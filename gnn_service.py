import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import redis
import json
import pickle
import pandas as pd
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import uvicorn

# ── GNN Architecture (must match 05_GNN.ipynb exactly) ───────────────────────
class MuleGNN(nn.Module):
    def __init__(self, in_channels=15, hidden_channels=64, out_channels=2):
        super().__init__()
        self.conv1   = torch.nn.Linear(in_channels, hidden_channels)
        self.conv2   = torch.nn.Linear(hidden_channels, hidden_channels)
        self.edge_fc = nn.Linear(2, hidden_channels)
        self.fc1     = nn.Linear(hidden_channels * 2, hidden_channels)
        self.fc2     = nn.Linear(hidden_channels, out_channels)

    def get_embedding(self, x, neighbor_embeddings=None):
        """
        Compute embedding for a single account using its features
        and pre-computed neighbor embeddings from Redis.
        No full graph needed — just local subgraph.
        """
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))

        if neighbor_embeddings is not None and len(neighbor_embeddings) > 0:
            # Aggregate neighbor embeddings (EdgeProp equivalent)
            neighbor_tensor = torch.tensor(
                neighbor_embeddings, dtype=torch.float32
            )
            aggregated = neighbor_tensor.mean(dim=0, keepdim=True)
            x_combined = torch.cat([x2, aggregated], dim=1)
        else:
            x_combined = torch.cat([x2, x2], dim=1)

        embedding = F.relu(self.fc1(x_combined))
        return embedding

app = FastAPI(title="GNN Embedding Microservice", version="1.0")

# ── Load model and Redis ──────────────────────────────────────────────────────
print("Loading GNN model...")
model = MuleGNN()
state = torch.load('best_gnn_model.pt', map_location='cpu')

# Filter only the weights our simplified forward pass uses
own_keys = set(model.state_dict().keys())
filtered = {k: v for k, v in state.items() if k in own_keys}
model.load_state_dict(filtered, strict=False)
model.eval()
print("GNN model loaded")

# torch.no_grad() context is already used per-request below; this just
# ensures no thread spawns extra intraop overhead fighting with uvicorn's
# own worker threads under concurrent load.
torch.set_num_threads(1)

# decode_responses=True + connection pooling (redis-py pools connections
# by default per Redis() instance) — no change needed here, but worth
# noting this client is shared across all concurrent requests safely,
# since redis-py's connection pool is thread-safe.
r = redis.Redis(host='localhost', port=6379, decode_responses=True)
r.ping()
print("Redis connected")

# ── Persistent embedding store (NEW) ─────────────────────────────────────────
# Layer 2 of the 3-tier lookup chain: Redis (hot cache) -> persistent
# store (this) -> live forward pass (last resort, see compute_embedding
# below). Built by export_embeddings.py from the real trained model
# against the full graph. Loaded once per worker process at startup —
# with workers=4, each worker is a separate OS process and needs its
# own in-memory copy; they cannot share a Python dict across processes.
# This in-memory load is fast (a single Parquet read, no network I/O)
# and safe to repeat per worker.
#
# REDIS PRE-WARMING IS NOT DONE HERE. An earlier version of this file
# pre-warmed Redis (579,735 SET calls) directly in this startup block,
# once per worker. Under workers=4 on Windows, that caused a crash
# loop: uvicorn's process supervisor treated the slow startup as an
# unresponsive worker, killed it, and respawned it, repeating
# indefinitely without the service ever stabilizing. Heavy synchronous
# I/O at multi-process startup is fragile on Windows in a way it isn't
# on Linux.
#
# Fix: pre-warming is now a SEPARATE one-time script (see
# warm_redis_cache.py) you run ONCE before starting this service, not
# something every worker repeats. This service now only loads the
# persistent store into memory here (fast, lightweight, per-worker is
# fine) and falls back to Tier 2 (persistent_embeddings dict) on a
# Redis miss at request time, same as before -- the only change is
# WHEN Redis gets pre-populated, not whether the 3-tier lookup works.
PERSISTENT_STORE_PATH = 'embeddings_store.parquet'
persistent_embeddings = {}  # account_id (str) -> list[float] (64-dim)

if os.path.exists(PERSISTENT_STORE_PATH):
    print(f"Loading persistent embedding store from {PERSISTENT_STORE_PATH}...")
    _df = pd.read_parquet(PERSISTENT_STORE_PATH)
    # Parquet round-trips list-of-float columns back as numpy ndarrays,
    # not plain Python lists -- json.dumps() cannot serialize ndarrays
    # directly, so convert explicitly here, once, at load time, rather
    # than at every single lookup call.
    persistent_embeddings = {
        account_id: (embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding))
        for account_id, embedding in zip(_df['account_id'], _df['embedding'])
    }
    print(f"Loaded {len(persistent_embeddings):,} persistent embeddings into memory")
    print("(Run warm_redis_cache.py separately, once, to pre-populate "
          "Redis -- not done automatically here, see comment above.)")
else:
    print(f"WARNING: {PERSISTENT_STORE_PATH} not found. Running without "
          f"persistent fallback — a Redis flush will degrade embedding "
          f"quality until live traffic rebuilds the cache. Run "
          f"export_embeddings.py to create this file.")

# ── Feature order must match training ────────────────────────────────────────
BEHAVIORAL_FEATURES = [
    'velocity_ratio', 'balance_retention_rate', 'fan_out_degree',
    'cross_bank_degree', 'round_amount_ratio', 'structuring_score',
    'amount_variance_score', 'session_burst_score', 'dormancy_ratio',
    'total_transactions', 'avg_amount', 'max_amount'
]

# ── Request schema ────────────────────────────────────────────────────────────
class EmbeddingRequest(BaseModel):
    account_id:      str
    behavioral:      List[float]   # 12 behavioral features
    neighbor_ids:    List[str]     # Account IDs of known neighbors
    amount:          float = 0.0
    payment_format:  str   = "unknown"

class EmbeddingResponse(BaseModel):
    account_id:      str
    embedding:       List[float]
    method:          str
    n_neighbors:     int
    is_provisional:  bool
    computed_at:     str

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/embed", response_model=EmbeddingResponse)
def compute_embedding(req: EmbeddingRequest):
    """
    Resolve a GNN embedding for an account using a 3-tier chain:
    1. Redis (hot cache) -- checked first, fastest path
    2. Persistent store (in-memory dict, loaded from
       embeddings_store.parquet at startup) -- a cache miss here means
       this account has a REAL trained embedding that just isn't in
       Redis right now (e.g. TTL expired), not that it was never
       trained. Returning this is strictly better than recomputing a
       weak, no-graph-context live embedding for an account the model
       has already seen during training.
    3. Live single-account forward pass -- TRUE last resort, only for
       accounts genuinely absent from the trained graph (new accounts
       since the last training run). This is the only tier that uses
       neighbor_ids/behavioral features from the request.
    """
    # ── Tier 1 — Redis hot cache ─────────────────────────────────────────────
    cached = r.get(f"embed:{req.account_id}")
    if cached:
        embedding_list = json.loads(cached)
        return EmbeddingResponse(
            account_id     = req.account_id,
            embedding      = embedding_list,
            method         = "redis_cache",
            n_neighbors    = -1,  # not applicable, not recomputed
            is_provisional = False,
            computed_at    = datetime.now().isoformat()
        )

    # ── Tier 2 — Persistent store (in-memory, loaded at startup) ───────────
    if req.account_id in persistent_embeddings:
        embedding_list = persistent_embeddings[req.account_id]
        # Backfill Redis so the next lookup for this account is a Tier 1
        # hit again, without needing another persistent-store lookup.
        r.set(f"embed:{req.account_id}", json.dumps(embedding_list), ex=21600)
        return EmbeddingResponse(
            account_id     = req.account_id,
            embedding      = embedding_list,
            method         = "persistent_store",
            n_neighbors    = -1,
            is_provisional = False,
            computed_at    = datetime.now().isoformat()
        )

    # ── Tier 3 — Live forward pass (true last resort) ───────────────────────
    # Account genuinely not in the trained graph. Use the simplified
    # single-account approximation with whatever neighbor context the
    # request provides.
    neighbor_embeddings = []
    if req.neighbor_ids:
        pipe = r.pipeline()
        for nid in req.neighbor_ids[:50]:
            pipe.get(f"embed:{nid}")
        results = pipe.execute()
        for res in results:
            if res:
                neighbor_embeddings.append(json.loads(res))

    n_neighbors = len(neighbor_embeddings)

    features = req.behavioral[:12]
    while len(features) < 15:
        features.append(0.0)
    x = torch.tensor([features], dtype=torch.float32)

    with torch.no_grad():
        embedding = model.get_embedding(x, neighbor_embeddings)
        embedding_np = embedding.squeeze().numpy()

    if n_neighbors >= 3:
        method = f"gnn_forward_{n_neighbors}_neighbors"
    elif n_neighbors > 0:
        method = f"gnn_forward_{n_neighbors}_neighbor"
    else:
        method = "gnn_forward_no_neighbors"

    r.set(
        f"embed:{req.account_id}",
        json.dumps(embedding_np.tolist()),
        ex=21600
    )

    return EmbeddingResponse(
        account_id     = req.account_id,
        embedding      = embedding_np.tolist(),
        method         = method,
        n_neighbors    = n_neighbors,
        is_provisional = True,
        computed_at    = datetime.now().isoformat()
    )

@app.get("/embed/{account_id}")
def get_embedding(account_id: str):
    """Retrieve stored embedding for an account."""
    emb_json  = r.get(f"embed:{account_id}")

    if not emb_json:
        return {"error": f"No embedding found for {account_id}"}

    return {
        "account_id": account_id,
        "embedding":  json.loads(emb_json),
        "meta":       {"method": "stored_gnn"}
    }

@app.get("/health")
def health():
    return {
        "status":      "operational",
        "model":       "MuleGNN",
        "torch":       torch.__version__,
        "redis":       r.ping(),
        "timestamp":   datetime.now().isoformat()
    }

@app.get("/stats")
def stats():
    stored = len(r.keys("embed:*"))
    return {
        "redis_cached_embeddings":     stored,
        "persistent_store_embeddings": len(persistent_embeddings),
        "persistent_store_loaded":     len(persistent_embeddings) > 0,
    }

if __name__ == "__main__":
    print("\nGNN Embedding Microservice starting on port 8002...")
    # FIX 1: was running with uvicorn's default single-worker, effectively
    # synchronous request handling — every /embed call had to fully
    # complete (Redis I/O + forward pass + Redis write) before the next
    # request was even picked up. Under consumer.py's rapid-fire calls
    # with a cold (just-flushed) Redis cache, this serialization was the
    # actual bottleneck, not the model or Redis individually being slow.
    #
    # workers=2 runs 2 separate worker processes, each with their own
    # model copy in memory, handling requests in parallel via uvicorn's
    # process manager.
    #
    # NOTE: originally set to workers=4, reduced to workers=2 due to a
    # real memory constraint on this machine (~6.2GB available out of
    # 16.8GB total, the rest consumed by Docker's Redis/Kafka/Zookeeper/
    # Postgres containers).
    #
    # FIX 2 (the actual root cause of the repeated crash loop): uvicorn's
    # default `timeout_worker_healthcheck` is 5 SECONDS. Measured
    # standalone, loading embeddings_store.parquet (579,735 rows) and
    # converting it to a Python dict takes ~4.8s by itself -- under real
    # startup conditions, with the worker ALSO loading the full PyTorch
    # model and competing with a sibling worker for CPU/memory at the
    # same time, this consistently exceeded 5 seconds. Uvicorn's
    # supervisor then concluded the worker was unresponsive and killed
    # it -- even though it was simply still busy, not actually dead.
    # That killed worker immediately respawned, hit the same slow
    # startup path, and got killed again: the crash loop.
    #
    # This was confirmed by direct measurement, not assumed: see the
    # standalone timing test that produced "Read: 1.6s, Convert: 3.2s"
    # during debugging. Extending the healthcheck timeout to 30s gives
    # workers a realistic window to finish startup before being judged
    # unresponsive.
    #
    # Safe because:
    #   - the model is read-only at inference time (model.eval(), no
    #     shared mutable state to race on)
    #   - each worker gets its own Redis connection pool, and Redis
    #     itself safely handles concurrent clients
    # Reload must stay False — reload=True is incompatible with workers>1.
    uvicorn.run("gnn_service:app", host="0.0.0.0", port=8002,
                workers=2, reload=False,
                timeout_worker_healthcheck=30)