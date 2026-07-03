import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import redis
import json
import pickle
import pandas as pd
import os
import threading
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import uvicorn

# -- GNN Architecture (must match 05_GNN.ipynb exactly) -----------------------
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
        No full graph needed -- just local subgraph.
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

# -- Load model and Redis ------------------------------------------------------
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

# -- Concurrency limiter for live GNN forward passes --------------------------
# PROBLEM THIS SOLVES:
# Tier 1 (Redis cache) and Tier 2 (persistent store) are cheap dict/Redis
# lookups -- no memory spike risk regardless of how many run concurrently.
# Tier 3 (live forward pass) allocates PyTorch tensors and runs actual
# matrix multiplications. FastAPI/uvicorn can dispatch many incoming HTTP
# requests concurrently (each in its own thread by default), and if a
# batch of 100 transactions contains many genuinely new accounts, dozens
# of Tier 3 forward passes can fire at once across overlapping requests.
# Each forward pass briefly holds several small tensors in memory, but
# enough of them stacking up at the same instant produces a sharp,
# short-lived memory spike -- this is what was crashing PowerShell on a
# 16.8GB machine running Docker + Kafka + Redis simultaneously.
#
# FIX: a bounded semaphore caps how many Tier 3 forward passes can be
# "in flight" at once, regardless of how many requests arrive concurrently.
# Extra requests simply queue briefly (sub-millisecond to a few ms wait)
# instead of all executing in parallel. This trades a small amount of
# worst-case latency for eliminating the memory spike entirely -- the
# right tradeoff for a fraud-detection system where a few extra
# milliseconds is invisible but a crash means zero detections.
#
# MAX_CONCURRENT_FORWARD_PASSES=5 was chosen conservatively for this
# 16.8GB development machine. In production, this is replaced entirely
# by NVIDIA Triton Inference Server's own dynamic batching and instance
# count controls -- this semaphore is a stopgap appropriate for
# single-machine demo/development, not a production concurrency
# control mechanism.
MAX_CONCURRENT_FORWARD_PASSES = 5
_forward_pass_semaphore = threading.Semaphore(MAX_CONCURRENT_FORWARD_PASSES)


def run_forward_pass(x, neighbor_embeddings):
    """
    Thread-safe, memory-bounded wrapper around model.get_embedding().
    Acquires the global semaphore before running the forward pass,
    releases it immediately after -- even if an exception occurs.
    """
    with _forward_pass_semaphore:
        with torch.no_grad():
            embedding = model.get_embedding(x, neighbor_embeddings)
        return embedding


# decode_responses=True + connection pooling (redis-py pools connections
# by default per Redis() instance) -- no change needed here, but worth
# noting this client is shared across all concurrent requests safely,
# since redis-py's connection pool is thread-safe.
r = redis.Redis(host='localhost', port=6379, decode_responses=True)
r.ping()
print("Redis connected")

# -- Persistent embedding store -----------------------------------------------
# Layer 2 of the 3-tier lookup chain: Redis (hot cache) -> persistent
# store (this) -> live forward pass (last resort, see compute_embedding
# below). Built by export_embeddings.py from the real trained model
# against the full graph. Loaded once per worker process at startup --
# with workers>1, each worker is a separate OS process and needs its
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
# warm_redis_cache.py). Run it once before starting the demo, not
# automatically here.
print("Loading persistent embedding store from embeddings_store.parquet...")
persistent_embeddings = {}
try:
    df = pd.read_parquet('embeddings_store.parquet')
    # Parquet round-trips list-of-float columns back as numpy ndarrays
    # of numpy.float32, not plain Python lists of float. json.dumps()
    # cannot serialize numpy.float32 -- it raises TypeError at the first
    # Tier 2 lookup that tries to backfill Redis. .tolist() converts to
    # native Python floats; this must happen once here at load time,
    # not at every lookup call. zip() over the two columns is also much
    # faster than per-row iterrows() for 579,735 rows.
    persistent_embeddings = {
        str(account_id): (embedding.tolist() if hasattr(embedding, 'tolist')
                          else list(embedding))
        for account_id, embedding in zip(df['account_id'], df['embedding'])
    }
    print(f"Loaded {len(persistent_embeddings):,} persistent embeddings into memory")
except FileNotFoundError:
    print("embeddings_store.parquet not found -- persistent store disabled, "
          "all lookups will use Tier 3 (live forward pass)")

print("(Run warm_redis_cache.py separately, once, to pre-populate Redis "
      "-- not done automatically here, see comment above.)")

# -- Feature order must match training -----------------------------------------
BEHAVIORAL_FEATURES = [
    'velocity_ratio', 'balance_retention_rate', 'fan_out_degree',
    'cross_bank_degree', 'round_amount_ratio', 'structuring_score',
    'amount_variance_score', 'session_burst_score', 'dormancy_ratio',
    'total_transactions', 'avg_amount', 'max_amount'
]

# -- Request/response schemas --------------------------------------------------
class EmbeddingRequest(BaseModel):
    account_id:   str
    behavioral:   List[float]
    neighbor_ids: Optional[List[str]] = []
    amount:       Optional[float] = 0.0
    payment_format: Optional[str] = "unknown"


class EmbeddingResponse(BaseModel):
    account_id:     str
    embedding:      List[float]
    method:         str
    n_neighbors:    int
    is_provisional: bool
    computed_at:    str


@app.post("/embed", response_model=EmbeddingResponse)
def compute_embedding(req: EmbeddingRequest):
    """
    3-tier embedding lookup for a single account:
    1. Redis cache (hot, sub-millisecond)
    2. Persistent store (579,735 pre-computed embeddings from training graph)
    3. Live single-account forward pass -- TRUE last resort, only for
       accounts genuinely outside the trained graph.
    """
    # -- Tier 1 -- Redis cache --------------------------------------------------
    cached = r.get(f"embed:{req.account_id}")
    if cached:
        return EmbeddingResponse(
            account_id     = req.account_id,
            embedding      = json.loads(cached),
            method         = "redis_cache",
            n_neighbors    = -1,
            is_provisional = False,
            computed_at    = datetime.now().isoformat()
        )

    # -- Tier 2 -- Persistent store ---------------------------------------------
    if req.account_id in persistent_embeddings:
        embedding_list = persistent_embeddings[req.account_id]
        r.set(f"embed:{req.account_id}", json.dumps(embedding_list), ex=21600)
        return EmbeddingResponse(
            account_id     = req.account_id,
            embedding      = embedding_list,
            method         = "persistent_store",
            n_neighbors    = -1,
            is_provisional = False,
            computed_at    = datetime.now().isoformat()
        )

    # -- Tier 3 -- Live forward pass (true last resort) -------------------------
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

    embedding = run_forward_pass(x, neighbor_embeddings)
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
    """Retrieve stored embedding for an account.
    Now checks Tier 1 (Redis) then Tier 2 (persistent store) --
    previously only checked Redis, causing false "not found" errors
    for any of the 579,735 trained accounts that hadn't yet been
    touched by live traffic."""
    emb_json = r.get(f"embed:{account_id}")
    if emb_json:
        return {
            "account_id": account_id,
            "embedding":  json.loads(emb_json),
            "meta":       {"method": "redis_cache"}
        }
    if account_id in persistent_embeddings:
        embedding = persistent_embeddings[account_id]
        r.set(f"embed:{account_id}", json.dumps(embedding), ex=21600)
        return {
            "account_id": account_id,
            "embedding":  embedding,
            "meta":       {"method": "persistent_store"}
        }
    return {"error": f"No embedding found for {account_id}"}


class BatchEmbeddingRequest(BaseModel):
    # List of account IDs to resolve embeddings for.
    # Each entry: {"account_id": str, "behavioral": List[float],
    #              "neighbor_ids": List[str]}
    # behavioral and neighbor_ids are only used if the account is not
    # in Redis or the persistent store (Tier 3 fallback).
    accounts: List[EmbeddingRequest]


@app.post("/embed_batch")
def compute_embedding_batch(req: BatchEmbeddingRequest):
    """
    Batch version of /embed -- resolves embeddings for up to ~100 accounts
    in a single call. Built to eliminate the per-transaction HTTP overhead
    that capped the original consumer.py at ~27 TPS.

    Same 3-tier lookup as /embed, but batched at each tier:
    1. Redis: one pipeline call for ALL accounts (not one GET per account)
    2. Persistent store: vectorized dict lookups, no I/O
    3. Live forward pass: only for accounts that miss both prior tiers,
       and even then bounded by the global semaphore (see
       run_forward_pass above) so a batch full of genuinely new accounts
       cannot spike memory by firing dozens of forward passes at once.
    """
    results = {}

    # -- Tier 1: Redis pipeline for all accounts at once ------------------------
    pipe = r.pipeline()
    for acct_req in req.accounts:
        pipe.get(f"embed:{acct_req.account_id}")
    cached_results = pipe.execute()

    cache_misses = []
    for acct_req, cached in zip(req.accounts, cached_results):
        if cached:
            results[acct_req.account_id] = {
                "embedding":      json.loads(cached),
                "method":         "redis_cache",
                "is_provisional": False,
            }
        else:
            cache_misses.append(acct_req)

    # -- Tier 2: persistent store for cache misses -------------------------------
    still_missing = []
    backfill_pipe = r.pipeline()
    for acct_req in cache_misses:
        aid = acct_req.account_id
        if aid in persistent_embeddings:
            embedding = persistent_embeddings[aid]
            results[aid] = {
                "embedding":      embedding,
                "method":         "persistent_store",
                "is_provisional": False,
            }
            # Backfill Redis so next lookup is Tier 1
            backfill_pipe.set(f"embed:{aid}", json.dumps(embedding), ex=21600)
        else:
            still_missing.append(acct_req)
    backfill_pipe.execute()

    # -- Tier 3: live GNN forward pass for genuinely unknown accounts -----------
    for acct_req in still_missing:
        aid = acct_req.account_id

        neighbor_embeddings = []
        if acct_req.neighbor_ids:
            npipe = r.pipeline()
            for nid in acct_req.neighbor_ids[:10]:
                npipe.get(f"embed:{nid}")
            nresults = npipe.execute()
            for nres in nresults:
                if nres:
                    neighbor_embeddings.append(json.loads(nres))

        features = acct_req.behavioral[:12]
        while len(features) < 15:
            features.append(0.0)
        x = torch.tensor([features], dtype=torch.float32)

        embedding = run_forward_pass(x, neighbor_embeddings)
        embedding_list = embedding.squeeze().numpy().tolist()

        n = len(neighbor_embeddings)
        method = (f"gnn_forward_{n}_neighbors" if n >= 3
                  else f"gnn_forward_{n}_neighbor" if n > 0
                  else "gnn_forward_no_neighbors")

        r.set(f"embed:{aid}", json.dumps(embedding_list), ex=21600)
        results[aid] = {
            "embedding":      embedding_list,
            "method":         method,
            "is_provisional": True,
        }

    return results


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
        "max_concurrent_forward_passes": MAX_CONCURRENT_FORWARD_PASSES,
        "forward_pass_slots_available": _forward_pass_semaphore._value,
    }


if __name__ == "__main__":
    print("\nGNN Embedding Microservice starting on port 8002...")
    # FIX 1: was running with uvicorn's default single-worker, effectively
    # synchronous request handling -- every /embed call had to fully
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
    # Postgres containers). For workers=1 (used after system restarts to
    # avoid a port-binding race condition), start_gnn.py is used instead
    # of this __main__ block.
    #
    # FIX 2 (the actual root cause of the repeated crash loop): uvicorn's
    # default timeout_worker_healthcheck is 5 SECONDS. Measured
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
    # Reload must stay False -- reload=True is incompatible with workers>1.
    uvicorn.run("gnn_service:app", host="0.0.0.0", port=8002,
                workers=2, reload=False,
                timeout_worker_healthcheck=30)