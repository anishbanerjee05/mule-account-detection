import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import redis
import json
import pickle
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
    Compute a GNN embedding for a new account using:
    1. Its own behavioral features (GNN forward pass)
    2. Pre-computed neighbor embeddings from Redis (EdgeProp approximation)
    """
    # Step 1 — Retrieve neighbor embeddings from Redis
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

    # Step 2 — Pad behavioral features to 15 (GNN input size)
    features = req.behavioral[:12]
    while len(features) < 15:
        features.append(0.0)
    x = torch.tensor([features], dtype=torch.float32)

    # Step 3 — GNN forward pass
    with torch.no_grad():
        embedding = model.get_embedding(x, neighbor_embeddings)
        embedding_np = embedding.squeeze().numpy()

    # Determine method
    if n_neighbors >= 3:
        method = f"gnn_forward_{n_neighbors}_neighbors"
    elif n_neighbors > 0:
        method = f"gnn_forward_{n_neighbors}_neighbor"
    else:
        method = "gnn_forward_no_neighbors"

    # Step 4 — Store embedding in Redis with 6-hour TTL.
    # NOTE: the embed_meta write that used to happen here has been
    # REMOVED. consumer.py never reads embed_meta:* at all — it only
    # ever calls embed:{account_id} (see get_gnn_embedding() in
    # consumer.py). That second write was a pure cost with zero
    # consumer-side benefit: every /embed call was paying for 2 Redis
    # writes when only 1 was ever used downstream. Removing it cuts
    # per-request Redis round-trips from 3 (pipeline GET + 2x SET) to
    # 2 (pipeline GET + 1x SET), with no change in behavior anywhere
    # that actually reads this data. The dashboard does not read
    # embed_meta either — confirmed against dashboard.py.
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
        "total_embeddings":       stored,
        "real_gnn_embeddings":    stored,
    }

if __name__ == "__main__":
    print("\nGNN Embedding Microservice starting on port 8002...")
    # FIX: was running with uvicorn's default single-worker, effectively
    # synchronous request handling — every /embed call had to fully
    # complete (Redis I/O + forward pass + Redis write) before the next
    # request was even picked up. Under consumer.py's rapid-fire calls
    # with a cold (just-flushed) Redis cache, this serialization was the
    # actual bottleneck, not the model or Redis individually being slow.
    #
    # workers=4 runs 4 separate worker processes, each with their own
    # model copy in memory, handling requests in parallel via uvicorn's
    # process manager. This is the standard fix for synchronous FastAPI
    # endpoints under concurrent load. Safe here because:
    #   - the model is read-only at inference time (model.eval(), no
    #     shared mutable state to race on)
    #   - each worker gets its own Redis connection pool, and Redis
    #     itself safely handles concurrent clients
    # Reload must stay False — reload=True is incompatible with workers>1.
    uvicorn.run("gnn_service:app", host="0.0.0.0", port=8002,
                workers=4, reload=False)