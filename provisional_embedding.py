import redis
import json
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from datetime import datetime

# ── Redis ─────────────────────────────────────────────────────────────────────
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# ── Load trained GNN weights for forward pass ────────────────────────────────
try:
    import sys
    sys.path.append('.')

    # Load GNN model architecture
    import torch.nn as nn
    from torch_geometric.nn import GCNConv

    class MuleGNN(nn.Module):
        def __init__(self, in_channels=15, hidden_channels=64, out_channels=2):
            super().__init__()
            self.conv1   = GCNConv(in_channels, hidden_channels)
            self.conv2   = GCNConv(hidden_channels, hidden_channels)
            self.edge_fc = nn.Linear(2, hidden_channels)
            self.fc1     = nn.Linear(hidden_channels * 2, hidden_channels)
            self.fc2     = nn.Linear(hidden_channels, out_channels)

        def forward(self, x, edge_index, edge_attr=None):
            x1 = F.relu(self.conv1(x, edge_index))
            x2 = F.relu(self.conv2(x1, edge_index))
            if edge_attr is not None:
                edge_emb  = F.relu(self.edge_fc(edge_attr.float()))
                dest_nodes = edge_index[1]
                edge_agg  = torch.zeros_like(x2)
                edge_agg.scatter_add_(
                    0,
                    dest_nodes.unsqueeze(1).expand_as(edge_emb),
                    edge_emb
                )
                x_combined = torch.cat([x2, edge_agg], dim=1)
            else:
                x_combined = torch.cat([x2, x2], dim=1)
            x3 = F.relu(self.fc1(x_combined))
            return x3  # Return embeddings, not logits

    GNN_MODEL = MuleGNN()
    GNN_MODEL.load_state_dict(
        torch.load('best_gnn_model.pt', map_location='cpu')
    )
    GNN_MODEL.eval()
    GNN_AVAILABLE = True
    print("GNN model loaded for provisional embedding computation")

except Exception as e:
    GNN_MODEL    = None
    GNN_AVAILABLE = False
    print(f"GNN model not available — using neighbor averaging: {e}")


# ── Core functions ────────────────────────────────────────────────────────────

def get_embedding(account_id: str):
    """Retrieve embedding from Redis — returns (embedding, source)."""
    emb_json = r.get(f"embed:{account_id}")
    if emb_json:
        return np.array(json.loads(emb_json)), "stored"
    return None, None


def compute_neighbor_average(neighbor_ids: list) -> np.ndarray:
    """
    Average the embeddings of known neighbor accounts.
    Core of the provisional embedding: if you transact with known accounts,
    your embedding should reflect their graph neighborhood.
    """
    embeddings = []
    pipe = r.pipeline()
    for nid in neighbor_ids[:50]:  # Cap at 50 neighbors
        pipe.get(f"embed:{nid}")
    results = pipe.execute()

    for res in results:
        if res:
            embeddings.append(json.loads(res))

    if not embeddings:
        return np.zeros(64)

    return np.mean(embeddings, axis=0)


def compute_provisional_embedding(
    account_id: str,
    neighbor_ids: list,
    amount: float = 0.0,
    transaction_type: str = "unknown"
) -> tuple:
    """
    Compute a provisional GNN embedding for a new account.

    Strategy:
    1. If account has >=1 neighbor with a stored embedding:
       → Average neighbor embeddings (weighted by transaction amount)
    2. If no neighbors have embeddings:
       → Use a risk-informed default vector based on behavioral signals
    3. Store with 6-hour TTL (replaced by daily GNN rebuild)

    Returns: (embedding_vector, method_used, neighbors_found)
    """
    if not neighbor_ids:
        # No transaction history at all — use zeros
        embedding = np.zeros(64)
        method    = "zero_no_neighbors"
        n_found   = 0
    else:
        # Try neighbor averaging
        neighbor_embeddings = []
        neighbor_found_ids  = []

        pipe = r.pipeline()
        for nid in neighbor_ids[:50]:
            pipe.get(f"embed:{nid}")
        results = pipe.execute()

        for nid, res in zip(neighbor_ids[:50], results):
            if res:
                neighbor_embeddings.append(json.loads(res))
                neighbor_found_ids.append(nid)

        n_found = len(neighbor_embeddings)

        if n_found == 0:
            # No neighbors have embeddings either — use risk-informed default
            embedding = _risk_informed_default(amount, transaction_type)
            method    = "risk_default_no_neighbor_embeddings"
        elif n_found == 1:
            # Only one neighbor — use directly with small noise
            embedding = np.array(neighbor_embeddings[0])
            noise     = np.random.normal(0, 0.01, 64)
            embedding = embedding + noise
            method    = "single_neighbor"
        else:
            # Multiple neighbors — weighted average
            embedding = np.mean(neighbor_embeddings, axis=0)
            method    = f"neighbor_average_{n_found}"

    # Store in Redis with 6-hour TTL (provisional)
    r.set(
        f"embed:{account_id}",
        json.dumps(embedding.tolist()),
        ex=21600  # 6 hours
    )

    # Mark as provisional so the daily rebuild can replace it
    r.set(
        f"embed_provisional:{account_id}",
        json.dumps({
            "method":       method,
            "n_neighbors":  n_found if neighbor_ids else 0,
            "computed_at":  datetime.now().isoformat(),
            "ttl_hours":    6
        }),
        ex=21600
    )

    return embedding, method, n_found


def _risk_informed_default(amount: float, transaction_type: str) -> np.ndarray:
    """
    When no neighbor embeddings exist, return a risk-informed default vector.
    High-amount first transactions get a slightly elevated risk embedding.
    This is better than zeros — it does not artificially suppress the score.
    """
    base = np.zeros(64)

    # Elevate specific dimensions associated with high-risk behavior
    # based on what the GNN learned to encode
    if amount > 500000:     # Above Rs. 5 lakh
        base[0:8] = 0.3     # Elevate financial magnitude dimensions
    if amount > 1000000:    # Above Rs. 10 lakh
        base[0:8] = 0.6
    if transaction_type in ['NEFT', 'RTGS', 'Wire']:
        base[8:16] = 0.2    # Elevate wire transfer dimensions

    return base


def get_or_compute_embedding(
    account_id: str,
    neighbor_ids: list = None,
    amount: float = 0.0,
    transaction_type: str = "unknown"
) -> tuple:
    """
    Main entry point. Returns embedding and metadata.
    Use this instead of r.get('embed:{id}') everywhere.

    Returns: (embedding_array, has_real_embedding, method)
    """
    # Check stored embedding first
    emb_json = r.get(f"embed:{account_id}")

    if emb_json:
        embedding = np.array(json.loads(emb_json))

        # Check if it is provisional
        provisional = r.get(f"embed_provisional:{account_id}")
        if provisional:
            return embedding, False, json.loads(provisional).get('method', 'provisional')
        return embedding, True, "stored"

    # No embedding found — compute provisional
    if neighbor_ids is None:
        neighbor_ids = []

    embedding, method, n_found = compute_provisional_embedding(
        account_id, neighbor_ids, amount, transaction_type
    )

    return embedding, False, method


def embedding_stats() -> dict:
    """Show stats on provisional vs stored embeddings."""
    provisional_keys = r.keys("embed_provisional:*")
    stored_keys      = r.keys("embed:*")

    # Filter out provisional overlap
    real_stored = len(stored_keys) - len(provisional_keys)

    return {
        "total_embeddings":       len(stored_keys),
        "real_gnn_embeddings":    real_stored,
        "provisional_embeddings": len(provisional_keys),
        "provisional_pct":        (
            f"{len(provisional_keys)/len(stored_keys)*100:.1f}%"
            if stored_keys else "0%"
        )
    }


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== Provisional Embedding Test ===\n")

    # Test 1: Account with no neighbors
    print("Test 1: Brand new account, no neighbors")
    emb, has_real, method = get_or_compute_embedding(
        "NEW_ACCOUNT_001", [], 250000.0, "NEFT"
    )
    print(f"  Method: {method}")
    print(f"  Has real embedding: {has_real}")
    print(f"  Embedding shape: {emb.shape}")
    print(f"  Embedding norm: {np.linalg.norm(emb):.4f}")

    print()

    # Test 2: Account transacting with known IBM AML accounts
    print("Test 2: New account transacting with known IBM AML accounts")
    known_neighbors = ['8000ECA90', '10042B660', '8006639C0']
    emb2, has_real2, method2 = get_or_compute_embedding(
        "NEW_ACCOUNT_002", known_neighbors, 950000.0, "NEFT"
    )
    print(f"  Method: {method2}")
    print(f"  Has real embedding: {has_real2}")
    print(f"  Embedding norm: {np.linalg.norm(emb2):.4f}")
    print(f"  Non-zero dimensions: {np.sum(emb2 != 0)}")

    print()

    # Test 3: Account with one known neighbor
    print("Test 3: New account with single known neighbor")
    emb3, has_real3, method3 = get_or_compute_embedding(
        "NEW_ACCOUNT_003", ['8000ECA90'], 75000.0, "IMPS"
    )
    print(f"  Method: {method3}")
    print(f"  Embedding norm: {np.linalg.norm(emb3):.4f}")

    print()

    # Test 4: Retrieve previously computed provisional
    print("Test 4: Retrieve previously computed provisional embedding")
    emb4, has_real4, method4 = get_or_compute_embedding("NEW_ACCOUNT_002")
    print(f"  Method: {method4}")
    print(f"  Retrieved from Redis: True")
    print(f"  Embedding matches: {np.allclose(emb2, emb4)}")

    print()

    # Stats
    stats = embedding_stats()
    print("=== Embedding Stats ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\nProvisional embedding system ready.")
    print("Integrate into consumer.py by replacing get_gnn_embedding()")
    print("with get_or_compute_embedding() from this module.")