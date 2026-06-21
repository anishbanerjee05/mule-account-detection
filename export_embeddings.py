"""
export_embeddings.py — Builds the persistent GNN embedding store.

Regenerates embeddings for every account in the trained graph CLEANLY
from best_gnn_model.pt + graph_data_full.pkl — not from whatever
happens to currently be in Redis. This matters because earlier
testing involved multiple Redis flushes; the persistent store should
be built from the authoritative trained source, not inherited state.

IMPORTANT: this uses the REAL MuleGNN architecture (GCNConv + edge_attr
projection + scatter_add aggregation), copied exactly from the
notebook cell that rebuilds the model for inference. This is NOT the
simplified Linear-layer approximation used in gnn_service.py's live
single-account inference path — that approximation exists for low-
latency per-transaction scoring. This batch export uses the real
trained forward pass against the real graph structure (including edge
features), since this file becomes the ground truth every other
lookup falls back to — accuracy matters more here than speed, and
speed isn't a constraint for a one-time/periodic batch job anyway.

Output: embeddings_store.parquet
  Columns: account_id (str), embedding (list[float], 64-dim)

Prerequisite: graph_data_full.pkl must exist. Create it by adding
this cell to 05_GNN.ipynb, immediately after the cell that builds
`graph_data` and prints "All variables rebuilt successfully":

    import pickle
    with open('graph_data_full.pkl', 'wb') as f:
        pickle.dump({'graph_data': graph_data.to('cpu'),
                     'all_nodes': all_nodes}, f)
    print("Saved full graph tensor + node ordering to graph_data_full.pkl")

Usage:
    python export_embeddings.py

Run this:
  - Once now, to create the initial persistent store
  - Any time the GNN model is retrained (manually, until Layer 1's
    scheduler exists per docs/embedding_persistence_architecture.md)
"""

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch_geometric.nn import GCNConv


# ── Model architecture — copied EXACTLY from the notebook's inference
# rebuild cell. Must stay byte-for-byte identical to that cell, or
# loaded weights will not align with the forward pass and embeddings
# will be silently wrong (no error, just garbage output). ────────────
class MuleGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super(MuleGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.edge_proj = nn.Linear(2, hidden_channels)
        self.fc1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        x1 = self.conv1(x, edge_index)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = self.conv2(x1, edge_index)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        edge_emb = self.edge_proj(edge_attr)
        edge_emb = F.relu(edge_emb)
        edge_agg = torch.zeros(x.shape[0], edge_emb.shape[1], device=x.device)
        dest_nodes = edge_index[1]
        edge_agg.scatter_add_(0, dest_nodes.unsqueeze(1).expand_as(edge_emb), edge_emb)
        x_combined = torch.cat([x2, edge_agg], dim=1)
        out = self.fc1(x_combined)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.fc2(out)
        return out

    def get_embedding(self, x, edge_index, edge_attr):
        """Same as forward(), but returns the 64-dim pre-fc2
        representation (x_combined after fc1+relu) rather than the
        final 2-class logits — this is the actual EMBEDDING used
        downstream by the hybrid XGBoost model (64 features), not a
        classification output. Matches gnn_service.py's live-inference
        return shape exactly (fc1 output, hidden_channels=64).
        dropout is disabled via model.eval(), so this is deterministic."""
        x1 = self.conv1(x, edge_index)
        x1 = F.relu(x1)
        x2 = self.conv2(x1, edge_index)
        x2 = F.relu(x2)
        edge_emb = self.edge_proj(edge_attr)
        edge_emb = F.relu(edge_emb)
        edge_agg = torch.zeros(x.shape[0], edge_emb.shape[1], device=x.device)
        dest_nodes = edge_index[1]
        edge_agg.scatter_add_(0, dest_nodes.unsqueeze(1).expand_as(edge_emb), edge_emb)
        x_combined = torch.cat([x2, edge_agg], dim=1)
        embedding = self.fc1(x_combined)
        embedding = F.relu(embedding)
        return embedding


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load the trained model ──────────────────────────────────────────────
    print("Loading trained GNN model (best_gnn_model.pt)...")
    model = MuleGNN(in_channels=15, hidden_channels=64, out_channels=2, dropout=0.3).to(
        device
    )
    model.load_state_dict(torch.load("best_gnn_model.pt", map_location=device))
    model.eval()
    print("Model loaded.")

    # ── Load the full graph tensor + node ordering ──────────────────────────
    print("Loading graph_data_full.pkl...")
    try:
        with open("graph_data_full.pkl", "rb") as f:
            saved = pickle.load(f)
    except FileNotFoundError:
        print("\nERROR: graph_data_full.pkl not found.")
        print("Add the save cell to 05_GNN.ipynb first (see this file's")
        print("module docstring for the exact code), run it once, then")
        print("re-run this export script.")
        return

    graph_data = saved["graph_data"]
    all_nodes = saved["all_nodes"]
    print(
        f"Loaded graph: {graph_data.x.shape[0]:,} nodes, "
        f"{graph_data.edge_index.shape[1]:,} edges"
    )

    x = graph_data.x.to(device)
    edge_index = graph_data.edge_index.to(device)
    edge_attr = graph_data.edge_attr.to(device)

    # ── Full-graph forward pass — one shot, computes ALL embeddings ────────
    print(f"\nRunning forward pass over {x.shape[0]:,} nodes...")
    with torch.no_grad():
        embeddings = model.get_embedding(x, edge_index, edge_attr)
        embeddings_np = embeddings.cpu().numpy()
    print(f"Embeddings computed: {embeddings_np.shape}")

    if embeddings_np.shape[0] != len(all_nodes):
        print(
            f"\nWARNING: embedding count ({embeddings_np.shape[0]:,}) does "
            f"not match all_nodes count ({len(all_nodes):,}). Account ID "
            f"mapping would be WRONG if we proceeded. Stopping."
        )
        return

    # ── Build the persistent store ──────────────────────────────────────────
    print("\nBuilding persistent store...")
    account_ids = [str(a) for a in all_nodes]
    df = pd.DataFrame(
        {
            "account_id": account_ids,
            "embedding": [emb.tolist() for emb in embeddings_np],
        }
    )

    output_path = "embeddings_store.parquet"
    df.to_parquet(output_path, index=False)
    print(f"Wrote {len(df):,} embeddings to {output_path}")

    print("\nDone. This file is the new persistent source of truth.")
    print("Next step: update gnn_service.py to check this file before")
    print("falling back to a live single-account forward pass.")


if __name__ == "__main__":
    main()
