"""
warm_redis_cache.py — One-time Redis pre-warm from the persistent
embedding store.

Run this ONCE, manually, before starting gnn_service.py, any time you
want Redis re-populated from embeddings_store.parquet (e.g. right
after a deliberate Redis flush, or after running export_embeddings.py
to refresh the persistent store with newly trained embeddings).

This was previously done automatically inside gnn_service.py's
startup code, once per worker process. Under workers=4 on Windows,
that caused a crash loop -- uvicorn's process supervisor killed
workers that took too long doing 579,735 synchronous Redis writes
during startup, then respawned them, repeating indefinitely. Splitting
this into a separate script run once, before the service starts,
avoids that entirely: gnn_service.py's own startup becomes fast again
(just an in-memory Parquet read), and Redis still ends up correctly
pre-populated.

Usage (run once, before starting gnn_service.py):
    python warm_redis_cache.py
"""

import json
import redis
import pandas as pd

PERSISTENT_STORE_PATH = 'embeddings_store.parquet'

print("Connecting to Redis...")
r = redis.Redis(host='localhost', port=6379, decode_responses=True)
r.ping()
print("Redis connected")

print(f"Loading {PERSISTENT_STORE_PATH}...")
df = pd.read_parquet(PERSISTENT_STORE_PATH)
print(f"Loaded {len(df):,} embeddings from disk")

print("Pre-warming Redis (bulk pipelined writes)...")
pipe = r.pipeline()
warmed = 0
for account_id, embedding in zip(df['account_id'], df['embedding']):
    # Same ndarray-to-list conversion needed here as in gnn_service.py --
    # Parquet round-trips list columns back as numpy arrays.
    embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
    pipe.set(f"embed:{account_id}", json.dumps(embedding_list), ex=21600)
    warmed += 1
    if warmed % 10000 == 0:
        pipe.execute()
        pipe = r.pipeline()
        print(f"  ...{warmed:,} / {len(df):,}")

pipe.execute()
print(f"\nDone. Pre-warmed Redis with {warmed:,} embeddings.")
print("You can now start gnn_service.py -- Tier 1 (Redis) will be hot")
print("from the very first request, not just from Tier 2 fallback.")