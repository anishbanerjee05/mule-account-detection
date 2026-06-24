"""
cep_offline_validator.py — Offline batch validation of CEP patterns.

Replays the full IBM AML transaction CSV through cep_engine.py's
check_patterns() directly, bypassing Kafka, GNN, and XGBoost entirely.
This gives statistically meaningful CEP precision numbers against the
full dataset in minutes instead of the 33+ hours a live pipeline run
would take.

What this does NOT test:
  - GNN embedding quality
  - XGBoost model scoring
  - Kafka throughput
  - The full end-to-end pipeline

What this DOES test:
  - CEP pattern detection logic correctness
  - CEP precision against ground truth labels
  - Which confirmed-laundering accounts the CEP engine catches
    that a model-only approach would miss

Run this AFTER resetting the environment (reset_demo_environment.ps1)
so Redis is clean. The script flushes its own CEP keys at the start
to ensure a clean slate.

Usage:
    python cep_offline_validator.py
    python cep_offline_validator.py --sample 100000   # first 100K rows
"""

import argparse
import json
import time
import pandas as pd
import redis
from datetime import datetime

# Import CEP engine directly — no Kafka, no GNN, no XGBoost
from cep_engine import check_patterns

# ── Config ───────────────────────────────────────────────────────────────────
CSV_PATH = "ibm_aml_small.csv"
CHUNK_SIZE = 50_000        # rows per chunk — avoids loading 650MB at once
PROGRESS_EVERY = 10_000   # print progress every N rows
LAUNDERING_LABEL_COL = "Is Laundering"

# Column mapping: CSV column name -> what check_patterns() expects
COL_MAP = {
    "account_from":  "Account",
    "account_to":    "Account.1",
    "bank_from":     "From Bank",
    "bank_to":       "To Bank",
    "amount_paid":   "Amount Paid",
    "timestamp":     "Timestamp",
}


def flush_cep_keys(r: redis.Redis):
    """Remove all CEP-related keys from Redis for a clean validation run."""
    patterns = [
        "cep_alerts:*", "cep_cooldown:*", "cep_last_seen:*",
        "acct_in:*", "acct_window:*"
    ]
    total = 0
    for pattern in patterns:
        keys = r.keys(pattern)
        if keys:
            r.delete(*keys)
            total += len(keys)
    print(f"  Flushed {total:,} CEP-related Redis keys")


def build_laundering_set(df_full: pd.DataFrame) -> set:
    """Return set of all confirmed laundering account IDs from full dataset."""
    mask = df_full[LAUNDERING_LABEL_COL] == 1
    accts = set(df_full.loc[mask, "Account"].astype(str).unique())
    accts |= set(df_full.loc[mask, "Account.1"].astype(str).unique())
    return accts


def run_validation(sample_size: int = None):
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    r.ping()
    print("Redis connected")

    # ── Step 1: flush CEP keys for a clean run ───────────────────────────────
    print("\n[1/5] Flushing existing CEP Redis keys...")
    flush_cep_keys(r)

    # ── Step 2: build ground truth set from full CSV ─────────────────────────
    print("\n[2/5] Loading ground truth labels from full CSV...")
    t0 = time.time()
    # Read only the columns we need for ground truth — much faster than
    # loading the full CSV into memory at once
    gt_df = pd.read_csv(CSV_PATH, usecols=["Account", "Account.1", LAUNDERING_LABEL_COL])
    laundering_accounts = build_laundering_set(gt_df)
    total_accounts = len(set(gt_df["Account"].unique()) | set(gt_df["Account.1"].unique()))
    print(f"  Confirmed laundering accounts: {len(laundering_accounts):,}")
    print(f"  Total unique accounts: {total_accounts:,}")
    print(f"  Base rate: {len(laundering_accounts)/total_accounts*100:.4f}%")
    print(f"  Loaded in {time.time()-t0:.1f}s")
    del gt_df  # free memory before loading full dataset

    # ── Step 3: replay transactions through CEP engine ───────────────────────
    print("\n[3/5] Replaying transactions through CEP engine...")
    print("  (Sorted by Timestamp — required for correct rolling windows)")

    t1 = time.time()
    total_rows = 0
    cep_fires = 0

    # Read in chunks to avoid loading 650MB at once
    chunks = []
    print("  Loading CSV in chunks...")
    for chunk in pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE):
        chunks.append(chunk)
        total_rows += len(chunk)
        if sample_size and total_rows >= sample_size:
            break
        if total_rows % 500_000 == 0:
            print(f"  Loaded {total_rows:,} rows...")

    print(f"  Loaded {total_rows:,} rows in {time.time()-t1:.1f}s. Sorting...")
    df = pd.concat(chunks, ignore_index=True)
    if sample_size:
        df = df.head(sample_size)

    # Sort by timestamp — absolutely critical for rolling window correctness.
    # Transactions arriving out of order would cause RAPID_FORWARD to fire
    # on events where the "forward" actually happened before the "receive",
    # which is physically impossible and would inflate false positives.
    df = df.sort_values("Timestamp").reset_index(drop=True)
    print(f"  Sorted {len(df):,} rows. Starting replay...")

    t2 = time.time()
    processed = 0

    for _, row in df.iterrows():
        # CSV timestamps are '2022/09/01 00:00' -- convert to ISO format
        # that cep_engine._parse_ts() expects: '2022-09-01T00:00:00'
        ts_raw = str(row["Timestamp"])
        ts_iso = ts_raw.replace("/", "-").replace(" ", "T")
        if ts_iso.count(":") == 1:
            ts_iso += ":00"
        event = {
            "account_from": str(row["Account"]),
            "account_to":   str(row["Account.1"]),
            "bank_from":    str(row["From Bank"]),
            "bank_to":      str(row["To Bank"]),
            "amount_paid":  float(row["Amount Paid"]),
            "timestamp":    ts_iso,
        }
        alerts = check_patterns(r, event)
        cep_fires += len(alerts)
        processed += 1

        if processed % PROGRESS_EVERY == 0:
            elapsed = time.time() - t2
            rate = processed / elapsed
            remaining = (len(df) - processed) / rate
            print(f"  {processed:,}/{len(df):,} rows | "
                  f"{rate:.0f} rows/sec | "
                  f"~{remaining/60:.1f} min remaining | "
                  f"{cep_fires} CEP fires so far")

    elapsed_total = time.time() - t2
    print(f"\n  Replay complete: {processed:,} rows in {elapsed_total:.1f}s "
          f"({processed/elapsed_total:.0f} rows/sec)")
    print(f"  Total CEP pattern fires: {cep_fires}")

    # ── Step 4: collect and analyze results ──────────────────────────────────
    print("\n[4/5] Collecting CEP alert results from Redis...")

    cep_keys = r.keys("cep_alerts:*")
    cep_flagged_accounts = {k.split(":", 1)[1] for k in cep_keys}

    pattern_counts = {}
    cep_account_patterns = {}  # account_id -> list of pattern types

    for account_id in cep_flagged_accounts:
        alerts_raw = r.lrange(f"cep_alerts:{account_id}", 0, -1)
        patterns = [json.loads(a)["pattern_type"] for a in alerts_raw]
        cep_account_patterns[account_id] = patterns
        for p in patterns:
            pattern_counts[p] = pattern_counts.get(p, 0) + 1

    # ── Step 5: precision calculation ────────────────────────────────────────
    print("\n[5/5] Computing precision against ground truth labels...")

    confirmed_catches = cep_flagged_accounts & laundering_accounts
    precision = len(confirmed_catches) / len(cep_flagged_accounts) * 100 if cep_flagged_accounts else 0
    base_rate = len(laundering_accounts) / total_accounts * 100
    lift = precision / base_rate if base_rate > 0 else 0

    # Per-pattern precision
    pattern_precision = {}
    for pattern in pattern_counts:
        pattern_accounts = {
            a for a, pats in cep_account_patterns.items() if pattern in pats
        }
        confirmed = pattern_accounts & laundering_accounts
        pattern_precision[pattern] = {
            "total_accounts": len(pattern_accounts),
            "confirmed_fraud": len(confirmed),
            "precision_pct": len(confirmed) / len(pattern_accounts) * 100 if pattern_accounts else 0
        }

    # ── Print full report ─────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("CEP OFFLINE BATCH VALIDATION REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Rows processed: {processed:,}")
    print(f"Sample: {'FULL DATASET' if not sample_size else f'First {sample_size:,} rows (sorted by timestamp)'}")
    print("="*70)
    print(f"\nTotal accounts flagged by CEP:       {len(cep_flagged_accounts):,}")
    print(f"Confirmed laundering among flagged:  {len(confirmed_catches):,}")
    print(f"CEP overall precision:               {precision:.2f}%")
    print(f"Dataset base rate:                   {base_rate:.4f}%")
    print(f"Lift over base rate:                 {lift:.1f}x")

    print("\n--- Per-Pattern Breakdown ---")
    for pattern, stats in sorted(pattern_precision.items(),
                                  key=lambda x: x[1]["precision_pct"], reverse=True):
        print(f"  {pattern:<35} "
              f"accounts: {stats['total_accounts']:>5} | "
              f"confirmed fraud: {stats['confirmed_fraud']:>4} | "
              f"precision: {stats['precision_pct']:.2f}%")

    print("\n--- Confirmed Fraud Catches (demo-citable) ---")
    for account_id in sorted(confirmed_catches):
        patterns = list(set(cep_account_patterns[account_id]))
        print(f"  {account_id}   patterns: {patterns}")

    print("\n" + "="*70)
    print("End of report.")
    print("="*70)

    return {
        "precision": precision,
        "lift": lift,
        "total_flagged": len(cep_flagged_accounts),
        "confirmed_catches": len(confirmed_catches),
        "pattern_precision": pattern_precision,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None,
                        help="Only process first N rows (sorted by timestamp). "
                             "Omit for full dataset.")
    args = parser.parse_args()
    run_validation(sample_size=args.sample)