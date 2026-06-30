# run_cross_bank_demo_fast.py
#
# Fast, vectorized version of the cross-bank federated detection
# simulation. The original run_cross_bank_demo.py called the full
# Redis-backed CEP engine (10 patterns, cooldowns, rolling windows)
# up to twice per row using df.iterrows() -- correct, but far too
# slow for a 500K+ row demo run (each check_patterns() call does
# several Redis round-trips; iterrows() alone is ~100x slower than
# vectorized pandas for large frames).
#
# THIS VERSION:
# - Uses vectorized pandas filtering instead of row-by-row iteration
#   to split transactions into Bank A's view and Bank B's view.
# - Uses pd.to_datetime() once on the whole column instead of calling
#   strptime() per row.
# - Replaces the full 10-pattern Redis-backed CEP engine with fast,
#   in-memory aggregate rules computed via groupby() -- conceptually
#   the same signal (high fan-in/fan-out, high transaction velocity)
#   but computed in milliseconds across the whole dataset at once
#   instead of incrementally per-transaction.
#
# This script is for the DEMO NARRATIVE (proving the federated
# correlation architecture works), not for replacing the real-time
# production cep_engine.py, which still runs per-transaction in
# consumer.py exactly as before. Nothing here changes the production
# detection logic.

import sys
import time
import json
import pandas as pd
import numpy as np
import redis

from cross_bank_simulation import (
    write_bank_alert,
    correlate_cross_bank_alerts,
    trace_cross_bank_money_flow,
)

CSV_PATH = "ibm_aml_small.csv"

# Lightweight, vectorized stand-ins for the heaviest-signal CEP patterns.
# Same conceptual checks as cep_engine.py's FAN_IN / FAN_OUT_SPIKE /
# GATHER_SCATTER, computed via groupby aggregation across the whole
# sample at once instead of incrementally per transaction.
FAN_IN_MIN_SENDERS      = 8    # unique senders to one account
FAN_OUT_MIN_RECEIVERS   = 8    # unique receivers from one account
HIGH_VELOCITY_MIN_TXNS  = 15   # total transactions for one account in sample


def find_top_bank_codes(csv_path: str, n: int = 10, sample_size: int = None):
    """
    sample_size=None means NO ROW LIMIT -- scan the entire file.
    This must stay consistent with how run_simulation_fast interprets
    sample_size=None (also "no limit"), otherwise this function and
    the simulation can silently look at two different row windows
    and pick bank codes that don't actually exist in the window the
    simulation processes.

    dtype=str is forced here to match the main simulation's load --
    without this, pandas infers numeric dtype for pure-digit bank
    codes, and the str() conversion of an inferred int64 can differ
    from the str() of a value loaded with dtype=str from the start
    (e.g. if the raw CSV has any leading zeros or whitespace that
    int parsing silently strips but string parsing preserves).
    """
    scope_label = f"first {sample_size:,} rows" if sample_size else "FULL dataset (no row limit)"
    print(f"Scanning {scope_label} for most active bank codes...")
    df = pd.read_csv(csv_path, usecols=['From Bank', 'To Bank'],
                     nrows=sample_size, dtype={'From Bank': str, 'To Bank': str})
    all_banks = pd.concat([df['From Bank'], df['To Bank']])
    top_banks = all_banks.value_counts().head(n)
    print(f"\nTop {n} most active bank codes in this window:")
    for bank, count in top_banks.items():
        print(f"  {bank}: {count:,} transactions")
    return [str(b) for b in top_banks.index]


def compute_fast_flags(df_slice: pd.DataFrame) -> dict:
    """
    Vectorized, in-memory equivalent of the heaviest-signal CEP checks.
    Returns {account_id: [pattern_names]} for every account that trips
    at least one rule, computed via groupby aggregation -- no Redis,
    no per-row Python loop, no cooldowns. This is fast by design: it
    exists purely to generate realistic-looking risk signals quickly
    enough to demonstrate cross-bank correlation on large samples.
    """
    flags = {}

    # FAN_IN equivalent: accounts receiving from many unique senders
    fan_in = df_slice.groupby('account_to')['account_from'].nunique()
    fan_in_flagged = fan_in[fan_in >= FAN_IN_MIN_SENDERS]
    for acct, n_senders in fan_in_flagged.items():
        flags.setdefault(acct, []).append(('FAN_IN', n_senders))

    # FAN_OUT_SPIKE equivalent: accounts sending to many unique receivers
    fan_out = df_slice.groupby('account_from')['account_to'].nunique()
    fan_out_flagged = fan_out[fan_out >= FAN_OUT_MIN_RECEIVERS]
    for acct, n_receivers in fan_out_flagged.items():
        flags.setdefault(acct, []).append(('FAN_OUT_SPIKE', n_receivers))

    # HIGH_VELOCITY equivalent: accounts with high total transaction count
    velocity = df_slice.groupby('account_from').size()
    velocity_flagged = velocity[velocity >= HIGH_VELOCITY_MIN_TXNS]
    for acct, n_txns in velocity_flagged.items():
        flags.setdefault(acct, []).append(('HIGH_VELOCITY', n_txns))

    return flags


def run_simulation_fast(bank_a_code: str, bank_b_code: str, sample_size: int = None):
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    r.ping()
    print("Redis connected\n")

    for prefix in ['bank_a:', 'bank_b:']:
        for key in r.keys(f"{prefix}*"):
            r.delete(key)
    print("Cleared previous simulation state\n")

    print(f"Bank A = '{bank_a_code}', Bank B = '{bank_b_code}'")
    print(f"Loading {'first ' + str(sample_size) + ' rows' if sample_size else 'full dataset'}...\n")

    t0 = time.time()
    cols = ['Account', 'Account.1', 'From Bank', 'To Bank',
            'Amount Paid', 'Timestamp']
    df = pd.read_csv(CSV_PATH, usecols=cols, nrows=sample_size,
                     dtype={'Account': str, 'Account.1': str,
                            'From Bank': str, 'To Bank': str})
    df = df.rename(columns={
        'Account': 'account_from', 'Account.1': 'account_to',
        'From Bank': 'bank_from', 'To Bank': 'bank_to',
        'Amount Paid': 'amount',
    })
    print(f"Loaded {len(df):,} transactions in {time.time()-t0:.2f}s\n")

    # Vectorized bank-relevance filtering -- a transaction is visible
    # to a bank if that bank is either the sender or receiver side.
    # This single boolean mask operation replaces a 500K-iteration
    # Python loop with one pandas-native pass over the whole column.
    t0 = time.time()
    bank_a_mask = (df['bank_from'] == bank_a_code) | (df['bank_to'] == bank_a_code)
    bank_b_mask = (df['bank_from'] == bank_b_code) | (df['bank_to'] == bank_b_code)
    both_mask   = bank_a_mask & bank_b_mask

    df_a = df[bank_a_mask]
    df_b = df[bank_b_mask]
    print(f"Filtered to bank slices in {time.time()-t0:.2f}s")
    print(f"Bank A node sees: {len(df_a):,} transactions")
    print(f"Bank B node sees: {len(df_b):,} transactions")
    print(f"Visible to both (inter-bank transfers): {both_mask.sum():,}\n")

    # Compute flags for each bank's slice independently and quickly
    t0 = time.time()
    flags_a = compute_fast_flags(df_a)
    flags_b = compute_fast_flags(df_b)
    print(f"Computed pattern flags in {time.time()-t0:.2f}s")
    print(f"Bank A flagged: {len(flags_a):,} accounts")
    print(f"Bank B flagged: {len(flags_b):,} accounts\n")

    # Write flagged accounts into each bank's isolated Redis keyspace --
    # this is the only part that touches Redis, and it's bounded by
    # the (small) number of FLAGGED accounts, not the full transaction
    # count, so it stays fast even on large samples.
    timestamp_now = pd.Timestamp.now().isoformat()
    for acct, pattern_list in flags_a.items():
        for pattern_name, evidence_value in pattern_list:
            write_bank_alert(r, "a", acct, pattern_name, 0.6, timestamp_now,
                            {"evidence_value": int(evidence_value)})
    for acct, pattern_list in flags_b.items():
        for pattern_name, evidence_value in pattern_list:
            write_bank_alert(r, "b", acct, pattern_name, 0.6, timestamp_now,
                            {"evidence_value": int(evidence_value)})

    print("=" * 60)
    print("CENTRAL HUB CORRELATION RESULTS")
    print("=" * 60)
    result = correlate_cross_bank_alerts(r, ["a", "b"])
    print(json.dumps(result, indent=2, default=str))

    if result['cross_bank_overlap_accounts']:
        print("\n" + "=" * 60)
        print("SAMPLE MONEY FLOW TRACE FOR FIRST CROSS-BANK ACCOUNT")
        print("=" * 60)
        first_account = result['cross_bank_overlap_accounts'][0]['account_id']
        trace = trace_cross_bank_money_flow(r, ["a", "b"], first_account)
        print(json.dumps(trace, indent=2, default=str))

    return result


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--find-banks":
        sample_for_scan = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else None
        find_top_bank_codes(CSV_PATH, sample_size=sample_for_scan)
    elif len(sys.argv) > 1 and sys.argv[1].lower() in ("full", "all"):
        # Explicit "run on everything" -- sample=None means pd.read_csv
        # has no nrows limit, processing the full 6.9M row dataset.
        bank_a = sys.argv[2] if len(sys.argv) > 2 else None
        bank_b = sys.argv[3] if len(sys.argv) > 3 else None
        if not bank_a or not bank_b:
            print("Auto-detecting top 2 bank codes (scanning full dataset)...")
            top_banks = find_top_bank_codes(CSV_PATH, n=2, sample_size=None)
            bank_a, bank_b = top_banks[0], top_banks[1]
            print(f"\nUsing Bank A = {bank_a}, Bank B = {bank_b}\n")
        run_simulation_fast(bank_a, bank_b, sample_size=None)
    elif len(sys.argv) > 1:
        sample = int(sys.argv[1]) if sys.argv[1].isdigit() else None
        bank_a = sys.argv[2] if len(sys.argv) > 2 else None
        bank_b = sys.argv[3] if len(sys.argv) > 3 else None

        if not bank_a or not bank_b:
            print("Auto-detecting top 2 bank codes (scanning the SAME "
                  "row window the simulation will process)...")
            top_banks = find_top_bank_codes(CSV_PATH, n=2, sample_size=sample)
            bank_a, bank_b = top_banks[0], top_banks[1]
            print(f"\nUsing Bank A = {bank_a}, Bank B = {bank_b}\n")

        run_simulation_fast(bank_a, bank_b, sample)
    else:
        print("Usage:")
        print("  python run_cross_bank_demo_fast.py --find-banks [sample_size]")
        print("  python run_cross_bank_demo_fast.py <sample_size>")
        print("  python run_cross_bank_demo_fast.py <sample_size> <bank_a> <bank_b>")
        print("  python run_cross_bank_demo_fast.py full")
        print("    Runs on the ENTIRE 6.9M row dataset, no limit")