# cross_bank_simulation_v2.py
#
# N-BANK federated detection simulation -- extends the original 2-bank
# proof of concept to handle an arbitrary number of banks, each running
# independently and feeding only alert-level signals (not raw
# transactions) to a central correlation hub.
#
# WHAT CHANGED FROM THE 2-BANK VERSION:
#
# 1. No hardcoded "a"/"b" bank identifiers. Any bank code present in
#    the dataset can be added to the simulation by listing it in
#    BANK_IDS. The correlation hub works identically whether there
#    are 2 banks or 50.
#
# 2. Uses the REAL cep_engine.py (all 10 validated patterns: FAN_IN,
#    FAN_OUT_SPIKE, GATHER_SCATTER, SCATTER_GATHER, SIMPLE_CYCLE,
#    STRUCTURING, SMURFING_RING, TEMPORAL_LAYERING,
#    RANDOM_STRUCTURING, CROSS_BANK_ACCELERATION) instead of a
#    lightweight 3-rule approximation. Each bank's node is namespaced
#    so its CEP state (rolling windows, cooldowns, daily aggregates)
#    never collides with another bank's node -- this matters because
#    cep_engine.py's internal Redis keys are account-keyed globally,
#    and an account participating in transactions visible to multiple
#    banks needs each bank's view of that account's risk to be
#    computed and stored independently, not overwritten by whichever
#    bank's loop ran last.
#
# WHAT THIS DOES NOT SOLVE (documented honestly, not pretended away):
#
# - Real network/infrastructure isolation. Every bank node here still
#   reads the same CSV file and writes to the same Redis instance,
#   just with different key prefixes. A production deployment runs
#   each bank's node on physically separate infrastructure inside
#   that bank's own data center, communicating with the central hub
#   only over an authenticated API -- that cannot be demonstrated on
#   a single laptop and is correctly scoped as Phase 2 infrastructure
#   work, not something this script claims to prove.
#
# - Model-weight federation (federated learning in the ML sense).
#   This script federates ALERTS (account ID + pattern + score) across
#   banks. It does not federate model training -- each bank is not
#   training its own XGBoost/GNN and contributing gradient updates to
#   a shared global model via FedAvg. That is a separate, larger body
#   of work (see ARCHITECTURE.md Section 7.5) and is not claimed here.
#
# - Cryptographic authentication between bank nodes and the hub. In
#   this simulation, "bank isolation" is enforced only by Redis key
#   prefixing, which has no security property -- any code with Redis
#   access could read another bank's keyspace. Production deployment
#   requires mutual TLS and signed API requests between each bank's
#   node and the central hub; this is infrastructure work, not
#   detection-logic work, and is out of scope for this script.

import sys
import json
import time
import pandas as pd
import redis
from datetime import datetime
from typing import List, Dict

import cep_engine


CSV_PATH = "ibm_aml_small.csv"


def get_bank_prefix(bank_id: str) -> str:
    """
    Redis key prefix for a given bank's isolated simulation namespace.
    bank_id here is the actual bank code from the dataset (e.g. "070"),
    sanitized for use in a Redis key (Redis keys cannot contain
    whitespace or certain special characters, though plain digits are
    always safe).
    """
    return f"bankX_{bank_id}:"


def find_top_bank_codes(csv_path: str, n: int, sample_size: int = None) -> List[str]:
    scope = f"first {sample_size:,} rows" if sample_size else "FULL dataset"
    print(f"Scanning {scope} for top {n} most active bank codes...")
    df = pd.read_csv(csv_path, usecols=['From Bank', 'To Bank'],
                     nrows=sample_size, dtype={'From Bank': str, 'To Bank': str})
    all_banks = pd.concat([df['From Bank'], df['To Bank']])
    top_banks = all_banks.value_counts().head(n)
    print(f"\nTop {n} bank codes in this window:")
    for bank, count in top_banks.items():
        print(f"  {bank}: {count:,} transactions")
    return [str(b) for b in top_banks.index]


def is_relevant_to_bank(bank_from: str, bank_to: str, bank_id: str) -> bool:
    """A bank sees a transaction if it is sender OR receiver."""
    return bank_from == bank_id or bank_to == bank_id


def write_bank_alert(r: redis.Redis, bank_id: str, account_id: str,
                     pattern_type: str, evidence: dict, timestamp: str):
    """Writes an alert into one bank's isolated keyspace."""
    prefix = get_bank_prefix(bank_id)
    alert = {
        "account_id":   account_id,
        "pattern_type": pattern_type,
        "timestamp":    timestamp,
        "bank_id":      bank_id,
        "evidence":     evidence,
    }
    pipe = r.pipeline()
    pipe.lpush(f"{prefix}alerts", json.dumps(alert))
    pipe.ltrim(f"{prefix}alerts", 0, 4999)
    pipe.sadd(f"{prefix}flagged_accounts", account_id)
    pipe.sadd(f"{prefix}flagged_patterns:{account_id}", pattern_type)
    pipe.execute()


def get_bank_flagged_accounts(r: redis.Redis, bank_id: str) -> set:
    return r.smembers(f"{get_bank_prefix(bank_id)}flagged_accounts")


def correlate_n_bank_alerts(r: redis.Redis, bank_ids: List[str]) -> dict:
    """
    Central hub correlation across N banks (not just 2). Finds every
    account flagged by 2 or more independent bank nodes, and reports
    exactly which banks flagged it and which patterns fired at each.

    This is the same logical operation as the 2-bank version --
    set intersection of independently-derived risk signals -- but
    generalized to operate over any number of bank node identifiers
    passed in bank_ids, with no code change required to add a third,
    fourth, or fiftieth bank to the simulation.
    """
    per_bank_flagged = {
        bank_id: get_bank_flagged_accounts(r, bank_id)
        for bank_id in bank_ids
    }

    account_to_banks: Dict[str, List[str]] = {}
    for bank_id, accounts in per_bank_flagged.items():
        for acct in accounts:
            account_to_banks.setdefault(acct, []).append(bank_id)

    multi_bank = {
        acct: banks for acct, banks in account_to_banks.items()
        if len(banks) >= 2
    }

    # For each multi-bank account, pull which patterns fired at each bank
    multi_bank_detail = []
    for acct, banks in multi_bank.items():
        bank_pattern_map = {}
        for bank_id in banks:
            prefix = get_bank_prefix(bank_id)
            patterns = r.smembers(f"{prefix}flagged_patterns:{acct}")
            bank_pattern_map[bank_id] = sorted(patterns)
        multi_bank_detail.append({
            "account_id":       acct,
            "flagged_by_banks": sorted(banks),
            "bank_count":       len(banks),
            "patterns_per_bank": bank_pattern_map,
        })

    multi_bank_detail.sort(key=lambda x: -x['bank_count'])

    return {
        "bank_ids_in_simulation": bank_ids,
        "per_bank_counts": {b: len(a) for b, a in per_bank_flagged.items()},
        "total_unique_flagged_accounts": len(account_to_banks),
        "cross_bank_overlap_count": len(multi_bank),
        "cross_bank_overlap_accounts": multi_bank_detail[:100],
        "max_banks_flagging_single_account": max(
            (d['bank_count'] for d in multi_bank_detail), default=0
        ),
        "interpretation": (
            f"{len(multi_bank)} accounts were independently flagged by "
            f"2 or more of the {len(bank_ids)} simulated bank nodes. "
            f"No bank's node read another bank's raw transaction data "
            f"at any point -- only account-level alert signals "
            f"(account ID + pattern type) were available to the "
            f"correlation hub, exactly matching the data-sharing "
            f"constraint RBI's data localization rules impose on a "
            f"real multi-bank deployment."
        ) if multi_bank else (
            f"No cross-bank overlap found across {len(bank_ids)} "
            f"simulated banks in current data."
        ),
    }


def run_n_bank_simulation(bank_ids: List[str], sample_size: int = None):
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    r.ping()
    print(f"Redis connected. Simulating {len(bank_ids)} banks: {bank_ids}\n")

    # Clean prior simulation state for these specific banks only
    for bank_id in bank_ids:
        prefix = get_bank_prefix(bank_id)
        for key in r.keys(f"{prefix}*"):
            r.delete(key)
    print("Cleared previous simulation state for these banks\n")

    t0 = time.time()
    cols = ['Account', 'Account.1', 'From Bank', 'To Bank',
            'Amount Paid', 'Timestamp']
    df = pd.read_csv(CSV_PATH, usecols=cols, nrows=sample_size,
                     dtype={'Account': str, 'Account.1': str,
                            'From Bank': str, 'To Bank': str})
    print(f"Loaded {len(df):,} transactions in {time.time()-t0:.2f}s\n")

    # Pre-parse timestamps once, vectorized -- not per-row inside the loop.
    # cep_engine.py expects ISO format strings; the raw CSV format is
    # '2022/09/01 00:08' (slash-separated, no seconds).
    t0 = time.time()
    df['iso_ts'] = pd.to_datetime(
        df['Timestamp'], format='%Y/%m/%d %H:%M'
    ).dt.strftime('%Y-%m-%dT%H:%M:%S')
    print(f"Vectorized timestamp parsing: {time.time()-t0:.2f}s\n")

    # Rename columns to valid Python identifiers BEFORE itertuples().
    # itertuples() only gives clean attribute names (row.Account) for
    # columns that are already valid identifiers; columns with spaces
    # or dots ('From Bank', 'Account.1') get renamed to positional
    # placeholders like _1, _2, _3 -- and critically, the FIRST
    # renamed column becomes _1, not _2, regardless of its original
    # position in the DataFrame. Relying on guessed positional names
    # here previously pulled data from the wrong columns entirely with
    # no error, since every field was still a valid string or number,
    # just the WRONG one. Renaming up front make this impossible to
    # get wrong, since every column then has its real name as the
    # attribute name.
    df = df.rename(columns={
        'Account.1':   'account_to_raw',
        'From Bank':    'bank_from_raw',
        'To Bank':      'bank_to_raw',
        'Amount Paid':  'amount_raw',
    })

    per_bank_counts = {b: 0 for b in bank_ids}

    print("Processing transactions through real cep_engine.py "
          "(this is the slow, accurate path -- not the fast "
          "approximation used in the speed-test version)...")
    t0 = time.time()
    total_rows = len(df)
    report_every = max(total_rows // 20, 1)

    for i, row in enumerate(df.itertuples(index=False)):
        bank_from = row.bank_from_raw
        bank_to   = row.bank_to_raw

        relevant_banks = [
            b for b in bank_ids if is_relevant_to_bank(bank_from, bank_to, b)
        ]
        if not relevant_banks:
            continue

        event = {
            "account_from": row.Account,
            "account_to":   row.account_to_raw,
            "bank_from":    bank_from,
            "bank_to":      bank_to,
            "amount_paid":  row.amount_raw,
            "timestamp":    row.iso_ts,
        }

        for bank_id in relevant_banks:
            per_bank_counts[bank_id] += 1
            # NAMESPACE ISOLATION: cep_engine.py's internal keys (account
            # windows, cooldowns, daily aggregates) are account-keyed
            # globally, not bank-keyed. To give each bank's node a truly
            # independent view of the SAME account's risk (which is what
            # real federated banks would each compute from their own data
            # slice), we suffix the account IDs with the bank ID before
            # calling check_patterns(). This means Bank A's CEP state for
            # account X and Bank B's CEP state for account X are tracked
            # as if they were different accounts internally -- which is
            # correct: each bank only sees ITS OWN transactions for
            # account X, and would compute different behavioral features
            # from that partial view than another bank with a different
            # partial view of the same account's activity.
            namespaced_event = dict(event)
            namespaced_event['account_from'] = f"{event['account_from']}__{bank_id}"
            namespaced_event['account_to']   = f"{event['account_to']}__{bank_id}"

            alerts = cep_engine.check_patterns(r, namespaced_event)
            for alert in alerts:
                # Strip the bank suffix back off before recording --
                # the OUTPUT alert should reference the real account ID,
                # only the internal CEP working-state was namespaced.
                real_account_id = alert['account_id'].rsplit('__', 1)[0]
                write_bank_alert(
                    r, bank_id, real_account_id, alert['pattern_type'],
                    alert.get('evidence', {}), event['timestamp']
                )

        if (i + 1) % report_every == 0:
            pct = (i + 1) / total_rows * 100
            print(f"  {i+1:,} / {total_rows:,} rows ({pct:.0f}%) "
                  f"-- {time.time()-t0:.1f}s elapsed")

    print(f"\nFull CEP pass completed in {time.time()-t0:.1f}s\n")

    for bank_id in bank_ids:
        print(f"Bank '{bank_id}' node processed: {per_bank_counts[bank_id]:,} transactions")

    print()
    print("=" * 60)
    print(f"CENTRAL HUB CORRELATION RESULTS -- {len(bank_ids)} BANKS")
    print("=" * 60)
    result = correlate_n_bank_alerts(r, bank_ids)
    print(json.dumps(result, indent=2, default=str))

    return result


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--find-banks":
        n = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 10
        sample = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else None
        find_top_bank_codes(CSV_PATH, n=n, sample_size=sample)

    elif len(sys.argv) > 1:
        # First arg: sample size (digits) or "full"
        if sys.argv[1].lower() == "full":
            sample_size = None
            remaining_args = sys.argv[2:]
        elif sys.argv[1].isdigit():
            sample_size = int(sys.argv[1])
            remaining_args = sys.argv[2:]
        else:
            sample_size = None
            remaining_args = sys.argv[1:]

        if remaining_args:
            # Explicit bank codes given
            bank_ids = remaining_args
        else:
            # Auto-detect top N banks (default N=3 for a meaningful
            # N-bank demo -- more convincing than just 2)
            num_banks = 3
            print(f"Auto-detecting top {num_banks} bank codes...")
            bank_ids = find_top_bank_codes(CSV_PATH, n=num_banks, sample_size=sample_size)
            print(f"\nUsing banks: {bank_ids}\n")

        run_n_bank_simulation(bank_ids, sample_size)

    else:
        print("Usage:")
        print("  python cross_bank_simulation_v2.py --find-banks [n] [sample_size]")
        print("    Lists top n bank codes (default 10) in that row window")
        print()
        print("  python cross_bank_simulation_v2.py <sample_size|full> [bank1 bank2 bank3 ...]")
        print("    Runs N-bank simulation. If no bank codes given, auto-picks top 3.")
        print()
        print("  Examples:")
        print("    python cross_bank_simulation_v2.py 100000")
        print("      Auto-picks top 3 banks, runs on first 100K rows")
        print("    python cross_bank_simulation_v2.py 100000 070 020 011 014")
        print("      Runs with 4 specific banks on first 100K rows")
        print("    python cross_bank_simulation_v2.py full 070 020")
        print("      Runs 2 specific banks on the FULL 6.9M row dataset")