"""
cep_polars_validator.py — High-speed offline CEP validation using Polars.

Implements all four CEP patterns as pure vectorized Polars operations
against the sorted transaction DataFrame. No Redis, no row-by-row
iteration, no Python loops over individual transactions.

Benchmark target: 6M rows in under 5 minutes.
Pandas/Redis baseline: 98 rows/sec (~17 hours for 6M rows).

Pattern implementations:
  RAPID_FORWARD         — self-join on account: receive then send
                          within 15 minutes, amount >= Rs.100
  FAN_OUT_SPIKE         — rolling 6-hour unique destination bank
                          count >= 10, per sender account
  CONCENTRATED_DESTINATION — rolling 6-hour transfer count to same
                          destination >= 5, per sender account
  BURST_ACTIVATION      — dormancy gap >= 30 days then 5+ transfers
                          within 1 hour, per account

Key difference from cep_engine.py (the production Redis-based engine):
  This script tests whether the PATTERN LOGIC is correct against the
  full dataset. It does not test the production code path. Both should
  agree on which accounts are flagged; if they diverge significantly,
  that is a signal worth investigating.

Usage:
    python cep_polars_validator.py               # full dataset
    python cep_polars_validator.py --sample 500000  # first 500K rows
"""

import argparse
import time
from datetime import datetime
import polars as pl

CSV_PATH = "ibm_aml_small.csv"


def load_data(sample_size=None):
    print(f"  Loading CSV with Polars lazy scan...")
    t0 = time.time()

    # Use lazy scan — Polars reads only needed columns, pushes down
    # predicates, never materializes the full CSV unless needed
    lf = pl.scan_csv(
        CSV_PATH,
        try_parse_dates=False,   # we parse manually for control
    )

    # Select and rename only what we need
    lf = lf.select([
        pl.col("Timestamp").alias("timestamp_raw"),
        pl.col("Account").alias("account_from"),
        pl.col("Account_duplicated_0").alias("account_to"),
        pl.col("From Bank").cast(pl.Utf8).alias("bank_from"),
        pl.col("To Bank").cast(pl.Utf8).alias("bank_to"),
        pl.col("Amount Paid").alias("amount"),
        pl.col("Is Laundering").alias("is_laundering"),
    ])

    df = lf.collect()

    # Parse timestamps — CSV format is '2022/09/01 00:00'
    df = df.with_columns(
        pl.col("timestamp_raw")
          .str.replace_all("/", "-")
          .str.replace(" ", "T")
          .str.replace(r"T(\d{2}:\d{2})$", r"T${1}:00")
          .str.to_datetime(format="%Y-%m-%dT%H:%M:%S")
          .alias("timestamp")
    ).drop("timestamp_raw")

    # Sort by timestamp — critical for rolling window correctness
    df = df.sort("timestamp")

    if sample_size:
        df = df.head(sample_size)

    print(f"  Loaded and sorted {len(df):,} rows in {time.time()-t0:.1f}s")
    return df


def get_laundering_accounts(df):
    """All confirmed laundering account IDs from both sender and receiver."""
    launder = df.filter(pl.col("is_laundering") == 1)
    senders = launder.select("account_from").rename({"account_from": "account_id"})
    receivers = launder.select("account_to").rename({"account_to": "account_id"})
    all_accounts = pl.concat([senders, receivers]).unique()
    return set(all_accounts["account_id"].to_list())


def detect_rapid_forward(df):
    """
    RAPID_FORWARD: account receives funds, then sends >= Rs.100
    within 15 minutes.

    Approach: self-join the DataFrame on account.
      Left side  = rows where account appears as RECEIVER (account_to)
      Right side = rows where account appears as SENDER (account_from)
    Join condition: same account, right.timestamp > left.timestamp,
    right.timestamp - left.timestamp <= 15 minutes,
    right.amount >= 100.
    """
    print("  Computing RAPID_FORWARD...")
    t0 = time.time()

    WINDOW_MINUTES = 15
    MIN_AMOUNT = 100.0

    # Build receive events: when did each account receive money?
    received = df.select([
        pl.col("account_to").alias("account"),
        pl.col("timestamp").alias("received_at"),
        pl.col("amount").alias("received_amount"),
    ])

    # Build send events: when did each account send money >= Rs.100?
    sent = df.filter(pl.col("amount") >= MIN_AMOUNT).select([
        pl.col("account_from").alias("account"),
        pl.col("timestamp").alias("sent_at"),
        pl.col("amount").alias("sent_amount"),
    ])

    # Join on account, then filter to events within the 15-minute window
    joined = received.join(sent, on="account", how="inner")
    joined = joined.filter(
        (pl.col("sent_at") > pl.col("received_at")) &
        ((pl.col("sent_at") - pl.col("received_at"))
         .dt.total_minutes() <= WINDOW_MINUTES)
    )

    flagged = set(joined["account"].unique().to_list())
    print(f"    {len(flagged):,} accounts flagged in {time.time()-t0:.1f}s")
    return flagged


def detect_fan_out_spike(df):
    """
    FAN_OUT_SPIKE: 10+ unique destination banks within a rolling
    6-hour window, per sender account.

    Approach: for each transaction, count unique destination banks
    in the 6 hours preceding it for the same sender account.
    Use a groupby_dynamic rolling window.
    """
    print("  Computing FAN_OUT_SPIKE...")
    t0 = time.time()

    MIN_BANKS = 10
    WINDOW_HOURS = 6

    # Sort by account_from then timestamp for groupby_dynamic
    sends = df.select([
        pl.col("account_from"),
        pl.col("timestamp"),
        pl.col("bank_to"),
    ]).sort(["account_from", "timestamp"])

    # Polars groupby_dynamic: for each account, rolling 6h window
    # Count unique destination banks in that window
    result = (
        sends
        .group_by_dynamic(
            "timestamp",
            every="1s",           # evaluate at every transaction
            period=f"{WINDOW_HOURS}h",
            closed="right",
            group_by="account_from",
        )
        .agg(pl.col("bank_to").n_unique().alias("unique_banks"))
        .filter(pl.col("unique_banks") >= MIN_BANKS)
    )

    flagged = set(result["account_from"].unique().to_list())
    print(f"    {len(flagged):,} accounts flagged in {time.time()-t0:.1f}s")
    return flagged


def detect_concentrated_destination(df):
    """
    CONCENTRATED_DESTINATION: 5+ transfers to the same destination
    account within a rolling 6-hour window.

    Approach: group by (account_from, account_to), use rolling 6h
    window, count transfers, flag where count >= 5.
    """
    print("  Computing CONCENTRATED_DESTINATION...")
    t0 = time.time()

    MIN_COUNT = 5
    WINDOW_HOURS = 6

    sends = df.select([
        pl.col("account_from"),
        pl.col("account_to"),
        pl.col("timestamp"),
    ]).sort(["account_from", "account_to", "timestamp"])

    # Add a count column of 1 per transaction for rolling sum
    sends = sends.with_columns(pl.lit(1).alias("count"))

    result = (
        sends
        .group_by_dynamic(
            "timestamp",
            every="1s",
            period=f"{WINDOW_HOURS}h",
            closed="right",
            group_by=["account_from", "account_to"],
        )
        .agg(pl.col("count").sum().alias("transfer_count"))
        .filter(pl.col("transfer_count") >= MIN_COUNT)
    )

    flagged = set(result["account_from"].unique().to_list())
    print(f"    {len(flagged):,} accounts flagged in {time.time()-t0:.1f}s")
    return flagged


def detect_burst_activation(df):
    """
    BURST_ACTIVATION: dormant 30+ days, then 5+ transfers within 1 hour.

    Approach:
    1. Per account, compute time since previous transaction (lag).
    2. Flag rows where lag >= 30 days (dormancy events).
    3. For each dormancy event, count subsequent transfers within 1 hour.
    4. Flag accounts where that count >= 5.
    """
    print("  Computing BURST_ACTIVATION...")
    t0 = time.time()

    DORMANCY_DAYS = 30
    BURST_WINDOW_HOURS = 1
    MIN_COUNT = 5

    # All activity per account (both sending and receiving)
    activity = pl.concat([
        df.select(pl.col("account_from").alias("account"), pl.col("timestamp")),
        df.select(pl.col("account_to").alias("account"), pl.col("timestamp")),
    ]).sort(["account", "timestamp"])

    # Compute lag (time since previous transaction for same account)
    activity = activity.with_columns(
        pl.col("timestamp")
          .shift(1)
          .over("account")
          .alias("prev_timestamp")
    ).with_columns(
        ((pl.col("timestamp") - pl.col("prev_timestamp"))
         .dt.total_seconds() / 86400)
        .alias("gap_days")
    )

    # Find dormancy reactivation events (gap >= 30 days)
    reactivations = activity.filter(pl.col("gap_days") >= DORMANCY_DAYS).select([
        pl.col("account"),
        pl.col("timestamp").alias("reactivation_at"),
    ])

    if len(reactivations) == 0:
        print(f"    0 accounts flagged in {time.time()-t0:.1f}s")
        return set()

    # For each reactivation, count sends within the next 1 hour
    sends = df.select([
        pl.col("account_from").alias("account"),
        pl.col("timestamp"),
    ])

    # Join reactivations to sends on account
    joined = reactivations.join(sends, on="account", how="inner")
    joined = joined.filter(
        (pl.col("timestamp") >= pl.col("reactivation_at")) &
        ((pl.col("timestamp") - pl.col("reactivation_at"))
         .dt.total_hours() <= BURST_WINDOW_HOURS)
    )

    # Count burst sends per reactivation event
    burst_counts = (
        joined
        .group_by(["account", "reactivation_at"])
        .agg(pl.len().alias("burst_count"))
        .filter(pl.col("burst_count") >= MIN_COUNT)
    )

    flagged = set(burst_counts["account"].unique().to_list())
    print(f"    {len(flagged):,} accounts flagged in {time.time()-t0:.1f}s")
    return flagged


def run_validation(sample_size=None):
    print(f"\n{'='*70}")
    print("CEP POLARS OFFLINE VALIDATOR")
    print(f"{'='*70}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[1/3] Loading and preparing data...")
    df = load_data(sample_size)

    # Build ground truth sets
    laundering_accounts = get_laundering_accounts(df)
    total_accounts = len(
        set(df["account_from"].unique().to_list()) |
        set(df["account_to"].unique().to_list())
    )
    base_rate = len(laundering_accounts) / total_accounts * 100
    print(f"  Confirmed laundering accounts: {len(laundering_accounts):,}")
    print(f"  Total unique accounts: {total_accounts:,}")
    print(f"  Base rate: {base_rate:.4f}%")

    # ── Run pattern detection ─────────────────────────────────────────────────
    print("\n[2/3] Running vectorized pattern detection...")
    t_patterns = time.time()

    flagged_by_pattern = {
        "RAPID_FORWARD":             detect_rapid_forward(df),
        "FAN_OUT_SPIKE":             detect_fan_out_spike(df),
        "CONCENTRATED_DESTINATION":  detect_concentrated_destination(df),
        "BURST_ACTIVATION":          detect_burst_activation(df),
    }

    all_flagged = set()
    for accounts in flagged_by_pattern.values():
        all_flagged |= accounts

    print(f"\n  All 4 patterns completed in {time.time()-t_patterns:.1f}s")

    # ── Compute precision ─────────────────────────────────────────────────────
    print("\n[3/3] Computing precision against ground truth labels...")

    confirmed = all_flagged & laundering_accounts
    overall_precision = len(confirmed) / len(all_flagged) * 100 if all_flagged else 0
    lift = overall_precision / base_rate if base_rate > 0 else 0

    # Per-pattern precision
    pattern_stats = {}
    for pattern, accounts in flagged_by_pattern.items():
        confirmed_p = accounts & laundering_accounts
        precision_p = len(confirmed_p) / len(accounts) * 100 if accounts else 0
        pattern_stats[pattern] = {
            "total":     len(accounts),
            "confirmed": len(confirmed_p),
            "precision": precision_p,
        }

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("CEP POLARS VALIDATION REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Rows processed: {len(df):,}")
    print(f"Sample: {'FULL DATASET' if not sample_size else f'First {sample_size:,} rows'}")
    print(f"{'='*70}")
    print(f"\nTotal accounts flagged by CEP:       {len(all_flagged):,}")
    print(f"Confirmed laundering among flagged:  {len(confirmed):,}")
    print(f"CEP overall precision:               {overall_precision:.2f}%")
    print(f"Dataset base rate:                   {base_rate:.4f}%")
    print(f"Lift over base rate:                 {lift:.1f}x")

    print("\n--- Per-Pattern Breakdown ---")
    for pattern, stats in sorted(pattern_stats.items(),
                                  key=lambda x: x[1]["precision"], reverse=True):
        print(f"  {pattern:<35} "
              f"accounts: {stats['total']:>6} | "
              f"confirmed: {stats['confirmed']:>4} | "
              f"precision: {stats['precision']:.2f}%")

    print(f"\n{'='*70}")
    print("End of report.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None,
                        help="Only process first N rows. Omit for full dataset.")
    args = parser.parse_args()

    t_total = time.time()
    run_validation(sample_size=args.sample)
    print(f"Total wall time: {time.time()-t_total:.1f}s")