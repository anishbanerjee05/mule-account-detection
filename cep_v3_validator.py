# cep_v3_validator_v3.py
# Revised after diagnostic analysis of IBM AML ground truth.
# Each pattern rewritten based on actual fraud account behavior.

import duckdb
import time
import sys

CSV_PATH = "ibm_aml_small.csv"

def connect():
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")
    con.execute("SET threads=4")
    return con

def load_view(con, sample_size=None):
    print("  Loading dataset...")
    t0 = time.time()
    con.execute(f"""
        CREATE OR REPLACE VIEW raw_txns AS
        SELECT
            CAST(Timestamp AS TIMESTAMP)  AS ts,
            CAST(Account    AS VARCHAR)   AS account_from,
            CAST(Account_1  AS VARCHAR)   AS account_to,
            CAST("From Bank" AS VARCHAR)  AS bank_from,
            CAST("To Bank"   AS VARCHAR)  AS bank_to,
            "Amount Paid"                 AS amount,
            "Is Laundering"               AS is_laundering
        FROM read_csv('{CSV_PATH}', header=true, nullstr='', ignore_errors=false)
        {f'LIMIT {sample_size}' if sample_size else ''}
    """)
    n = con.execute("SELECT COUNT(*) FROM raw_txns").fetchone()[0]
    print(f"  {n:,} rows in {time.time()-t0:.1f}s")
    return n

def get_ground_truth(con):
    rows = con.execute("""
        SELECT DISTINCT account_id FROM (
            SELECT account_from AS account_id FROM raw_txns WHERE is_laundering = 1
            UNION
            SELECT account_to   AS account_id FROM raw_txns WHERE is_laundering = 1
        )
    """).fetchall()
    return {r[0] for r in rows}

def evaluate(flagged_accounts, ground_truth, total_accounts, name):
    flagged   = set(flagged_accounts)
    confirmed = flagged & ground_truth
    base_rate = len(ground_truth) / total_accounts if total_accounts > 0 else 0
    precision = len(confirmed) / len(flagged) if flagged else 0
    lift      = precision / base_rate if base_rate > 0 else 0
    print(f"\n  {name}")
    print(f"  {'─'*55}")
    print(f"  Flagged:   {len(flagged):,}")
    print(f"  Confirmed: {len(confirmed):,}")
    print(f"  Precision: {precision:.2%}")
    print(f"  Lift:      {lift:.1f}x  (base: {base_rate:.4%})")
    return {"pattern": name, "flagged": len(flagged),
            "confirmed": len(confirmed), "precision": precision, "lift": lift}


# ── TEMPORAL_LAYERING ─────────────────────────────────────────────────────────
# Diagnostic showed: 3,023 fraud pairs in 0.70-1.10 passthrough range.
# The original query counted pairs not accounts, and the cross-join exploded.
#
# Revised approach: flag accounts that have BOTH:
#   (a) at least one day where they received funds
#   (b) a LATER day (1-7 days after) where they sent similar total funds
# Both conditions must hold for the SAME account on matching day pairs.
# We cap at the first matching pair per account to avoid double-counting.

def validate_temporal_layering(con, gt, total_accounts):
    print("\nValidating TEMPORAL_LAYERING (revised)...")
    t0 = time.time()

    rows = con.execute("""
        WITH daily_in AS (
            SELECT account_to AS account_id,
                   DATE_TRUNC('day', ts) AS day,
                   SUM(amount) AS inbound
            FROM raw_txns
            GROUP BY account_to, DATE_TRUNC('day', ts)
            HAVING SUM(amount) > 0
        ),
        daily_out AS (
            SELECT account_from AS account_id,
                   DATE_TRUNC('day', ts) AS day,
                   SUM(amount) AS outbound
            FROM raw_txns
            GROUP BY account_from, DATE_TRUNC('day', ts)
            HAVING SUM(amount) > 0
        ),
        matched AS (
            SELECT
                i.account_id,
                i.day AS recv_day,
                o.day AS send_day,
                i.inbound,
                o.outbound,
                o.outbound / i.inbound AS ratio,
                DATEDIFF('day', i.day, o.day) AS delay
            FROM daily_in i
            JOIN daily_out o
              ON i.account_id = o.account_id
             AND DATEDIFF('day', i.day, o.day) BETWEEN 1 AND 7
            WHERE i.inbound > 0
              AND o.outbound / i.inbound BETWEEN 0.70 AND 1.10
        )
        SELECT DISTINCT account_id
        FROM matched
    """).fetchall()

    flagged = [r[0] for r in rows]
    print(f"  Completed in {time.time()-t0:.1f}s")
    return evaluate(flagged, gt, total_accounts, "TEMPORAL_LAYERING")


def analyze_temporal_layering(con, gt, total_accounts):
    print("\n  TEMPORAL_LAYERING threshold grid (passthrough_lo x max_days):")
    print(f"  {'Pass Lo':>8} {'Max Days':>9} {'Flagged':>9} {'Confirmed':>10} {'Precision':>10} {'Lift':>6}")
    print(f"  {'─'*58}")
    for plo in [0.70, 0.80, 0.90]:
        for maxd in [1, 3, 7]:
            rows = con.execute(f"""
                WITH daily_in AS (
                    SELECT account_to AS account_id,
                           DATE_TRUNC('day', ts) AS day,
                           SUM(amount) AS inbound
                    FROM raw_txns GROUP BY account_to, DATE_TRUNC('day', ts)
                    HAVING SUM(amount) > 0
                ),
                daily_out AS (
                    SELECT account_from AS account_id,
                           DATE_TRUNC('day', ts) AS day,
                           SUM(amount) AS outbound
                    FROM raw_txns GROUP BY account_from, DATE_TRUNC('day', ts)
                    HAVING SUM(amount) > 0
                )
                SELECT DISTINCT i.account_id
                FROM daily_in i
                JOIN daily_out o
                  ON i.account_id = o.account_id
                 AND DATEDIFF('day', i.day, o.day) BETWEEN 1 AND {maxd}
                WHERE i.inbound > 0
                  AND o.outbound / i.inbound BETWEEN {plo} AND 1.10
            """).fetchall()
            flagged   = {r[0] for r in rows}
            confirmed = len(flagged & gt)
            prec      = confirmed / len(flagged) if flagged else 0
            base      = len(gt) / total_accounts
            lift      = prec / base if base > 0 else 0
            print(f"  {plo:>8.2f} {maxd:>9}d {len(flagged):>9,} "
                  f"{confirmed:>10,} {prec:>9.2%} {lift:>6.1f}x")


# ── RANDOM_STRUCTURING ────────────────────────────────────────────────────────
# Diagnostic showed:
#   FRAUD:  avg 60.7 txns/day, avg total Rs.2.88L, avg CV 1.057
#   LEGIT:  avg  5.3 txns/day, avg total Rs.24K,   avg CV 0.988
#
# The real discriminator is VOLUME — fraud accounts do 60+ sub-threshold
# transactions per day. Legit accounts do ~5. CV alone doesn't separate them.
#
# Revised: flag accounts with 20+ sub-threshold transactions per day
# totalling more than Rs.5L. This targets the high-volume structuring
# typology visible in fraud accounts.

def validate_random_structuring(con, gt, total_accounts):
    print("\nValidating RANDOM_STRUCTURING (revised — volume-based)...")
    t0 = time.time()

    rows = con.execute("""
        WITH daily AS (
            SELECT account_from AS account_id,
                   DATE_TRUNC('day', ts) AS day,
                   COUNT(*) AS txn_count,
                   SUM(amount) AS total_amount,
                   AVG(amount) AS avg_amount,
                   STDDEV_POP(amount) AS std_amount
            FROM raw_txns
            WHERE amount < 50000
            GROUP BY account_from, DATE_TRUNC('day', ts)
        )
        SELECT DISTINCT account_id
        FROM daily
        WHERE txn_count >= 20
          AND total_amount >= 500000
    """).fetchall()

    flagged = [r[0] for r in rows]
    print(f"  Completed in {time.time()-t0:.1f}s")
    return evaluate(flagged, gt, total_accounts, "RANDOM_STRUCTURING (volume)")


def analyze_random_structuring(con, gt, total_accounts):
    print("\n  RANDOM_STRUCTURING threshold grid (min_txns x min_total):")
    print(f"  {'Min Txns':>9} {'Min Total':>11} {'Flagged':>9} "
          f"{'Confirmed':>10} {'Precision':>10} {'Lift':>6}")
    print(f"  {'─'*60}")
    for min_txns in [10, 20, 30, 50]:
        for min_total in [100000, 500000, 1000000]:
            rows = con.execute(f"""
                WITH daily AS (
                    SELECT account_from AS account_id,
                           DATE_TRUNC('day', ts) AS day,
                           COUNT(*) AS txn_count,
                           SUM(amount) AS total_amount
                    FROM raw_txns
                    WHERE amount < 50000
                    GROUP BY account_from, DATE_TRUNC('day', ts)
                )
                SELECT DISTINCT account_id
                FROM daily
                WHERE txn_count >= {min_txns}
                  AND total_amount >= {min_total}
            """).fetchall()
            flagged   = {r[0] for r in rows}
            confirmed = len(flagged & gt)
            prec      = confirmed / len(flagged) if flagged else 0
            base      = len(gt) / total_accounts
            lift      = prec / base if base > 0 else 0
            print(f"  {min_txns:>9} {min_total:>11,} {len(flagged):>9,} "
                  f"{confirmed:>10,} {prec:>9.2%} {lift:>6.1f}x")


# ── CROSS_BANK_ACCELERATION ───────────────────────────────────────────────────
# Diagnostic showed:
#   FRAUD:  6,089,463 cross-bank pairs (avg ratio 11.5M — not 0.7-1.05)
#   LEGIT:  1,137,280 cross-bank pairs (avg ratio 1.1M)
#
# The passthrough ratio filter (0.70-1.05) is not discriminating because
# fraud and legit have similar ratios at the pair level.
#
# The real signal: fraud accounts have 5-10x MORE cross-bank transactions
# than legit accounts. A cross-bank transit mule is identified by FREQUENCY
# not by amount ratio.
#
# Revised: flag accounts where cross-bank outbound transactions
# exceed their same-bank outbound transactions by a ratio of 3:1 or more.
# Pure cross-bank actors = transit mules.

def validate_cross_bank_accel(con, gt, total_accounts):
    print("\nValidating CROSS_BANK_ACCELERATION (revised — frequency-based)...")
    t0 = time.time()

    rows = con.execute("""
        WITH account_bank_stats AS (
            SELECT
                account_from AS account_id,
                COUNT(*) AS total_out_txns,
                COUNT(CASE WHEN bank_from != bank_to THEN 1 END) AS cross_bank_txns,
                COUNT(CASE WHEN bank_from  = bank_to THEN 1 END) AS same_bank_txns
            FROM raw_txns
            GROUP BY account_from
            HAVING COUNT(*) >= 10
        )
        SELECT DISTINCT account_id
        FROM account_bank_stats
        WHERE cross_bank_txns >= 10
          AND same_bank_txns = 0
    """).fetchall()

    flagged = [r[0] for r in rows]
    print(f"  Completed in {time.time()-t0:.1f}s")
    return evaluate(flagged, gt, total_accounts, "CROSS_BANK_ACCEL (pure cross-bank)")


def analyze_cross_bank_accel(con, gt, total_accounts):
    print("\n  CROSS_BANK_ACCELERATION threshold grid:")
    print(f"  {'Min Txns':>9} {'Condition':>30} {'Flagged':>9} "
          f"{'Confirmed':>10} {'Precision':>10} {'Lift':>6}")
    print(f"  {'─'*75}")

    configs = [
        (10,  "same_bank_txns = 0",      "100% cross-bank"),
        (10,  "cross_bank_txns * 1.0 / total_out_txns >= 0.95", ">95% cross-bank"),
        (10,  "cross_bank_txns * 1.0 / total_out_txns >= 0.90", ">90% cross-bank"),
        (20,  "same_bank_txns = 0",      "100% cross-bank, 20+ txns"),
        (50,  "same_bank_txns = 0",      "100% cross-bank, 50+ txns"),
    ]

    for min_txns, condition, label in configs:
        rows = con.execute(f"""
            WITH stats AS (
                SELECT account_from AS account_id,
                       COUNT(*) AS total_out_txns,
                       COUNT(CASE WHEN bank_from != bank_to THEN 1 END) AS cross_bank_txns,
                       COUNT(CASE WHEN bank_from  = bank_to THEN 1 END) AS same_bank_txns
                FROM raw_txns
                GROUP BY account_from
                HAVING COUNT(*) >= {min_txns}
            )
            SELECT DISTINCT account_id FROM stats
            WHERE {condition}
        """).fetchall()
        flagged   = {r[0] for r in rows}
        confirmed = len(flagged & gt)
        prec      = confirmed / len(flagged) if flagged else 0
        base      = len(gt) / total_accounts
        lift      = prec / base if base > 0 else 0
        print(f"  {min_txns:>9} {label:>30} {len(flagged):>9,} "
              f"{confirmed:>10,} {prec:>9.2%} {lift:>6.1f}x")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = None
    grid   = "--grid" in sys.argv
    for arg in sys.argv[1:]:
        if arg.isdigit():
            sample = int(arg)

    print("=" * 60)
    print("CEP v3 Validator v2 — Data-driven thresholds")
    print("=" * 60)

    con = connect()
    load_view(con, sample)

    gt = get_ground_truth(con)
    total_accounts = con.execute(
        "SELECT COUNT(DISTINCT account_from) FROM raw_txns"
    ).fetchone()[0]
    print(f"  Ground truth: {len(gt):,} confirmed fraud accounts")
    print(f"  Total accounts: {total_accounts:,}")
    print(f"  Base fraud rate: {len(gt)/total_accounts:.4%}")

    results = []
    results.append(validate_temporal_layering(con, gt, total_accounts))
    results.append(validate_random_structuring(con, gt, total_accounts))
    results.append(validate_cross_bank_accel(con, gt, total_accounts))

    if grid:
        print("\n" + "=" * 60)
        print("THRESHOLD GRID ANALYSIS")
        print("=" * 60)
        analyze_temporal_layering(con, gt, total_accounts)
        analyze_random_structuring(con, gt, total_accounts)
        analyze_cross_bank_accel(con, gt, total_accounts)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'Pattern':<35} {'Flagged':>8} {'Confirmed':>10} "
          f"{'Precision':>10} {'Lift':>6}")
    print(f"  {'─'*72}")
    for r in results:
        print(f"  {r['pattern']:<35} {r['flagged']:>8,} {r['confirmed']:>10,} "
              f"  {r['precision']:>8.2%} {r['lift']:>5.1f}x")