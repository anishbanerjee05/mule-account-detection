# cep_diagnostic_v2.py
# Diagnoses LAYERING_CHAIN and AMOUNT_SYMMETRY threshold issues.
# Compares actual evidence values between confirmed fraud vs non-fraud
# to find the right thresholds empirically.
import duckdb
import time

CSV_PATH = "ibm_aml_small.csv"

con = duckdb.connect()
con.execute("SET memory_limit='4GB'")
con.execute("SET threads=4")
con.execute("""
    CREATE VIEW raw_txns AS
    SELECT
        CAST(Timestamp AS TIMESTAMP) AS ts,
        CAST(Account    AS VARCHAR)  AS account_from,
        CAST(Account_1  AS VARCHAR)  AS account_to,
        CAST("From Bank" AS VARCHAR) AS bank_from,
        CAST("To Bank"   AS VARCHAR) AS bank_to,
        "Amount Paid"                AS amount,
        "Is Laundering"              AS is_laundering
    FROM read_csv('ibm_aml_small.csv', header=true, nullstr='', ignore_errors=false)
""")

laundering = {r[0] for r in con.execute("""
    SELECT DISTINCT account_id FROM (
        SELECT account_from AS account_id FROM raw_txns WHERE is_laundering = 1
        UNION
        SELECT account_to AS account_id FROM raw_txns WHERE is_laundering = 1
    )
""").fetchall()}

print(f"Confirmed laundering accounts: {len(laundering):,}")
print("="*70)

# ── LAYERING_CHAIN diagnostic ──────────────────────────────────────────────
print("\nLAYERING_CHAIN — distribution of amount difference ratio")
print("(A->B->C same day, amounts match within X% -- what X actually separates?)")

t0 = time.time()
lc_data = con.execute("""
    SELECT
        t1.account_to AS account_id,
        ABS(t1.amount - t2.amount) / GREATEST(t1.amount, t2.amount) AS ratio_diff,
        LEAST(t1.amount, t2.amount) AS min_amount,
        t1.amount AS amount_ab,
        t2.amount AS amount_bc
    FROM raw_txns t1
    JOIN raw_txns t2
        ON t1.account_to = t2.account_from
        AND t1.account_from <> t2.account_to
        AND DATE_TRUNC('day', t1.ts) = DATE_TRUNC('day', t2.ts)
        AND t1.amount >= 1000
        AND t2.amount >= 1000
        AND ABS(t1.amount - t2.amount) / GREATEST(t1.amount, t2.amount) < 0.50
""").fetchall()

fraud_lc = [(r[1], r[2]) for r in lc_data if r[0] in laundering]
legit_lc = [(r[1], r[2]) for r in lc_data if r[0] not in laundering]

print(f"\n  Fraud intermediary accounts ({len(set(r[0] for r in lc_data if r[0] in laundering))} unique):")
if fraud_lc:
    ratios_f = sorted([r[0] for r in fraud_lc])
    amounts_f = sorted([r[1] for r in fraud_lc])
    print(f"    ratio_diff: min={min(ratios_f):.4f} mean={sum(ratios_f)/len(ratios_f):.4f} "
          f"median={ratios_f[len(ratios_f)//2]:.4f} max={max(ratios_f):.4f}")
    print(f"    min_amount: min={min(amounts_f):,.0f} mean={sum(amounts_f)/len(amounts_f):,.0f} "
          f"median={amounts_f[len(amounts_f)//2]:,.0f} max={max(amounts_f):,.0f}")

print(f"\n  Non-fraud intermediary accounts ({len(set(r[0] for r in lc_data if r[0] not in laundering))} unique):")
if legit_lc:
    ratios_l = sorted([r[0] for r in legit_lc])
    amounts_l = sorted([r[1] for r in legit_lc])
    print(f"    ratio_diff: min={min(ratios_l):.4f} mean={sum(ratios_l)/len(ratios_l):.4f} "
          f"median={ratios_l[len(ratios_l)//2]:.4f} max={max(ratios_l):.4f}")
    print(f"    min_amount: min={min(amounts_l):,.0f} mean={sum(amounts_l)/len(amounts_l):,.0f} "
          f"median={amounts_l[len(amounts_l)//2]:,.0f} max={max(amounts_l):,.0f}")

print(f"\n  Precision grid (amount_tolerance x min_amount):")
print(f"  {'Tolerance':>10} {'Min Amt':>12} {'Flagged':>8} {'Confirmed':>10} {'Precision':>10}")
for tol in [0.01, 0.02, 0.05, 0.10]:
    for min_amt in [1000, 10000, 50000, 100000]:
        flagged = set(r[0] for r in lc_data
                      if r[1] <= tol and r[2] >= min_amt)
        confirmed = flagged & laundering
        prec = len(confirmed)/len(flagged)*100 if flagged else 0
        print(f"  {tol:>10.2f} {min_amt:>12,.0f} {len(flagged):>8,} "
              f"{len(confirmed):>10} {prec:>9.2f}%")

print(f"\n  Query time: {time.time()-t0:.1f}s")

# ── AMOUNT_SYMMETRY diagnostic ─────────────────────────────────────────────
print("\n" + "="*70)
print("\nAMOUNT_SYMMETRY — distribution of symmetry ratio and min_amount")
print("(Does in/out symmetry actually separate fraud from non-fraud?)")

t0 = time.time()
sym_data = con.execute("""
    SELECT account_id,
           total_in,
           total_out,
           ABS(total_in - total_out) / GREATEST(total_in, total_out) AS asymmetry
    FROM (
        SELECT account_id, day,
            SUM(CASE WHEN direction='in'  THEN amount ELSE 0 END) AS total_in,
            SUM(CASE WHEN direction='out' THEN amount ELSE 0 END) AS total_out
        FROM (
            SELECT account_to   AS account_id, DATE_TRUNC('day', ts) AS day,
                   amount, 'in' AS direction FROM raw_txns
            UNION ALL
            SELECT account_from AS account_id, DATE_TRUNC('day', ts) AS day,
                   amount, 'out' AS direction FROM raw_txns
        )
        GROUP BY account_id, day
    )
    WHERE total_in >= 1000 AND total_out >= 1000
""").fetchall()

fraud_sym = [(r[1], r[2], r[3]) for r in sym_data if r[0] in laundering]
legit_sym = [(r[1], r[2], r[3]) for r in sym_data if r[0] not in laundering]

print(f"\n  Fraud accounts ({len(set(r[0] for r in sym_data if r[0] in laundering))} unique):")
if fraud_sym:
    asym_f = sorted([r[2] for r in fraud_sym])
    in_f = sorted([r[0] for r in fraud_sym])
    print(f"    asymmetry: min={min(asym_f):.4f} mean={sum(asym_f)/len(asym_f):.4f} "
          f"median={asym_f[len(asym_f)//2]:.4f} max={max(asym_f):.4f}")
    print(f"    total_in:  min={min(in_f):,.0f} mean={sum(in_f)/len(in_f):,.0f} "
          f"median={in_f[len(in_f)//2]:,.0f}")
    pct_symmetric = sum(1 for r in fraud_sym if r[2] < 0.05) / len(fraud_sym) * 100
    print(f"    % with asymmetry < 5% (very symmetric): {pct_symmetric:.1f}%")

print(f"\n  Non-fraud accounts ({len(set(r[0] for r in sym_data if r[0] not in laundering))} unique):")
if legit_sym:
    asym_l = sorted([r[2] for r in legit_sym])
    in_l = sorted([r[0] for r in legit_sym])
    print(f"    asymmetry: min={min(asym_l):.4f} mean={sum(asym_l)/len(asym_l):.4f} "
          f"median={asym_l[len(asym_l)//2]:.4f} max={max(asym_l):.4f}")
    print(f"    total_in:  min={min(in_l):,.0f} mean={sum(in_l)/len(in_l):,.0f} "
          f"median={in_l[len(in_l)//2]:,.0f}")
    pct_symmetric = sum(1 for r in legit_sym if r[2] < 0.05) / len(legit_sym) * 100
    print(f"    % with asymmetry < 5% (very symmetric): {pct_symmetric:.1f}%")

print(f"\n  Key question: if both groups have similar % of symmetric days,")
print(f"  the pattern has no discriminating power regardless of threshold.")

print(f"\n  Precision grid (max_asymmetry x min_total_in):")
print(f"  {'Max Asymm':>10} {'Min In':>12} {'Flagged':>8} {'Confirmed':>10} {'Precision':>10}")
for asym in [0.01, 0.02, 0.05, 0.10]:
    for min_in in [10000, 50000, 100000, 500000]:
        flagged = set(r[0] for r in sym_data
                      if r[3] <= asym and r[1] >= min_in)
        confirmed = flagged & laundering
        prec = len(confirmed)/len(flagged)*100 if flagged else 0
        print(f"  {asym:>10.2f} {min_in:>12,.0f} {len(flagged):>8,} "
              f"{len(confirmed):>10} {prec:>9.2f}%")

print(f"\n  Query time: {time.time()-t0:.1f}s")
print("\n" + "="*70)
con.close()