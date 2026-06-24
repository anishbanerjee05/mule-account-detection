# cep_diagnostic.py
# Diagnoses why GATHER_SCATTER and PASS_THROUGH_RATIO have weak precision.
# Compares the actual evidence values behind each pattern's fires,
# split by confirmed fraud vs non-fraud, to find the right thresholds.
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

# Ground truth
laundering = {r[0] for r in con.execute("""
    SELECT DISTINCT account_id FROM (
        SELECT account_from AS account_id FROM raw_txns WHERE is_laundering = 1
        UNION
        SELECT account_to AS account_id FROM raw_txns WHERE is_laundering = 1
    )
""").fetchall()}

print(f"Confirmed laundering accounts: {len(laundering):,}")
print("="*70)

# ── GATHER_SCATTER diagnostic ─────────────────────────────────────────────
print("\nGATHER_SCATTER — distribution of n_senders and n_receivers per day")
print("(fraud vs non-fraud accounts, to find the right threshold)")

t0 = time.time()
gs_data = con.execute("""
    SELECT
        gs.account_id,
        gs.day,
        gs.n_senders,
        go.n_receivers
    FROM (
        SELECT account_to AS account_id, DATE_TRUNC('day', ts) AS day,
               COUNT(DISTINCT account_from) AS n_senders
        FROM raw_txns
        GROUP BY account_to, DATE_TRUNC('day', ts)
        HAVING COUNT(DISTINCT account_from) >= 2
    ) gs
    JOIN (
        SELECT account_from AS account_id, DATE_TRUNC('day', ts) AS day,
               COUNT(DISTINCT account_to) AS n_receivers
        FROM raw_txns
        GROUP BY account_from, DATE_TRUNC('day', ts)
        HAVING COUNT(DISTINCT account_to) >= 2
    ) go ON gs.account_id = go.account_id AND gs.day = go.day
""").fetchall()

fraud_gs = [(r[2], r[3]) for r in gs_data if r[0] in laundering]
legit_gs = [(r[2], r[3]) for r in gs_data if r[0] not in laundering]

print(f"\n  Fraud accounts ({len(set(r[0] for r in gs_data if r[0] in laundering))} unique):")
if fraud_gs:
    senders_f = [r[0] for r in fraud_gs]
    receivers_f = [r[1] for r in fraud_gs]
    print(f"    n_senders:   min={min(senders_f)} mean={sum(senders_f)/len(senders_f):.1f} median={sorted(senders_f)[len(senders_f)//2]} max={max(senders_f)}")
    print(f"    n_receivers: min={min(receivers_f)} mean={sum(receivers_f)/len(receivers_f):.1f} median={sorted(receivers_f)[len(receivers_f)//2]} max={max(receivers_f)}")

print(f"\n  Non-fraud accounts ({len(set(r[0] for r in gs_data if r[0] not in laundering))} unique):")
if legit_gs:
    senders_l = [r[0] for r in legit_gs]
    receivers_l = [r[1] for r in legit_gs]
    print(f"    n_senders:   min={min(senders_l)} mean={sum(senders_l)/len(senders_l):.1f} median={sorted(senders_l)[len(senders_l)//2]} max={max(senders_l)}")
    print(f"    n_receivers: min={min(receivers_l)} mean={sum(receivers_l)/len(receivers_l):.1f} median={sorted(receivers_l)[len(receivers_l)//2]} max={max(receivers_l)}")

# Distribution at different threshold combinations
print(f"\n  Precision at different threshold combinations:")
print(f"  {'Senders':>8} {'Receivers':>10} {'Flagged':>8} {'Confirmed':>10} {'Precision':>10}")
for s_thresh in [3, 5, 8, 10]:
    for r_thresh in [3, 5, 8, 10]:
        flagged = set(r[0] for r in gs_data if r[2] >= s_thresh and r[3] >= r_thresh)
        confirmed = flagged & laundering
        prec = len(confirmed)/len(flagged)*100 if flagged else 0
        print(f"  {s_thresh:>8} {r_thresh:>10} {len(flagged):>8,} {len(confirmed):>10} {prec:>9.2f}%")

print(f"\n  Query time: {time.time()-t0:.1f}s")

# ── PASS_THROUGH_RATIO diagnostic ─────────────────────────────────────────
print("\n" + "="*70)
print("\nPASS_THROUGH_RATIO — distribution of ratio and total_in per day")

t0 = time.time()
ptr_data = con.execute("""
    SELECT account_id, total_in, total_out, (total_out/total_in) AS ratio
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

fraud_ptr = [(r[1], r[2], r[3]) for r in ptr_data if r[0] in laundering]
legit_ptr = [(r[1], r[2], r[3]) for r in ptr_data if r[0] not in laundering]

print(f"\n  Fraud accounts ({len(set(r[0] for r in ptr_data if r[0] in laundering))} unique):")
if fraud_ptr:
    ratios_f = sorted([r[2] for r in fraud_ptr])
    ins_f = sorted([r[0] for r in fraud_ptr])
    print(f"    ratio: min={min(ratios_f):.3f} mean={sum(ratios_f)/len(ratios_f):.3f} median={ratios_f[len(ratios_f)//2]:.3f} max={max(ratios_f):.3f}")
    print(f"    total_in: min={min(ins_f):,.0f} mean={sum(ins_f)/len(ins_f):,.0f} median={ins_f[len(ins_f)//2]:,.0f} max={max(ins_f):,.0f}")

print(f"\n  Non-fraud accounts ({len(set(r[0] for r in ptr_data if r[0] not in laundering))} unique):")
if legit_ptr:
    ratios_l = sorted([r[2] for r in legit_ptr])
    ins_l = sorted([r[0] for r in legit_ptr])
    print(f"    ratio: min={min(ratios_l):.3f} mean={sum(ratios_l)/len(ratios_l):.3f} median={ratios_l[len(ratios_l)//2]:.3f} max={max(ratios_l):.3f}")
    print(f"    total_in: min={min(ins_l):,.0f} mean={sum(ins_l)/len(ins_l):,.0f} median={ins_l[len(ins_l)//2]:,.0f} max={max(ins_l):,.0f}")

print(f"\n  Precision at different threshold combinations:")
print(f"  {'Min Total In':>12} {'Ratio':>8} {'Flagged':>8} {'Confirmed':>10} {'Precision':>10}")
for min_in in [100000, 500000, 1000000, 5000000]:
    for ratio in [0.95, 0.97, 0.99]:
        flagged = set(r[0] for r in ptr_data if r[1] >= min_in and r[3] >= ratio)
        confirmed = flagged & laundering
        prec = len(confirmed)/len(flagged)*100 if flagged else 0
        print(f"  {min_in:>12,.0f} {ratio:>8.2f} {len(flagged):>8,} {len(confirmed):>10} {prec:>9.2f}%")

print(f"\n  Query time: {time.time()-t0:.1f}s")
print("\n" + "="*70)
con.close()