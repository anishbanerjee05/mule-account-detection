"""
cep_diagnostic.py — Diagnoses WHY CEP patterns aren't discriminating
between fraud and non-fraud accounts.

For each pattern type, pulls every alert ever fired, splits them into
"fired on a confirmed-fraud account" vs "fired on a non-fraud account",
and compares the actual evidence values (gap_hours, amount,
transfer_count, etc.) between the two groups.

If there's a genuine separating threshold hiding in the data, this
will show it as a clear difference in distributions between the two
groups. If the distributions look the same, that tells us the pattern
itself isn't discriminating, regardless of where we set the threshold.

Requires ibm_aml_small.csv in the current directory.

Usage:
    python cep_diagnostic.py
"""

import json
import redis
import pandas as pd
import numpy as np

r = redis.Redis(host='localhost', port=6379, decode_responses=True)
r.ping()

print("Loading IBM AML ground truth labels...")
df = pd.read_csv('ibm_aml_small.csv')
laundering_accounts = set(
    df[df['Is Laundering'] == 1]['Account'].astype(str).unique()
)
print(f"Total confirmed laundering accounts: {len(laundering_accounts):,}\n")

# ── Pull every alert from every account's cep_alerts list ──────────────────
print("Loading all CEP alerts...")
cep_keys = r.keys('cep_alerts:*')

all_alerts = []
for key in cep_keys:
    account_id = key.split(':', 1)[1]
    raw = r.lrange(key, 0, -1)
    for item in raw:
        try:
            alert = json.loads(item)
            alert['account_id'] = account_id
            alert['is_confirmed_fraud'] = account_id in laundering_accounts
            all_alerts.append(alert)
        except Exception:
            pass

print(f"Total alert events loaded: {len(all_alerts)}\n")

alerts_df = pd.DataFrame(all_alerts)

print("=" * 70)
print("DIAGNOSTIC — FEATURE DISTRIBUTIONS BY PATTERN, FRAUD vs NON-FRAUD")
print("=" * 70)

for pattern_type in alerts_df['pattern_type'].unique():
    subset = alerts_df[alerts_df['pattern_type'] == pattern_type].copy()
    fraud_subset = subset[subset['is_confirmed_fraud'] == True]
    nonfraud_subset = subset[subset['is_confirmed_fraud'] == False]

    print(f"\n--- {pattern_type} ---")
    print(f"  Total fires: {len(subset)}  "
          f"(Fraud: {len(fraud_subset)}, Non-fraud: {len(nonfraud_subset)})")

    if len(fraud_subset) == 0:
        print("  No confirmed-fraud examples to compare — cannot analyze separability.")
        continue

    # Extract evidence fields — these differ per pattern type
    evidence_df = pd.json_normalize(subset['evidence'])
    evidence_df['is_confirmed_fraud'] = subset['is_confirmed_fraud'].values

    numeric_cols = evidence_df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        fraud_vals = evidence_df[evidence_df['is_confirmed_fraud'] == True][col].dropna()
        nonfraud_vals = evidence_df[evidence_df['is_confirmed_fraud'] == False][col].dropna()

        if len(fraud_vals) == 0 or len(nonfraud_vals) == 0:
            continue

        print(f"\n  Field: {col}")
        print(f"    Fraud      — mean: {fraud_vals.mean():>12,.4f}  "
              f"median: {fraud_vals.median():>12,.4f}  "
              f"min: {fraud_vals.min():>12,.4f}  max: {fraud_vals.max():>12,.4f}")
        print(f"    Non-fraud  — mean: {nonfraud_vals.mean():>12,.4f}  "
              f"median: {nonfraud_vals.median():>12,.4f}  "
              f"min: {nonfraud_vals.min():>12,.4f}  max: {nonfraud_vals.max():>12,.4f}")

        # Quick separability check — does a simple threshold exist?
        if col in ('gap_hours',):
            # for gap_hours, fraud might cluster at LOWER values (faster forwarding)
            for threshold in [0.05, 0.1, 0.25, 0.5, 1.0]:
                fraud_below = (fraud_vals <= threshold).sum()
                nonfraud_below = (nonfraud_vals <= threshold).sum()
                fraud_pct = fraud_below / len(fraud_vals) * 100
                nonfraud_pct = nonfraud_below / len(nonfraud_vals) * 100
                print(f"      <= {threshold:>5.2f}h: fraud {fraud_pct:>5.1f}% "
                      f"vs non-fraud {nonfraud_pct:>5.1f}%")

        if col in ('forwarded_amount', 'received_amount', 'total_amount'):
            for threshold in [1, 100, 1000, 10000]:
                fraud_above = (fraud_vals >= threshold).sum()
                nonfraud_above = (nonfraud_vals >= threshold).sum()
                fraud_pct = fraud_above / len(fraud_vals) * 100
                nonfraud_pct = nonfraud_above / len(nonfraud_vals) * 100
                print(f"      >= Rs.{threshold:>7,}: fraud {fraud_pct:>5.1f}% "
                      f"vs non-fraud {nonfraud_pct:>5.1f}%")

print("\n" + "=" * 70)
print("How to read this:")
print("  If fraud and non-fraud rows show SIMILAR mean/median/min/max for")
print("  a field, that field does NOT separate fraud from noise at any")
print("  threshold — the pattern is firing on the SHAPE of the behavior")
print("  but the shape itself is common to both fraud and legitimate use.")
print("  If they show CLEARLY DIFFERENT distributions, there is a usable")
print("  threshold we can tighten the pattern around.")
print("=" * 70)