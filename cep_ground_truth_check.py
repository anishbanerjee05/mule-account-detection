"""
cep_ground_truth_check.py — Validates CEP-only catches against IBM AML's
ground truth "Is Laundering" labels.

A CEP-only catch is an account that:
  - Had a genuine CEP behavioral pattern fire (RAPID_FORWARD, etc.)
  - Was NOT independently flagged by the XGBoost+GNN model's score

This script answers the question that actually matters for the demo:
of those CEP-only catches, how many are CONFIRMED fraud accounts per
the dataset's own labels — not just "unusual," but actually laundering?

Requires ibm_aml_small.csv in the current directory (same file
consumer.py already loads at startup).

Usage:
    python cep_ground_truth_check.py
"""

import json
import redis
import pandas as pd
from collections import Counter

r = redis.Redis(host='localhost', port=6379, decode_responses=True)
r.ping()

print("Loading IBM AML ground truth labels...")
df = pd.read_csv('ibm_aml_small.csv')
laundering_accounts = set(
    df[df['Is Laundering'] == 1]['Account'].astype(str).unique()
)
print(f"Total confirmed laundering accounts in dataset: {len(laundering_accounts):,}\n")

# ── Pull every account that has a CEP alert ─────────────────────────────────
cep_keys = r.keys('cep_alerts:*')
cep_accounts = {k.split(':', 1)[1] for k in cep_keys}

# ── Pull the main alert queue to separate CEP-only vs model-flagged-too ────
main_alerts_raw = r.lrange('alerts', 0, 999)
model_flagged_accounts = set()
cep_only_accounts = set()
cep_only_alert_by_account = {}

for item in main_alerts_raw:
    try:
        a = json.loads(item)
        aid = a.get('account_id', '')
        if a.get('tier') == "CEP — PATTERN MATCH":
            cep_only_accounts.add(aid)
            cep_only_alert_by_account[aid] = a
        else:
            model_flagged_accounts.add(aid)
    except Exception:
        pass

print("=" * 70)
print("GROUND TRUTH VALIDATION — CEP-ONLY CATCHES")
print("=" * 70)

# ── Core question: of CEP-only catches, how many are CONFIRMED fraud? ──────
cep_only_confirmed = cep_only_accounts & laundering_accounts
cep_only_unconfirmed = cep_only_accounts - laundering_accounts

total_cep_only = len(cep_only_accounts)
confirmed_count = len(cep_only_confirmed)
precision_pct = (confirmed_count / total_cep_only * 100) if total_cep_only > 0 else 0

print(f"\nTotal CEP-only catches (model missed, CEP caught): {total_cep_only}")
print(f"  Confirmed laundering (ground truth):              {confirmed_count}")
print(f"  Not confirmed in current dataset:                 {len(cep_only_unconfirmed)}")
print(f"  CEP-only precision rate:                           {precision_pct:.2f}%")

# ── Compare to overall dataset base rate, for context ───────────────────────
overall_laundering_rate = len(laundering_accounts) / df['Account'].nunique() * 100
print(f"\nFor context — overall dataset laundering rate (all accounts): "
      f"{overall_laundering_rate:.4f}%")
if precision_pct > 0:
    lift = precision_pct / overall_laundering_rate if overall_laundering_rate > 0 else 0
    print(f"CEP-only catches are confirmed-fraud at {lift:.1f}x the base rate")

# ── Breakdown by pattern type for CONFIRMED CEP-only catches ───────────────
print("\n--- Pattern Types Among CONFIRMED CEP-Only Catches ---")
pattern_counter = Counter()
for aid in cep_only_confirmed:
    alert = cep_only_alert_by_account.get(aid, {})
    for p in alert.get('cep_patterns', []):
        pattern_counter[p] += 1
for pattern, count in pattern_counter.most_common():
    print(f"  {pattern:<28} {count:>6}")

# ── Show the actual confirmed examples — these are your demo gold ──────────
print("\n--- CONFIRMED Fraud Accounts CEP Caught (Model Missed) ---")
print("These are real, demo-citable examples: real fraud, model score low, CEP caught it.\n")
shown = 0
for aid in cep_only_confirmed:
    if shown >= 10:
        break
    alert = cep_only_alert_by_account.get(aid, {})
    print(f"  Account: {aid:<16} "
          f"Model score: {alert.get('probability', 0):.4f}  "
          f"CEP patterns: {alert.get('cep_patterns', [])}  "
          f"Amount: Rs. {alert.get('amount', 0):,.2f}")
    shown += 1

if shown == 0:
    print("  None found — see note below.")

print("\n" + "=" * 70)
print("End of report.")
print("=" * 70)