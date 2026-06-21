"""
cep_summary.py — Post-run summary of CEP pattern detection.

Run this AFTER a producer.py run has completed (or at any point during
a run, for a partial snapshot) to get a clean, demo-presentable summary
of what the CEP engine actually caught: total alerts, breakdown by
pattern type, accounts caught ONLY by CEP (model missed them), and the
top accounts by alert count.

Usage:
    python cep_summary.py
"""

import json
import redis
from collections import Counter
from datetime import datetime

r = redis.Redis(host='localhost', port=6379, decode_responses=True)
r.ping()

print("=" * 70)
print("CEP PATTERN DETECTION — SUMMARY REPORT")
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# ── Gather all CEP alerts across all accounts ───────────────────────────────
cep_keys = r.keys('cep_alerts:*')
print(f"\nAccounts with at least one CEP alert: {len(cep_keys)}")

all_alerts = []
account_alert_counts = Counter()

for key in cep_keys:
    account_id = key.split(':', 1)[1]
    raw = r.lrange(key, 0, -1)
    for item in raw:
        try:
            alert = json.loads(item)
            all_alerts.append(alert)
            account_alert_counts[account_id] += 1
        except Exception:
            pass

print(f"Total CEP alert events (across all accounts): {len(all_alerts)}")

# ── Breakdown by pattern type ────────────────────────────────────────────────
pattern_counts = Counter(a['pattern_type'] for a in all_alerts)
print("\n--- Breakdown by Pattern Type ---")
for pattern, count in pattern_counts.most_common():
    print(f"  {pattern:<28} {count:>6}")

# ── Cross-reference with model-flagged alerts ───────────────────────────────
# Pull the main alerts list to see which CEP-flagged accounts the MODEL
# also independently flagged vs which it missed entirely.
main_alerts_raw = r.lrange('alerts', 0, 999)
model_flagged_accounts = set()
cep_only_accounts_in_queue = set()

for item in main_alerts_raw:
    try:
        a = json.loads(item)
        aid = a.get('account_id', '')
        if a.get('tier') == "CEP — PATTERN MATCH":
            cep_only_accounts_in_queue.add(aid)
        else:
            model_flagged_accounts.add(aid)
    except Exception:
        pass

cep_accounts = set(account_alert_counts.keys())
caught_by_both = cep_accounts & model_flagged_accounts
cep_detected_but_not_in_main_queue = cep_accounts - model_flagged_accounts - cep_only_accounts_in_queue

print("\n--- CEP vs Model Overlap ---")
print(f"  Accounts with a CEP pattern AND independently flagged by the model: {len(caught_by_both)}")
print(f"  Accounts pinned in dashboard as CEP-ONLY catches:                  {len(cep_only_accounts_in_queue)}")

# ── Top accounts by CEP alert volume ─────────────────────────────────────────
print("\n--- Top 10 Accounts by CEP Alert Count ---")
for account_id, count in account_alert_counts.most_common(10):
    patterns_for_acct = Counter(
        a['pattern_type'] for a in all_alerts
        if a.get('account_id') == account_id
    )
    pattern_summary = ", ".join(f"{p}×{c}" for p, c in patterns_for_acct.most_common())
    print(f"  {account_id:<16} {count:>4} alerts   [{pattern_summary}]")

# ── Sample CEP-only catches for demo narration ──────────────────────────────
print("\n--- Sample CEP-Only Catches (model score did not independently flag) ---")
shown = 0
for item in main_alerts_raw:
    try:
        a = json.loads(item)
        if a.get('tier') == "CEP — PATTERN MATCH" and shown < 5:
            print(f"  Account: {a.get('account_id')}  "
                  f"Score: {a.get('probability', 0):.4f}  "
                  f"Patterns: {a.get('cep_patterns', [])}")
            shown += 1
    except Exception:
        pass
if shown == 0:
    print("  (none found in current main alert queue — check cep_alerts:* "
        "directly, or the model may have independently flagged everything "
        "CEP also caught in this run)")

print("\n" + "=" * 70)
print("End of report.")
print("=" * 70)