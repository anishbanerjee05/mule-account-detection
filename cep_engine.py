"""
cep_engine.py — Complex Event Processing engine for mule detection.

Detects multi-transaction patterns across time that individual
transaction scoring cannot see. Called from consumer.py after every
transaction is scored.

Patterns detected:
  1. RAPID_FORWARD          — account receives funds, then forwards
                               (sends) within 15 minutes, with a
                               minimum forwarded amount of Rs. 100
  2. CONCENTRATED_DESTINATION — 5+ transfers to the same destination
                               account within a rolling 6-hour window
  3. BURST_ACTIVATION       — account silent for 30+ days, then 5+
                               transfers within 1 hour
  4. FAN_OUT_SPIKE          — 10+ unique destination banks within a
                               rolling 6-hour window

THRESHOLD TUNING NOTE (post-validation, June 2026):
Original thresholds (2h RAPID_FORWARD window, 3-transfer/24h
CONCENTRATED_DESTINATION) were spec-derived, not empirically
validated. A ground-truth check against IBM AML confirmed-laundering
labels showed CEP-only catches (accounts the model's score missed)
were confirmed fraud at only 0.33% — statistically indistinguishable
from the dataset's overall ~0.35% base rate, i.e. no better than
random.

A follow-up diagnostic (cep_diagnostic.py) comparing the actual
evidence values behind each pattern firing, split by confirmed-fraud
vs non-fraud, found:
  - FAN_OUT_SPIKE: strongly separable as-is. Fraud accounts averaged
    55.7 unique banks / 179 transfers vs non-fraud's 10.3 banks / 17.9
    transfers. Left UNCHANGED.
  - RAPID_FORWARD: gap_hours showed real separation (fraud median
    ~14 seconds vs non-fraud median ~2 minutes) but the 2-hour window
    was so loose that 100% of BOTH groups passed it, making the
    threshold useless as a filter despite the underlying signal being
    real. TIGHTENED to 15 minutes. Also added a Rs. 100 minimum
    forwarded_amount floor, since fraud cleared Rs. 100 at 88.1% vs
    non-fraud's 69.9%.
  - CONCENTRATED_DESTINATION: showed almost no separation — transfer
    counts were nearly identical between fraud (mean 3.10) and
    non-fraud (mean 3.03), meaning nearly every alert fired right at
    the 3-transfer minimum for both groups equally. TIGHTENED to 5+
    transfers within a narrower 6-hour window (was 3+ within 24h) to
    make this a genuinely less common, more specific signal rather
    than a frequent coincidence.

These new thresholds should be re-validated with cep_ground_truth_check.py
and cep_diagnostic.py after the next full producer run, before treating
the resulting precision number as demo-final.

All time comparisons use the event's own `timestamp` field (the
producer's demo-compressed-but-proportional clock during the demo;
real wall-clock time in production via webhook_receiver.py). This
file contains no demo-specific logic — it is written exactly as it
would run in production.

Redis structures used (new, in addition to what consumer.py already
maintains):

  acct_in:{account_id}        Hash, 24h TTL — most recent INBOUND
                               transaction for this account (amount,
                               timestamp). Currently nothing tracks
                               inbound money; consumer.py only updates
                               state for account_from (the sender).

  acct_window:{account_id}    List, capped, 24h TTL — recent outbound
                               transactions for this account as JSON:
                               {timestamp, dest_account, dest_bank,
                               amount}. Used for rolling-window pattern
                               checks (CONCENTRATED_DESTINATION,
                               FAN_OUT_SPIKE, BURST_ACTIVATION) since
                               consumer.py's acct:{id} hash only stores
                               ALL-TIME running totals, not a queryable
                               history.

  cep_alerts:{account_id}     List, capped at 200, no expiry — detected
                               pattern events as JSON: {pattern_type,
                               timestamp, evidence}.

  cep_last_seen:{account_id}  String, 45-day TTL — timestamp of this
                               account's previous transaction (any
                               direction), used for BURST_ACTIVATION's
                               dormancy check.

  cep_cooldown:{account_id}:{pattern}  String, TTL = cooldown window —
                               marks that this pattern already fired
                               recently for this account, so a
                               naturally high-volume account does not
                               re-trigger the same pattern on every
                               single transaction. The underlying
                               behaviour is still logged once per
                               cooldown window, not suppressed entirely.

IMPORTANT — RAPID_FORWARD consumption: once a RAPID_FORWARD alert
fires, the triggering acct_in:{account_id} record is DELETED. Without
this, an account that received money once but sends frequently (e.g.
a high-volume hub account) would re-trigger RAPID_FORWARD on every
subsequent unrelated outbound transaction, since the same stale
inbound record would keep matching. Deleting it means RAPID_FORWARD
fires exactly once per genuine receive-then-forward event, which is
what "receives funds then forwards quickly" is actually meant to
describe — a single act of forwarding, not an ongoing state.
"""

import json
import redis
from datetime import datetime, timedelta

# ── Pattern thresholds ───────────────────────────────────────────────────────
# RAPID_FORWARD — tightened from 2h to 15min based on gap_hours
# separability found in diagnostic (fraud median ~14s, non-fraud ~2min).
# Added amount floor since fraud clears Rs.100 far more often (88.1%)
# than non-fraud (69.9%) on forwarded_amount.
RAPID_FORWARD_WINDOW_HOURS          = 0.25   # 15 minutes
RAPID_FORWARD_MIN_AMOUNT            = 100.0  # Rs. 100 floor

# CONCENTRATED_DESTINATION — tightened from 3+/24h to 5+/6h. The
# original thresholds showed almost zero separation between fraud and
# non-fraud (both clustered right at the 3-transfer minimum).
CONCENTRATED_DEST_WINDOW_HOURS      = 6
CONCENTRATED_DEST_MIN_COUNT         = 5

BURST_DORMANCY_DAYS                 = 30
BURST_ACTIVATION_WINDOW_HOURS       = 1
BURST_ACTIVATION_MIN_COUNT          = 5

# FAN_OUT_SPIKE — left unchanged. Diagnostic confirmed strong
# separability as-is (fraud avg 55.7 banks vs non-fraud avg 10.3).
FAN_OUT_SPIKE_WINDOW_HOURS          = 6
FAN_OUT_SPIKE_MIN_BANKS             = 10

# Cooldown — how long a pattern stays "already alerted" for a given
# account before it's allowed to fire again. Prevents naturally
# high-volume accounts from spamming the same pattern on every single
# transaction once the underlying condition is met. The condition is
# still genuinely true each time; this only controls how often we
# surface a NEW alert for it.
FAN_OUT_SPIKE_COOLDOWN_HOURS        = 6
CONCENTRATED_DEST_COOLDOWN_HOURS    = 6
BURST_ACTIVATION_COOLDOWN_HOURS     = 24  # this one is rare/high-signal,
                                           # long cooldown is appropriate

WINDOW_MAX_LEN = 200     # cap on acct_window list length
WINDOW_TTL_SECONDS = 86400  # 24h, matches consumer.py's acct:{id} TTL
CEP_ALERTS_MAX_LEN = 200
LAST_SEEN_TTL_SECONDS = 45 * 86400  # 45 days — comfortably longer than the
                                     # 30-day dormancy threshold so a real
                                     # gap is never lost to TTL expiry


def _parse_ts(ts_str: str) -> datetime:
    return datetime.fromisoformat(ts_str)


def _push_alert(r: redis.Redis, account_id: str, pattern_type: str,
                 timestamp: str, evidence: dict):
    alert = {
        "pattern_type": pattern_type,
        "account_id":   account_id,
        "timestamp":    timestamp,
        "evidence":     evidence,
    }
    r.lpush(f"cep_alerts:{account_id}", json.dumps(alert))
    r.ltrim(f"cep_alerts:{account_id}", 0, CEP_ALERTS_MAX_LEN - 1)
    return alert


def _record_outbound(r: redis.Redis, account_id: str, timestamp: str,
                      dest_account: str, dest_bank: str, amount: float):
    """Append this outbound transaction to the account's rolling window."""
    entry = {
        "timestamp":     timestamp,
        "dest_account":  dest_account,
        "dest_bank":     dest_bank,
        "amount":        amount,
    }
    key = f"acct_window:{account_id}"
    r.lpush(key, json.dumps(entry))
    r.ltrim(key, 0, WINDOW_MAX_LEN - 1)
    r.expire(key, WINDOW_TTL_SECONDS)


def _get_window(r: redis.Redis, account_id: str) -> list:
    """Return this account's recent outbound transactions, newest first."""
    raw = r.lrange(f"acct_window:{account_id}", 0, WINDOW_MAX_LEN - 1)
    return [json.loads(x) for x in raw]


def _on_cooldown(r: redis.Redis, account_id: str, pattern_type: str) -> bool:
    """True if this pattern already fired for this account within its
    cooldown window — i.e. we should NOT alert again yet."""
    return r.exists(f"cep_cooldown:{account_id}:{pattern_type}") > 0


def _start_cooldown(r: redis.Redis, account_id: str, pattern_type: str,
                     hours: float):
    r.set(f"cep_cooldown:{account_id}:{pattern_type}", "1",
          ex=int(hours * 3600))


def check_patterns(r: redis.Redis, event: dict) -> list:
    """
    Run all four CEP pattern checks against the current transaction.
    Call this from consumer.py AFTER the existing scoring step.

    event must contain: account_from, account_to, bank_from, bank_to,
    amount_paid, timestamp (ISO format string).

    Returns a list of detected pattern alert dicts (empty if none fired).
    """
    detected = []

    account_from = str(event['account_from'])
    account_to   = str(event['account_to'])
    bank_to      = str(event['bank_to'])
    amount       = float(event['amount_paid'])
    timestamp    = str(event['timestamp'])
    now          = _parse_ts(timestamp)

    # ── Pattern 3 precheck — BURST_ACTIVATION dormancy gap ──────────────────
    # Must run BEFORE we overwrite cep_last_seen below, since we need the
    # PREVIOUS timestamp to compute the gap.
    last_seen_key = f"cep_last_seen:{account_from}"
    last_seen_raw = r.get(last_seen_key)

    # ── Pattern 1 — RAPID_FORWARD ────────────────────────────────────────────
    # account_from is now SENDING. Check if it RECEIVED money recently
    # (within the window) — i.e. it's forwarding funds it just got.
    # Tightened window (15min) + amount floor (Rs.100), per diagnostic.
    inbound_key = f"acct_in:{account_from}"
    inbound = r.hgetall(inbound_key)
    if inbound:
        received_at = _parse_ts(inbound['timestamp'])
        gap_hours = (now - received_at).total_seconds() / 3600
        if (0 <= gap_hours <= RAPID_FORWARD_WINDOW_HOURS
                and amount >= RAPID_FORWARD_MIN_AMOUNT):
            alert = _push_alert(r, account_from, "RAPID_FORWARD", timestamp, {
                "received_amount":    float(inbound['amount']),
                "received_at":        inbound['timestamp'],
                "forwarded_amount":   amount,
                "forwarded_to":       account_to,
                "gap_hours":          round(gap_hours, 3),
            })
            detected.append(alert)
            # Consume this inbound record — it has now been "used" to
            # explain one forwarding event. Without this, a high-volume
            # account that received money once would re-trigger
            # RAPID_FORWARD on every subsequent unrelated send, since
            # the same stale inbound record would keep matching.
            r.delete(inbound_key)

    # Record this transaction's receiver as having just received money,
    # for the NEXT transaction (if any) to check against.
    r.hset(f"acct_in:{account_to}", mapping={
        "timestamp": timestamp,
        "amount":    str(amount),
        "from":      account_from,
    })
    r.expire(f"acct_in:{account_to}", WINDOW_TTL_SECONDS)

    # ── Record this outbound transaction in the rolling window ──────────────
    _record_outbound(r, account_from, timestamp, account_to, bank_to, amount)
    window = _get_window(r, account_from)

    # ── Pattern 2 — CONCENTRATED_DESTINATION ────────────────────────────────
    # 5+ transfers to the SAME destination within 6 hours (tightened
    # from 3+/24h — original showed no fraud/non-fraud separation).
    cutoff_window = now - timedelta(hours=CONCENTRATED_DEST_WINDOW_HOURS)
    same_dest_recent = [
        w for w in window
        if w['dest_account'] == account_to and _parse_ts(w['timestamp']) >= cutoff_window
    ]
    if (len(same_dest_recent) >= CONCENTRATED_DEST_MIN_COUNT
            and not _on_cooldown(r, account_from, "CONCENTRATED_DESTINATION")):
        alert = _push_alert(r, account_from, "CONCENTRATED_DESTINATION", timestamp, {
            "destination_account": account_to,
            "transfer_count":      len(same_dest_recent),
            "window_hours":        CONCENTRATED_DEST_WINDOW_HOURS,
            "total_amount":        sum(w['amount'] for w in same_dest_recent),
        })
        detected.append(alert)
        _start_cooldown(r, account_from, "CONCENTRATED_DESTINATION",
                         CONCENTRATED_DEST_COOLDOWN_HOURS)

    # ── Pattern 4 — FAN_OUT_SPIKE ────────────────────────────────────────────
    # 10+ unique destination BANKS within a rolling 6-hour window.
    # Left unchanged — diagnostic confirmed strong separability.
    cutoff_6h = now - timedelta(hours=FAN_OUT_SPIKE_WINDOW_HOURS)
    recent_6h = [w for w in window if _parse_ts(w['timestamp']) >= cutoff_6h]
    unique_banks_6h = {w['dest_bank'] for w in recent_6h}
    if (len(unique_banks_6h) >= FAN_OUT_SPIKE_MIN_BANKS
            and not _on_cooldown(r, account_from, "FAN_OUT_SPIKE")):
        alert = _push_alert(r, account_from, "FAN_OUT_SPIKE", timestamp, {
            "unique_banks":  len(unique_banks_6h),
            "window_hours":  FAN_OUT_SPIKE_WINDOW_HOURS,
            "transfer_count": len(recent_6h),
        })
        detected.append(alert)
        _start_cooldown(r, account_from, "FAN_OUT_SPIKE",
                         FAN_OUT_SPIKE_COOLDOWN_HOURS)

    # ── Pattern 3 — BURST_ACTIVATION ─────────────────────────────────────────
    # Account silent for 30+ days, then 5+ transfers within 1 hour.
    # Real elapsed time, exactly as it would run in production — the demo's
    # producer.py is responsible for feeding realistic gaps, this logic
    # does not know or care that compression happened.
    if last_seen_raw:
        last_seen_at = _parse_ts(last_seen_raw)
        dormancy_days = (now - last_seen_at).total_seconds() / 86400
        if dormancy_days >= BURST_DORMANCY_DAYS:
            cutoff_1h = now - timedelta(hours=BURST_ACTIVATION_WINDOW_HOURS)
            recent_1h = [w for w in window if _parse_ts(w['timestamp']) >= cutoff_1h]
            if (len(recent_1h) >= BURST_ACTIVATION_MIN_COUNT
                    and not _on_cooldown(r, account_from, "BURST_ACTIVATION")):
                alert = _push_alert(r, account_from, "BURST_ACTIVATION", timestamp, {
                    "dormancy_days":   round(dormancy_days, 1),
                    "burst_count":     len(recent_1h),
                    "window_hours":    BURST_ACTIVATION_WINDOW_HOURS,
                    "last_active_at":  last_seen_raw,
                })
                detected.append(alert)
                _start_cooldown(r, account_from, "BURST_ACTIVATION",
                                 BURST_ACTIVATION_COOLDOWN_HOURS)

    # Update last_seen for the NEXT transaction's dormancy check.
    r.set(last_seen_key, timestamp, ex=LAST_SEEN_TTL_SECONDS)

    return detected


def get_cep_alerts(r: redis.Redis, account_id: str) -> list:
    """Retrieve all CEP alerts recorded for an account (for the dashboard)."""
    raw = r.lrange(f"cep_alerts:{account_id}", 0, CEP_ALERTS_MAX_LEN - 1)
    return [json.loads(x) for x in raw]


# ── Standalone smoke test ───────────────────────────────────────────────────
if __name__ == "__main__":
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    r.ping()
    print("Redis connected — running CEP engine smoke test\n")

    base = datetime(2026, 1, 1, 9, 0, 0)

    def evt(acc_from, acc_to, bank_to, amount, minutes_offset):
        return {
            "account_from": acc_from,
            "account_to":   acc_to,
            "bank_from":    "TESTBANK",
            "bank_to":      bank_to,
            "amount_paid":  amount,
            "timestamp":    (base + timedelta(minutes=minutes_offset)).isoformat(),
        }

    # Test 1: RAPID_FORWARD — A receives from X, then sends to B within
    # 15 minutes, amount >= Rs.100
    print("Test 1: RAPID_FORWARD")
    r.delete("acct_in:MULE_A", "acct_window:MULE_A", "cep_alerts:MULE_A", "cep_last_seen:MULE_A")
    check_patterns(r, evt("SOURCE_X", "MULE_A", "BANK1", 50000, 0))
    result = check_patterns(r, evt("MULE_A", "SINK_B", "BANK2", 49000, 10))
    print("  Detected:", [a['pattern_type'] for a in result], "\n")

    # Test 1b: RAPID_FORWARD should NOT fire if amount is below Rs.100
    print("Test 1b: RAPID_FORWARD — should NOT fire (amount below floor)")
    r.delete("acct_in:MULE_A2", "acct_window:MULE_A2", "cep_alerts:MULE_A2", "cep_last_seen:MULE_A2")
    check_patterns(r, evt("SOURCE_X", "MULE_A2", "BANK1", 50000, 0))
    result = check_patterns(r, evt("MULE_A2", "SINK_B", "BANK2", 50, 10))
    print("  Detected:", [a['pattern_type'] for a in result], "(expected: [])\n")

    # Test 1c: RAPID_FORWARD should NOT fire if gap exceeds 15 minutes
    print("Test 1c: RAPID_FORWARD — should NOT fire (gap too long)")
    r.delete("acct_in:MULE_A3", "acct_window:MULE_A3", "cep_alerts:MULE_A3", "cep_last_seen:MULE_A3")
    check_patterns(r, evt("SOURCE_X", "MULE_A3", "BANK1", 50000, 0))
    result = check_patterns(r, evt("MULE_A3", "SINK_B", "BANK2", 49000, 30))
    print("  Detected:", [a['pattern_type'] for a in result], "(expected: [])\n")

    # Test 2: CONCENTRATED_DESTINATION — 5 transfers to same dest in 6h
    print("Test 2: CONCENTRATED_DESTINATION")
    r.delete("acct_in:MULE_C", "acct_window:MULE_C", "cep_alerts:MULE_C", "cep_last_seen:MULE_C")
    check_patterns(r, evt("MULE_C", "DEST_1", "BANK3", 10000, 0))
    check_patterns(r, evt("MULE_C", "DEST_1", "BANK3", 12000, 60))
    check_patterns(r, evt("MULE_C", "DEST_1", "BANK3", 11000, 120))
    check_patterns(r, evt("MULE_C", "DEST_1", "BANK3", 13000, 180))
    result = check_patterns(r, evt("MULE_C", "DEST_1", "BANK3", 9000, 240))
    print("  Detected:", [a['pattern_type'] for a in result], "\n")

    # Test 3: FAN_OUT_SPIKE — 10 unique banks in 6h
    print("Test 3: FAN_OUT_SPIKE")
    r.delete("acct_in:MULE_D", "acct_window:MULE_D", "cep_alerts:MULE_D", "cep_last_seen:MULE_D")
    result = []
    for i in range(10):
        result = check_patterns(r, evt("MULE_D", f"DEST_{i}", f"BANK_{i}", 5000, i * 10))
    print("  Detected:", [a['pattern_type'] for a in result], "\n")

    # Test 4: BURST_ACTIVATION — dormant 31 days, then 5 transfers in 1h
    print("Test 4: BURST_ACTIVATION")
    r.delete("acct_in:MULE_E", "acct_window:MULE_E", "cep_alerts:MULE_E", "cep_last_seen:MULE_E")
    check_patterns(r, evt("MULE_E", "DEST_X", "BANKX", 5000, 0))
    result = []
    for i in range(5):
        result = check_patterns(r, evt(
            "MULE_E", f"DEST_{i}", "BANKX", 5000,
            31 * 24 * 60 + i * 5  # 31 days later, 5 min apart
        ))
    print("  Detected:", [a['pattern_type'] for a in result])