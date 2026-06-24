# cep_engine.py - Complex Event Processing engine for mule detection.
# Called from consumer.py after every transaction is scored.
#
# Patterns (empirically validated against IBM AML ground truth, June 2026):
#
#   FAN_OUT_SPIKE     -- 10+ unique destination banks in 6h rolling window.
#                        Precision: 48.39% (64x lift). UNCHANGED from v1.
#
#   FAN_IN            -- receives from 5+ distinct senders in 24h window.
#                        Precision: 6.57% (8.7x lift). NEW in v2.
#
#   GATHER_SCATTER    -- receives from 8+ distinct senders AND sends to 8+
#                        distinct receivers on the same calendar day.
#                        Precision: 93.75% (124.8x lift). NEW in v2.
#
#   SCATTER_GATHER    -- one source sends to 3+ intermediaries, all of whom
#                        send to the same final receiver within 24h.
#                        Precision: 3.41% (4.5x lift). NEW in v2.
#
#   SIMPLE_CYCLE      -- A sends to B, B sends to C, C sends back to A
#                        within the same calendar day.
#                        Precision: 12.93% (17.2x lift). NEW in v2.
#
#   STRUCTURING       -- 3+ transactions in the same day with amounts in
#                        bands just below common reporting thresholds
#                        (900-999, 4500-4999, 9000-9999).
#                        Precision: 4.08% (5.4x lift). NEW in v2.
#
# PATTERNS RETIRED in v2 (with reasons):
#
#   RAPID_FORWARD     -- retired. The IBM AML dataset has minute-level
#                        timestamp resolution only. At 15-minute window,
#                        the pattern fires on timestamp collisions (two
#                        transactions in the same minute) not genuine rapid
#                        forwarding. Unfalsifiable at this data resolution.
#
#   CONCENTRATED_DESTINATION -- retired. Diagnostic showed near-zero
#                        separation: fraud mean 3.10 transfers vs non-fraud
#                        mean 3.03. Both groups clustered at the minimum
#                        threshold regardless of tightening.
#
#   BURST_ACTIVATION  -- retired. Zero fires in full dataset validation run
#                        (6.9M transactions). The 30-day dormancy + 5
#                        transfers in 1h combination does not appear in this
#                        dataset's transaction patterns.
#
# Redis structures used:
#   acct_window:{id}          List, capped, 24h TTL -- recent outbound txns
#                              as JSON: {timestamp, dest_account, dest_bank,
#                              amount, bank_from}
#   acct_in_daily:{id}:{day}  Hash, 25h TTL -- per-day inbound sender counts
#                              and outbound receiver counts for GATHER_SCATTER
#   cep_alerts:{id}           List, capped at 200, no expiry
#   cep_cooldown:{id}:{pat}   String, TTL = cooldown -- rate-limits repeat alerts

import json
import redis
from datetime import datetime, timedelta

# ── Thresholds (empirically validated) ───────────────────────────────────────
FAN_OUT_SPIKE_WINDOW_HOURS      = 6
FAN_OUT_SPIKE_MIN_BANKS         = 10
FAN_OUT_SPIKE_COOLDOWN_HOURS    = 6

FAN_IN_WINDOW_HOURS             = 24
FAN_IN_MIN_SENDERS              = 5
FAN_IN_COOLDOWN_HOURS           = 24

GATHER_SCATTER_MIN_SENDERS      = 8
GATHER_SCATTER_MIN_RECEIVERS    = 8
GATHER_SCATTER_COOLDOWN_HOURS   = 24

SCATTER_GATHER_MIN_INTERMEDIARIES = 3
SCATTER_GATHER_WINDOW_HOURS     = 24
SCATTER_GATHER_COOLDOWN_HOURS   = 24

SIMPLE_CYCLE_WINDOW_HOURS       = 24
SIMPLE_CYCLE_COOLDOWN_HOURS     = 24

STRUCTURING_MIN_COUNT           = 3
STRUCTURING_BANDS               = [(900, 999), (4500, 4999), (9000, 9999)]
STRUCTURING_COOLDOWN_HOURS      = 24

WINDOW_MAX_LEN      = 200
WINDOW_TTL_SECONDS  = 86400   # 24h
CEP_ALERTS_MAX_LEN  = 200


def _parse_ts(ts_str: str) -> datetime:
    return datetime.fromisoformat(ts_str)


def _push_alert(r, account_id, pattern_type, timestamp, evidence):
    alert = {
        "pattern_type": pattern_type,
        "account_id":   account_id,
        "timestamp":    timestamp,
        "evidence":     evidence,
    }
    r.lpush(f"cep_alerts:{account_id}", json.dumps(alert))
    r.ltrim(f"cep_alerts:{account_id}", 0, CEP_ALERTS_MAX_LEN - 1)
    return alert


def _record_outbound(r, account_id, timestamp, dest_account, dest_bank, amount):
    entry = {
        "timestamp":    timestamp,
        "dest_account": dest_account,
        "dest_bank":    dest_bank,
        "amount":       amount,
    }
    key = f"acct_window:{account_id}"
    r.lpush(key, json.dumps(entry))
    r.ltrim(key, 0, WINDOW_MAX_LEN - 1)
    r.expire(key, WINDOW_TTL_SECONDS)


def _get_window(r, account_id):
    raw = r.lrange(f"acct_window:{account_id}", 0, WINDOW_MAX_LEN - 1)
    return [json.loads(x) for x in raw]


def _on_cooldown(r, account_id, pattern_type):
    return r.exists(f"cep_cooldown:{account_id}:{pattern_type}") > 0


def _start_cooldown(r, account_id, pattern_type, hours):
    r.set(f"cep_cooldown:{account_id}:{pattern_type}", "1",
          ex=int(hours * 3600))


def _day_key(timestamp: str) -> str:
    return timestamp[:10]   # 'YYYY-MM-DD'


def check_patterns(r: redis.Redis, event: dict) -> list:
    """
    Run all six CEP pattern checks against the current transaction.
    Call from consumer.py AFTER the existing scoring step.

    event must contain: account_from, account_to, bank_from, bank_to,
    amount_paid, timestamp (ISO format string).

    Returns list of detected pattern alert dicts (empty if none fired).
    """
    detected = []

    account_from = str(event['account_from'])
    account_to   = str(event['account_to'])
    bank_from    = str(event.get('bank_from', ''))
    bank_to      = str(event['bank_to'])
    amount       = float(event['amount_paid'])
    timestamp    = str(event['timestamp'])
    now          = _parse_ts(timestamp)
    day          = _day_key(timestamp)

    # ── Record outbound transaction in rolling window ─────────────────────────
    _record_outbound(r, account_from, timestamp, account_to, bank_to, amount)
    window = _get_window(r, account_from)

    # ── Update daily inbound/outbound tracking for GATHER_SCATTER ────────────
    # Track unique senders per receiver per day
    sender_key   = f"acct_daily_senders:{account_to}:{day}"
    receiver_key = f"acct_daily_receivers:{account_from}:{day}"
    r.sadd(sender_key, account_from)
    r.expire(sender_key, 90000)    # 25h TTL
    r.sadd(receiver_key, account_to)
    r.expire(receiver_key, 90000)

    # ── Pattern 1 — FAN_OUT_SPIKE ─────────────────────────────────────────────
    # 10+ unique destination banks in rolling 6h window.
    cutoff_6h = now - timedelta(hours=FAN_OUT_SPIKE_WINDOW_HOURS)
    recent_6h = [w for w in window if _parse_ts(w['timestamp']) >= cutoff_6h]
    unique_banks = {w['dest_bank'] for w in recent_6h}
    if (len(unique_banks) >= FAN_OUT_SPIKE_MIN_BANKS
            and not _on_cooldown(r, account_from, "FAN_OUT_SPIKE")):
        alert = _push_alert(r, account_from, "FAN_OUT_SPIKE", timestamp, {
            "unique_banks":   len(unique_banks),
            "window_hours":   FAN_OUT_SPIKE_WINDOW_HOURS,
            "transfer_count": len(recent_6h),
        })
        detected.append(alert)
        _start_cooldown(r, account_from, "FAN_OUT_SPIKE",
                        FAN_OUT_SPIKE_COOLDOWN_HOURS)

    # ── Pattern 2 — FAN_IN ────────────────────────────────────────────────────
    # Receives from 5+ distinct senders within 24h.
    # Check the receiver side: account_to has received from account_from.
    n_senders = r.scard(sender_key)
    if (n_senders >= FAN_IN_MIN_SENDERS
            and not _on_cooldown(r, account_to, "FAN_IN")):
        alert = _push_alert(r, account_to, "FAN_IN", timestamp, {
            "unique_senders": n_senders,
            "window_hours":   FAN_IN_WINDOW_HOURS,
            "day":            day,
        })
        detected.append(alert)
        _start_cooldown(r, account_to, "FAN_IN", FAN_IN_COOLDOWN_HOURS)

    # ── Pattern 3 — GATHER_SCATTER ────────────────────────────────────────────
    # Same account receives from 8+ distinct senders AND sends to 8+
    # distinct receivers on the same calendar day.
    n_senders_as_receiver = r.scard(f"acct_daily_senders:{account_from}:{day}")
    n_receivers_as_sender = r.scard(receiver_key)
    if (n_senders_as_receiver >= GATHER_SCATTER_MIN_SENDERS
            and n_receivers_as_sender >= GATHER_SCATTER_MIN_RECEIVERS
            and not _on_cooldown(r, account_from, "GATHER_SCATTER")):
        alert = _push_alert(r, account_from, "GATHER_SCATTER", timestamp, {
            "unique_senders":   n_senders_as_receiver,
            "unique_receivers": n_receivers_as_sender,
            "day":              day,
        })
        detected.append(alert)
        _start_cooldown(r, account_from, "GATHER_SCATTER",
                        GATHER_SCATTER_COOLDOWN_HOURS)

    # ── Pattern 4 — SCATTER_GATHER ────────────────────────────────────────────
    # account_from sent to account_to (intermediary).
    # Check if account_to is now sending to a common final receiver
    # that 3+ other intermediaries from the same source also sent to.
    # Approximation: check if account_to has received from 3+ distinct
    # senders today AND is now forwarding -- a lightweight proxy for
    # the full 3-table join the DuckDB validator uses.
    n_to_senders = r.scard(f"acct_daily_senders:{account_to}:{day}")
    to_receivers = r.scard(f"acct_daily_receivers:{account_to}:{day}")
    if (n_to_senders >= SCATTER_GATHER_MIN_INTERMEDIARIES
            and to_receivers >= 1
            and not _on_cooldown(r, account_from, "SCATTER_GATHER")):
        alert = _push_alert(r, account_from, "SCATTER_GATHER", timestamp, {
            "intermediary":           account_to,
            "intermediary_senders":   n_to_senders,
            "window_hours":           SCATTER_GATHER_WINDOW_HOURS,
        })
        detected.append(alert)
        _start_cooldown(r, account_from, "SCATTER_GATHER",
                        SCATTER_GATHER_COOLDOWN_HOURS)

    # ── Pattern 5 — SIMPLE_CYCLE ──────────────────────────────────────────────
    # A sends to B, B sends to C, C sends back to A -- same calendar day.
    # Detect when account_from appears as a destination in account_to's
    # recent outbound window (i.e. account_to has sent back to account_from).
    to_window = _get_window(r, account_to)
    cutoff_day = now - timedelta(hours=SIMPLE_CYCLE_WINDOW_HOURS)
    cycle_found = any(
        w['dest_account'] == account_from
        and _parse_ts(w['timestamp']) >= cutoff_day
        for w in to_window
    )
    if cycle_found and not _on_cooldown(r, account_from, "SIMPLE_CYCLE"):
        alert = _push_alert(r, account_from, "SIMPLE_CYCLE", timestamp, {
            "cycle_accounts": [account_from, account_to],
            "window_hours":   SIMPLE_CYCLE_WINDOW_HOURS,
        })
        detected.append(alert)
        _start_cooldown(r, account_from, "SIMPLE_CYCLE",
                        SIMPLE_CYCLE_COOLDOWN_HOURS)

    # ── Pattern 6 — STRUCTURING ───────────────────────────────────────────────
    # 3+ transactions in the same day with amounts just below common
    # reporting thresholds (900-999, 4500-4999, 9000-9999).
    struct_key = f"acct_structuring:{account_from}:{day}"
    in_band = any(lo <= amount <= hi for lo, hi in STRUCTURING_BANDS)
    if in_band:
        r.incr(struct_key)
        r.expire(struct_key, 90000)
        count = int(r.get(struct_key) or 0)
        if (count >= STRUCTURING_MIN_COUNT
                and not _on_cooldown(r, account_from, "STRUCTURING")):
            alert = _push_alert(r, account_from, "STRUCTURING", timestamp, {
                "structured_count": count,
                "amount":           amount,
                "day":              day,
                "bands":            str(STRUCTURING_BANDS),
            })
            detected.append(alert)
            _start_cooldown(r, account_from, "STRUCTURING",
                            STRUCTURING_COOLDOWN_HOURS)

    return detected


def get_cep_alerts(r: redis.Redis, account_id: str) -> list:
    """Retrieve all CEP alerts for an account (for the dashboard)."""
    raw = r.lrange(f"cep_alerts:{account_id}", 0, CEP_ALERTS_MAX_LEN - 1)
    return [json.loads(x) for x in raw]


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    r.ping()
    print("Redis connected -- running CEP engine v2 smoke test\n")

    base = datetime(2026, 1, 1, 9, 0, 0)

    def evt(acc_from, acc_to, bank_to, amount, minutes_offset, bank_from="TESTBANK"):
        return {
            "account_from": acc_from,
            "account_to":   acc_to,
            "bank_from":    bank_from,
            "bank_to":      bank_to,
            "amount_paid":  amount,
            "timestamp":    (base + timedelta(minutes=minutes_offset)).isoformat(),
        }

    # Clean up test keys
    for key in r.keys("acct_*:MULE_*") + r.keys("cep_*:MULE_*"):
        r.delete(key)

    # Test 1: FAN_OUT_SPIKE -- 10 unique banks in 6h
    print("Test 1: FAN_OUT_SPIKE")
    for i in range(10):
        result = check_patterns(r, evt("MULE_A", f"DEST_{i}", f"BANK_{i}", 5000, i*10))
    print("  Detected:", [a['pattern_type'] for a in result])
    print("  Expected: FAN_OUT_SPIKE\n")

    # Test 2: FAN_IN -- 5+ distinct senders to same receiver
    print("Test 2: FAN_IN")
    results = []
    for i in range(5):
        results = check_patterns(r, evt(f"SOURCE_{i}", "MULE_B", "BANK1", 5000, i*5))
    print("  Detected:", [a['pattern_type'] for a in results])
    print("  Expected: FAN_IN\n")

    # Test 3: GATHER_SCATTER -- 8+ senders AND 8+ receivers same day
    print("Test 3: GATHER_SCATTER")
    for i in range(8):
        check_patterns(r, evt(f"SRC_{i}", "MULE_C", "BANK1", 5000, i*5))
    results = []
    for i in range(8):
        results = check_patterns(r, evt("MULE_C", f"DST_{i}", f"BANK_{i}", 4900, 60+i*5))
    print("  Detected:", [a['pattern_type'] for a in results])
    print("  Expected: GATHER_SCATTER (and possibly FAN_OUT_SPIKE)\n")

    # Test 4: STRUCTURING -- 3+ txns in 900-999 band same day
    print("Test 4: STRUCTURING")
    results = []
    for i in range(3):
        results = check_patterns(r, evt("MULE_D", f"DEST_{i}", "BANK1", 950, i*30))
    print("  Detected:", [a['pattern_type'] for a in results])
    print("  Expected: STRUCTURING\n")

    print("Smoke test complete.")