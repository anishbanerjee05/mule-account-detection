# account_baseline.py
#
# Per-account behavioral baseline tracking for anomaly detection.
#
# PURPOSE:
# Every account has its own normal behavior. A payment processor receiving
# 500 inbound transfers per day is normal for that account. A personal
# account receiving 500 inbound transfers is anomalous.
#
# Fixed thresholds cannot distinguish these. Statistical baselines can.
#
# This module maintains rolling daily statistics for every account:
#   - Daily sender count (how many unique accounts sent to this account)
#   - Daily inbound amount (total inbound Rs. per day)
#   - Daily transaction count (outbound transactions per day)
#   - Daily outbound amount (total outbound Rs. per day)
#
# For each metric we track:
#   - n:   number of days observed
#   - mu:  rolling mean (Welford's online algorithm -- no history stored)
#   - M2:  sum of squared deviations (for variance/std computation)
#
# Welford's algorithm updates mean and variance in O(1) per observation
# without storing historical values. Numerically stable for large n.
#
# ANOMALY SCORE:
# z_score = (today_value - mu) / std
#
# z >= 2.0: unusual (5% probability under normal distribution)
# z >= 3.0: very unusual (0.13% probability)
# z >= 4.0: almost certainly anomalous (0.003% probability)
#
# We use z >= 3.0 as the detection threshold because:
# - It is the standard statistical definition of an outlier
# - It adapts to each account's own history
# - It requires no hardcoded amount or count thresholds
# - It is defensible to regulators: "today's value was 3 standard
#   deviations above this account's 30-day mean"
#
# REDIS STRUCTURE:
# acct_baseline:{account_id} -> hash with fields:
#   in_senders_n, in_senders_mu, in_senders_M2   -- inbound unique senders/day
#   in_amount_n,  in_amount_mu,  in_amount_M2    -- inbound total amount/day
#   out_count_n,  out_count_mu,  out_count_M2    -- outbound txn count/day
#   out_amount_n, out_amount_mu, out_amount_M2   -- outbound total amount/day
#   last_updated_day                              -- date string YYYY-MM-DD
#
# acct_daily:{account_id}:{YYYY-MM-DD} -> hash with fields:
#   in_senders:  set of unique inbound sender IDs (stored as comma-separated)
#   in_amount:   total inbound amount today
#   out_count:   outbound transaction count today
#   out_amount:  total outbound amount today
# TTL: 25 hours (captures one full day, expires before next day's data)

import json
import math
import redis
from datetime import datetime, timedelta
from typing import Optional, Tuple


# ── Welford's online algorithm ────────────────────────────────────────────────

def _welford_update(n: float, mu: float, M2: float,
                    new_value: float) -> Tuple[float, float, float]:
    """
    Update running mean and variance using Welford's online algorithm.
    Numerically stable, O(1) time and space.

    Returns updated (n, mu, M2) where:
        variance = M2 / n          (population variance)
        std      = sqrt(M2 / n)    (population std deviation)
    """
    n  += 1
    delta  = new_value - mu
    mu    += delta / n
    delta2 = new_value - mu
    M2    += delta * delta2
    return n, mu, M2


def _welford_std(n: float, M2: float) -> float:
    """Compute standard deviation from Welford accumulators."""
    if n < 2:
        return 0.0
    variance = M2 / n
    return math.sqrt(max(variance, 0.0))


def _z_score(value: float, mu: float, std: float) -> float:
    """
    Compute z-score: how many standard deviations is value from mean.
    Returns 0.0 if std is too small (account behavior is perfectly consistent
    or we have insufficient history).
    """
    if std < 1e-9:
        return 0.0
    return (value - mu) / std


# ── Baseline read/write ───────────────────────────────────────────────────────

def get_baseline(r: redis.Redis, account_id: str) -> dict:
    """
    Read baseline statistics for an account.
    Returns dict with all Welford accumulators, or empty dict if no history.
    """
    raw = r.hgetall(f"acct_baseline:{account_id}")
    if not raw:
        return {}
    return {k: float(v) if k != 'last_updated_day' else v
            for k, v in raw.items()}


def _update_baseline_field(r: redis.Redis, account_id: str,
                            prefix: str, value: float):
    """
    Update Welford accumulators for one metric (prefix) with a new daily value.
    Called once per day per metric when the day rolls over.
    """
    key = f"acct_baseline:{account_id}"
    raw = r.hmget(key, f"{prefix}_n", f"{prefix}_mu", f"{prefix}_M2")

    n  = float(raw[0]) if raw[0] else 0.0
    mu = float(raw[1]) if raw[1] else 0.0
    M2 = float(raw[2]) if raw[2] else 0.0

    n, mu, M2 = _welford_update(n, mu, M2, value)

    r.hset(key, mapping={
        f"{prefix}_n":  str(n),
        f"{prefix}_mu": str(mu),
        f"{prefix}_M2": str(M2),
    })
    # Baseline never expires — it accumulates indefinitely
    r.persist(key)


# ── Daily accumulator ─────────────────────────────────────────────────────────

def record_outbound_transaction(r: redis.Redis, account_id: str,
                                 amount: float, day: str):
    """
    Record one outbound transaction for today.
    Called by consumer.py for every transaction where account_id is the sender.
    """
    daily_key = f"acct_daily:{account_id}:{day}"
    pipe = r.pipeline()
    pipe.hincrbyfloat(daily_key, "out_count",  1.0)
    pipe.hincrbyfloat(daily_key, "out_amount", amount)
    pipe.expire(daily_key, 90000)  # 25h TTL
    pipe.execute()


def record_inbound_transaction(r: redis.Redis, account_id: str,
                                sender_id: str, amount: float, day: str):
    """
    Record one inbound transaction for today.
    Called by consumer.py for every transaction where account_id is the receiver.
    """
    daily_key = f"acct_daily:{account_id}:{day}"
    pipe = r.pipeline()
    # Track unique senders using a Redis set
    pipe.sadd(f"acct_daily_in_senders:{account_id}:{day}", sender_id)
    pipe.expire(f"acct_daily_in_senders:{account_id}:{day}", 90000)
    pipe.hincrbyfloat(daily_key, "in_amount", amount)
    pipe.expire(daily_key, 90000)
    pipe.execute()


def get_today_stats(r: redis.Redis, account_id: str, day: str) -> dict:
    """
    Get today's accumulated stats for an account.
    Returns dict with in_senders, in_amount, out_count, out_amount.
    """
    daily_key   = f"acct_daily:{account_id}:{day}"
    sender_key  = f"acct_daily_in_senders:{account_id}:{day}"

    pipe = r.pipeline()
    pipe.hgetall(daily_key)
    pipe.scard(sender_key)
    results = pipe.execute()

    daily = results[0] or {}
    n_senders = results[1] or 0

    return {
        "in_senders":  int(n_senders),
        "in_amount":   float(daily.get("in_amount", 0)),
        "out_count":   int(float(daily.get("out_count", 0))),
        "out_amount":  float(daily.get("out_amount", 0)),
    }


def rollover_daily_baseline(r: redis.Redis, account_id: str, day: str):
    """
    Called once per day (when the date changes) to push yesterday's
    daily stats into the Welford baseline accumulators.

    This is called lazily — when we see a transaction on a new day,
    we check if yesterday's data has been rolled over and do so if not.
    """
    baseline_key = f"acct_baseline:{account_id}"
    last_day = r.hget(baseline_key, "last_updated_day")

    if last_day == day:
        return  # Already rolled over today

    # Get yesterday's stats (day before current day)
    yesterday = (datetime.strptime(day, "%Y-%m-%d") -
                 timedelta(days=1)).strftime("%Y-%m-%d")
    yesterday_stats = get_today_stats(r, account_id, yesterday)

    # Only update baseline if we have actual activity yesterday
    # (don't count silent days — they would artificially lower the mean)
    if yesterday_stats["out_count"] > 0 or yesterday_stats["in_amount"] > 0:
        _update_baseline_field(r, account_id, "in_senders",
                               float(yesterday_stats["in_senders"]))
        _update_baseline_field(r, account_id, "in_amount",
                               yesterday_stats["in_amount"])
        _update_baseline_field(r, account_id, "out_count",
                               float(yesterday_stats["out_count"]))
        _update_baseline_field(r, account_id, "out_amount",
                               yesterday_stats["out_amount"])

    r.hset(baseline_key, "last_updated_day", day)


# ── Anomaly scoring ───────────────────────────────────────────────────────────

def compute_anomaly_scores(r: redis.Redis, account_id: str,
                           day: str) -> dict:
    """
    Compute z-scores for today's behavior vs this account's historical baseline.

    Returns dict with:
        in_senders_z:  z-score for today's unique inbound sender count
        in_amount_z:   z-score for today's total inbound amount
        out_count_z:   z-score for today's outbound transaction count
        out_amount_z:  z-score for today's total outbound amount
        days_observed: how many days of history we have
        today:         today's raw stats
        baseline:      mean and std for each metric
        interpretation: human-readable explanation of the highest anomaly

    z >= 3.0 means the value is statistically anomalous for this account.
    z < 2.0 means the value is within normal range for this account.
    """
    # Roll over yesterday's data into baseline if needed
    rollover_daily_baseline(r, account_id, day)

    # Get today's stats and baseline
    today = get_today_stats(r, account_id, day)
    baseline_key = f"acct_baseline:{account_id}"
    raw = r.hgetall(baseline_key)

    if not raw:
        # No history at all — cannot compute z-scores
        return {
            "in_senders_z":  0.0,
            "in_amount_z":   0.0,
            "out_count_z":   0.0,
            "out_amount_z":  0.0,
            "days_observed": 0,
            "today":         today,
            "baseline":      {},
            "sufficient_history": False,
            "interpretation": "Insufficient history for anomaly scoring",
        }

    def _get_accum(prefix):
        n  = float(raw.get(f"{prefix}_n",  0))
        mu = float(raw.get(f"{prefix}_mu", 0))
        M2 = float(raw.get(f"{prefix}_M2", 0))
        std = _welford_std(n, M2)
        return n, mu, std

    n_s, mu_s, std_s = _get_accum("in_senders")
    n_a, mu_a, std_a = _get_accum("in_amount")
    n_c, mu_c, std_c = _get_accum("out_count")
    n_o, mu_o, std_o = _get_accum("out_amount")

    days_observed = int(max(n_s, n_a, n_c, n_o))

    # Need at least 7 days of history for meaningful z-scores
    sufficient_history = days_observed >= 7

    z_in_senders = _z_score(today["in_senders"], mu_s, std_s)
    z_in_amount  = _z_score(today["in_amount"],  mu_a, std_a)
    z_out_count  = _z_score(today["out_count"],  mu_c, std_c)
    z_out_amount = _z_score(today["out_amount"], mu_o, std_o)

    # Build human-readable interpretation of the highest anomaly
    anomalies = []
    if z_in_senders >= 3.0:
        anomalies.append(
            f"Inbound senders: {today['in_senders']} today vs "
            f"mean {mu_s:.1f} (z={z_in_senders:.1f})"
        )
    if z_in_amount >= 3.0:
        anomalies.append(
            f"Inbound amount: Rs.{today['in_amount']:,.0f} today vs "
            f"mean Rs.{mu_a:,.0f} (z={z_in_amount:.1f})"
        )
    if z_out_count >= 3.0:
        anomalies.append(
            f"Outbound count: {today['out_count']} today vs "
            f"mean {mu_c:.1f} (z={z_out_count:.1f})"
        )
    if z_out_amount >= 3.0:
        anomalies.append(
            f"Outbound amount: Rs.{today['out_amount']:,.0f} today vs "
            f"mean Rs.{mu_o:,.0f} (z={z_out_amount:.1f})"
        )

    interpretation = (
        " | ".join(anomalies) if anomalies
        else "Behavior within normal range for this account"
    )

    return {
        "in_senders_z":  round(z_in_senders, 3),
        "in_amount_z":   round(z_in_amount,  3),
        "out_count_z":   round(z_out_count,  3),
        "out_amount_z":  round(z_out_amount, 3),
        "days_observed": days_observed,
        "sufficient_history": sufficient_history,
        "today":  today,
        "baseline": {
            "in_senders_mean": round(mu_s, 2),
            "in_senders_std":  round(std_s, 2),
            "in_amount_mean":  round(mu_a, 2),
            "in_amount_std":   round(std_a, 2),
            "out_count_mean":  round(mu_c, 2),
            "out_count_std":   round(std_c, 2),
            "out_amount_mean": round(mu_o, 2),
            "out_amount_std":  round(std_o, 2),
        },
        "interpretation": interpretation,
    }


def is_anomalous(scores: dict, z_threshold: float = 3.0) -> Tuple[bool, str]:
    """
    Returns (True, reason) if any z-score exceeds the threshold.
    Returns (False, "") if behavior is within normal range.

    Uses z_threshold=3.0 by default (0.13% probability = genuine anomaly).
    """
    if not scores.get("sufficient_history", False):
        return False, ""

    checks = [
        ("in_senders_z",  "inbound sender count"),
        ("in_amount_z",   "inbound amount"),
        ("out_count_z",   "outbound transaction count"),
        ("out_amount_z",  "outbound amount"),
    ]

    max_z    = 0.0
    max_name = ""
    for field, name in checks:
        z = scores.get(field, 0.0)
        if z > max_z:
            max_z    = z
            max_name = name

    if max_z >= z_threshold:
        return True, (
            f"z={max_z:.2f} on {max_name} — "
            f"{max_z:.1f} standard deviations above this account's "
            f"{scores['days_observed']}-day historical mean"
        )
    return False, ""