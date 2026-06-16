import json
import requests
import redis
import numpy as np
from confluent_kafka import Consumer, Producer
from datetime import datetime
import math

print("Starting mule detection consumer v4 (dynamic features)...")

# Redis for live account state
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Confluent Kafka consumer
print("Connecting to Kafka...")
consumer = Consumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'mule-detector-v5',
    'auto.offset.reset': 'earliest',
    'enable.auto.commit': True,
    'fetch.min.bytes': 1,
    'fetch.wait.max.ms': 500,
})
consumer.subscribe(['transactions-raw'])

# Kafka producer for risk scores
risk_producer = Producer({'bootstrap.servers': 'localhost:9092'})

API_URL = "http://localhost:8000"

processed = 0
flagged   = 0

# ── Dynamic feature computation ───────────────────────────────────────────────
def update_account_state(r, account_id, amount, bank_from, bank_to, payment_format):
    key = f"acct:{account_id}"
    state = r.hgetall(key)

    # Initialise if new account
    if not state:
        state = {
            'total_txn':        '0',
            'total_sent':       '0.0',
            'total_received':   '0.0',
            'max_amount':       '0.0',
            'sum_amount':       '0.0',
            'unique_dest':      '',
            'unique_banks':     '',
            'round_count':      '0',
            'first_seen':       str(processed),
            'last_seen':        str(processed),
            'sum_sq_amount':    '0.0',
        }

    # Update state
    total_txn      = int(state['total_txn']) + 1
    total_sent     = float(state['total_sent']) + amount
    max_amount     = max(float(state['max_amount']), amount)
    sum_amount     = float(state['sum_amount']) + amount
    sum_sq_amount  = float(state['sum_sq_amount']) + amount ** 2

    # Unique destinations (store as comma-separated, cap at 100)
    unique_dest_set = set(state['unique_dest'].split(',')) if state['unique_dest'] else set()
    unique_dest_set.add(bank_to)
    unique_dest_set.discard('')

    # Unique banks
    unique_banks_set = set(state['unique_banks'].split(',')) if state['unique_banks'] else set()
    unique_banks_set.add(bank_to)
    unique_banks_set.discard('')

    # Round amount flag
    round_count = int(state['round_count']) + (1 if amount % 1000 == 0 and amount > 0 else 0)

    # Save updated state
    new_state = {
        'total_txn':      str(total_txn),
        'total_sent':     str(total_sent),
        'max_amount':     str(max_amount),
        'sum_amount':     str(sum_amount),
        'sum_sq_amount':  str(sum_sq_amount),
        'unique_dest':    ','.join(list(unique_dest_set)[:100]),
        'unique_banks':   ','.join(list(unique_banks_set)[:50]),
        'round_count':    str(round_count),
        'first_seen':     state['first_seen'],
        'last_seen':      str(processed),
    }
    r.hset(key, mapping=new_state)
    r.expire(key, 86400)  # 24 hour TTL

    return new_state, total_txn, total_sent, max_amount, sum_amount, sum_sq_amount, unique_dest_set, unique_banks_set, round_count

def compute_features(state, total_txn, total_sent, max_amount,
                     sum_amount, sum_sq_amount, unique_dest_set,
                     unique_banks_set, round_count, amount):

    avg_amount = sum_amount / total_txn if total_txn > 0 else 0

    # Velocity ratio — proxy using sent/max_amount
    velocity_ratio = min(amount / (max_amount + 1e-9), 1.0)

    # Balance retention — inverse of velocity
    balance_retention = max(0, 1 - velocity_ratio)

    # Fan-out degree — unique destinations (normalised)
    fan_out = min(len(unique_dest_set) / 10.0, 1.0)

    # Cross-bank degree
    cross_bank = min(len(unique_banks_set) / 5.0, 1.0)

    # Dormancy spike — high amount relative to average
    dormancy_spike = min(amount / (avg_amount + 1e-9) / 10.0, 1.0) if avg_amount > 0 else 0.5

    # Round amount
    round_flag  = 1.0 if amount % 1000 == 0 and amount > 0 else 0.0
    round_ratio = round_count / total_txn if total_txn > 0 else 0.0

    # Type entropy — fixed at low value (single type stream)
    type_entropy = 0.1

    # Account age at activation (normalised by stream position)
    first_seen = int(state.get('first_seen', 0))
    acct_age   = min(first_seen / 10000.0, 1.0)

    # Fan-in — not directly computable, use 0
    fan_in = 0.0

    # Regularity index
    regularity = 0.5

    # Max amount ratio
    max_amount_ratio = min(max_amount / (avg_amount + 1e-9) / 5.0, 1.0)

    # Total transactions (normalised)
    total_txn_norm = min(total_txn / 100.0, 1.0)

    # Structuring score
    structuring = round_ratio * 0.5

    # Freq acceleration
    freq_accel = min(total_txn / 10.0, 1.0)

    # Amount variance score
    if total_txn > 1:
        mean_sq = sum_sq_amount / total_txn
        var     = mean_sq - (avg_amount ** 2)
        std     = math.sqrt(max(var, 0))
        cv      = std / (avg_amount + 1e-9)
        amount_variance = 1 / (1 + cv)
    else:
        amount_variance = 1.0

    # Session burst — proxy: high single transaction relative to count
    session_burst = min(amount / (total_txn * avg_amount + 1e-9), 1.0)

    # Dormancy ratio
    dormancy_ratio = max(0, 1 - total_txn_norm)

    # Onboarding velocity
    onboarding = min(amount / (max_amount + 1e-9), 1.0)

    return [
        velocity_ratio,    # 0
        balance_retention, # 1
        fan_out,           # 2
        dormancy_spike,    # 3
        round_flag,        # 4
        round_ratio,       # 5
        type_entropy,      # 6
        acct_age,          # 7
        fan_in,            # 8
        regularity,        # 9
        max_amount_ratio,  # 10
        total_txn_norm,    # 11
        avg_amount,        # 12
        max_amount,        # 13
        structuring,       # 14
        freq_accel,        # 15
        amount_variance,   # 16
        session_burst,     # 17
        dormancy_ratio,    # 18
        onboarding,        # 19
    ]

# ── Main loop ─────────────────────────────────────────────────────────────────
print("Listening for transactions...\n")
print(f"{'Time':<10} {'Account':<18} {'Amount':>15} {'Prob':>8} {'Tier':<22} {'Alert'}")
print("-" * 85)

try:
    while True:
        msg = consumer.poll(timeout=1.0)

        if msg is None:
            if processed > 0 and processed % 100 == 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Processed: {processed:,} | Flagged: {flagged}")
            continue

        if msg.error():
            print(f"Kafka error: {msg.error()}")
            continue

        try:
            event      = json.loads(msg.value().decode('utf-8'))
            account_id = str(event['account_from'])
            amount     = float(event['amount_paid'])
            is_launder = int(event['is_laundering'])
            bank_from  = str(event['bank_from'])
            bank_to    = str(event['bank_to'])
            pay_fmt    = str(event['payment_format'])

            # Update account state and compute features dynamically
            state, total_txn, total_sent, max_amt, sum_amt, sum_sq, \
            unique_dest, unique_banks, round_cnt = update_account_state(
                r, account_id, amount, bank_from, bank_to, pay_fmt
            )

            features = compute_features(
                state, total_txn, total_sent, max_amt,
                sum_amt, sum_sq, unique_dest, unique_banks,
                round_cnt, amount
            )

            # Score via API
            resp = requests.post(
                f"{API_URL}/score/account",
                json={"account_id": account_id, "features": features},
                timeout=5
            )
            result       = resp.json()
            prob         = result['fraud_probability']
            tier         = result['risk_tier']
            flagged_flag = result['flagged']

            # Store alert
            if flagged_flag:
                flagged += 1
                alert = {
                    "account_id":     account_id,
                    "amount":         amount,
                    "probability":    prob,
                    "tier":           tier,
                    "timestamp":      event['timestamp'],
                    "is_laundering":  is_launder,
                    "payment_format": pay_fmt,
                    "features": {
                        "velocity_ratio":    round(features[0], 4),
                        "fan_out_degree":    round(features[2], 4),
                        "dormancy_spike":    round(features[3], 4),
                        "total_txn":         total_txn,
                        "max_amount":        max_amt,
                    }
                }
                r.lpush("alerts", json.dumps(alert))
                r.ltrim("alerts", 0, 999)

            processed += 1

            # Print every flagged + every 50th
            if flagged_flag or processed % 50 == 0:
                ts        = datetime.now().strftime("%H:%M:%S")
                alert_str = "🚨 FLAGGED" if flagged_flag else ""
                laund_str = "⚠ LAUNDER" if is_launder else ""
                print(f"{ts:<10} {account_id[:16]:<18} {amount:>15,.2f} "
                      f"{prob:>8.4f} {tier:<22} {alert_str}{laund_str}")

        except Exception as e:
            print(f"Error: {e}")
            continue

except KeyboardInterrupt:
    print(f"\nStopped. Processed: {processed:,} | Flagged: {flagged}")
finally:
    consumer.close()