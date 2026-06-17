import json
import math
import pickle
import redis
import numpy as np
import pandas as pd
from confluent_kafka import Consumer, Producer
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("Starting mule detection consumer v7 (hybrid in-process model)...")

# ── Load models directly — no HTTP calls ──────────────────────────────────────
print("Loading hybrid XGBoost model...")
with open('xgb_hybrid_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)
with open('scaler_hybrid.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('hybrid_model_config.json', 'r') as f:
    config = json.load(f)

optimal_threshold = config['optimal_threshold']
L3_THRESHOLD      = 0.70
L2_THRESHOLD      = 0.50
L1_THRESHOLD      = 0.30

print(f"Model loaded. Optimal threshold: {optimal_threshold:.4f}")
print(f"Tiers — L1: {L1_THRESHOLD}, L2: {L2_THRESHOLD}, L3: {L3_THRESHOLD}")
print(f"Features: {config['n_total_features']} (12 behavioral + 64 GNN embeddings)")

# ── Redis ─────────────────────────────────────────────────────────────────────
r = redis.Redis(host='localhost', port=6379, decode_responses=True)
r.ping()
print("Redis connected")

# ── Load IBM AML laundering labels ────────────────────────────────────────────
print("Loading IBM AML laundering account labels...")
try:
    df_ibm = pd.read_csv('ibm_aml_small.csv')
    laundering_accounts = set(
        df_ibm[df_ibm['Is Laundering'] == 1]['Account'].astype(str).unique()
    )
    print(f"Confirmed laundering accounts: {len(laundering_accounts):,}")
except Exception as e:
    laundering_accounts = set()
    print(f"Could not load IBM AML labels: {e}")

# ── Kafka consumer ────────────────────────────────────────────────────────────
group_id = f'mule-detector-{int(datetime.now().timestamp())}'
print(f"Consumer group: {group_id}")

consumer = Consumer({
    'bootstrap.servers':  'localhost:9092',
    'group.id':           group_id,
    'auto.offset.reset':  'earliest',
    'enable.auto.commit': True,
    'fetch.min.bytes':    1,
    'fetch.wait.max.ms':  500,
})
consumer.subscribe(['transactions-raw'])
print("Subscribed to transactions-raw")

# ── Kafka producer for risk scores ────────────────────────────────────────────
risk_producer = Producer({'bootstrap.servers': 'localhost:9092'})

processed = 0
flagged   = 0

# ── Account state management ──────────────────────────────────────────────────
def update_account_state(account_id, amount, bank_to):
    key   = f"acct:{account_id}"
    state = r.hgetall(key)

    if not state:
        state = {
            'total_txn':     '0',
            'total_sent':    '0.0',
            'max_amount':    '0.0',
            'sum_amount':    '0.0',
            'sum_sq_amount': '0.0',
            'unique_dest':   '',
            'unique_banks':  '',
            'round_count':   '0',
            'first_seen':    str(processed),
        }

    total_txn     = int(state['total_txn'])      + 1
    total_sent    = float(state['total_sent'])   + amount
    max_amount    = max(float(state['max_amount']), amount)
    sum_amount    = float(state['sum_amount'])   + amount
    sum_sq_amount = float(state['sum_sq_amount']) + amount ** 2

    unique_dest  = set(state['unique_dest'].split(','))  if state['unique_dest']  else set()
    unique_banks = set(state['unique_banks'].split(',')) if state['unique_banks'] else set()
    unique_dest.discard('')
    unique_banks.discard('')
    unique_dest.add(bank_to)
    unique_banks.add(bank_to)

    round_count = int(state['round_count']) + (
        1 if amount % 1000 == 0 and amount > 0 else 0
    )

    new_state = {
        'total_txn':     str(total_txn),
        'total_sent':    str(total_sent),
        'max_amount':    str(max_amount),
        'sum_amount':    str(sum_amount),
        'sum_sq_amount': str(sum_sq_amount),
        'unique_dest':   ','.join(list(unique_dest)[:100]),
        'unique_banks':  ','.join(list(unique_banks)[:50]),
        'round_count':   str(round_count),
        'first_seen':    state.get('first_seen', str(processed)),
    }
    r.hset(key, mapping=new_state)
    r.expire(key, 86400)

    return (new_state, total_txn, total_sent, max_amount,
            sum_amount, sum_sq_amount, unique_dest, unique_banks, round_count)


# ── Behavioral feature computation ────────────────────────────────────────────
def compute_behavioral_features(state, total_txn, total_sent, max_amount,
                                 sum_amount, sum_sq_amount, unique_dest,
                                 unique_banks, round_count, amount):
    avg_amount = sum_amount / total_txn if total_txn > 0 else amount

    velocity_ratio    = (min(amount / (max_amount + 1e-9), 1.0)
                         if total_txn > 1 else 0.95)
    balance_retention = max(0.0, 1.0 - velocity_ratio)
    fan_out           = float(len(unique_dest))
    cross_bank        = float(len(unique_banks))
    round_ratio       = round_count / total_txn if total_txn > 0 else 0.0
    structuring       = round_ratio * 0.5
    dormancy_ratio    = max(0.0, 1.0 - min(total_txn / 10.0, 1.0))

    if total_txn > 1:
        mean_sq         = sum_sq_amount / total_txn
        var             = mean_sq - avg_amount ** 2
        std             = math.sqrt(max(var, 0))
        cv              = std / (avg_amount + 1e-9)
        amount_variance = 1.0 / (1.0 + cv)
    else:
        amount_variance = 1.0

    session_burst = min(amount / (avg_amount * 5 + 1e-9), 1.0)

    return [
        velocity_ratio,    # velocity_ratio
        balance_retention, # balance_retention_rate
        fan_out,           # fan_out_degree       (raw count)
        cross_bank,        # cross_bank_degree    (raw count)
        round_ratio,       # round_amount_ratio
        structuring,       # structuring_score
        amount_variance,   # amount_variance_score
        session_burst,     # session_burst_score
        dormancy_ratio,    # dormancy_ratio
        float(total_txn),  # total_transactions   (raw)
        avg_amount,        # avg_amount            (raw)
        max_amount,        # max_amount            (raw)
    ]


# ── GNN embedding retrieval ───────────────────────────────────────────────────
def get_gnn_embedding(account_id):
    emb_json = r.get(f"embed:{account_id}")
    if emb_json:
        return json.loads(emb_json), True
    return [0.0] * 64, False


# ── Risk tier ─────────────────────────────────────────────────────────────────
def get_risk_tier(prob):
    if prob >= L3_THRESHOLD:   return "L3 — AUTO FREEZE"
    elif prob >= L2_THRESHOLD: return "L2 — SOFT HOLD"
    elif prob >= L1_THRESHOLD: return "L1 — REVIEW"
    else:                      return "CLEAR"


# ── Main loop ─────────────────────────────────────────────────────────────────
print("\nListening for transactions...\n")
print(f"{'Time':<10} {'Account':<18} {'Amount':>15} "
      f"{'Prob':>8} {'Tier':<22} {'GNN':<6} {'Alert'}")
print("-" * 90)

try:
    while True:
        msg = consumer.poll(timeout=1.0)

        if msg is None:
            if processed > 0 and processed % 500 == 0:
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
            bank_to    = str(event['bank_to'])

            # Step 1 — Update account state in Redis
            (state, total_txn, total_sent, max_amt,
             sum_amt, sum_sq, unique_dest,
             unique_banks, round_cnt) = update_account_state(
                account_id, amount, bank_to
            )

            # Step 2 — Compute 12 behavioral features
            behavioral = compute_behavioral_features(
                state, total_txn, total_sent, max_amt,
                sum_amt, sum_sq, unique_dest,
                unique_banks, round_cnt, amount
            )

            # Step 3 — Retrieve 64-dim GNN embedding from Redis
            gnn_embedding, has_embed = get_gnn_embedding(account_id)

            # Step 4 — Concatenate 76-feature hybrid vector
            features_hybrid = behavioral + gnn_embedding

            # Step 5 — Scale and score in-process
            X        = np.array(features_hybrid).reshape(1, -1)
            X_scaled = scaler.transform(X)
            prob     = float(xgb_model.predict_proba(X_scaled)[0][1])

            flagged_flag = prob >= optimal_threshold
            tier         = get_risk_tier(prob)

            # Step 6 — Account-level laundering status
            is_confirmed_launder = (
                account_id in laundering_accounts or
                is_launder == 1 or
                r.get(f"launder_status:{account_id}") == "1"
            )
            if is_launder == 1 or account_id in laundering_accounts:
                r.set(f"launder_status:{account_id}", "1", ex=86400 * 7)

            # Step 7 — Store alert in Redis if flagged
            if flagged_flag:
                flagged += 1
                alert = {
                    "account_id":         account_id,
                    "amount":             amount,
                    "probability":        round(prob, 4),
                    "tier":               tier,
                    "timestamp":          event['timestamp'],
                    "is_laundering":      is_launder,
                    "account_laundering": is_confirmed_launder,
                    "payment_format":     event.get('payment_format', 'unknown'),
                    "total_txn":          total_txn,
                    "fan_out":            len(unique_dest),
                    "cross_bank":         len(unique_banks),
                    "has_gnn_embed":      has_embed,
                    "max_amount":         max_amt,
                    "avg_amount":         (sum_amt / total_txn
                                           if total_txn > 0 else amount),
                }
                r.lpush("alerts", json.dumps(alert))
                r.ltrim("alerts", 0, 999)

            # Step 8 — Publish risk score to Kafka
            risk_producer.produce(
                'risk-scores',
                value=json.dumps({
                    "account_id":  account_id,
                    "probability": round(prob, 4),
                    "tier":        tier,
                    "flagged":     flagged_flag,
                    "timestamp":   event['timestamp'],
                }).encode('utf-8')
            )
            risk_producer.poll(0)

            processed += 1

            # Step 9 — Print every flagged + every 10th
            if flagged_flag or processed % 10 == 0:
                ts        = datetime.now().strftime("%H:%M:%S")
                alert_str = "FLAGGED" if flagged_flag         else ""
                laund_str = "LAUNDER" if is_confirmed_launder else ""
                gnn_str   = "GNN+" if has_embed else "GNN-"
                print(f"{ts:<10} {account_id[:16]:<18} {amount:>15,.2f} "
                      f"{prob:>8.4f} {tier:<22} {gnn_str:<6} "
                      f"{alert_str} {laund_str}")

        except Exception as e:
            print(f"Error: {e}")
            continue

except KeyboardInterrupt:
    print(f"\nStopped. Processed: {processed:,} | Flagged: {flagged}")
finally:
    consumer.close()