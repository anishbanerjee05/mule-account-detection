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


from topic_config import get_active_topic

# ── GNN embedding retrieval — batch-capable ───────────────────────────────────
BATCH_SIZE = 100  # number of transactions to accumulate before one batch call

def get_gnn_embeddings_batch(batch_requests):
    """
    Fetch GNN embeddings for a batch of accounts in a single HTTP call.
    batch_requests: list of dicts with keys account_id, behavioral, neighbor_ids.
    Returns dict: account_id -> (embedding_list, is_cached).
    Falls back to individual Redis checks if batch endpoint unavailable.
    """
    # First pass: check Redis for all accounts in one pipeline
    pipe = r.pipeline()
    for req in batch_requests:
        pipe.get(f"embed:{req['account_id']}")
    cached = pipe.execute()

    results = {}
    missing = []
    for req, cached_val in zip(batch_requests, cached):
        aid = req['account_id']
        if cached_val:
            results[aid] = (json.loads(cached_val), True)
        else:
            missing.append(req)

    if not missing:
        return results

    # Second pass: call /embed_batch for cache misses
    try:
        import requests as req_lib
        payload = {"accounts": [
            {
                "account_id":     m['account_id'],
                "behavioral":     m['behavioral'],
                "neighbor_ids":   m['neighbor_ids'],
                "amount":         m.get('amount', 0.0),
                "payment_format": m.get('payment_format', 'unknown'),
            }
            for m in missing
        ]}
        resp = req_lib.post(
            "http://localhost:8002/embed_batch",
            json=payload,
            timeout=5.0   # 5s for a batch of 50, vs 1s per individual call
        )
        if resp.status_code == 200:
            batch_results = resp.json()
            for m in missing:
                aid = m['account_id']
                if aid in batch_results:
                    entry = batch_results[aid]
                    is_cached = not entry.get('is_provisional', True)
                    results[aid] = (entry['embedding'], is_cached)
                else:
                    results[aid] = ([0.0] * 64, False)
            return results
    except Exception:
        pass

    # Fallback: zeros for accounts where batch call failed
    for m in missing:
        aid = m['account_id']
        if aid not in results:
            results[aid] = ([0.0] * 64, False)
    return results


# ── Risk tier ─────────────────────────────────────────────────────────────────
def get_risk_tier(prob):
    if prob >= L3_THRESHOLD:   return "L3 — AUTO FREEZE"
    elif prob >= L2_THRESHOLD: return "L2 — SOFT HOLD"
    elif prob >= L1_THRESHOLD: return "L1 — REVIEW"
    else:                      return "CLEAR"


# ── Subscribe to active Kafka topic ──────────────────────────────────────────
active_topic = get_active_topic()
consumer.subscribe([active_topic])
print(f"Subscribed to {active_topic}")

# ── Main loop — batched ───────────────────────────────────────────────────────
# Accumulates BATCH_SIZE transactions, fetches all GNN embeddings in one
# HTTP call, then scores all transactions in-process with XGBoost.
# This eliminates per-transaction HTTP latency (~50ms each) and replaces it
# with one call per batch (~100ms for 50 transactions = ~20x throughput gain).
print("\nListening for transactions (batch mode, size={})...\n".format(BATCH_SIZE))
print(f"{'Time':<10} {'Account':<18} {'Amount':>15} "
      f"{'Prob':>8} {'Tier':<22} {'GNN':<6} {'Alert'}")
print("-" * 90)

pending_events  = []   # raw events waiting to be processed
pending_states  = []   # per-event account state (computed before GNN call)

try:
    while True:
        # ── Drain Kafka into pending buffer ───────────────────────────────────
        msg = consumer.poll(timeout=0.05)

        if msg is not None and not msg.error():
            try:
                event = json.loads(msg.value().decode('utf-8'))
                account_id = str(event['account_from'])
                amount     = float(event['amount_paid'])
                bank_to    = str(event['bank_to'])

                # Update account state immediately (Redis, sequential — fast)
                (state, total_txn, total_sent, max_amt,
                 sum_amt, sum_sq, unique_dest,
                 unique_banks, round_cnt) = update_account_state(
                    account_id, amount, bank_to
                )
                behavioral = compute_behavioral_features(
                    state, total_txn, total_sent, max_amt,
                    sum_amt, sum_sq, unique_dest,
                    unique_banks, round_cnt, amount
                )

                pending_events.append(event)
                pending_states.append({
                    'account_id':   account_id,
                    'amount':       amount,
                    'bank_to':      bank_to,
                    'behavioral':   behavioral,
                    'total_txn':    total_txn,
                    'sum_amt':      sum_amt,
                    'max_amt':      max_amt,
                    'unique_dest':  unique_dest,
                    'unique_banks': unique_banks,
                })
            except Exception as e:
                print(f"Parse error: {e}")

        elif msg is not None and msg.error():
            print(f"Kafka error: {msg.error()}")

        # ── Process batch when full OR when queue is empty and we have items ──
        queue_empty = (msg is None)
        batch_ready = len(pending_events) >= BATCH_SIZE
        flush_now   = queue_empty and pending_events

        if not (batch_ready or flush_now):
            if queue_empty and processed > 0 and processed % 5000 == 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Processed: {processed:,} | Flagged: {flagged}")
            continue

        # ── Batch GNN embedding fetch ─────────────────────────────────────────
        batch_requests = [
            {
                'account_id':     s['account_id'],
                'behavioral':     s['behavioral'],
                'neighbor_ids':   list(s['unique_dest'])[:10],
                'amount':         s['amount'],
                'payment_format': pending_events[i].get('payment_format', 'unknown'),
            }
            for i, s in enumerate(pending_states)
        ]
        embeddings = get_gnn_embeddings_batch(batch_requests)

        # ── Score all transactions in batch ───────────────────────────────────
        feature_matrix = []
        for s in pending_states:
            aid = s['account_id']
            gnn_emb, _ = embeddings.get(aid, ([0.0] * 64, False))
            feature_matrix.append(s['behavioral'] + gnn_emb)

        X        = np.array(feature_matrix)
        X_scaled = scaler.transform(X)
        probs    = xgb_model.predict_proba(X_scaled)[:, 1]

        # ── Emit alerts and CEP checks for each transaction in batch ──────────
        ts = datetime.now().strftime("%H:%M:%S")
        alert_pipe = r.pipeline()

        for i, (event, s) in enumerate(zip(pending_events, pending_states)):
            account_id   = s['account_id']
            amount       = s['amount']
            prob         = float(probs[i])
            is_launder   = int(event['is_laundering'])
            flagged_flag = prob >= optimal_threshold
            tier         = get_risk_tier(prob)
            gnn_emb, has_embed = embeddings.get(account_id, ([0.0] * 64, False))

            is_confirmed_launder = (
                account_id in laundering_accounts or
                is_launder == 1 or
                r.get(f"launder_status:{account_id}") == "1"
            )
            if is_launder == 1 or account_id in laundering_accounts:
                alert_pipe.set(f"launder_status:{account_id}", "1",
                               ex=86400 * 7)

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
                    "total_txn":          s['total_txn'],
                    "fan_out":            len(s['unique_dest']),
                    "cross_bank":         len(s['unique_banks']),
                    "has_gnn_embed":      has_embed,
                    "max_amount":         s['max_amt'],
                    "avg_amount":         (s['sum_amt'] / s['total_txn']
                                           if s['total_txn'] > 0 else amount),
                }
                alert_pipe.lpush("alerts", json.dumps(alert))
                alert_pipe.ltrim("alerts", 0, 999)

            # CEP check (per transaction, not batchable — uses rolling state)
            try:
                from cep_engine import check_patterns
                cep_alerts = check_patterns(r, {
                    'account_from':  account_id,
                    'account_to':    str(event['account_to']),
                    'bank_from':     str(event.get('bank_from', '')),
                    'bank_to':       s['bank_to'],
                    'amount_paid':   amount,
                    'timestamp':     event['timestamp'],
                })
                for cep_alert in cep_alerts:
                    print(f"  >> CEP PATTERN: {cep_alert['pattern_type']} "
                          f"on {cep_alert['account_id']}")
            except Exception:
                pass

            # Publish to Kafka risk-scores topic
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

            processed += 1

            # Print every flagged + every 50th (batch-aligned)
            if flagged_flag or processed % 500 == 0:
                alert_str = "FLAGGED" if flagged_flag         else ""
                laund_str = "LAUNDER" if is_confirmed_launder else ""
                gnn_str   = "GNN+" if has_embed else "GNN-"
                print(f"{ts:<10} {account_id[:16]:<18} {amount:>15,.2f} "
                      f"{prob:>8.4f} {tier:<22} {gnn_str:<6} "
                      f"{alert_str} {laund_str}")

        # Flush Redis pipeline and Kafka producer
        alert_pipe.execute()
        risk_producer.poll(0)

        # Clear batch
        pending_events.clear()
        pending_states.clear()

except KeyboardInterrupt:
    print(f"\nStopped. Processed: {processed:,} | Flagged: {flagged}")
finally:
    consumer.close()