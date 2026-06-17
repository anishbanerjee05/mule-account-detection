from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import JSONResponse
from schema_adapter import SchemaAdapter
from confluent_kafka import Producer
from datetime import datetime
import json
import uvicorn
import redis
import hashlib
import hmac
import os

app = FastAPI(
    title="Mule Detection — Bank Webhook Receiver",
    description="Receives transaction events from bank CBS and publishes to Kafka",
    version="1.0.0"
)

# ── Load schema adapters ──────────────────────────────────────────────────────
print("Loading schema adapters...")
adapters = {
    'sbi':   SchemaAdapter('schema_configs/sbi_finacle.yaml'),
    'kotak': SchemaAdapter('schema_configs/kotak_cbs.yaml'),
}
print(f"Adapters loaded: {list(adapters.keys())}")

# ── Kafka producer ────────────────────────────────────────────────────────────
producer = Producer({'bootstrap.servers': 'localhost:9092'})
print("Kafka producer connected")

# ── Redis for deduplication ───────────────────────────────────────────────────
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# ── Stats tracking ────────────────────────────────────────────────────────────
stats = {
    'total_received':   0,
    'total_published':  0,
    'total_rejected':   0,
    'total_duplicates': 0,
    'by_bank':          {}
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def delivery_report(err, msg):
    if err:
        print(f"Kafka delivery failed: {err}")

def is_duplicate(transaction_id: str, bank_code: str) -> bool:
    """Check if this transaction has already been processed."""
    key = f"txn_seen:{bank_code}:{transaction_id}"
    if r.exists(key):
        return True
    r.setex(key, 3600, "1")  # Remember for 1 hour
    return False

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/webhook/{bank_code}/transaction")
async def receive_transaction(bank_code: str, request: Request):
    """
    Receive a transaction event from a bank's CBS.
    Translates via schema adapter and publishes to Kafka.

    Usage:
        POST /webhook/sbi/transaction
        POST /webhook/kotak/transaction

    Body: Bank-specific transaction JSON
    """
    stats['total_received'] += 1

    # Validate bank code
    if bank_code not in adapters:
        stats['total_rejected'] += 1
        raise HTTPException(
            status_code=400,
            detail=f"Unknown bank code: '{bank_code}'. Supported: {list(adapters.keys())}"
        )

    # Parse request body
    try:
        raw_event = await request.json()
    except Exception:
        stats['total_rejected'] += 1
        raise HTTPException(status_code=422, detail="Invalid JSON body")

    # Translate via schema adapter
    adapter  = adapters[bank_code]
    standard = adapter.translate(raw_event)

    # Validate required fields
    valid, missing = adapter.validate(standard)
    if not valid:
        stats['total_rejected'] += 1
        raise HTTPException(
            status_code=422,
            detail=f"Missing required fields after translation: {missing}"
        )

    # Deduplication check
    txn_id = standard.get('transaction_id', '')
    if txn_id and is_duplicate(txn_id, bank_code):
        stats['total_duplicates'] += 1
        return JSONResponse(
            status_code=200,
            content={
                "status":    "duplicate",
                "message":   "Transaction already processed",
                "txn_id":    txn_id,
                "bank":      bank_code,
            }
        )

    # Add event metadata
    standard['_received_at'] = datetime.now().isoformat()
    standard['_source']      = f"webhook:{bank_code}"

    # Publish to Kafka
    try:
        producer.produce(
            'transactions-raw',
            key=str(standard.get('account_from', '')),
            value=json.dumps(standard).encode('utf-8'),
            callback=delivery_report
        )
        producer.poll(0)
        stats['total_published'] += 1
        stats['by_bank'][bank_code] = stats['by_bank'].get(bank_code, 0) + 1

    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Kafka publish failed: {str(e)}")

    return JSONResponse(
        status_code=200,
        content={
            "status":      "accepted",
            "bank":        bank_code,
            "account":     standard.get('account_from'),
            "amount":      standard.get('amount_paid'),
            "currency":    standard.get('currency'),
            "timestamp":   standard.get('timestamp'),
            "txn_id":      txn_id,
            "adapted":     True,
        }
    )

@app.post("/webhook/{bank_code}/batch")
async def receive_batch(bank_code: str, request: Request):
    """
    Receive a batch of transaction events.
    Useful for end-of-day reconciliation or bulk testing.
    """
    if bank_code not in adapters:
        raise HTTPException(status_code=400, detail=f"Unknown bank: {bank_code}")

    try:
        raw_events = await request.json()
        if not isinstance(raw_events, list):
            raise HTTPException(status_code=422, detail="Body must be a JSON array")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid JSON: {str(e)}")

    adapter = adapters[bank_code]
    translated, errors = adapter.translate_batch(raw_events)

    published = 0
    for standard in translated:
        try:
            producer.produce(
                'transactions-raw',
                key=str(standard.get('account_from', '')),
                value=json.dumps(standard).encode('utf-8'),
                callback=delivery_report
            )
            published += 1
        except Exception:
            errors += 1

    producer.flush()
    stats['total_published'] += published
    stats['total_rejected']  += errors

    return JSONResponse(
        status_code=200,
        content={
            "status":    "batch_accepted",
            "bank":      bank_code,
            "received":  len(raw_events),
            "published": published,
            "errors":    errors,
        }
    )

@app.get("/webhook/health")
def health():
    """Health check endpoint."""
    try:
        r.ping()
        redis_status = "connected"
    except Exception:
        redis_status = "disconnected"

    return {
        "status":       "operational",
        "timestamp":    datetime.now().isoformat(),
        "banks":        list(adapters.keys()),
        "kafka":        "connected",
        "redis":        redis_status,
        "stats":        stats,
        "endpoints": {
            "single":  "POST /webhook/{bank_code}/transaction",
            "batch":   "POST /webhook/{bank_code}/batch",
            "health":  "GET  /webhook/health",
        }
    }

@app.get("/webhook/stats")
def get_stats():
    """Detailed statistics on received transactions."""
    return {
        "total_received":   stats['total_received'],
        "total_published":  stats['total_published'],
        "total_rejected":   stats['total_rejected'],
        "total_duplicates": stats['total_duplicates'],
        "success_rate":     (
            f"{stats['total_published']/stats['total_received']*100:.1f}%"
            if stats['total_received'] > 0 else "N/A"
        ),
        "by_bank": stats['by_bank'],
    }

@app.get("/webhook/test/{bank_code}")
async def test_bank(bank_code: str):
    """
    Test endpoint — sends a sample transaction for the specified bank.
    Use this to verify the webhook is working correctly.
    """
    if bank_code not in adapters:
        raise HTTPException(status_code=400, detail=f"Unknown bank: {bank_code}")

    sample_events = {
        'sbi': {
            "DEBIT_ACCT_NO":  "SBI_TEST_001",
            "CREDIT_ACCT_NO": "SBI_TEST_002",
            "TXN_AMOUNT":     "250000.00",
            "TXN_CCY":        "INR",
            "TXN_TYPE":       "NEFT",
            "TXN_DATE":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "UTR_NO":         f"SBITEST{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "BRANCH_CODE":    "SBI_TEST",
            "DEST_BANK_CODE": "HDFC_TEST",
            "IS_FRAUD":       "0"
        },
        'kotak': {
            "FROM_ACCOUNT_ID": "KTK_TEST_001",
            "TO_ACCOUNT_ID":   "KTK_TEST_002",
            "TRANSACTION_AMT": "175000.50",
            "CURRENCY_CODE":   "INR",
            "PAYMENT_MODE":    "IMPS",
            "CREATED_AT":      datetime.now().isoformat() + "Z",
            "TXN_REF_ID":      f"KTKTEST{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "SOURCE_IFSC":     "KKBK_TEST",
            "DEST_IFSC":       "HDFC_TEST",
            "FRAUD_FLAG":      "0"
        }
    }

    sample = sample_events[bank_code]
    adapter  = adapters[bank_code]
    standard = adapter.translate(sample)

    producer.produce(
        'transactions-raw',
        key=str(standard.get('account_from', '')),
        value=json.dumps(standard).encode('utf-8')
    )
    producer.flush()

    return {
        "status":       "test_published",
        "bank":         bank_code,
        "raw_event":    sample,
        "translated":   standard,
        "kafka_topic":  "transactions-raw"
    }

if __name__ == "__main__":
    print("\nStarting webhook receiver on port 8001...")
    print("Endpoints:")
    print("  POST http://localhost:8001/webhook/sbi/transaction")
    print("  POST http://localhost:8001/webhook/kotak/transaction")
    print("  POST http://localhost:8001/webhook/{bank}/batch")
    print("  GET  http://localhost:8001/webhook/health")
    print("  GET  http://localhost:8001/webhook/test/{bank}")
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)