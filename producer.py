import pandas as pd
import json
import time
from confluent_kafka import Producer
from datetime import datetime, timedelta
from topic_config import get_active_topic

print("Starting IBM AML transaction producer (confluent-kafka)...")

# Reads active_topic.txt (written by reset_demo_environment.ps1) so this
# always targets a freshly-verified-empty topic rather than a hardcoded
# name that might still hold a stale backlog from a previous session.
TOPIC = get_active_topic()
print(f"Target topic: {TOPIC}")

# -- Demo time compression settings -----------------------------------------
# The IBM AML dataset's real timestamps span ~30 days. Replaying that at
# real speed is obviously not viable for a live demo, but stamping every
# event with datetime.now() (the old behaviour) destroys ALL time-gap
# information, which silently breaks any CEP pattern that depends on
# "quiet for a while, then a burst" (e.g. BURST_ACTIVATION).
#
# Fix: preserve every account's REAL relative time gaps from the source
# data, just compressed by a fixed ratio so the whole dataset's time span
# plays out over DEMO_DURATION_MINUTES instead of 30 real days. A 30-day
# gap and a 2-hour gap both shrink by the same factor, so their *shape*
# relative to each other survives -- the detection logic in cep_engine.py
# does not need to know or care that compression happened at all.
DEMO_DURATION_MINUTES = 12  # total wall-clock runtime for a full replay

print("Loading IBM AML dataset...")
df = pd.read_csv('ibm_aml_small.csv')
df_focus = df[df['Payment Format'].isin(
    ['ACH', 'Cheque', 'Credit Card', 'Cash', 'Bitcoin']
)].copy()

print(f"Loaded {len(df_focus):,} transactions")
print(f"Laundering transactions: {df_focus['Is Laundering'].sum():,}")

# -- Parse real timestamps and sort chronologically -------------------------
print("Parsing real timestamps and computing compression ratio...")
df_focus['_real_ts'] = pd.to_datetime(df_focus['Timestamp'], format='%Y/%m/%d %H:%M')
df_focus = df_focus.sort_values('_real_ts').reset_index(drop=True)

real_start = df_focus['_real_ts'].iloc[0]
real_end   = df_focus['_real_ts'].iloc[-1]
real_span_seconds = (real_end - real_start).total_seconds()

demo_span_seconds = DEMO_DURATION_MINUTES * 60
compression_ratio = demo_span_seconds / real_span_seconds if real_span_seconds > 0 else 1.0

print(f"Real data span:    {real_start} -> {real_end}  "
      f"({real_span_seconds / 86400:.1f} days)")
print(f"Demo replay span:  {DEMO_DURATION_MINUTES} minutes")
print(f"Compression ratio: 1 real second -> {compression_ratio:.6f} demo seconds "
      f"(~{1/compression_ratio:.0f}x speedup)")

# -- Precompute each event's demo-clock timestamp ----------------------------
# demo_offset = how far into the (compressed) demo this event's timestamp
# falls, anchored to wall-clock "now" at producer startup. This preserves
# relative gaps between ANY two events (same account or not) exactly,
# scaled by compression_ratio.
demo_start = datetime.now()
df_focus['_demo_offset_seconds'] = (
    (df_focus['_real_ts'] - real_start).dt.total_seconds() * compression_ratio
)

producer = Producer({'bootstrap.servers': 'localhost:9092'})

def delivery_report(err, msg):
    pass

print("Kafka producer connected")
print(f"Streaming transactions over ~{DEMO_DURATION_MINUTES} minutes "
      f"(Ctrl+C to stop)\n")

sent = 0
flagged = 0
last_demo_offset = 0.0

for _, row in df_focus.iterrows():
    # Compute this event's demo-clock timestamp (real gaps, compressed).
    demo_timestamp = demo_start + timedelta(seconds=row['_demo_offset_seconds'])

    event = {
        "event_id":          f"EVT_{sent:08d}",
        "timestamp":         demo_timestamp.isoformat(),
        "account_from":      str(row['Account']),
        "account_to":        str(row['Account.1']),
        "bank_from":         str(row['From Bank']),
        "bank_to":           str(row['To Bank']),
        "amount_paid":       float(row['Amount Paid']),
        "amount_received":   float(row['Amount Received']),
        "currency_paid":     str(row['Payment Currency']),
        "currency_received": str(row['Receiving Currency']),
        "payment_format":    str(row['Payment Format']),
        "is_laundering":     int(row['Is Laundering'])
    }

    producer.produce(
        TOPIC,
        key=event['account_from'],
        value=json.dumps(event).encode('utf-8'),
        callback=delivery_report
    )
    producer.poll(0)

    sent += 1
    if event['is_laundering'] == 1:
        flagged += 1

    if sent % 1000 == 0:
        producer.flush()
        print(f"Sent: {sent:,} | Laundering: {flagged} | "
              f"Rate: {flagged/sent*100:.3f}% | "
              f"Demo-elapsed: {row['_demo_offset_seconds']:.1f}s")

    # Sleep just enough to pace delivery against the compressed demo clock,
    # rather than a fixed 0.002s -- this keeps wall-clock pacing roughly
    # aligned with _demo_offset_seconds without blocking on huge gaps.
    offset_delta = row['_demo_offset_seconds'] - last_demo_offset
    sleep_for = min(max(offset_delta, 0.0), 0.05)  # cap per-event sleep
    if sleep_for > 0:
        time.sleep(sleep_for)
    last_demo_offset = row['_demo_offset_seconds']

producer.flush()

# Send an explicit end-of-stream marker as the final message. consumer.py
# watches for this and exits cleanly on its own, instead of needing to
# be manually Ctrl+C'd every time a run finishes -- and instead of being
# left running for hours by accident (which is what caused the stale
# Kafka backlog problem earlier).
end_marker = {"event_type": "END_OF_STREAM", "total_sent": sent}
producer.produce(
    TOPIC,
    key="__END_OF_STREAM__",
    value=json.dumps(end_marker).encode('utf-8'),
    callback=delivery_report
)
producer.flush()

print(f"\nComplete. Total: {sent:,}, Laundering: {flagged}")
print("End-of-stream marker sent -- consumer will stop automatically.")