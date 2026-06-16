import pandas as pd
import json
import time
from confluent_kafka import Producer
from datetime import datetime

print("Starting IBM AML transaction producer (confluent-kafka)...")

df = pd.read_csv('ibm_aml_small.csv')
df_focus = df[df['Payment Format'].isin(
    ['ACH', 'Cheque', 'Credit Card', 'Cash', 'Bitcoin']
)].copy()

print(f"Loaded {len(df_focus):,} transactions")
print(f"Laundering transactions: {df_focus['Is Laundering'].sum():,}")

producer = Producer({'bootstrap.servers': 'localhost:9092'})

def delivery_report(err, msg):
    pass

print("Kafka producer connected")
print("Streaming transactions... (Ctrl+C to stop)\n")

sent = 0
flagged = 0

for _, row in df_focus.iterrows():
    event = {
        "event_id":          f"EVT_{sent:08d}",
        "timestamp":         datetime.now().isoformat(),
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
        'transactions-raw',
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
        print(f"Sent: {sent:,} | Laundering: {flagged} | Rate: {flagged/sent*100:.3f}%")

    time.sleep(0.002)

producer.flush()
print(f"\nComplete. Total: {sent:,}, Laundering: {flagged}")