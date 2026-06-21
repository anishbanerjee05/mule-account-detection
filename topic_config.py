"""
topic_config.py — Single source of truth for which Kafka topic is
currently active.

Why this exists: manually keeping producer.py and consumer.py in sync
on a topic name, and manually trusting `kafka-topics --delete` to have
actually worked, was the root cause of an entire debugging session —
old message backlogs silently surviving a "reset" and being mistaken
for new bugs.

reset_demo_environment.ps1 always creates a uniquely-named, freshly
verified-empty topic and writes its name to active_topic.txt. This
module just reads that file. If the file doesn't exist yet (e.g. very
first run), it falls back to the original 'transactions-raw' name so
nothing breaks for anyone who hasn't run the reset script yet.

Usage, in both producer.py and consumer.py:
    from topic_config import get_active_topic
    TOPIC = get_active_topic()
"""

import os

DEFAULT_TOPIC = "transactions-raw"
TOPIC_FILE = "active_topic.txt"


def get_active_topic() -> str:
    if os.path.exists(TOPIC_FILE):
        with open(TOPIC_FILE, "r") as f:
            topic = f.read().strip()
            if topic:
                return topic
    return DEFAULT_TOPIC