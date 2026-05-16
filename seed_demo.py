import sqlite3
import json
import os
import random
from datetime import datetime, timedelta

DB_PATH = "./data/demo.db"
os.makedirs("data", exist_ok=True)

# Delete existing DB to start fresh
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.executescript("""
CREATE TABLE IF NOT EXISTS eval_runs (
    run_id              TEXT PRIMARY KEY,
    prompt_version      TEXT NOT NULL,
    model               TEXT NOT NULL,
    timestamp           TEXT NOT NULL,
    total_cases         INTEGER,
    passed              INTEGER,
    failed              INTEGER,
    pass_rate           REAL,
    avg_latency_ms      REAL,
    total_tokens        INTEGER,
    per_category_accuracy TEXT,
    case_scores_json    TEXT
);
""")

runs = [
    ("v1.0.0", "gpt-4o-mini", datetime.now() - timedelta(days=6), 0.817, 49, 11, 1240),
    ("v1.0.0", "gpt-4o-mini", datetime.now() - timedelta(days=5), 0.833, 50, 10, 1190),
    ("v1.1.0", "gpt-4o-mini", datetime.now() - timedelta(days=4), 0.867, 52, 8,  1150),
    ("v1.1.0", "gpt-4o-mini", datetime.now() - timedelta(days=3), 0.883, 53, 7,  1120),
    ("v1.1.0", "gpt-4o-mini", datetime.now() - timedelta(days=2), 0.800, 48, 12, 1380),
    ("v1.1.0", "gpt-4o-mini", datetime.now() - timedelta(days=1), 0.917, 55, 5,  1090),
]

per_cat = json.dumps({
    "SPAM": 0.91, "HAM": 0.93, "PHISHING": 0.88, "NEWSLETTER": 0.90
})

for i, run in enumerate(runs):
    cur.execute("""
        INSERT INTO eval_runs 
        (run_id, prompt_version, model, timestamp, total_cases, passed, failed,
         pass_rate, avg_latency_ms, total_tokens, per_category_accuracy, case_scores_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        f"run_{i+1:03d}",
        run[0], run[1],
        run[2].isoformat(),
        60, run[4], run[5],
        run[3], run[6],
        random.randint(8000, 12000),
        per_cat,
        "[]"
    ))

conn.commit()
conn.close()
print(f"demo.db seeded with {len(runs)} eval runs.")