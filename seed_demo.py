import sqlite3
import json
import os
from datetime import datetime, timedelta

DB_PATH = "./data/demo.db"
os.makedirs("data", exist_ok=True)

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.executescript("""
CREATE TABLE IF NOT EXISTS eval_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_version TEXT NOT NULL,
    model TEXT NOT NULL,
    run_at TEXT NOT NULL,
    accuracy REAL,
    precision REAL,
    recall REAL,
    f1 REAL,
    total_cases INTEGER,
    passed INTEGER,
    failed INTEGER,
    avg_latency_ms REAL
);

CREATE TABLE IF NOT EXISTS eval_cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    case_id TEXT,
    expected_label TEXT,
    predicted_label TEXT,
    confidence REAL,
    latency_ms REAL,
    passed INTEGER,
    FOREIGN KEY (run_id) REFERENCES eval_runs(id)
);
""")

runs = [
    ("v1.0.0", "gpt-4o-mini", datetime.now() - timedelta(days=6), 0.817, 0.801, 0.796, 0.798, 60, 49, 11, 1240),
    ("v1.0.0", "gpt-4o-mini", datetime.now() - timedelta(days=5), 0.833, 0.821, 0.809, 0.815, 60, 50, 10, 1190),
    ("v1.1.0", "gpt-4o-mini", datetime.now() - timedelta(days=4), 0.867, 0.854, 0.841, 0.847, 60, 52, 8,  1150),
    ("v1.1.0", "gpt-4o-mini", datetime.now() - timedelta(days=3), 0.883, 0.871, 0.862, 0.866, 60, 53, 7,  1120),
    ("v1.1.0", "gpt-4o-mini", datetime.now() - timedelta(days=2), 0.800, 0.788, 0.779, 0.783, 60, 48, 12, 1380),
    ("v1.1.0", "gpt-4o-mini", datetime.now() - timedelta(days=1), 0.917, 0.908, 0.895, 0.901, 60, 55, 5,  1090),
]

labels = ["SPAM", "HAM", "PHISHING", "NEWSLETTER"]

for run in runs:
    cur.execute("""
        INSERT INTO eval_runs 
        (prompt_version, model, run_at, accuracy, precision, recall, f1, total_cases, passed, failed, avg_latency_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (run[0], run[1], run[2].isoformat(), run[3], run[4], run[5], run[6], run[7], run[8], run[9], run[10]))
    
    run_id = cur.lastrowid
    
    for i in range(1, 61):
        import random
        expected = labels[i % 4]
        passed = random.random() < run[3]
        predicted = expected if passed else random.choice([l for l in labels if l != expected])
        cur.execute("""
            INSERT INTO eval_cases
            (run_id, case_id, expected_label, predicted_label, confidence, latency_ms, passed)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (run_id, f"case_{i:03d}", expected, predicted, round(random.uniform(0.65, 0.99), 2), round(random.uniform(900, 1600), 1), int(passed)))

conn.commit()
conn.close()
print(f"demo.db seeded with {len(runs)} eval runs and {len(runs)*60} cases.")