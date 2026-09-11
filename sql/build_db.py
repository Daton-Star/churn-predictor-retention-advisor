"""Materialize cleaned transactions into a local SQLite database so the
.sql files in this folder can be run directly (sqlite3, DBeaver, VS Code's
SQLite extension, etc.) instead of only through pandas.

WHY this exists alongside the pandas pipeline in src/
--------------------------------------------------------
data_prep.py already builds the churn label and RFM features in pandas.
This script deliberately does NOT reimplement that logic in Python -- it
reuses clean_transactions() from src/data_prep.py so both paths clean the
data identically, then hands the result to SQLite. The four .sql files in
this folder recompute recency/frequency/monetary and the churn label a
second time, but *in SQL*, as a deliberate demonstration: the same business
questions answered with CTEs and window functions instead of pandas
groupby, for contexts (most analyst tooling, most interviews) where SQL is
the expected language.

Run: `python sql/build_db.py` (from the repo root) -> sql/online_retail.db
"""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import DATA_RAW_PATH  # noqa: E402
from data_prep import clean_transactions, load_raw_transactions  # noqa: E402

DB_PATH = Path(__file__).resolve().parent / "online_retail.db"


def main():
    print(f"Loading and cleaning transactions from {DATA_RAW_PATH} ...")
    raw = load_raw_transactions()
    cleaned = clean_transactions(raw)

    # SQLite has no boolean type; store is_cancelled as 0/1 explicitly
    # rather than relying on pandas' implicit True/False -> 1/0 cast, so
    # the .sql files can compare it in a CASE without surprises.
    cleaned = cleaned.copy()
    cleaned["is_cancelled"] = cleaned["is_cancelled"].astype(int)

    print(f"Writing {len(cleaned):,} cleaned rows to {DB_PATH} ...")
    conn = sqlite3.connect(DB_PATH)
    cleaned.to_sql("transactions", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_customer ON transactions(customer_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_invoice ON transactions(invoice)")
    conn.commit()
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
