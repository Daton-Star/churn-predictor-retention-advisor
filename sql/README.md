# SQL analysis

The same churn/RFM logic from `src/data_prep.py`, reimplemented in SQL
(CTEs, window functions, `CASE` segmentation) against a SQLite copy of the
cleaned transaction data — for contexts where SQL, not pandas, is the
expected tool.

**Cross-check:** query `03_revenue_at_risk.sql` finds 2,893 active + 2,985
churned = 5,878 customers at a 50.8% churn rate — this matches the pandas
pipeline's output in `reports/model_comparison.json` exactly, which is a
useful sanity check that both implementations agree.

## Files

| File | Business question |
|---|---|
| `01_rfm_summary.sql` | Top 20 customers by lifetime spend, and which have gone quiet |
| `02_churn_rate_by_country.sql` | Which markets (≥20 customers) have the highest churn rate |
| `03_revenue_at_risk.sql` | How much historical revenue sits with churned vs. active customers |
| `04_recency_risk_tiers.sql` | Customer counts/value by a simple 4-tier recency risk segmentation |

Real output from running all four against the actual dataset is in
[`QUERY_RESULTS.md`](QUERY_RESULTS.md) — no database client needed to see
the results.

## Regenerate

```bash
python sql/build_db.py     # -> sql/online_retail.db (not committed; ~50MB, rebuilt from data_raw/)
python sql/run_queries.py  # -> sql/QUERY_RESULTS.md
```

Or open `sql/online_retail.db` directly in any SQLite client (DB Browser
for SQLite, the VS Code SQLite extension, `sqlite3` CLI) and run the
`.sql` files interactively.
