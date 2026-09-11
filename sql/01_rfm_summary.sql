-- Per-customer RFM summary + churn label, computed entirely in SQL.
--
-- This intentionally recomputes what data_prep.py already builds in
-- pandas, in SQL, as a second implementation of the same business logic:
-- cancelled invoices (Invoice starting with 'C') and non-positive
-- quantity/price rows are excluded from monetary/frequency totals; a
-- customer is "churned" if their most recent purchase is more than 90
-- days before the dataset's last recorded transaction.
--
-- Business question: who are our top-20 customers by lifetime spend, and
-- which of them have gone quiet?

WITH snapshot AS (
    SELECT MAX(invoice_date) AS snapshot_date FROM transactions
),
real_purchases AS (
    SELECT *
    FROM transactions
    WHERE is_cancelled = 0 AND quantity > 0 AND price > 0
),
customer_rfm AS (
    SELECT
        customer_id,
        COUNT(DISTINCT invoice) AS frequency,
        ROUND(SUM(quantity * price), 2) AS monetary,
        MIN(invoice_date) AS first_purchase,
        MAX(invoice_date) AS last_purchase,
        COUNT(DISTINCT stock_code) AS distinct_products
    FROM real_purchases
    GROUP BY customer_id
)
SELECT
    r.customer_id,
    r.frequency,
    r.monetary,
    ROUND(r.monetary / r.frequency, 2) AS avg_order_value,
    r.distinct_products,
    CAST(julianday(s.snapshot_date) - julianday(r.last_purchase) AS INTEGER) AS recency_days,
    CAST(julianday(s.snapshot_date) - julianday(r.first_purchase) AS INTEGER) AS tenure_days,
    CASE
        WHEN CAST(julianday(s.snapshot_date) - julianday(r.last_purchase) AS INTEGER) > 90
        THEN 1 ELSE 0
    END AS churned
FROM customer_rfm r
CROSS JOIN snapshot s
ORDER BY r.monetary DESC
LIMIT 20;
