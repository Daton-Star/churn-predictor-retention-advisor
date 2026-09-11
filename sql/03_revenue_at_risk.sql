-- Business question: how much historical revenue sits with customers who
-- are currently churned vs. still active? Translates a customer count
-- into a dollar figure a retention budget can be sized against.

WITH snapshot AS (
    SELECT MAX(invoice_date) AS snapshot_date FROM transactions
),
real_purchases AS (
    SELECT * FROM transactions WHERE is_cancelled = 0 AND quantity > 0 AND price > 0
),
customer_status AS (
    SELECT
        p.customer_id,
        SUM(p.quantity * p.price) AS monetary,
        CASE
            WHEN CAST(julianday(s.snapshot_date) - julianday(MAX(p.invoice_date)) AS INTEGER) > 90
            THEN 'Churned' ELSE 'Active'
        END AS status
    FROM real_purchases p
    CROSS JOIN snapshot s
    GROUP BY p.customer_id
),
status_totals AS (
    SELECT
        status,
        COUNT(*) AS num_customers,
        SUM(monetary) AS total_revenue
    FROM customer_status
    GROUP BY status
)
SELECT
    status,
    num_customers,
    ROUND(total_revenue, 0) AS total_revenue,
    ROUND(100.0 * total_revenue / SUM(total_revenue) OVER (), 1) AS pct_of_total_revenue
FROM status_totals;
