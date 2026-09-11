-- Business question: a simple "traffic light" segmentation retention
-- teams can act on directly -- how many customers (and how much revenue)
-- sit in each recency-based risk tier?

WITH snapshot AS (
    SELECT MAX(invoice_date) AS snapshot_date FROM transactions
),
real_purchases AS (
    SELECT * FROM transactions WHERE is_cancelled = 0 AND quantity > 0 AND price > 0
),
customer_recency AS (
    SELECT
        p.customer_id,
        SUM(p.quantity * p.price) AS monetary,
        CAST(julianday(s.snapshot_date) - julianday(MAX(p.invoice_date)) AS INTEGER) AS recency_days
    FROM real_purchases p
    CROSS JOIN snapshot s
    GROUP BY p.customer_id
),
tiered AS (
    SELECT
        *,
        CASE
            WHEN recency_days <= 30 THEN '1. Active (<=30d)'
            WHEN recency_days <= 90 THEN '2. Cooling (31-90d)'
            WHEN recency_days <= 180 THEN '3. At risk (91-180d)'
            ELSE '4. Lost (>180d)'
        END AS risk_tier
    FROM customer_recency
)
SELECT
    risk_tier,
    COUNT(*) AS num_customers,
    ROUND(AVG(monetary), 0) AS avg_customer_value,
    ROUND(SUM(monetary), 0) AS total_revenue
FROM tiered
GROUP BY risk_tier
ORDER BY risk_tier;
