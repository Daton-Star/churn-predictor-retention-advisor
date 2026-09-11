-- Business question: which markets have the highest churn rate, among
-- countries with a large enough customer base (>=20) for the rate to be
-- meaningful? This is the kind of table that decides where a retention
-- team focuses a limited campaign budget first.

WITH snapshot AS (
    SELECT MAX(invoice_date) AS snapshot_date FROM transactions
),
real_purchases AS (
    SELECT * FROM transactions WHERE is_cancelled = 0 AND quantity > 0 AND price > 0
),
customer_country AS (
    -- A customer can appear under slightly different recorded country
    -- values across orders (address changes, data entry). Their most
    -- recent order's country is used as the single label for that
    -- customer, via ROW_NUMBER() rather than an arbitrary MIN/MAX.
    SELECT customer_id, country
    FROM (
        SELECT
            customer_id,
            country,
            ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY invoice_date DESC) AS rn
        FROM real_purchases
    )
    WHERE rn = 1
),
customer_status AS (
    SELECT
        p.customer_id,
        SUM(p.quantity * p.price) AS monetary,
        CASE
            WHEN CAST(julianday(s.snapshot_date) - julianday(MAX(p.invoice_date)) AS INTEGER) > 90
            THEN 1 ELSE 0
        END AS churned
    FROM real_purchases p
    CROSS JOIN snapshot s
    GROUP BY p.customer_id
)
SELECT
    cc.country,
    COUNT(*) AS num_customers,
    ROUND(100.0 * SUM(cs.churned) / COUNT(*), 1) AS churn_rate_pct,
    ROUND(SUM(cs.monetary), 0) AS total_revenue
FROM customer_status cs
JOIN customer_country cc ON cc.customer_id = cs.customer_id
GROUP BY cc.country
HAVING COUNT(*) >= 20
ORDER BY churn_rate_pct DESC;
