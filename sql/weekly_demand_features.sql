-- Weekly demand feature dataset for product-level forecasting
-- PostgreSQL dialect
WITH sold AS (
    SELECT
        oi.product_id,
        DATE_TRUNC('week', o.created_at) AS week,
        SUM(oi.quantity) AS units_sold
    FROM order_items oi
    JOIN orders o ON o.order_id = oi.order_id
    GROUP BY oi.product_id, DATE_TRUNC('week', o.created_at)
),
wish AS (
    SELECT
        w.product_id,
        DATE_TRUNC('week', w.created_at) AS week,
        COUNT(*) AS wishlist_count
    FROM wishlist w
    GROUP BY w.product_id, DATE_TRUNC('week', w.created_at)
),
cart AS (
    SELECT
        c.product_id,
        DATE_TRUNC('week', c.created_at) AS week,
        SUM(c.quantity) AS cart_total
    FROM cart_items c
    GROUP BY c.product_id, DATE_TRUNC('week', c.created_at)
),
msg AS (
    SELECT
        m.product_id,
        DATE_TRUNC('week', m.timestamp) AS week,
        COUNT(DISTINCT m.sender_user) AS unique_message_users
    FROM messages m
    GROUP BY m.product_id, DATE_TRUNC('week', m.timestamp)
),
merged AS (
    SELECT
        COALESCE(s.product_id, w.product_id, c.product_id, m.product_id) AS product_id,
        COALESCE(s.week, w.week, c.week, m.week) AS week,
        COALESCE(s.units_sold, 0) AS units_sold,
        COALESCE(w.wishlist_count, 0) AS wishlist_count,
        COALESCE(c.cart_total, 0) AS cart_total,
        COALESCE(m.unique_message_users, 0) AS unique_message_users
    FROM sold s
    FULL OUTER JOIN wish w
        ON s.product_id = w.product_id AND s.week = w.week
    FULL OUTER JOIN cart c
        ON COALESCE(s.product_id, w.product_id) = c.product_id
       AND COALESCE(s.week, w.week) = c.week
    FULL OUTER JOIN msg m
        ON COALESCE(s.product_id, w.product_id, c.product_id) = m.product_id
       AND COALESCE(s.week, w.week, c.week) = m.week
),
final_features AS (
    SELECT
        product_id,
        week,
        units_sold,
        wishlist_count,
        cart_total,
        unique_message_users,
        LAG(units_sold) OVER (
            PARTITION BY product_id
            ORDER BY week
        ) AS prev_units_sold,
        AVG(units_sold) OVER (
            PARTITION BY product_id
            ORDER BY week
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS rolling_3wk_avg_units_sold
    FROM merged
)

SELECT
    product_id,
    week,
    units_sold,
    wishlist_count,
    cart_total,
    unique_message_users,
    prev_units_sold,
    rolling_3wk_avg_units_sold
FROM final_features
WHERE prev_units_sold IS NOT NULL
ORDER BY product_id, week;
