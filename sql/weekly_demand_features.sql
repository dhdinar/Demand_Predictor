
# MySQL-compatible version (no FULL OUTER JOIN, use UNION of LEFT JOINs)

-- Helper: get week start date (Monday)
-- MySQL: YEARWEEK returns week number, but we want the actual date
-- We'll use DATE_SUB to get the Monday of each week

-- Sold




SELECT
    t.product_id,
    t.week,
    t.units_sold,
    t.wishlist_count,
    t.cart_total,
    t.unique_message_users,
    prev.prev_units_sold,
    roll.rolling_3wk_avg_units_sold
FROM (
    SELECT
        base.product_id,
        base.week,
        COALESCE(s.units_sold, 0) AS units_sold,
        COALESCE(w.wishlist_count, 0) AS wishlist_count,
        COALESCE(c.cart_total, 0) AS cart_total,
        COALESCE(m.unique_message_users, 0) AS unique_message_users
    FROM (
        SELECT DISTINCT
            p.id AS product_id,
            DATE(DATE_SUB(o.created_at, INTERVAL (WEEKDAY(o.created_at)) DAY)) AS week
        FROM order_items oi
        JOIN orders o ON o.id = oi.order_id
        JOIN products p ON p.id = oi.product_id
        UNION
        SELECT DISTINCT
            w.product_id,
            DATE(DATE_SUB(w.created_at, INTERVAL (WEEKDAY(w.created_at)) DAY)) AS week
        FROM wishlists w
        UNION
        SELECT DISTINCT
            c.product_id,
            DATE(DATE_SUB(c.created_at, INTERVAL (WEEKDAY(c.created_at)) DAY)) AS week
        FROM cart_items c
        UNION
        SELECT DISTINCT
            m.product_id,
            DATE(DATE_SUB(m.timestamp, INTERVAL (WEEKDAY(m.timestamp)) DAY)) AS week
        FROM messages m
    ) base
    LEFT JOIN (
        SELECT oi.product_id, DATE(DATE_SUB(o.created_at, INTERVAL (WEEKDAY(o.created_at)) DAY)) AS week, SUM(oi.quantity) AS units_sold
        FROM order_items oi
        JOIN orders o ON o.id = oi.order_id
        GROUP BY oi.product_id, week
    ) s ON s.product_id = base.product_id AND s.week = base.week
    LEFT JOIN (
        SELECT w.product_id, DATE(DATE_SUB(w.created_at, INTERVAL (WEEKDAY(w.created_at)) DAY)) AS week, COUNT(*) AS wishlist_count
        FROM wishlists w
        GROUP BY w.product_id, week
    ) w ON w.product_id = base.product_id AND w.week = base.week
    LEFT JOIN (
        SELECT c.product_id, DATE(DATE_SUB(c.created_at, INTERVAL (WEEKDAY(c.created_at)) DAY)) AS week, SUM(c.quantity) AS cart_total
        FROM cart_items c
        GROUP BY c.product_id, week
    ) c ON c.product_id = base.product_id AND c.week = base.week
    LEFT JOIN (
        SELECT m.product_id, DATE(DATE_SUB(m.timestamp, INTERVAL (WEEKDAY(m.timestamp)) DAY)) AS week, COUNT(DISTINCT m.sender_user_id) AS unique_message_users
        FROM messages m
        GROUP BY m.product_id, week
    ) m ON m.product_id = base.product_id AND m.week = base.week
) t
LEFT JOIN (
    SELECT
        base2.product_id,
        base2.week,
        SUM(oi2.quantity) AS prev_units_sold
    FROM (
        SELECT DISTINCT
            p.id AS product_id,
            DATE(DATE_SUB(o.created_at, INTERVAL (WEEKDAY(o.created_at)) DAY)) AS week
        FROM order_items oi
        JOIN orders o ON o.id = oi.order_id
        JOIN products p ON p.id = oi.product_id
        UNION
        SELECT DISTINCT
            w.product_id,
            DATE(DATE_SUB(w.created_at, INTERVAL (WEEKDAY(w.created_at)) DAY)) AS week
        FROM wishlists w
        UNION
        SELECT DISTINCT
            c.product_id,
            DATE(DATE_SUB(c.created_at, INTERVAL (WEEKDAY(c.created_at)) DAY)) AS week
        FROM cart_items c
        UNION
        SELECT DISTINCT
            m.product_id,
            DATE(DATE_SUB(m.timestamp, INTERVAL (WEEKDAY(m.timestamp)) DAY)) AS week
        FROM messages m
    ) base2
    LEFT JOIN order_items oi2 ON oi2.product_id = base2.product_id
    LEFT JOIN orders o2 ON o2.id = oi2.order_id
    WHERE DATE(DATE_SUB(o2.created_at, INTERVAL (WEEKDAY(o2.created_at)) DAY)) = DATE_SUB(base2.week, INTERVAL 7 DAY)
    GROUP BY base2.product_id, base2.week
) prev ON prev.product_id = t.product_id AND prev.week = t.week
LEFT JOIN (
    SELECT
        base3.product_id,
        base3.week,
        AVG(sub.units_sold) AS rolling_3wk_avg_units_sold
    FROM (
        SELECT DISTINCT
            p.id AS product_id,
            DATE(DATE_SUB(o.created_at, INTERVAL (WEEKDAY(o.created_at)) DAY)) AS week
        FROM order_items oi
        JOIN orders o ON o.id = oi.order_id
        JOIN products p ON p.id = oi.product_id
        UNION
        SELECT DISTINCT
            w.product_id,
            DATE(DATE_SUB(w.created_at, INTERVAL (WEEKDAY(w.created_at)) DAY)) AS week
        FROM wishlists w
        UNION
        SELECT DISTINCT
            c.product_id,
            DATE(DATE_SUB(c.created_at, INTERVAL (WEEKDAY(c.created_at)) DAY)) AS week
        FROM cart_items c
        UNION
        SELECT DISTINCT
            m.product_id,
            DATE(DATE_SUB(m.timestamp, INTERVAL (WEEKDAY(m.timestamp)) DAY)) AS week
        FROM messages m
    ) base3
    JOIN (
        SELECT
            oi2.product_id,
            DATE(DATE_SUB(o2.created_at, INTERVAL (WEEKDAY(o2.created_at)) DAY)) AS week,
            SUM(oi2.quantity) AS units_sold
        FROM order_items oi2
        JOIN orders o2 ON o2.id = oi2.order_id
        GROUP BY oi2.product_id, week
    ) sub ON sub.product_id = base3.product_id
        AND sub.week BETWEEN DATE_SUB(base3.week, INTERVAL 14 DAY) AND base3.week
    GROUP BY base3.product_id, base3.week
) roll ON roll.product_id = t.product_id AND roll.week = t.week
-- WHERE prev.prev_units_sold IS NOT NULL
ORDER BY t.product_id, t.week;
