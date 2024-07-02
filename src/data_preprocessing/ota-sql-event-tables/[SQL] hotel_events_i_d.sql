-- Databricks notebook source
INSERT INTO TABLE hotel_events_i_d
WITH OrderedEvents AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY uid ORDER BY event_timestamp) AS row_num
    FROM base_ota_events_i_d
    WHERE   LOWER(content_group) IN ('ota', 'ota hotel') AND
            event_timestamp > (SELECT MAX(max_event_timestamp) FROM hotel_events_i_d)
    ORDER BY uid, event_timestamp
),
GroupedEvents AS (
    SELECT *,
           CASE 
               WHEN LAG(event_name) OVER (PARTITION BY uid ORDER BY event_timestamp) != event_name 
                        OR 
                    LAG(uid) OVER (ORDER BY uid, event_timestamp) != uid 
                THEN 1
                ELSE 0
           END AS group_change
    FROM OrderedEvents
),
LabeledEvents AS (
    SELECT *,
           SUM(group_change) OVER (ORDER BY uid, event_timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS group_label
    FROM GroupedEvents
),
AggregatedEvents AS (
    SELECT uid, user_id, ga_session_id, group_label, first_touch_source, event_name, hotel_id, booking_id, hotel_destination_id, banner_id, carousel_id, recommendlocation_id,
           MIN(event_timestamp) AS min_event_timestamp,
           MAX(event_timestamp) AS max_event_timestamp
    FROM LabeledEvents
    GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
),
PrepEvents AS (
    SELECT  user_id, ga_session_id, event_name, 
            DATE_DIFF(SECOND, min_event_timestamp, max_event_timestamp) AS duration,
            hotel_id, booking_id, hotel_destination_id, banner_id, carousel_id, recommendlocation_id,
            uid,
            min_event_timestamp, max_event_timestamp,
            LEAD(max_event_timestamp) OVER (PARTITION BY uid ORDER BY max_event_timestamp) AS next_event_timestamp
    FROM AggregatedEvents
)
SELECT  uid, user_id, ga_session_id, event_name, duration, 
        CASE WHEN DATE_DIFF(SECOND, max_event_timestamp, next_event_timestamp) IS NULL THEN 0
                ELSE DATE_DIFF(SECOND, max_event_timestamp, next_event_timestamp)
            END AS shift_duration,
        duration + IFNULL(DATE_DIFF(SECOND, max_event_timestamp, next_event_timestamp), 0) AS total_duration,
        hotel_id, booking_id, hotel_destination_id, banner_id, carousel_id, recommendlocation_id,
        min_event_timestamp, max_event_timestamp
FROM PrepEvents