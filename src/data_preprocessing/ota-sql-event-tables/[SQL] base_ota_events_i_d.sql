-- Databricks notebook source
INSERT INTO TABLE base_ota_events_i_d 
WITH basedEvent AS (
    SELECT  event_date, event_timestamp, event_name, user_id, ga_session_id,
            CONCAT(user_id, ga_session_id) AS uid,
            prop.key, COALESCE(params.value.string_value, params.value.int_value, params.value.float_value, params.value.double_value) as values
    FROM   events_i_d LATERAL VIEW EXPLODE(event_params) exploded_user_prop AS prop
    WHERE
            LOWER(app_info.id) = 'th.in.robinhood'
            AND user_id IS NOT NULL
            AND user_id != ' '
            AND ga_session_id IS NOT NULL
            AND event_name IN (SELECT event_name FROM ota_event_name)
            AND event_timestamp > (SELECT MAX(event_timestamp) FROM base_ota_events_i_d)
), pivotTable AS (
    SELECT 
        *
    FROM 
        basedEvent
    PIVOT(
        MAX(string_value) FOR key IN (
        'content_group',
        'first_touch_source',
        'carousel_id',
        'carousel_name',
        'hotel_id',
        'hotel_name',
        'booking_id',
        'hotel_destination_id',
        'banner_id',
        'recommendlocation_id',
        'search_term'
    )
  )
)
SELECT DISTINCT * FROM pivotTable
WHERE LOWER(content_group) IN ('ota hotel', 'ota', 'ota flight')