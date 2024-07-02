-- Databricks notebook source
CREATE OR REPLACE TABLE base_carousel_events_a_d
WITH basedEvent AS (
    SELECT  event_date, event_timestamp, event_name, user_id, ga_session_id,
            CONCAT(user_id, ga_session_id) AS uid,
            prop.key, prop.value.string_value
    FROM
            events_i_d LATERAL VIEW EXPLODE(event_params) exploded_user_prop AS prop
    WHERE
            LOWER(app_info.id) = 'th.in.robinhood'
            AND user_id IS NOT NULL
            AND user_id != ' '
            AND ga_session_id IS NOT NULL
            AND event_name IN ('hotel_main_carousel_click', 'hotel_main_carouselviewall_click', 'hotel_carouselviewall_select_click') --from confluence
            AND prop.key IN ('hotel_id', 'hotel_name', 'carousel_id', 'carousel_name') --from 1. display key calue
), basePlaylist AS (
SELECT * FROM basedEvent
PIVOT (
        MAX(string_value)
        FOR key IN ('hotel_id', 'hotel_name', 'carousel_id', 'carousel_name')
)
)
SELECT DISTINCT * FROM basePlaylist GROUP BY ALL