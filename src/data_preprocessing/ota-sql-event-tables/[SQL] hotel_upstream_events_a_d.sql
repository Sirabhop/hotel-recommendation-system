-- Databricks notebook source
CREATE OR REPLACE TABLE hotel_upstream_events_a_d AS
WITH baseTable AS (
  SELECT
    a.user_id,
    a.event_date,
    a.event_timestamp,
    a.event_name,
    b.upstream_event_tag,
    CONCAT(a.user_id, a.ga_session_id) AS uid,
    prop.key,
    prop.value.string_value
  FROM
    events_i_d a 
  LEFT JOIN hotel_event_name b
    ON a.event_name = b.event_name
  LATERAL VIEW EXPLODE(a.event_params) exploded_user_prop AS prop
  WHERE
    LOWER(app_info.id) = 'th.in.robinhood'
    AND a.user_id IS NOT NULL
    AND a.user_id != ' '
    AND a.ga_session_id IS NOT NULL
    AND prop.key IN ('hotel_id', 'hotel_name','first_touch_source')
    AND b.upstream_event_tag != 'N'
),
pivotedTable AS (
  SELECT
    *
  FROM
    baseTable PIVOT (
      FIRST(string_value) FOR key IN ('hotel_id', 'hotel_name','first_touch_source')
    )
),
countTable AS (
  SELECT
    *
  FROM
    pivotedTable
  PIVOT (
    COUNT(DISTINCT uid) AS impression, COUNT(DISTINCT user_id) AS reach
    FOR upstream_event_tag IN ('search', 'location_bubble', 'search_recommend', 'favorite_list', 'carousel', 'banner')
  )
),
finalTable AS (
  SELECT
    *
  FROM
    countTable
  WHERE
    hotel_id IS NOT NULL
)
SELECT * FROM finalTable