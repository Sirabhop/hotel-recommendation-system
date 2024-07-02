-- Databricks notebook source
CREATE OR REPLACE TABLE playground_prod.ml_hotel_recommendation.base_training_data_a_d AS
WITH completeData AS (
  SELECT
    from_json(
      request,
      'hotelId STRING, cityId STRING, checkInDate STRING, checkOutDate STRING, room ARRAY<STRUCT<roomType:STRING, adults:INT, children:INT, childAge1:INT, childAge2:INT>>, currency STRING, countryId STRING, roomCode STRING, roomCategory STRING, price DOUBLE, supplierId STRING, supplierName STRING, checkPrice INT, mealCode STRING'
    ) as hotel_data,
    bookingUrn AS booking_id
  from
    reservation_booking_details_i_d
),
mergedData (
  SELECT
    a.uid, a.user_id, a.booking_id, a.total_duration,
    a.max_event_timestamp AS event_timestamp,
    c.*,
    CASE WHEN a.hotel_id IS NULL THEN b.booking_hotel_id ELSE a.hotel_id END AS hotel_id,
    CASE WHEN b.booking_hotel_id IS NOT NULL THEN 1 ELSE 0 END AS complete_flag
  FROM
    hotel_events_i_d a
    LEFT JOIN (
      SELECT
        hotel_data.hotelID AS booking_hotel_id,
        booking_id
      FROM
        completeData
    ) b ON a.booking_id = b.booking_id
    INNER JOIN hotel_event_name c
      ON a.event_name = c.event_name
), 
finalData (
SELECT  uid, 
        user_id AS raw_user_id, 
        main_action, 
        hotel_id, 
        complete_flag,
        ROUND(SUM(total_duration/60), 2) AS duration,
        MAX(event_timestamp) AS event_timestamp
FROM mergedData
GROUP BY uid, user_id, main_action, hotel_id, complete_flag
)
SELECT  b.user_id,
        a.main_action, 
        c.hotel_id, 
        CASE  WHEN a.duration IS NULL THEN 0
              ELSE a.duration
        END AS duration,
        a.complete_flag,
        a.event_timestamp
 FROM finalData a
 INNER JOIN user_info_i_d b
  ON a.raw_user_id = b.raw_user_id
 INNER JOIN hotel_info_a_d c
  ON a.hotel_id = c.hotel_code
 WHERE  a.main_action IS NOT NULL