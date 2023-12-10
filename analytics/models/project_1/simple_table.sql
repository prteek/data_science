{{ config(
  materialized='table',
) }}

SELECT
  *
FROM strava.activities
limit 50;
