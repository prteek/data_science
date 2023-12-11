{{ config(
  materialized='ephemeral',
) }}

SELECT *
FROM strava.streams
