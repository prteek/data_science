-- Define the source tables for our model
{{ config(
  materialized='table',
) }}

select moving,
    heartrate,
    time
from {{ref('filtered_intermediate_table')}}
