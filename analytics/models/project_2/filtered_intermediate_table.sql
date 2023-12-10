{{ config(
  materialized='view',
) }}

select *
from {{ref('intermediate_table')}}
where moving = true
limit 10;
