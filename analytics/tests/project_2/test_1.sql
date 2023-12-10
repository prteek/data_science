select *
from {{ ref('complex_table') }}
where heartrate <= 0 -- no semicolon a the end
