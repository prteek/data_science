.ONESHELL:

.PHONY: all \
	update_wages_median_table \
	update_wages_cua_table \
	clean

DB=ds_dev
# Define bucket based on the value of DB
ifneq ($(filter $(DB),ds_prd),)
  BUCKET=bricklane-ds-prd
else
  BUCKET=bricklane-ds-dev
endif

all: wages_median.csv \
	update_wages_median_table \
	wages_cua.csv \
	update_wages_cua_table

wages_median.csv:
	Rscript -e "targets::tar_make('wages_median.csv')"

update_wages_median_table: wages_median.csv
	cat $< | csv2a -o s3://$(BUCKET)/lgps/wages_median \
					-db $(DB) \
					-t wages_median \
					-m overwrite \
					--dtype region='string' \
							year='integer' \
							median_combined='double' \
							etl_dt_id='integer'

wages_cua.csv:
	Rscript -e "targets::tar_make('wages_cua.csv')"

update_wages_cua_table: wages_cua.csv
	cat $< | csv2a -o s3://$(BUCKET)/lgps/wages_cua \
					-db $(DB) \
					-t wages_cua \
					-m overwrite \
					--dtype region='string' \
							year='integer' \
							values='double' \
							percentiles='double' \
							etl_dt_id='integer'

clean:
	rm -r *.csv
