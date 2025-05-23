library(this.path)
setwd(here())
renv::load()

library(conflicted)
library(targets)

conflicts_prefer(dplyr::filter)

tar_source("r/get_wage_data.r")
tar_source("r/fit_lnorm_params.r")
tar_source("r/bootstrap_median.r")
tar_source("r/clean_wage_dataset.r")
tar_source("r/monotonic_constraint_medians.r")


tar_option_set(
    packages = c("tidyr", "dplyr", "readr")
    # controller = crew_controller_local(workers = 8)
)

list(
    tar_target(
        wage_data_raw,
        get_wage_data()
    ),
    tar_target(
        wage_data,
        wage_data_raw %>% clean_wage_dataset()
    ),
    tar_target(
        wages.csv,
        wage_data %>%
            mutate(etl_dt_id = as.integer(format(Sys.Date(), "%Y%m%d"))) %>%
            write.csv("wages_cua.csv", row.names = FALSE),
        format = "file"
    ),
    tar_target(
        wage_data_coefs,
        wage_data %>%
            group_by(region, year, source) %>%
            filter(sum(!is.na(values)) >= 3) %>%
            group_modify(~ fit_lnorm_params(.x)) %>%
            ungroup()
    ),
    tar_target(
        wage_data_combined_median,
        wage_data_coefs %>%
            filter(source %in% c("male_full_time", "female_full_time")) %>%
            bootstrap_median()
    ),
    tar_target(
        wage_data_combined_median_monotonic,
        wage_data_combined_median %>%
            monotonic_constraint_medians()
    ),
    tar_target(
        wages_median.csv,
        wage_data_combined_median_monotonic %>%
            mutate(etl_dt_id = as.integer(format(Sys.Date(), "%Y%m%d"))) %>%
            write.csv("wages_median.csv", row.names = FALSE),
        format = "file"
    )
)
