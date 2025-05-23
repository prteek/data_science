use("tidyr")
use("dplyr")
use("conflicted")
use("this.path")
source(file.path(here(), "generate_rlnorm_samples.r"))

conflicts_prefer(dplyr::filter)

bootstrap_median <- function(df) {
    median_combined <- df %>%
        group_by(region, year, source) %>%
        group_modify(~ generate_rlnorm_samples(.x)) %>%
        mutate(index = row_number()) %>%
        group_by(region, year, index) %>%
        summarise(combined_wages = sum(samples)) %>%
        group_by(region, year) %>%
        summarise(median_combined = round(median(combined_wages), 0)) %>%
        ungroup()

    return(median_combined)
}
