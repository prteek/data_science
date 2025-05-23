use("tidyr")
use("dplyr")
use("conflicted")


fit_predict_isoreg <- function(df) {
    model <- isoreg(df$median_combined ~ df$year)
    out <- tibble(year = df$year, median_combined = model$yf)
    return(out)
}

monotonic_constraint_medians <- function(df) {
    medians <- df %>%
        group_by(region) %>%
        group_modify(~ fit_predict_isoreg(.x)) %>%
        ungroup()

    return(medians)
}
