use("tibble")

fit_lnorm_params <- function(df) {
    values <- df$values
    percentiles <- df$percentiles
    log_values <- log(values)
    z_scores <- qnorm(percentiles / 100)
    model <- lm(log_values ~ z_scores)
    mu <- coef(model)[1]
    sigma <- coef(model)[2]
    return(tibble(
        mu = mu,
        sigma = sigma
    ))
}
