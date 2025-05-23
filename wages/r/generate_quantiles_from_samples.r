use("tibble")

generate_quantiles_from_samples <- function(df, quantiles) {
    values <- quantile(df$samples, quantiles)
    return(tibble(quantiles = quantiles, values = values))
}
