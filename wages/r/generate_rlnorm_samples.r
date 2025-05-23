use("tibble")

generate_rlnorm_samples <- function(df, n_samples = 500) {
    return(tibble(samples = rlnorm(n_samples, meanlog = df$mu, sdlog = df$sigma)))
}
