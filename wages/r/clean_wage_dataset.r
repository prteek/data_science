use("tidyr")
use("dplyr")

clean_wage_dataset <- function(df) {
    df_clean <- df %>%
        mutate(
            region = str_remove(str_remove(tolower(region), " ua$"), " /.*$")
        ) %>%
        pivot_longer(
            cols = c(`10`, `20`, `25`, `30`, `40`, `50`, `60`, `70`, `75`, `80`, `90`),
            names_to = "percentiles",
            values_to = "values"
        ) %>%
        mutate(
            percentiles = as.numeric(percentiles),
            values = as.numeric(values)
        )

    return(df_clean)
}
