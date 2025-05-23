use("dplyr")
use("tidyr")
use("readxl", "read_excel")
use("purrr", "map_df")
use("stringr")

DATA_DIR <- file.path(dirname(this.path::here()), "data")

process_wage_sheet <- function(filepath, sheet) {
    sheet_data <- readxl::read_excel(path = filepath, sheet = sheet, range = "A5:Q1000") %>%
        filter(!is.na(Code)) %>%
        rename(`50` = Median, region = Description) %>%
        select(-c(Code, `(thousand)`, `change...5`, Mean, `change...7`)) %>%
        relocate(`50`, .after = `40`) %>%
        mutate_all(~ na_if(., "x")) %>%
        mutate(
            year = as.numeric(gsub(".*(\\d{4}).*", "\\1", basename(filepath))),
            source = str_to_lower(str_replace_all(sheet, "[ -]", "_"))
        )

    return(sheet_data)
}


process_wage_file <- function(filepath) {
    sheets <- c("Male Full-Time", "Female Full-Time", "Female Part-Time", "Male Part-Time")
    file_data <- map_df(sheets, \(x) process_wage_sheet(filepath, x))
    return(file_data)
}


get_wage_data <- function() {
    files <- list.files(DATA_DIR)
    filepaths <- lapply(files, \(x) file.path(DATA_DIR, x))
    df <- map_df(filepaths, process_wage_file)
}
