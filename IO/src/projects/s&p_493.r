#%%
# Load necessary libraries
library(dplyr)
library(ggplot2)
library(lubridate)
# Load necessary libraries
library(rvest)
library(dplyr)
library(lubridate)
library(quantmod)
library(tidyverse)

#%%
#%%
# Scrape S&P 500 companies from Wikipedia
url <- "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
webpage <- read_html(url)
table <- html_nodes(webpage, "table")[[1]]
tickers <- html_text(html_nodes(table, "td:nth-child(1)"))

# Remove newline characters
tickers <- gsub("\n", "", tickers)
#%%
#%%
# Define start and end dates
start <- as.Date("2000-01-01")
end <- as.Date("2023-11-09")

# Download stock data
data <- do.call(cbind, lapply(tickers, function(ticker) {
    tryCatch(
        {
            getSymbols(ticker, src = "yahoo", from = start, to = end, auto.assign = FALSE)
        },
        error = function(e) {
            NULL
        }
    )
}))


# Convert to data frame and reshape
df <- data.frame(Date = index(data), coredata(data)) %>%
    pivot_longer(
        cols = -Date,
        names_to = c("Symbol", ".value"),
        names_pattern = "(.*)\\.(.*)"
    ) %>%
    arrange(Symbol, Date)

#%%
#%%

# Take tickers of top 7 tech giants in s&p 500
top_tickers <- c("AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA")

# Create new columns 'type' and 'adj_close' based on conditions
df <- df %>%
    mutate(
        type = ifelse(Symbol %in% top_tickers, "Tech-7", "Others"),
    )

# Group by 'type' and 'Date', and calculate average 'adj_close'
df_grouped <- df %>%
    group_by(type, Date) %>%
    summarise(mean_close = mean(Close, weights=NULL, na.rm = TRUE)) %>%
    filter (Date >= as.Date("2020-01-01")) %>%
    mutate(price_index = mean_close / mean_close[1] * 100)

# Plot
ggplot(df_grouped, aes(x = Date, y = price_index, color = type)) +
    geom_line()
#%%
#%%
