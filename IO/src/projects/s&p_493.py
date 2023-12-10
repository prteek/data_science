# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import bs4 as bs
import requests
import yfinance as yf
from datetime import datetime
import duckdb

sql = lambda q: duckdb.sql(q).df()
sns.set()

# %%
resp = requests.get("http://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
soup = bs.BeautifulSoup(resp.text, "lxml")
table = soup.find("table", {"class": "wikitable sortable"})

tickers = []

for row in table.findAll("tr")[1:]:
    ticker = row.findAll("td")[0].text
    tickers.append(ticker)


print(tickers)

tickers = [s.replace("\n", "") for s in tickers]
start = datetime(2000, 1, 1)
end = datetime(2023, 11, 9)
data = yf.download(tickers, start=start, end=end)

df = (
    data.stack()
    .reset_index()
    .rename(index=str, columns={"level_1": "Symbol"})
    .sort_values(["Symbol", "Date"])
)

# %%
# Take tickers of top 7 tech giants in s&p 500
top_tickers = ("AAPL", "MSFT", "AMZN", "GOOG", "GOOGL", "FB", "TSLA")
sns.lineplot(

    sql(
        f"""with grouped as (select *,
        case when Symbol in {top_tickers} then 'TG' else 'Other' end as type,
        case when Symbol in {top_tickers} then Close else Close/2 end as adj_close
        from df
        )
        select avg(adj_close) as adj_close,
                Date,
                type
        from grouped
        group by type, Date
        """
    ),
    y="adj_close",
    x="Date",
    hue="type",
)
plt.xlim(datetime(2018, 1, 1), datetime(2023, 12, 10))
