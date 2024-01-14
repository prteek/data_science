#%%
import pandas as pd
import numpy as np
from lets_plot import *
import os
import duckdb
import yfinance as yf

sql = lambda q: duckdb.sql(q).df()

def calculate_macd(df, n_fast=12, n_slow=26, n_signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence) for a given DataFrame.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - n_fast (int): The number of periods for fast moving average. Default is 12.
    - n_slow (int): The number of periods for slow moving average. Default is 26.
    - n_signal (int): The number of periods for signal line. Default is 9.

    Returns:
    - macd (DataFrame): The DataFrame containing the calculated MACD values.

    Example Usage:
    df = pd.read_csv('data.csv')
    macd = calculate_macd(df, n_fast=10, n_slow=20, n_signal=5)
    """
    macd = sql(f"""
    select date,
    close,
    ma_fast-ma_slow as macd,
    avg(ma_fast-ma_slow) over(order by date asc rows between {n_signal} preceding and current row) as signal,
    ma_fast-ma_slow - avg(ma_fast-ma_slow) over(order by date asc rows between {n_signal} preceding and current row) as histogram
    from (
    select date,
    close,
    avg(close) over(order by date asc rows BETWEEN {n_slow} preceding and current row) as ma_slow,
    avg(close) over(order by date asc rows BETWEEN {n_fast} preceding and current row) as ma_fast
    from df
    )
    """)

    return macd


def plot_macd(macd, title="TICKR"):
    """
    Plot MACD

    Plot the Moving Average Convergence Divergence (MACD) using ggplot library.

    Parameters:
        macd (DataFrame): DataFrame containing the MACD data.
        title (str, optional): Title for the plot. Defaults to "TICKR".

    Returns:
        gggrid: ggplot grid object containing the plot.

    """
    p0 = ggplot(macd, aes(x='date')) + geom_line(aes(y='close'), color='black') + ggtitle(title)
    p1 = ggplot(macd, aes(x='date')) + geom_line(aes(y='macd'), color='red') + geom_line(aes(y='signal'),
                                                                                         color='green') + geom_bar(
        aes(y='histogram'), color='blue', stat="identity")

    return gggrid([p0, p1], ncol=1)


def simple_macd_strategy(macd):
    macd['buy'] = (macd['macd'].shift(1) < macd['signal'].shift(1)) & (macd['macd'] >= macd['signal'])
    macd['sell'] = (macd['macd'].shift(1) > macd['signal'].shift(1)) & (macd['macd'] <= macd['signal'])
    return macd


def modified_macd_strategy(macd):
    macd['buy'] = (macd['macd'].shift(1) < macd['signal'].shift(1)) & (macd['macd'] >= macd['signal']) & (macd['macd'] <=0)
    macd['sell'] = (macd['macd'].shift(1) > macd['signal'].shift(1)) & (macd['macd'] <= macd['signal']) & (macd['macd'] >= 0)
    return macd


def buy_and_hold_strategy(macd):
    macd['buy'] = False
    macd['sell'] = False
    macd.loc[0, "buy"] = True
    macd.loc[len(macd)-1, "sell"] = True
    return macd

#%%
if __name__ == '__main__':
    #%%
    TICKRS = ['AAPL', 'NVDA', 'BRK-A', 'TSLA', 'TMUS', 'MCD', 'NKE', 'V',
              'ADBE', 'GOOGL', 'AMZN', 'T', 'KO', 'DIS', 'JPM', 'MA', 'META', 'MSFT', 'PFE', 'PG']
    start_date = "2023-01-01"
    for TICKR in TICKRS:
        asset = yf.Ticker(TICKR)
        df = (asset
              .history(start=start_date)
              .reset_index()
              .rename(columns={'Date':'date',
                               'Open':'open',
                               'Close':'close',
                               'High':'high',
                               'Low':'low',
                               'Volume':'volume'})
              .assign(date=lambda x: x['date'].dt.date)
              )
        macd = calculate_macd(df)
        p = plot_macd(macd, title=TICKR)
        p.show()


    #%%
    start_date = "2023-01-01"
    end_date = "2024-01-13"
    for TICKR in TICKRS:
        asset = yf.Ticker(TICKR)
        df = (asset
              .history(start=start_date, end=end_date)
              .reset_index()
              .rename(columns={'Date': 'date',
                               'Open': 'open',
                               'Close': 'close',
                               'High': 'high',
                               'Low': 'low',
                               'Volume': 'volume'})
              .assign(date=lambda x: x['date'].dt.date)
              )
        macd = calculate_macd(df)
        macd_strategy = simple_macd_strategy(macd)
        balance = 100
        balance_series = []
        all_in = False
        position = {}
        active_position = None
        for i, row in macd_strategy.iterrows():
            balance_series.append(balance)

            if row['buy'] and (not active_position):
                if all_in:
                    trade = balance
                else: trade = 100
                active_position = row['date']
                position[row['date']] = dict()
                position[row['date']]['buy_price'] = row['close']
                position[row['date']]['quantity'] = trade / row['close']
                balance = balance - trade

            if row['sell'] and active_position:
                position[active_position]['sell_price'] = row['close']
                position[active_position]['profit'] = (row['close'] - position[active_position]['buy_price']) * position[active_position]['quantity']
                balance += position[active_position]['sell_price']*position[active_position]['quantity']
                active_position = None

            profited = balance > balance_series[0]

        if active_position: balance = balance + trade
        macd['balance'] = balance_series

        p0 = ggplot(macd, aes(x='date')) + geom_line(aes(y='balance'), color='black') + ggtitle(f"{TICKR}, end-balance: {round(balance)}")
        p1 = ggplot(macd, aes(x='date')) + geom_line(aes(y='macd'), color='red') + geom_line(aes(y='signal'), color='green') + geom_bar(aes(y='histogram'), color='blue', stat='identity')
        p2 = ggplot(macd, aes(x='date')) + geom_line(aes(y='close'), color='black')
        gggrid([p0, p1, p2], ncol=1).show()

    #%%

    # df = pd.read_excel('S&S ISA (20-5-18-12-1-24).xlsx', skiprows=50, skipfooter=4)
    # x = (df
    #  .dropna(subset=['Money out'])
    #      .rename(columns={"Transaction\ndate": "tdate"})
    #  .loc[lambda x: x['Money out']!="Money out"]
    #  .assign(money_out=lambda x: x['Money out'].replace(to_replace="\£([0-9,\.]+).*", value=r"\1", regex=True).replace(",","", regex=True))
    #  .astype({"money_out":"float"})
    #      .dropna(subset=['tdate'])
    #      .assign(tdate=lambda x: pd.to_datetime(x['tdate'], format='%d/%m/%Y'))
    #      .query("tdate<'2023-11-01'")
    #  )
    #
    # y = (df
    #  .dropna(subset=['Details of transaction'])
    #  .rename(columns={"Transaction\ndate":"tdate"})
    #  .loc[lambda x: (x['Details of transaction'].str.contains("Sale"))]
    #  .assign(money_in=lambda x: x['Money in'].replace(to_replace="\£([0-9,\.]+).*", value=r"\1", regex=True).replace(",","", regex=True))
    #  .astype({"money_in":"float"})
    #  .dropna(subset=['tdate'])
    #  .assign(tdate=lambda x: pd.to_datetime(x['tdate'], format='%d/%m/%Y'))
    #  .query("tdate<'2023-11-01'")
    #  )


    #%%

