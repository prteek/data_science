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
    return macd


def weekly_buy_and_hold_strategy(macd, dayofweek=1):
    macd['buy'] = (macd['date'].dt.dayofweek == dayofweek)
    macd['sell'] = False
    return macd


#%%
if __name__ == '__main__':
    #%%
    TICKRS = ['AAPL', 'NVDA', 'BRK-B', 'TSLA', 'TMUS', 'MCD', 'NKE', 'V',
              'ADBE', 'GOOGL', 'AMZN', 'T', 'KO', 'DIS', 'JPM', 'MA', 'META', 'MSFT', 'PFE', 'PG']
    start_date = "2022-01-01"
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
    PORTFOLIO = ['AAPL', 'NVDA', 'BRK-B', 'TMUS', 'MCD', 'NKE', 'V', 'ADBE']
    portfolio_profit = 0
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

        positions = {}
        total_invested = 0
        total_profit = 0
        profit_series = []
        invested_series = []
        is_active = False
        trade = 20
        platform_fee = 0.45 # [%]
        macd = calculate_macd(df)
        macd_strategy = modified_macd_strategy(macd)
        weekly_top_up = True
        sell_at_end = False
        for i, row in macd_strategy.iterrows():
            if row['date'].dayofweek == 0 and weekly_top_up : trade += 20
            if row['buy']:
                # 1. Set that a position is active
                # 2. Open a new position with current date as i.d.
                # 3. Set the current position to active
                # 4. Log buy price for current position
                # 5. Calculate the quantity bought, adjusting for fee
                # 6. Update the total amount under-investment currently
                # 7. Update the amount left over to trade

                is_active = True
                positions[row['date']] = dict()
                positions[row['date']]['is_active'] = True
                positions[row['date']]['buy_price'] = row['close']
                positions[row['date']]['quantity'] = trade*(1-platform_fee/100) / row['close'] # platform fee 0.45%
                total_invested += trade
                trade -= trade

            if (row['sell'] and is_active) or ((i == len(macd_strategy)-1) and is_active and sell_at_end):
                # If there is a sell signal and a position is open
                # Or, this is the last data point in the period and there is a command to sell at the end of the period
                # Check for all open positions and process them
                for active_position in positions:
                    if not positions[active_position]['is_active']: continue
                    # 1. Set the selling price for position
                    # 2. Calculate the gain from sell adjusting for fee
                    # 3. Calculate profit from adjusted gain
                    # 4. Log closing price for the position
                    # 5. Log closing date for the position
                    # 6. Close the position
                    # 7. Free gain amount from invested pool
                    # 8. Add profit to running total
                    # 9. Add gain amount to trade pool
                    # 10. Finally, we declare all open positions are closed
                    positions[active_position]['sell_price'] = row['close']
                    positions[active_position]['gain'] = positions[active_position]['sell_price']*positions[active_position]['quantity']*(1-platform_fee/100) # Platform fee
                    positions[active_position]['profit'] = positions[active_position]['gain'] - positions[active_position]['buy_price'] * positions[active_position]['quantity']
                    positions[active_position]['close'] = row['date']
                    positions[active_position]['is_active'] = False
                    total_invested -= positions[active_position]['gain']
                    total_profit += positions[active_position]['profit']
                    trade += positions[active_position]['gain']
                is_active = False

            profit_series.append(total_profit)
            invested_series.append(total_invested)
            profited = total_profit > 0

        macd['profit'] = profit_series
        macd['invested'] = invested_series

        p0 = ggplot(macd, aes(x='date')) + geom_line(aes(y='profit'), color='black') + \
             ggtitle(f"{TICKR}, end-profit: {round(total_profit)}") #, total-invested: {round(total_invested)}, %: {round(total_profit/total_invested*100,2)}")
        p1 = ggplot(macd, aes(x='date')) + geom_line(aes(y='invested'), color='blue')
        p2 = ggplot(macd, aes(x='date')) + geom_line(aes(y='macd'), color='red') + geom_line(aes(y='signal'), color='green') + geom_bar(aes(y='histogram'), color='blue', stat='identity')
        p3 = ggplot(macd, aes(x='date')) + geom_line(aes(y='close'), color='black')
        gggrid([p0, p1, p2, p3], ncol=1).show()

        portfolio_profit += total_profit

    print(f"portfolio profit: {portfolio_profit}")

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

