import numpy as np
import collections
import pandas as pd
import os
from scipy.stats import t
from urllib.request import urlretrieve
from datetime import datetime

x = 'd'


def get_data(fund_ticker, rfr_ticker='PLOPLN6M', interval='d'):
    """

    Args:
        fund_ticker (object): inPzu fund under scrutiny
        rfr_ticker: risk free rate fund_ticker
        interval (object):
    """
    global x

    x = str(interval)

    fund_csv_file = str(fund_ticker) + '_' + x + '.csv'
    rfr_csv_file = str(rfr_ticker) + '_' + x + '.csv'

    if os.path.exists(fund_csv_file):
        ctime = os.path.getmtime(fund_csv_file)
        ctime_datetime = datetime.fromtimestamp(ctime).date()
        today = datetime.today().date()
        if ctime_datetime != today:
            try:
                url = f'https://stooq.com/q/d/l/?s={fund_ticker}&i={interval}'
                urlretrieve(url, fund_csv_file)
                url = f'https://stooq.com/q/d/l/?s={rfr_ticker}&i={interval}'
                urlretrieve(url, rfr_csv_file)
            except IOError as e:
                print(f'Error retrieving file {e}')
    else:
        try:
            url = f'https://stooq.com/q/d/l/?s={fund_ticker}&i={interval}'
            urlretrieve(url, fund_csv_file)
            url = f'https://stooq.com/q/d/l/?s={rfr_ticker}&i={interval}'
            urlretrieve(url, rfr_csv_file)
        except IOError as e:
            print(f'Error retrieving file {e}')

    fund_datas = pd.read_csv(fund_csv_file, index_col='Date', usecols=['Date', 'Close'])
    rfr_datas = pd.read_csv(rfr_csv_file, index_col='Date', usecols=['Date', 'Close'])
    fund_datas.merge(rfr_datas, left_index=True, right_index=True, suffixes=('_fund', '_rfr')).to_csv('merged_data.csv',
                                                                                                      index=True)
    fund_datas = pd.read_csv('merged_data.csv', usecols=['Close_fund']).values.flatten()
    rfr_datas = pd.read_csv('merged_data.csv', usecols=['Close_rfr']).values.flatten()
    return fund_datas, rfr_datas


def time_series_to_returns(price_series):
    """
    Convert price series of fund into return series.
    Args:
        price_series: one dimensional array (list).

    Returns:
        time series: one dimensional array of returns.

    """
    time_series_returns = np.diff(np.log(price_series))
    return time_series_returns


def mean_return(price_series, interval='d'):
    """
    Calculate mean return of price series.
    Args:
        price_series:
        interval: string 'd'-days, 'w'-weeks, 'm'-months, 'q'-quarters
    """
    interval = interval.lower()
    if interval == 'd':
        n = 250
    elif interval == 'w':
        n = 52
    elif interval == 'm':
        n = 12
    elif interval == 'q':
        n = 4
    elif interval == 'y':
        n = 1
    else:
        print('Please input correct interval: "d" or "w" or "m" or "y".')

    returns = time_series_to_returns(price_series)
    mean_ret = np.mean(returns) * n
    risk = np.std(returns) * np.sqrt(n)
    return mean_ret, risk


def max_drawdown(price_series):
    """
    Calculate the maximum drawdown of a time series.

    Parameters:
    - time_series (list or numpy array): A 1-dimensional array of numbers representing the time series.

    Returns:
    - max_dur_dd (float): The maximum drawdown of the time series.
    - max_dd (float): The maximum drawdown of the time series.
    """
    cum_max = np.maximum.accumulate(price_series)
    max_dur_dd = max(collections.Counter(cum_max).values())
    drawdown = (cum_max - price_series) / cum_max
    max_dd = max(drawdown)
    return max_dur_dd, max_dd


def sharpe_ratio(price_series, risk_free_rate, interval='d'):
    """
    Calculate sharp ration of price series: as (returns - rf rate )/standard return of rf rate.
    Args:
        interval:
        price_series: one dimensional array (list) of prices.
        risk_free_rate:

    Returns:
        Sharpe ratio of returns.

    Further reading:
    https://abcportfela.pl/what-is-the-sharpe-ratio-and-how-is-it-used-to-measure-investment-risk/
    """
    interval = interval.lower()
    if interval == 'd':
        n = 250
    elif interval == 'w':
        n = 52
    elif interval == 'm':
        n = 12
    elif interval == 'q':
        n = 4
    elif interval == 'y':
        n = 1
    else:
        print('Please input correct interval: "d" or "w" or "m" or "y".')

    returns = time_series_to_returns(price_series)
    excess_returns = returns - np.log(1 + (risk_free_rate[1:] / 100)) / 365
    mean_excess_returns = np.mean(excess_returns) * n
    std_excess_returns = np.std(excess_returns) * np.sqrt(n)
    sharpe_ratio_result = np.exp(mean_excess_returns / std_excess_returns) - 1
    return sharpe_ratio_result


def sortino_ratio(price_series, risk_free_rate, interval='d'):
    interval = interval.lower()
    if interval == 'd':
        n = 250
    elif interval == 'w':
        n = 52
    elif interval == 'm':
        n = 12
    elif interval == 'q':
        n = 4
    elif interval == 'y':
        n = 1
    else:
        print('Please input correct interval: "d" or "w" or "m" or "y".')

    returns = time_series_to_returns(price_series)
    downside_returns = returns.copy()
    downside_returns[returns >= 0] = 0
    downside_std = np.std(downside_returns) * np.sqrt(n)
    excess_returns = returns - np.log(1 + (risk_free_rate[1:] / 100)) / 365
    mean_excess_returns = np.mean(excess_returns) * n
    sortino_ratio_result = np.exp(mean_excess_returns / downside_std) - 1
    return sortino_ratio_result


def value_at_risk(price_series, confidence_level=0.95, nominal=1000, interval='d'):
    """
    Calculate value at risk of price series
    Args:
        price_series: one dimensional array of list of prices.
        confidence_level: default value is 0.95
        nominal: portfolio nominal value, default value is 1000.
        interval: d - days, w - weeks, m - months, etc.

    Returns:
        value at risk

    """
    interval = interval.lower()
    if interval == 'd':
        n = 250
    elif interval == 'w':
        n = 52
    elif interval == 'm':
        n = 12
    elif interval == 'q':
        n = 4
    elif interval == 'y':
        n = 1
    else:
        print('Please input correct interval: "d" or "w" or "m" or "y".')

    # calculate value at risk given time series
    return_series = time_series_to_returns(price_series)
    volatility = np.std(return_series) * np.sqrt(n)
    val_at_risk = volatility * t.ppf(1 - confidence_level, len(return_series) - 1)
    dollar_val_at_risk = nominal * val_at_risk
    return val_at_risk, dollar_val_at_risk


# Exemplary inputs
time_series, risk_f_rate = get_data('1623.N')

mean_ret_risk = mean_return(time_series)
max_dr_down = max_drawdown(time_series)
sharpe_ratio_value = sharpe_ratio(time_series, risk_f_rate)
sortino_ratio_value = sortino_ratio(time_series, risk_f_rate)
v_at_r = value_at_risk(time_series, 0.95)

print("Mean return and risk:", mean_ret_risk)
print("Maximum drawdown and length of drawdown:", max_dr_down)
print("Współczynnik Sharpe'a wynosi:", sharpe_ratio_value)
print("Sortino Ratio wynosi:", sortino_ratio_value)
print("Value at risk wynosi:", v_at_r)
