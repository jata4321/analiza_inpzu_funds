import collections
import os
from datetime import datetime
from urllib.request import urlretrieve

import dash_bootstrap_components as dbc
import plotly.express as px
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, dash_table, Input, Output
from scipy.stats import t

x = 'd'


def get_data(fund_ticker, rfr_ticker='PLOPLN3M', interval='d'):
    """

    Args:
        fund_ticker (object): inPzu fund under scrutiny
        rfr_ticker: risk free rate fund_ticker
        interval (object):
    """
    global x

    x = str(interval)

    fund_csv_file = './data/' + str(fund_ticker) + '_' + x + '.csv'
    rfr_csv_file = './data/' + str(rfr_ticker) + '_' + x + '.csv'

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
        n = 250

    returns = time_series_to_returns(price_series)
    mean_returns = np.mean(returns) * n
    risk = np.std(returns) * np.sqrt(n)
    return mean_returns, risk


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
    max_dd = -max(drawdown)
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
        n = 250

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
        n = 250

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
        n = 250

    # calculate value at risk given time series
    return_series = time_series_to_returns(price_series)
    volatility = np.std(return_series) * np.sqrt(n)
    val_at_risk = volatility * t.ppf(1 - confidence_level, len(return_series) - 1)
    dollar_val_at_risk = nominal * val_at_risk
    wipeout = abs(val_at_risk / (np.mean(return_series) * n))
    return val_at_risk, dollar_val_at_risk, wipeout


# Dictionary of inputs
funds_dict = dict(in_ostr='1623.n', in_obl_pl='1624.n', in_obl_ry_ro='1625.n', in_obl_ry_ws='1643.n',
                  in_obl_inf='1216.n', in_akc_pol='1621.n', in_akc_ry_ws='1378.n', in_akc_ry_ro='1622.n',
                  in_akc_am='2824.n', in_akc_eu='2671.n', in_akc_ce='2701.n', in_akc_sn='1223.n', in_akc_si='1232.n',
                  in_akc_sze='1222.n', in_akc_rz='1262.n', in_akc_rs='1334.n', TBSP='^TBSP', ETFSP500='ETFSP500.PL',
                  ETFSDAX='ETFDAX.PL', WIG='WIG')
name = []
mean_ret = []
mean_risk = []
drawdown_dur = []
drawdown_depth = []
sharpe = []
sortino = []
value_risk = []
wipe = []

for key, val in funds_dict.items():
    time_series, risk_f_rate = get_data(val)

    mean_ret_risk = mean_return(time_series)
    max_draw_down = max_drawdown(time_series)
    sharpe_ratio_value = sharpe_ratio(time_series, risk_f_rate)
    sortino_ratio_value = sortino_ratio(time_series, risk_f_rate)
    v_at_r = value_at_risk(time_series, 0.95)

    name.append(key)
    mean_ret.append(mean_ret_risk[0])
    mean_risk.append(mean_ret_risk[1])
    drawdown_dur.append(max_draw_down[0])
    drawdown_depth.append(max_draw_down[1])
    sharpe.append(sharpe_ratio_value)
    sortino.append(sortino_ratio_value)
    value_risk.append(v_at_r[0])
    wipe.append(v_at_r[2])

dff = pd.DataFrame({'Name': name,
                    'Return': mean_ret,
                    'Risk': mean_risk,
                    'Drawdown_Duration': drawdown_dur,
                    'Drawdown_Depth': drawdown_depth,
                    'Sharpe': sharpe,
                    'Sortino': sortino,
                    'VaR': value_risk,
                    'Wipeout': wipe
                    })

# dff.sort_values(by=['Return'], inplace=True)

'''
You can add two more indicators:
    1) provided investment horizon, what is expected return taking into wipeout period
    2) how distressed the market is based on latest drawdown and normal distribution of returns for
    particular fund. 
'''

app = Dash(__name__, external_stylesheets=[dbc.themes.GRID])

app.layout = html.Div([
    dbc.Row([
        html.H1('Risk & Return for inPZU Funds!', style={'textAlign': 'center'})
    ]),
    dbc.Row([
        dbc.Col([dcc.Dropdown(id='y-select',
                              options=dff.columns,
                              value=dff.columns[1]),
                 dcc.Dropdown(id='x-select',
                              options=dff.columns,
                              value=dff.columns[2]),
                 dcc.RadioItems(id='return-select',
                                options=['All returns', 'Positive returns only'],
                                value='All returns')], width=2),
        dbc.Col(dcc.Graph(id='scatter-plot', figure={}), width=8),
        dbc.Col(html.P(id='click-output'))
        # dbc.Col(dash_table.DataTable(dff.to_dict('records'),
        #                              columns=[{'format': {'locale': {'decimal': '.2'}}}]))
    ])
])


@app.callback(
    Output('scatter-plot', 'figure'),
    Output('click-output', 'children'),
    Input('y-select', 'value'),
    Input('x-select', 'value'),
    Input('return-select', 'value'),
    Input('scatter-plot', 'clickData'),
)
def figure_plot(y_select, x_select, return_select, clicked):
    if return_select == 'Positive returns only':
        filter_dff = dff['Return'] >= 0
        df = dff[filter_dff]
    else:
        df = dff.copy()
    figure = px.scatter(df, x=x_select, y=y_select, trendline="ols", hover_name=df['Name'], height=600)
    if clicked is None:
        fund_info = ''
    else:
        point_clicked = clicked['points'][0]['pointNumber']
        fund_info = dff.loc[point_clicked]

    return figure, str(fund_info)


if __name__ == '__main__':
    app.run(debug=True)
