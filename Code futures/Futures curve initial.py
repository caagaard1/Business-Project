import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import statsmodels.api as sm
from statsmodels.api import OLS
from scipy import stats
from sklearn.metrics import r2_score
from statsmodels.tsa.stattools import adfuller
from scipy.optimize import minimize

day_ahead_df = pd.read_csv(r'C:\Users\chrsr\PycharmProjects\Business_Project\Data\Day ahead\20250121 - Electricity day ahead prices, avg.csv')
futures_df = pd.read_csv(r'C:\Users\chrsr\PycharmProjects\Business_Project\Data\Futures\20250122 - Futures curve.csv')

day_ahead_df = day_ahead_df.set_index(pd.to_datetime(day_ahead_df['Date delivery'], format = '%d/%m/%Y') ).drop(columns = {'Date delivery'})
futures_df = futures_df.set_index(pd.to_datetime(futures_df['Dates'], format = '%d/%m/%Y')).drop(columns = {'Dates'}).dropna()
futures_df = futures_df.apply(pd.to_numeric, errors = 'coerce')

day_ahead_df_monthly = day_ahead_df.resample('ME').mean()

futures_month_df = futures_df.copy()
for i in range(0,len(futures_df.columns)):
    futures_month_df[f'M+{i}'] = pd.to_datetime(futures_month_df.index.to_period('M').astype(str))
    futures_month_df[f'M+{i}'] = futures_month_df[f'M+{i}'] + pd.DateOffset(months = i) + pd.offsets.MonthEnd(0)


def futures_trades_pull(date,futures_df, futures_month_df):
    date_ID = date + pd.offsets.MonthEnd(0)
    date_first = date - pd.DateOffset(months = 9)
    date_range = pd.DataFrame(futures_month_df.loc[date_first:date].index)
    date_value_range = pd.DataFrame({'Dates': date_range['Dates'],
                                     'price': np.float64(0),
                                     'time delta': 0,
                                     'diff real' : float(0)}).set_index(date_range['Dates']).drop(columns = 'Dates')
    month_mean = float(day_ahead_df_monthly.loc[date + pd.offsets.MonthEnd(0)].iloc[0])
    for i in date_range['Dates']:
        col = (futures_month_df.loc[i] == date_ID).idxmax()
        date_value_range.loc[i, 'price'] = float(futures_df.loc[i].loc[col])
        date_value_range.loc[i, 'diff real'] = float(futures_df.loc[i].loc[col]) - month_mean
        date_value_range.loc[i, 'time delta'] = int((i - date).days)

    return date_value_range

date = pd.to_datetime('2023-10-01')
date_last_period = date + pd.offsets.MonthEnd(0)

plt.figure()
plt.plot(futures_trades_pull(date, futures_df, futures_month_df), label='Daily MOC pricing for forward contract')
plt.plot(day_ahead_df.loc[date_last_period - pd.DateOffset(months =10): date_last_period], color='green', label=f'Day ahead prices for {date.strftime('%B')} {date.year}')
plt.axhline(y = float(day_ahead_df_monthly.loc[date_last_period].iloc[0]), color = 'red', linestyle = '--', label=f'Avg. realized day ahead price for {date.strftime('%B')} {date.year}')
plt.title(f"Forward price for {date.strftime('%B')} {date.year}")
plt.legend()
plt.tight_layout()
plt.show()

#Average convergence of each futures contract period regression