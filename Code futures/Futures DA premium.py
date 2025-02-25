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
from dateutil.relativedelta import relativedelta
import matplotlib
matplotlib.use("TkAgg")

"""We witness from file "futures vs.dayahead correlation" that the correlation between futures prices for differing months are largely perfect,
and as such we will focus our analysis on M+1, i.e., the front-month futures price. This contract is both the most widely traded, easiest to understand,
and the most correlated with the day-ahead prices, making it suitable for our continued analysis. 
"""

futures_df_full = pd.read_excel(r'C:\Users\chrsr\Business_Project_GitHub\Data\futures vs. dayahaead correlation.xlsx', sheet_name=1)
futures_df_full = futures_df_full.set_index(pd.to_datetime(futures_df_full['Dates'], format = '%d/%m/%Y')).drop(columns = {'Dates'}).dropna()
futures_df_full = futures_df_full.apply(pd.to_numeric, errors = 'coerce')
futures_df = futures_df_full['M+1'].copy()

plot_year = 2024

plt.figure(figsize=(10, 6))
plt.plot(futures_df.loc[futures_df.index.year == plot_year], label = 'M+1')
plt.title('Front-month futures curve')
plt.xlabel('Dates')
plt.ylabel('€ / MWh')


"""We witness large discontinuous changes in the front-month price at the end of each month, this is due to the rollover effect where in the
front-month contract changes to the new month which has been separately traded. In an attempt to combat this, giving us a smoother futures
curve, we will apply a in month weighting of the two futures contracts, the adjusted futures contract price will be found as:

M+1, adj = M+1 * (D_1 / M_0) + M+2 * (M_0 - D_1) / M_0
Where: D_1 is the days in the month until next month, M_0 is the total number of days in the month, M+2 is the month after the front-months futures contract

While this creates an artificial futures contract price that is not traded in real life, it will provide us with a much more smooth and
continuous function, allowing us to more robustly determine statistical properties and compare it with other continuous forms of information
such as interest rates, stock market indices, etc. 

"""

futures_adj_df = pd.read_excel(r'C:\Users\chrsr\Business_Project_GitHub\Data\futures vs. dayahaead correlation.xlsx', sheet_name=0)
futures_adj_df = futures_adj_df.set_index(pd.to_datetime(futures_adj_df['Dates'], format = '%d/%m/%Y')).drop(columns = {'Dates'}).dropna()
futures_adj_df = futures_adj_df.apply(pd.to_numeric, errors = 'coerce')
futures_adj_df = futures_adj_df['M+1, adj.'].copy()
futures_adj_df = pd.DataFrame(futures_adj_df)

plt.plot(futures_adj_df.loc[futures_adj_df.index.year == plot_year], label = 'M+1, adj.')


months_ends_temp = futures_adj_df.loc[futures_adj_df.index.year == plot_year].resample('ME').last().index
for date in months_ends_temp:
    plt.axvline(date, color = 'red', linestyle = '--', alpha = 0.5)

plt.legend()
plt.tight_layout()
plt.show()

"""By our adjustment we reduce the large up and down jumps witnessed at the end of each month. We will now begin by looking at the premium paid
on the futures contract over the day ahead price for the day, which would imply the additional (or less) price an investor is willing to 
pay to hedge their risk on future day-ahead prices
"""

futures_adj_DApremium = (futures_adj_df.copy()).sub(futures_df_full['Day Ahead'], axis = 0)
futures_adj_DApremium['30-day MVA'] = futures_adj_DApremium.iloc[:,0].rolling(window=30).mean()

plt.figure(figsize=(10, 6))

plt.plot(futures_adj_DApremium.iloc[:,0], label = 'Premium')
plt.plot(futures_adj_DApremium.iloc[:,1], label = '30-day MVA')

plt.title('Premium Output Over Time')
plt.xlabel('Time Delta')
plt.ylabel('Premium')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

"""Plotting the premium paid over the day-ahead price and the 30-day moving average we observe that the premium is generally positive as well
having no obvious patterns except for it being very high during the period of 2022 where day-ahead prices skyrocketed. We will now attempt to
understand the premium paid in the context of other macroeconomic factors such as interest rates (should dictate the carry in the futures
contract), stock market returns, gas prices, gas storage levels, and economic sentiment (through producer price index or similar)."""

