import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import statsmodels.api as sm
from statsmodels.api import OLS
from scipy import stats
from scipy.stats import t
from sklearn.metrics import r2_score
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.optimize import minimize
from dateutil.relativedelta import relativedelta
import matplotlib
from Functions_stat import test_linear_regression_assumptions
from Functions_stat import run_xgboost_regression
from Functions_stat import run_lasso_regression
from Functions_stat import run_pca

matplotlib.use("TkAgg")

"""We witness from file "futures vs.dayahead correlation" that the correlation between futures prices for differing months are largely perfect,
and as such we will focus our analysis on M+1, i.e., the front-month futures price. This contract is both the most widely traded, easiest to understand,
and the most correlated with the day-ahead prices, making it suitable for our continued analysis. 
"""

#Set up day ahead prices for comparison later
day_ahead_prices = pd.read_csv(r'C:\Users\chrsr\Business_Project_GitHub\Data\Day ahead\20250121 - Electricity day ahead prices, avg.csv')
day_ahead_prices = day_ahead_prices.set_index(pd.to_datetime(day_ahead_prices['Date delivery'], format ='%d/%m/%Y')).drop(columns = {'Date delivery'})
day_ahead_prices = day_ahead_prices.rename(columns = {'Price': 'Day ahead price'})
day_ahead_prices = day_ahead_prices.rename_axis('Date')

intervals = sorted(list({7, 30, 60}))

#Set up of MVA of day-ahead prices
for i in intervals:
    day_ahead_prices[f'DA MVA {i}'] = day_ahead_prices['Day ahead price'].rolling(window = i).mean()

# Set up of rolling vol of day-ahead prices
for i in intervals:
    day_ahead_prices[f'DA rolling vol {i}'] = day_ahead_prices['Day ahead price'].rolling(window=i).std()


#Set up futures dataframe
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

futures_adj_DA_df = pd.DataFrame(futures_adj_df)
futures_adj_DA_df = pd.merge(futures_adj_DA_df, day_ahead_prices, left_index = True, right_index = True, how = 'left')



futures_adj_DApremium = pd.DataFrame(futures_adj_df)
futures_adj_DApremium['M+1, adj, premium'] = futures_adj_DA_df['M+1, adj.'] - futures_adj_DA_df['Day ahead price']

intervals = sorted(list({7, 30, 60}))

#Set up of Premiums on MVA of day ahead prices
for i in intervals:
    futures_adj_DApremium[f'Premium on MVA {i}'] = futures_adj_DA_df['M+1, adj.'] - futures_adj_DA_df[f'DA MVA {i}']


plt.figure(figsize=(10, 6))

plt.plot(futures_adj_DApremium, label = futures_adj_DApremium.columns)

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

eco_correl_df = pd.read_excel(r'C:\Users\chrsr\Business_Project_GitHub\Data\20250226 - Eco correl data.xlsx')
eco_correl_df = eco_correl_df.set_index(pd.to_datetime(eco_correl_df['Dates'], format = '%d/%m/%Y')).drop(columns = {'Dates'})


futures_premium_eco_df = pd.merge(futures_adj_DApremium, eco_correl_df, left_index = True, right_index = True, how = 'left').dropna()
futures_premium_eco_df['Gas Storage (TWh), deseason'] =futures_premium_eco_df['Gas Storage (TWh)'] - seasonal_decompose(futures_premium_eco_df['Gas Storage (TWh)'], model='additive', period = 365).seasonal


correl_matrix_1 = futures_premium_eco_df.corr()


"""Technically to be done later, but: Looking at day-ahead volatility"""


#Day ahead plot showing MVA and rolling Vol
plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(day_ahead_prices.iloc[:, 0:4], label = day_ahead_prices.iloc[:, 0:4].columns)
plt.title('Day ahead prices')
plt.legend()

plt.subplot(2,1,2)
plt.plot(day_ahead_prices.iloc[:, 4:7], label = day_ahead_prices.iloc[:, 4:7].columns)
plt.legend()
plt.title('Day ahead volatility')
plt.tight_layout()
plt.show()

#Merging into a new dataframe

futures_premium_reg_df = pd.merge(futures_adj_DApremium, day_ahead_prices, left_index = True, right_index = True, how ='left').dropna()

correl_matrix_2 = futures_premium_reg_df.corr()

"""The correlation matrix gives us valuable information:
1. The spot premium does not seem to correlated in any way with the day-ahead price on a specific day
2. The spot premium is somewhat correlated with the long-term moving averages and rolling volatilities of day ahead prices
3. The moving average premiums are heavily correlated with long-term behaviors (price and volatility) of day ahead prices

This would indicate that the premium on futures prices are more so dependent on the medium-term (7-60 days) market conditions which would
make intuitive sense given that the futures contract is for an entire month, meaning that while daily price movements give information
actors in the market are more concerned about medium-term behavior. 

However, we should also be cautious in interpreting the results directly, as the correlation between day ahead prices and the premium
is also somewhat mechanical, as the day ahead prices have been used in the calculation of the premium!!  
"""

#Create a regression model on %change adj premium and %change DA rolling vol

#Percentage change
target_df = pd.DataFrame({'target': futures_premium_reg_df['M+1, adj.'].rolling(window = 7).mean().dropna().pct_change()})

feature_df = pd.DataFrame({
    'f1_DA_7MVA' : futures_premium_reg_df['DA MVA 7'].pct_change(),
    'f2_DA_30MVA' : futures_premium_reg_df['DA MVA 30'].pct_change(),
    'f3_DA_60MVA' : futures_premium_reg_df['DA MVA 60'].pct_change(),
    'f4_DAVol_7MVA' : futures_premium_reg_df['DA rolling vol 7'].pct_change(),
    'f5_DAVol_30MVA' : futures_premium_reg_df['DA rolling vol 30'].pct_change(),
    'f6_DAVol_60MVA' : futures_premium_reg_df['DA rolling vol 60'].pct_change(),
    'f7_FUVol_7MVA' : futures_premium_reg_df['M+1, adj.'].rolling(window=7).std().dropna().pct_change(),
    'f8_FUVol_30MVA' : futures_premium_reg_df['M+1, adj.'].rolling(window=30).std().dropna().pct_change(),
    'f9_FUVol_60MVA' : futures_premium_reg_df['M+1, adj.'].rolling(window=60).std().dropna().pct_change(),
    'f10_FU_7MVA_VAL' : futures_premium_reg_df['M+1, adj.'].rolling(window = 7).mean().dropna(),
    'f11_FU_30MVA_VAL' : futures_premium_reg_df['M+1, adj.'].rolling(window = 30).mean().dropna(),
    'f12_FU_60MVA_VAL' : futures_premium_reg_df['M+1, adj.'].rolling(window = 60).mean().dropna(),
    'f13_DA_7MVA_VAL' : futures_premium_reg_df['DA MVA 7'],
    'f14_DA_30MVA_VAL' : futures_premium_reg_df['DA MVA 30'],
    'f15_DA_60MVA_VAL' : futures_premium_reg_df['DA MVA 60'],
}).shift(1).dropna()

regression_df = target_df.join(feature_df, how = 'left').dropna()

correl_matrix_3 = regression_df.corr()

#Absolute relationship
target_df = pd.DataFrame({'target': futures_premium_reg_df['M+1, adj, premium']})

feature_df = pd.DataFrame({
    'f1_DA_7MVA' : futures_premium_reg_df['DA MVA 7'],
    'f2_DA_30MVA' : futures_premium_reg_df['DA MVA 30'],
    'f3_DA_60MVA' : futures_premium_reg_df['DA MVA 60'],
    'f4_DAVol_7MVA' : futures_premium_reg_df['DA rolling vol 7'],
    'f5_DAVol_30MVA' : futures_premium_reg_df['DA rolling vol 30'],
    'f6_DAVol_60MVA' : futures_premium_reg_df['DA rolling vol 60'],
    'f7_FUVol_7MVA' : futures_premium_reg_df['M+1, adj.'].rolling(window=7).std().dropna(),
    'f8_FUVol_30MVA' : futures_premium_reg_df['M+1, adj.'].rolling(window=30).std().dropna(),
    'f9_FUVol_60MVA' : futures_premium_reg_df['M+1, adj.'].rolling(window=60).std().dropna(),
    'f10_FU_7MVA_VAL' : futures_premium_reg_df['M+1, adj.'].rolling(window = 7).mean().dropna(),
    'f11_FU_30MVA_VAL' : futures_premium_reg_df['M+1, adj.'].rolling(window = 30).mean().dropna(),
    'f12_FU_60MVA_VAL' : futures_premium_reg_df['M+1, adj.'].rolling(window = 60).mean().dropna(),
    'f13_DA_7MVA_VAL' : futures_premium_reg_df['DA MVA 7'],
    'f14_DA_30MVA_VAL' : futures_premium_reg_df['DA MVA 30'],
    'f15_DA_60MVA_VAL' : futures_premium_reg_df['DA MVA 60'],
}).shift(1).dropna()

regression_df = target_df.join(feature_df, how = 'left').dropna()

correl_matrix_3 = regression_df.corr()










test_linear_regression_assumptions(regression_df, 'target', feature_cols=list(feature_df.columns), plot=False)
temp, model = run_xgboost_regression(regression_df['target'], regression_df[feature_df.columns], max_depth=6, learning_rate=0.1)


pca_df, explained_var, pca_model = run_pca(X=feature_df, n_components=6)
regression_df_pca = target_df.join(pca_df, how = 'left').dropna()
temp, model = run_xgboost_regression(regression_df_pca['target'], regression_df_pca[pca_df.columns], max_depth=6, learning_rate=0.1)




pred_values = pd.DataFrame({'forecast' : model.predict(regression_df_pca[pca_df.columns])}, index=regression_df_pca.index)
temp_df = pd.DataFrame(regression_df['target'])
temp_df['forecast'] = pd.DataFrame(pred_values)

plt.figure()
plt.subplot(2,1,1)
plt.plot(temp_df, label = temp_df.columns)
plt.legend()

plt.subplot(2,1,2)
plt.plot(temp_df['target'] - temp_df['forecast'], label='residuals')
plt.show()



pred_values = pd.DataFrame({'forecast' : model.predict(regression_df[feature_df.columns])}, index=regression_df.index)
temp_df = pd.DataFrame(regression_df['target'])
temp_df['forecast'] = pd.DataFrame(pred_values)

plt.figure()
plt.subplot(2,1,1)
plt.plot(temp_df, label = temp_df.columns)
plt.legend()

plt.subplot(2,1,2)
plt.plot(temp_df['target'] - temp_df['forecast'], label='residuals')
plt.show()


'--------------------------------'


pred_values = pd.DataFrame({'forecast %': model.predict(regression_df[feature_df.columns])}, index=regression_df.index)
pred_values += 1
pred_values['t-1 value'] = pd.Series(futures_premium_reg_df['M+1, adj.'].shift(1)[futures_premium_reg_df.index >= min(regression_df.index)])
pred_values['forecast'] = pred_values['forecast %'] * pred_values['t-1 value']

observed = pd.Series(futures_premium_reg_df['M+1, adj.'][futures_premium_reg_df.index >= min(regression_df.index)])
temp_df = pd.DataFrame({'target': observed})
temp_df['forecast'] = pd.DataFrame(pred_values['forecast'])

plt.figure()
plt.subplot(2,1,1)
plt.plot(temp_df, label = temp_df.columns)
plt.legend()

plt.subplot(2,1,2)
plt.plot(temp_df['target'] - temp_df['forecast'], label='residuals')
plt.show()