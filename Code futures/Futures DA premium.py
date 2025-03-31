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

futures_DA_df = pd.DataFrame(futures_df) #WE NO LONGER ADJUST AS IT DOENS'T MAKE SENSE FOR FURTHER ANALYSIS
futures_DA_df = pd.merge(futures_DA_df, day_ahead_prices, left_index = True, right_index = True, how = 'left')

futures_DApremium = pd.DataFrame(futures_df)
futures_DApremium['M+1, premium'] = futures_DA_df['M+1'] - futures_DA_df['Day ahead price']

intervals = sorted(list({7, 30, 60}))

#Set up of Premiums on MVA of day ahead prices
for i in intervals:
    futures_DApremium[f'Premium on MVA {i}'] = futures_DA_df['M+1'] - futures_DA_df[f'DA MVA {i}']

"""Plotting the premium paid over the day-ahead price and the 30-day moving average we observe that the premium is generally positive as well
having no obvious patterns except for it being very high during the period of 2022 where day-ahead prices skyrocketed. We will now attempt to
understand the premium paid in the context of other macroeconomic factors such as interest rates (should dictate the carry in the futures
contract), stock market returns, gas prices, gas storage levels, and economic sentiment (through producer price index or similar)."""

eco_correl_df = pd.read_excel(r'C:\Users\chrsr\Business_Project_GitHub\Data\20250226 - Eco correl data.xlsx')
eco_correl_df = eco_correl_df.set_index(pd.to_datetime(eco_correl_df['Dates'], format = '%d/%m/%Y')).drop(columns = {'Dates'})


futures_premium_eco_df = pd.merge(futures_DApremium, eco_correl_df, left_index = True, right_index = True, how = 'left').dropna()
futures_premium_eco_df['Gas Storage (TWh), deseason'] =futures_premium_eco_df['Gas Storage (TWh)'] - seasonal_decompose(futures_premium_eco_df['Gas Storage (TWh)'], model='additive', period = 365).seasonal


correl_matrix_1 = futures_premium_eco_df.corr()

#Merging into a new dataframe

futures_premium_reg_df = pd.merge(futures_DApremium, day_ahead_prices, left_index = True, right_index = True, how ='left').dropna()

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

#Calculate the estimated futures prices based on the model, turn them into a PDF, and compare with current price

from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d


futures_sim_pdf_data = pd.read_csv(r'C:\Users\chrsr\Business_Project_GitHub\historical_futures_sim_data_2018-01-03 v5.csv')
futures_sim_pdf_data.rename(columns = {futures_sim_pdf_data.columns[0] : 'Dates'}, inplace=True)
futures_sim_pdf_data.set_index(pd.to_datetime(futures_sim_pdf_data.iloc[:,0]), inplace = True)
futures_sim_pdf_data.drop(columns = {futures_sim_pdf_data.columns[0]}, inplace = True)
futures_sim_pdf_data.dropna()
futures_sim_pdf_data = futures_sim_pdf_data.clip(lower = -1000, upper =1000)

kde_pdf_data = pd.DataFrame()
linspace_data = pd.DataFrame()

for i in futures_sim_pdf_data.index:
    calc_data = np.array((futures_sim_pdf_data.iloc[futures_sim_pdf_data.index == i,:]).iloc[0,:])
    kde_calc = gaussian_kde(calc_data)

    x_calc = np.linspace(min(calc_data), max(calc_data), 1000)
    new_row_linspace = pd.DataFrame([x_calc])
    new_row_linspace.index = [pd.Timestamp(i)]

    linspace_data = pd.concat([linspace_data, new_row_linspace])

    pdf_calc = kde_calc(x_calc)
    new_row_pdf = pd.DataFrame([pdf_calc])
    new_row_pdf.index = [pd.Timestamp(i)]

    kde_pdf_data = pd.concat([kde_pdf_data, new_row_pdf])

    print(i)

def get_mode_centered_interval(x, y, p=0.95):
    dx = np.diff(x).mean()
    # Normalize the PDF
    y = y / (y.sum() * dx)
    # Find index of the mode (highest point)
    mode_idx = np.argmax(y)
    # Sort indices by descending PDF (density)
    sorted_indices = np.argsort(y)[::-1]
    # Accumulate area until we reach the desired probability
    area = 0.0
    included_indices = []
    for idx in sorted_indices:
        area += y[idx] * dx
        included_indices.append(idx)
        if area >= p:
            break
    # Find bounds of the selected region
    interval_x = x[sorted(included_indices)]
    return interval_x[0], interval_x[-1]


comparison_data = pd.DataFrame({'future current' : float(1), 'future sim': float(1), 'sim -5%': float(1), 'sim +5%': float(1)}, index = futures_sim_pdf_data.index)

for i in futures_sim_pdf_data.index:
    date = i.strftime('%Y-%m-%d')
    x = linspace_data.loc[date].values.astype(float)
    y = kde_pdf_data.loc[date].values.astype(float)

    current = futures_DA_df.iloc[futures_DA_df.index == date,0].iloc[0]
    max_like = x[np.argmax(y)]
    a, b = get_mode_centered_interval(x, y, p=0.05)

    comparison_data.loc[i,'future current'] = current
    comparison_data.loc[i, 'future sim'] = max_like
    comparison_data.loc[i, 'sim -5%'] = a
    comparison_data.loc[i, 'sim +5%'] = b

comparison_rolling = comparison_data.copy()
comparison_rolling['future sim'] = comparison_rolling['future sim'].rolling(window = 7).mean()
comparison_rolling['sim -5%'] = comparison_rolling['sim -5%'].rolling(window = 7).mean()
comparison_rolling['sim +5%'] = comparison_rolling['sim +5%'].rolling(window = 7).mean()
comparison_rolling = comparison_rolling.dropna()

r2_score(comparison_data['future current'], comparison_data['future sim'])
r2_score(comparison_rolling['future current'], comparison_rolling['future sim'])

plt.figure()
plt.plot(comparison_rolling.iloc[:,0:2], label =comparison_rolling.columns[0:2] )
plt.plot(comparison_rolling.iloc[:,2:4], label =comparison_rolling.columns[2:4], alpha = 0.2 )
plt.legend()
plt.show()



date = '2022-02-05'
plt.figure()
x = linspace_data.iloc[linspace_data.index == date,:].iloc[0]
y = kde_pdf_data.iloc[kde_pdf_data.index == date,:].iloc[0]
target = futures_DA_df.iloc[futures_DA_df.index == date,0].iloc[0]

interp_pdf = interp1d(x, y, kind='linear', fill_value="extrapolate")
y_value = interp_pdf(target)

plt.plot(x, y)
plt.scatter(x.iloc[np.argmax(y)], max(y), color = 'red', label = 'MLV')
plt.scatter(target, y_value, color = 'green', label ='Current price')
plt.legend()

plt.axvline(comparison_data['sim -5%'][comparison_data.index == date].iloc[0] , color = 'orange', linestyle = '--', alpha = 0.3)
plt.axvline(comparison_data['sim +5%'][comparison_data.index == date].iloc[0] , color = 'orange', linestyle = '--', alpha = 0.3)




plt.text(target + 2, y.median(), f'futures price: {target:.2f}')
plt.text(x.iloc[np.argmax(y)], max(y), f'Max likelihood value: {x.iloc[np.argmax(y)]:.2f}')
plt.axvline(target, color = 'red')