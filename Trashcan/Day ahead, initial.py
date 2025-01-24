import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import statsmodels.api as sm
from statsmodels.api import OLS
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from scipy.optimize import minimize

limiter = input("Do you want to exclude 2022 from seasonality analysis? (y/n): ").lower() == 'y'

# Initialize the data file
data_file = r'/Data/Day ahead/20241209 - Electricity day ahead prices, data avg.csv'

# Set up a dataframe that contains the prices and the delivery date including the hour
df = pd.read_csv(data_file)
df['Date delivery'] = pd.to_datetime(df['Date delivery'], format= '%d/%m/%Y')



# Create new dataframe with selected columns and additional time features
# Create base columns
base_columns = {
    'Date delivery': df['Date delivery'],
    'price': df['Price'],
    'month': df['Date delivery'].dt.month,
    'weekday': df['Date delivery'].dt.weekday,
}

# Add month indicator columns
month_columns = {
    f'month_{i}': df['Date delivery'].dt.month == i    for i in range(1, 13)
}

# Add month indicator columns
weekday_columns = {
    f'weekday_{i+1}': df['Date delivery'].dt.weekday == i
    for i in range(0, 7)
}

# Combine all columns and create dataframe
new_df = pd.DataFrame({
    **base_columns,
    **month_columns,
    **weekday_columns
}).reset_index()

# Remove rows with 'Date delivery' values in 2019 and 2022
new_df = new_df.reset_index()
new_df = new_df.iloc[:,2:].copy()

temp_df = pd.DataFrame(new_df['price'])
pt = PowerTransformer(method = 'yeo-johnson')
temp_df_YJ = pt.fit_transform(temp_df)
new_df['price'] = pd.DataFrame(temp_df_YJ)

if limiter == 1:
    new_df = new_df[(new_df['Date delivery'].dt.year != 2019) & (new_df['Date delivery'].dt.year != 2022)]

# Display first few rows of the new dataframe
print(new_df.info())


# Create a copy of the price data and subtract seasonal components
adjusted_prices = new_df['price'].copy()

##### Monthly seasonality
# Perform one-way ANOVA test for monthly seasonality

# Initialize list to store months with significant seasonality
significant_months = []

# Initialize dictionary to store F-statistics and p-values
seasonality_stats = {}

# Record months with significant seasonality
for month in range(1, 13):
    # Split data into two groups: current month vs other months
    month_group = new_df[new_df[f'month_{month}']]['price'].values
    other_months_group = new_df[~new_df[f'month_{month}']]['price'].values

    # Perform one-way ANOVA test
    f_stat_monthly, p_value_monthly = stats.f_oneway(month_group, other_months_group)

    # Store statistics
    seasonality_stats[month] = {
        'f_stat': f_stat_monthly,
        'p_value': p_value_monthly
    }

    # Record if statistically significant
    if p_value_monthly < 0.05:
        significant_months.append(month)

print("\nMonths showing significant seasonality:", significant_months)

# Calculate seasonal dummy parameters using linear regression
X = pd.concat([new_df[f'month_{m}'] for m in significant_months], axis=1)
y = new_df['price']

# Fit regression model
monthly_model = sm.OLS(y, X).fit()

# Display regression results
print("\nSeasonal Dummy Variable Parameters:")
print(monthly_model.summary().tables[1])

# Subtract monthly seasonal component using F-statistic
monthly_adjustment = monthly_model.predict(X)
new_df['monthly adjustments'] = monthly_adjustment
adjusted_prices -= monthly_adjustment

#Add the adjusted prices into the new_df
new_df['S_adjusted_prices'] = pd.DataFrame(adjusted_prices)

##### Weekly seasonality
# Initialize list to store months with significant seasonality
significant_weekdays = []

# Initialize dictionary to store F-statistics and p-values
seasonality_stats = {}

# Record weekdays with significant seasonality
for weekday in range(1, 8):
    # Split data into two groups: current month vs other months
    weekday_group = new_df[new_df[f'weekday_{weekday}']]['S_adjusted_prices'].values
    other_weekday_group = new_df[~new_df[f'weekday_{weekday}']]['S_adjusted_prices'].values

    # Perform one-way ANOVA test
    f_stat_monthly, p_value_monthly = stats.f_oneway(weekday_group, other_weekday_group)

    # Store statistics
    seasonality_stats[weekday] = {
        'f_stat': f_stat_monthly,
        'p_value': p_value_monthly
    }

    # Record if statistically significant
    if p_value_monthly < 0.05:
        significant_weekdays.append(weekday)

print("\nWeekdays showing significant seasonality:", significant_weekdays)

# Calculate seasonal dummy parameters using linear regression
X = pd.concat([new_df[f'weekday_{m}'] for m in significant_weekdays], axis=1)
y = new_df['S_adjusted_prices']

# Fit regression model
weekly_model = sm.OLS(y, X).fit()

# Display regression results
print("\nSeasonal Weekly Dummy Variable Parameters:")
print(weekly_model.summary().tables[1])

# Subtract monthly seasonal component using F-statistic
M_adjusted_prices = new_df['S_adjusted_prices'].copy()

weekly_adjustment = weekly_model.predict(X)
new_df['weekly adjustments'] = weekly_adjustment
W_M_adjusted_prices = M_adjusted_prices -weekly_adjustment
new_df['S_adjusted_prices'] = pd.DataFrame(W_M_adjusted_prices)
new_df['Date delivery'] = pd.to_datetime(new_df['Date delivery'])

if limiter == 1:
    temp_df_m = pd.DataFrame()
    temp_df_m['price'] = pd.DataFrame(df['Price'])
    temp_df_m = pd.DataFrame(pt.fit_transform(temp_df_m[['price']]))
    temp_df_m['Date delivery'] = pd.DataFrame(df['Date delivery'])
    temp_df_m['price'] = pd.DataFrame(temp_df_m.iloc[:,0])
    temp_df_m = temp_df_m.drop(0, axis=1)
    #temp_df_m = temp_df_m[temp_df_m['Date delivery'].dt.year != 2019]
    merged_df = pd.merge(new_df, temp_df_m, on='Date delivery', how = 'outer')
    merged_df['price'] = merged_df['price_x'].combine_first(merged_df['price_y'])
    merged_df = merged_df.drop(columns = ['price_x', 'price_y'])
    merged_df['month'] = merged_df['Date delivery'].dt.month
    merged_df['weekday'] = merged_df['Date delivery'].dt.weekday
    for i in range (1,13):
        merged_df[f'month_{i}'] = merged_df['Date delivery'].dt.month == i
    for i in range (1,8):
        merged_df[f'weekday_{i}'] = merged_df['Date delivery'].dt.weekday == i

    merged_df['weekly adjustments'] = weekly_model.predict(pd.concat([merged_df[f'weekday_{m}'] for m in significant_weekdays], axis=1))
    merged_df['monthly adjustments'] = monthly_model.predict(pd.concat([merged_df[f'month_{m}'] for m in significant_months], axis=1))
    merged_df['S_adjusted_prices'] = merged_df['price'] - merged_df['weekly adjustments'] - merged_df['monthly adjustments']

    new_df = merged_df

new_df = new_df.set_index(new_df['Date delivery'])
new_df = new_df.sort_index(ascending=True)  # Explicitly sort ascending
new_df = new_df.asfreq('D')


#ARIMA component

#ARIMA model

if input("Do you want to recalibrate ARIMA? (y/n): ").lower() == 'y':
    best_aic = float('inf')
    best_order = None
    best_model = None

    for p in range(5):
        for d in range(5):
            for q in range(5):
                try:
                    model = sm.tsa.ARIMA(new_df['S_adjusted_prices'],
                                         order=(p, d, q),
                                         freq='D',
                                         enforce_invertibility=True,
                                         enforce_stationarity=True)
                    results = model.fit()
                    aic = results.aic
                    if best_aic is None:
                        best_aic = aic
                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p, d, q)
                        best_model = results

                    print(f"ARIMA({p},{d},{q}) AIC: {aic}")

                except Exception as e:
                    print(f"Failed for ARIMA({p},{d},{q}): {str(e)}")
                    continue

    if best_model is not None:
        print("\nBest ARIMA order based on AIC:", best_order)
        print("Best AIC value:", best_aic)
        print("\nBest model parameters:")
        print(best_model.summary())
else:
    if limiter == 1:
        best_order = (2,1,4)
    else:
        best_order = (1,1,3) #1,1,3 if not excluding 2022

    best_model = sm.tsa.ARIMA(new_df['S_adjusted_prices'],
                                         order=best_order,
                                         freq='D',
                                         enforce_invertibility=True,
                                         enforce_stationarity=True)
    best_model = best_model.fit()
    best_aic = best_model.aic

    print("\nBest ARIMA order based on AIC:", best_order)
    print("Best AIC value:", best_aic)

# Store fitted prices of the best ARIMA model in a new column in new_df
new_df['ARIMA_fitted'] = best_model.fittedvalues
new_df['SARIMA_fitted'] = new_df['ARIMA_fitted'] + new_df['monthly adjustments'] + new_df['weekly adjustments']

new_df = new_df.iloc[1:] #drop the first row due to ARIMA fit

new_df['SARIMA_residuals'] = new_df['price'] - new_df['SARIMA_fitted']
new_df['original_price'] = pt.inverse_transform(new_df[['price']])
temp_df = pd.DataFrame(new_df['SARIMA_fitted'])
temp_df.rename(columns={'SARIMA_fitted' : 'price'}, inplace = True)
new_df['trans_SARIMA_fitted'] = pt.inverse_transform(temp_df[['price']])
new_df['trans_residuals'] = new_df['original_price'] - new_df['trans_SARIMA_fitted']
new_df.to_csv('SVJ_data_SARIMA.csv', index=True)
print(r2_score(new_df['trans_SARIMA_fitted'], new_df['original_price']))
print(f"p-value of stationarity test of residuals: {adfuller(new_df['trans_residuals'])[1]:.4f}")



#LEVY JUMPS + time varying volatility -> see chat GPT recommendations




