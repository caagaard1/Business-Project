# Create histogram of residuals
import pandas as pd

plt.figure(figsize=(10, 6))
plt.subplot(2,1,1)
plt.hist(new_df['AR_adjusted_prices'], bins=50, edgecolor='black')
plt.title('Histogram of AR Adjusted Prices (Residuals)')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.grid(True)

# Create Q-Q plot
plt.subplot(2,1,2)
stats.probplot(new_df['AR_adjusted_prices'].values, dist="norm", plot=plt)
plt.title('Q-Q Plot of AR Adjusted Prices (Residuals)')
plt.grid(True)
plt.tight_layout()
plt.show()


# Create a figure with two subplots
plt.figure(figsize=(12, 8))

# Plot original prices
plt.subplot(2, 1, 1)
plt.plot(new_df.index, new_df['original_price'], label='Original Prices')
plt.title('Model fit')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)

# Plot original prices
plt.subplot(2, 1, 1)
plt.plot(new_df.index, new_df['trans_SARIMA_fitted'], label='Fitted prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)

# Plot residuals (AR adjusted prices)
plt.subplot(2, 1, 2)
plt.plot(new_df.index, new_df['trans_residuals'], label='Residuals', color='orange')
plt.title('Residuals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()



#PACF tests

# Fit AR model for lags 1-4 using OLS regression
X_ar = pd.DataFrame()
y_ar = new_df['S_adjusted_prices'].iloc[4:]  # Start from 5th observation due to lags

# Create lagged variables
for lag in range(1, 5):
    X_ar[f'lag_{lag}'] = new_df['S_adjusted_prices'].shift(lag).iloc[4:]

# Fit AR model
ar_model = sm.OLS(y_ar, X_ar).fit()

# Display regression results
print("\nAutoregressive Model Parameters (Lags 1-4):")
print(ar_model.summary().tables[1])

# Calculate AR adjusted prices
ar_predictions = ar_model.predict(X_ar)
ar_adjusted_prices = y_ar - ar_predictions

# Update the main dataframe with AR adjusted prices
new_df['AR_adjusted_prices'] = pd.NA  # Initialize with NA
new_df.loc[new_df.index[4:], 'AR_adjusted_prices'] = ar_adjusted_prices
new_df = new_df.dropna()
new_df['AR_adjusted_prices'] = new_df['AR_adjusted_prices'].astype('float64')


# Loop through different combinations of p, d, and q
best_aic = float('inf')
best_order = None
best_model = None
new_df = full_data
new_df['Date delivery'] = pd.to_datetime(new_df['Date delivery'])
new_df = new_df.set_index('Date delivery')
new_df = new_df.sort_index(ascending = True)
new_df = new_df.asfreq('D')

best_aic = float('inf')
best_order = None
best_model = None

for p in range(5):
    for d in range(5):
        for q in range(5):
            try:
                model = sm.tsa.ARIMA(new_df['price'],
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

#Initialize the transformation used on prices
new_df['Date delivery'] = pd.to_datetime(new_df['Date delivery'])
new_df = new_df.set_index('Date delivery').asfreq('D')
temp_df = pd.DataFrame(new_df['original_price'])
pt = PowerTransformer(method = 'yeo-johnson')
temp_df_YJ = pt.fit_transform(temp_df)

temp_df = pd.DataFrame()
temp_df['original_price'] = pd.DataFrame(best_model.fittedvalues)
temp_df['fitted_trans'] = pt.inverse_transform(temp_df[['original_price']])
temp_df.rename(columns={'original_price' : 'fitted_nontrans'}, inplace = True)
temp_df['original_price'] = new_df['original_price']
temp_df.iloc[0,1] = temp_df.iloc[0,2] #adjustment for first value


plt.figure(figsize=(12,8))
plt.plot(new_df.index, new_df['trans_residuals'].rolling(window=2).std(), label='2-day')
plt.plot(new_df.index, new_df['trans_residuals'].rolling(window=30).std(), label='30-day')
plt.plot(new_df.index, new_df['trans_residuals'].rolling(window=2).std().rolling(window=7).std(), label='7-day std of 2-day std')
plt.legend()
plt.show()

vol_data = {'30-day std' : new_df['trans_residuals'].rolling(window=30).std(),
            '2-day std': new_df['trans_residuals'].rolling(window=2).std(),
            '14-day of 2-day' : new_df['trans_residuals'].rolling(window=2).std().rolling(window=14).std()}
vol_df = pd.DataFrame(vol_data)

for i in range(10):
    vol_df[f'2-day lag_{i}'] = vol_df['2-day std'].shift(i)
for i in range(10):
    vol_df[f'14-day of 2-day lag_{i}'] = vol_df['14-day of 2-day'].shift(i)

vol_df.dropna()

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
# Heatmap for covariance matrix
sns.heatmap(vol_df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation  Matrix')
plt.show()


# Create a figure with two subplots
plt.figure(figsize=(12, 8))

# Plot original prices
plt.subplot(2, 1, 1)
plt.plot(data.index, data, label='Modelled residuals')
plt.title('Model fit')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.legend()
plt.grid(True)

# Plot residuals (AR adjusted prices)
plt.subplot(2, 1, 2)
plt.plot(rolling_volatility.index, rolling_volatility, label='Rolling vol', color='orange')
plt.title('Volatility')
plt.xlabel('Date')
plt.ylabel('Rolling volatility')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
