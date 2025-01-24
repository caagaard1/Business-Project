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

# Initialize the data file
data_file = r'C:\Users\chrsr\PycharmProjects\Business_Project\Data\Day ahead\20250121 - Electricity day ahead prices, avg.csv'

# Set up a dataframe that contains the prices and the delivery date including the hour
df = pd.read_csv(data_file)
df['Date delivery'] = pd.to_datetime(df['Date delivery'], format= '%d/%m/%Y')
df = df.rename(columns ={'Price': 'price'})
df = df.set_index(df['Date delivery']).asfreq('D').dropna()

if input("Do you want to recalibrate ARIMA? (y/n): ").lower() == 'y':
    best_aic = float('inf')
    best_order = None
    best_model = None

    for p in range(5):
        for d in range(5):
            for q in range(5):
                try:
                    model = sm.tsa.ARIMA(df['price'],
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
    best_order = (2,1,4)
    best_model = sm.tsa.ARIMA(df['price'],
                                         order=best_order,
                                         freq='D',
                                         enforce_invertibility=True,
                                         enforce_stationarity=True)
    best_model = best_model.fit()
    best_aic = best_model.aic
    print(best_model.summary())

    print("\nBest ARIMA order based on AIC:", best_order)
    print("Best AIC value:", best_aic)

# Store fitted prices of the best ARIMA model in a new column in df
df['ARIMA_fitted'] = best_model.fittedvalues

df = df.dropna() #drop the first row due to ARIMA fit
print(best_model.forecast(steps=30))
ARIMA_parameters = pd.DataFrame(best_model.params)

df['ARIMA_residuals'] = df['price'] - df['ARIMA_fitted']
df.to_csv('SVJ_data_ARIMA_clean.csv', index=True)
ARIMA_parameters.to_csv('ARIMA_model_params.csv', index=True, header=[f'ARIMA: {best_order}'])
print(r2_score(df['ARIMA_fitted'], df['price']))
print(f"p-value of stationarity test of residuals: {adfuller(df['ARIMA_residuals'])[1]:.4f}")




