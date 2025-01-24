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


full_data = pd.read_csv(r'/Trashcan/SVJ_data_SARIMA.csv')
full_data['Date delivery'] = pd.to_datetime(full_data['Date delivery'])
full_data = full_data.set_index('Date delivery').asfreq('D')
methods = {'base', 'monthly', 'weekly', 'monthly + weekly'}
monthly_data = full_data['price'].resample('ME').mean()
monthly_data =pd.DataFrame(monthly_data)

ARIMA_data = pd.DataFrame()
ARIMA_data['base'] = full_data['price']
ARIMA_data['monthly'] = full_data['price'] - full_data['monthly adjustments']
ARIMA_data['weekly'] = full_data['price'] - full_data['weekly adjustments']
ARIMA_data['monthly + weekly'] = full_data['price'] - full_data['monthly adjustments'] - full_data['weekly adjustments']

for i in range(0, len(methods)):
    best_aic = float('inf')
    best_order = None
    best_model = None

    for p in range(5):
        for d in range(5):
            for q in range(5):
                try:
                    model = sm.tsa.SARIMAX(ARIMA_data.iloc[:, i],
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
                except Exception as e:
                    # Ignore errors in the run window while looping
                    print(e)
                    continue
    locals()[f'best_model_{i}'] = best_model
    print(f"\nBest {ARIMA_data.columns[i]} ARIMA order based on AIC:", best_order)
    print("Best AIC value:", best_aic)
    print(r2_score(best_model.fittedvalues, ARIMA_data.iloc[:, 0]))

r2_data = pd.DataFrame()
r2_data['base'] = best_model_0.fittedvalues
r2_data['monthly'] =best_model_1.fittedvalues + full_data['monthly adjustments']
r2_data['weekly'] =best_model_2.fittedvalues + full_data['weekly adjustments']
r2_data['monthly + weekly'] =best_model_3.fittedvalues + full_data['monthly adjustments'] + full_data['weekly adjustments']

for i in range(0, len(methods)):
    print(ARIMA_data.columns[i],locals()[f'best_model_{i}'].model.order)
    print(r2_score(r2_data.iloc[:, i], ARIMA_data.iloc[:, 0]))
    plt.plot(r2_data.iloc[:, i])
plt.plot(ARIMA_data.iloc[:, 0], label='Price Data')


for i in range(0, len(methods)):
    forecast_steps = 360
    forecast = locals()[f'best_model_{i}'].forecast(steps=forecast_steps)
    plt.plot(forecast, label=ARIMA_data.columns[i])
plt.plot(ARIMA_data.iloc[:, 0], label='Price Data')
plt.legend()
plt.show()
