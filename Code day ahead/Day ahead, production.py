import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

#Step by step:
#1. date for forecast and simulation
#2. check if SARIMA required data is available, e.g., just put a stop at 2020 or something
#3. Create a dataframe containing the dates from t-30
#4. fill in prices, dates, weekday flag, month flag, rolling 7-day vol for all the values of t-30 to t.
#5. fill in dates, weekday flag, month flag for the next 30 days
#6. Forecast SARIMA price for t+1
#7. Calculate a jump based 7-day vol and starting price of SARIMA
#8. save price for date, calculate updated rolling vol value
#9. Forecast SARIMA price for t+2 based on #8 saved price and rolling vol
#

# Load and preprocess data, sets up the dataframe so that it is easily callable. The iloc function removes the first row to the ARIMA setup
full_data = pd.read_csv(r'SVJ_data_ARIMA_clean.csv')
full_data['Date delivery'] = pd.to_datetime(full_data['Date delivery'])
full_data = pd.DataFrame({
    'price': full_data['price'],
    'errors': full_data['ARIMA_residuals'],
    'rolling volatility' : full_data['price'].shift(1).rolling(window=7).std(),
    'rolling volatility LT' : full_data['price'].shift(1).rolling(window=7).std().rolling(window=30).mean()
    }).set_index(full_data['Date delivery']).iloc[1:]

#ARIMA parameters
ARIMA_parameters = pd.read_csv(r'ARIMA_model_params.csv')
ARIMA_parameters = ARIMA_parameters.set_index(ARIMA_parameters.columns[0])

#Stochastic time-varying volatility jump process parameters
with open('calibrated_params.txt', 'r') as file:
        calibrated_params = [float(item) for item in file.read().split(',')]


ar_params = []
ma_params = []
for key in ARIMA_parameters.index:
    if key.startswith('ar'):
        ar_params.append(float(ARIMA_parameters.loc[key].iloc[0]))
    if key.startswith('ma'):
        ma_params.append(float(ARIMA_parameters.loc[key].iloc[0]))

ar_steps = len(ar_params)
ma_steps = len(ma_params)

d = 1

def ARIMA_step(dataframe, ar_params, ma_params, d):
    """
    Predict the next price based on ARIMA parameters (with differencing).

    Parameters:
        dataframe (pd.DataFrame): DataFrame with columns ['price', 'errors'].
        ar_params (list): AR coefficients [phi1, phi2, ...].
        ma_params (list): MA coefficients [theta1, theta2, ...].
        d (int): Degree of differencing (default=1).

    Returns:
        float: Predicted price for the next period.
    """
    # Get the last observed prices and errors
    prices = dataframe['price'].values
    errors = dataframe['errors'].values

    # Calculate differenced prices if d > 0
    if d > 0:
        differenced_prices = pd.Series(prices).diff().dropna().values
    else:
        differenced_prices = prices

    # Initialize the prediction with zero (no constant term)
    predicted_diff = 0

    # Add AR terms (Auto-Regressive component)
    for i, phi in enumerate(ar_params):
        if i < len(differenced_prices):  # Ensure we don't go out of bounds
            predicted_diff += phi * differenced_prices[-(i + 1)]


    # Add MA terms (Moving Average component)
    for j, theta in enumerate(ma_params):
        if j < len(errors):  # Ensure we don't go out of bounds
            predicted_diff += theta * errors[-(j + 1)]


    # Reverse differencing to return to the original scale
    ARIMA_price = predicted_diff + prices[-1] if d > 0 else predicted_diff

    return ARIMA_price

def initialize_dataset(target_date):
    date_last = pd.to_datetime(target_date)
    date_first = date_last - pd.Timedelta(days=max(ar_steps, ma_steps, 8))
    price_data = full_data.loc[date_first:date_last]
    return price_data

def jump_step(ARIMA_value, price_dataframe, calibrated_params, sim_variance, LT_vol , seed = None):
    sigma_0, lambda_0, jump_mean_pos, jump_std_pos, jump_mean_neg, jump_std_neg, kappa, theta, eta, alpha, beta_pos, beta_neg, gamma_pos, gamma_neg, prob_positive = calibrated_params
    dt = 1/7
    theta = LT_vol **2

    sigma = price_dataframe['rolling volatility'].iloc[-1]
    if sim_variance is None:
        variance = sigma ** 2
    else:
        variance = sim_variance

    S = ARIMA_value

    #Mean-reverting variance process
    dW_v = np.random.normal(0, np.sqrt(dt)) # Brownian increment for volatility
    #print(f'variance + kappa * (theta - variance) * dt + eta * np.sqrt(max(variance, 0)) * dW_v \n {variance} + {kappa} * ({theta} - {variance}) * {dt} + {eta} * np.sqrt * {dW_v}')


    variance = max(
        0,  # Ensure variance is non-negative
        variance + kappa * (theta - variance) * dt + eta * np.sqrt(max(variance, 0)) * dW_v
    )
    sigma_t = np.sqrt(variance)

    #Jump intensity (frequency) as a function of volatility
    lambda_t = lambda_0 + alpha * sigma_t #initial jump intensity, regression on volatility, and variance after mean-reversion process
    N_t = np.random.poisson(lambda_t * dt) #number of jumps in time period (one day)

    #Jump calculation positive or negative determined by probability
    J_t = 0

    for _ in range(N_t):
            if np.random.rand() < prob_positive:  # Positive jump
                J_t += np.random.normal(
                    jump_mean_pos + beta_pos * sigma_t,
                    jump_std_pos + gamma_pos * sigma_t
                )
                J_t = min(J_t, 250)
            else:  # Negative jump
                J_t += np.random.normal(
                    jump_mean_neg + beta_neg * sigma_t,
                    jump_std_neg + gamma_neg * sigma_t
                )
                J_t = max(J_t, -250)
    #Brownian movement of variance inherent in jump
    dW = np.random.normal(0 , np.sqrt(dt))

    #Jump impact on price
    dS = J_t + sigma_t * dW
    jump_price = S + dS
    jump_size = dS
    sim_variance = variance
    #print(f'S: {S}, variance: {variance}, lambda_t: {lambda_t}, N_t: {N_t}, dS: {dS}')

    return(jump_price, jump_size, sim_variance)

def initialize_multistep(target_date, x = None):
    price_dataframe = initialize_dataset(target_date)
    sim_variance = None

    if x is not None:
        np.random.seed(x)
    LT_vol = price_dataframe['rolling volatility LT'].iloc[-1]

    for i in range(270):
        ARIMA_value =  ARIMA_step(price_dataframe, ar_params, ma_params, d)
        #ARIMA_value =  float(price_dataframe['price'].iloc[-1])
        jump_adj_price, jump_size, sim_variance = jump_step(ARIMA_value, price_dataframe, calibrated_params, sim_variance, LT_vol , seed = x)
        temp_row  = pd.DataFrame(
            {price_dataframe.columns[0] : [jump_adj_price],
                  price_dataframe.columns[1]: [jump_size],
             price_dataframe.columns[2] : price_dataframe['price'].iloc[-7:].std()}
            ,index = [price_dataframe.index.max() + pd.Timedelta(days = 1)])
        price_dataframe = pd.concat([price_dataframe, temp_row], ignore_index = False)
    return price_dataframe

def one_period_simulator(target_date):
    one_period_simulation = pd.DataFrame()
    one_period_simulation_vol = pd.DataFrame()
    for x in range(100):
        one_period_simulation[f'sim {x}'] = initialize_multistep(target_date)['price']
        one_period_simulation_vol[f'sim {x}'] = initialize_multistep(target_date)['rolling volatility']

    return (one_period_simulation, one_period_simulation_vol)

target_date = '2023-02-01'

one_period_simulation, one_period_simulation_vol = one_period_simulator(target_date)
plt.figure(figsize=(12,8))
plt.subplot(2,1,1)
plt.plot(one_period_simulation, linestyle = '--', alpha = 0.3)
plt.plot(one_period_simulation.mean(axis=1), color = 'red', label = f'Forecast mean: {one_period_simulation.mean(axis=1).mean()}')
plt.plot(full_data['price'].loc[one_period_simulation.index.min():one_period_simulation.index.max()], label = f'Actual mean: {full_data['price'].loc[one_period_simulation.index.min():one_period_simulation.index.max()].mean()}')
plt.axvline(x = pd.to_datetime(target_date), color = 'red', linestyle = '--')
plt.legend()
plt.subplot(2,1,2)
plt.plot(one_period_simulation_vol, linestyle = '--', alpha = 0.3)
plt.plot(one_period_simulation_vol.mean(axis=1), color = 'red', label = f'Forecast mean: {one_period_simulation_vol.mean(axis=1).mean()}')
plt.plot(full_data['rolling volatility'].loc[one_period_simulation_vol.index.min():one_period_simulation_vol.index.max()], label = f'Actual mean: {full_data['rolling volatility'].loc[one_period_simulation.index.min():one_period_simulation.index.max()].mean()}')
plt.axvline(x = pd.to_datetime(target_date), color = 'red', linestyle = '--')
plt.legend()
plt.show()

#We need a volatility dependent futures premium on 1) historical realized prices, 2) a probability density of what was forecasted at the time