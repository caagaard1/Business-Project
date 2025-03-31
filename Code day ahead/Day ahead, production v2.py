import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load and preprocess data, sets up the dataframe so that it is easily callable
day_ahead_df_full = pd.read_csv(r'C:\Users\chrsr\Business_Project_GitHub\Data\Day ahead\20250121 - Electricity day ahead prices, avg.csv')
day_ahead_df_full['Date delivery'] = pd.to_datetime(day_ahead_df_full['Date delivery'], format = '%d/%m/%Y')
day_ahead_df_full = day_ahead_df_full.set_index('Date delivery')[['Price']].asfreq('D').dropna()
day_ahead_df_full = day_ahead_df_full[day_ahead_df_full.index.year < 2025].dropna()

#Ensure all data is positive
pos_constant = abs(int(day_ahead_df_full.min().iloc[0])) + 1
day_ahead_df_full += pos_constant

#Calculate log-returns
day_ahead_df_log = pd.DataFrame(np.log(day_ahead_df_full / day_ahead_df_full.shift(1))).dropna()
day_ahead_df_log.rename(columns = {'Price' : 'Log_return'}, inplace = True)

#Calculate volatility
day_ahead_df_log['rolling volatility'] = day_ahead_df_log['Log_return'].shift(1).rolling(window=7).std()
day_ahead_df_log['rolling volatility LT'] = day_ahead_df_log['rolling volatility'].rolling(window=30).mean()


futures_df_full = pd.read_excel(r'C:\Users\chrsr\Business_Project_GitHub\Data\futures vs. dayahaead correlation.xlsx', sheet_name=1)
futures_df_full = futures_df_full.set_index(pd.to_datetime(futures_df_full['Dates'], format = '%d/%m/%Y')).drop(columns = {'Dates'}).dropna()
futures_df_full = futures_df_full.apply(pd.to_numeric, errors = 'coerce')
futures_df = futures_df_full['M+1'].copy()
futures_df.rename('futures_price', inplace = True)

full_data = day_ahead_df_log.join(futures_df)
full_data = full_data.iloc[full_data.index < max(futures_df.index)]

#Recover calibrated parameters
with open(r'C:\Users\chrsr\Business_Project_GitHub\calibrated_params_log v6.0.txt', 'r') as file:
    calibrated_params = [float(item) for item in file.read().split(',')]

# Simulate an SVJ process with mean-reverting volatility
def svj_simulation(calibrated_params, ST_vol, LT_vol, duration, seed=None):


    if seed is not None:
        np.random.seed(seed)
    T = duration
    dt = 1 / 365
    n_steps = duration
    times = np.linspace(0, T-1, n_steps)

    sigma_0, lambda_0, jump_mean_pos, jump_std_pos, jump_mean_neg, jump_std_neg, kappa, theta, eta, alpha, beta_pos, beta_neg, gamma_pos, gamma_neg, prob_positive = calibrated_params

    sigma_0 = ST_vol
    theta = LT_vol ** 2

    # Initialize price, volatility, and variance
    S = []
    sigma = []
    variance = sigma_0**2  # Initial variance
    J_t_data = []
    variance_data = []
    dW_v_data = []
    N_t_data = []
    dS_data = []


    for t in range(1, n_steps):

        # Mean-reverting variance process
        dW_v = np.random.normal(0, np.sqrt(dt))  # Brownian increment for volatility
        variance = max(
            0,  # Ensure variance is non-negative
            variance + kappa * (theta - variance) * dt + eta * np.sqrt(max(variance, 0)) * dW_v
        )
        sigma_t = np.sqrt(variance)  # Convert variance to standard deviation

        # Update jump intensity and size based on volatility
        lambda_t = lambda_0 + alpha * sigma_t
        N_t = np.random.poisson(lambda_t * dt)  # Number of jumps in this interval
        J_t = 0

        for _ in range(N_t):
            if np.random.rand() < prob_positive:  # Positive jump
                J_t += np.random.normal(
                    jump_mean_pos + beta_pos * sigma_t * dt,
                    jump_std_pos + gamma_pos * sigma_t * dt
                )
            else:  # Negative jump
                J_t += np.random.normal(
                    jump_mean_neg + beta_neg * sigma_t * dt,
                    jump_std_neg + gamma_neg * sigma_t * dt
                )

        # Brownian motion for the price
        dW = np.random.normal(0, np.sqrt(dt))

        # Update price using random walk with time-varying volatility and jumps
        dS = sigma_t * dW + J_t
        S.append(dS)
        sigma.append(sigma_t)

        dS_data.append(dS)
        J_t_data.append(J_t)
        variance_data.append(variance)
        N_t_data.append(N_t)
        dW_v_data.append(dW_v)

    debug_data = pd.DataFrame({
        'Price change': dS_data,
        'Jump size': J_t_data,
        'Variance': variance_data,
        'Jumps in interval': N_t_data,
        'Brownian time element': dW_v_data
    })

    return np.array(S), np.array(sigma), times, debug_data, seed

def calculate_futures_log_return():
    simulations_per_day = 100
    simulation_duration = 60 #days
    start_date = '2018-01-03'
    end_date = '2024-11-29'

    historical_futures_pdf_data = pd.DataFrame()
    historical_DA_est_vol = pd.DataFrame()
    simulation_days = pd.date_range(start=pd.to_datetime(start_date), end=pd.to_datetime(end_date))
    simulation_days = [d.strftime('%Y-%m-%d') for d in simulation_days]
    simulation_data = pd.DataFrame()

    for i in simulation_days:
        print(i)

        futures_price_ref = full_data['futures_price'].shift(1)[full_data.index == i].iloc[0]
        ST_vol = full_data['rolling volatility'][full_data.index == i].iloc[0]
        LT_vol = full_data['rolling volatility LT'][full_data.index == i].iloc[0]

        new_row_sim = pd.DataFrame()
        for x in range(1,simulations_per_day +1):
            simulated_log_returns, simulated_volatility, time_points, debug_data, seed = svj_simulation(calibrated_params, ST_vol, LT_vol, simulation_duration, seed=None)

            average_month = (pd.to_datetime(i) + pd.offsets.MonthEnd(2)).month
            super_cap = 1000
            #Calculations for output average value
            output_start_date = pd.to_datetime(i) + pd.Timedelta(days = 1)
            sim_log_df = pd.DataFrame({'sim_log_return': simulated_log_returns}, index=pd.date_range(start = output_start_date , periods = simulation_duration -1, freq ='D'))
            cum_sim_log_df = sim_log_df.cumsum()

            sim_log_df = sim_log_df[sim_log_df.index.month == average_month].dropna()
            cum_sim_log_df = sim_log_df.cumsum()
            futures_price_ref_temp = futures_price_ref
            futures_price_dev = futures_price_ref_temp * np.exp(cum_sim_log_df)

            futures_price_dev = futures_price_dev.clip(lower=-abs(super_cap * 0.5), upper=abs(super_cap))

            # Calculate average futures value for simulation

            futures_price_sim_avg = futures_price_dev.mean()

            new_row_sim[f'sim {x}'] = futures_price_sim_avg



        new_row_sim.index = [i]
        simulation_data = pd.concat([simulation_data, new_row_sim])
        print(simulation_data.isna().sum().sum() == 0)
        simulation_data.to_csv(f'historical_futures_sim_data_{start_date} v5.csv', index=True)

