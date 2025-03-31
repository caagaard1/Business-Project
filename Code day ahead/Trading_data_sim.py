import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d

#Select correct file for parameters
with open(r'C:\Users\chrsr\Business_Project_GitHub\calibrated_params_log v6.0.txt', 'r') as file:
    model_parameters = [float(item) for item in file.read().split(',')]

#Import dataframe with day-ahead and futures prices
data_df = pd.read_excel(r'C:\Users\chrsr\Business_Project_GitHub\Data\trading data for model.xlsx', sheet_name = 0)
data_df = data_df.set_index(data_df['Dates']).drop(columns = {'Dates'})
start_date = data_df.shift(1).dropna().index.min()
end_date = data_df.dropna().index.max()

data_df['DA_log'] = np.log(data_df['Day ahead avg'] / data_df['Day ahead avg'].shift(1) )

super_cap = 1000

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
    simulation_days = pd.date_range(start=pd.to_datetime(start_date), end=pd.to_datetime(end_date))
    simulation_days = [d.strftime('%Y-%m-%d') for d in simulation_days]
    simulation_data = pd.DataFrame()

    for i in simulation_days:
        print(i)

        futures_price_ref = data_df['Future M+1'].shift(1)[data_df.index == i].iloc[0]
        ST_vol = data_df['DA_log'].rolling(window = 7).std()[data_df.index == i].iloc[0]
        LT_vol = data_df['DA_log'].rolling(window = 7).std().rolling(window=30).mean()[data_df.index == i].iloc[0]

        new_row_sim = pd.DataFrame()
        for x in range(1,simulations_per_day +1):
            simulated_log_returns, simulated_volatility, time_points, debug_data, seed = svj_simulation(model_parameters, ST_vol, LT_vol, simulation_duration, seed=None)

            average_month = (pd.to_datetime(i) + pd.offsets.MonthEnd(2)).month
            #Calculations for output average value
            output_start_date = pd.to_datetime(i) + pd.Timedelta(days = 1)
            sim_log_df = pd.DataFrame({'sim_log_return': simulated_log_returns}, index=pd.date_range(start = output_start_date , periods = simulation_duration -1, freq ='D'))
            sim_log_df = sim_log_df[sim_log_df.index.month == average_month].dropna()
            cum_sim_log_df = sim_log_df.cumsum()
            futures_price_ref_temp = futures_price_ref
            futures_price_dev = futures_price_ref_temp * np.exp(cum_sim_log_df)

            futures_price_dev = futures_price_dev.clip(lower=-abs(super_cap * 0.5), upper=abs(super_cap))

            #Calculate average futures value for simulation

            futures_price_sim_avg = futures_price_dev.mean()

            new_row_sim[f'sim {x}'] = futures_price_sim_avg



        new_row_sim.index = [i]
        simulation_data = pd.concat([simulation_data, new_row_sim])
        print(simulation_data.isna().sum().sum() == 0)
        simulation_data.to_csv(f'trading_futures_sim_data_{start_date.strftime('%Y-%m-%d')} v1.csv', index=True)

    return simulation_data

data_sim = calculate_futures_log_return()

def get_kde(data_sim):
    futures_sim_pdf_data = data_sim
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
    return linspace_data, kde_pdf_data

linspace_data, kde_pdf_data = get_kde(data_sim)

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

def calculate_kde_intervals(futures_sim_pdf_data, linspace_data, kde_pdf_date):
    comparison_data = pd.DataFrame(
        {'future current': float(1), 'future sim': float(1), 'sim -5%': float(1), 'sim +5%': float(1)},
        index=futures_sim_pdf_data.index)

    for i in futures_sim_pdf_data.index:
        date = i
        x = linspace_data.loc[date].values.astype(float)
        y = kde_pdf_data.loc[date].values.astype(float)

        current = data_df['Future M+1'].iloc[data_df.index == date].iloc[0]
        max_like = x[np.argmax(y)]
        a, b = get_mode_centered_interval(x, y, p=0.05)

        comparison_data.loc[i, 'future current'] = current
        comparison_data.loc[i, 'future sim'] = max_like
        comparison_data.loc[i, 'sim -5%'] = a
        comparison_data.loc[i, 'sim +5%'] = b

    return comparison_data

comparison_data = calculate_kde_intervals(data_sim, linspace_data, kde_pdf_data)
comparison_data.to_csv(f'comparison data for trading v2.csv')