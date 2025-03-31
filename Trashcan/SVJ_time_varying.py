import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

# Simulate an SVJ process with mean-reverting volatility
def svj_simulation(
    sigma_0, lambda_0, jump_mean_pos, jump_std_pos, jump_mean_neg, jump_std_neg, kappa, theta, eta, alpha, beta_pos, beta_neg, gamma_pos, gamma_neg, prob_positive, seed=None
):
    """
    Simulates an SVJ process with random-walk prices and mean-reverting volatility.

    Parameters:
        sigma_0: Initial volatility
        lambda_0: Baseline jump intensity
        jump_mean_pos: Mean size of positive jumps
        jump_std_pos: Std deviation of positive jumps
        jump_mean_neg: Mean size of negative jumps
        jump_std_neg: Std deviation of negative jumps
        T: Total time
        dt: Time step
        S0: Initial value of the process
        kappa: Speed of mean reversion for volatility
        theta: Long-term variance
        eta: Volatility of volatility
        alpha: Sensitivity of jump intensity to volatility
        beta_pos: Scaling factor for positive jump mean based on volatility
        beta_neg: Scaling factor for negative jump mean based on volatility
        gamma_pos: Scaling factor for positive jump std deviation based on volatility
        gamma_neg: Scaling factor for negative jump std deviation based on volatility
        prob_positive: Probability of a jump being positive
        seed: Random seed for reproducibility

    Returns:
        Simulated time series (S), volatility series (sigma), and corresponding time points.
    """
    if seed is not None:
        np.random.seed(seed)

    n_steps = int(T / dt)
    times = np.linspace(0, T, n_steps)

    # Initialize price, volatility, and variance
    S = [S0]
    sigma = [sigma_0]
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
                    jump_mean_pos + beta_pos * sigma_t,
                    jump_std_pos + gamma_pos * sigma_t
                )
            else:  # Negative jump
                J_t += np.random.normal(
                    jump_mean_neg + beta_neg * sigma_t,
                    jump_std_neg + gamma_neg * sigma_t
                )

        # Brownian motion for the price
        dW = np.random.normal(0, np.sqrt(dt))
        #print(f'variance + kappa * (theta - variance) * dt + eta * np.sqrt(max(variance, 0)) * dW_v \n {variance} + {kappa} * ({theta} - {variance}) * {dt} + {eta} * np.sqrt * {dW_v}')

        # Update price using random walk with time-varying volatility and jumps
        dS = sigma_t * dW + J_t
        S.append(S0 + dS)
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

# Load and preprocess data
full_data = pd.read_csv(r'../Code day ahead/SVJ_data_ARIMA_clean.csv')
full_data['Date delivery'] = pd.to_datetime(full_data['Date delivery'])
data = full_data.set_index('Date delivery')[['ARIMA_residuals']].asfreq('D').dropna()

# Total time and step size
T = len(data)  # Total time
dt = 1  # Daily intervals
S0 = 0  # Prices revolve around zero

# Threshold for identifying jumps
threshold = 50  # Threshold movement to be considered a jump

# Initial parameters
sigma_0 = data[data.abs() < threshold].std().iloc[0]  # Initial volatility excluding jumps
extreme_moves = (data.abs() > threshold).sum().iloc[0]  # Moves exceeding the threshold
lambda_0 = extreme_moves / len(data)  # Baseline jump intensity
jumps = data[np.abs(data) > threshold]  # Observations that are jumps

# Separate positive and negative jumps
positive_jumps = jumps[jumps['ARIMA_residuals'] > 0]
negative_jumps = jumps[jumps['ARIMA_residuals'] < 0]
prob_positive = len(positive_jumps) / len(jumps.dropna())  # Probability of a jump being positive

# Mean and std for positive and negative jumps
jump_mean_pos = positive_jumps.mean().iloc[0]
jump_std_pos = positive_jumps.std().iloc[0]
jump_mean_neg = negative_jumps.mean().iloc[0]
jump_std_neg = negative_jumps.std().iloc[0]

# Calculate rolling volatility
delta_P = data.abs().iloc[:, 0]  # Absolute values of price differences
rolling_volatility = delta_P.rolling(window=7).std().dropna()  # Rolling standard deviation (volatility)

# Fit an AR(1) process to the rolling volatility
lagged_volatility = rolling_volatility.shift(1).dropna()  # Lagged rolling volatility
aligned_volatility = rolling_volatility.loc[lagged_volatility.index]  # Align indices
lagged_volatility_param = sm.add_constant(lagged_volatility)  # Add constant for regression

# Perform AR(1) regression (Ornstein-Uhlenbeck process approximation)
OU_regression = sm.OLS(aligned_volatility, lagged_volatility_param).fit()

# Extract AR(1) parameters
phi = OU_regression.params.iloc[1]  # Slope (persistence of volatility)
if 0 < phi < 1:  # Ensure stationarity
    kappa = -np.log(phi)  # Speed of mean reversion
    theta = (OU_regression.params['const'] / (1 - phi)) ** 2 # Long-term variance level
else:
    raise ValueError("Phi is outside the range (0, 1). Recheck data or model assumptions.")

# Volatility of volatility (eta)
eta = OU_regression.resid.std()  # Standard deviation of residuals

# Regression for positive and negative jumps
beta_pos = sm.OLS(positive_jumps.abs(), sm.add_constant(rolling_volatility.loc[positive_jumps.index])).fit().params.iloc[1]
gamma_pos = sm.OLS(positive_jumps.abs(), sm.add_constant(rolling_volatility.loc[positive_jumps.index])).fit().bse.iloc[1]
beta_neg = -sm.OLS(negative_jumps.abs(), sm.add_constant(rolling_volatility.loc[negative_jumps.index])).fit().params.iloc[1]
gamma_neg = sm.OLS(negative_jumps.abs(), sm.add_constant(rolling_volatility.loc[negative_jumps.index])).fit().bse.iloc[1]

# Calculate jump intensity (lambda_t)
is_jump = (data['ARIMA_residuals'].abs() > threshold).astype(int)
lambda_t = is_jump.rolling(window=7).sum() / 7 # Average jumps per window

# Ensure aligned indices between lambda_t and rolling_volatility
aligned_data = pd.DataFrame({
    'lambda_t': lambda_t,
    'sigma_t': rolling_volatility
}).dropna()

# Perform regression of jump intensity on volatility

alpha = sm.OLS(aligned_data['lambda_t'], sm.add_constant(aligned_data['sigma_t'])).fit().params.iloc[1]

print(f"Estimated Parameters:")
print(f"Sigma_0 (Initial Volatility): {sigma_0}")
print(f"Lambda_0 (Baseline Jump Intensity): {lambda_0}")
print(f"Positive Jump Mean: {jump_mean_pos}")
print(f"Positive Jump Std Dev: {jump_std_pos}")
print(f"Negative Jump Mean: {jump_mean_neg}")
print(f"Negative Jump Std Dev: {jump_std_neg}")
print(f"Probability of Positive Jumps: {prob_positive}")
print(f"Kappa (Mean Reversion Speed): {kappa}")
print(f"Phi: {phi}")
print(f"Theta (Long-Term Variance): {theta}")
print(f"Eta (Volatility of Volatility): {eta}")
print(f"Alpha (Jump Intensity Sensitivity): {alpha}")
print(f"Beta_Pos (Positive Jump Mean Sensitivity): {beta_pos}")
print(f"Gamma_Pos (Positive Jump Std Dev Sensitivity): {gamma_pos}")
print(f"Beta_Neg (Negative Jump Mean Sensitivity): {beta_neg}")
print(f"Gamma_Neg (Negative Jump Std Dev Sensitivity): {gamma_neg}")


# Plot the results
plt.figure(figsize=(12, 6))
# Simulate the process
simulated_price, simulated_volatility, time_points, debug_data, seed = svj_simulation(
    sigma_0, lambda_0, jump_mean_pos, jump_std_pos, jump_mean_neg, jump_std_neg,
    kappa, theta, eta, alpha, beta_pos, beta_neg, gamma_pos, gamma_neg,
    prob_positive, seed=42
)

# Plot simulated prices
plt.subplot(2, 1, 1)
plt.plot(time_points, simulated_price, label=f"Simulated Price")
plt.title("Pre-calibration SVJ Process with Mean-Reverting Volatility and Random Walk Prices")
plt.ylabel("Price")
plt.grid()
plt.legend()

# Plot simulated volatility
plt.subplot(2, 1, 2)
plt.plot(time_points, pd.Series(simulated_volatility).rolling(window = 7).mean(), label=f"Simulated Volatility 7-day MVA")
plt.ylabel("Volatility")
plt.xlabel("Time")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
print(debug_data.iloc[0:50,:])


#Calibration based on the defined loss function of CHATGPT, however, be aware that many of the models parameters are already based upon the dataset, CROSS REFERENCE WITH OTHER RANDOM SEEDS
#Items for the loss function: 1) positive and negative jump mean, 2) number of jumps, 3) jump neg and pos std, 4) absolute value jump mean, 5) absolute value jump std, 6) correlation between rolling vol and jumps,
#7) mean rolling vol, 8) rolling vol standard deviation, 9) min and max values for positive and negative jumps

#Data sets for metric comparison
his_prices = pd.DataFrame({'his_prices': data['ARIMA_residuals']})
his_volatility = pd.DataFrame({'his_volatility': data['ARIMA_residuals'].rolling(window=7).std().dropna()})

#Historical metrics

#1) Jump mean
his_pos_jump_mean = his_prices[his_prices > threshold].mean()
his_neg_jump_mean = his_prices[his_prices < -threshold].mean()

#2) Number of jumps
his_pos_jump_num = his_prices[his_prices > threshold].count()
his_neg_jump_num = his_prices[his_prices < -threshold].count()

#3) Jump neg and pos std
his_pos_jump_std = his_prices[his_prices > threshold].std()
his_neg_jump_std = his_prices[his_prices < -threshold].std()

#4) Absolute jump mean
his_abs_jump_mean = his_prices.abs()[his_prices.abs() > threshold].mean()

#5) Absolute value jump std
his_abs_jump_std = his_prices.abs()[his_prices.abs() > threshold].std()

#6) Correlation between lagged rolling vol and jumps size and frequency
x_correl_vol_size = his_volatility.shift(1)[his_volatility.index.isin(his_prices.abs()[his_prices.abs() > threshold].dropna().index)]
y_correl_vol_size = his_prices.abs()[his_prices.abs() > threshold].dropna()

his_correl_vol_size = sm.OLS(y_correl_vol_size, x_correl_vol_size).fit().params.iloc[0]

x_correl_vol_freq = his_volatility.shift(1).dropna()
y_correl_vol_freq = (his_prices.abs() > threshold).astype(int)
y_correl_vol_freq = y_correl_vol_freq[y_correl_vol_freq.index.isin(x_correl_vol_freq.index)]
his_correl_vol_freq = sm.OLS(y_correl_vol_freq,x_correl_vol_freq).fit().params.iloc[0]

#7) Mean rolling vol
his_vol_mean = his_volatility.mean()

#8) Rolling vol standard deviation
his_vol_std = his_volatility.std()

#9 Min and max values for positive and negative jumps
his_pos_max_jump = his_prices.max()
his_neg_min_jump = his_prices.min()

historical_stats = pd.DataFrame({
    'Metric': [
        'Positive Jump Mean',
        'Negative Jump Mean',
        'Positive Jump Number',
        'Negative Jump Number',
        'Positive Jump Std',
        'Negative Jump Std',
        'Absolute Jump Mean',
        'Absolute Jump Std',
        'Correlation Vol-Size',
        'Correlation Vol-Freq',
        'Mean Rolling Vol',
        'Rolling Vol Std',
        'Positive Max Jump',
        'Negative Min Jump'
    ],
    'Value': [
        float(his_pos_jump_mean.iloc[0]),
        float(his_neg_jump_mean.iloc[0]),
        float(his_pos_jump_num.iloc[0]),
        float(his_neg_jump_num.iloc[0]),
        float(his_pos_jump_std.iloc[0]),
        float(his_neg_jump_std.iloc[0]),
        float(his_abs_jump_mean.iloc[0]),
        float(his_abs_jump_std.iloc[0]),
        float(his_correl_vol_size),
        float(his_correl_vol_freq),
        float(his_vol_mean.iloc[0]),
        float(his_vol_std.iloc[0]),
        float(his_pos_max_jump.iloc[0]),
        float(his_neg_min_jump.iloc[0])
    ]
})

#Loss function
def loss_function(params, historical_stats):
    try:
        sigma_0, lambda_0, jump_mean_pos, jump_std_pos, jump_mean_neg, jump_std_neg, kappa, theta, eta, alpha, beta_pos, beta_neg, gamma_pos, gamma_neg, prob_positive = params

        #Simulated metrics
        simulated_price, simulated_volatility, time_points, debug_data, seed = svj_simulation(
            sigma_0, lambda_0, jump_mean_pos, jump_std_pos, jump_mean_neg, jump_std_neg,
            kappa, theta, eta, alpha, beta_pos, beta_neg, gamma_pos, gamma_neg,
            prob_positive, seed=42
        )

        sim_prices = pd.DataFrame({'sim_prices': simulated_price})
        sim_prices = sim_prices.set_index(his_prices.index).asfreq('D')

        sim_volatility = pd.DataFrame({'sim_volatility': pd.Series(simulated_volatility).rolling(window=7).mean().dropna()})
        sim_volatility = sim_volatility.set_index(his_volatility.index).asfreq('D')

        #1) Jump mean
        sim_pos_jump_mean = sim_prices[sim_prices > threshold].mean()
        sim_neg_jump_mean = sim_prices[sim_prices < -threshold].mean()

        #2) Number of jumps
        sim_pos_jump_num = sim_prices[sim_prices > threshold].count()
        sim_neg_jump_num = sim_prices[sim_prices < -threshold].count()

        #3) Jump neg and pos std
        sim_pos_jump_std = sim_prices[sim_prices > threshold].std()
        sim_neg_jump_std = sim_prices[sim_prices < -threshold].std()

        #4) Absolute jump mean
        sim_abs_jump_mean = sim_prices.abs()[sim_prices.abs() > threshold].mean()

        #5) Absolute value jump std
        sim_abs_jump_std = sim_prices.abs()[sim_prices.abs() > threshold].std()

        #6) Correlation between lagged rolling vol and jumps size and frequency
        x_correl_vol_size = sim_volatility.shift(1)[sim_volatility.index.isin(sim_prices.abs()[sim_prices.abs() > threshold].dropna().index)]
        y_correl_vol_size = sim_prices.abs()[sim_prices.index.isin(x_correl_vol_size.index)]

        sim_correl_vol_size = sm.OLS(y_correl_vol_size, x_correl_vol_size).fit().params.iloc[0]

        x_correl_vol_freq = sim_volatility.shift(1).dropna()
        y_correl_vol_freq = (sim_prices.abs() > threshold).astype(int)
        y_correl_vol_freq = y_correl_vol_freq[y_correl_vol_freq.index.isin(x_correl_vol_freq.index)]
        sim_correl_vol_freq = sm.OLS(y_correl_vol_freq,x_correl_vol_freq).fit().params.iloc[0]

        #7) Mean rolling vol
        sim_vol_mean = sim_volatility.mean()

        #8) Rolling vol standard deviation
        sim_vol_std = sim_volatility.std()

        #9 Min and max values for positive and negative jumps
        sim_pos_max_jump = sim_prices.max()
        sim_neg_min_jump = sim_prices.min()

        #Loss definition
        loss = (
            # 1) Jump mean
            ((float(sim_pos_jump_mean.iloc[0]) - historical_stats.loc[historical_stats['Metric'] == 'Positive Jump Mean', 'Value'].iloc[0]) /
             historical_stats.loc[historical_stats['Metric'] == 'Positive Jump Mean', 'Value'].iloc[0])**2 +
            ((float(sim_neg_jump_mean.iloc[0]) - historical_stats.loc[historical_stats['Metric'] == 'Negative Jump Mean', 'Value'].values[0]) /
             historical_stats.loc[historical_stats['Metric'] == 'Negative Jump Mean', 'Value'].values[0])**2 +
            # 2) Number of jumps
            ((float(sim_pos_jump_num.iloc[0]) - historical_stats.loc[historical_stats['Metric'] == 'Positive Jump Number', 'Value'].values[0]) /
             historical_stats.loc[historical_stats['Metric'] == 'Positive Jump Number', 'Value'].values[0])**2 +
            ((float(sim_neg_jump_num.iloc[0]) - historical_stats.loc[historical_stats['Metric'] == 'Negative Jump Number', 'Value'].values[0]) /
             historical_stats.loc[historical_stats['Metric'] == 'Negative Jump Number', 'Value'].values[0])**2 +
            # 3) Jump neg and pos std
            ((float(sim_pos_jump_std.iloc[0]) - historical_stats.loc[historical_stats['Metric'] == 'Positive Jump Std', 'Value'].values[0]) /
             historical_stats.loc[historical_stats['Metric'] == 'Positive Jump Std', 'Value'].values[0])**2 +
            ((float(sim_neg_jump_std.iloc[0]) - historical_stats.loc[historical_stats['Metric'] == 'Negative Jump Std', 'Value'].values[0]) /
             historical_stats.loc[historical_stats['Metric'] == 'Negative Jump Std', 'Value'].values[0])**2 +
            # 4) Absolute jump mean
            ((float(sim_abs_jump_mean.iloc[0]) - historical_stats.loc[historical_stats['Metric'] == 'Absolute Jump Mean', 'Value'].values[0]) /
             historical_stats.loc[historical_stats['Metric'] == 'Absolute Jump Mean', 'Value'].values[0])**2 +
            # 5) Absolute value jump std
            ((float(sim_abs_jump_std.iloc[0]) - historical_stats.loc[historical_stats['Metric'] == 'Absolute Jump Std', 'Value'].values[0]) /
             historical_stats.loc[historical_stats['Metric'] == 'Absolute Jump Std', 'Value'].values[0])**2 +
            # 6) Correlation between lagged rolling vol and jumps size and frequency
            ((float(sim_correl_vol_size) - historical_stats.loc[historical_stats['Metric'] == 'Correlation Vol-Size', 'Value'].values[0]) /
             abs(historical_stats.loc[historical_stats['Metric'] == 'Correlation Vol-Size', 'Value'].values[0]))**2 +
            ((float(sim_correl_vol_freq) - historical_stats.loc[historical_stats['Metric'] == 'Correlation Vol-Freq', 'Value'].values[0]) /
             abs(historical_stats.loc[historical_stats['Metric'] == 'Correlation Vol-Freq', 'Value'].values[0]))**2 +
            # 7) Mean rolling vol
            ((float(sim_vol_mean.iloc[0]) - historical_stats.loc[historical_stats['Metric'] == 'Mean Rolling Vol', 'Value'].values[0]) /
             historical_stats.loc[historical_stats['Metric'] == 'Mean Rolling Vol', 'Value'].values[0])**2 +
            # 8) Rolling vol standard deviation
            ((float(sim_vol_std.iloc[0]) - historical_stats.loc[historical_stats['Metric'] == 'Rolling Vol Std', 'Value'].values[0]) /
             historical_stats.loc[historical_stats['Metric'] == 'Rolling Vol Std', 'Value'].values[0])**2 +
            # 9) Min and max values for positive and negative jumps
            ((float(sim_pos_max_jump.iloc[0]) - historical_stats.loc[historical_stats['Metric'] == 'Positive Max Jump', 'Value'].values[0]) /
             abs(historical_stats.loc[historical_stats['Metric'] == 'Positive Max Jump', 'Value'].values[0]))**2 +
            ((float(sim_neg_min_jump.iloc[0]) - historical_stats.loc[historical_stats['Metric'] == 'Negative Min Jump', 'Value'].values[0]) /
             abs(historical_stats.loc[historical_stats['Metric'] == 'Negative Min Jump', 'Value'].values[0]))**2
        )
    except Exception as e:
        print("Error in loss calculation: ", e)
        return np.inf
    return loss

def specific_loss_function(params, historical_stats):
    try:
        sigma_0, lambda_0, jump_mean_pos, jump_std_pos, jump_mean_neg, jump_std_neg, kappa, theta, eta, alpha, beta_pos, beta_neg, gamma_pos, gamma_neg, prob_positive = params

        # Simulated metrics
        simulated_price, simulated_volatility, time_points, debug_data, seed = svj_simulation(
            sigma_0, lambda_0, jump_mean_pos, jump_std_pos, jump_mean_neg, jump_std_neg,
            kappa, theta, eta, alpha, beta_pos, beta_neg, gamma_pos, gamma_neg,
            prob_positive, seed=42
        )

        sim_prices = pd.DataFrame({'sim_prices': simulated_price})
        sim_prices = sim_prices.set_index(his_prices.index).asfreq('D')

        sim_volatility = pd.DataFrame({'sim_volatility': pd.Series(simulated_volatility).rolling(window=7).mean().dropna()})
        sim_volatility = sim_volatility.set_index(his_volatility.index).asfreq('D')

        # Metrics
        sim_pos_jump_mean = sim_prices[sim_prices > threshold].mean()
        sim_neg_jump_mean = sim_prices[sim_prices < -threshold].mean()

        sim_pos_jump_num = sim_prices[sim_prices > threshold].count()
        sim_neg_jump_num = sim_prices[sim_prices < -threshold].count()

        sim_pos_jump_std = sim_prices[sim_prices > threshold].std()
        sim_neg_jump_std = sim_prices[sim_prices < -threshold].std()

        sim_abs_jump_mean = sim_prices.abs()[sim_prices.abs() > threshold].mean()
        sim_abs_jump_std = sim_prices.abs()[sim_prices.abs() > threshold].std()

        x_correl_vol_size = sim_volatility.shift(1)[sim_volatility.index.isin(sim_prices.abs()[sim_prices.abs() > threshold].dropna().index)]
        y_correl_vol_size = sim_prices.abs()[sim_prices.index.isin(x_correl_vol_size.index)]
        sim_correl_vol_size = sm.OLS(y_correl_vol_size, x_correl_vol_size).fit().params.iloc[0]

        x_correl_vol_freq = sim_volatility.shift(1).dropna()
        y_correl_vol_freq = (sim_prices.abs() > threshold).astype(int)
        y_correl_vol_freq = y_correl_vol_freq[y_correl_vol_freq.index.isin(x_correl_vol_freq.index)]
        sim_correl_vol_freq = sm.OLS(y_correl_vol_freq, x_correl_vol_freq).fit().params.iloc[0]

        sim_vol_mean = sim_volatility.mean()
        sim_vol_std = sim_volatility.std()

        sim_pos_max_jump = sim_prices.max()
        sim_neg_min_jump = sim_prices.min()

        # Individual loss components
        loss_components = {
            "jump_mean_pos": ((float(sim_pos_jump_mean.iloc[0]) - historical_stats.loc[historical_stats['Metric'] == 'Positive Jump Mean', 'Value'].iloc[0]) /
                              historical_stats.loc[historical_stats['Metric'] == 'Positive Jump Mean', 'Value'].iloc[0])**2,
            "jump_mean_neg": ((float(sim_neg_jump_mean.iloc[0]) - historical_stats.loc[historical_stats['Metric'] == 'Negative Jump Mean', 'Value'].values[0]) /
                              historical_stats.loc[historical_stats['Metric'] == 'Negative Jump Mean', 'Value'].values[0])**2,
            "jump_num_pos": ((float(sim_pos_jump_num.iloc[0]) - historical_stats.loc[historical_stats['Metric'] == 'Positive Jump Number', 'Value'].values[0]) /
                             historical_stats.loc[historical_stats['Metric'] == 'Positive Jump Number', 'Value'].values[0])**2,
            "jump_num_neg": ((float(sim_neg_jump_num.iloc[0]) - historical_stats.loc[historical_stats['Metric'] == 'Negative Jump Number', 'Value'].values[0]) /
                             historical_stats.loc[historical_stats['Metric'] == 'Negative Jump Number', 'Value'].values[0])**2,
            "jump_std_pos": ((float(sim_pos_jump_std.iloc[0]) - historical_stats.loc[historical_stats['Metric'] == 'Positive Jump Std', 'Value'].values[0]) /
                             historical_stats.loc[historical_stats['Metric'] == 'Positive Jump Std', 'Value'].values[0])**2,
            "jump_std_neg": ((float(sim_neg_jump_std.iloc[0]) - historical_stats.loc[historical_stats['Metric'] == 'Negative Jump Std', 'Value'].values[0]) /
                             historical_stats.loc[historical_stats['Metric'] == 'Negative Jump Std', 'Value'].values[0])**2,
            "abs_jump_mean": ((float(sim_abs_jump_mean.iloc[0]) - historical_stats.loc[historical_stats['Metric'] == 'Absolute Jump Mean', 'Value'].values[0]) /
                              historical_stats.loc[historical_stats['Metric'] == 'Absolute Jump Mean', 'Value'].values[0])**2,
            "abs_jump_std": ((float(sim_abs_jump_std.iloc[0]) - historical_stats.loc[historical_stats['Metric'] == 'Absolute Jump Std', 'Value'].values[0]) /
                             historical_stats.loc[historical_stats['Metric'] == 'Absolute Jump Std', 'Value'].values[0])**2,
            "correlation_vol_size": ((float(sim_correl_vol_size) - historical_stats.loc[historical_stats['Metric'] == 'Correlation Vol-Size', 'Value'].values[0]) /
                                     abs(historical_stats.loc[historical_stats['Metric'] == 'Correlation Vol-Size', 'Value'].values[0]))**2,
            "correlation_vol_freq": ((float(sim_correl_vol_freq) - historical_stats.loc[historical_stats['Metric'] == 'Correlation Vol-Freq', 'Value'].values[0]) /
                                      abs(historical_stats.loc[historical_stats['Metric'] == 'Correlation Vol-Freq', 'Value'].values[0]))**2,
            "mean_rolling_vol": ((float(sim_vol_mean.iloc[0]) - historical_stats.loc[historical_stats['Metric'] == 'Mean Rolling Vol', 'Value'].values[0]) /
                                 historical_stats.loc[historical_stats['Metric'] == 'Mean Rolling Vol', 'Value'].values[0])**2,
            "rolling_vol_std": ((float(sim_vol_std.iloc[0]) - historical_stats.loc[historical_stats['Metric'] == 'Rolling Vol Std', 'Value'].values[0]) /
                                historical_stats.loc[historical_stats['Metric'] == 'Rolling Vol Std', 'Value'].values[0])**2,
            "max_jump_pos": ((float(sim_pos_max_jump.iloc[0]) - historical_stats.loc[historical_stats['Metric'] == 'Positive Max Jump', 'Value'].values[0]) /
                             abs(historical_stats.loc[historical_stats['Metric'] == 'Positive Max Jump', 'Value'].values[0]))**2,
            "min_jump_neg": ((float(sim_neg_min_jump.iloc[0]) - historical_stats.loc[historical_stats['Metric'] == 'Negative Min Jump', 'Value'].values[0]) /
                             abs(historical_stats.loc[historical_stats['Metric'] == 'Negative Min Jump', 'Value'].values[0]))**2
        }

        # Print individual components
        for component, value in loss_components.items():
            with np.printoptions(precision = 10, suppress=True):
                print(f"{component}: {value}")

        # Total loss
        loss = sum(loss_components.values())

    except Exception as e:
        print("Error in loss calculation: ", e)
        return np.inf

    return loss

initial_params = [
    float(sigma_0),
    float(lambda_0),
    float(jump_mean_pos),
    float(jump_std_pos),
    float(jump_mean_neg),
    float(jump_std_neg),
    float(kappa),
    float(theta),
    float(eta),
    float(alpha),
    float(beta_pos),
    float(beta_neg),
    float(gamma_pos),
    float(gamma_neg),
    float(prob_positive)
]

bounds = [
    (15.0, 25.0),          # Sigma_0 (Initial Volatility)
    (0.05, 0.2),           # Lambda_0 (Baseline Jump Intensity)
    (70.0, 90.0),          # Positive Jump Mean
    (30.0, 50.0),          # Positive Jump Std Dev
    (-100.0, -70.0),       # Negative Jump Mean
    (30.0, 50.0),          # Negative Jump Std Dev
    (0.05, 0.2),           # Kappa (Mean Reversion Speed)
    (200.0, 300.0),        # Theta (Long-Term Variance)
    (3.0, 8.0),            # Eta (Volatility of Volatility)
    (0.005, 0.02),         # Alpha (Jump Intensity Sensitivity)
    (1.5, 2.0),            # Beta_Pos (Positive Jump Mean Sensitivity)
    (-1.5, -1.0),          # Beta_Neg (Negative Jump Mean Sensitivity)
    (0.1, 0.3),            # Gamma_Pos (Positive Jump Std Dev Sensitivity)
    (0.1, 0.3),            # Gamma_Neg (Negative Jump Std Dev Sensitivity)
    (prob_positive, prob_positive)  # Probability of Positive Jumps (fixed)
]

limiter = input("Do calibrate simulation model? (y/n): ").lower() == 'y'

if limiter == 1:
    result = minimize(
        loss_function, initial_params, args=(historical_stats,),method='L-BFGS-B', bounds = bounds,
        options={'maxiter':10000}
    )
    with np.printoptions(precision = 10, suppress=True):
        print(f'minimize calibrated parameters: {result.x}')
        print(f'initial parameters: {initial_params}')
    print(result.success)

    result_2 = differential_evolution(loss_function, bounds = bounds, args=(historical_stats,), disp=True)
    with np.printoptions(precision = 10, suppress=True):
        print(f'diff evolution calibrated parameters: {result_2.x}')
        print(f'initial parameters: {initial_params}')
    print(result_2.success)

    calibrated_params = result_2.x
    with open("../Code day ahead/calibrated_params v2.txt", "w") as file:
        count = 0
        for item in calibrated_params:
            count += 1
            if count < len(calibrated_params):
                x = ','
            else:
                x = ''
            file.write(f'{item}{x}')

else:
    with open('calibrated_params.txt', 'r') as file:
        calibrated_params = [float(item) for item in file.read().split(',')]


sigma_0, lambda_0, jump_mean_pos, jump_std_pos, jump_mean_neg, jump_std_neg, kappa, theta, eta, alpha, beta_pos, beta_neg, gamma_pos, gamma_neg, prob_positive = calibrated_params
# Plot the results
plt.figure(figsize=(12, 6))
# Simulate the process
simulated_price, simulated_volatility, time_points, debug_data, seed = svj_simulation(
    sigma_0, lambda_0, jump_mean_pos, jump_std_pos, jump_mean_neg, jump_std_neg,
    kappa, theta, eta, alpha, beta_pos, beta_neg, gamma_pos, gamma_neg,
    prob_positive, seed=42
)

# Plot simulated prices
plt.subplot(2, 1, 1)
plt.plot(time_points, simulated_price, label=f"Simulated Price")
plt.title("Post-calibration SVJ Process with Mean-Reverting Volatility and Random Walk Prices")
plt.ylabel("Price")
plt.grid()
plt.legend()

# Plot simulated volatility
plt.subplot(2, 1, 2)
plt.plot(time_points, pd.Series(simulated_volatility).rolling(window = 7).mean(), label=f"Simulated Volatility 7-day MVA")
plt.ylabel("Volatility")
plt.xlabel("Time")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

#calibrated parameters: [ 18.7596284452   0.0564839399  70.9919952736  30.0561548979 -70.1542294122  30.0646342149   0.0503326854 299.6500140103 7.6405327341   0.0050895421   1.5903645119  -1.0030781081  0.1053767042   0.1401420032   0.5169082126]
#try capping the extreme outlier. Also add vol max and min to loss function.