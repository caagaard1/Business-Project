from random import seed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Load and preprocess data
full_data = pd.read_csv(r'/Trashcan/SVJ_data_SARIMA.csv')
full_data['Date delivery'] = pd.to_datetime(full_data['Date delivery'])
data = full_data.set_index('Date delivery')[['trans_residuals']].asfreq('D').dropna()

# Calculate absolute changes
absolute_returns = data

# Initial parameter guesses
threshold = 50  # Define a threshold for extreme moves (e.g., $10)
mu_g = 0 #absolute_returns.mean().iloc[0]
sigma_g = absolute_returns.std().iloc[0]
extreme_moves = (absolute_returns.abs() > threshold).sum().iloc[0]
lambda_jump_g = extreme_moves / len(absolute_returns)
jumps = absolute_returns[absolute_returns.abs() > threshold]
jump_mean_g = jumps.mean().iloc[0]
jump_std_g = jumps.std().iloc[0]
initial_params = [mu_g, sigma_g, lambda_jump_g, jump_mean_g, jump_std_g]


def svj_simulation_symmetric(mu, sigma, lambda_jump, jump_mean, jump_std, T, dt, S0, seed=None):
    """
    Simulates an SVJ process with symmetric jumps.

    Parameters:
        mu: Drift (mean return)
        sigma: Diffusion coefficient (volatility)
        lambda_jump: Jump intensity (jumps per unit time)
        jump_mean: Mean size of jumps (can be positive or negative)
        jump_std: Std deviation of jumps
        T: Total time for simulation
        dt: Time step size
        S0: Initial value of the process

    Returns:
        Simulated time series as a NumPy array.
    """
    if seed is not None:
        np.random.seed(seed)

    n_steps = int(T / dt)
    times = np.linspace(0, T, n_steps)
    S = [S0]

    for t in range(1, n_steps):
        dW = np.random.normal(0, np.sqrt(dt))  # Brownian increment
        N_t = np.random.poisson(lambda_jump * dt)  # Number of jumps in this interval
        J_t = np.sum(np.random.normal(jump_mean, jump_std, N_t))  # Symmetric jump size

        # Update process
        dS = mu * dt + sigma * dW + J_t
        S.append(S0 + dS)

    return np.array(S), times


def svj_likelihood_symmetric(params, data, dt, seed=None, alpha = 0.5):
    """
    Calculates the negative log-likelihood for the SVJ process with symmetric jumps.

    Parameters:
        params: List of parameters [mu, sigma, lambda_jump, jump_mean, jump_std]
        data: Observed time series
        dt: Time step size

    Returns:
        Negative log-likelihood value.
    """
    try:
        mu, sigma, lambda_jump, jump_mean, jump_std = params
        T = len(data) * dt
        simulated, _ = svj_simulation_symmetric(mu, sigma, lambda_jump, jump_mean, jump_std, T, dt, S0, seed=seed)

        # Residuals and log-likelihood

        residuals = data - simulated
        likelihood = -np.sum((residuals ** 2) / (2 * sigma ** 2)) - np.log(sigma)
        regularized_likelihood = likelihood - alpha * sigma # Gaussian likelihood

        return -regularized_likelihood  # Return negative for minimization
    except Exception as e:
        print("Error in likelihood calculations:", e)
        return np.inf

def calibrate_svj_symmetric(data, dt):
    """
    Calibrates the SVJ process parameters to the observed data.

    Parameters:
        data: Observed time series
        dt: Time step size

    Returns:
        Optimized parameters as a dictionary.
    """
    # Initial parameter guesses
    initial_params = [mu_g, sigma_g, lambda_jump_g, jump_mean_g, jump_std_g]  # mu, sigma, lambda_jump, jump_mean, jump_std

    # Bounds for the parameters
    bounds = [
        (0, 0),  # mu
        (0.001, 200),  # sigma
        (0.001, 0.2),  # lambda_jump
        (-50, +50),  # jump_mean
        (0.001, 150)  # jump_std
    ]

    # Minimize negative log-likelihood
    result = minimize(svj_likelihood_symmetric, initial_params, args=(data, dt, 42), bounds=bounds, method='L-BFGS-B', options={"disp":True})
    if result.success:
        optimized_params = result.x
        return {
            "mu": optimized_params[0],
            "sigma": optimized_params[1],
            "lambda_jump": optimized_params[2],
            "jump_mean": optimized_params[3],
            "jump_std": optimized_params[4]
        }
    else:
        print(result.x)
        raise ValueError("Optimization failed.")


# Simulate observed data
T = len(data)
dt = 1
S0 = float(data.mean().iloc[0])
print(f'mu {mu_g} sigma {sigma_g}, lamda_jump {lambda_jump_g}, jump_mean {jump_mean_g}, jump_std {jump_std_g}, T {T}, dt {dt}, S0 {S0}')


# Calibrate the model
calibrated_params = calibrate_svj_symmetric(data.values.squeeze(), dt)
print("Calibrated Parameters:", calibrated_params)

# Simulate with calibrated parameters
simulated, _ = svj_simulation_symmetric(
    calibrated_params["mu"], calibrated_params["sigma"],
    calibrated_params["lambda_jump"], calibrated_params["jump_mean"],
    calibrated_params["jump_std"], T, dt, S0
)

# Plot observed vs simulated
plt.subplot(2,1,1)
plt.title("Calibration Results")
plt.ylabel("Value")
plt.plot(data.index, data.values.squeeze(), label="Observed Data")
plt.legend()
plt.subplot(2,1,2)
plt.plot(data.index, simulated, label="Simulated Data", linestyle="--")
plt.xlabel("Time")
plt.legend()
plt.tight_layout()
plt.show()

#Residuals, remove S-1 component, only calculate jumps around zero, no drift, the model should be calculated as jump + SARIMA component, volatility varying jump component