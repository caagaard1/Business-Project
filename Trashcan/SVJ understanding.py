import numpy as np
import matplotlib.pyplot as plt

def svj_simulation_symmetric(mu, sigma, lambda_jump, jump_mean, jump_std, T, dt, S0):
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
        Simulated time series as a NumPy array and the corresponding time points.
    """
    n_steps = int(T / dt)  # Number of time steps
    times = np.linspace(0, T, n_steps)  # Time points
    S = [S0]  # Initialize the process

    for t in range(1, n_steps):
        dW = np.random.normal(0, np.sqrt(dt))  # Brownian increment
        N_t = np.random.poisson(lambda_jump * dt)  # Number of jumps in this interval
        J_t = np.sum(np.random.normal(jump_mean, jump_std, N_t))  # Symmetric jump size

        # Update process
        dS = mu * dt + sigma * dW + J_t
        S.append(S0 + dS)

    return np.array(S), times

# Parameters for the SVJ process
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Load and preprocess data
full_data = pd.read_csv(r'/Trashcan/SVJ_data_SARIMA.csv')
full_data['Date delivery'] = pd.to_datetime(full_data['Date delivery'])
data = full_data.set_index('Date delivery')[['trans_residuals']].asfreq('D').dropna()

# Calculate absolute changes
absolute_returns = data.diff().dropna()

# Initial parameter guesses
threshold = 10  # Define a threshold for extreme moves (e.g., $10)
mu = absolute_returns.mean().iloc[0]
sigma = absolute_returns.std().iloc[0]
extreme_moves = (absolute_returns.abs() > threshold).sum().iloc[0]
lambda_jump = extreme_moves / len(absolute_returns)
jumps = absolute_returns[absolute_returns.abs() > threshold]
jump_mean = jumps.mean().iloc[0]
jump_std = jumps.std().iloc[0]
T = len(data)
dt = 1
S0 = float(data.mean().iloc[0])

plt.figure(figsize=(10, 6))
plt.title("Stochastic Volatility with Jumps (SVJ) Simulation")
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid()

print(f'mu {mu} sigma {sigma}, lamda_jump {lambda_jump}, jump_mean {jump_mean}, jump_std {jump_std}, T {T}, dt {dt}, S0 {S0}')

# Simulate the process
for i in range (1, 2):
    simulated_process, time_points = svj_simulation_symmetric(mu, sigma, lambda_jump, jump_mean, jump_std, T, dt, S0)
    plt.plot(time_points, simulated_process, label="SVJ Process")

plt.show()
