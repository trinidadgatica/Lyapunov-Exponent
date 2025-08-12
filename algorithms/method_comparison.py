import numpy as np
from algorithms.dynamics import create_trajectories
from algorithms.lyapunov import compute_lyapunov_exponents_from_trajectory, compute_lyapunov_sum_from_determinants, compute_lyapunov_from_eigenvalue_product, equation_name_dd



def run_lce_method_comparison(equation, temperature, pressure, frequency, radius, times, step):
    trajectories, model = create_trajectories([equation], temperature, pressure, frequency, radius, times, step)

    radius_data = trajectories[f'Radius_{equation}']
    velocity_data = trajectories[f'Velocity_{equation}']
    scaled_time = times * frequency

    lce_qr, hist_qr = compute_lyapunov_exponents_from_trajectory(radius_data, velocity_data, scaled_time, model, equation_name_dd[equation], keep=True)
    lce_eig, hist_eig = compute_lyapunov_from_eigenvalue_product(radius_data, velocity_data, scaled_time, model, equation_name_dd[equation], keep=True)
    lce_det, hist_det = compute_lyapunov_sum_from_determinants(radius_data, velocity_data, scaled_time, model, equation_name_dd[equation], keep=True)

    return lce_qr, lce_eig, lce_det, hist_qr, hist_eig, hist_det


def compute_tail_mean(history, tail_frac=0.1, index=0):
    tail_len = max(1, int(len(history) * tail_frac))
    return np.mean(history[-tail_len:, index])
