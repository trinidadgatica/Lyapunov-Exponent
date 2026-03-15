import numpy as np
import pandas as pd

from core.lyapunov import (
    EQUATION_DISPLAY_NAMES,
    compute_lce_from_eigenvalue_product_trajectory,
    compute_lce_qr_from_trajectory,
    compute_lce_sum_from_determinants_trajectory,
)
from models.bubble_models import simulate_bubble_trajectories


def run_method_comparison_experiment(
    equation: str,
    temperature: float,
    pressure: float,
    frequency: float,
    radius: float,
    times: np.ndarray,
    step: float
) -> tuple:
    trajectories, model = simulate_bubble_trajectories([equation], temperature, pressure, frequency, radius, times, step)

    radius_data = trajectories[f'Radius_{equation}']
    velocity_data = trajectories[f'Velocity_{equation}']
    scaled_time = times * frequency

    lce_qr, hist_qr = compute_lce_qr_from_trajectory(radius_data, velocity_data, scaled_time, model, EQUATION_DISPLAY_NAMES[equation], keep=True)
    lce_eig, hist_eig = compute_lce_from_eigenvalue_product_trajectory(radius_data, velocity_data, scaled_time, model, EQUATION_DISPLAY_NAMES[equation], keep=True)
    lce_det, hist_det = compute_lce_sum_from_determinants_trajectory(radius_data, velocity_data, scaled_time, model, EQUATION_DISPLAY_NAMES[equation], keep=True)

    return lce_qr, lce_eig, lce_det, hist_qr, hist_eig, hist_det


def get_final_period_indices(
    times: np.ndarray,
    frequency: float,
    periods: int,
    M: int = 1
) -> np.ndarray:
    """
    Get the indices of the last M periods of oscillation.

    Parameters
    ----------
    times : np.ndarray
        Time array (e.g., from np.arange).
    frequency : float
        Acoustic frequency [Hz].
    periods : int
        Total number of simulated periods.
    M : int, optional
        Number of last periods to extract (default = 1).

    Returns
    -------
    indices : np.ndarray
        Array of indices corresponding to the last M periods.
    """
    # Period duration
    T = 1 / frequency

    # Clamp M so it doesn't exceed total simulated periods
    M = min(M, periods)

    # Start and end times
    t_start = (periods - M) * T
    t_end = periods * T

    # Boolean mask and indices
    mask = (times >= t_start) & (times <= t_end)
    return np.where(mask)[0]

def _filter_finite_values(x):
    x = np.asarray(x).ravel()
    return x[np.isfinite(x)]

def _compute_quantiles(x):
    x = _filter_finite_values(x)
    try:
        q05, q25, q50, q75, q95 = np.nanquantile(x, [0.05, 0.25, 0.50, 0.75, 0.95], method="linear")
    except TypeError:  # NumPy < 1.23
        q05, q25, q50, q75, q95 = np.nanquantile(x, [0.05, 0.25, 0.50, 0.75, 0.95], interpolation="linear")
    return float(q05), float(q25), float(q50), float(q75), float(q95)

def _compute_mad(x):
    x = _filter_finite_values(x)
    m = np.nanmedian(x)
    return float(np.nanmedian(np.abs(x - m)))

def _compute_robust_scale(x):
    q05, q25, q50, q75, q95 = _compute_quantiles(x)
    iqr = q75 - q25
    if np.isfinite(iqr) and iqr > 0:
        return iqr
    s = 1.4826 * _compute_mad(x)
    return s if (np.isfinite(s) and s > 0) else np.nan

def median_error_over_iqr(x, lam_star):
    x = _filter_finite_values(x)
    if x.size == 0 or not np.isfinite(lam_star): return np.nan
    scale = _compute_robust_scale(x)
    if not np.isfinite(scale) or scale == 0:     return np.nan
    return float(np.nanmedian(np.abs(x - lam_star)) / scale)

def wasserstein_over_iqr(x, lam_star):
    x = _filter_finite_values(x)
    if x.size == 0 or not np.isfinite(lam_star): return np.nan
    scale = _compute_robust_scale(x)
    if not np.isfinite(scale) or scale == 0:     return np.nan
    return float(np.nanmean(np.abs(x - lam_star)) / scale)


def last_period_summary_table(last_period_sample: dict[str, np.ndarray], final_values: dict[str, float],
                        order: list[str] | None = None, decimals: int = 3) -> pd.DataFrame:
    """
    last_period_sample:  {name -> 1D array of samples from the last period}
    finals: {name -> scalar final λ*}
    order:  optional fixed row order
    """
    rows = []
    names = order or list(last_period_sample.keys())
    for name in names:
        x = last_period_sample.get(name, None)
        if x is None: continue
        x = _filter_finite_values(x)
        if x.size == 0: continue
        lam = float(final_values[name])

        q05, q25, q50, q75, q95 = _compute_quantiles(x)
        rmed = median_error_over_iqr(x, lam)
        w1   = wasserstein_over_iqr(x, lam)

        rows.append({
            "Method": name,
            "λ* (final)": lam,
            "q05 (5%)": q05,
            "q95 (95%)": q95,
            "r_med/IQR": rmed,
            "W1/IQR": w1
        })

    df = pd.DataFrame(rows)
    # nice ordering & rounding
    cols = ["Method", "λ* (final)", "q05 (5%)", "q95 (95%)", "r_med/IQR", "W1/IQR"]
    df = df[cols]
    num_cols = ["λ* (final)", "q05 (5%)", "q95 (95%)", "r_med/IQR", "W1/IQR"]
    df[num_cols] = df[num_cols].round(decimals)
    return df


