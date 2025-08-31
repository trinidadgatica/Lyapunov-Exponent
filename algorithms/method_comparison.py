import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from algorithms.dynamics import create_trajectories
from algorithms.lyapunov import (
    compute_lyapunov_exponents_from_trajectory,
    compute_lyapunov_sum_from_determinants_fixed,
    compute_lyapunov_from_eigenvalue_product_fixed,
    equation_name_dd
)
from utils.plot_information import (
    PLOT_WIDTH, PLOT_HEIGHT,
    X_LABEL_FONT_SIZE, Y_LABEL_FONT_SIZE,
    X_TICK_FONT_SIZE, Y_TICK_FONT_SIZE,
    LEGEND_POSITION, LEGEND_FONT_SIZE,
    LEGEND_TITLE_FONT_SIZE, colors)



def run_lce_method_comparison(
    equation: str,
    temperature: float,
    pressure: float,
    frequency: float,
    radius: float,
    times: np.ndarray,
    step: float
) -> tuple:
    trajectories, model = create_trajectories([equation], temperature, pressure, frequency, radius, times, step)

    radius_data = trajectories[f'Radius_{equation}']
    velocity_data = trajectories[f'Velocity_{equation}']
    scaled_time = times * frequency

    lce_qr, hist_qr = compute_lyapunov_exponents_from_trajectory(radius_data, velocity_data, scaled_time, model, equation_name_dd[equation], keep=True)
    lce_eig, hist_eig = compute_lyapunov_from_eigenvalue_product_fixed(radius_data, velocity_data, scaled_time, model, equation_name_dd[equation], keep=True)
    lce_det, hist_det = compute_lyapunov_sum_from_determinants_fixed(radius_data, velocity_data, scaled_time, model, equation_name_dd[equation], keep=True)

    return lce_qr, lce_eig, lce_det, hist_qr, hist_eig, hist_det


def get_last_period_indices(
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


def period_stats(history: np.ndarray) -> dict:
    """
    Calculate min, max, and mean for a NumPy array time series.
    """
    return {
        "min": np.min(history),
        "max": np.max(history),
        "mean": np.mean(history)
    }


def _finite(x):
    x = np.asarray(x).ravel()
    return x[np.isfinite(x)]

def _quantiles(x):
    x = _finite(x)
    try:
        q05, q25, q50, q75, q95 = np.nanquantile(x, [0.05, 0.25, 0.50, 0.75, 0.95], method="linear")
    except TypeError:  # NumPy < 1.23
        q05, q25, q50, q75, q95 = np.nanquantile(x, [0.05, 0.25, 0.50, 0.75, 0.95], interpolation="linear")
    return float(q05), float(q25), float(q50), float(q75), float(q95)

def _mad(x):
    x = _finite(x)
    m = np.nanmedian(x)
    return float(np.nanmedian(np.abs(x - m)))

def _scale_iqr_or_mad(x):
    q05, q25, q50, q75, q95 = _quantiles(x)
    iqr = q75 - q25
    if np.isfinite(iqr) and iqr > 0:
        return iqr
    s = 1.4826 * _mad(x)
    return s if (np.isfinite(s) and s > 0) else np.nan

def r_med_over_IQR(x, lam_star):
    x = _finite(x)
    if x.size == 0 or not np.isfinite(lam_star): return np.nan
    scale = _scale_iqr_or_mad(x)
    if not np.isfinite(scale) or scale == 0:     return np.nan
    return float(np.nanmedian(np.abs(x - lam_star)) / scale)

def W1_over_IQR(x, lam_star):
    x = _finite(x)
    if x.size == 0 or not np.isfinite(lam_star): return np.nan
    scale = _scale_iqr_or_mad(x)
    if not np.isfinite(scale) or scale == 0:     return np.nan
    return float(np.nanmean(np.abs(x - lam_star)) / scale)


def last1_summary_table(last1: dict[str, np.ndarray], finals: dict[str, float],
                        order: list[str] | None = None, decimals: int = 3) -> pd.DataFrame:
    """
    last1:  {name -> 1D array (last 1 period samples)}
    finals: {name -> scalar final λ*}
    order:  optional fixed row order
    """
    rows = []
    names = order or list(last1.keys())
    for name in names:
        x = last1.get(name, None)
        if x is None: continue
        x = _finite(x)
        if x.size == 0: continue
        lam = float(finals[name])

        q05, q25, q50, q75, q95 = _quantiles(x)
        rmed = r_med_over_IQR(x, lam)
        w1   = W1_over_IQR(x, lam)

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


