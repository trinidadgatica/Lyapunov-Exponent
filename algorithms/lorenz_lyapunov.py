"""
Module for computing Lyapunov exponents and benchmarking methods for the Lorenz system.
"""

import time
import numpy as np
import nolds
from scipy.integrate import solve_ivp

def lorenz(t: float, state: list[float], sigma: float, rho: float, beta: float) -> list[float]:
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]


def jacobian_lorenz(x: float, y: float, z: float, sigma: float, rho: float, beta: float) -> np.ndarray:
    return np.array([
        [-sigma, sigma, 0],
        [rho - z, -1, -x],
        [y, x, -beta]
    ])


def compute_lce_qr_lorenz(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, time: np.ndarray,
    sigma: float, rho: float, beta: float, keep: bool = False
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    N = len(x)
    dt = time[1] - time[0]
    d = 3
    W = np.eye(d)
    LCE_vals = np.zeros(d)
    history = np.zeros((N, d)) if keep else None

    for i in range(N):
        J = jacobian_lorenz(x[i], y[i], z[i], sigma, rho, beta)
        W = W + dt * J @ W
        W, R = np.linalg.qr(W)
        for j in range(d):
            LCE_vals[j] += np.log(np.abs(R[j, j]))
            if keep:
                history[i, j] = LCE_vals[j] / ((i + 1) * dt)

    LCE_vals = LCE_vals / (N * dt)
    return (LCE_vals, history) if keep else LCE_vals


def compute_lce_det_lorenz(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, time: np.ndarray,
    sigma: float, rho: float, beta: float, keep: bool = False
) -> tuple[float, np.ndarray] | float:
    N = len(time)
    dt = time[1] - time[0]
    sum_log_det = 0.0
    history = np.zeros(N) if keep else None

    for i in range(N):
        J = jacobian_lorenz(x[i], y[i], z[i], sigma, rho, beta)
        det_J = np.linalg.det(J)
        sum_log_det += np.log(np.abs(det_J))
        if keep:
            history[i] = sum_log_det / ((i + 1) * dt)

    sum_LCE = sum_log_det / (N * dt)
    return (sum_LCE, history) if keep else sum_LCE


def compute_lce_eigprod_lorenz(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, time: np.ndarray,
    sigma: float, rho: float, beta: float, keep: bool = False
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    d = 3
    N = len(time)
    dt = time[1] - time[0]
    J_total = np.eye(d)
    history = np.full((N, d), np.nan) if keep else None

    for i in range(N):
        J = jacobian_lorenz(x[i], y[i], z[i], sigma, rho, beta)
        J_total = J @ J_total

        if keep:
            if np.any(np.isnan(J_total)) or np.any(np.isinf(J_total)):
                print(f"[WARNING] NaN/Inf in J_total at step {i}, rho={rho}")
                break
            eigvals = np.linalg.eigvals(J_total)
            lyap_step = [np.log(abs(ev)) / ((i + 1) * dt) for ev in eigvals]
            history[i, :] = np.real(lyap_step)

    if np.any(np.isnan(J_total)) or np.any(np.isinf(J_total)):
        return [np.nan] * d, history

    eigvals = np.linalg.eigvals(J_total)
    lyap_exponents = [np.log(abs(ev)) / (N * dt) for ev in eigvals]
    return (np.real(lyap_exponents), history) if keep else np.real(lyap_exponents)

def compute_lce_eckmann(x_1d: np.ndarray, dt: float, params: dict) -> np.ndarray:
    """
    Full spectrum from a scalar observable (Eckmann et al.) via nolds.lyap_e.
    params: emb_dim, matrix_dim, tau, min_tsep, min_nb
    Returns spectrum in 1/time units.
    """
    x = np.ascontiguousarray(x_1d, dtype=float)
    spec_per_sample = nolds.lyap_e(
        x,
        emb_dim=params["emb_dim"],
        matrix_dim=params["matrix_dim"],
        tau=params["tau"],
        min_tsep=params["min_tsep"],
        min_nb=params["min_nb"]
    )
    return np.asarray(spec_per_sample, dtype=float) / dt

def compute_lce_rosenstein(x_1d: np.ndarray, dt: float, params: dict) -> float:
    """
    Largest LCE from a scalar observable (Rosenstein) via nolds.lyap_r.
    params: emb_dim, tau, min_tsep, trajectory_len, fit (optional)
    Returns a single float in 1/time units.
    """
    lle_per_sample = nolds.lyap_r(
        x_1d,
        emb_dim=params["emb_dim"],
        lag=params["tau"],
        min_tsep=params["min_tsep"],
        trajectory_len=params["trajectory_len"],
        fit=params.get("fit", "RANSAC"),
        debug_plot=False
    )
    return float(lle_per_sample) / dt

def integrate_lorenz(
    case: dict, t0: float, t1: float, dt: float,
    rtol: float = 1e-9, atol: float = 1e-9,
    x0: float = 1.0, y0: float = 1.0, z0: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    t_eval = np.arange(t0, t1, dt)
    sol = solve_ivp(
        lorenz,
        (t0, t1),
        [x0, y0, z0],
        t_eval=t_eval,
        args=(case["sigma"], case["rho"], case["beta"]),
        rtol=rtol,
        atol=atol
    )
    return sol.t, sol.y  # time, (3 x N)

def slice_after_transient(t: np.ndarray, Y: np.ndarray, transient_time: float) -> tuple[np.ndarray, np.ndarray]:
    """Drop initial transient by time (seconds)."""
    mask = t >= (t[0] + transient_time)
    return t[mask], Y[:, mask]

def benchmark_case(
    case: dict,
    t0: float,
    t1: float,
    dt: float,
    transient_time: float,
    eck_params: dict,
    ros_params: dict,
    observable: str = "x"
) -> list[dict]:
    """
    Run one parameter case; return list of rows with timing + LCEs for each method.
    """
    # integrate
    t, Y = integrate_lorenz(case, t0, t1, dt)
    t_keep, Y_keep = slice_after_transient(t, Y, transient_time)
    x, y, z = Y_keep

    # choose scalar observable
    obs = {"x": x, "y": y, "z": z}[observable]

    rows = []

    # 1) QR
    t0_perf = time.perf_counter()
    lce_qr = compute_lce_qr_lorenz(x, y, z, t_keep, case["sigma"], case["rho"], case["beta"], keep=False)
    t_qr = time.perf_counter() - t0_perf
    rows.append(dict(
        case=case["name"], method="QR", time_sec=t_qr,
        lce1=float(lce_qr[0]), lce2=float(lce_qr[1]), lce3=float(lce_qr[2]),
        sigma=case["sigma"], rho=case["rho"], beta=case["beta"]
    ))

    # 2) Eckmann (nolds.lyap_e)
    t0_perf = time.perf_counter()
    lce_eck = compute_lce_eckmann(obs, dt, eck_params)
    t_eck = time.perf_counter() - t0_perf
    l1 = float(lce_eck[0]) if lce_eck.size > 0 else np.nan
    l2 = float(lce_eck[1]) if lce_eck.size > 1 else np.nan
    l3 = float(lce_eck[2]) if lce_eck.size > 2 else np.nan
    rows.append(dict(
        case=case["name"], method="Eckmann", time_sec=t_eck,
        lce1=l1, lce2=l2, lce3=l3,
        sigma=case["sigma"], rho=case["rho"], beta=case["beta"]
    ))

    # 3) Rosenstein (nolds.lyap_r)
    t0_perf = time.perf_counter()
    lce_ros = compute_lce_rosenstein(obs, dt, ros_params)
    t_ros = time.perf_counter() - t0_perf
    rows.append(dict(
        case=case["name"], method="Rosenstein", time_sec=t_ros,
        lce1=float(lce_ros), lce2=np.nan, lce3=np.nan,
        sigma=case["sigma"], rho=case["rho"], beta=case["beta"]
    ))

    return rows