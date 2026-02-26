"""
Module for computing Lyapunov exponents and benchmarking methods for the Lorenz system.
"""

import time
import numpy as np
import nolds
from numpy.linalg import norm, qr
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

def advance_tangent_linear_map(W: np.ndarray, J: np.ndarray, dt: float) -> np.ndarray:
    """
    One step RK4 for tangent linear map evolution.
    """
    k1 = J @ W
    k2 = J @ (W + 0.5 * dt * k1)
    k3 = J @ (W + 0.5 * dt * k2)
    k4 = J @ (W + dt * k3)
    return W + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


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

        W = advance_tangent_linear_map(W, J, dt)
        # W = W + dt * J @ W
        W, R = np.linalg.qr(W)
        for j in range(d):
            LCE_vals[j] += np.log(np.abs(R[j, j]))
            if keep:
                history[i, j] = LCE_vals[j] / ((i + 1) * dt)

    LCE_vals = LCE_vals / (N * dt)
    return (LCE_vals, history) if keep else LCE_vals

def _phi_from_J(J: np.ndarray, dt: float) -> np.ndarray:
    """Per-step flow propagator Phi ≈ exp(J*dt) """
    A = J * dt

    return np.eye(J.shape[0]) + A + 0.5 * (A @ A)


def compute_lyapunov_from_eigenvalue_product_lorenz(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, time: np.ndarray,
    sigma: float, rho: float, beta: float, keep: bool = False
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    Lyapunov exponents for Lorenz via Eq. (6.18)-style product-of-maps:
      Product P = Φ_{N-1} ... Φ_0, with Φ_i ≈ exp(J_mid * Δt).
    We avoid overflow by maintaining P ≡ Q @ T (QR similarity), so eigenvalues(P)=eig(T).
    LEs = log|eig(T)| / (time[-1] - time[0]).

    Inputs follow your usual signature (x,y,z,time,sigma,rho,beta,keep).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    time = np.asarray(time, dtype=float)

    N = len(time)
    assert len(x) == len(y) == len(z) == N, "x,y,z,time must have same length"
    if N < 2:
        raise ValueError("Need at least two samples.")
    if not np.all(np.diff(time) > 0):
        raise ValueError("time must be strictly increasing.")

    d = 3
    # Maintain product as Q @ T (T upper-triangular) so eig(product) = eig(T).
    Q = np.eye(d)
    T = np.eye(d)

    if keep:
        history = np.zeros((N, d))
    t0 = time[0]
    dt = time[1] - time[0]  # assume constant step size

    for i in range(N):
        # Midpoint state for better local accuracy
        #xm = 0.5 * (x[i] + x[i + 1])
        #ym = 0.5 * (y[i] + y[i + 1])
        #zm = 0.5 * (z[i] + z[i + 1])

        J = jacobian_lorenz(x[i], y[i], z[i], sigma, rho, beta)
        Phi = _phi_from_J(J, dt)

        # Update product: P <- Φ @ P while keeping P = Q @ T
        Z = Phi @ Q
        Q, R = np.linalg.qr(Z, mode="reduced")
        T = R @ T  # stays upper-triangular; accumulates safely

        if keep:
            evals = np.linalg.eigvals(T)
            lam_running = np.log(np.abs(evals) + np.finfo(float).tiny) / (dt * i)
            history[i] = lam_running

    # Final exponents
    evals_final = np.linalg.eigvals(T)
    LCE = np.log(np.abs(evals_final) + np.finfo(float).tiny) / (time[-1] - time[0])
    LCE = np.sort(LCE)[::-1]

    if keep:
        return LCE, history
    return LCE

def compute_lyapunov_sum_from_determinants_lorenz(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, time: np.ndarray,
    sigma: float, rho: float, beta: float,
    keep: bool = False,
) -> float | tuple[float, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    t = np.asarray(time, dtype=float)

    if not (x.shape == y.shape == z.shape == t.shape):
        raise ValueError("x, y, z, and time must have the same shape.")
    N = x.size
    if N < 2:
        raise ValueError("Need at least two samples.")

    dt = float(t[1] - t[0])
    total_logdet = 0.0
    c = 0.0
    elapsed = 0.0
    hist = np.empty(N - 1, float) if keep else None

    for i in range(N):
        J = jacobian_lorenz(x[i], y[i], z[i], sigma, rho, beta)
        # step = dt * float(np.trace(J))
        M = np.eye(J.shape[0]) + dt * J
        sign, logdet = np.linalg.slogdet(M)     # ~ dt*tr(J) + O(dt^2)
        step =  float(logdet) if sign != 0 else np.nan

        # Kahan summation
        yk = step - c
        sk = total_logdet + yk
        c  = (sk - total_logdet) - yk
        total_logdet = sk

        elapsed += dt
        if keep:
            hist[i] = total_logdet / elapsed

    sum_lce = total_logdet / elapsed
    return (sum_lce, hist) if keep else sum_lce



def compute_lyapunov_sum_from_determinants_lorenz1(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, time: np.ndarray,
    sigma: float, rho: float, beta: float, keep: bool = False
) -> float | tuple[float, np.ndarray]:
    """
    Sum of Lyapunov exponents for Lorenz using the continuous-time trace route:
        sum_i λ_i ≈ (1/N) Σ_k tr(J_c(x_k))   (per unit time)

    Returns a scalar (per unit time). If keep=True, also returns the running estimate.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    t = np.asarray(time, dtype=float)

    if not (x.shape == y.shape == z.shape == t.shape):
        raise ValueError("x, y, z, and time must have the same shape.")
    N = x.size
    if N < 2:
        raise ValueError("Need at least two samples.")

    dt = float(t[1] - t[0])
    if not np.allclose(np.diff(t), dt, rtol=1e-6, atol=1e-12):
        raise ValueError("time must be uniformly spaced.")

    total = 0.0
    comp = 0.0  # Kahan compensation
    history = np.empty(N, dtype=float) if keep else None

    for i in range(N):
        Jc = jacobian_lorenz(x[i], y[i], z[i], sigma, rho, beta)
        trJ = float(np.trace(Jc))

        # Kahan summation
        yk = trJ - comp
        sk = total + yk
        comp = (sk - total) - yk
        total = sk

        if keep:
            history[i] = total / (i + 1)  # per-unit-time estimate

    sum_LCE = total / N  # per-unit-time
    return (sum_LCE, history) if keep else sum_LCE


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

    # 2) EIGENVALUE PRODUCT
    t0_perf = time.perf_counter()
    lce_eig = compute_lyapunov_from_eigenvalue_product_lorenz(x, y, z, t_keep, case["sigma"], case["rho"], case["beta"], keep=False)
    t_qr = time.perf_counter() - t0_perf
    rows.append(dict(
        case=case["name"], method="EIG", time_sec=t_qr,
        lce1=float(lce_eig[0]), lce2=float(lce_eig[1]), lce3=float(lce_eig[2]),
        sigma=case["sigma"], rho=case["rho"], beta=case["beta"]
    ))
    
    # 3) Determinant
    t0_perf = time.perf_counter()
    lce_det = compute_lyapunov_sum_from_determinants_lorenz(x, y, z, t_keep, case["sigma"], case["rho"], case["beta"], keep=False)
    t_qr = time.perf_counter() - t0_perf
    rows.append(dict(
        case=case["name"], method="DET", time_sec=t_qr,
        lce1=float(lce_det),
        sigma=case["sigma"], rho=case["rho"], beta=case["beta"]
    ))
    

    # 4) Eckmann (nolds.lyap_e)
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

    # 5) Rosenstein (nolds.lyap_r)
    t0_perf = time.perf_counter()
    lce_ros = compute_lce_rosenstein(obs, dt, ros_params)
    t_ros = time.perf_counter() - t0_perf
    rows.append(dict(
        case=case["name"], method="Rosenstein", time_sec=t_ros,
        lce1=float(lce_ros), lce2=np.nan, lce3=np.nan,
        sigma=case["sigma"], rho=case["rho"], beta=case["beta"]
    ))

    return rows