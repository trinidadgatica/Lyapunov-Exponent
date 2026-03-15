import numpy as np

from core.tangent_dynamics import _build_tangent_map_from_jacobian, rk4_step_tangent_map
from models.bubble_models import simulate_bubble_trajectories

EQUATION_DISPLAY_NAMES = dict()
EQUATION_DISPLAY_NAMES['RP'] = 'Rayleigh-Plesset'
EQUATION_DISPLAY_NAMES['KM'] = 'Keller-Miksis'
EQUATION_DISPLAY_NAMES['G'] = 'Gilmore'


def compute_lce_qr_from_trajectory(
    radius: np.ndarray,
    velocity: np.ndarray,
    time: np.ndarray,
    model,
    equation_name: str,
    keep: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Compute Lyapunov Exponents from trajectory data.

    Parameters:
        radius (ndarray): Radius values (N,).
        velocity (ndarray): Velocity values (N,).
        time (ndarray): Time values (N,).
        general_model: Object with method Jacobian_RP(R, V, t) returning 2x2 Jacobian.
        keep (bool): If True, store history of LCEs.

    Returns:
        LCE (ndarray): Final values of the Lyapunov exponents.
        history (ndarray): (Optional) Evolution over time.
    """
    N = len(radius)
    assert len(velocity) == N and len(time) == N, "All arrays must be the same length"
    dt = time[1] - time[0]  # Assume constant time step

    d = 2  # Phase space dimension (R, V)

    W = np.eye(d)[:, :d]
    LCE_vals = np.zeros(d)

    if keep:
        history = np.zeros((N, d))

    for i in range(N):
        if equation_name == 'Rayleigh-Plesset':
            J = model.Jacobian_RP(radius[i], velocity[i], time[i])  # 2x2 Jacobian
        elif equation_name == 'Keller-Miksis':
            J = model.Jacobian_KM(radius[i], velocity[i], time[i])  # 2x2 Jacobian
        elif equation_name == 'Gilmore':
            J = model.Jacobian_G(radius[i], velocity[i], time[i])  # 2x2 Jacobian
        else:
            raise ValueError('Wrong equation name')
        
        W = rk4_step_tangent_map(W, J, dt)

        #W = W + dt * J @ W
        W, R = np.linalg.qr(W)
        for j in range(d):
            LCE_vals[j] += np.log(np.abs(R[j, j]))
            if keep:
                history[i, j] = LCE_vals[j] / ((i + 1) * dt)
    LCE_vals = LCE_vals / (N * dt)

    if keep:
        return LCE_vals, history
    else:
        return LCE_vals


def find_trajectory_cut_index(signal: np.ndarray, tolerance: float = 1e-6, min_consecutive: int = 3) -> int:
    """
    Finds the index at which to cut a signal due to:
    - NaN values
    - Prolonged near-zero values
    """
    nan_indices = np.where(np.isnan(signal))[0]
    if nan_indices.size > 0:
        return nan_indices[0]

    near_zero_mask = np.abs(signal) < tolerance
    count = 0
    for i, is_near_zero in enumerate(near_zero_mask):
        count = count + 1 if is_near_zero else 0
        if count >= min_consecutive:
            return i - min_consecutive + 1

    return len(signal)


def compute_lce_grid(
    grid: list[tuple[float, float]],
    equation: str,
    temperature: float,
    frequency: float = None,
    pressure: float = None,
    filename_suffix: str = ""
) -> list[np.ndarray]:
    """
    Computes Lyapunov exponents over a grid of (radius, freq) or (radius, pressure).
    
    Args:
        grid (list of tuples): Grid of (radius, frequency) or (radius, pressure).
        equation (str): 'RP', 'KM', or 'G'.
        temperature (float): In °C.
        frequency (float or None): Fixed frequency (Hz), if sweeping over pressure.
        pressure (float or None): Fixed pressure (Pa), if sweeping over frequency.
        filename_suffix (str): For log and .npy filenames.
    """
    exponents = []
    log_filename = f"loggers/{equation}_cut_log{filename_suffix}.txt"

    with open(log_filename, "w") as log_file:
        log_file.write("Equation\tCut_Index\tRadius_Next\tVelocity_Next\n")
        for radius_um, value in grid:
            R0 = radius_um * 1e-6

            if frequency is None:
                f = value * 1e6
                P = pressure
            else:
                f = frequency
                P = value * 1e6

            step = 1e-3 / f
            periods = 200
            times = np.arange(0, periods / f, step)
            time = times * f

            trajectories, model = simulate_bubble_trajectories(
                [equation], temperature, P, f, R0, times, step)

            radius = trajectories[f"Radius_{equation}"]
            velocity = trajectories[f"Velocity_{equation}"]

            cut_idx = min(find_trajectory_cut_index(radius), find_trajectory_cut_index(velocity, 1e-10))

            r_next = radius[cut_idx + 1] if cut_idx < len(radius) - 1 else radius[-1]
            v_next = velocity[cut_idx + 1] if cut_idx < len(velocity) - 1 else velocity[-1]

            log_file.write(f"{equation}\t{cut_idx}\t{r_next:.6e}\t{v_next:.6e}\n")

            exps = compute_lce_qr_from_trajectory(
                radius[:cut_idx],
                velocity[:cut_idx],
                time[:cut_idx],
                model,
                EQUATION_DISPLAY_NAMES[equation],
                keep=False
            )
            exponents.append(exps)

    return exponents


def compute_jacobian_eigenvalues(
    radius: np.ndarray,
    velocity: np.ndarray,
    time: np.ndarray,
    model,
    equation_name: str
) -> np.ndarray:
    """
    Compute eigenvalues of the Jacobian at each time step.
    """
    eigenvalues = []

    for r, v, t in zip(radius, velocity, time):
        if equation_name == 'Rayleigh-Plesset':
            J = model.Jacobian_RP(r, v, t)
        elif equation_name == 'Keller-Miksis':
            J = model.Jacobian_KM(r, v, t)
        elif equation_name == 'Gilmore':
            J = model.Jacobian_G(r, v, t)
        else:
            raise ValueError("Unknown equation name.")

        trace = np.trace(J)
        det = np.linalg.det(J)
        disc = (trace / 2)**2 - det

        sqrt_disc = np.sqrt(disc) if disc >= 0 else complex(0, (-disc)**0.5)
        λ1 = trace / 2 + sqrt_disc
        λ2 = trace / 2 - sqrt_disc
        eigenvalues.append([λ1, λ2])

    return np.array(eigenvalues)


def compute_lce_from_eigenvalue_product(
    radius: np.ndarray,
    velocity: np.ndarray,
    time: np.ndarray,
    general_model,
    equation_name: str,
    keep: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Estimate Lyapunov exponents via eigenvalues of the Jacobian product,
    using compute_eigenvalues() to access Jacobians.

    Parameters:
        keep (bool): If True, stores the evolution of exponents.

    Returns:
        lyap_exponents (ndarray): Final Lyapunov exponents.
        history (ndarray, optional): Time evolution of Lyapunov estimates.
    """
    import numpy as np

    d = 2
    N = len(time)
    dt = time[1] - time[0]

    J_total = np.eye(d)
    if keep:
        history = np.zeros((N, d))

    for i in range(N):
        if equation_name == 'Rayleigh-Plesset':
            J = general_model.Jacobian_RP(radius[i], velocity[i], time[i])
        elif equation_name == 'Keller-Miksis':
            J = general_model.Jacobian_KM(radius[i], velocity[i], time[i])
        elif equation_name == 'Gilmore':
            J = general_model.Jacobian_G(radius[i], velocity[i], time[i])
        else:
            raise ValueError('Wrong equation name')

        J_total = J @ J_total

        if keep:
            # Compute eigenvalues at this cumulative step
            a, b = J_total[0][0], J_total[0][1]
            c, d_ = J_total[1][0], J_total[1][1]

            trace = a + d_
            det = a * d_ - b * c
            discriminant = (trace / 2) ** 2 - det

            if discriminant >= 0:
                sqrt_disc = discriminant ** 0.5
            else:
                sqrt_disc = complex(0, (-discriminant) ** 0.5)

            lambda1 = trace / 2 + sqrt_disc
            lambda2 = trace / 2 - sqrt_disc

            eigvals = [lambda1, lambda2]
            lyap_step = [np.log(abs(ev)) / ((i + 1) * dt) for ev in eigvals]
            history[i, :] = lyap_step

    # Final exponents
    a, b = J_total[0][0], J_total[0][1]
    c, d_ = J_total[1][0], J_total[1][1]
    trace = a + d_
    det = a * d_ - b * c
    discriminant = (trace / 2) ** 2 - det

    if discriminant >= 0:
        sqrt_disc = discriminant ** 0.5
    else:
        sqrt_disc = complex(0, (-discriminant) ** 0.5)

    lambda1 = trace / 2 + sqrt_disc
    lambda2 = trace / 2 - sqrt_disc
    eigvals = [lambda1, lambda2]
    lyap_exponents = [ev ** (1/ N) for ev in eigvals]

    if keep:
        return np.array(lyap_exponents), history
    else:
        return np.array(lyap_exponents)


def compute_lce_sum_from_determinants(
    radius: np.ndarray,
    velocity: np.ndarray,
    time: np.ndarray,
    general_model,
    equation_name: str,
    keep: bool = False
) -> float | tuple[float, np.ndarray]:
    """
    Estimate the sum of Lyapunov Exponents using the log-det method.

    Parameters:
        keep (bool): If True, store trajectory of cumulative sum.

    Returns:
        sum_LCE (float): Sum of the Lyapunov exponents.
        history (ndarray, optional): Time evolution of the sum.
    """
    import numpy as np

    N = len(time)
    dt = time[1] - time[0]
    sum_log_det = 0.0

    if keep:
        history = np.zeros(N)

    for i in range(N):
        if equation_name == 'Rayleigh-Plesset':
            J = general_model.Jacobian_RP(radius[i], velocity[i], time[i])
        elif equation_name == 'Keller-Miksis':
            J = general_model.Jacobian_KM(radius[i], velocity[i], time[i])
        elif equation_name == 'Gilmore':
            J = general_model.Jacobian_G(radius[i], velocity[i], time[i])
        else:
            raise ValueError('Wrong equation name')

        det_J = np.linalg.det(J)
        sum_log_det += np.log(np.abs(det_J))

        if keep:
            history[i] = sum_log_det / ((i + 1) * dt)

    sum_LCE = sum_log_det / (N)

    if keep:
        return sum_LCE, history
    else:
        return sum_LCE


def compute_lce_from_eigenvalue_product_trajectory(
    radius: np.ndarray,
    velocity: np.ndarray,
    time: np.ndarray,
    model,
    equation_name: str,
    keep: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Lyapunov exponents via Eq. (6.18)-style eigenvalues of the product of per-step maps.

    We are in continuous time with samples at t[i], so the correct per-step map is
    Phi_i ≈ exp( J(t_i, x_i) * Δt_i ) (midpoint option below). We avoid overflow by
    accumulating an equivalent upper-triangular factor T whose eigenvalues equal those
    of the full product. Then λ_i = (1/T_total) * log |eigvals(T)|.

    Returns
    -------
    LCE : (d,) ndarray of Lyapunov exponents
    history : (N-1, d) running estimates (if keep=True)
    """
    radius = np.asarray(radius, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    time = np.asarray(time, dtype=float)

    N = len(time)
    assert len(radius) == N and len(velocity) == N, "All arrays must be the same length"
    if N < 2:
        raise ValueError("Need at least two samples")

    d = 2  # (R, V)
    # --- helper to get the Jacobian at (t, R, V)
    if equation_name == 'Rayleigh-Plesset':
        J_fn = lambda t, R, V: model.Jacobian_RP(R, V, t)
    elif equation_name == 'Keller-Miksis':
        J_fn = lambda t, R, V: model.Jacobian_KM(R, V, t)
    elif equation_name == 'Gilmore':
        J_fn = lambda t, R, V: model.Jacobian_G(R, V, t)
    else:
        raise ValueError("Wrong equation name")

    # Accumulate via QR similarity: product = Q * T, eigenvalues(product) = eigenvalues(T)
    Q = np.eye(d)
    T = np.eye(d)

    if keep:
        history = np.zeros((N, d))

    t0 = time[0]
    dt = time[1] - time[0]  # assume constant step size

    for i in range(N):
        if dt <= 0:
            raise ValueError("time must be strictly increasing")

        # Midpoint evaluation improves accuracy for flows
        #t_mid = time[i] + 0.5 * dt
        #R_mid = 0.5 * (radius[i] + radius[i + 1])
        #V_mid = 0.5 * (velocity[i] + velocity[i + 1])

        J = np.asarray(J_fn(time[i], radius[i], velocity[i]), dtype=float)
        if J.shape != (d, d):
            raise ValueError(f"Jacobian must be {d}x{d}, got {J.shape}")

        Phi = _build_tangent_map_from_jacobian(J, dt)

        # Update: product <- Phi @ product  ; maintain as Q @ T with QR
        Z = Phi @ Q
        Q, R = np.linalg.qr(Z, mode="reduced")
        T = R @ T  # still upper triangular; eigenvalues(product) = eig(T)

        # Optional running estimate from current T
        if keep:
            # Eigenvalues of current product; normalize by elapsed time
            evals = np.linalg.eigvals(T)
            LCE_step = np.log(np.abs(evals)) / (dt * i)
            history[i, 0] = LCE_step[0]
            history[i, 1] = LCE_step[1]
           

    # Final exponents from T over total time
    evals_final = np.linalg.eigvals(T)
    LCE = np.log(np.abs(evals_final) + np.finfo(float).tiny) / (time[-1] - time[0])
    LCE = np.sort(LCE)[::-1]  # λ1 ≥ λ2

    if keep:
        return LCE, history
    else:
        return LCE
    
def compute_lce_sum_from_determinants_trajectory(
    radius: np.ndarray,
    velocity: np.ndarray,
    time: np.ndarray,
    model,
    equation_name: str,
    keep: bool = False,
) -> float | tuple[float, np.ndarray]:
    """
    Sum of LEs via one-step maps Φ_k formed from the continuous-time Jacobian J_c:
        Φ_k ≈ exp( J_c(x(t_k)) Δt_k )  or  Φ_k ≈ I + Δt_k J_c(x(t_k))
    Estimate:
        sum_i λ_i ≈ (1 / T_total) * Σ_k log|det Φ_k|,   T_total = Σ_k Δt_k.
    This mirrors the normalization in your QR/Eig implementations.

    Returns a scalar (per unit time). If keep=True, returns a running time-normalized history (length N-1).
    """
    N = len(time)
    assert len(radius) == N and len(velocity) == N, "All arrays must be the same length"
    if N < 2:
        raise ValueError("Need at least two samples")
    
    dt = float(time[1] - time[0])

    # Running Kahan sum
    total_logdet = 0.0
    c = 0.0
    elapsed = 0.0

    hist = np.empty(N, float) if keep else None

    for i in range(N):
        if equation_name == 'Rayleigh-Plesset':
            J = model.Jacobian_RP(radius[i], velocity[i], time[i])  # 2x2 Jacobian
        elif equation_name == 'Keller-Miksis':
            J = model.Jacobian_KM(radius[i], velocity[i], time[i])  # 2x2 Jacobian
        elif equation_name == 'Gilmore':
            J = model.Jacobian_G(radius[i], velocity[i], time[i])  # 2x2 Jacobian
        else:
            raise ValueError('Wrong equation name')

        # step = dt * float(np.trace(J))
        M = np.eye(J.shape[0]) + dt * J
        sign, logdet = np.linalg.slogdet(M)     # ~ dt*tr(J) + O(dt^2)
        step =  float(logdet) if sign != 0 else np.nan

        # Kahan summation of the numerator Σ log|det Φ_k|
        y = step - c
        s = total_logdet + y
        c = (s - total_logdet) - y
        total_logdet = s

        elapsed += dt

        if keep:
            hist[i] = total_logdet / elapsed  # per-unit-time running estimate

    sum_lce = total_logdet / elapsed  # per-unit-time
    return (sum_lce, hist) if keep else sum_lce