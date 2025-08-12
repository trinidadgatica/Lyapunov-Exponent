import numpy as np
from algorithms.dynamics import create_trajectories

equation_name_dd = dict()
equation_name_dd['RP'] = 'Rayleigh-Plesset'
equation_name_dd['KM'] = 'Keller-Miksis'
equation_name_dd['G'] = 'Gilmore'


def advance_tangent_linear_map(W, J, dt):
    """
    One step RK4 for tangent linear map evolution.
    """
    k1 = J @ W
    k2 = J @ (W + 0.5 * dt * k1)
    k3 = J @ (W + 0.5 * dt * k2)
    k4 = J @ (W + dt * k3)
    return W + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def compute_lyapunov_exponents_from_trajectory(radius, velocity, time, model, equation_name, keep=False):
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
        
        W = advance_tangent_linear_map(W, J, dt)

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


def find_cut_index(signal, tolerance=1e-6, min_consecutive=3):
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


def compute_lyapunov_grid(grid, equation, temperature, frequency=None, pressure=None, filename_suffix=""):
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

            trajectories, model = create_trajectories(
                [equation], temperature, P, f, R0, times, step)

            radius = trajectories[f"Radius_{equation}"]
            velocity = trajectories[f"Velocity_{equation}"]

            cut_idx = min(find_cut_index(radius), find_cut_index(velocity, 1e-10))

            r_next = radius[cut_idx + 1] if cut_idx < len(radius) - 1 else radius[-1]
            v_next = velocity[cut_idx + 1] if cut_idx < len(velocity) - 1 else velocity[-1]

            log_file.write(f"{equation}\t{cut_idx}\t{r_next:.6e}\t{v_next:.6e}\n")

            exps = compute_lyapunov_exponents_from_trajectory(
                radius[:cut_idx],
                velocity[:cut_idx],
                time[:cut_idx],
                model,
                equation_name_dd[equation],
                keep=False
            )
            exponents.append(exps)

    return exponents


def compute_eigenvalues(radius, velocity, time, model, equation_name):
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


def compute_lyapunov_from_eigenvalue_product(radius, velocity, time, general_model, equation_name, keep=False):
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


def compute_lyapunov_sum_from_determinants(radius, velocity, time, general_model, equation_name, keep=False):
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
