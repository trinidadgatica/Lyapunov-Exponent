import numpy as np

def lorenz(t, state, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]


def jacobian_lorenz(x, y, z, sigma, rho, beta):
    return np.array([
        [-sigma, sigma, 0],
        [rho - z, -1, -x],
        [y, x, -beta]
    ])


def compute_lce_qr_lorenz(x, y, z, time, sigma, rho, beta, keep=False):
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


def compute_lce_det_lorenz(x, y, z, time, sigma, rho, beta, keep=False):
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


def compute_lce_eigprod_lorenz(x, y, z, time, sigma, rho, beta, keep=False):
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
