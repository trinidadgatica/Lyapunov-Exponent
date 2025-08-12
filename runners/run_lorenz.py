import sys
import os
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

print(f"[DEBUG] Root added to sys.path: {project_root}")

from algorithms.lorenz_lyapunov import lorenz, compute_lce_qr_lorenz, compute_lce_eigprod_lorenz, compute_lce_det_lorenz


sigma = 10
beta = 8 / 3
rho_values = [28, 0.5] # other possible values are 99.96, 166.30

t0, t1, dt = 0, 50, 0.01
t_eval = np.arange(t0, t1, dt)

results = {
    "QR": {},
    "Det": {},
    "Eigen": {}
}

histories = {
    "QR": {},
    "Det": {},
    "Eigen": {}
}

for rho in rho_values:
    sol = solve_ivp(
        lorenz,
        (t0, t1),
        [1.0, 1.0, 1.0],
        t_eval=t_eval,
        args=(sigma, rho, beta),
        rtol=1e-9,
        atol=1e-9
    )
    x, y, z = sol.y
    time = sol.t

    lce_vals_analytical, history_analytical = compute_lce_qr_lorenz(x, y, z, time, sigma, rho, beta, keep=True)
    sum_LCE, history_determinant = compute_lce_det_lorenz(x, y, z, time, sigma, rho, beta, keep=True)
    lce_vals_eigenvalue, history_eigenvalue = compute_lce_eigprod_lorenz(x, y, z, time, sigma, rho, beta, keep=True)

    results["QR"][rho] = lce_vals_analytical
    results["Det"][rho] = [sum_LCE, None, None]  
    results["Eigen"][rho] = lce_vals_eigenvalue

    histories["QR"][rho] = history_analytical
    histories["Det"][rho] = history_determinant
    histories["Eigen"][rho] = history_eigenvalue


methods = ['QR', 'Det', 'Eigen']
rows = []

for method in methods:
    for rho, exponents in results[method].items():
        row = {
            'Method': method,
            'Rho': rho,
            'LCE1': exponents[0],
            'LCE2': exponents[1],
            'LCE3': exponents[2]
        }
        rows.append(row)

df = pd.DataFrame(rows)
df = df.sort_values(['Rho', 'Method']).reset_index(drop=True)

print(df)