import pandas as pd
import numpy as np


from core.lyapunov import (
    compute_lyapunov_exponents_from_trajectory,
    compute_lyapunov_sum_from_determinants,
    compute_lyapunov_from_eigenvalue_product,
    equation_name_dd
)
from models.bubble_models import create_trajectories


configs: list[tuple[str, float, float, float]] = [
    ("C1", 0.3e6, 1.2e6, 10e-6),
    ("C2", 1.5e6, 1.2e6, 5e-6),
    ("C3", 2.0e6, 0.8e6, 0.08e-6),
]

temperature: float = 20.0  # Celsius
equation: str = "G"      # will be mapped with equation_name_dd
periods: int = 10
step: float = 1e-3         # non-dimensional dt in periods, since you multiply by frequency later

results: dict = {}
for label, presure, frequency, initial_radius in configs:
    print(f"\nRunning config {label}: P={presure:.2e}, f={frequency:.2e}, R0={initial_radius:.2e}")

    # Time grid: 10 periods with step scaled by frequency so that time*frequency is "cycles"
    time = np.arange(0, periods / frequency, step / frequency)

    trajectories, model = create_trajectories(
        [equation], temperature, presure, frequency, initial_radius, time, step
    )

    radius_data = trajectories[f'Radius_{equation}']
    velocity_data = trajectories[f'Velocity_{equation}']

    # --- Call your Lyapunov algorithms ---
    # 1) Eigenvalue product (keep history)
    eigvals, eig_hist = compute_lyapunov_from_eigenvalue_product(
        radius_data, velocity_data, time * frequency, model, equation_name_dd[equation], keep=True
    )

    # 2) Determinant method (sum only; keep history)
    sum_lce, det_hist = compute_lyapunov_sum_from_determinants(
        radius_data, velocity_data, time * frequency, model, equation_name_dd[equation], keep=True
    )

    # 3) QR (baseline; keep history)
    lce_qr, qr_hist = compute_lyapunov_exponents_from_trajectory(
        radius_data, velocity_data, time * frequency, model, equation_name_dd[equation], keep=True
    )

    # --- Store results ---
    results[label] = {
        "eig_product": {"final": eigvals, "history": eig_hist},
        "determinant": {"final": sum_lce, "history": det_hist},
        "qr": {"final": lce_qr, "history": qr_hist},
    }


invalid_threshold: float = 1e3   # adjust if your scale warrants a different bound

rows: list[dict] = []
for (label, _, _, _) in configs:
    # Pull histories
    eig_hist = results[label]["eig_product"]["history"]   # shape (N,2)
    det_hist = results[label]["determinant"]["history"]   # shape (N,)
    qr_hist  = results[label]["qr"]["history"]            # shape (N,2)

    # Eigenvalue Product
    invalid_eig = (np.any(~np.isfinite(eig_hist), axis=1) |
                   np.any(np.abs(eig_hist) > invalid_threshold, axis=1))
    pct_eig = 100.0 * invalid_eig.mean() if eig_hist.size else np.nan

    # Determinant Sum
    invalid_det = (~np.isfinite(det_hist)) | (np.abs(det_hist) > invalid_threshold)
    pct_det = 100.0 * invalid_det.mean() if det_hist.size else np.nan

    # QR baseline

    invalid_qr = (np.any(~np.isfinite(qr_hist), axis=1) |
                    np.any(np.abs(qr_hist) > invalid_threshold, axis=1))
    pct_qr = 100.0 * invalid_qr.mean() if qr_hist.size else np.nan
    rows.append({
        "Case": label,
        "Algorithm": "QR (baseline)",
        "% Invalid Timesteps": np.round(pct_qr, 1)
    })

    rows.append({
        "Case": label,
        "Algorithm": "Eigenvalue Product (before)",
        "% Invalid Timesteps": np.round(pct_eig, 1)
    })
    rows.append({
        "Case": label,
        "Algorithm": "Determinant Sum (before)",
        "% Invalid Timesteps": np.round(pct_det, 1)
    })

df_invalid_long = pd.DataFrame(rows)

# Wide table for paper
order_cases: list[str] = [c[0] for c in configs]
order_algs: list[str] = [
    "QR (baseline)",
    "Eigenvalue Product (before)",
    "Determinant Sum (before)"
]

table_invalid = (df_invalid_long
                 .pivot(index="Case", columns="Algorithm", values="% Invalid Timesteps")
                 .reindex(index=order_cases, columns=order_algs))

print("\n--- % INVALID TIMESTEPS OVER 10 PERIODS (Gilmore, T=20°C) ---")
print(table_invalid.to_string())