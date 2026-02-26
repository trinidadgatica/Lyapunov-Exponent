import time
import numpy as np
import pandas as pd

from models.bubble_models import create_trajectories
from core import lyapunov as lya
from models.lorenz import compute_lce_eckmann, compute_lce_rosenstein

# ================== USER PARAMS ==================
equation: str = 'G'      # 'RP', 'KM', 'G'
temperature: float = 20
periods: int = 10
step: float = 1e-3

configs: list[tuple[str, float, float, float]] = [
    ("C1", 0.3e6, 1.2e6, 10e-6),
    ("C2", 1.5e6, 1.2e6, 5e-6),
]

# ================== NOLDS PARAMS ==================
nolds_params: dict = {
    "C1": {"emb_dim": 5, "tau": 6, "min_tsep": 18, "trajectory_len": 5000},
    "C2": {"emb_dim": 5, "tau": 83, "min_tsep": 249, "trajectory_len": 5000},
}

# ================== RUN ==================
all_results: dict = {}

for configuration_name, acoustic_pressure, frequency, initial_radius in configs:
    print(f"\n=== Running {configuration_name} ===")
    integration_time = np.arange(0, periods / frequency, step / frequency)

    trajectories, model = create_trajectories(
        [equation], temperature, acoustic_pressure, frequency,
        initial_radius, integration_time, step
    )

    radius_data   = trajectories[f'Radius_{equation}']
    velocity_data = trajectories[f'Velocity_{equation}']

    results: dict = {}

    # ---------- QR ----------
    t0 = time.time()
    lce_qr = lya.compute_lyapunov_exponents_from_trajectory(
        radius_data, velocity_data, integration_time * frequency,
        model, lya.equation_name_dd[equation], keep=False
    )
    results["QR"] = {"values": list(np.atleast_1d(lce_qr)), "time": time.time() - t0}

    # ---------- Eckmann ----------
    params_eck = {
        "emb_dim": nolds_params[configuration_name]["emb_dim"],
        "tau": nolds_params[configuration_name]["tau"],
        "min_tsep": nolds_params[configuration_name]["min_tsep"],
        "trajectory_len": nolds_params[configuration_name]["trajectory_len"],
        "matrix_dim": 2,
        "min_nb": 2 * nolds_params[configuration_name]["emb_dim"],
    }
    t0 = time.time()
    lce_eck = compute_lce_eckmann(radius_data, step, params_eck)
    results["Eckmann"] = {"values": list(np.atleast_1d(lce_eck)), "time": time.time() - t0}

    # ---------- Rosenstein ----------
    params_ros = {
        "emb_dim": nolds_params[configuration_name]["emb_dim"],
        "tau": nolds_params[configuration_name]["tau"],
        "min_tsep": nolds_params[configuration_name]["min_tsep"],
        "trajectory_len": nolds_params[configuration_name]["trajectory_len"],
        "matrix_dim": 2,
    }
    t0 = time.time()
    lce_ros = compute_lce_rosenstein(radius_data, step, params_ros)
    results["Rosenstein"] = {"values": list(np.atleast_1d(lce_ros)), "time": time.time() - t0}

    # ---------- Eigenvalue-product ----------
    t0 = time.time()
    eigvals = lya.compute_lyapunov_from_eigenvalue_product_fixed(
        radius_data, velocity_data, integration_time * frequency,
        model, lya.equation_name_dd[equation], keep=False
    )
    results["EigenvalueProduct"] = {"values": list(np.atleast_1d(eigvals)), "time": time.time() - t0}

    # ---------- Determinant-sum ----------
    t0 = time.time()
    sum_lce = lya.compute_lyapunov_sum_from_determinants_fixed(
        radius_data, velocity_data, integration_time * frequency,
        model, lya.equation_name_dd[equation], keep=False
    )
    results["DeterminantSum"] = {"values": list(np.atleast_1d(sum_lce)), "time": time.time() - t0}

    # store per config
    all_results[configuration_name] = results

# ================== PRINT ==================
rows: list[dict] = []
for config, algos in all_results.items():
    for algo, vals in algos.items():
        if isinstance(vals, dict):  # normal case
            values = vals.get("values", None)
            t = vals.get("time", None)

        rows.append({
            "Config": config,
            "Algorithm": algo,
            "Values": values,
            "Time (s)": t
        })

df = pd.DataFrame(rows)
print(df.to_string(index=False))