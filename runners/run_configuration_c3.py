import time
import numpy as np
import pandas as pd

from models.bubble_models import create_trajectories
from core import lyapunov as lya
from models.lorenz import compute_lce_eckmann, compute_lce_rosenstein

# ================== FIXED / USER PARAMS (C3) ==================
equation: str = "G"     # 'RP', 'KM', 'G'
temperature: float = 20  # Celsius
periods: int = 10

configuration_name = "C3"
acoustic_pressure = 2.0e6   # Pa
frequency = 0.8e6           # Hz  
initial_radius = 0.08e-6    # m

# Integrator step control for create_trajectories: ALWAYS 1e-3
step_for_creator = 1e-3

# Trim length before collapse
N_KEEP = 1061

# Base time-series params (will clamp trajectory_len safely later)
base_params = {
    "emb_dim": 5,
    "matrix_dim": 2,
    "tau": 6,
    "min_tsep": 18,          # Theiler window in samples
    "trajectory_len": 1061,  # desired; will be clamped as needed
    "use_log_radius": False,
}

print(f"\n=== Running {configuration_name} (equation={equation}) ===")

# ================== Build time vector (like your original runner) ==================
# NOTE: integration_time spacing = step_for_creator / frequency
integration_time = np.arange(0.0, periods / frequency, step_for_creator / frequency)

# ================== Simulate ==================
trajectories, model = create_trajectories(
    [equation],
    temperature,
    acoustic_pressure,
    frequency,
    initial_radius,
    integration_time,
    step_for_creator
)

radius_data   = np.asarray(trajectories[f"Radius_{equation}"], float)
velocity_data = np.asarray(trajectories[f"Velocity_{equation}"], float)

# ================== Cut to first 1061 points ==================
if len(integration_time) < N_KEEP:
    raise ValueError(
        f"Series has only {len(integration_time)} points (< {N_KEEP}). "
        f"Increase integration span or lower step to ensure at least {N_KEEP} samples."
    )

radius_data       = radius_data[:N_KEEP]
velocity_data     = velocity_data[:N_KEEP]
integration_time  = integration_time[:N_KEEP]

# True sampling interval of the time series (seconds/sample)
dt_series = float(np.median(np.diff(integration_time)))  # should be 1e-3 / 0.8e6 = 1.25e-9

# Optional log-transform for radius if requested
if base_params["use_log_radius"]:
    eps = np.finfo(float).eps
    radius_data = np.log(np.maximum(radius_data, eps))

# ================== Prepare Eckmann & Rosenstein params ==================
N = len(radius_data)
emb_dim = base_params["emb_dim"]
tau = base_params["tau"]
min_tsep = base_params["min_tsep"]
matrix_dim = base_params["matrix_dim"]

# Conservative clamp for Rosenstein trajectory_len:
# Remove embedding overhead ( (m-1)*tau ), at least one extra step, and min_tsep
max_traj_len = N - (emb_dim - 1) * tau - 1 - min_tsep
# Keep it in a reasonable range
max_traj_len = int(max(20, min(max_traj_len, N // 3)))

traj_len_req = int(base_params["trajectory_len"])
traj_len_eff = int(min(traj_len_req, max_traj_len))

if traj_len_eff < traj_len_req:
    print(
        f"[WARN] trajectory_len clamped from {traj_len_req} to {traj_len_eff} "
        f"(N={N}, emb_dim={emb_dim}, tau={tau}, min_tsep={min_tsep})."
    )

# Separate dicts (Eckmann vs Rosenstein)
params_eck = {
    "emb_dim": emb_dim,
    "tau": tau,
    "min_tsep": min_tsep,
    "matrix_dim": matrix_dim,
    "min_nb": 2 * emb_dim,   # typical robust choice
}
params_ros = {
    "emb_dim": emb_dim,
    "tau": tau,
    "min_tsep": min_tsep,
    "trajectory_len": traj_len_eff,
    "matrix_dim": matrix_dim,   # harmless if ignored by your impl
}

# ================== RUN ALL METHODS ==================
results: dict = {}

# ---------- QR ----------
t0 = time.time()
lce_qr = lya.compute_lyapunov_exponents_from_trajectory(
    radius_data,
    velocity_data,
    integration_time * frequency,     # keep your scaling convention (time * f)
    model,
    lya.equation_name_dd[equation],
    keep=False
)
results["QR"] = {"values": list(np.atleast_1d(lce_qr)), "time": time.time() - t0}

# ---------- Eckmann (spectrum) ----------
t0 = time.time()
lce_eck = compute_lce_eckmann(radius_data, dt_series, params_eck)
results["Eckmann"] = {"values": list(np.atleast_1d(lce_eck)), "time": time.time() - t0}

# ---------- Rosenstein (LLE) ----------
t0 = time.time()
lce_ros = compute_lce_rosenstein(radius_data, dt_series, params_ros)
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

# ================== PRINT TABLE ==================
rows = []
for algo, out in results.items():
    rows.append({
        "Config": configuration_name,
        "Algorithm": algo,
        "Values": out.get("values"),
        "Time (s)": out.get("time"),
    })

df = pd.DataFrame(rows, columns=["Config", "Algorithm", "Values", "Time (s)"])
print(df.to_string(index=False))
