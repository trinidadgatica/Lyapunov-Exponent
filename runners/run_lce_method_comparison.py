import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

print(f"[DEBUG] Root added to sys.path: {project_root}")

from algorithms.method_comparison import (run_lce_method_comparison, compute_tail_mean)
from visualization.lyapunov_plotly import plot_lce_histories

# --- Parameters ---
equation = 'G'
temperature = 20
frequency = 1e6
pressure = 0.5e6
radius = 10e-6
periods = 10
step_sizes = [1e-3, 1e-4, 1e-5, 1e-6]

# --- Time Loop ---
df_tail = []
df_final = []

for step in step_sizes:
    times = np.arange(0, periods / frequency, step/ frequency)
    print(f"Δt = {step:.1e}, steps = {len(times)}")

    lce_qr, lce_eig, lce_det, hist_qr, hist_eig, hist_det = run_lce_method_comparison(
        equation, temperature, pressure, frequency, radius, times, step
    )

    tail_len = int(len(times) * 0.1)

    df_tail.append({
        "Δt": step,
        "QR λ₁": compute_tail_mean(hist_qr, index=0),
        "QR λ₂": compute_tail_mean(hist_qr, index=1),
        "Eigen λ₁": compute_tail_mean(hist_eig, index=0),
        "Eigen λ₂": compute_tail_mean(hist_eig, index=1),
        "Det λ₁": np.mean(hist_det[-tail_len:])
    })

    df_final.append({
        "Δt": step,
        "QR λ₁": hist_qr[-1, 0],
        "QR λ₂": hist_qr[-1, 1],
        "Eigen λ₁": hist_eig[-1, 0],
        "Eigen λ₂": hist_eig[-1, 1],
        "Det λ₁": hist_det[-1]
    })

    plot_lce_histories(times, hist_qr, hist_eig, hist_det, filename=f"results/lce_history_{step:.0e}.html")

# --- Results ---
df_tail = pd.DataFrame(df_tail).sort_values("Δt")
df_final = pd.DataFrame(df_final).sort_values("Δt")

print(df_tail)
print(df_final)

# --- Plot Final LCE1 vs Step ---
plt.figure(figsize=(8, 5))
plt.plot(df_final["Δt"], df_final["QR λ₁"], marker='o', label='QR')
plt.plot(df_final["Δt"], df_final["Eigen λ₁"], marker='s', label='Eigenvalue')
plt.plot(df_final["Δt"], df_final["Det λ₁"], marker='^', label='Determinant')
plt.xscale("log")
plt.xlabel("Step Size (Δt)")
plt.ylabel("Largest Lyapunov Exponent")
plt.title("Method Robustness: Lyapunov Exponent vs Step Size")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
 