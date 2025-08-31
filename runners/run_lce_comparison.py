import numpy as np
import pandas as pd

from algorithms.method_comparison import (
    get_last_period_indices,
    run_lce_method_comparison,
    last1_summary_table,   
)

# --- Parameters ---
equation: str = "G"
temp_celsius: float = 20
n_periods: int = 10
dt: float = 1e-3                          


configs: list[tuple[str, float, float, float]] = [
    ("C1", 0.3e6, 1.2e6, 10e-6),
    ("C2", 1.5e6, 1.2e6, 5e-6)
]


tables_by_cfg: dict[str, pd.DataFrame] = {}

for cfg_name, p_init_pa, freq_hz, r_init_m in configs:
    print(f"\n=== Running {cfg_name} @ Δt={dt:.1e} ===")

    # time grid and last-1-period indices
    t_grid = np.arange(0, n_periods / freq_hz, dt / freq_hz)
    idx_last1 = get_last_period_indices(t_grid, freq_hz, n_periods, 1)

    # integrate & compute histories
    lce_qr, lce_eig, lce_det, hist_qr, hist_eig, hist_det = run_lce_method_comparison(
        equation, temp_celsius, p_init_pa, freq_hz, r_init_m, t_grid, dt
    )

    # slices for last 1 period
    seg_qr_last1  = hist_qr[idx_last1[0]: idx_last1[-1], :]
    seg_eig_last1 = hist_eig[idx_last1[0]: idx_last1[-1], :]
    seg_det_last1 = hist_det[idx_last1[0]: idx_last1[-1]]

    last1 = {
        "QR λ1":    seg_qr_last1[:, 0],
        "QR λ2":    seg_qr_last1[:, 1],
        "Eigen λ1": seg_eig_last1[:, 0],
        "Eigen λ2": seg_eig_last1[:, 1],
        "Det λ1":   seg_det_last1,
    }
    finals = {
        "QR λ1":    float(hist_qr[-1, 0]),
        "QR λ2":    float(hist_qr[-1, 1]),
        "Eigen λ1": float(hist_eig[-1, 0]),
        "Eigen λ2": float(hist_eig[-1, 1]),
        "Det λ1":   float(hist_det[-1]),
    }

    table = last1_summary_table(
        last1, finals,
        order=["QR λ1", "QR λ2", "Eigen λ1", "Eigen λ2", "Det λ1"],
        decimals=3,
    )

    # add helpful columns
    table.insert(0, "Config", cfg_name)
    table.insert(1, "Δt", dt)

    print(table)
    tables_by_cfg[cfg_name] = table


out_path = "results/Last1_LCE_summary_by_config.xlsx"
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    for cfg_name, df in tables_by_cfg.items():
        df.to_excel(writer, sheet_name=cfg_name[:31], index=False)

print(f"\nSaved: {out_path}")
