from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from experiments.method_comparison import (
    get_final_period_indices,
    last_period_summary_table,
    run_method_comparison_experiment,
)
from utils.logging_utils import get_logger, setup_logging

setup_logging(log_to_file=True)
logger = get_logger(__name__)


def main() -> None:
    logger.info("Starting reproduction of Table 6: method comparison.")

    Path("results").mkdir(parents=True, exist_ok=True)

    equation: str = "G"
    temp_celsius: float = 20
    n_periods: int = 10
    dt: float = 1e-3

    configs: list[tuple[str, float, float, float]] = [
        ("C1", 0.3e6, 1.2e6, 10e-6),
        ("C2", 1.5e6, 1.2e6, 5e-6),
    ]

    tables_by_cfg: dict[str, pd.DataFrame] = {}

    try:
        for idx, (cfg_name, p_init_pa, freq_hz, r_init_m) in enumerate(configs, start=1):
            logger.info(
                "[%d/%d] Running %s with dt=%s, pressure=%s, frequency=%s, initial_radius=%s",
                idx,
                len(configs),
                cfg_name,
                dt,
                p_init_pa,
                freq_hz,
                r_init_m,
            )

            t_grid = np.arange(0, n_periods / freq_hz, dt / freq_hz)
            last_period_indices = get_final_period_indices(t_grid, freq_hz, n_periods, 2)

            logger.info("%s: computing LCE histories.", cfg_name)
            lce_qr, lce_eig, lce_det, hist_qr, hist_eig, hist_det = run_method_comparison_experiment(
                equation,
                temp_celsius,
                p_init_pa,
                freq_hz,
                r_init_m,
                t_grid,
                dt,
            )

            last_period_qr = hist_qr[last_period_indices[0] : last_period_indices[-1], :]
            last_period_eig = hist_eig[last_period_indices[0] : last_period_indices[-1], :]
            last_period_det = hist_det[last_period_indices[0] : last_period_indices[-1]]

            last_period_samples = {
                "QR λ1": last_period_qr[:, 0],
                "QR λ2": last_period_qr[:, 1],
                "Eigen λ1": last_period_eig[:, 0],
                "Eigen λ2": last_period_eig[:, 1],
                "Det λ1": last_period_det,
            }
            last_period_means = {
                "QR λ1": float(np.mean(last_period_qr[:, 0])),
                "QR λ2": float(np.mean(last_period_qr[:, 1])),
                "Eigen λ1": float(np.mean(last_period_eig[:, 0])),
                "Eigen λ2": float(np.mean(last_period_eig[:, 1])),
                "Det λ1": float(np.mean(last_period_det)),
            }

            table = last_period_summary_table(
                last_period_samples,
                last_period_means,
                order=["QR λ1", "QR λ2", "Eigen λ1", "Eigen λ2", "Det λ1"],
                decimals=3,
            )

            table.insert(0, "Config", cfg_name)
            table.insert(1, "Δt", dt)

            print(table)
            tables_by_cfg[cfg_name] = table
            logger.info("%s: summary table generated.", cfg_name)

        out_path = Path("results/last_period_lce_summary_by_config.xlsx")
        logger.info("Saving Excel workbook to %s", out_path)

        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            for cfg_name, df in tables_by_cfg.items():
                df.to_excel(writer, sheet_name=cfg_name[:31], index=False)

        if not out_path.exists():
            logger.error("Expected output workbook was not created: %s", out_path)
            raise FileNotFoundError(f"Missing output workbook: {out_path}")

        logger.info("Saved Excel workbook successfully: %s", out_path)

    except Exception:
        logger.exception("Table 6 reproduction failed.")
        raise

    logger.info("Finished reproduction of Table 6.")


if __name__ == "__main__":
    main()