from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from core.lyapunov import (
    compute_lce_qr_from_trajectory,
    compute_lce_from_eigenvalue_product,
    compute_lce_sum_from_determinants,
    EQUATION_DISPLAY_NAMES,
)
from models.bubble_models import simulate_bubble_trajectories
from utils.logging_utils import get_logger, setup_logging

setup_logging(log_to_file=True)
logger = get_logger(__name__)


def main() -> None:
    logger.info("Starting reproduction of Table 4: reliability comparison.")

    Path("results").mkdir(parents=True, exist_ok=True)

    configs: list[tuple[str, float, float, float]] = [
        ("C1", 0.3e6, 1.2e6, 10e-6),
        ("C2", 1.5e6, 1.2e6, 5e-6),
        ("C3", 2.0e6, 0.8e6, 0.08e-6),
    ]

    temperature: float = 20.0
    equation: str = "G"
    periods: int = 10
    step: float = 1e-3

    results: dict = {}

    try:
        for idx, (label, pressure, frequency, initial_radius) in enumerate(configs, start=1):
            logger.info(
                "[%d/%d] Running config %s: P=%s, f=%s, R0=%s",
                idx,
                len(configs),
                label,
                pressure,
                frequency,
                initial_radius,
            )

            time = np.arange(0, periods / frequency, step / frequency)

            trajectories, model = simulate_bubble_trajectories(
                [equation],
                temperature,
                pressure,
                frequency,
                initial_radius,
                time,
                step,
            )

            radius_data = trajectories[f"Radius_{equation}"]
            velocity_data = trajectories[f"Velocity_{equation}"]

            logger.info("%s: running eigenvalue-product method.", label)
            eigvals, eig_hist = compute_lce_from_eigenvalue_product(
                radius_data,
                velocity_data,
                time * frequency,
                model,
                EQUATION_DISPLAY_NAMES[equation],
                keep=True,
            )

            logger.info("%s: running determinant method.", label)
            sum_lce, det_hist = compute_lce_sum_from_determinants(
                radius_data,
                velocity_data,
                time * frequency,
                model,
                EQUATION_DISPLAY_NAMES[equation],
                keep=True,
            )

            logger.info("%s: running QR method.", label)
            lce_qr, qr_hist = compute_lce_qr_from_trajectory(
                radius_data,
                velocity_data,
                time * frequency,
                model,
                EQUATION_DISPLAY_NAMES[equation],
                keep=True,
            )

            results[label] = {
                "eig_product": {"final": eigvals, "history": eig_hist},
                "determinant": {"final": sum_lce, "history": det_hist},
                "qr": {"final": lce_qr, "history": qr_hist},
            }
            logger.info("%s: finished successfully.", label)

        invalid_threshold: float = 1e3
        rows: list[dict] = []

        for label, _, _, _ in configs:
            eig_hist = results[label]["eig_product"]["history"]
            det_hist = results[label]["determinant"]["history"]
            qr_hist = results[label]["qr"]["history"]

            invalid_eig = (
                np.any(~np.isfinite(eig_hist), axis=1)
                | np.any(np.abs(eig_hist) > invalid_threshold, axis=1)
            )
            pct_eig = 100.0 * invalid_eig.mean() if eig_hist.size else np.nan

            invalid_det = (~np.isfinite(det_hist)) | (np.abs(det_hist) > invalid_threshold)
            pct_det = 100.0 * invalid_det.mean() if det_hist.size else np.nan

            invalid_qr = (
                np.any(~np.isfinite(qr_hist), axis=1)
                | np.any(np.abs(qr_hist) > invalid_threshold, axis=1)
            )
            pct_qr = 100.0 * invalid_qr.mean() if qr_hist.size else np.nan

            rows.append(
                {
                    "Case": label,
                    "Algorithm": "QR (baseline)",
                    "% Invalid Timesteps": np.round(pct_qr, 1),
                }
            )
            rows.append(
                {
                    "Case": label,
                    "Algorithm": "Eigenvalue Product (before)",
                    "% Invalid Timesteps": np.round(pct_eig, 1),
                }
            )
            rows.append(
                {
                    "Case": label,
                    "Algorithm": "Determinant Sum (before)",
                    "% Invalid Timesteps": np.round(pct_det, 1),
                }
            )

        df_invalid_long = pd.DataFrame(rows)

        order_cases: list[str] = [c[0] for c in configs]
        order_algs: list[str] = [
            "QR (baseline)",
            "Eigenvalue Product (before)",
            "Determinant Sum (before)",
        ]

        table_invalid = (
            df_invalid_long.pivot(
                index="Case",
                columns="Algorithm",
                values="% Invalid Timesteps",
            ).reindex(index=order_cases, columns=order_algs)
        )

        logger.info("Generated Table 4 dataframe with %d rows.", len(df_invalid_long))
        print("\n--- % INVALID TIMESTEPS OVER 10 PERIODS (Gilmore, T=20°C) ---")
        print(table_invalid.to_string())

    except Exception:
        logger.exception("Table 4 reproduction failed.")
        raise

    logger.info("Finished reproduction of Table 4.")


if __name__ == "__main__":
    main()