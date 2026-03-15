from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
import time

import numpy as np
import pandas as pd

import core.lyapunov as lyapunov
from models.bubble_models import simulate_bubble_trajectories
from models.lorenz import compute_lce_eckmann, compute_lce_rosenstein
from utils.logging_utils import get_logger, setup_logging

setup_logging(log_to_file=True)
logger = get_logger(__name__)


def main() -> None:
    logger.info("Starting reproduction of Table 3: numerical comparison.")

    Path("results").mkdir(parents=True, exist_ok=True)

    equation: str = "G"
    temperature: float = 20
    periods: int = 10
    step: float = 1e-3

    configs: list[tuple[str, float, float, float]] = [
        ("C1", 0.3e6, 1.2e6, 10e-6),
        ("C2", 1.5e6, 1.2e6, 5e-6),
    ]

    nolds_params: dict = {
        "C1": {"emb_dim": 5, "tau": 6, "min_tsep": 18, "trajectory_len": 5000},
        "C2": {"emb_dim": 5, "tau": 83, "min_tsep": 249, "trajectory_len": 5000},
    }

    all_results: dict = {}

    try:
        for idx, (configuration_name, acoustic_pressure, frequency, initial_radius) in enumerate(configs, start=1):
            logger.info(
                "[%d/%d] Running %s with equation=%s, pressure=%s, frequency=%s, initial_radius=%s",
                idx,
                len(configs),
                configuration_name,
                equation,
                acoustic_pressure,
                frequency,
                initial_radius,
            )

            integration_time = np.arange(0, periods / frequency, step / frequency)

            trajectories, model = simulate_bubble_trajectories(
                [equation],
                temperature,
                acoustic_pressure,
                frequency,
                initial_radius,
                integration_time,
                step,
            )

            radius_data = trajectories[f"Radius_{equation}"]
            velocity_data = trajectories[f"Velocity_{equation}"]

            results: dict = {}

            logger.info("%s: running QR method.", configuration_name)
            start_time = time.time()
            lce_qr = lyapunov.compute_lce_qr_from_trajectory(
                radius_data,
                velocity_data,
                integration_time * frequency,
                model,
                lyapunov.EQUATION_DISPLAY_NAMES[equation],
                keep=False,
            )
            results["QR"] = {"values": list(np.atleast_1d(lce_qr)), "time": time.time() - start_time}

            params_eck = {
                "emb_dim": nolds_params[configuration_name]["emb_dim"],
                "tau": nolds_params[configuration_name]["tau"],
                "min_tsep": nolds_params[configuration_name]["min_tsep"],
                "trajectory_len": nolds_params[configuration_name]["trajectory_len"],
                "matrix_dim": 2,
                "min_nb": 2 * nolds_params[configuration_name]["emb_dim"],
            }
            logger.info("%s: running Eckmann method.", configuration_name)
            start_time = time.time()
            lce_eck = compute_lce_eckmann(radius_data, step, params_eck)
            results["Eckmann"] = {"values": list(np.atleast_1d(lce_eck)), "time": time.time() - start_time}

            params_ros = {
                "emb_dim": nolds_params[configuration_name]["emb_dim"],
                "tau": nolds_params[configuration_name]["tau"],
                "min_tsep": nolds_params[configuration_name]["min_tsep"],
                "trajectory_len": nolds_params[configuration_name]["trajectory_len"],
                "matrix_dim": 2,
            }
            logger.info("%s: running Rosenstein method.", configuration_name)
            start_time = time.time()
            lce_ros = compute_lce_rosenstein(radius_data, step, params_ros)
            results["Rosenstein"] = {"values": list(np.atleast_1d(lce_ros)), "time": time.time() - start_time}

            logger.info("%s: running eigenvalue-product method.", configuration_name)
            start_time = time.time()
            eigenvalue_product_lce = lyapunov.compute_lce_from_eigenvalue_product_trajectory(
                radius_data,
                velocity_data,
                integration_time * frequency,
                model,
                lyapunov.EQUATION_DISPLAY_NAMES[equation],
                keep=False,
            )
            results["EigenvalueProduct"] = {
                "values": list(np.atleast_1d(eigenvalue_product_lce)),
                "time": time.time() - start_time,
            }

            logger.info("%s: running determinant-sum method.", configuration_name)
            start_time = time.time()
            determinant_sum_lce = lyapunov.compute_lce_sum_from_determinants_trajectory(
                radius_data,
                velocity_data,
                integration_time * frequency,
                model,
                lyapunov.EQUATION_DISPLAY_NAMES[equation],
                keep=False,
            )
            results["DeterminantSum"] = {
                "values": list(np.atleast_1d(determinant_sum_lce)),
                "time": time.time() - start_time,
            }

            all_results[configuration_name] = results
            logger.info("%s: all methods finished successfully.", configuration_name)

        table_rows: list[dict] = []
        for config, algos in all_results.items():
            for algo, vals in algos.items():
                values = vals.get("values", None)
                elapsed_time = vals.get("time", None)

                table_rows.append(
                    {
                        "Config": config,
                        "Algorithm": algo,
                        "Values": values,
                        "Time (s)": elapsed_time,
                    }
                )

        results_table = pd.DataFrame(table_rows)
        logger.info("Generated Table 3 dataframe with %d rows.", len(results_table))
        print(results_table.to_string(index=False))

    except Exception:
        logger.exception("Table 3 reproduction failed.")
        raise

    logger.info("Finished reproduction of Table 3.")


if __name__ == "__main__":
    main()