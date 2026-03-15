from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

import core.lyapunov as lyapunov
from models.bubble_models import simulate_bubble_trajectories
from models.lorenz import compute_lce_eckmann, compute_lce_rosenstein
from utils.logging_utils import get_logger, setup_logging

setup_logging(log_to_file=True)
logger = get_logger(__name__)


def main() -> None:
    logger.info("Starting reproduction of Table 7: configuration C3.")

    Path("results").mkdir(parents=True, exist_ok=True)

    equation: str = "G"
    temperature: float = 20
    periods: int = 10

    configuration_name = "C3"
    acoustic_pressure = 2.0e6
    frequency = 0.8e6
    initial_radius = 0.08e-6

    step_for_creator = 1e-3
    n_keep = 1061

    base_params = {
        "emb_dim": 5,
        "matrix_dim": 2,
        "tau": 6,
        "min_tsep": 18,
        "trajectory_len": 1061,
        "use_log_radius": False,
    }

    try:
        logger.info(
            "Running %s with equation=%s, pressure=%s, frequency=%s, initial_radius=%s",
            configuration_name,
            equation,
            acoustic_pressure,
            frequency,
            initial_radius,
        )

        integration_time = np.arange(
            0.0,
            periods / frequency,
            step_for_creator / frequency,
        )

        trajectories, model = simulate_bubble_trajectories(
            [equation],
            temperature,
            acoustic_pressure,
            frequency,
            initial_radius,
            integration_time,
            step_for_creator,
        )

        radius_data = np.asarray(trajectories[f"Radius_{equation}"], float)
        velocity_data = np.asarray(trajectories[f"Velocity_{equation}"], float)

        if len(integration_time) < n_keep:
            logger.error(
                "Series has only %d points (< %d).",
                len(integration_time),
                n_keep,
            )
            raise ValueError(
                f"Series has only {len(integration_time)} points (< {n_keep}). "
                f"Increase integration span or lower step to ensure at least {n_keep} samples."
            )

        logger.info("Trimming series to first %d points.", n_keep)
        radius_data = radius_data[:n_keep]
        velocity_data = velocity_data[:n_keep]
        integration_time = integration_time[:n_keep]

        dt_series = float(np.median(np.diff(integration_time)))

        if base_params["use_log_radius"]:
            logger.info("Applying log transform to radius series.")
            eps = np.finfo(float).eps
            radius_data = np.log(np.maximum(radius_data, eps))

        n = len(radius_data)
        emb_dim = base_params["emb_dim"]
        tau = base_params["tau"]
        min_tsep = base_params["min_tsep"]
        matrix_dim = base_params["matrix_dim"]

        max_traj_len = n - (emb_dim - 1) * tau - 1 - min_tsep
        max_traj_len = int(max(20, min(max_traj_len, n // 3)))

        traj_len_req = int(base_params["trajectory_len"])
        traj_len_eff = int(min(traj_len_req, max_traj_len))

        if traj_len_eff < traj_len_req:
            logger.warning(
                "trajectory_len clamped from %d to %d (N=%d, emb_dim=%d, tau=%d, min_tsep=%d).",
                traj_len_req,
                traj_len_eff,
                n,
                emb_dim,
                tau,
                min_tsep,
            )

        params_eck = {
            "emb_dim": emb_dim,
            "tau": tau,
            "min_tsep": min_tsep,
            "matrix_dim": matrix_dim,
            "min_nb": 2 * emb_dim,
        }
        params_ros = {
            "emb_dim": emb_dim,
            "tau": tau,
            "min_tsep": min_tsep,
            "trajectory_len": traj_len_eff,
            "matrix_dim": matrix_dim,
        }

        results: dict = {}

        logger.info("%s: running QR method.", configuration_name)
        t0 = time.time()
        lce_qr = lyapunov.compute_lce_qr_from_trajectory(
            radius_data,
            velocity_data,
            integration_time * frequency,
            model,
            lyapunov.EQUATION_DISPLAY_NAMES[equation],
            keep=False,
        )
        results["QR"] = {"values": list(np.atleast_1d(lce_qr)), "time": time.time() - t0}

        logger.info("%s: running Eckmann method.", configuration_name)
        t0 = time.time()
        lce_eck = compute_lce_eckmann(radius_data, dt_series, params_eck)
        results["Eckmann"] = {"values": list(np.atleast_1d(lce_eck)), "time": time.time() - t0}

        logger.info("%s: running Rosenstein method.", configuration_name)
        t0 = time.time()
        lce_ros = compute_lce_rosenstein(radius_data, dt_series, params_ros)
        results["Rosenstein"] = {"values": list(np.atleast_1d(lce_ros)), "time": time.time() - t0}

        logger.info("%s: running eigenvalue-product method.", configuration_name)
        t0 = time.time()
        eigvals = lyapunov.compute_lce_from_eigenvalue_product_trajectory(
            radius_data,
            velocity_data,
            integration_time * frequency,
            model,
            lyapunov.EQUATION_DISPLAY_NAMES[equation],
            keep=False,
        )
        results["EigenvalueProduct"] = {
            "values": list(np.atleast_1d(eigvals)),
            "time": time.time() - t0,
        }

        logger.info("%s: running determinant-sum method.", configuration_name)
        t0 = time.time()
        sum_lce = lyapunov.compute_lce_sum_from_determinants_trajectory(
            radius_data,
            velocity_data,
            integration_time * frequency,
            model,
            lyapunov.EQUATION_DISPLAY_NAMES[equation],
            keep=False,
        )
        results["DeterminantSum"] = {
            "values": list(np.atleast_1d(sum_lce)),
            "time": time.time() - t0,
        }

        rows = []
        for algo, out in results.items():
            rows.append(
                {
                    "Config": configuration_name,
                    "Algorithm": algo,
                    "Values": out.get("values"),
                    "Time (s)": out.get("time"),
                }
            )

        df = pd.DataFrame(rows, columns=["Config", "Algorithm", "Values", "Time (s)"])
        logger.info("Generated Table 7 dataframe with %d rows.", len(df))
        print(df.to_string(index=False))

    except Exception:
        logger.exception("Table 7 reproduction failed.")
        raise

    logger.info("Finished reproduction of Table 7.")


if __name__ == "__main__":
    main()