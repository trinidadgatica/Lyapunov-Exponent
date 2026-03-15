from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.nolds_parameter_search import (
    EMB_GRID,
    TAU_GRID_BASE,
    THEILER_MULTS,
    TRAJ_LEN_FRACS,
    USE_LOG_RADIUS,
    find_best_params_grid,
)
from models.bubble_models import simulate_bubble_trajectories
from utils.logging_utils import get_logger, setup_logging

setup_logging(log_to_file=True)
logger = get_logger(__name__)


def main() -> None:
    equation = "G"
    temperature = 20
    periods = 10
    step = 1e-3

    configs = [
        ("C1", 0.3e6, 1.2e6, 10e-6),
        ("C2", 1.5e6, 1.2e6, 5e-6),
    ]

    logger.info(
        "Starting standalone parameter search runner for %d configurations.",
        len(configs),
    )

    for idx, (
        configuration_name,
        acoustic_pressure,
        frequency,
        initial_radius,
    ) in enumerate(configs, start=1):
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
        dt_series = float(np.median(np.diff(integration_time)))

        radius_data = None

        try:
            trajectories, _ = simulate_bubble_trajectories(
                [equation],
                temperature,
                acoustic_pressure,
                frequency,
                initial_radius,
                integration_time,
                step,
            )
            logger.info("%s: trajectories created successfully.", configuration_name)

            radius_data = trajectories[f"Radius_{equation}"]

            best = find_best_params_grid(
                radius=radius_data,
                dt=dt_series,
                drive_freq_hz=frequency,
                emb_grid=EMB_GRID,
                tau_grid_base=TAU_GRID_BASE,
                theiler_mults=THEILER_MULTS,
                traj_len_fracs=TRAJ_LEN_FRACS,
                use_log_radius=USE_LOG_RADIUS,
            )

            lam1, lam2 = best["eckmann"]["spectrum"][:2]
            lle = best["rosenstein"]["lle"]
            chosen = best["params"]

            logger.info("%s: chosen params = %s", configuration_name, chosen)
            logger.info(
                "%s: Eckmann λ1, λ2 [1/s] = %s, %s",
                configuration_name,
                lam1,
                lam2,
            )
            logger.info("%s: Rosenstein LLE [1/s] = %s", configuration_name, lle)

        except Exception:
            logger.exception(
                "%s: parameter search failed. N=%d, dt=%s, f=%s Hz",
                configuration_name,
                len(radius_data) if radius_data is not None else -1,
                dt_series,
                frequency,
            )
            raise

    logger.info("Finished standalone parameter search runner.")


if __name__ == "__main__":
    main()