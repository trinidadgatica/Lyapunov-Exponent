from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.nolds_parameter_search import find_best_params_grid
from models.bubble_models import simulate_bubble_trajectories
from utils.logging_utils import get_logger, setup_logging

setup_logging(log_to_file=True)
logger = get_logger(__name__)


def main() -> None:
    logger.info("Starting C3 parameter search runner.")

    Path("results").mkdir(parents=True, exist_ok=True)

    # Fixed setup
    equation: str = "G"
    temperature: float = 20.0
    periods: int = 10
    step: float = 1e-3

    # C3 setup
    configuration_name: str = "C3"
    acoustic_pressure: float = 2.0e6   # Pa
    frequency: float = 0.8e6           # Hz
    initial_radius: float = 0.08e-6    # m

    # Parameter-search setup
    n_keep: int = 1061
    emb_grid: tuple[int, ...] = (4, 5, 6, 7, 8)
    tau_grid_base: tuple[int, ...] = (2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 64, 83)
    theiler_mults: tuple[int, ...] = (1, 2, 3)
    traj_len_fracs: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25, 0.30)
    use_log_radius: bool = False

    logger.info(
        "Running %s with equation=%s, pressure=%s, frequency=%s, initial_radius=%s",
        configuration_name,
        equation,
        acoustic_pressure,
        frequency,
        initial_radius,
    )

    integration_time = np.arange(0.0, periods / frequency, step / frequency)

    trajectories, _model = simulate_bubble_trajectories(
        [equation],
        temperature,
        acoustic_pressure,
        frequency,
        initial_radius,
        integration_time,
        step,
    )

    total_samples = len(integration_time)
    if total_samples < n_keep:
        logger.error("Series has only %d points (< %d).", total_samples, n_keep)
        raise ValueError(
            f"Series has only {total_samples} points (< {n_keep}). "
            f"Increase integration span or lower step to ensure at least {n_keep} samples."
        )

    logger.info("Trimming series to first %d samples.", n_keep)
    trimmed_time = integration_time[:n_keep]
    radius_series = np.asarray(trajectories[f"Radius_{equation}"], float)[:n_keep]

    dt_series = float(np.median(np.diff(trimmed_time)))
    logger.info("Computed trimmed dt=%s", dt_series)

    try:
        best = find_best_params_grid(
            radius=radius_series,
            dt=dt_series,
            drive_freq_hz=frequency,
            emb_grid=emb_grid,
            tau_grid_base=tau_grid_base,
            theiler_mults=theiler_mults,
            traj_len_fracs=traj_len_fracs,
            use_log_radius=use_log_radius,
        )

        chosen_params = best["params"]
        lam1, lam2 = best["eckmann"]["spectrum"][:2]
        rosenstein_lle = best["rosenstein"]["lle"]

        logger.info("Chosen parameters: %s", chosen_params)
        logger.info("Eckmann λ1, λ2 [1/s]: %s, %s", lam1, lam2)
        logger.info("Rosenstein LLE [1/s]: %s", rosenstein_lle)

        print("\n=== C3 parameter search result ===")
        print("Chosen params:", chosen_params)
        print("Eckmann λ1, λ2 [1/s]:", lam1, lam2)
        print("Rosenstein LLE [1/s]:", rosenstein_lle)

    except Exception:
        logger.exception("%s parameter search failed.", configuration_name)
        raise

    logger.info("Finished C3 parameter search runner.")


if __name__ == "__main__":
    main()