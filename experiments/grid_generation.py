from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np

from core.lyapunov import compute_lce_grid
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def generate_fixed_frequency_scans(
    *,
    frequency: float,
    temperature: float,
    n_points: int,
    initial_radius_min: float,
    initial_radius_max: float,
    acoustic_pressure_min: float,
    acoustic_pressure_max: float,
    results_dir: str | Path = "results",
) -> None:
    """
    Generate Lyapunov grids for the fixed-frequency experiment.

    Parameters
    ----------
    frequency : float
        Fixed driving frequency in Hz.
    temperature : float
        Temperature in degrees Celsius.
    n_points : int
        Number of grid points per axis.
    initial_radius_min : float
        Minimum initial radius.
    initial_radius_max : float
        Maximum initial radius.
    acoustic_pressure_min : float
        Minimum acoustic pressure in MPa.
    acoustic_pressure_max : float
        Maximum acoustic pressure in MPa.
    results_dir : str | Path, optional
        Output directory for saved .npy files.
    """
    logger.info("Starting fixed-frequency scan.")

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    initial_radii: np.ndarray = np.linspace(
        initial_radius_min,
        initial_radius_max,
        n_points,
    )
    acoustic_pressures: np.ndarray = np.linspace(
        acoustic_pressure_min,
        acoustic_pressure_max,
        n_points,
    )

    logger.info(
        "Fixed-frequency settings: frequency=%s Hz, temperature=%s C, N=%d",
        frequency,
        temperature,
        n_points,
    )

    grid: list[tuple[float, float]] = list(product(initial_radii, acoustic_pressures))
    logger.info("Constructed fixed-frequency grid with %d points.", len(grid))

    for idx, eq in enumerate(["RP", "KM", "G"], start=1):
        output_path = results_path / f"{eq}_fix_freq.npy"
        logger.info("[%d/3] %s: starting grid computation.", idx, eq)

        try:
            results = compute_lce_grid(
                grid=grid,
                equation=eq,
                temperature=temperature,
                frequency=frequency,
                pressure=None,
                filename_suffix="_fix_freq",
            )

            logger.info("%s: computation finished, saving to %s", eq, output_path)
            np.save(output_path, results, allow_pickle=True)

            if not output_path.exists():
                logger.error("%s: expected output file was not created.", eq)
                raise FileNotFoundError(f"Missing output file: {output_path}")

            logger.info("[%d/3] %s: finished successfully.", idx, eq)

        except Exception:
            logger.exception("%s: fixed-frequency scan failed.", eq)
            raise

    logger.info("Finished fixed-frequency scan.")


def generate_fixed_pressure_scans(
    *,
    acoustic_pressure: float,
    temperature: float,
    n_points: int,
    initial_radius_min: float,
    initial_radius_max: float,
    frequency_min: float,
    frequency_max: float,
    results_dir: str | Path = "results",
) -> None:
    """
    Generate Lyapunov grids for the fixed-pressure experiment.

    Parameters
    ----------
    acoustic_pressure : float
        Fixed acoustic pressure in Pa.
    temperature : float
        Temperature in degrees Celsius.
    n_points : int
        Number of grid points per axis.
    initial_radius_min : float
        Minimum initial radius.
    initial_radius_max : float
        Maximum initial radius.
    frequency_min : float
        Minimum frequency in MHz.
    frequency_max : float
        Maximum frequency in MHz.
    results_dir : str | Path, optional
        Output directory for saved .npy files.
    """
    logger.info("Starting fixed-pressure scan (_fix_pa).")

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    initial_radii: np.ndarray = np.linspace(
        initial_radius_min,
        initial_radius_max,
        n_points,
    )
    frequencies: np.ndarray = np.linspace(
        frequency_min,
        frequency_max,
        n_points,
    )

    logger.info(
        "Fixed-pressure settings: acoustic_pressure=%s Pa, temperature=%s C, N=%d",
        acoustic_pressure,
        temperature,
        n_points,
    )

    grid: list[tuple[float, float]] = list(product(initial_radii, frequencies))
    logger.info("Constructed fixed-pressure grid with %d points.", len(grid))

    for idx, eq in enumerate(["RP", "KM", "G"], start=1):
        output_path = results_path / f"{eq}_fix_pa.npy"
        logger.info("[%d/3] %s: starting grid computation.", idx, eq)

        try:
            results = compute_lce_grid(
                grid=grid,
                equation=eq,
                temperature=temperature,
                pressure=acoustic_pressure,
                frequency=None,
                filename_suffix="_fix_pa",
            )

            logger.info("%s: computation finished, saving to %s", eq, output_path)
            np.save(output_path, results, allow_pickle=True)

            if not output_path.exists():
                logger.error("%s: expected output file was not created.", eq)
                raise FileNotFoundError(f"Missing output file: {output_path}")

            logger.info("[%d/3] %s: finished successfully.", idx, eq)

        except Exception:
            logger.exception("%s: fixed-pressure scan failed.", eq)
            raise

    logger.info("Finished fixed-pressure scan (_fix_pa).")