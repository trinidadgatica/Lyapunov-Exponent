from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import numpy as np

from core.lyapunov import compute_lce_grid
from utils.logging_utils import get_logger

logger = get_logger(__name__)

def _compute_and_save_grid(
    *,
    equation: str,
    grid: list[tuple[float, float]],
    temperature: float,
    frequency: float | None,
    pressure: float | None,
    filename_suffix: str,
    output_path: str | Path,
) -> str:
    """
    Worker function for one equation.
    Runs the Lyapunov grid computation and saves the result.
    """
    output_path = Path(output_path)

    results = compute_lce_grid(
        grid=grid,
        equation=equation,
        temperature=temperature,
        frequency=frequency,
        pressure=pressure,
        filename_suffix=filename_suffix,
    )

    np.save(output_path, results, allow_pickle=True)

    if not output_path.exists():
        raise FileNotFoundError(f"Missing output file after save: {output_path}")

    return str(output_path)


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
    skip_existing: bool = True,
    max_workers: int | None = None,
) -> None:
    """
    Generate Lyapunov grids for the fixed-frequency experiment.

    Saves:
        RP_fix_freq.npy
        KM_fix_freq.npy
        G_fix_freq.npy
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

    jobs: list[dict] = []
    for equation in ["RP", "KM", "G"]:
        output_path = results_path / f"{equation}_fix_freq.npy"

        if skip_existing and output_path.exists():
            logger.info("%s: output already exists, skipping %s", equation, output_path)
            continue

        jobs.append(
            {
                "equation": equation,
                "grid": grid,
                "temperature": temperature,
                "frequency": frequency,
                "pressure": None,
                "filename_suffix": "_fix_freq",
                "output_path": output_path,
            }
        )

    if not jobs:
        logger.info("All fixed-frequency outputs already exist. Nothing to do.")
        return

    logger.info(
        "Running %d fixed-frequency equation job(s) with max_workers=%s",
        len(jobs),
        max_workers,
    )

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_compute_and_save_grid, **job): job["equation"]
            for job in jobs
        }

        for future in as_completed(futures):
            equation = futures[future]
            try:
                saved_path = future.result()
                logger.info("%s: fixed-frequency scan finished successfully: %s", equation, saved_path)
            except Exception:
                logger.exception("%s: fixed-frequency scan failed.", equation)
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
    skip_existing: bool = True,
    max_workers: int | None = None,
) -> None:
    """
    Generate Lyapunov grids for the fixed-pressure experiment.

    Saves:
        RP_fix_pa.npy
        KM_fix_pa.npy
        G_fix_pa.npy
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

    jobs: list[dict] = []
    for equation in ["RP", "KM", "G"]:
        output_path = results_path / f"{equation}_fix_pa.npy"

        if skip_existing and output_path.exists():
            logger.info("%s: output already exists, skipping %s", equation, output_path)
            continue

        jobs.append(
            {
                "equation": equation,
                "grid": grid,
                "temperature": temperature,
                "frequency": None,
                "pressure": acoustic_pressure,
                "filename_suffix": "_fix_pa",
                "output_path": output_path,
            }
        )

    if not jobs:
        logger.info("All fixed-pressure outputs already exist. Nothing to do.")
        return

    logger.info(
        "Running %d fixed-pressure equation job(s) with max_workers=%s",
        len(jobs),
        max_workers,
    )

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_compute_and_save_grid, **job): job["equation"]
            for job in jobs
        }

        for future in as_completed(futures):
            equation = futures[future]
            try:
                saved_path = future.result()
                logger.info("%s: fixed-pressure scan finished successfully: %s", equation, saved_path)
            except Exception:
                logger.exception("%s: fixed-pressure scan failed.", equation)
                raise

    logger.info("Finished fixed-pressure scan (_fix_pa).")