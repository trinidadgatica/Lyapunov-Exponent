from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.grid_generation import (  # noqa: E402
    generate_fixed_frequency_scans,
    generate_fixed_pressure_scans,
)
from utils.logging_utils import get_logger, setup_logging

setup_logging(log_to_file=True)
logger = get_logger(__name__)


def validate_outputs() -> None:
    """Check that the required precomputed files were created."""
    required_files = [
        "results/RP_fix_freq.npy",
        "results/KM_fix_freq.npy",
        "results/G_fix_freq.npy",
        "results/RP_fix_pa.npy",
        "results/KM_fix_pa.npy",
        "results/G_fix_pa.npy",
    ]

    missing_files = [path for path in required_files if not Path(path).exists()]

    if missing_files:
        missing_str = "\n".join(f"  - {path}" for path in missing_files)
        raise FileNotFoundError(
            "The preparation step finished, but some expected files are missing:\n"
            f"{missing_str}"
        )

    logger.info("All required pre-example files were created.")


def main() -> None:
    """Generate reusable data needed by the paper figure scripts."""
    logger.info("Starting pre_examples generation.")

    Path("results").mkdir(parents=True, exist_ok=True)
    Path("figures").mkdir(parents=True, exist_ok=True)

    # ==========================================================
    # Paper-level experiment settings
    # ==========================================================
    n_points: int = 5
    temperature: float = 20.0  # °C

    # Optimization controls
    skip_existing: bool = False
    max_workers: int | None = 3

    # Fixed-frequency scan settings
    fixed_frequency: float = 1e6  # Hz
    initial_radius_min_freq: float = 1.0
    initial_radius_max_freq: float = 50.0
    acoustic_pressure_min: float = 0.2  # MPa
    acoustic_pressure_max: float = 3.0  # MPa

    # Fixed-pressure scan settings
    fixed_acoustic_pressure: float = 0.1e6  # Pa
    initial_radius_min_pa: float = 1.0
    initial_radius_max_pa: float = 50.0
    frequency_min: float = 0.02  # MHz
    frequency_max: float = 2.0  # MHz

    logger.info(
        "Optimization settings: skip_existing=%s, max_workers=%s",
        skip_existing,
        max_workers,
    )

    logger.info("Running fixed-frequency generation.")
    generate_fixed_frequency_scans(
        frequency=fixed_frequency,
        temperature=temperature,
        n_points=n_points,
        initial_radius_min=initial_radius_min_freq,
        initial_radius_max=initial_radius_max_freq,
        acoustic_pressure_min=acoustic_pressure_min,
        acoustic_pressure_max=acoustic_pressure_max,
        results_dir="results",
        skip_existing=skip_existing,
        max_workers=max_workers,
    )

    logger.info("Running fixed-pressure generation.")
    generate_fixed_pressure_scans(
        acoustic_pressure=fixed_acoustic_pressure,
        temperature=temperature,
        n_points=n_points,
        initial_radius_min=initial_radius_min_pa,
        initial_radius_max=initial_radius_max_pa,
        frequency_min=frequency_min,
        frequency_max=frequency_max,
        results_dir="results",
        skip_existing=skip_existing,
        max_workers=max_workers,
    )

    validate_outputs()
    logger.info("Finished pre_examples generation.")


if __name__ == "__main__":
    main()