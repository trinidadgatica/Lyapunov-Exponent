from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from plotting.lyapunov_maps import plot_max_lce_map
from utils.logging_utils import get_logger, setup_logging

setup_logging(log_to_file=True)
logger = get_logger(__name__)


def require_file(path: str) -> Path:
    p = Path(path)
    if not p.exists():
        logger.error("Missing required file: %s", p)
        raise FileNotFoundError(
            f"Missing required file: {p}\n"
            f"Run the grid-generation scripts first:\n"
            f"  python -m scripts.run_fixed_frequency_scan\n"
            f"  python -m scripts.run_fixed_pressure_scan\n"
        )
    logger.info("Found required file: %s", p)
    return p


def main() -> None:
    logger.info("Starting reproduction of Figure 3: Lyapunov maps.")

    N: int = 50
    initial_radii: np.ndarray = np.linspace(1, 50, N)
    acoustic_pressures: np.ndarray = np.linspace(0.2, 3, N)
    frequencies: np.ndarray = np.linspace(0.02, 2, N)

    try:
        logger.info("Loading fixed-frequency sweep results.")
        res_RP_f = np.load(require_file("results/RP_fix_freq.npy"), allow_pickle=True)
        res_KM_f = np.load(require_file("results/KM_fix_freq.npy"), allow_pickle=True)
        res_G_f = np.load(require_file("results/G_fix_freq.npy"), allow_pickle=True)

        logger.info("Generating fixed-frequency Lyapunov maps.")
        for eq, data in zip(["RP", "KM", "G"], [res_RP_f, res_KM_f, res_G_f]):
            logger.info("%s: plotting fixed-frequency map.", eq)
            max_le = np.max(data, axis=1)
            plot_max_lce_map(
                x=initial_radii,
                y=acoustic_pressures,
                max_exponents=max_le,
                equation=eq,
                xlabel="Initial Radius (μm)",
                ylabel="Acoustic Pressure (MPa)",
                filename=f"results/LE_map_freq_{eq}.pdf",
            )
            logger.info("%s: fixed-frequency map saved.", eq)

        logger.info("Loading fixed-pressure sweep results.")
        res_RP_pa = np.load(require_file("results/RP_fix_pa.npy"), allow_pickle=True)
        res_KM_pa = np.load(require_file("results/KM_fix_pa.npy"), allow_pickle=True)
        res_G_pa = np.load(require_file("results/G_fix_pa.npy"), allow_pickle=True)

        logger.info("Generating fixed-pressure Lyapunov maps.")
        for eq, data in zip(["RP", "KM", "G"], [res_RP_pa, res_KM_pa, res_G_pa]):
            logger.info("%s: plotting fixed-pressure map.", eq)
            max_le = np.max(data, axis=1)
            plot_max_lce_map(
                x=initial_radii,
                y=frequencies,
                max_exponents=max_le,
                equation=eq,
                xlabel="Initial Radius (μm)",
                ylabel="Frequency (MHz)",
                filename=f"results/LE_map_pa_{eq}.pdf",
            )
            logger.info("%s: fixed-pressure map saved.", eq)

    except Exception:
        logger.exception("Lyapunov map reproduction failed.")
        raise

    logger.info("Finished reproduction of Figure 3.")


if __name__ == "__main__":
    main()