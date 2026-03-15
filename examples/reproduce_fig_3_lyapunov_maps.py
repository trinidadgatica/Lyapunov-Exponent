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
            f"Run the preparation script first:\n"
            f"  python examples/pre_examples.py\n"
        )
    logger.info("Found required file: %s", p)
    return p


def main() -> None:
    logger.info("Starting reproduction of Figure 3: Lyapunov maps.")

    Path("figures").mkdir(parents=True, exist_ok=True)

    n_points = 5
    initial_radii = np.linspace(1, 50, n_points)
    acoustic_pressures = np.linspace(0.2, 3, n_points)
    frequencies = np.linspace(0.02, 2, n_points)

    try:
        logger.info("Loading fixed-frequency sweep results.")
        res_rp_f = np.load(require_file("results/RP_fix_freq.npy"), allow_pickle=True)
        res_km_f = np.load(require_file("results/KM_fix_freq.npy"), allow_pickle=True)
        res_g_f = np.load(require_file("results/G_fix_freq.npy"), allow_pickle=True)

        logger.info("Generating fixed-frequency Lyapunov maps.")
        for eq, data in zip(["RP", "KM", "G"], [res_rp_f, res_km_f, res_g_f]):
            logger.info("%s: plotting fixed-frequency map.", eq)
            max_le = np.max(data, axis=1)
            plot_max_lce_map(
                x=initial_radii,
                y=acoustic_pressures,
                max_exponents=max_le,
                xlabel="Initial Radius (μm)",
                ylabel="Acoustic Pressure (MPa)",
                save_path=f"figures/LE_map_freq_{eq}.pdf",
                xticks=[1, 10, 20, 30, 40, 50],
                yticks=[0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                show=True,
                close=False,
                low_pa=False,
            )
            logger.info("%s: fixed-frequency map saved.", eq)

        logger.info("Loading fixed-pressure sweep results.")
        res_rp_pa = np.load(require_file("results/RP_fix_pa.npy"), allow_pickle=True)
        res_km_pa = np.load(require_file("results/KM_fix_pa.npy"), allow_pickle=True)
        res_g_pa = np.load(require_file("results/G_fix_pa.npy"), allow_pickle=True)

        logger.info("Generating fixed-pressure Lyapunov maps.")
        for eq, data in zip(["RP", "KM", "G"], [res_rp_pa, res_km_pa, res_g_pa]):
            logger.info("%s: plotting fixed-pressure map.", eq)
            max_le = np.max(data, axis=1)
            plot_max_lce_map(
                x=initial_radii,
                y=frequencies,
                max_exponents=max_le,
                xlabel="Initial Radius (μm)",
                ylabel="Frequency (MHz)",
                save_path=f"figures/LE_map_pa_{eq}.pdf",
                xticks=[1, 10, 20, 30, 40, 50],
                yticks=[0.02, 0.4, 0.8, 1.2, 1.6, 2.0],
                show=True,
                close=False,
                low_pa=False,
            )
            logger.info("%s: fixed-pressure map saved.", eq)

    except Exception:
        logger.exception("Lyapunov map reproduction failed.")
        raise

    logger.info("Finished reproduction of Figure 3.")


if __name__ == "__main__":
    main()