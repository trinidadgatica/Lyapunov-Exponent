from __future__ import annotations

from pathlib import Path

import numpy as np

from plotting.correlation_dimension_maps import (
    plot_d2_map_fixed_frequency,
    plot_d2_map_fixed_pressure,
)
from utils.logging_utils import get_logger, setup_logging


setup_logging(log_to_file=True)
logger = get_logger(__name__)


def _require(path: str) -> Path:
    p = Path(path)
    if not p.exists():
        logger.error("Missing required file: %s", p)
        raise FileNotFoundError(f"Missing required file: {p}")
    logger.info("Found required file: %s", p)
    return p


def main() -> None:
    logger.info("Starting correlation-dimension map generation.")

    N: int = 50
    initial_radii: np.ndarray = np.linspace(1, 50, N)

    logger.info("Preparing D2 maps with fixed frequency.")
    pressures: np.ndarray = np.linspace(0.2, 3, N)

    results_RP_f = np.load(_require("results/RP_fix_freq.npy"), allow_pickle=True)
    results_KM_f = np.load(_require("results/KM_fix_freq.npy"), allow_pickle=True)
    results_G_f = np.load(_require("results/G_fix_freq.npy"), allow_pickle=True)

    logger.info("Creating fixed-frequency D2 map for RP.")
    plot_d2_map_fixed_frequency(initial_radii, pressures, N, results_RP_f, "RP")

    logger.info("Creating fixed-frequency D2 map for KM.")
    plot_d2_map_fixed_frequency(initial_radii, pressures, N, results_KM_f, "KM")

    logger.info("Creating fixed-frequency D2 map for G.")
    plot_d2_map_fixed_frequency(initial_radii, pressures, N, results_G_f, "G")

    logger.info("Preparing D2 maps with fixed pressure.")
    frequencies: np.ndarray = np.linspace(0.02, 2, N)

    results_RP_pa = np.load(_require("results/RP_fix_pa.npy"), allow_pickle=True)
    results_KM_pa = np.load(_require("results/KM_fix_pa.npy"), allow_pickle=True)
    results_G_pa = np.load(_require("results/G_fix_pa.npy"), allow_pickle=True)

    logger.info("Creating fixed-pressure D2 map for RP.")
    plot_d2_map_fixed_pressure(initial_radii, frequencies, N, results_RP_pa, "RP")

    logger.info("Creating fixed-pressure D2 map for KM.")
    plot_d2_map_fixed_pressure(initial_radii, frequencies, N, results_KM_pa, "KM")

    logger.info("Creating fixed-pressure D2 map for G.")
    plot_d2_map_fixed_pressure(initial_radii, frequencies, N, results_G_pa, "G")

    logger.info("Creating low-pressure D2 map for RP.")
    plot_d2_map_fixed_pressure(
        initial_radii,
        frequencies,
        N,
        results_RP_pa,
        "RP",
        low_pressure=True,
    )

    logger.info("Creating low-pressure D2 map for KM.")
    plot_d2_map_fixed_pressure(
        initial_radii,
        frequencies,
        N,
        results_KM_pa,
        "KM",
        low_pressure=True,
    )

    logger.info("Creating low-pressure D2 map for G.")
    plot_d2_map_fixed_pressure(
        initial_radii,
        frequencies,
        N,
        results_G_pa,
        "G",
        low_pressure=True,
    )

    logger.info("Finished correlation-dimension map generation.")


if __name__ == "__main__":
    main()