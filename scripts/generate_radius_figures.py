from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from models.bubble_models import simulate_bubble_trajectories
from utils.logging_utils import get_logger, setup_logging

setup_logging(log_to_file=True)
logger = get_logger(__name__)


def _save_radius_plot(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    logger.info("Saving figure: %s", output_path)

    fig, ax = plt.subplots()
    ax.plot(x, y, label=title.split()[0])
    ax.set_title(title)
    ax.set_xlabel("Periods")
    ax.set_ylabel("Radius (non-dimensional)")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    if not output_path.exists():
        logger.error("Expected output figure was not created: %s", output_path)
        raise FileNotFoundError(f"Missing output figure: {output_path}")


def main() -> None:
    logger.info("Starting radius figure generation.")

    Path("results").mkdir(parents=True, exist_ok=True)

    temperature: float = 20
    periods: int = 10
    step: float = 1e-3

    configs: list[tuple[str, float, float, float]] = [
        ("C1", 0.3e6, 1.2e6, 10e-6), # Pa, F, R0
        ("C2", 1.5e6, 1.2e6, 5e-6),
    ]

    for idx, (cfg_name, pressure, frequency, initial_radius) in enumerate(configs, start=1):
        logger.info(
            "[%d/%d] Running %s with pressure=%s, frequency=%s, initial_radius=%s",
            idx,
            len(configs),
            cfg_name,
            pressure,
            frequency,
            initial_radius,
        )

        integration_time = np.arange(0, periods / frequency, step / frequency)

        try:
            trajectories, model = simulate_bubble_trajectories(
                ["RP", "KM", "G"],
                temperature,
                pressure,
                frequency,
                initial_radius,
                integration_time,
                step,
            )
            logger.info("%s: trajectories created successfully.", cfg_name)

            radius_rp = trajectories["Radius_RP"]
            radius_km = trajectories["Radius_KM"]
            radius_g = trajectories["Radius_G"]

            _save_radius_plot(
                integration_time * frequency,
                radius_rp,
                f"Rayleigh-Plesset ({cfg_name})",
                Path(f"results/Radius_RP_{cfg_name}.pdf"),
            )

            _save_radius_plot(
                integration_time * frequency,
                radius_km,
                f"Keller-Miksis ({cfg_name})",
                Path(f"results/Radius_KM_{cfg_name}.pdf"),
            )

            _save_radius_plot(
                integration_time * frequency,
                radius_g,
                f"Gilmore ({cfg_name})",
                Path(f"results/Radius_G_{cfg_name}.pdf"),
            )

            logger.info("%s: all radius figures saved successfully.", cfg_name)

        except Exception:
            logger.exception("%s: radius figure generation failed.", cfg_name)
            raise

    logger.info("Finished radius figure generation.")


if __name__ == "__main__":
    main()