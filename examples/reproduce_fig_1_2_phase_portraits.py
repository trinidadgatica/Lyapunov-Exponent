from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plotting.phase_portraits import (
    create_phase_portrait_composite_figure,
    plot_stable_rp_phase_portrait,
    print_phase_portrait_lyapunov_table,
)
from utils.logging_utils import get_logger, setup_logging


setup_logging(log_to_file=True)
logger = get_logger(__name__)


def main() -> None:
    logger.info("Starting reproduction of Figures 1 and 2: phase portraits.")

    Path("figures").mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Generating Figure 1: stable RP phase portrait.")
        plot_stable_rp_phase_portrait(
            save_path="figures/fig_1_stable_rp_phase_portrait.pdf",
            show=False,
            close=True,
        )
        logger.info("Figure 1 generated successfully.")

        logger.info("Generating Figure 2: composite phase portrait figure.")
        _, _, lyap_table_data = create_phase_portrait_composite_figure(
            save_path="figures/fig_2_phase_portraits_composite.pdf",
            show=False,
            close=True,
        )
        logger.info("Figure 2 generated successfully.")

        print_phase_portrait_lyapunov_table(lyap_table_data)

    except Exception:
        logger.exception("Phase portrait figure reproduction failed.")
        raise

    logger.info("Finished reproduction of Figures 1 and 2.")


if __name__ == "__main__":
    main()