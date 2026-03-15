from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plotting.phase_portraits import (  # noqa: E402
    create_phase_portrait_composite_figure,
    plot_stable_rp_phase_portrait,
)
from utils.logging_utils import get_logger, setup_logging  # noqa: E402

setup_logging(log_to_file=True)
logger = get_logger(__name__)


def main() -> None:
    logger.info("Starting reproduction of Figures 1 and 2: phase portraits.")

    try:
        logger.info("Generating single phase plot.")
        plot_stable_rp_phase_portrait()
        logger.info("Single phase plot generated successfully.")

        logger.info("Generating composite phase portrait figure.")
        create_phase_portrait_composite_figure()
        logger.info("Composite phase portrait figure generated successfully.")

    except Exception:
        logger.exception("Phase portrait figure reproduction failed.")
        raise

    logger.info("Finished reproduction of Figures 1 and 2.")


if __name__ == "__main__":
    main()