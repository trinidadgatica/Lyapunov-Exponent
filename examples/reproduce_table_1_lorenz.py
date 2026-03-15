from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from models.lorenz import benchmark_case
from utils.logging_utils import get_logger, setup_logging

setup_logging(log_to_file=True)
logger = get_logger(__name__)


def main() -> None:
    logger.info("Starting reproduction of Table 1: Lorenz benchmarks.")

    Path("results").mkdir(parents=True, exist_ok=True)

    cases: list[dict] = [
        dict(name="chaotic_16_45.92_4", sigma=16.0, rho=45.92, beta=4.0),
        dict(name="stable_10_0.5_8_3", sigma=10.0, rho=0.5, beta=8.0 / 3.0),
    ]

    t0: float = 0.0
    t1: float = 50.0
    dt: float = 0.01
    transient_time: float = 0.0
    observable: str = "x"

    min_tsep: int = int(2.0 / dt)
    eck_params: dict = dict(
        emb_dim=9,
        matrix_dim=3,
        tau=10,
        min_tsep=min_tsep,
        min_nb=20,
    )
    ros_params: dict = dict(
        emb_dim=9,
        tau=10,
        min_tsep=min_tsep,
        trajectory_len=60,
        fit="RANSAC",
    )

    all_rows: list[dict] = []

    try:
        for idx, case in enumerate(cases, start=1):
            logger.info("[%d/%d] Running Lorenz case: %s", idx, len(cases), case["name"])
            rows = benchmark_case(
                case,
                t0,
                t1,
                dt,
                transient_time,
                eck_params,
                ros_params,
                observable,
            )
            all_rows.extend(rows)
            logger.info("%s: completed successfully.", case["name"])

        df = pd.DataFrame(all_rows)
        df = df[["case", "method", "time_sec", "lce1", "lce2", "lce3", "sigma", "rho", "beta"]]
        df = df.sort_values(["case", "method"]).reset_index(drop=True)

        logger.info("Generated Table 1 dataframe with %d rows.", len(df))
        print(df.to_string(index=False))

    except Exception:
        logger.exception("Table 1 reproduction failed.")
        raise

    logger.info("Finished reproduction of Table 1.")


if __name__ == "__main__":
    main()