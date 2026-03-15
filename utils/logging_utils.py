from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOGGER_CONFIGURED = False


def setup_logging(
    level: int = logging.INFO,
    log_to_file: bool = False,
    log_file: str = "loggers/run.log",
) -> None:
    """
    Configure project-wide logging only once.
    """
    global _LOGGER_CONFIGURED

    if _LOGGER_CONFIGURED:
        return

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
    ]

    if log_to_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )

    _LOGGER_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger for a module or script.
    """
    return logging.getLogger(name)