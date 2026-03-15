from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def ensure_parent_dir(save_path: str | Path) -> Path:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(
    fig: plt.Figure,
    save_path: str | Path,
    *,
    dpi: int = 300,
    bbox_inches: str = "tight",
) -> None:
    output_path = ensure_parent_dir(save_path)
    fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)


def finalize_figure(
    fig: plt.Figure,
    *,
    save_path: str | Path | None = None,
    show: bool = True,
    close: bool = False,
    dpi: int = 300,
    bbox_inches: str = "tight",
    use_tight_layout: bool = True,
) -> None:
    if use_tight_layout:
        fig.tight_layout()

    if save_path is not None:
        save_figure(fig, save_path, dpi=dpi, bbox_inches=bbox_inches)

    if show:
        plt.show()

    if close:
        plt.close(fig)


def reshape_grid_values(
    x: np.ndarray,
    y: np.ndarray,
    values: Iterable[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x)
    y = np.asarray(y)
    X, Y = np.meshgrid(x, y)
    Z = np.asarray(values).reshape(X.shape, order="F")
    return X, Y, Z