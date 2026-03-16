from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from plotting.styles import (
    DEFAULT_LYAPUNOV_LINTHRESH,
    DEFAULT_LYAPUNOV_NUM_DIVS,
    LEGEND_FONT_SIZE,
    PLOT_WIDTH,
    X_LABEL_FONT_SIZE,
    X_TICK_FONT_SIZE,
    Y_LABEL_FONT_SIZE,
    Y_TICK_FONT_SIZE,
)
from plotting.utils import finalize_figure, reshape_grid_values


def _build_lyapunov_colormap(
    num_divs: int,
    linthresh: float,
    zmin: float,
    zmax: float,
) -> tuple[ListedColormap, BoundaryNorm, np.ndarray]:
    neg_levels = -np.logspace(
        np.log10(linthresh),
        np.log10(max(-zmin, linthresh * 10)),
        num_divs,
    )[::-1]

    pos_levels = np.logspace(
        np.log10(linthresh),
        np.log10(max(zmax, linthresh * 10)),
        num_divs,
    )

    all_levels = np.unique(
        np.concatenate((neg_levels, [-linthresh], [linthresh], pos_levels))
    )
    all_levels = np.sort(all_levels)

    base_cmap = plt.get_cmap("coolwarm", 256)
    blue = base_cmap(np.linspace(0.0, 0.45, num_divs))
    red = base_cmap(np.linspace(0.55, 1.0, num_divs))
    gray = np.array([[0, 0, 0, 0]])
    cmap = ListedColormap(np.vstack((blue, gray, red)))
    norm = BoundaryNorm(all_levels, ncolors=len(cmap.colors), clip=False)

    return cmap, norm, all_levels


def plot_max_lce_map(
    x: np.ndarray,
    y: np.ndarray,
    max_exponents: np.ndarray,
    xlabel: str,
    ylabel: str,
    save_path: str,
    xticks: list[float] | None = None,
    yticks: list[float] | None = None,
    show: bool = True,
    close: bool = False,
    low_pa: bool = False,
    linthresh: float = DEFAULT_LYAPUNOV_LINTHRESH,
    num_divs: int = DEFAULT_LYAPUNOV_NUM_DIVS,
) -> tuple[plt.Figure, plt.Axes]:
    X, Y, Z = reshape_grid_values(x, y, max_exponents)

    zmin = float(np.min(Z))
    zmax = float(np.max(Z))
    cmap, norm, levels = _build_lyapunov_colormap(
        num_divs=num_divs,
        linthresh=linthresh,
        zmin=zmin,
        zmax=zmax,
    )

    factor = 1
    fig, ax = plt.subplots(
        figsize=(factor * PLOT_WIDTH, factor * (PLOT_WIDTH - 1.5))
    )

    contour = ax.contourf(
        X,
        Y,
        Z,
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend="both",
    )

    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("Max Lyapunov Exponent", fontsize=X_LABEL_FONT_SIZE)
    cbar.set_ticks(levels)
    cbar.set_ticklabels([f"{value:.1e}" for value in levels])
    cbar.ax.tick_params(labelsize=LEGEND_FONT_SIZE)

    ax.set_xlabel(xlabel, fontsize=X_LABEL_FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=Y_LABEL_FONT_SIZE)

    if xticks is None:
        xticks = [1, 10, 20, 30, 40, 50]
    ax.set_xticks(xticks)

    if yticks is not None:
        ax.set_yticks(yticks)

    ax.tick_params(
        axis="x",
        labelsize=X_TICK_FONT_SIZE,
        direction="out",
        length=4,
        width=1,
        top=False,
        bottom=True,
    )
    ax.tick_params(
        axis="y",
        labelsize=Y_TICK_FONT_SIZE,
        direction="out",
        length=4,
        width=1,
        left=True,
        right=False,
    )

    final_save_path = save_path

    finalize_figure(
        fig,
        save_path=final_save_path,
        show=show,
        close=close,
        dpi=300,
        bbox_inches="tight",
        use_tight_layout=True,
    )
    return fig, ax