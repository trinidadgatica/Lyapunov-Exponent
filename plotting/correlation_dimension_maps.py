from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from core.dimensions import compute_lce_dimension_metrics
from plotting.styles import (
    DEFAULT_D2_MAX,
    DEFAULT_MAP_LEVELS,
    LEGEND_FONT_SIZE,
    PLOT_WIDTH,
    X_LABEL_FONT_SIZE,
    X_TICK_FONT_SIZE,
    Y_LABEL_FONT_SIZE,
    Y_TICK_FONT_SIZE,
)
from plotting.utils import finalize_figure, reshape_grid_values


def _compute_d2_values(results: np.ndarray) -> np.ndarray:
    d2_vals = [
        compute_lce_dimension_metrics(result)["Correlation D₂ (approx)"]
        for result in results
    ]
    return np.clip(np.nan_to_num(d2_vals, nan=DEFAULT_D2_MAX), 0, DEFAULT_D2_MAX)


def plot_d2_map(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    save_path: str,
    xticks: list[float] | None = None,
    yticks: list[float] | None = None,
    show: bool = True,
    close: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Generic D2 contour map plotter.
    """
    X, Y, Z = reshape_grid_values(x, y, values)

    levels = np.linspace(0, DEFAULT_D2_MAX, DEFAULT_MAP_LEVELS)
    norm = Normalize(vmin=0, vmax=DEFAULT_D2_MAX)

    factor = 0.7
    fig, ax = plt.subplots(
        figsize=(factor * PLOT_WIDTH, factor * (PLOT_WIDTH - 1.5))
    )

    contour = ax.contourf(
        X,
        Y,
        Z,
        levels=levels,
        norm=norm,
        cmap="viridis",
        extend="both",
    )

    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("Correlation Dimension $D_2$", fontsize=X_LABEL_FONT_SIZE)
    cbar.ax.tick_params(labelsize=LEGEND_FONT_SIZE)

    ax.set_xlabel(xlabel, fontsize=X_LABEL_FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=Y_LABEL_FONT_SIZE)

    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)

    ax.tick_params(axis="x", labelsize=X_TICK_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=Y_TICK_FONT_SIZE)
    ax.grid(True)

    finalize_figure(
        fig,
        save_path=save_path,
        show=show,
        close=close,
        dpi=300,
        bbox_inches="tight",
        use_tight_layout=True,
    )
    return fig, ax


def plot_d2_map_fixed_frequency(
    initial_radii: np.ndarray,
    pressures: np.ndarray,
    grid_size: int,
    results: np.ndarray,
    equation_name: str,
    save_path: str | None = None,
    show: bool = True,
    close: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create D2 contour plot for constant frequency across pressure-radius space.
    """
    x = np.linspace(initial_radii.min(), initial_radii.max(), grid_size)
    y = np.linspace(pressures.min(), pressures.max(), grid_size)

    d2_vals = _compute_d2_values(results)

    if save_path is None:
        save_path = f"results/{equation_name}_d2_map_freq.pdf"

    return plot_d2_map(
        x,
        y,
        d2_vals,
        xlabel="Initial Radius (μm)",
        ylabel="Acoustic Pressure (MPa)",
        save_path=save_path,
        xticks=[1, 10, 20, 30, 40, 50],
        yticks=[0.2, 0.5, 1, 1.5, 2, 2.5, 3],
        show=show,
        close=close,
    )


def plot_d2_map_fixed_pressure(
    initial_radii: np.ndarray,
    freqs: np.ndarray,
    grid_size: int,
    results: np.ndarray,
    equation_name: str,
    low_pressure: bool = False,
    save_path: str | None = None,
    show: bool = True,
    close: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create D2 contour plot for constant pressure across frequency-radius space.
    """
    x = np.linspace(initial_radii.min(), initial_radii.max(), grid_size)
    y = np.linspace(freqs.min(), freqs.max(), grid_size)

    d2_vals = _compute_d2_values(results)

    if save_path is None:
        suffix = "_pa"
        save_path = f"results/{equation_name}_d2_map{suffix}.pdf"

    return plot_d2_map(
        x,
        y,
        d2_vals,
        xlabel="Initial Radius (μm)",
        ylabel="Frequency (MHz)",
        save_path=save_path,
        xticks=[1, 10, 20, 30, 40, 50],
        yticks=[0.02, 0.4, 0.8, 1.2, 1.6, 2],
        show=show,
        close=close,
    )