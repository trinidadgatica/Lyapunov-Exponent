import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import AutoMinorLocator

from utils.plot_information import *


def _build_custom_colormap(
    num_divs: int,
    linthresh: float,
    zmin: float,
    zmax: float
) -> tuple:
    neg_levels = -np.logspace(np.log10(linthresh), np.log10(max(-zmin, linthresh * 10)), num_divs)[::-1]
    pos_levels = np.logspace(np.log10(linthresh), np.log10(max(zmax, linthresh * 10)), num_divs)

    all_levels = np.unique(np.concatenate((neg_levels, [-linthresh], [linthresh], pos_levels)))
    all_levels = np.sort(all_levels)

    base_cmap = plt.get_cmap('coolwarm', 256)
    blue = base_cmap(np.linspace(0.0, 0.45, num_divs))
    red = base_cmap(np.linspace(0.55, 1.0, num_divs))
    gray = np.array([[0, 0, 0, 0]])
    cmap = ListedColormap(np.vstack((blue, gray, red)))
    norm = BoundaryNorm(all_levels, ncolors=len(cmap.colors), clip=False)

    return cmap, norm, all_levels


def plot_le_map1(
    x: np.ndarray,
    y: np.ndarray,
    max_exponents: np.ndarray,
    equation: str,
    xlabel: str,
    ylabel: str,
    filename: str,
    low_pa: bool = False
) -> None:
    X, Y = np.meshgrid(x, y)
    Z = np.array(max_exponents).reshape(X.shape, order='F')

    zmin, zmax = np.min(Z), np.max(Z)
    linthresh = 1e-3
    num_divs = 4
    cmap, norm, levels = _build_custom_colormap(num_divs, linthresh, zmin, zmax)

    factor = 0.7
    plt.figure(figsize=(factor * PLOT_WIDTH, factor * (PLOT_WIDTH - 1.5)))
    contour = plt.contourf(X, Y, Z, levels=levels, cmap=cmap, norm=norm, extend='both')
    cbar = plt.colorbar(contour)
    cbar.set_label('Max Lyapunov Exponent', fontsize=X_LABEL_FONT_SIZE)

    ticks = levels
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{v:.1e}' for v in ticks])
    cbar.ax.tick_params(labelsize=LEGEND_FONT_SIZE)

    plt.xlabel(xlabel, fontsize=X_LABEL_FONT_SIZE)
    plt.ylabel(ylabel, fontsize=Y_LABEL_FONT_SIZE)
    plt.xticks([1, 10, 20, 30, 40, 50], fontsize=X_TICK_FONT_SIZE)
    plt.grid(True, which='both', linestyle=':', linewidth=0.5)
    plt.tight_layout()

    if low_pa:
        filename = filename.replace(".pdf", "_01.pdf")

    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.show()


def plot_le_map(
    x: np.ndarray,
    y: np.ndarray,
    max_exponents: np.ndarray,
    equation: str,
    xlabel: str,
    ylabel: str,
    filename: str,
    low_pa: bool = False
) -> None:
    X, Y = np.meshgrid(x, y)
    Z = np.array(max_exponents).reshape(X.shape, order='F')

    zmin, zmax = np.min(Z), np.max(Z)
    linthresh = 1e-3
    num_divs = 4
    cmap, norm, levels = _build_custom_colormap(num_divs, linthresh, zmin, zmax)

    factor = 1
    fig, ax = plt.subplots(figsize=(factor * PLOT_WIDTH, factor * (PLOT_WIDTH - 1.5)))
    contour = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, norm=norm, extend='both')
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Max Lyapunov Exponent', fontsize=X_LABEL_FONT_SIZE)

    ticks = levels
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{v:.1e}' for v in ticks])
    cbar.ax.tick_params(labelsize=LEGEND_FONT_SIZE)

    # Axis labels
    ax.set_xlabel(xlabel, fontsize=X_LABEL_FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=Y_LABEL_FONT_SIZE)

    # Clean ticks: only bottom and left, only major
    ax.set_xticks([1, 10, 20, 30, 40, 50])
    ax.tick_params(axis="x", labelsize=X_TICK_FONT_SIZE,
                   direction="out", length=4, width=1,
                   top=False, bottom=True)

    ax.tick_params(axis="y", labelsize=Y_TICK_FONT_SIZE,
                   direction="out", length=4, width=1,
                   left=True, right=False)

    plt.tight_layout()

    if low_pa:
        filename = filename.replace(".pdf", "_01.pdf")

    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.show()
