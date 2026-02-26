import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from algorithms.dimensions import compute_lyapunov_dimensions
from utils.plot_information import *


def create_d2_map_fixed_freq(
    initial_radii: np.ndarray,
    pressures: np.ndarray,
    grid_size: int,
    results: np.ndarray,
    equation_name: str
) -> None:
    """
    Create D₂ contour plot for constant frequency across pressure-radius space.
    """
    x = np.linspace(initial_radii.min(), initial_radii.max(), grid_size)
    y = np.linspace(pressures.min(), pressures.max(), grid_size)
    X, Y = np.meshgrid(x, y)

    d2_vals = [compute_lyapunov_dimensions(r)["Correlation D₂ (approx)"] for r in results]
    Z = np.clip(np.nan_to_num(d2_vals, nan=2.5), 0, 2.5).reshape(X.shape, order='F')

    levels = np.linspace(0, 2.5, 300)
    norm = Normalize(vmin=0, vmax=2.5)

    factor = 0.7
    plt.figure(figsize=(factor * PLOT_WIDTH, factor * (PLOT_WIDTH - 1.5)))
    contour = plt.contourf(X, Y, Z, levels=levels, norm=norm, cmap='viridis', extend='both')

    cbar = plt.colorbar(contour)
    cbar.set_label('Correlation Dimension $D_2$', fontsize=X_LABEL_FONT_SIZE)
    cbar.ax.tick_params(labelsize=LEGEND_FONT_SIZE)

    plt.xlabel('Initial Radius (μm)', fontsize=X_LABEL_FONT_SIZE)
    plt.ylabel('Acoustic Pressure (MPa)', fontsize=Y_LABEL_FONT_SIZE)
    plt.xticks([1, 10, 20, 30, 40, 50], fontsize=X_TICK_FONT_SIZE)
    plt.yticks([0.2, 0.5, 1, 1.5, 2, 2.5, 3], fontsize=Y_TICK_FONT_SIZE)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/{equation_name}_d2_map_freq.pdf", dpi=300)
    plt.show()


def create_d2_map_fixed_pa(
    initial_radii: np.ndarray,
    freqs: np.ndarray,
    grid_size: int,
    results: np.ndarray,
    equation_name: str,
    low_pressure: bool = False
) -> None:
    """
    Create D₂ contour plot for constant pressure across freq-radius space.
    """
    x = np.linspace(initial_radii.min(), initial_radii.max(), grid_size)
    y = np.linspace(freqs.min(), freqs.max(), grid_size)
    X, Y = np.meshgrid(x, y)

    d2_vals = [compute_lyapunov_dimensions(r)["Correlation D₂ (approx)"] for r in results]
    Z = np.clip(np.nan_to_num(d2_vals, nan=2.5), 0, 2.5).reshape(X.shape, order='F')

    levels = np.linspace(0, 2.5, 300)
    norm = Normalize(vmin=0, vmax=2.5)

    factor = 0.7
    plt.figure(figsize=(factor * PLOT_WIDTH, factor * (PLOT_WIDTH - 1.5)))
    contour = plt.contourf(X, Y, Z, levels=levels, norm=norm, cmap='viridis', extend='both')

    cbar = plt.colorbar(contour)
    cbar.set_label('Correlation Dimension $D_2$', fontsize=X_LABEL_FONT_SIZE)
    cbar.ax.tick_params(labelsize=LEGEND_FONT_SIZE)

    plt.xlabel('Initial Radius (μm)', fontsize=X_LABEL_FONT_SIZE)
    plt.ylabel('Frequency (MHz)', fontsize=Y_LABEL_FONT_SIZE)
    plt.xticks([1, 10, 20, 30, 40, 50], fontsize=X_TICK_FONT_SIZE)
    plt.yticks([0.02, 0.4, 0.8, 1.2, 1.6, 2], fontsize=Y_TICK_FONT_SIZE)
    plt.grid(True)
    plt.tight_layout()

    suffix = "_pa_01" if low_pressure else "_pa"
    plt.savefig(f"results/{equation_name}_d2_map{suffix}.pdf", dpi=300)
    plt.show()
