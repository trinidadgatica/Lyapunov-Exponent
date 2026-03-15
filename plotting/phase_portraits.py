from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import BoundaryNorm, Normalize
from tabulate import tabulate

from core.lyapunov import EQUATION_DISPLAY_NAMES, compute_lce_qr_from_trajectory
from models.bubble_models import simulate_bubble_trajectories
from plotting.styles import (
    LEGEND_FONT_SIZE,
    PLOT_WIDTH,
    X_LABEL_FONT_SIZE,
    X_TICK_FONT_SIZE,
    Y_LABEL_FONT_SIZE,
    Y_TICK_FONT_SIZE,
    primary_color,
    quaternary_color,
)


def plot_stable_rp_phase_portrait(
    save_path: str = "figures/stable_RP.pdf",
    show: bool = True,
    close: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    temperature = 20
    periods = 10
    equation = "RP"

    acoustic_pressure = 0.3e6
    frequency = 1.2e6
    initial_radius = 10e-6
    step = 1e-3 / frequency
    times = np.arange(0, periods / frequency, step)
    time = times * frequency

    trajectories, general_model = simulate_bubble_trajectories(
        [equation],
        temperature,
        acoustic_pressure,
        frequency,
        initial_radius,
        times,
        step,
    )

    radius = trajectories[f"Radius_{equation}"]
    velocity = (
        trajectories[f"Velocity_{equation}"]
        * initial_radius
        * frequency
        / general_model.sound_velocity
    )

    factor = 1.4
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(factor * PLOT_WIDTH, (PLOT_WIDTH - 2)),
        gridspec_kw={"width_ratios": [1, 1]},
    )

    ax_r = axes[0]
    ax_v = ax_r.twinx()

    ax_r.plot(time, radius, color=primary_color, label=r"$R_n$")
    ax_v.plot(
        time,
        velocity,
        color=quaternary_color,
        linestyle="--",
        label=r"$\dot{R}_n$",
    )

    ax_r.set_xlabel("Time (periods)", fontsize=X_LABEL_FONT_SIZE)
    ax_r.set_ylabel(r"$R_n$", fontsize=Y_LABEL_FONT_SIZE, color=primary_color)
    ax_v.set_ylabel(r"$\dot{R}_n$", fontsize=Y_LABEL_FONT_SIZE, color=quaternary_color)

    ax_r.tick_params(axis="y", labelsize=Y_TICK_FONT_SIZE, colors=primary_color)
    ax_v.tick_params(axis="y", labelsize=Y_TICK_FONT_SIZE, colors=quaternary_color)
    ax_r.tick_params(axis="x", labelsize=X_TICK_FONT_SIZE)

    lines_r, labels_r = ax_r.get_legend_handles_labels()
    lines_v, labels_v = ax_v.get_legend_handles_labels()
    ax_r.legend(
        lines_r + lines_v,
        labels_r + labels_v,
        loc="upper right",
        fontsize=LEGEND_FONT_SIZE,
    )

    base_cmap = plt.get_cmap("Blues_r", 256)
    num_points = len(radius)
    colors_local = base_cmap(np.linspace(0, 1, num_points - 1))

    ax_phase = axes[1]
    for i in range(num_points - 1):
        ax_phase.plot(
            radius[i:i + 2],
            velocity[i:i + 2],
            color=colors_local[i],
            lw=1.2,
        )

    ax_phase.set_xlabel(r"$R_n$", fontsize=X_LABEL_FONT_SIZE)
    ax_phase.set_ylabel(r"$\dot{R}_n$", fontsize=Y_LABEL_FONT_SIZE)
    ax_phase.tick_params(axis="x", labelsize=X_TICK_FONT_SIZE)
    ax_phase.tick_params(axis="y", labelsize=Y_TICK_FONT_SIZE)

    levels = np.linspace(0, periods, periods + 1)
    norm = BoundaryNorm(levels, ncolors=base_cmap.N, clip=False)
    sm = plt.cm.ScalarMappable(cmap=base_cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=axes[1], pad=0.08, shrink=0.9)
    cbar.set_label("Periods", fontsize=X_LABEL_FONT_SIZE)
    cbar.set_ticks(np.arange(0, periods + 1, 2))
    cbar.ax.tick_params(labelsize=LEGEND_FONT_SIZE)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    if close:
        plt.close(fig)

    return fig, axes


def create_phase_portrait_composite_figure(
    save_path: str = "figures/composite_phase_3x3.pdf",
    show: bool = True,
    close: bool = False,
) -> tuple[plt.Figure, np.ndarray, list[list[str]]]:
    regimes = ["stable", "transient", "collapse"]
    equations = ["RP", "KM", "G"]
    temperature = 20
    periods = 10

    param_config = {
        "stable": {
            "acoustic_pressure": 0.3e6,
            "frequency": 1.2e6,
            "initial_radius": 10e-6,
        },
        "transient": {
            "acoustic_pressure": 1.5e6,
            "frequency": 1.2e6,
            "initial_radius": 5e-6,
        },
        "collapse": {
            "acoustic_pressure": 2e6,
            "frequency": 0.8e6,
            "initial_radius": 8e-8,
        },
    }

    lyap_table_data: list[list[str]] = []

    factor_x = 1.9
    factor_y = 1.8
    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(factor_x * PLOT_WIDTH, factor_y * (PLOT_WIDTH - 1.5)),
        sharex=False,
        sharey=False,
    )

    cmap = cm.get_cmap("Blues_r")
    norm = Normalize(vmin=0, vmax=100)

    for col, eq in enumerate(equations):
        for row, regime in enumerate(regimes):
            params = param_config[regime]
            frequency = params["frequency"]
            step = 1e-3 / frequency
            times = np.arange(0, periods / frequency, step)
            time = 2 * np.pi * times * frequency if eq == "G" else times * frequency

            trajectories, general_model = simulate_bubble_trajectories(
                [eq],
                temperature,
                params["acoustic_pressure"],
                frequency,
                params["initial_radius"],
                times,
                step,
            )

            radius = trajectories[f"Radius_{eq}"]
            velocity = (
                trajectories[f"Velocity_{eq}"]
                * params["initial_radius"]
                * frequency
                / general_model.sound_velocity
            )

            lce_vals = compute_lce_qr_from_trajectory(
                radius,
                velocity,
                time,
                general_model,
                EQUATION_DISPLAY_NAMES[eq],
                keep=False,
            )

            lyap_table_data.append(
                [
                    EQUATION_DISPLAY_NAMES[eq],
                    regime.capitalize(),
                    f"{lce_vals[0]:.2e}",
                    f"{lce_vals[1]:.2e}",
                ]
            )

            num_points = len(radius)
            colors_phase = cmap(np.linspace(0, 1, num_points - 1))

            ax = axes[row, col]
            for i in range(num_points - 1):
                ax.plot(
                    radius[i:i + 2],
                    velocity[i:i + 2],
                    color=colors_phase[i],
                    lw=2,
                )

            if row == 0:
                ax.set_title(
                    f"{EQUATION_DISPLAY_NAMES[eq]}",
                    fontsize=X_LABEL_FONT_SIZE + 2,
                )

            if col == 0:
                ax.set_ylabel(
                    f"C{row + 1}\n" + r"$\dot{R}_n$",
                    fontsize=Y_LABEL_FONT_SIZE + 2,
                )

            ax.set_xlabel(r"$R_n$", fontsize=X_LABEL_FONT_SIZE)
            ax.tick_params(axis="x", labelsize=X_TICK_FONT_SIZE)
            ax.tick_params(axis="y", labelsize=Y_TICK_FONT_SIZE)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Periods", fontsize=X_LABEL_FONT_SIZE)
    tick_locs = np.linspace(0, 100, 6)
    tick_labels = [f"{i}" for i in range(0, 11, 2)]
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels(tick_labels)
    cbar.ax.tick_params(labelsize=X_TICK_FONT_SIZE)

    print("\nLyapunov Exponents Table:")
    print(
        tabulate(
            lyap_table_data,
            headers=["Equation", "Regime", "λ₁", "λ₂"],
            tablefmt="grid",
        )
    )

    fig.tight_layout(rect=[0, 0, 0.91, 1])

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    if close:
        plt.close(fig)

    return fig, axes, lyap_table_data


def print_phase_portrait_lyapunov_table(lyap_table_data: list[list[str]]) -> None:
    print("\nLyapunov Exponents Table:")
    print(
        tabulate(
            lyap_table_data,
            headers=["Equation", "Regime", "λ₁", "λ₂"],
            tablefmt="grid",
        )
    )