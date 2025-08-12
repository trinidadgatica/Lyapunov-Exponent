import sys
import os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

print(f"[DEBUG] Root added to sys.path: {project_root}")

from visualization.le_maps import plot_le_map

N = 50
initial_radii = np.linspace(1, 50, N)
acoustic_pressures = np.linspace(0.2, 3, N)
frequencies = np.linspace(0.02, 2, N)

# --- Load results for fixed frequency sweep ---
res_RP_f = np.load("results/RP_fix_freq.npy", allow_pickle=True)
res_KM_f = np.load("results/KM_fix_freq.npy", allow_pickle=True)
res_G_f = np.load("results/G_fix_freq.npy", allow_pickle=True)

# --- Plot LE maps (fixed frequency) ---
for eq, data in zip(['RP', 'KM', 'G'], [res_RP_f, res_KM_f, res_G_f]):
    max_le = np.max(data, axis=1)
    plot_le_map(
        x=initial_radii,
        y=acoustic_pressures,
        max_exponents=max_le,
        equation=eq,
        xlabel='Initial Radius (μm)',
        ylabel='Acoustic Pressure (MPa)',
        filename=f'results/LE_map_freq_{eq}.pdf'
    )

# --- Load results for fixed pressure sweep ---
res_RP_pa = np.load("results/RP_fix_pa.npy", allow_pickle=True)
res_KM_pa = np.load("results/KM_fix_pa.npy", allow_pickle=True)
res_G_pa = np.load("results/G_fix_pa.npy", allow_pickle=True)

res_RP_pa_01 = np.load("results/RP_fix_pa_01.npy", allow_pickle=True)
res_KM_pa_01 = np.load("results/KM_fix_pa_01.npy", allow_pickle=True)
res_G_pa_01 = np.load("results/G_fix_pa_01.npy", allow_pickle=True)

# --- Plot LE maps (fixed pressure) ---
for eq, data in zip(['RP', 'KM', 'G'], [res_RP_pa, res_KM_pa, res_G_pa]):
    max_le = np.max(data, axis=1)
    plot_le_map(
        x=initial_radii,
        y=frequencies,
        max_exponents=max_le,
        equation=eq,
        xlabel='Initial Radius (μm)',
        ylabel='Frequency (MHz)',
        filename=f'results/LE_map_pa_{eq}.pdf'
    )

# --- Plot LE maps (low pressure variant) ---
for eq, data in zip(['RP', 'KM', 'G'], [res_RP_pa_01, res_KM_pa_01, res_G_pa_01]):
    max_le = np.max(data, axis=1)
    plot_le_map(
        x=initial_radii,
        y=frequencies,
        max_exponents=max_le,
        equation=eq,
        xlabel='Initial Radius (μm)',
        ylabel='Frequency (MHz)',
        filename=f'results/LE_map_pa_{eq}.pdf',
        low_pa=True
    )
