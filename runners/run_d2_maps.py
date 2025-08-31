import numpy as np
from visualization.d2_maps import create_d2_map_fixed_freq, create_d2_map_fixed_pa

N: int = 50
initial_radii: np.ndarray = np.linspace(1, 50, N)

# --- D2 maps with fixed frequency ---
pressures: np.ndarray = np.linspace(0.2, 3, N)
results_RP_f = np.load("results/RP_fix_freq.npy", allow_pickle=True)
results_KM_f = np.load("results/KM_fix_freq.npy", allow_pickle=True)
results_G_f = np.load("results/G_fix_freq.npy", allow_pickle=True)

create_d2_map_fixed_freq(initial_radii, pressures, N, results_RP_f, 'RP')
create_d2_map_fixed_freq(initial_radii, pressures, N, results_KM_f, 'KM')
create_d2_map_fixed_freq(initial_radii, pressures, N, results_G_f, 'G')

# --- D2 maps with fixed pressure ---
frequencies: np.ndarray = np.linspace(0.02, 2, N)
results_RP_pa = np.load("results/RP_fix_pa.npy", allow_pickle=True)
results_KM_pa = np.load("results/KM_fix_pa.npy", allow_pickle=True)
results_G_pa = np.load("results/G_fix_pa.npy", allow_pickle=True)

results_RP_pa_01 = np.load("results/RP_fix_pa_01.npy", allow_pickle=True)
results_KM_pa_01 = np.load("results/KM_fix_pa_01.npy", allow_pickle=True)
results_G_pa_01 = np.load("results/G_fix_pa_01.npy", allow_pickle=True)

create_d2_map_fixed_pa(initial_radii, frequencies, N, results_RP_pa, 'RP')
create_d2_map_fixed_pa(initial_radii, frequencies, N, results_KM_pa, 'KM')
create_d2_map_fixed_pa(initial_radii, frequencies, N, results_G_pa, 'G')

create_d2_map_fixed_pa(initial_radii, frequencies, N, results_RP_pa_01, 'RP', low_pressure=True)
create_d2_map_fixed_pa(initial_radii, frequencies, N, results_KM_pa_01, 'KM', low_pressure=True)
create_d2_map_fixed_pa(initial_radii, frequencies, N, results_G_pa_01, 'G', low_pressure=True)
