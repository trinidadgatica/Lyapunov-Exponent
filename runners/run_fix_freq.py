import sys
import os
from itertools import product
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

print(f"[DEBUG] Root added to sys.path: {project_root}")

from algorithms.lyapunov import compute_lyapunov_grid


# Fixed frequency
frequency = 1e6  # Hz
temperature = 20  # °C

N = 50
initial_radii = np.linspace(1, 50, N)
acoustic_pressures = np.linspace(0.2, 3, N)  # MPa

# Grid of (radius, pressure)
grid = list(product(initial_radii, acoustic_pressures))

for eq in ['RP', 'KM', 'G']:
    results = compute_lyapunov_grid(
        grid=grid,
        equation=eq,
        temperature=temperature,
        frequency=frequency,
        pressure=None,
        filename_suffix="_fix_freq"
    )
    np.save(f"results/{eq}_fix_freq.npy", results, allow_pickle=True)
