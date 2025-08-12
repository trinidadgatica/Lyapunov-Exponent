import sys
import os
from itertools import product
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

print(f"[DEBUG] Root added to sys.path: {project_root}")

from algorithms.lyapunov import compute_lyapunov_grid

# Fixed pressure (low)
acoustic_pressure = 0.1e6  # Pa
temperature = 20  # °C

N = 50
initial_radii = np.linspace(1, 50, N)
frequencies = np.linspace(0.02, 2, N)  # MHz

# Grid of (radius, frequency)
grid = list(product(initial_radii, frequencies))

for eq in ['RP', 'KM', 'G']:
    results = compute_lyapunov_grid(
        grid=grid,
        equation=eq,
        temperature=temperature,
        pressure=acoustic_pressure,
        frequency=None,
        filename_suffix="_fix_pa_01"
    )
    np.save(f"results/{eq}_fix_pa_01.npy", results, allow_pickle=True)
