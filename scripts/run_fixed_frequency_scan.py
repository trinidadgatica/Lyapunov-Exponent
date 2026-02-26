from itertools import product
import numpy as np

from core.lyapunov import compute_lyapunov_grid

# Fixed frequency
frequency: float = 1e6  # Hz
temperature: float = 20  # °C

N: int = 50
initial_radii: np.ndarray = np.linspace(1, 50, N)
acoustic_pressures: np.ndarray = np.linspace(0.2, 3, N)  # MPa

# Grid of (radius, pressure)
grid: list[tuple[float, float]] = list(product(initial_radii, acoustic_pressures))

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
