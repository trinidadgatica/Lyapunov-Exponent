from itertools import product
import numpy as np

from core.lyapunov import compute_lyapunov_grid

# Fixed pressure (low)
acoustic_pressure: float = 0.1e6  # Pa
temperature: float = 20  # °C

N: int = 50
initial_radii: np.ndarray = np.linspace(1, 50, N)
frequencies: np.ndarray = np.linspace(0.02, 2, N)  # MHz

# Grid of (radius, frequency)
grid: list[tuple[float, float]] = list(product(initial_radii, frequencies))

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
