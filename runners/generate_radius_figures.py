import numpy as np
import matplotlib.pyplot as plt

from models.bubble_models import create_trajectories

temperature: float = 20
periods: int = 10
step: float = 1e-3

# --- Configurations ---
configs: list[tuple[str, float, float, float]] = [
    ("C1", 0.3e6, 1.2e6, 10e-6),     # Pa, f, R0
    ("C2", 1.5e6, 1.2e6, 5e-6)
]

for cfg_name, pressure, frequency, initial_radius in configs:
    print(f"\n=== Running {cfg_name} ===")

    integration_time = np.arange(0, periods / frequency, step / frequency)

    trajectories, model = create_trajectories(['RP', 'KM', 'G'], temperature, pressure, frequency, initial_radius, integration_time, step)

    radius_rp = trajectories['Radius_RP']
    radius_km = trajectories['Radius_KM']
    radius_g = trajectories['Radius_G']

    plt.plot(integration_time * frequency, radius_rp, label='RP')
    plt.title(f'Rayleigh-Plesset ({cfg_name})')
    plt.xlabel('Periods')
    plt.ylabel('Radius (non-dimensional)')
    plt.savefig(f'results/Radius_RP_{cfg_name}.pdf')
    plt.show()

    plt.plot(integration_time * frequency, radius_km, label='KM')
    plt.title(f'Keller-Miksis ({cfg_name})')
    plt.xlabel('Periods')
    plt.ylabel('Radius (non-dimensional)')
    plt.savefig(f'results/Radius_KM_{cfg_name}.pdf')
    plt.show()

    plt.plot(integration_time * frequency, radius_g, label='G')
    plt.title(f'Gilmore ({cfg_name})')
    plt.xlabel('Periods')
    plt.ylabel('Radius (non-dimensional)')
    plt.savefig(f'results/Radius_G_{cfg_name}.pdf')
    plt.show()

