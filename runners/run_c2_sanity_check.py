# quiero agregar la serie de tiempo de C2 , radio y velocity 

# agregar los valores que toman los lyapunov exponents en el tiempo para QR 


import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.bubble_models import create_trajectories
from algorithms import lyapunov as lya

# ================== FIXED / USER PARAMS (C3) ==================
equation: str = "G"     # 'RP', 'KM', 'G'
temperature: float = 20  # Celsius
periods: int = 100

configuration_name = "C2"
acoustic_pressure = 1.5e6   # Pa
frequency = 1.2e6           # Hz  
initial_radius = 5e-6    # m

step_for_creator = 1e-3
integration_time = np.arange(0.0, periods / frequency, step_for_creator / frequency)


trajectories, model = create_trajectories([equation], temperature, acoustic_pressure, frequency, initial_radius, integration_time, step_for_creator)


plt.plot(integration_time * frequency, trajectories[f"Radius_{equation}"])
plt.xlabel('Time (s)')
plt.ylabel('Radius (m)')
plt.title(f'Bubble Radius over Time ({configuration_name})')
plt.grid()
plt.savefig(f'C2_Radius_Time_{periods}.pdf')
plt.show()

plt.plot(integration_time * frequency, trajectories[f"Velocity_{equation}"])
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title(f'Bubble Velocity over Time ({configuration_name})')
plt.grid()
plt.savefig(f'C2_Velocity_Time_{periods}.pdf')
plt.show()

LCE, hist = lya.compute_lyapunov_exponents_from_trajectory(trajectories[f"Radius_{equation}"], trajectories[f"Velocity_{equation}"],
    integration_time * frequency, model, lya.equation_name_dd[equation], keep=True)

plt.plot(integration_time * frequency, hist[:, 0], label='LCE 1')
plt.plot(integration_time * frequency, hist[:, 1], label='LCE 2')
plt.xlabel('Time (s)')
plt.ylabel('Lyapunov Exponents')
plt.title(f'Lyapunov Exponents over Time ({configuration_name})')
plt.legend()
plt.grid()
plt.savefig(f'C2_Lyapunov_Exponents_{periods}.pdf')
plt.show()

