import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.bubble_models import create_trajectories
from algorithms import lyapunov as lya

# ================== FIXED / USER PARAMS (C3) ==================
equation: str = "G"     # 'RP', 'KM', 'G'
temperature: float = 20  # Celsius
periods: int = 10


#("C1", 0.3e6, 1.2e6, 10e-6),
#("C2", 1.5e6, 1.2e6, 5e-6)
#("C3", 2.0e6, 0.8e6, 0.08e-6)

configuration_name = "C1"
acoustic_pressure = 0.3e6   # Pa
frequency = 1.2e6           # Hz  
initial_radius = 10e-6    # m

# Integrator step control for create_trajectories: ALWAYS 1e-3
step_for_creator = 1e-3

integration_time = np.arange(0.0, periods / frequency, step_for_creator / frequency)

# ================== Simulate ==================
trajectories, model = create_trajectories(
    [equation],
    temperature,
    acoustic_pressure,
    frequency,
    initial_radius,
    integration_time,
    step_for_creator
)

radius_data   = trajectories[f"Radius_{equation}"]
velocity_data = trajectories[f"Velocity_{equation}"]

#np.savetxt('radius_data.txt', radius_data)
#np.savetxt('velocity_data.txt', velocity_data)


#plt.plot(integration_time * frequency, radius_data)
#plt.show()

#plt.plot(integration_time* frequency, velocity_data)
#plt.show()

lce_qr, hist = lya.compute_lyapunov_exponents_from_trajectory(
    radius_data,
    velocity_data,
    integration_time * frequency,     # keep your scaling convention (time * f)
    model,
    lya.equation_name_dd[equation],
    keep=True)

print('LCE (QR): ', lce_qr)
plt.plot(integration_time * frequency, hist[:, 0], label="λ1 (QR)")
plt.plot(integration_time * frequency, hist[:, 1], label="λ2 (QR)")
plt.xlabel("Time")
plt.ylabel("Lyapunov Exponents")
plt.legend()
plt.show()

eigvals, hist_eig = lya.compute_lyapunov_from_eigenvalue_product_fixed(
    radius_data, velocity_data, integration_time * frequency,
    model, lya.equation_name_dd[equation], keep=True)

print('LCE (Eig): ', eigvals)
plt.plot(integration_time * frequency, hist_eig[:, 0], label="λ1 (Eig)")
plt.plot(integration_time * frequency, hist_eig[:, 1], label="λ2 (Eig)")    
plt.xlabel("Time")
plt.ylabel("Lyapunov Exponents")
plt.legend()
plt.show()


sum_lce, his_det = lya.compute_lyapunov_sum_from_determinants_fixed(
    radius_data, velocity_data, integration_time * frequency,
    model, lya.equation_name_dd[equation], keep=True)

print('LCE (Det): ', sum_lce)
plt.plot(integration_time * frequency, his_det, label="λ1 (Det)")
plt.show()