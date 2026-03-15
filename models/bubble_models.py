import numpy as np
from experiments.experiment_maker import BubbleExperimentFactory
from core.main import BubbleModelBase

# Physical constants
ATMOSPHERIC_PRESSURE = 1e5
ADIABATIC_INDEX = 1.33
VAPOR_PRESSURE = 3.2718e3

def simulate_bubble_trajectories(
    equation_list: list,
    temperature: float,
    acoustic_pressure: float,
    frequency: float,
    initial_radius: float,
    times: np.ndarray,
    step: float
) -> tuple[dict, BubbleExperimentFactory]:
    """
    Generates bubble dynamics trajectories using selected models.

    Args:
        equation_list (list): List of model equation names ('RP', 'KM', 'G').
        temperature (float): Fluid temperature.
        acoustic_pressure (float): Acoustic pressure.
        frequency (float): Driving frequency.
        initial_radius (float): Initial bubble radius.
        times (np.ndarray): Array of time points.
        step (float): Integration step size.

    Returns:
        tuple: (dict of simulation results per equation, ExperimentMaker instance)
    """
    initial_velocity = 0

    # Temperature-dependent fluid properties
    sound_velocity = BubbleModelBase.sound_velocity_generator_temperature(temperature)
    surface_tension = BubbleModelBase.surface_tension_generator_temperature(temperature)
    density = BubbleModelBase.density_generator_temperature(temperature)
    viscosity = BubbleModelBase.viscosity_generator_temperature(temperature)

    model = BubbleExperimentFactory(
        acoustic_pressure,
        frequency,
        initial_radius,
        initial_velocity,
        ATMOSPHERIC_PRESSURE,
        surface_tension,
        density,
        viscosity,
        sound_velocity,
        VAPOR_PRESSURE,
        ADIABATIC_INDEX
    )

    results = {}

    if 'RP' in equation_list:
        inertial, pressure, radius, velocity = model.RP_functions(
            time=times * frequency, solver='ODEINT', step=step * frequency)
        results.update({
            'Radius_RP': radius,
            'Velocity_RP': velocity,
            'Inertial_RP': inertial,
            'Pressure_RP': pressure
        })

    if 'KM' in equation_list:
        inertial, pressure, radius, velocity = model.KM_functions(
            time=times * frequency, solver='ODEINT', step=step * frequency)
        results.update({
            'Radius_KM': radius,
            'Velocity_KM': velocity,
            'Inertial_KM': inertial,
            'Pressure_KM': pressure
        })

    if 'G' in equation_list:
        inertial, pressure, radius, velocity = model.G_functions(
            time=2 * np.pi * times * frequency, solver='ODEINT', step=2 * np.pi * step * frequency)
        results.update({
            'Radius_G': radius,
            'Velocity_G': velocity,
            'Inertial_G': inertial,
            'Pressure_G': pressure
        })

    return results, model
