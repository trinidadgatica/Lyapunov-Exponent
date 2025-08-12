import numpy as np
from algorithms.experiment_maker import ExperimentMaker
from algorithms.main import Model

# Physical constants
ATMOSPHERIC_PRESSURE = 1e5
ADIABATIC_INDEX = 1.33
VAPOR_PRESSURE = 3.2718e3

def create_trajectories(equation_list, temperature, acoustic_pressure, frequency, initial_radius, times, step):
    """
    Generates bubble dynamics trajectories using selected models.

    Returns:
        dict: Simulation results per equation.
        ExperimentMaker: Initialized model for Jacobian access.
    """
    initial_velocity = 0

    # Temperature-dependent fluid properties
    sound_velocity = Model.sound_velocity_generator_temperature(temperature)
    surface_tension = Model.surface_tension_generator_temperature(temperature)
    density = Model.density_generator_temperature(temperature)
    viscosity = Model.viscosity_generator_temperature(temperature)

    model = ExperimentMaker(
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
