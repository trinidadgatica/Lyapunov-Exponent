import warnings
import numpy as np
from scipy.integrate import odeint, quad
from scipy.signal import find_peaks
from scipy.optimize import root

from core.main import Model
from core.ode_runner import Solver

warnings.filterwarnings("ignore")


class ExperimentMaker(Model):
    def __int__(self, model_instance: Model):
        super().__init__(model_instance.pa, model_instance.f, model_instance.r0,
                         model_instance.j0, model_instance.p0, model_instance.sigma,
                         model_instance.rho, model_instance.mu, model_instance.c,
                         model_instance.pv, model_instance.kappa)

    def RP_functions(self, time: np.ndarray, solver: str, step: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Function that calculates inertial and pressure functions for the Rayleigh-Plesset equation.

        :param time: Variable of time.
        :type time: numpy.ndarray

        :return: A tuple containing the inertial function (numpy array), pressure function (numpy array),
        array of radius, and array of velocity.

        References:
        - H.G. Flynn (1975). Cavitation dynamics I.
        """
        if solver == "ODEINT":
            result_rayleigh_plesset = odeint(self.rayleigh_plesset_equation, [1, 0], time, tfirst=True,
                                             atol=1e-16)
        elif solver == "ODE":
            result_rayleigh_plesset = Solver.runner_ode_rp(self, time, step)

        else:
            raise ValueError("Invalid equation")

        radius = result_rayleigh_plesset[:, 0]
        velocity = result_rayleigh_plesset[:, 1]

        reynolds_number_fl = (self.initial_radius ** 2 * self.density) / (4 * self.viscosity * self.period)
        weber_number_fl = (2 * self.surface_tension * self.period ** 2) / (self.density *
                                                                           (
                                                                                   self.initial_radius ** 3))
        non_dimensional_pressure_fl = self.acoustic_pressure
        thoma_number_fl = (self.period ** 2) * non_dimensional_pressure_fl / (self.density *
                                                                              (
                                                                                      self.initial_radius ** 2))
        pressure_in_infinity_fl = self.atmospheric_pressure + \
                                  self.acoustic_pressure * np.sin(
            self.angular_frequency * time * self.period)
        non_dimensional_initial_pressure_fl = ((self.period ** 2) / (self.initial_radius ** 2)) * \
                                              (self.atmospheric_pressure + (
                                                      2 * self.surface_tension / self.initial_radius) -
                                               self.vapor_pressure) / self.density

        external_pressure_wall = - thoma_number_fl * (
                pressure_in_infinity_fl + self.vapor_pressure) / self.acoustic_pressure
        internal_pressure = non_dimensional_initial_pressure_fl / (radius ** (3 * self.adiabatic_index))
        surface_tension_effect = - weber_number_fl / radius
        viscosity_effect = - (1 / reynolds_number_fl) * (velocity / radius)

        pressure_contribution = external_pressure_wall + internal_pressure + surface_tension_effect + viscosity_effect

        IF = - (3 * (velocity ** 2) / (radius * 2))
        PF = pressure_contribution / radius

        return np.array(IF), np.array(PF), radius, velocity

    def KM_functions(self, time: np.ndarray, solver: str, step: float = 0.001) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Function that calculates inertial and pressure functions for the Keller-Miksis equation.

        :param step: Variable of step in time.
        :type step: float

        :param time: Variable of time.
        :type time: numpy.ndarray

        :return: A tuple containing the inertial function (numpy array), pressure function (numpy array),
        array of radius, and array of velocity.

        References:
        - H.G. Flynn (1975). Cavitation dynamics I.
        """

        if solver == "ODEINT":
            result_keller_miksis = odeint(self.keller_miksis_equation, [1, 0], time, tfirst=True, atol=1e-16)
        elif solver == "ODE":
            result_keller_miksis = Solver.runner_ode_km(self, time, step)
        else:
            raise ValueError("Invalid equation")

        radius = result_keller_miksis[:, 0]
        velocity = result_keller_miksis[:, 1]
        acceleration = np.diff(result_keller_miksis[:, 1]) / step
        acceleration = np.insert(acceleration, 0, 0)

        p_in = (self.atmospheric_pressure - self.vapor_pressure + 2 * (
                self.surface_tension / self.initial_radius)) * \
               (1 / radius) ** (3 * self.adiabatic_index)
        p_out = - self.atmospheric_pressure - 2 * (
                self.surface_tension / (
                radius * self.initial_radius)) - 4 * self.viscosity * (
                    velocity) / (
                        radius * self.period) - self.acoustic_pressure * np.sin(
            self.angular_frequency
            * time * self.period) + self.vapor_pressure
        pressure = p_in + p_out

        eq_1 = (self.period ** 2) / (self.initial_radius ** 2)
        eq_2 = (1 + (velocity * self.initial_radius / (
                self.period * self.sound_velocity))) * (
                       pressure / self.density)
        eq_3 = radius * self.initial_radius / (self.density * self.sound_velocity)

        eq_4 = ((self.atmospheric_pressure - self.vapor_pressure +
                 2 * self.surface_tension / self.initial_radius) *
                (- 3 * self.adiabatic_index * (radius ** (- 3 * self.adiabatic_index - 1))
                 * velocity / self.period))

        eq_5 = 2 * self.surface_tension * velocity / (
                (radius ** 2) * self.initial_radius * self.period)
        eq_6 = - 4 * self.viscosity * (acceleration / (radius * (self.period ** 2)))
        eq_7 = 4 * self.viscosity * (velocity ** 2) / ((radius ** 2) * (self.period ** 2))
        eq_8 = - self.acoustic_pressure * np.cos(
            2 * np.pi * time * self.period * self.frequency) * 2 * np.pi
        eq_9 = (3 / 2) * (velocity ** 2) * (1 - ((velocity * self.initial_radius) / (
                3 * self.sound_velocity * self.period)))

        pressure_contribution = eq_1 * (eq_2 + eq_3 * (eq_4 + eq_5 + eq_6 + eq_7 + eq_8))

        IF = - eq_9 / ((1 - ((velocity * self.initial_radius) / (self.sound_velocity * self.period))) * radius)
        PF = pressure_contribution / (
                (1 - ((velocity * self.initial_radius) / (self.sound_velocity * self.period))) * radius)

        return np.array(IF), np.array(PF), radius, velocity

    def G_functions(self, time: np.ndarray, solver: str, step: float = 0.001) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Function that calculates inertial and pressure functions for the Gilmore equation.

        :param solver:
        :param time: Variable of time.
        :type time: numpy.ndarray

        :param step: Variable of step in time.
        :type step: float

        :return: A tuple containing the inertial function (numpy array), pressure function (numpy array),
        array of radius, and array of velocity.

        References:
        - H.G. Flynn (1975). Cavitation dynamics I.
        """

        n = 7
        A = 304e6
        B = 303.9e6

        if solver == "ODEINT":
            result_gilmore = odeint(self.gilmore_equation, [1, 0], time, tfirst=True, atol=1e-16)
        elif solver == "ODE":
            result_gilmore = Solver.runner_ode_g(self, time, step)
        else:
            raise ValueError("Invalid equation")

        radius = result_gilmore[:, 0]
        velocity = result_gilmore[:, 1]
        acceleration = np.diff(result_gilmore[:, 1]) / (step * 2 * np.pi * self.frequency)
        acceleration = np.insert(acceleration, 0, 0)

        non_dimensional_enthalpy = self.enthalpy(radius, velocity, time, n, A, B)
        non_dimensional_delta_enthalpy = self.delta_enthalpy(radius, velocity, time, n, A, B)
        non_dimensional_sound_velocity = (self.sound_velocity *
                                          (self.initial_radius * self.angular_frequency) ** (-1))
        non_dimensional_speed_sound = np.sqrt(non_dimensional_sound_velocity ** 2 + (n - 1) * non_dimensional_enthalpy)
        common_term = self.common_factor_gilmore(radius, velocity, B)
        non_dimensional_viscosity = self.viscosity * self.angular_frequency / self.atmospheric_pressure
        non_dimensional_density = self.density * (
                self.initial_radius * self.angular_frequency) ** 2 / self.atmospheric_pressure

        eq_1 = (1 + (velocity / non_dimensional_speed_sound)) * non_dimensional_enthalpy
        eq_2 = (1 / non_dimensional_speed_sound) * (
                1 - (velocity / non_dimensional_speed_sound)) * radius * non_dimensional_delta_enthalpy

        eq_3 = acceleration * radius * (1 + ((A / self.atmospheric_pressure) ** (1 / n)) * (
                common_term ** (- 1 / n)) * 4 * non_dimensional_viscosity / (
                                                non_dimensional_speed_sound * radius * non_dimensional_density))
        eq_4 = (3 / 2) * (1 - (velocity / (3 * non_dimensional_speed_sound))) * (velocity ** 2)

        pressure_contribution = eq_1 + eq_2
        eq_3_1 = radius * (1 + ((A / self.atmospheric_pressure) ** (1 / n)) * (
                common_term ** (- 1 / n)) * 4 * non_dimensional_viscosity / (
                                   non_dimensional_speed_sound * radius * non_dimensional_density))
        IF = - eq_4 / eq_3_1
        PF = pressure_contribution / ((1 - (velocity / non_dimensional_speed_sound)) * radius)

        return np.array(IF), np.array(PF), radius, velocity

    def Jacobian_RP(self, x1: float, x2: float, tau: float) -> np.ndarray:
        gas_pressure = self.atmospheric_pressure +  (2 * self.surface_tension / self.initial_radius) - self.vapor_pressure
        overline_pressure = ((self.period ** 2) * gas_pressure) / ((self.initial_radius ** 2) * self.density)
        
        # thoma_number = overline_pressure * (self.period ** 2) / (self.density * (self.initial_radius ** 2))
        thoma_number = self.atmospheric_pressure * (self.period ** 2) / (self.density * (self.initial_radius ** 2))

        reynolds_number = ((self.initial_radius ** 2) * self.density) / (4 * self.viscosity * self.period)
        weber_number = (2 * self.surface_tension * (self.period ** 2)) / (self.density * (self.initial_radius ** 3))
    
        P_infty = self.acoustic_pressure * np.sin(self.angular_frequency * tau * self.period) + self.atmospheric_pressure

        A = -thoma_number * (P_infty - self.vapor_pressure) / self.atmospheric_pressure
        B =  overline_pressure / (x1 ** (3 * self.adiabatic_index)) - weber_number / x1
        C = - (1 / reynolds_number) * (x2 / x1)
        D = - 1.5 * (x2 ** 2)
        G = A + B + C + D

        dfdx1 = (1 / x1) * (
            -3 * self.adiabatic_index * (overline_pressure / (x1 ** (3 * self.adiabatic_index + 1)))
            + weber_number / (x1 ** 2)
            + (1 / reynolds_number) * (x2 / x1 ** 2)
        ) - (G / (x1 ** 2))
 
        dfdx2 = - (1 /  (reynolds_number * (x1 ** 2))) -  3 * (x2 / x1)

        return np.array([[0, 1], [dfdx1, dfdx2]])

    def Jacobian_KM(self, x1: float, x2: float, tau: float) -> np.ndarray:

        gas_pressure = self.atmospheric_pressure +  (2 * self.surface_tension / self.initial_radius) - self.vapor_pressure
        p_infty = self.acoustic_pressure * np.sin(self.angular_frequency * tau * self.period) + self.atmospheric_pressure
        pressure_surface = (2 * self.surface_tension / (self.initial_radius * x1)) + (4 * self.viscosity * x2 / (x1 * self.period))
        # Pressure at time tau
        P = gas_pressure * (x1** (-3 * self.adiabatic_index)) + self.vapor_pressure - pressure_surface - p_infty
        derivative_factor = (x1 * self.initial_radius / (self.density * self.sound_velocity))

        # A
        A = (3 / 2) * (x2**2) -  (1 / 2) * (x2**3) * self.initial_radius / (self.period * self.sound_velocity)

        # B
        B = (1 + x2 * self.initial_radius/(self.sound_velocity * self.period)) * (P / self.density)

        # C
        C = derivative_factor * gas_pressure * (-3*self.adiabatic_index) * x1**(-3*self.adiabatic_index - 1) * (x2/self.period)

        #D
        D = derivative_factor * (2*self.surface_tension*x2/((x1**2) * self.initial_radius *self.period) + 4*self.viscosity*(x2**2)/((x1*self.period)**2))

        #E
        E = derivative_factor* (- self.acoustic_pressure * np.cos(2 * np.pi * tau * self.period * self.frequency) * 2 * np.pi)


        K = B + C + D + E
        L = -A + (self.period**2/self.initial_radius**2) * K
        M = (1 - x2*self.initial_radius/(self.sound_velocity*self.period))*x1 \
            + derivative_factor * (self.period**2/self.initial_radius**2)*(4*self.viscosity/(x1*self.period))

        # Partial derivatives
        # dB
        dP_dx1 = gas_pressure * (-3*self.adiabatic_index) * x1**(-3*self.adiabatic_index - 1) \
                + (2*self.surface_tension/(self.initial_radius* x1**2)) + (4*self.viscosity*x2/(self.period * x1**2))
        dB_dx1 = (1 + x2 * self.initial_radius /(self.sound_velocity*self.period)) * (1/self.density) * dP_dx1
        dB_dx2 = self.initial_radius * P / (self.sound_velocity * self.period * self.density) 
        - (1 + x2 * self.initial_radius/(self.sound_velocity*self.period)) * (1 / self.density) * (4 * self.viscosity / (self.period * x1))

        # dC
        dC_dx1 = derivative_factor * gas_pressure * (- 3 * self.adiabatic_index) * (-3 * self.adiabatic_index - 1) \
            *(x1 ** (-3 * self.adiabatic_index -2)) * x2 / self.period
        dC_dx2 = derivative_factor * gas_pressure * (-3*self.adiabatic_index)*x1**(-3*self.adiabatic_index - 1)*(1/self.period) +\
        (self.initial_radius / (self.density * self.sound_velocity)) * gas_pressure * (-3*self.adiabatic_index)*x1**(-3*self.adiabatic_index - 1)*(x2/self.period) 

        # dD
        dD_dx1 = derivative_factor * (
            (-4 * self.surface_tension * x2) / (x1**3 * self.period) - (8 * self.viscosity * x2**2) / (x1**3 * self.period**2)
        )

        dD_dx2 = derivative_factor * (2 * self.surface_tension / (x1 * self.period) + (8 * self.viscosity * x2) / ((x1 * self.period)**2)) +\
        (self.initial_radius / (self.density * self.sound_velocity)) * (2*self.surface_tension*x2/((x1 ** 2)* self.initial_radius *self.period) + 4*self.viscosity*(x2**2)/((x1*self.period)**2))

        dE_dx1 = 0
        dE_dx2 = (self.initial_radius / (self.density * self.sound_velocity))* (- self.acoustic_pressure * np.cos(2 * np.pi * tau * self.period * self.frequency) * 2 * np.pi)


        # ∂L/∂x1
        dL_dx1 = (self.period**2 / self.initial_radius**2) * (dB_dx1 + dC_dx1 + dD_dx1 + dE_dx1)

        # ∂L/∂x2
        dL_dx2 = -3 * x2  + 1.5 * (x2**2) * self.initial_radius / (self.sound_velocity * self.period)+ (self.period**2 / self.initial_radius**2) * (dB_dx2 + dC_dx2 + dD_dx2 + dE_dx2)

        # ∂M/∂x1
        dM_dx1 = (1 - x2 * self.initial_radius / (self.sound_velocity * self.period))

        # ∂M/∂x2
        dM_dx2 = - self.initial_radius * x1 / (self.sound_velocity * self.period)

        # Jacobian entries using quotient rule
        dfdx1 = (dL_dx1 * M - dM_dx1 * L) / M**2
        dfdx2 = (dL_dx2 * M - dM_dx2 * L) / M**2

        return np.array([[0, 1], [dfdx1, dfdx2]])

    def Jacobian_G(self, x1: float, x2: float, tau: float) -> np.ndarray:
        """
        Calculate the Jacobian matrix of the Gilmore equation at fixed nondimensional time tau.
        
        x1: nondimensional radius Rn
        x2: nondimensional radial velocity Ṙn
        tau: nondimensional time
        """
        # Constants and aliases
        n = 7
        A = 304e6
        B = 303.9e6
        T = 1 / self.frequency
        R0 = self.initial_radius
        mu = self.viscosity
        sigma = self.surface_tension
        rho = self.density
        c = self.sound_velocity
        P0 = self.atmospheric_pressure
        PA = self.acoustic_pressure
        gamma = self.adiabatic_index
        
        # Precomputations
        NH1 = (n / (n - 1)) * (A ** (1 / n)) / rho
        NH2 = (P0 + 2 * sigma / R0) * x1 ** (-3 * gamma)
        NH3 = -2 * sigma / (R0 * x1) - 4 * mu * x2 / (T * x1) + B
        NH4 = PA * np.sin(2 * np.pi * tau * T) + P0 + B
        H = NH1 * ((NH2 + NH3) ** (1 - 1 / n) - NH4 ** (1 - 1 / n))

        C = np.sqrt((c * T / R0) ** 2 + (n - 1) * H)

        # dH terms
        NHA = (A ** (1 / n)) / rho
        NHB = (NH2 + NH3) ** (-1 / n)
        
        NHC = (
            (P0 + 2 * sigma / R0) * (-3 * gamma) * x1 ** (-3 * gamma - 1) * x2 / T
            + 2 * sigma * x2 / (T * R0 * x1 ** 2)
            + 4 * mu * x2 ** 2 / ((T ** 2) * (x1 ** 2))
        )
        NHD = NH4 ** (-1 / n)
        NHE = PA * np.cos(2 * np.pi * tau * T) * 2 * np.pi * T 

        dH = NHA * (NHB * NHC - NHD * NHE)
        
        dNHB_dx1 = - (1 / n) * (NH2 + NH3) ** ((-1 / n) - 1) * (
            (-3 * gamma) * (P0 + 2 * sigma / R0) * x1 ** (-3 * gamma - 1)
            + 2 * sigma / (R0 * x1 ** 2)
            + 4 * mu * x2 / (T * x1 ** 2)
        )
        dNHB_dx2 = - (1 / n) * (NH2 + NH3) ** ((-1 / n) - 1) * (-4 * mu / (T * x1))
        
        dNHC_dx1 = (
            (P0 + 2 * sigma / R0) * (-3 * gamma) * (-3 * gamma - 1) * x1 ** (-3 * gamma - 2) * x2 / T
            - 4 * sigma * x2 / (T * R0 * x1 ** 3)
            - 8 * mu * x2 ** 2 / ((T ** 2) * (x1 ** 3)))
        
        dNHC_dx2 = (P0 + 2 * sigma / R0) * (-3 * gamma) * x1 ** (-3 * gamma - 1) / T + (2 * sigma / (T * R0 * x1 ** 2)
                + 8 * mu * x2 / ((T ** 2) * (x1 ** 2)))

        ddH_dx1 = NHA * (dNHB_dx1 * NHC + NHB * dNHC_dx1)
        ddH_dx2 = NHA * (dNHB_dx2 * NHC + NHB * dNHC_dx2)

        dNH2_dx1 = (P0 + 2 * sigma / R0) * (-3 * gamma) * (x1 ** (-3 * gamma - 1))

        dNH3_dx1 = 2 * sigma / (R0 * (x1 ** 2)) + 4 * mu * x2 / (T * (x1 ** 2))

        dNH3_dx2 = -4 * mu / (T * x1)

        
        dH_dx1 = NH1 * ((n - 1) / n) * (NH2 + NH3) ** (-1 / n) * (dNH2_dx1 + dNH3_dx1)
        dH_dx2 = NH1 * ((n - 1) / n) * (NH2 + NH3) ** (-1 / n) * dNH3_dx2

        M1 = (1 - R0 * x2 / (T * C))
        M2 = 1 +   1 / (R0 * C) * NHA * NHB * 4 * mu / x1
        M = x1 * M1 * M2

        C_denominator = np.sqrt(((c * T / R0) ** 2) + (n - 1) * H)

        dC_dx1 = (n - 1) / (2 * C_denominator) * dH_dx1
        dC_dx2 = (n - 1) / (2 * C_denominator) * dH_dx2


        dM1_dx1 = R0 * x2 / (T * (C**2)) * dC_dx1
        dM1_dx2 = - R0 / (T * C) + R0 * x2 / (T * (C**2)) * dC_dx2


        dM2_dx1 = ((NHA / R0) * (dNHB_dx1 * 4 * mu / (C * x1) - NHB * dC_dx1 * 4 * mu / (x1 * (C**2))
                - NHB * 4 * mu / ((x1**2) * C)))

        dM2_dx2 = ((NHA * 4 * mu / (R0 * x1)) * (dNHB_dx2 / C - NHB * dC_dx2 / (C**2)))

        dM_dx1 = M1 * M2 + x1 * (dM1_dx1 * M2 + M1 * dM2_dx1)
        dM_dx2 = x1 * (dM1_dx2 * M2 + M1 * dM2_dx2)

        L1 = (-3 / 2) * x2 ** 2 * (1 - R0 * x2 / (3 * T * C))
        L2 = H * (1 + R0 * x2 / (T * C))
        L3 = R0 * (x1 / C) * dH * (1 - R0 * x2 / (T * C))
        L = L1 + (T ** 2 / R0 ** 2) * (L2 + L3)


        dL1_dx1 = -0.5 * (x2**3) * R0 / (T * (C**2)) * dC_dx1
        dL1_dx2 = (-3 * x2 * (1 - R0 * x2 / (3 * T * C)) + 1.5 * (x2**2) * R0 / (3 * T * C) -
            1.5 * (x2**3) * R0 / (3 * T * (C**2)) * dC_dx2)


        dL2_dx1 = dH_dx1 * (1 + R0 * x2 / (T * C)) - H * R0 * x2 / (T * (C**2)) * dC_dx1
        dL2_dx2 = dH_dx2 * (1 + R0 * x2 / (T * C)) + H * R0 / (T * C) - H * R0 * x2 / (T * (C**2)) * dC_dx2


        dL3_dx1 = ((R0 / C) * dH * (1 - R0 * x2 / (T * C)) -
            (R0 * x1 / (C**2)) * dC_dx1 * dH * (1 - R0 * x2 / (T * C)) +
            (R0 * x1 / C) * ddH_dx1 * (1 - R0 * x2 / (T * C)) +
            (R0 * x1 / C) * dH * R0 * (x2 / (T * (C**2))) * dC_dx1)


        dL3_dx2_1 = -(R0 * x1 / (C**2)) * dC_dx2 * dH * (1 - R0 * x2 / (T * C))
        dL3_dx2_2 = (R0 * x1 / C) * ddH_dx2 * (1 - R0 * x2 / (T * C))
        dL3_dx2_3 = - (R0**2) * (x1 / (T * (C**2))) * dH 
        dL3_dx2_4 = ((R0**2) * x1 / (T * (C**3))) * dH * x2 * dC_dx2

        dL3_dx2 = dL3_dx2_1 + dL3_dx2_2 + dL3_dx2_3 + dL3_dx2_4

        dL_dx1 = dL1_dx1 + ((T**2) / (R0**2)) * (dL2_dx1 + dL3_dx1)
        dL_dx2 = dL1_dx2 + ((T**2) / (R0**2)) * (dL2_dx2 + dL3_dx2)


        # df/dx
        df_dx1 = (dL_dx1 * M - dM_dx1 * L) / M ** 2
        df_dx2 = (dL_dx2 * M - dM_dx2 * L) / M ** 2

        return np.array([[0, 1], [df_dx1, df_dx2]])
