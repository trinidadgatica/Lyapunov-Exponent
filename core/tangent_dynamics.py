import numpy as np


def rk4_step_tangent_map(W: np.ndarray, J: np.ndarray, dt: float) -> np.ndarray:
    """
    One step RK4 for tangent linear map evolution.
    """
    k1 = J @ W
    k2 = J @ (W + 0.5 * dt * k1)
    k3 = J @ (W + 0.5 * dt * k2)
    k4 = J @ (W + dt * k3)
    return W + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def _build_tangent_map_from_jacobian(J: np.ndarray, dt: float) -> np.ndarray:
    """Per-step flow propagator Phi ≈ exp(J*dt) """
    A = J * dt

    return np.eye(J.shape[0]) + A + 0.5 * (A @ A)