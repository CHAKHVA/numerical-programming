"""ODE solvers for ball trajectory simulation."""

import numpy as np

from constants import GRAVITY


class ODESolver:
    def __init__(self, g: float = GRAVITY):
        self.g = g

    def euler_step(self, state: np.ndarray, dt: float) -> np.ndarray:
        """Euler method for one step."""
        x, y, vx, vy = state
        new_x = x + vx * dt
        new_y = y + vy * dt
        new_vx = vx
        new_vy = vy + self.g * dt
        return np.array([new_x, new_y, new_vx, new_vy])

    def rk4_step(self, state: np.ndarray, dt: float) -> np.ndarray:
        """Runge-Kutta 4th order method for one step."""

        def f(s, t):
            _, _, vx, vy = s
            return np.array([vx, vy, 0, self.g])

        k1 = f(state, 0)
        k2 = f(state + 0.5 * dt * k1, dt / 2)
        k3 = f(state + 0.5 * dt * k2, dt / 2)
        k4 = f(state + dt * k3, dt)

        return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
