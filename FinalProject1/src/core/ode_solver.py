import numpy as np

from src.constants import GRAVITY


class ODESolver:
    """Class for solving ODEs using different numerical methods"""

    def __init__(self, gravity: float = GRAVITY):
        self.g = gravity

    def rk4(
        self, f: callable, t0: float, y0: np.ndarray, h: float, n: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Runge-Kutta 4th order method

        Args:
            f: Function that defines the ODE system
            t0: Initial time
            y0: Initial conditions
            h: Step size
            n: Number of steps

        Returns:
            Tuple of (time points, solution)
        """
        t = np.zeros(n + 1)
        y = np.zeros((n + 1, len(y0)))
        t[0] = t0
        y[0] = y0

        for i in range(n):
            k1 = f(t[i], y[i])
            k2 = f(t[i] + h / 2, y[i] + h / 2 * k1)
            k3 = f(t[i] + h / 2, y[i] + h / 2 * k2)
            k4 = f(t[i] + h, y[i] + h * k3)

            y[i + 1] = y[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t[i + 1] = t[i] + h

        return t, y

    def projectile_motion(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Projectile motion equations

        Args:
            t: Time (unused in this case)
            y: State vector [x, y, vx, vy]

        Returns:
            Derivatives [dx/dt, dy/dt, dvx/dt, dvy/dt]
        """
        _, _, vx, vy = y
        return np.array([vx, vy, 0, -self.g])
