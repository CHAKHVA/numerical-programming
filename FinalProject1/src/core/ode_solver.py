from enum import Enum

import numpy as np

from src.constants import (
    AIR_DENSITY,
    DRAG_COEFFICIENT,
    GRAVITY,
    PROJECTILE_MASS,
    PROJECTILE_RADIUS,
)


class SolverMethod(Enum):
    """Enum for available solver methods"""

    EULER = "euler"
    RK4 = "rk4"
    IMPROVED_EULER = "improved_euler"
    RK2 = "rk2"


class ODESolver:
    """Class for solving ODEs using different numerical methods"""

    def __init__(
        self, gravity: float = GRAVITY, method: SolverMethod = SolverMethod.RK4
    ):
        self.g = gravity
        self.method = method
        # Dictionary mapping method enum to actual solver functions
        self._solvers = {
            SolverMethod.EULER: self._euler,
            SolverMethod.RK4: self._rk4,
            SolverMethod.IMPROVED_EULER: self._improved_euler,
            SolverMethod.RK2: self._rk2,
        }

    def solve(
        self, f: callable, t0: float, y0: np.ndarray, h: float, n: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve ODE using the selected method

        Args:
            f: Function that defines the ODE system
            t0: Initial time
            y0: Initial conditions
            h: Step size
            n: Number of steps

        Returns:
            Tuple of (time points, solution)
        """
        solver = self._solvers.get(self.method)
        if solver is None:
            raise ValueError(f"Unknown solver method: {self.method}")
        return solver(f, t0, y0, h, n)

    def _euler(
        self, f: callable, t0: float, y0: np.ndarray, h: float, n: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Basic Euler method"""
        t = np.zeros(n + 1)
        y = np.zeros((n + 1, len(y0)))
        t[0] = t0
        y[0] = y0

        for i in range(n):
            y[i + 1] = y[i] + h * f(t[i], y[i])
            t[i + 1] = t[i] + h

        return t, y

    def _improved_euler(
        self, f: callable, t0: float, y0: np.ndarray, h: float, n: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Improved Euler method"""
        t = np.zeros(n + 1)
        y = np.zeros((n + 1, len(y0)))
        t[0] = t0
        y[0] = y0

        for i in range(n):
            k1 = f(t[i], y[i])
            k2 = f(t[i] + h, y[i] + h * k1)
            y[i + 1] = y[i] + h / 2 * (k1 + k2)
            t[i + 1] = t[i] + h

        return t, y

    def _rk2(
        self, f: callable, t0: float, y0: np.ndarray, h: float, n: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """2nd order Runge-Kutta method (Heun's method)"""
        t = np.zeros(n + 1)
        y = np.zeros((n + 1, len(y0)))
        t[0] = t0
        y[0] = y0

        for i in range(n):
            k1 = f(t[i], y[i])
            k2 = f(t[i] + 2 * h / 3, y[i] + 2 * h * k1 / 3)

            y[i + 1] = y[i] + h / 4 * (k1 + 3 * k2)
            t[i + 1] = t[i] + h

        return t, y

    def _rk4(
        self, f: callable, t0: float, y0: np.ndarray, h: float, n: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """4th order Runge-Kutta method"""
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
        Projectile motion equations with air resistance

        Args:
            t: Time (unused in this case)
            y: State vector [x, y, vx, vy]

        Returns:
            Derivatives [dx/dt, dy/dt, dvx/dt, dvy/dt]
        """
        _, _, vx, vy = y

        # Calculate velocity magnitude
        v = np.sqrt(vx**2 + vy**2)

        # Air resistance force calculation
        area = np.pi * PROJECTILE_RADIUS**2
        drag_force = 0.5 * AIR_DENSITY * DRAG_COEFFICIENT * area * v**2

        # Calculate acceleration components due to drag
        if v > 0:  # Avoid division by zero
            ax = -(drag_force / PROJECTILE_MASS) * (vx / v)
            ay = -(drag_force / PROJECTILE_MASS) * (vy / v) - self.g
        else:
            ax = 0
            ay = -self.g

        return np.array([vx, vy, ax, ay])

    @property
    def available_methods(self) -> list[str]:
        """Get list of available solver methods"""
        return [method.value for method in SolverMethod]

    def set_method(self, method: SolverMethod) -> None:
        """Change the solver method"""
        if method not in SolverMethod:
            raise ValueError(
                f"Unknown method {method}. Available methods: {self.available_methods}"
            )
        self.method = method
