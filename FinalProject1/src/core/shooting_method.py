import numpy as np

from src.constants import (
    RK4_NUM_STEPS,
    RK4_STEP_SIZE,
    SHOOTING_MAX_ITERATIONS,
    SHOOTING_TOLERANCE,
    VELOCITY_UPDATE_FACTOR,
)

from .ode_solver import ODESolver, SolverMethod


class ShootingMethod:
    """Class to handle the shooting method"""

    def __init__(self, ode_solver: ODESolver):
        self.solver = ode_solver

    def solve(
        self,
        target_x: float,
        target_y: float,
        initial_position: tuple[float, float],
        initial_guess: list[float],
        tolerance: float = SHOOTING_TOLERANCE,
        max_iter: int = SHOOTING_MAX_ITERATIONS,
        solver_method: SolverMethod = SolverMethod.RK4,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """
        Find the correct initial velocity using shooting method

        Args:
            target_x, target_y: Target coordinates
            initial_position: Starting position (x0, y0)
            initial_guess: Initial velocity guess [vx0, vy0]
            tolerance: Convergence tolerance
            max_iter: Maximum iterations
            solver_method: Numerical method to use for solving ODEs

        Returns:
            Tuple of (initial velocity, time points, trajectory)
            Returns (None, None, None) if no solution is found
        """
        self.solver.set_method(solver_method)
        v0_guess = initial_guess.copy()

        for i in range(max_iter):
            y0 = [initial_position[0], initial_position[1], v0_guess[0], v0_guess[1]]

            t, y = self.solver.solve(
                self.solver.projectile_motion, 0, y0, RK4_STEP_SIZE, RK4_NUM_STEPS
            )

            x_final = y[-1, 0]
            y_final = y[-1, 1]

            if (
                abs(x_final - target_x) < tolerance
                and abs(y_final - target_y) < tolerance
            ):
                print(
                    f"Converged after {i+1} iterations using {solver_method.value} method."
                )
                return v0_guess, t, y

            # Update velocity guess
            v0_guess[0] += (target_x - x_final) / VELOCITY_UPDATE_FACTOR
            v0_guess[1] += (target_y - y_final) / VELOCITY_UPDATE_FACTOR

        print(f"Did not converge using {solver_method.value} method.")
        return None, None, None
