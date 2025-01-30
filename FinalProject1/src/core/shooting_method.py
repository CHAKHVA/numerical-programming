from enum import Enum

import numpy as np

from src.constants import (
    GRAVITY,
    RK4_NUM_STEPS,
    RK4_STEP_SIZE,
    SHOOTING_MAX_ITERATIONS,
    SHOOTING_TOLERANCE,
    VELOCITY_UPDATE_FACTOR,
)

from .ode_solver import ODESolver, SolverMethod


class UpdateStrategy(Enum):
    FIXED = "fixed"
    ADAPTIVE = "adaptive"


class ShootingMethod:
    """Class to handle the shooting method with multiple update strategies"""

    def __init__(self, ode_solver: ODESolver):
        self.solver = ode_solver

        # Strategy parameters
        self.learning_rate = 1.0 / VELOCITY_UPDATE_FACTOR
        self.adapt_rate = 1.1
        self.decay_rate = 0.5

    def _estimate_required_velocity(
        self, target_x: float, target_y: float, initial_position: tuple[float, float]
    ) -> tuple[list[float], float]:
        """Estimate required velocity range based on target position"""
        dx = target_x - initial_position[0]
        dy = target_y - initial_position[1]
        distance = np.sqrt(dx**2 + dy**2)

        theta = np.arctan2(dy, dx)
        v_min = (
            np.sqrt(distance * GRAVITY / np.sin(2 * theta)) if theta > 0 else distance
        )

        vx_guess = v_min * np.cos(theta)
        vy_guess = v_min * np.sin(theta)

        return [vx_guess, vy_guess], distance

    def _fixed_update(
        self, v0_guess: list[float], x_error: float, y_error: float
    ) -> list[float]:
        """Fixed learning rate update"""
        v0_guess[0] += x_error * self.learning_rate
        v0_guess[1] += y_error * self.learning_rate
        return v0_guess

    def _adaptive_update(
        self,
        v0_guess: list[float],
        x_error: float,
        y_error: float,
        prev_error: float | None,
    ) -> tuple[list[float], float]:
        """Adaptive learning rate update"""
        current_error = np.sqrt(x_error**2 + y_error**2)

        # Adjust learning rate based on error change
        if prev_error is not None:
            if current_error < prev_error:
                self.learning_rate *= self.adapt_rate
            else:
                self.learning_rate *= self.decay_rate

        # Apply update
        v0_guess[0] += x_error * self.learning_rate
        v0_guess[1] += y_error * self.learning_rate

        return v0_guess, current_error

    def solve(
        self,
        target_x: float,
        target_y: float,
        initial_position: tuple[float, float],
        initial_guess: list[float] | None = None,
        tolerance: float = SHOOTING_TOLERANCE,
        max_iter: int = SHOOTING_MAX_ITERATIONS,
        solver_method: SolverMethod = SolverMethod.RK4,
        update_strategy: UpdateStrategy = UpdateStrategy.ADAPTIVE,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """
        Find the correct initial velocity using shooting method with specified update strategy

        Args:
            target_x, target_y: Target coordinates
            initial_position: Starting position (x0, y0)
            initial_guess: Initial velocity guess [vx0, vy0] (optional)
            tolerance: Convergence tolerance
            max_iter: Maximum iterations
            solver_method: Numerical method to use for solving ODEs
            update_strategy: Strategy for updating velocity guess

        Returns:
            Tuple of (initial velocity, time points, trajectory)
            Returns (None, None, None) if no solution is found
        """
        self.solver.set_method(solver_method)

        # Get initial velocity estimate if not provided
        if initial_guess is None:
            initial_guess, _ = self._estimate_required_velocity(
                target_x, target_y, initial_position
            )

        v0_guess = initial_guess.copy()
        best_error = float("inf")
        best_solution = None

        # Strategy-specific variables
        prev_error = None

        print(f"Using {update_strategy.value} update strategy")

        for i in range(max_iter):
            # Solve ODE
            y0 = [initial_position[0], initial_position[1], v0_guess[0], v0_guess[1]]
            t, y = self.solver.solve(
                self.solver.projectile_motion, 0, y0, RK4_STEP_SIZE, RK4_NUM_STEPS
            )

            # Calculate errors
            x_final = y[-1, 0]
            y_final = y[-1, 1]
            x_error = target_x - x_final
            y_error = target_y - y_final

            current_error = np.sqrt(x_error**2 + y_error**2)

            # Track best solution
            if current_error < best_error:
                best_error = current_error
                best_solution = (v0_guess.copy(), t, y)

                if current_error < tolerance:
                    print(
                        f"Converged after {i+1} iterations using {solver_method.value} method "
                        f"with {update_strategy.value} strategy. Error: {current_error:.6f}"
                    )
                    return v0_guess, t, y

            # Update velocity based on selected strategy
            if update_strategy == UpdateStrategy.FIXED:
                v0_guess = self._fixed_update(v0_guess, x_error, y_error)

            elif update_strategy == UpdateStrategy.ADAPTIVE:
                v0_guess, prev_error = self._adaptive_update(
                    v0_guess, x_error, y_error, prev_error
                )

            # Add small random perturbation if stuck
            if i > max_iter // 2 and current_error > best_error * 1.5:
                v0_guess[0] += np.random.uniform(-0.5, 0.5)
                v0_guess[1] += np.random.uniform(0, 0.5)

        print(
            f"Did not converge using {solver_method.value} method with {update_strategy.value} "
            f"strategy. Best error: {best_error:.6f}"
        )

        # Return best found solution if error is reasonable
        if best_error < tolerance * 10 and best_solution is not None:
            return best_solution

        return None, None, None
