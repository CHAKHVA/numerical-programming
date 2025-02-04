import numpy as np
from scipy.optimize import minimize

from src.constants import (
    FPS,
    SIMULATION_EXTENSION,
    VELOCITY_SCALE,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
)
from src.ode_solvers import ODESolver


class TrajectoryPredictor:
    def __init__(self, solver_type: str = "rk4"):
        self.solver = ODESolver()
        self.solver_type = solver_type

    def reconstruct_trajectory(
        self, positions: list[tuple[int, int]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruct full trajectory using shooting method."""
        dt = 1.0 / FPS

        if len(positions) < 2:
            raise ValueError("Need at least 2 positions to reconstruct trajectory")

        x0, y0 = positions[0]
        x1, y1 = positions[1]

        # Scale down velocities
        vx_init = (x1 - x0) / dt * VELOCITY_SCALE
        vy_init = ((y1 - y0) / dt - 0.5 * self.solver.g * dt) * VELOCITY_SCALE

        print(f"Initial velocity estimate: vx={vx_init:.2f}, vy={vy_init:.2f}")

        def shooting_error(v_init):
            vx0, vy0 = v_init
            state = np.array([x0, y0, vx0, vy0])

            trajectory = []
            for _ in range(len(positions)):
                trajectory.append(state[:2])
                state = (
                    self.solver.rk4_step(state, dt)
                    if self.solver_type == "rk4"
                    else self.solver.euler_step(state, dt)
                )

            error = sum(
                np.linalg.norm(np.array(pos) - np.array(traj))
                for pos, traj in zip(positions, trajectory)
            )
            return error

        result = minimize(shooting_error, [vx_init, vy_init], method="Nelder-Mead")
        vx0, vy0 = result.x

        print(f"Optimized velocities: vx={vx0:.2f}, vy={vy0:.2f}")

        # Generate full trajectory
        state = np.array([x0, y0, vx0, vy0])
        points = []
        velocities = []

        # Continue simulation until ball hits ground or goes off screen
        while True:
            points.append(state[:2])
            velocities.append(state[2:])
            state = (
                self.solver.rk4_step(state, dt)
                if self.solver_type == "rk4"
                else self.solver.euler_step(state, dt)
            )

            # Print every 10th point for debugging
            if len(points) % 10 == 0:
                print(
                    f"Position: ({state[0]:.1f}, {state[1]:.1f}), "
                    f"Velocity: ({state[2]:.1f}, {state[3]:.1f})"
                )

            # Stop if ball hits ground or goes off screen
            if (
                state[1] > VIDEO_HEIGHT
                or state[0] > VIDEO_WIDTH
                or state[1] < 0
                or state[0] < 0
                or len(points) > len(positions) * SIMULATION_EXTENSION
            ):
                break

        return np.array(points), np.array(velocities)

    def shoot_interceptor(
        self,
        target_trajectory: np.ndarray,
        target_velocities: np.ndarray,
        shooter_pos: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate shooter ball to intercept target."""
        dt = 1.0 / FPS

        def simulate_shot(v_init):
            state = np.array([shooter_pos[0], shooter_pos[1], v_init[0], v_init[1]])
            trajectory = []
            velocities = []

            for _ in range(len(target_trajectory)):
                trajectory.append(state[:2])
                velocities.append(state[2:])
                state = (
                    self.solver.rk4_step(state, dt)
                    if self.solver_type == "rk4"
                    else self.solver.euler_step(state, dt)
                )

                if state[1] > VIDEO_HEIGHT:
                    break

            return np.array(trajectory), np.array(velocities)

        def intercept_error(v_init):
            shot_traj, _ = simulate_shot(v_init)
            min_dist = float("inf")

            for i in range(min(len(shot_traj), len(target_trajectory))):
                dist = np.linalg.norm(shot_traj[i] - target_trajectory[i])
                min_dist = min(min_dist, dist)

            return min_dist

        target_speed = np.linalg.norm(target_velocities[0]) * VELOCITY_SCALE
        angle = np.arctan2(
            target_trajectory[10][1] - shooter_pos[1],
            target_trajectory[10][0] - shooter_pos[0],
        )
        v_init_guess = [target_speed * np.cos(angle), target_speed * np.sin(angle)]

        result = minimize(intercept_error, v_init_guess, method="Nelder-Mead")
        return simulate_shot(result.x)
