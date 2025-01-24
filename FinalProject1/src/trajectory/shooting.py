import numpy as np

from src.trajectory.calculator import TrajectoryCalculator
from src.trajectory.params import TrajectoryParams


class ShootingMethod:
    def __init__(self, start_pos: tuple[float, float], target_pos: tuple[float, float]):
        self.start_pos = start_pos
        self.target_pos = target_pos

    def find_parameters(self) -> tuple[float, float]:
        # Initial guesses based on distance and angle
        dx = self.target_pos[0] - self.start_pos[0]
        dy = self.target_pos[1] - self.start_pos[1]
        distance = np.sqrt(dx**2 + dy**2)

        # Grid search for initial parameters
        best_error = float("inf")
        best_params = None

        # Search ranges
        velocities = np.linspace(
            np.sqrt(9.81 * distance) * 0.5, np.sqrt(9.81 * distance) * 1.5, 50
        )
        angles = np.linspace(0, np.pi, 50)

        for v0 in velocities:
            for theta in angles:
                params = TrajectoryParams(
                    start_pos=self.start_pos,
                    target_pos=self.target_pos,
                    initial_velocity=v0,
                    angle=theta,
                )

                calculator = TrajectoryCalculator(params)
                x, y = calculator.calculate_trajectory()
                distances = np.sqrt(
                    (x - self.target_pos[0]) ** 2 + (y - self.target_pos[1]) ** 2
                )
                error = np.min(distances)

                if error < best_error:
                    best_error = error
                    best_params = (v0, theta)

                if error < params.tolerance:
                    return v0, theta

        return best_params
