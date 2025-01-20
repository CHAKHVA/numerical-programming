import numpy as np

from src.trajectory.params import TrajectoryParams


class TrajectoryCalculator:
    def __init__(self, params: TrajectoryParams):
        self.params = params

    def calculate_trajectory(self) -> tuple[np.ndarray, np.ndarray]:
        t = np.arange(0, self.params.max_time, self.params.time_step)

        x = (
            self.params.start_pos[0]
            + self.params.initial_velocity * np.cos(self.params.angle) * t
        )

        y = (
            self.params.start_pos[1]
            + self.params.initial_velocity * np.sin(self.params.angle) * t
            - 0.5 * self.params.gravity * t**2
        )

        # Keep only points above ground and before target
        valid_points = y >= 0
        return x[valid_points], y[valid_points]

    def hits_target(self) -> bool:
        x, y = self.calculate_trajectory()
        distances = np.sqrt(
            (x - self.params.target_pos[0]) ** 2 + (y - self.params.target_pos[1]) ** 2
        )
        return np.min(distances) <= self.params.tolerance
