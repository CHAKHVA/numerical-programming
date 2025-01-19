import numpy as np

from src.trajectory.params import TrajectoryParams


class TrajectoryCalculator:
    """Calculate projectile motion trajectory."""

    def __init__(self, params: TrajectoryParams):
        self.params = params

    def calculate_trajectory(self) -> tuple[np.ndarray, np.ndarray]:
        """Calculate trajectory points using projectile motion equations."""
        t = np.arange(0, self.params.max_time, self.params.time_step)

        x = (self.params.start_pos[0] +
             self.params.initial_velocity * np.cos(self.params.angle) * t)

        y = (self.params.start_pos[1] +
             self.params.initial_velocity * np.sin(self.params.angle) * t -
             0.5 * self.params.gravity * t ** 2)

        # Keep only points above ground and before target
        valid_points = y >= 0
        return x[valid_points], y[valid_points]

    def hits_target(self) -> bool:
        """Check if trajectory hits the target within tolerance."""
        x, y = self.calculate_trajectory()
        distances = np.sqrt((x - self.params.target_pos[0]) ** 2 +
                            (y - self.params.target_pos[1]) ** 2)
        return np.min(distances) <= self.params.tolerance

import matplotlib.pyplot as plt

def test_trajectory():
    # Test parameters
    params = TrajectoryParams(
        start_pos=(0, 0),          # Start at origin
        target_pos=(100, 50),      # Target at (100, 50)
        initial_velocity=30.0,      # Initial velocity
        angle=np.pi/4,             # 45 degrees
        gravity=9.81,
        time_step=0.01,
        max_time=10.0,
        tolerance=0.1
    )

    # Calculate trajectory
    calculator = TrajectoryCalculator(params)
    x, y = calculator.calculate_trajectory()
    hits = calculator.hits_target()

    # Plot trajectory
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', label='Trajectory')
    plt.plot(params.start_pos[0], params.start_pos[1], 'go', label='Start')
    plt.plot(params.target_pos[0], params.target_pos[1], 'ro', label='Target')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.title(f'Hits target: {hits}')
    plt.show()

if __name__ == "__main__":
    test_trajectory()