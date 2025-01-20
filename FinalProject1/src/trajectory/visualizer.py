import matplotlib.pyplot as plt
import numpy as np

from .calculator import TrajectoryCalculator
from .params import TrajectoryParams
from .shooting import ShootingMethod


class TrajectoryVisualizer:
    def __init__(
        self, circles: list[tuple[int, int, int]], shooting_point: tuple[float, float]
    ):
        self.circles = circles
        self.shooting_point = shooting_point
        self.fig, self.ax = plt.subplots(figsize=(10, 8))

    def _hits_circle(self, x: float, y: float, circle: tuple[int, int, int]) -> bool:
        cx, cy, r = circle
        distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        return distance <= r

    def animate(self):
        # Setup plot
        self.ax.set_aspect("equal")
        self.ax.grid(True)

        # Plot shooting point and circles
        self.ax.plot(
            self.shooting_point[0],
            self.shooting_point[1],
            "bo",
            markersize=10,
            label="Shooting Point",
        )
        for x, y, r in self.circles:
            circle = plt.Circle((x, y), r, fill=False, color="red")
            self.ax.add_artist(circle)

        # Set plot limits
        margin = 50
        all_x = [x for x, _, _ in self.circles] + [self.shooting_point[0]]
        all_y = [y for _, y, _ in self.circles] + [self.shooting_point[1]]
        self.ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        self.ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

        # Draw trajectory to each ball
        for i, target in enumerate(self.circles, 1):
            target_pos = (target[0], target[1])

            # Calculate trajectory
            shooter = ShootingMethod(self.shooting_point, target_pos)
            v0, theta = shooter.find_parameters()
            params = TrajectoryParams(
                start_pos=self.shooting_point,
                target_pos=target_pos,
                initial_velocity=v0,
                angle=theta,
            )
            calculator = TrajectoryCalculator(params)
            x, y = calculator.calculate_trajectory()

            # Draw trajectory point by point
            (line,) = self.ax.plot([], [], "b-", label=f"Shot {i}")

            # Determine animation step size based on trajectory length
            step_size = max(1, len(x) // 50)

            for i in range(0, len(x), step_size):
                line.set_data(x[: i + 1], y[: i + 1])
                plt.pause(0.01)

                # Check if trajectory hits the target
                if self._hits_circle(x[i], y[i], target):
                    # Complete the line until the hit point
                    line.set_data(x[: i + 1], y[: i + 1])
                    plt.pause(0.2)
                    break

        self.ax.legend()
        plt.show()
