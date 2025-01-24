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

        # Get image dimensions for y-axis inversion
        self.max_y = (
            max(p[1] for p in [shooting_point] + [(c[0], c[1]) for c in circles]) + 50
        )

        self.scale = 0.01

    def _convert_to_physics_coords(self, x: float, y: float) -> tuple[float, float]:
        """Convert image coordinates to physics coordinates"""
        return x * self.scale, (self.max_y - y) * self.scale

    def _convert_to_image_coords(self, x: float, y: float) -> tuple[float, float]:
        """Convert physics coordinates back to image coordinates"""
        return x / self.scale, self.max_y - (y / self.scale)

    def _hits_circle(self, x: float, y: float, circle: tuple[int, int, int]) -> bool:
        cx, cy, r = circle
        distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        return distance <= r

    def animate(self):
        self.ax.set_aspect("equal")
        self.ax.grid(True)

        # Convert shooting point to physics coordinates
        phys_shooting_x, phys_shooting_y = self._convert_to_physics_coords(
            *self.shooting_point
        )

        # Plot shooting point
        self.ax.plot(
            self.shooting_point[0],
            self.shooting_point[1],
            "bo",
            markersize=10,
            label="Shooting Point",
        )

        # Draw circles
        for x, y, r in self.circles:
            circle = plt.Circle((x, y), r, fill=False, color="red")
            self.ax.add_artist(circle)
            self.ax.plot(x, y, "rx", markersize=5)

        # Set plot limits with y-axis inverted and extra vertical space
        margin_x = 50
        margin_y = 200
        all_x = [x for x, _, _ in self.circles] + [self.shooting_point[0]]
        all_y = [y for _, y, _ in self.circles] + [self.shooting_point[1]]
        self.ax.set_xlim(min(all_x) - margin_x, max(all_x) + margin_x)
        self.ax.set_ylim(max(all_y) + margin_y, min(all_y) - margin_y)

        # Draw trajectory to each ball
        for i, target in enumerate(self.circles, 1):
            # Convert target position to physics coordinates
            phys_target_x, phys_target_y = self._convert_to_physics_coords(
                target[0], target[1]
            )
            phys_target_radius = target[2] * self.scale

            # Calculate trajectory
            shooter = ShootingMethod(
                (phys_shooting_x, phys_shooting_y), (phys_target_x, phys_target_y)
            )
            v0, theta = shooter.find_parameters()

            params = TrajectoryParams(
                start_pos=(phys_shooting_x, phys_shooting_y),
                target_pos=(phys_target_x, phys_target_y),
                initial_velocity=v0,
                angle=theta,
                gravity=9.81,
                time_step=0.01,
                max_time=10.0,
                tolerance=phys_target_radius,
            )

            calculator = TrajectoryCalculator(params)
            phys_x, phys_y = calculator.calculate_trajectory()

            # Convert trajectory points back to image coordinates
            x = [px / self.scale for px in phys_x]
            y = [self.max_y - (py / self.scale) for py in phys_y]

            # Draw trajectory
            (line,) = self.ax.plot(
                [],
                [],
                "b:",
                label=f"Shot {i}",
                alpha=0.7,
                linewidth=1.5,
            )

            step_size = max(1, len(x) // 50)
            for j in range(0, len(x), step_size):
                line.set_data(x[: j + 1], y[: j + 1])
                plt.pause(0.01)

                # Check if trajectory hits the target
                if self._hits_circle(x[j], y[j], target):
                    line.set_data(x[: j + 1], y[: j + 1])
                    plt.pause(0.2)
                    break

        self.ax.legend()
        plt.show()
