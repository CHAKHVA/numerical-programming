import matplotlib.pyplot as plt
from typing import List, Tuple
from .params import TrajectoryParams
from .calculator import TrajectoryCalculator
from .shooting import ShootingMethod


class TrajectoryVisualizer:
    def __init__(self, circles: List[Tuple[int, int, int]], shooting_point: Tuple[float, float]):
        self.circles = circles
        self.shooting_point = shooting_point
        self.fig, self.ax = plt.subplots(figsize=(10, 8))

    def animate(self):
        """Animate trajectories from shooting point to each ball"""
        # Setup plot
        self.ax.set_aspect('equal')
        self.ax.grid(True)

        # Plot shooting point and circles
        self.ax.plot(self.shooting_point[0], self.shooting_point[1], 'bo', markersize=10)
        for x, y, r in self.circles:
            circle = plt.Circle((x, y), r, fill=False, color='red')
            self.ax.add_artist(circle)

        # Set plot limits
        margin = 50
        all_x = [x for x, _, _ in self.circles] + [self.shooting_point[0]]
        all_y = [y for _, y, _ in self.circles] + [self.shooting_point[1]]
        self.ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        self.ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

        # Draw trajectory to each ball
        for target in self.circles:
            target_pos = (target[0], target[1])

            # Calculate trajectory
            shooter = ShootingMethod(self.shooting_point, target_pos)
            v0, theta = shooter.find_parameters()
            params = TrajectoryParams(
                start_pos=self.shooting_point,
                target_pos=target_pos,
                initial_velocity=v0,
                angle=theta
            )
            calculator = TrajectoryCalculator(params)
            x, y = calculator.calculate_trajectory()

            # Draw trajectory point by point
            line, = self.ax.plot([], [], 'b-')
            for i in range(len(x)):
                line.set_data(x[:i + 1], y[:i + 1])
                plt.pause(0.001)

            plt.pause(0.5)  # Pause between trajectories

        plt.show()