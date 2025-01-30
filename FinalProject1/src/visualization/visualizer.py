import matplotlib.pyplot as plt
import numpy as np

from src.constants import (
    FIGURE_SIZE,
    GRID_ALPHA,
    TARGET_CIRCLE_ALPHA,
    TARGET_MARKER_SIZE,
    TARGET_PAUSE,
    TRAJECTORY_ANIMATION_PAUSE,
    TRAJECTORY_LINE_WIDTH,
)
from src.models.shape import Shape


class Visualizer:
    """Class to handle visualization and animation"""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.fig = None

    def setup_plot(self) -> None:
        """Initialize the plot"""
        self.fig = plt.figure(figsize=FIGURE_SIZE)
        plt.xlim(0, self.width)
        plt.ylim(0, self.height)
        plt.xlabel("Horizontal Distance (pixels)")
        plt.ylabel("Vertical Distance (pixels)")
        plt.title("Projectile Motion to Hit Targets")
        plt.grid(True, linestyle="--", alpha=GRID_ALPHA)
        plt.gca().set_aspect("equal")

    def plot_target(self, shape: Shape) -> None:
        """Plot a single target"""
        target_x, target_y = shape.position
        plt.scatter(target_x, target_y, color="red", s=TARGET_MARKER_SIZE)

        radius = shape.radius[0] if isinstance(shape.radius, tuple) else shape.radius
        circle = plt.Circle(
            (target_x, target_y),
            radius,
            fill=False,
            color="red",
            linestyle="--",
            alpha=TARGET_CIRCLE_ALPHA,
        )
        plt.gca().add_patch(circle)

    def animate_trajectory(
        self, trajectory: np.ndarray, target_x: float, target_y: float
    ) -> None:
        """Animate trajectory drawing"""
        num_points = len(trajectory)
        for i in range(1, num_points):
            plt.plot(
                trajectory[i - 1 : i + 1, 0],
                trajectory[i - 1 : i + 1, 1],
                "b:",
                linewidth=TRAJECTORY_LINE_WIDTH,
            )
            plt.pause(TRAJECTORY_ANIMATION_PAUSE)

    def add_trajectory_to_legend(self, shape: Shape, index: int) -> None:
        """Add trajectory to legend (without showing it)"""
        plt.plot([], [], "b:", label=f"Target {index}: {shape}")

    def pause_between_targets(self) -> None:
        """Pause between target animations"""
        plt.pause(TARGET_PAUSE)

    def show_legend(self) -> None:
        """Display the legend"""
        plt.legend()

    def show(self) -> None:
        """Display the plot"""
        plt.show()
