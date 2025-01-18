# Project Structure
'''
hit_ball_target/
│
├── src/
│   ├── __init__.py
│   ├── edge_detection.py
│   ├── ball_detection.py
│   ├── trajectory.py
│   └── visualization.py
│
├── tests/
│   ├── __init__.py
│   ├── test_edge_detection.py
│   ├── test_ball_detection.py
│   └── test_trajectory.py
│
└── main.py
'''


# src/ball_detection.py
import numpy as np
from typing import List, Tuple


class HoughCircleDetector:
    def __init__(self, min_radius: int = 10, max_radius: int = 50,
                 threshold: float = 0.5, step_angle: float = 1.0):
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.threshold = threshold
        self.step_angle = np.deg2rad(step_angle)

    def detect_circles(self, edges: np.ndarray) -> List[Tuple[int, int, int]]:
        """Detect circles using Hough Transform."""
        height, width = edges.shape
        angles = np.arange(0, 2 * np.pi, self.step_angle)

        # Initialize accumulator
        accumulator = {}
        for r in range(self.min_radius, self.max_radius + 1):
            accumulator[r] = np.zeros((height, width))

        # Vote in accumulator
        y_indices, x_indices = np.nonzero(edges)
        for i, j in zip(y_indices, x_indices):
            for theta in angles:
                for r in range(self.min_radius, self.max_radius + 1):
                    # Calculate circle center
                    a = int(j - r * np.cos(theta))
                    b = int(i - r * np.sin(theta))

                    if 0 <= a < width and 0 <= b < height:
                        accumulator[r][b, a] += 1

        # Find circles
        circles = []
        for r, acc in accumulator.items():
            # Find peaks in accumulator
            threshold = np.max(acc) * self.threshold
            peaks = np.argwhere(acc >= threshold)

            for center_y, center_x in peaks:
                # Check if circle already exists
                is_new = True
                for existing_y, existing_x, existing_r in circles:
                    if (abs(center_y - existing_y) <= r / 2 and
                            abs(center_x - existing_x) <= r / 2 and
                            abs(r - existing_r) <= r / 2):
                        is_new = False
                        break

                if is_new:
                    circles.append((center_y, center_x, r))

        return circles


# src/trajectory.py
import numpy as np
from typing import Tuple, List
from dataclasses import dataclass


@dataclass
class TrajectoryParams:
    initial_velocity: float
    angle: float
    start_pos: Tuple[float, float]
    target_pos: Tuple[float, float]
    gravity: float = 9.81
    time_step: float = 0.01
    max_time: float = 10.0
    tolerance: float = 0.1


class TrajectoryCalculator:
    def __init__(self, params: TrajectoryParams):
        self.params = params

    def calculate_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate trajectory points given initial conditions."""
        t = np.arange(0, self.params.max_time, self.params.time_step)

        # Calculate x and y positions
        x = (self.params.start_pos[0] +
             self.params.initial_velocity * np.cos(self.params.angle) * t)
        y = (self.params.start_pos[1] +
             self.params.initial_velocity * np.sin(self.params.angle) * t -
             0.5 * self.params.gravity * t ** 2)

        return x, y

    def hit_target(self) -> bool:
        """Check if trajectory hits target within tolerance."""
        x, y = self.calculate_trajectory()

        # Find minimum distance to target
        distances = np.sqrt((x - self.params.target_pos[0]) ** 2 +
                            (y - self.params.target_pos[1]) ** 2)
        return np.min(distances) <= self.params.tolerance


class ShootingMethod:
    def __init__(self, start_pos: Tuple[float, float], target_pos: Tuple[float, float],
                 v_range: Tuple[float, float] = (0, 100),
                 angle_range: Tuple[float, float] = (0, np.pi / 2)):
        self.start_pos = start_pos
        self.target_pos = target_pos
        self.v_range = v_range
        self.angle_range = angle_range

    def optimize(self) -> Tuple[float, float]:
        """Find optimal initial velocity and angle using shooting method."""
        v_min, v_max = self.v_range
        angle_min, angle_max = self.angle_range

        best_params = None
        min_error = float('inf')

        # Grid search with refinement
        n_iterations = 3
        n_points = 10

        for _ in range(n_iterations):
            v_values = np.linspace(v_min, v_max, n_points)
            angle_values = np.linspace(angle_min, angle_max, n_points)

            for v in v_values:
                for angle in angle_values:
                    params = TrajectoryParams(
                        initial_velocity=v,
                        angle=angle,
                        start_pos=self.start_pos,
                        target_pos=self.target_pos
                    )

                    calculator = TrajectoryCalculator(params)
                    x, y = calculator.calculate_trajectory()

                    # Calculate error as minimum distance to target
                    distances = np.sqrt((x - self.target_pos[0]) ** 2 +
                                        (y - self.target_pos[1]) ** 2)
                    error = np.min(distances)

                    if error < min_error:
                        min_error = error
                        best_params = (v, angle)

            # Refine search range around best parameters
            if best_params is not None:
                v_best, angle_best = best_params
                v_range = (v_best - (v_max - v_min) / n_points,
                           v_best + (v_max - v_min) / n_points)
                angle_range = (angle_best - (angle_max - angle_min) / n_points,
                               angle_best + (angle_max - angle_min) / n_points)

                v_min, v_max = v_range
                angle_min, angle_max = angle_range

        return best_params


# src/visualization.py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from typing import List, Tuple


class Visualizer:
    def __init__(self, image_size: Tuple[int, int]):
        self.image_size = image_size
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.setup_plot()

    def setup_plot(self):
        """Set up the plot with proper limits and labels."""
        self.ax.set_xlim(0, self.image_size[1])
        self.ax.set_ylim(0, self.image_size[0])
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        self.ax.set_title('Ball Trajectory Simulation')
        self.ax.grid(True)

    def plot_edges(self, edges: np.ndarray):
        """Plot detected edges."""
        self.ax.imshow(edges, cmap='gray')

    def plot_circles(self, circles: List[Tuple[int, int, int]]):
        """Plot detected circles."""
        for y, x, r in circles:
            circle = plt.Circle