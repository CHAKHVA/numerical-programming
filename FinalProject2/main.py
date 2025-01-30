import random
from enum import Enum

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


class NumericalMethod(Enum):
    EULER = "euler"
    RK4 = "rk4"
    RK2 = "rk2"


class BallTracker:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.observed_points = []

    def _detect_ball(self, frame: np.ndarray) -> tuple[float, float] | None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=50,
        )

        if circles is not None:
            circle = circles[0][0]
            return float(circle[0]), float(circle[1])
        return None

    def extract_trajectory(self) -> list[tuple[float, float, float]]:
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {self.video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            points = []
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                pos = self._detect_ball(frame)
                if pos is not None:
                    time = frame_count / fps
                    points.append((time, pos[0], pos[1]))
                frame_count += 1

            cap.release()

            if not points:
                raise ValueError("No ball detected in video")

            return points

        except Exception as e:
            print(f"Error processing video: {e}")
            raise


class TrajectoryReconstructor:
    def __init__(self, g: float = 9.81):
        self.g = g

    def reconstruct_initial_conditions(
        self, points: list[tuple[float, float, float]]
    ) -> tuple[float, float, float, float]:
        times = np.array([p[0] for p in points])
        x_pos = np.array([p[1] for p in points])
        y_pos = np.array([p[2] for p in points])

        # Fit x = x0 + vx*t
        x_coeffs = np.polyfit(times, x_pos, 1)
        vx0, x0 = x_coeffs[0], x_coeffs[1]

        # Fit y = y0 + vy*t - 0.5*g*t^2
        transformed_y = y_pos + 0.5 * self.g * times**2
        y_coeffs = np.polyfit(times, transformed_y, 1)
        vy0, y0 = y_coeffs[0], y_coeffs[1]

        return x0, y0, vx0, vy0

    def get_position_at_time(
        self, t: float, initial_conditions: tuple[float, float, float, float]
    ) -> tuple[float, float]:
        x0, y0, vx0, vy0 = initial_conditions
        x = x0 + vx0 * t
        y = y0 + vy0 * t - 0.5 * self.g * t**2
        return x, y


class Interceptor:
    def __init__(self, g: float = 9.81):
        self.g = g

    def calculate_intercept(
        self,
        target_conditions: tuple[float, float, float, float],
        launch_pos: tuple[float, float],
        method: NumericalMethod,
    ) -> tuple[float, float] | None:
        best_v = None
        min_dist = float("inf")

        # Get target trajectory info
        x0, y0, vx0, vy0 = target_conditions
        launch_x, launch_y = launch_pos

        # Calculate target-to-launcher vector
        dx = x0 - launch_x
        dy = y0 - launch_y
        distance = np.sqrt(dx**2 + dy**2)

        # Estimate required velocity ranges based on distance and target motion
        max_vx = max(abs(vx0) * 3, distance / 2)
        max_vy = max(abs(vy0) * 2, np.sqrt(2 * 9.81 * (y0 - launch_y)))

        # Create velocity ranges with smaller steps for more precision
        vx_range = np.arange(-max_vx, max_vx, max_vx / 20)
        vy_range = np.arange(max_vy / 2, max_vy * 1.5, max_vy / 20)

        print(
            f"Searching velocities: vx={-max_vx:.1f} to {max_vx:.1f}, vy={max_vy/2:.1f} to {max_vy*1.5:.1f}"
        )

        for vx in vx_range:
            for vy in vy_range:
                t_hit = self._find_collision_time(
                    launch_pos, (vx, vy), target_conditions, method, dt=0.01, max_t=3.0
                )

                if t_hit is not None:
                    dist = self._calculate_miss_distance(
                        launch_pos, (vx, vy), target_conditions, t_hit
                    )

                    if dist < min_dist:
                        min_dist = dist
                        best_v = (vx, vy)

                        if dist < 20:
                            print(f"Found solution with distance {dist:.1f}")
                            return vx, vy

        if best_v is not None:
            print(f"Best miss distance found: {min_dist:.1f}")

        return best_v if min_dist < 50 else None

    def _find_collision_time(
        self,
        launch_pos: tuple[float, float],
        launch_v: tuple[float, float],
        target_conditions: tuple[float, float, float, float],
        method: NumericalMethod,
        dt: float = 0.01,
        max_t: float = 3.0,
    ) -> float | None:
        times = np.arange(0, max_t, dt)
        for t in times:
            d = self._calculate_miss_distance(
                launch_pos, launch_v, target_conditions, t
            )
            if d < 15:  # More lenient collision detection
                return t
        return None

    def _calculate_miss_distance(
        self,
        launch_pos: tuple[float, float],
        launch_v: tuple[float, float],
        target_conditions: tuple[float, float, float, float],
        t: float,
    ) -> float:
        # Interceptor position
        x_i = launch_pos[0] + launch_v[0] * t
        y_i = launch_pos[1] + launch_v[1] * t - 0.5 * self.g * t**2

        # Target position
        x0, y0, vx0, vy0 = target_conditions
        x_t = x0 + vx0 * t
        y_t = y0 + vy0 * t - 0.5 * self.g * t**2

        return np.sqrt((x_i - x_t) ** 2 + (y_i - y_t) ** 2)


class Animator:
    def __init__(self, width: int = 800, height: int = 600):
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.width = width
        self.height = height

    def setup(self):
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_xlabel("X Position")
        self.ax.set_ylabel("Y Position")
        self.ax.grid(True)

    def animate_interception(
        self,
        target_conditions: tuple[float, float, float, float],
        launch_pos: tuple[float, float],
        launch_v: tuple[float, float],
        method: NumericalMethod,
    ):
        self.setup()

        # Original ball's trajectory
        t = np.linspace(0, 5, 100)
        x0, y0, vx0, vy0 = target_conditions
        x_target = x0 + vx0 * t
        y_target = y0 + vy0 * t - 0.5 * 9.81 * t**2

        # Plot predicted trajectory
        self.ax.plot(x_target, y_target, "r--", alpha=0.3, label="Predicted Trajectory")

        def update(frame):
            self.ax.cla()
            self.setup()

            current_t = frame / 20

            # Target ball position
            x_t = x0 + vx0 * current_t
            y_t = y0 + vy0 * current_t - 0.5 * 9.81 * current_t**2

            # Interceptor position
            x_i = launch_pos[0] + launch_v[0] * current_t
            y_i = launch_pos[1] + launch_v[1] * current_t - 0.5 * 9.81 * current_t**2

            # Plot trajectories and current positions
            self.ax.plot(
                x_target, y_target, "r--", alpha=0.3, label="Predicted Trajectory"
            )
            self.ax.plot(x_t, y_t, "ro", label="Target Ball")
            self.ax.plot(x_i, y_i, "bo", label="Interceptor")
            self.ax.plot(launch_pos[0], launch_pos[1], "k^", label="Launch Position")

            self.ax.legend()

        anim = FuncAnimation(self.fig, update, frames=100, interval=50, repeat=False)

        plt.show()


def main():
    try:
        # 1. Track ball from video and reconstruct trajectory
        tracker = BallTracker("videos/test4.mp4")
        points = tracker.extract_trajectory()

        # 2. Reconstruct complete trajectory
        reconstructor = TrajectoryReconstructor()
        initial_conditions = reconstructor.reconstruct_initial_conditions(points)

        # Get target's initial position and velocity
        x0, y0, vx0, vy0 = initial_conditions
        print(f"Target initial position: ({x0:.1f}, {y0:.1f})")
        print(f"Target initial velocity: ({vx0:.1f}, {vy0:.1f})")

        # 3. Choose better launch position based on target trajectory
        if vx0 > 0:  # Target moving right
            launch_x = max(50, x0 - random.uniform(50, 150))
        else:  # Target moving left
            launch_x = min(750, x0 + random.uniform(150, 300))

        # Adjust launch height based on target height
        launch_y = min(y0 - 100, 300)  # Launch from lower than target
        launch_pos = (launch_x, launch_y)
        print(f"Launch position: ({launch_x:.1f}, {launch_y:.1f})")

        # 4. Calculate interception
        interceptor = Interceptor()
        launch_v = None
        max_attempts = 5

        while launch_v is None and max_attempts > 0:
            launch_v = interceptor.calculate_intercept(
                initial_conditions, launch_pos, NumericalMethod.RK4
            )

            if launch_v is None:
                print(f"Retrying... {max_attempts} attempts left")
                # Adjust position based on target direction
                if vx0 > 0:
                    dx = random.uniform(-70, -20)
                else:
                    dx = random.uniform(20, 70)

                dy = random.uniform(-30, 30)
                launch_x = launch_pos[0] + dx
                launch_y = min(300, max(50, launch_pos[1] + dy))
                launch_pos = (launch_x, launch_y)
                max_attempts -= 1

        if launch_v is None:
            print("Could not find interception solution after multiple attempts")
            return

        print(
            f"Found solution: initial velocity = ({launch_v[0]:.1f}, {launch_v[1]:.1f})"
        )

        # 5. Animate the result
        animator = Animator()
        animator.animate_interception(
            initial_conditions, launch_pos, launch_v, NumericalMethod.RK4
        )

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
