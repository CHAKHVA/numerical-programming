from typing import List, Optional, Tuple

import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


class ODESolver:
    """Class to handle different ODE solving methods"""

    def __init__(self, g: float = 9.81):
        self.g = g

    def euler_step(self, state: np.ndarray, dt: float) -> np.ndarray:
        """Euler method for one step"""
        x, y, vx, vy = state
        # In screen coordinates, positive y is downward
        new_x = x + vx * dt
        new_y = y + vy * dt
        new_vx = vx  # x-velocity remains constant
        new_vy = vy + self.g * dt  # Add gravity (positive since y is down)
        return np.array([new_x, new_y, new_vx, new_vy])

    def rk4_step(self, state: np.ndarray, dt: float) -> np.ndarray:
        """Runge-Kutta 4th order method for one step"""

        def f(s, t):
            _, _, vx, vy = s
            # Physics equations: a = dv/dt, v = dx/dt
            # In screen coordinates, positive y is downward
            return np.array(
                [
                    vx,  # dx/dt = vx
                    vy,  # dy/dt = vy
                    0,  # dvx/dt = 0 (no horizontal acceleration)
                    self.g,  # dvy/dt = g (positive since y is down)
                ]
            )

        k1 = f(state, 0)
        k2 = f(state + 0.5 * dt * k1, dt / 2)
        k3 = f(state + 0.5 * dt * k2, dt / 2)
        k4 = f(state + dt * k3, dt)

        return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def detect_ball(frame):
    """Detect blue ball in frame"""
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    lower_bound = np.array([100, 50, 50])
    upper_bound = np.array([130, 255, 255])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue

        perimeter = cv2.arcLength(cnt, True)
        circularity = (
            (4 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0
        )

        if circularity > 0.7:
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            return (int(x), int(y)), int(radius)

    return None, None


def extract_ball_positions(
    video_path: str, display: bool = False
) -> List[Tuple[int, int]]:
    """Extract ball positions from video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    positions = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        pos, radius = detect_ball(frame)

        if pos is not None:
            positions.append(pos)

            if display:
                cv2.circle(frame, pos, radius, (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Frame: {frame_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Detection", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    if display:
        cv2.destroyAllWindows()

    return positions


def reconstruct_trajectory(
    positions: List[Tuple[int, int]], fps: int = 30, method: str = "rk4"
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct full trajectory using shooting method"""
    solver = ODESolver()
    dt = 1.0 / fps

    # Initial guess for velocity using first positions
    if len(positions) < 2:
        raise ValueError("Need at least 2 positions to reconstruct trajectory")

    x0, y0 = positions[0]
    x1, y1 = positions[1]
    vx_init = (x1 - x0) / dt
    # For y velocity, account for gravity: y = y0 + v0*t + 0.5*g*t^2
    vy_init = (y1 - y0) / dt - 0.5 * solver.g * dt

    def shooting_error(v_init):
        vx0, vy0 = v_init
        state = np.array([x0, y0, vx0, vy0])

        trajectory = []
        for _ in range(len(positions)):
            trajectory.append(state[:2])
            state = (
                solver.rk4_step(state, dt)
                if method == "rk4"
                else solver.euler_step(state, dt)
            )

        error = sum(
            np.linalg.norm(np.array(pos) - np.array(traj))
            for pos, traj in zip(positions, trajectory)
        )
        return error

    # Use shooting method to find initial velocities
    result = minimize(shooting_error, [vx_init, vy_init], method="Nelder-Mead")
    vx0, vy0 = result.x

    # Generate full trajectory
    state = np.array([x0, y0, vx0, vy0])
    points = []
    velocities = []

    # Simulate for longer to get full trajectory
    for _ in range(int(2 * len(positions))):  # Simulate longer to show full path
        points.append(state[:2])
        velocities.append(state[2:])
        state = (
            solver.rk4_step(state, dt)
            if method == "rk4"
            else solver.euler_step(state, dt)
        )

        # Stop if ball hits ground (adjust Y threshold as needed)
        if state[1] > 480:  # Assuming 480 is video height
            break

    return np.array(points), np.array(velocities)


def shoot_interceptor(
    target_trajectory: np.ndarray,
    target_velocities: np.ndarray,
    shooter_pos: Tuple[float, float],
    fps: int = 30,
    method: str = "rk4",
) -> np.ndarray:
    """Simulate shooter ball to intercept target"""
    solver = ODESolver()
    dt = 1.0 / fps

    def simulate_shot(v_init):
        state = np.array([shooter_pos[0], shooter_pos[1], v_init[0], v_init[1]])
        trajectory = []

        for _ in range(len(target_trajectory)):
            trajectory.append(state[:2])
            state = (
                solver.rk4_step(state, dt)
                if method == "rk4"
                else solver.euler_step(state, dt)
            )

            if state[1] > 480:  # Ground collision check
                break

        return np.array(trajectory)

    def intercept_error(v_init):
        shot_traj = simulate_shot(v_init)
        min_dist = float("inf")

        for i in range(min(len(shot_traj), len(target_trajectory))):
            dist = np.linalg.norm(shot_traj[i] - target_trajectory[i])
            min_dist = min(min_dist, dist)

        return min_dist

    # Optimize to find intercepting trajectory
    target_speed = np.linalg.norm(target_velocities[0])  # Use same speed as target
    angle = np.arctan2(
        target_trajectory[10][1] - shooter_pos[1],
        target_trajectory[10][0] - shooter_pos[0],
    )
    v_init_guess = [target_speed * np.cos(angle), target_speed * np.sin(angle)]

    result = minimize(intercept_error, v_init_guess, method="Nelder-Mead")
    return simulate_shot(result.x)


def create_animation(
    original_trajectory: np.ndarray,
    intercept_trajectory: np.ndarray,
    method: str = "RK4",
):
    """Create matplotlib animation showing both trajectories"""
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 640)  # Adjust based on your video dimensions
    ax.set_ylim(480, 0)  # Inverted y-axis to match video coordinates
    ax.grid(True)
    ax.set_title(f"Ball Trajectory Simulation ({method})")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    # Initialize lines and points
    (original_trail,) = ax.plot([], [], "b-", alpha=0.3, label="Original Ball Trail")
    (intercept_trail,) = ax.plot([], [], "r-", alpha=0.3, label="Interceptor Trail")
    (original_point,) = ax.plot([], [], "bo", markersize=10, label="Original Ball")
    (intercept_point,) = ax.plot([], [], "ro", markersize=10, label="Interceptor")
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes)

    def init():
        """Initialize animation"""
        original_trail.set_data([], [])
        intercept_trail.set_data([], [])
        original_point.set_data([], [])
        intercept_point.set_data([], [])
        time_text.set_text("")
        ax.legend()
        return (
            original_trail,
            intercept_trail,
            original_point,
            intercept_point,
            time_text,
        )

    def animate(frame):
        """Animation function called for each frame"""
        # Update original ball position and trail
        if frame < len(original_trajectory):
            orig_trail_x = original_trajectory[: frame + 1, 0]
            orig_trail_y = original_trajectory[: frame + 1, 1]
            original_trail.set_data(orig_trail_x, orig_trail_y)
            original_point.set_data(
                [original_trajectory[frame, 0]], [original_trajectory[frame, 1]]
            )

        # Update interceptor position and trail
        if frame < len(intercept_trajectory):
            int_trail_x = intercept_trajectory[: frame + 1, 0]
            int_trail_y = intercept_trajectory[: frame + 1, 1]
            intercept_trail.set_data(int_trail_x, int_trail_y)
            intercept_point.set_data(
                [intercept_trajectory[frame, 0]], [intercept_trajectory[frame, 1]]
            )

        # Update time text
        time_text.set_text(f"Frame: {frame}")

        return (
            original_trail,
            intercept_trail,
            original_point,
            intercept_point,
            time_text,
        )

    frames = max(len(original_trajectory), len(intercept_trajectory))
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=frames, interval=50, blit=True
    )

    plt.show()


def main():
    # 1. Extract ball positions from video
    video_path = "test.mp4"
    positions = extract_ball_positions(video_path, display=True)

    # 2. Reconstruct full trajectory using both methods
    points_rk4, velocities_rk4 = reconstruct_trajectory(positions, method="rk4")
    points_euler, velocities_euler = reconstruct_trajectory(positions, method="euler")

    # 3. Choose random shooter position (adjust ranges based on your video)
    shooter_pos = (np.random.uniform(50, 200), np.random.uniform(200, 400))

    # 4. Simulate interceptor using both methods
    intercept_traj_rk4 = shoot_interceptor(
        points_rk4, velocities_rk4, shooter_pos, method="rk4"
    )
    intercept_traj_euler = shoot_interceptor(
        points_euler, velocities_euler, shooter_pos, method="euler"
    )

    # 5. Create matplotlib animations for both methods
    print("\nClose the RK4 animation window to see the Euler method animation.")
    create_animation(points_rk4, intercept_traj_rk4, method="RK4")
    create_animation(points_euler, intercept_traj_euler, method="Euler")


if __name__ == "__main__":
    main()
