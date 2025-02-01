from typing import List, Optional, Tuple

import cv2
import numpy as np


class ODESolver:
    """Class to handle different ODE solving methods"""

    def __init__(self, g: float = 9.81):
        self.g = g

    def euler_step(self, state: np.ndarray, dt: float) -> np.ndarray:
        """Euler method for one step"""
        x, y, vx, vy = state
        new_x = x + vx * dt
        new_y = y + vy * dt
        new_vx = vx
        new_vy = vy + self.g * dt
        return np.array([new_x, new_y, new_vx, new_vy])

    def rk4_step(self, state: np.ndarray, dt: float) -> np.ndarray:
        """Runge-Kutta 4th order method for one step"""

        def f(s):
            _, _, vx, vy = s
            return np.array([vx, vy, 0, self.g])

        k1 = f(state)
        k2 = f(state + 0.5 * dt * k1)
        k3 = f(state + 0.5 * dt * k2)
        k4 = f(state + dt * k3)

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
    vy_init = (y1 - y0) / dt

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
    from scipy.optimize import minimize

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
    from scipy.optimize import minimize

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
    output_path: str = "simulation.mp4",
):
    """Create animation showing both trajectories"""
    height, width = 480, 640  # Adjust to your video dimensions
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

    for i in range(max(len(original_trajectory), len(intercept_trajectory))):
        frame = np.full((height, width, 3), 255, dtype=np.uint8)

        # Draw original trajectory
        if i < len(original_trajectory):
            pos = original_trajectory[i].astype(int)
            cv2.circle(frame, tuple(pos), 10, (255, 0, 0), -1)  # Original ball in blue

            # Draw trail
            for j in range(max(0, i - 10), i):
                prev_pos = original_trajectory[j].astype(int)
                cv2.circle(frame, tuple(prev_pos), 2, (255, 0, 0), -1)

        # Draw interceptor trajectory
        if i < len(intercept_trajectory):
            pos = intercept_trajectory[i].astype(int)
            cv2.circle(frame, tuple(pos), 10, (0, 0, 255), -1)  # Interceptor in red

            # Draw trail
            for j in range(max(0, i - 10), i):
                prev_pos = intercept_trajectory[j].astype(int)
                cv2.circle(frame, tuple(prev_pos), 2, (0, 0, 255), -1)

        # Add frame counter
        cv2.putText(
            frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
        )

        out.write(frame)

    out.release()


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

    # 5. Create animations for both methods
    create_animation(points_rk4, intercept_traj_rk4, "simulation_rk4.mp4")
    create_animation(points_euler, intercept_traj_euler, "simulation_euler.mp4")

    print(f"Shooter position: {shooter_pos}")
    print("Animations saved as 'simulation_rk4.mp4' and 'simulation_euler.mp4'")


if __name__ == "__main__":
    main()
