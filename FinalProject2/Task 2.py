import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.signal import savgol_filter

# Constants
g = 9.81  # Gravitational acceleration (m/s^2)
dt = 0.01  # Time step for numerical methods (s)

# Blue color range in HSV
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])


def extract_positions(video_path, debug=False):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    positions = []
    times = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise ValueError("Unable to retrieve FPS from video.")

    frame_time_interval = 1 / fps
    frame_idx = 0

    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    if frame_width == 0:
        raise ValueError("Unable to retrieve frame width from video.")

    pixels_to_meters = 10 / frame_width

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask for blue detection
        mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)

        if debug:
            cv2.imshow("Mask", mask)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            if radius > 2:
                height = frame.shape[0]
                y_cartesian = height - y
                positions.append((x, y_cartesian))
                times.append(frame_idx * frame_time_interval)

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    if len(positions) < 2:
        raise ValueError(
            "Insufficient ball positions detected. Adjust HSV thresholds or check the video."
        )

    return np.array(times), np.array(positions) * pixels_to_meters


def smooth_trajectory(positions):
    """Smooths and interpolates the trajectory data."""
    x = positions[:, 0]
    y = positions[:, 1]

    # Interpolate and smooth
    t = np.arange(len(x))
    interp_x = interp1d(t, x, kind="cubic", fill_value="extrapolate")(t)
    interp_y = interp1d(t, y, kind="cubic", fill_value="extrapolate")(t)

    # Apply Savitzky-Golay filter
    smooth_x = savgol_filter(interp_x, 11, 3)
    smooth_y = savgol_filter(interp_y, 11, 3)

    return smooth_x, smooth_y


def extrapolate_trajectory(times, x, y, vx, vy):
    """Extrapolates the trajectory using equations of motion."""
    t_extrapolate = np.linspace(
        times[-1], times[-1] + 2, num=50
    )  # Extend for 2 seconds
    x_extrapolate = x[-1] + vx * (t_extrapolate - times[-1])
    y_extrapolate = (
        y[-1]
        + vy * (t_extrapolate - times[-1])
        - 0.5 * g * (t_extrapolate - times[-1]) ** 2
    )
    return t_extrapolate, x_extrapolate, y_extrapolate


def ball_motion(t, state):
    """ODE for projectile motion under gravity."""
    x, y, vx, vy = state
    return [vx, vy, 0, -g]


def rk4_method(f, t_span, y0, dt, target_trajectory=None, collision_threshold=0.1):
    """Runge-Kutta 4th order method for solving ODEs with early stop for collision."""
    t_values = np.arange(t_span[0], t_span[1] + dt, dt)
    y_values = np.zeros((len(t_values), len(y0)))
    y_values[0] = np.array(y0)

    for i in range(1, len(t_values)):
        t = t_values[i - 1]
        y = y_values[i - 1]

        k1 = np.array(f(t, y))
        k2 = np.array(f(t + dt / 2, y + dt * k1 / 2))
        k3 = np.array(f(t + dt / 2, y + dt * k2 / 2))
        k4 = np.array(f(t + dt, y + dt * k3))

        y_next = y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        y_values[i] = y_next

        # Early stop if collision is detected
        if target_trajectory is not None:
            target_position = target_trajectory[min(i, len(target_trajectory) - 1)]
            distance = np.linalg.norm(y_next[:2] - target_position[:2])  # Compare x, y
            if distance <= collision_threshold:
                print(f"Collision detected at time {t:.2f} s, position: {y_next[:2]}")
                return t_values[: i + 1], y_values[
                    : i + 1
                ]  # Return truncated trajectory

    return t_values, y_values


def shooting_method_dynamic(target_trajectory, start_position, dt):
    """Uses the shooting method to calculate the initial velocity and angle for interception."""

    def objective(params):
        v0, angle = params
        vx = v0 * np.cos(angle)
        vy = v0 * np.sin(angle)
        y0 = [start_position[0], start_position[1], vx, vy]

        _, second_trajectory = rk4_method(
            ball_motion, (0, len(target_trajectory) * dt), y0, dt
        )
        min_length = min(len(target_trajectory), len(second_trajectory))
        target_trimmed = target_trajectory[:min_length]
        second_trimmed = second_trajectory[:min_length]
        distances = np.linalg.norm(
            target_trimmed[:, :2] - second_trimmed[:, :2], axis=1
        )
        return np.sum(distances)

    avg_vx = np.mean(np.gradient(target_trajectory[:, 0], dt))
    avg_vy = np.mean(np.gradient(target_trajectory[:, 1], dt))
    initial_v0 = np.sqrt(avg_vx**2 + avg_vy**2)
    initial_angle = np.arctan2(avg_vy, avg_vx)
    initial_guess = [initial_v0, initial_angle]

    bounds = [(1, 100), (0, np.pi / 2)]
    result = minimize(
        objective,
        initial_guess,
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-6},
    )
    if not result.success:
        raise ValueError("Shooting method failed to converge.")

    v0, angle = result.x
    vx = v0 * np.cos(angle)
    vy = v0 * np.sin(angle)
    return [start_position[0], start_position[1], vx, vy]


def animate_trajectories_collision(
    original_trajectory, second_trajectory, collision_threshold=0.1
):
    """
    Animates the original and second ball trajectories with fixed zoom levels.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set fixed axis limits
    x_min, x_max = 0, 20  # Example fixed horizontal range
    y_min, y_max = 0, 10  # Example fixed vertical range
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")

    # Plot the full trajectories
    ax.plot(
        original_trajectory[:, 0],
        original_trajectory[:, 1],
        label="Original Ball",
        color="blue",
        linestyle="--",
        alpha=0.5,
    )
    ax.plot(
        second_trajectory[:, 0],
        second_trajectory[:, 1],
        label="Second Ball",
        color="orange",
        linestyle="--",
        alpha=0.5,
    )

    # Animated lines and markers
    (original_line,) = ax.plot(
        [], [], label="Original Ball (Animated)", color="red", linewidth=2
    )
    (second_line,) = ax.plot(
        [], [], label="Second Ball (Animated)", color="orange", linewidth=2
    )
    (collision_marker,) = ax.plot([], [], "ro", label="Collision Point", markersize=8)

    collision_detected = False

    def update(frame):
        nonlocal collision_detected
        if frame < len(original_trajectory) and frame < len(second_trajectory):
            original_line.set_data(
                original_trajectory[:frame, 0], original_trajectory[:frame, 1]
            )
            second_line.set_data(
                second_trajectory[:frame, 0], second_trajectory[:frame, 1]
            )

            # Check for collision
            distance = np.linalg.norm(
                original_trajectory[frame] - second_trajectory[frame]
            )
            if distance <= collision_threshold and not collision_detected:
                collision_detected = True
                collision_marker.set_data(
                    [original_trajectory[frame, 0]], [original_trajectory[frame, 1]]
                )

                print(
                    f"Collision occurred at frame {frame}, position: {original_trajectory[frame]}"
                )
        return original_line, second_line, collision_marker

    total_frames = min(len(original_trajectory), len(second_trajectory))
    anim = FuncAnimation(fig, update, frames=total_frames, interval=50, blit=True)
    plt.show()


def main(video_path, debug=False):
    """Main function to execute the trajectory analysis and simulation."""
    # Step 1: Extract positions from video
    times, positions = extract_positions(video_path, debug=debug)

    # Step 2: Smooth trajectory
    x_smooth, y_smooth = smooth_trajectory(positions)

    # Step 3: Compute initial velocities
    vx0 = np.gradient(x_smooth, times)
    vy0 = np.gradient(y_smooth, times)
    initial_conditions = [x_smooth[0], y_smooth[0], vx0[0], vy0[0]]

    # Step 4: Extrapolate trajectory
    t_span = (0, times[-1])
    t_extrapolate, x_extrapolate, y_extrapolate = extrapolate_trajectory(
        times, x_smooth, y_smooth, vx0[-1], vy0[-1]
    )

    # Step 5: Simulate the second ball trajectory using RK4
    start_position = [0, 0]
    second_initial_conditions = shooting_method_dynamic(
        np.column_stack((x_smooth, y_smooth)), start_position, dt
    )

    _, rk4_trajectory = rk4_method(ball_motion, t_span, second_initial_conditions, dt)

    # Combine trajectories for animation
    original_trajectory = np.column_stack((x_smooth, y_smooth))
    second_trajectory = rk4_trajectory[:, :2]

    # Step 6: Animate the trajectories and detect collision

    animate_trajectories_collision(
        original_trajectory, second_trajectory, collision_threshold=0.1
    )


# Run the program with your video file path
video_path = "test.mp4"  # Replace with your video path
main(video_path, debug=True)
