import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# Constants
g = 9.81  # Acceleration due to gravity


# Step 1: Extract Ball Positions from the Video
def detect_ball(frame):
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range for the ball color (adjust as needed)
    lower_bound = np.array([100, 50, 50])
    upper_bound = np.array([130, 255, 255])

    # Threshold the HSV image to get only the ball color
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour (assuming it's the ball)
        largest_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
        return (int(x), int(y)), int(radius)
    return None, None


def extract_ball_positions(video_path):
    cap = cv2.VideoCapture(video_path)
    positions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        (x, y), radius = detect_ball(frame)
        if x is not None and y is not None:
            positions.append((x, y))

    cap.release()
    return positions


# Step 2: Define ODEs for Ball Motion
def ball_ode(state, g):
    x, y, vx, vy = state
    dxdt = vx
    dydt = vy
    dvxdt = 0
    dvydt = -g
    return np.array([dxdt, dydt, dvxdt, dvydt])


# Step 3: Implement Euler Method
def euler_method(ode_func, initial_state, t_span, dt, g):
    t = np.arange(t_span[0], t_span[1], dt)
    states = [initial_state]
    for _ in t[1:]:
        current_state = states[-1]
        derivative = ode_func(current_state, g)
        new_state = current_state + derivative * dt
        states.append(new_state)
    return np.array(states).T


# Step 4: Implement Runge-Kutta 4th Order (RK4) Method
def rk4_method(ode_func, initial_state, t_span, dt, g):
    t = np.arange(t_span[0], t_span[1], dt)
    states = [initial_state]
    for _ in t[1:]:
        current_state = states[-1]
        k1 = ode_func(current_state, g)
        k2 = ode_func(current_state + 0.5 * dt * k1, g)
        k3 = ode_func(current_state + 0.5 * dt * k2, g)
        k4 = ode_func(current_state + dt * k3, g)
        new_state = current_state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        states.append(new_state)
    return np.array(states).T


# Step 5: Reconstruct Trajectory Using Shooting Method (Grid Search)
def shooting_method_error(params, target_positions, t_span, dt, g, method="rk4"):
    vx0, vy0 = params
    initial_state = np.array([target_positions[0][0], target_positions[0][1], vx0, vy0])
    if method == "euler":
        sol = euler_method(ball_ode, initial_state, t_span, dt, g)
    else:
        sol = rk4_method(ball_ode, initial_state, t_span, dt, g)
    predicted_positions = list(zip(sol[0], sol[1]))
    error = np.sum(
        [
            np.linalg.norm(np.array(pred) - np.array(actual))
            for pred, actual in zip(predicted_positions, target_positions)
        ]
    )
    return error


def reconstruct_trajectory(ball_positions, dt=0.01, g=9.81, method="rk4"):
    t_span = (0, len(ball_positions) * dt)

    # Grid search for initial velocities (vx0, vy0)
    best_error = float("inf")
    best_params = None
    for vx0 in np.linspace(-10, 10, 20):  # Adjust range as needed
        for vy0 in np.linspace(-10, 10, 20):
            error = shooting_method_error(
                [vx0, vy0], ball_positions, t_span, dt, g, method
            )
            if error < best_error:
                best_error = error
                best_params = [vx0, vy0]

    vx0, vy0 = best_params
    initial_state = np.array([ball_positions[0][0], ball_positions[0][1], vx0, vy0])
    if method == "euler":
        sol = euler_method(ball_ode, initial_state, t_span, dt, g)
    else:
        sol = rk4_method(ball_ode, initial_state, t_span, dt, g)
    return sol


# Step 6: Simulate Interception
def intercept_ball(trajectory, intercept_point, dt=0.01, g=9.81):
    x0, y0 = intercept_point
    t_intercept = len(trajectory[0]) // 2  # Intercept at midpoint
    target_x, target_y = trajectory[0][t_intercept], trajectory[1][t_intercept]

    # Calculate initial velocity for the second ball
    vx0 = (target_x - x0) / (t_intercept * dt)
    vy0 = (target_y - y0 + 0.5 * g * (t_intercept * dt) ** 2) / (t_intercept * dt)

    # Simulate the second ball's trajectory
    initial_state = np.array([x0, y0, vx0, vy0])
    t_span = (0, t_intercept * dt)
    sol = rk4_method(ball_ode, initial_state, t_span, dt, g)
    return sol


# Step 7: Create Animation
def create_animation(trajectory, intercept_trajectory):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 800)
    ax.set_ylim(0, 600)

    (ball,) = ax.plot([], [], "bo", markersize=10)
    (intercept_ball,) = ax.plot([], [], "ro", markersize=10)

    def animate(i):
        ball.set_data(trajectory[0][i], trajectory[1][i])
        if i < len(intercept_trajectory[0]):
            intercept_ball.set_data(
                intercept_trajectory[0][i], intercept_trajectory[1][i]
            )
        return ball, intercept_ball

    ani = animation.FuncAnimation(
        fig, animate, frames=len(trajectory[0]), interval=50, blit=True
    )
    plt.show()


# Main Execution
if __name__ == "__main__":
    # Step 1: Extract ball positions from video
    video_path = "test.mp4"  # Replace with your video path
    ball_positions = extract_ball_positions(video_path)

    # Step 2: Reconstruct trajectory using RK4
    dt = 0.01  # Time step
    trajectory = reconstruct_trajectory(ball_positions, dt=dt, method="rk4")

    # Step 3: Simulate interception
    intercept_point = (100, 100)  # Random point
    intercept_trajectory = intercept_ball(trajectory, intercept_point, dt=dt)

    # Step 4: Create animation
    create_animation(trajectory, intercept_trajectory)
