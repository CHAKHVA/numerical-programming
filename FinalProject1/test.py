import cv2
import matplotlib.pyplot as plt
import numpy as np

# Constants
g = 9.81  # Acceleration due to gravity (m/s^2)


# Step 1: Edge Detection and Shape Identification
def detect_shapes(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")

    # Get image dimensions
    height, width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Identify shapes and their properties
    shapes = []
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Fit a circle or ellipse
        if len(approx) >= 5:  # Ellipse
            (x, y), (major_axis, minor_axis), angle = cv2.fitEllipse(contour)
            # Convert y-coordinate to match mathematical coordinate system
            y = height - y
            shapes.append(("ellipse", (x, y), (major_axis / 2, minor_axis / 2)))
        else:  # Circle or polygon
            (x, y), radius = cv2.minEnclosingCircle(contour)
            # Convert y-coordinate to match mathematical coordinate system
            y = height - y
            shapes.append(("circle", (x, y), radius))

    return shapes, height, width


# Step 2: Custom ODE Solver (RK4 Method)
def rk4(f, t0, y0, h, n):
    """
    Runge-Kutta 4th order method for solving ODEs.
    """
    t = np.zeros(n + 1)
    y = np.zeros((n + 1, len(y0)))
    t[0] = t0
    y[0] = y0

    for i in range(n):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h / 2, y[i] + h / 2 * k1)
        k3 = f(t[i] + h / 2, y[i] + h / 2 * k2)
        k4 = f(t[i] + h, y[i] + h * k3)

        y[i + 1] = y[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t[i + 1] = t[i] + h

    return t, y


# Step 3: Projectile Motion Equations
def projectile_motion(t, y):
    """
    Defines the ODE for projectile motion.
    y[0] = x position, y[1] = y position
    y[2] = x velocity, y[3] = y velocity
    """
    x, y_pos, vx, vy = y
    dxdt = vx
    dydt = vy
    dvxdt = 0  # No horizontal acceleration
    dvydt = -g  # Vertical acceleration due to gravity
    return np.array([dxdt, dydt, dvxdt, dvydt])


# Step 4: Shooting Method
def shooting_method(target_x, target_y, initial_guess, tolerance=1e-2, max_iter=100):
    """
    Shooting method to find the correct initial velocity.
    """
    v0_guess = initial_guess.copy()  # Make a copy to avoid modifying the original
    for i in range(max_iter):
        # Initial conditions: [x0, y0, vx0, vy0]
        y0 = [0, 0, v0_guess[0], v0_guess[1]]

        # Solve the ODE using RK4
        t, y = rk4(projectile_motion, t0=0, y0=y0, h=0.1, n=100)

        # Extract the final position
        x_final = y[-1, 0]
        y_final = y[-1, 1]

        # Check if the ball is close to the target
        if abs(x_final - target_x) < tolerance and abs(y_final - target_y) < tolerance:
            print(f"Converged after {i+1} iterations.")
            return v0_guess, t, y

        # Update the guess for initial velocity
        v0_guess[0] += (target_x - x_final) / 10
        v0_guess[1] += (target_y - y_final) / 10

    print("Did not converge.")
    return None, None, None


def animate_trajectory(trajectory, target_x, target_y, plt_obj):
    """
    Animates the drawing of a trajectory path point by point.
    """
    num_points = len(trajectory)
    for i in range(1, num_points):
        # Plot segment of the trajectory with dotted line
        plt_obj.plot(
            trajectory[i - 1 : i + 1, 0],
            trajectory[i - 1 : i + 1, 1],
            "b:",  # blue dotted line
            linewidth=1.5,
        )
        plt_obj.pause(0.0001)  # Pause briefly to create animation effect


def main(image_path):
    # Detect shapes in the image
    shapes, height, width = detect_shapes(image_path)
    print(f"Detected shapes: {shapes}")

    # Create a plot for visualization
    plt.figure(figsize=(10, 6))

    # Set up the plot
    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.xlabel("Horizontal Distance (pixels)")
    plt.ylabel("Vertical Distance (pixels)")
    plt.title("Projectile Motion to Hit Targets")
    plt.grid(True, linestyle="--", alpha=0.7)

    # Plot all targets first
    for shape in shapes:
        shape_type, (target_x, target_y), radius = shape
        plt.scatter(target_x, target_y, color="red", s=100)
        # Draw target circle
        circle = plt.Circle(
            (target_x, target_y),
            radius[0] if isinstance(radius, tuple) else radius,
            fill=False,
            color="red",
            linestyle="--",
            alpha=0.5,
        )
        plt.gca().add_patch(circle)

    # Iterate over each shape and perform shooting with animation
    for i, shape in enumerate(shapes):
        shape_type, (target_x, target_y), radius = shape
        print(
            f"\nTarget {i+1}: {shape_type} at ({target_x:.1f}, {target_y:.1f}) with radius {radius}"
        )

        # Initial guess for velocity [vx0, vy0]
        initial_guess = [10, 10]

        # Find the correct initial velocity using the shooting method
        v0, t, trajectory = shooting_method(target_x, target_y, initial_guess)

        if v0 is not None:
            # Animate the trajectory drawing
            animate_trajectory(trajectory, target_x, target_y, plt)

            # Add legend entry for this trajectory
            plt.plot(
                [],
                [],
                "b:",
                label=f"Target {i+1}: {shape_type} at ({target_x:.1f}, {target_y:.1f})",
            )
            plt.legend()
        else:
            print("Failed to find a solution for this target.")

        plt.pause(0.1)  # Pause between targets

    plt.show()


"""
# Step 5: Main Function
def main(image_path):
    # Detect shapes in the image
    shapes, height, width = detect_shapes(image_path)
    print(f"Detected shapes: {shapes}")

    # Create a plot for visualization
    plt.figure(figsize=(10, 6))

    # Iterate over each shape and perform shooting
    for shape in shapes:
        shape_type, (target_x, target_y), radius = shape
        print(
            f"\nTarget: {shape_type} at ({target_x:.1f}, {target_y:.1f}) with radius {radius}"
        )

        # Initial guess for velocity [vx0, vy0]
        initial_guess = [10, 10]

        # Find the correct initial velocity using the shooting method
        v0, t, trajectory = shooting_method(target_x, target_y, initial_guess)

        if v0 is not None:
            # Plot the trajectory
            plt.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                label=f"{shape_type} at ({target_x:.1f}, {target_y:.1f})",
            )
            plt.scatter(target_x, target_y, color="red")
        else:
            print("Failed to find a solution for this target.")

    # Set the plot limits to match the image dimensions
    plt.xlim(0, width)
    plt.ylim(0, height)

    # Finalize the plot
    plt.xlabel("Horizontal Distance (pixels)")
    plt.ylabel("Vertical Distance (pixels)")
    plt.title("Projectile Motion to Hit Targets")
    plt.legend()
    plt.grid()
    plt.show()
"""

# Run the program
image_path = "images/test3.png"  # Replace with the path to your image
main(image_path)
