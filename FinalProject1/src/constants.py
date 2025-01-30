# Physical constants
GRAVITY = 9.81  # Acceleration due to gravity (m/s^2)
AIR_DENSITY = 1.225  # Air density (kg/m^3)
DRAG_COEFFICIENT = 0.47  # Drag coefficient for a sphere
PROJECTILE_MASS = 0.1  # Mass of projectile (kg)
PROJECTILE_RADIUS = 0.02  # Radius of projectile (m)

# Image processing constants
GAUSSIAN_KERNEL_SIZE = (5, 5)
GAUSSIAN_SIGMA = 1.0
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150
CONTOUR_EPSILON_FACTOR = 0.01
ELLIPSE_MIN_POINTS = 5

# Numerical method constants
RK4_STEP_SIZE = 0.1
RK4_NUM_STEPS = 100
SHOOTING_TOLERANCE = 1e-2
SHOOTING_MAX_ITERATIONS = 100
VELOCITY_UPDATE_FACTOR = 10  # Divisor for velocity updates in shooting method

# Visualization constants
FIGURE_SIZE = (10, 6)
GRID_ALPHA = 0.7
TARGET_MARKER_SIZE = 10
TARGET_CIRCLE_ALPHA = 0.5
TRAJECTORY_LINE_WIDTH = 1.5
TRAJECTORY_ANIMATION_PAUSE = 0.0001
TARGET_PAUSE = 0.1

# Initial conditions
DEFAULT_INITIAL_VELOCITY = [10, 10]
DEFAULT_INITIAL_POSITION = (400, 300)
