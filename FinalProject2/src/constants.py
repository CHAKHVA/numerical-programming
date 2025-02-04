"""Configuration parameters for the ball trajectory simulation."""

# Physics parameters
GRAVITY = 300  # Gravity constant (pixels/s^2)
FPS = 30  # Frames per second

# Video dimensions
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480

# Ball detection parameters
BALL_COLOR_LOWER = [100, 50, 50]  # HSV lower bounds
BALL_COLOR_UPPER = [130, 255, 255]  # HSV upper bounds
MIN_BALL_AREA = 100  # Minimum ball contour area
MIN_CIRCULARITY = 0.7  # Minimum circularity for ball detection
GAUSSIAN_KERNEL = (11, 11)  # Gaussian blur kernel size
MORPH_KERNEL = (5, 5)  # Morphological operations kernel size

# Simulation parameters
VELOCITY_SCALE = 0.1  # Scale factor for velocity
SIMULATION_EXTENSION = 5.0

# Animation parameters
ANIMATION_INTERVAL = 50  # Animation interval in milliseconds
TRAIL_ALPHA = 0.3  # Transparency of trajectory trails
ARROW_SCALE = 100  # Scale for velocity arrows
POINT_SIZE = 10  # Size of ball points in animation

# Random shooter parameters
SHOOTER_X_RANGE = (50, 200)  # Random x range for shooter position
SHOOTER_Y_RANGE = (200, 400)  # Random y range for shooter position

# File paths
DEFAULT_VIDEO_PATH = "test.mp4"
