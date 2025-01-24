import cv2

from src.detection.circle_detector import HoughCircleDetector
from src.detection.edge_detector import CannyEdgeDetector
from src.trajectory.visualizer import TrajectoryVisualizer


def main():
    # Read image
    image_path = "images/1.jpg"
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    # Use default parameters for detectors
    canny = CannyEdgeDetector()
    hough = HoughCircleDetector(min_radius=10)

    # Detect edges and circles
    edges = canny.detect(image)
    circles = hough.detect(edges)

    # Set shooting point
    shooting_point = (50, 500)

    # Visualize
    visualizer = TrajectoryVisualizer(circles, shooting_point)
    visualizer.animate()


if __name__ == "__main__":
    main()
