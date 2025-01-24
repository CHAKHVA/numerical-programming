import cv2
import numpy as np

from src.detection.circle_detector import HoughCircleDetector
from src.detection.edge_detector import CannyEdgeDetector


def test_detection(image_path: str):
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    # Detect edges and circles
    canny = CannyEdgeDetector()
    hough = HoughCircleDetector(min_radius=10)

    # Get detections
    edges = canny.detect(image)
    circles = hough.detect(edges)

    # Print results
    print(f"Detected {len(circles)} circles")
    for i, (x, y, r) in enumerate(circles):
        print(f"Ball {i+1}: center=({x}, {y}), radius={r}")

    # Show images
    cv2.imshow("Original", image)
    cv2.imshow("Edges", edges.astype(np.uint8) * 255)

    # Draw circles on original image
    result = image.copy()
    for x, y, r in circles:
        cv2.circle(result, (x, y), r, (0, 255, 0), 2)
        cv2.circle(result, (x, y), 2, (0, 0, 255), 3)

    cv2.imshow("Detected Circles", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_detection("images/3.jpg")
