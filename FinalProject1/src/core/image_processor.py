import cv2

from src.constants import (
    CANNY_HIGH_THRESHOLD,
    CANNY_LOW_THRESHOLD,
    CONTOUR_EPSILON_FACTOR,
    ELLIPSE_MIN_POINTS,
    GAUSSIAN_KERNEL_SIZE,
    GAUSSIAN_SIGMA,
)
from src.models.shape import Shape


class ImageProcessor:
    """Class to handle image processing and shape detection"""

    def __init__(self, image_path: str):
        self.image_path = image_path
        self.image = None
        self.height = 0
        self.width = 0

    def load_image(self) -> None:
        """Load and validate the image"""
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise ValueError(f"Could not load image at {self.image_path}")
        self.height, self.width = self.image.shape[:2]

    def detect_shapes(self) -> list[Shape]:
        """Detect shapes in the image"""
        self.load_image()

        # Process image
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA)
        edges = cv2.Canny(blurred, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)

        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        shapes = []
        for contour in contours:
            epsilon = CONTOUR_EPSILON_FACTOR * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) >= ELLIPSE_MIN_POINTS:  # Ellipse
                (x, y), (major_axis, minor_axis), angle = cv2.fitEllipse(contour)
                y = self.height - y  # Convert to mathematical coordinates
                shapes.append(
                    Shape("ellipse", (x, y), (major_axis / 2, minor_axis / 2))
                )
            else:  # Circle
                (x, y), radius = cv2.minEnclosingCircle(contour)
                y = self.height - y  # Convert to mathematical coordinates
                shapes.append(Shape("circle", (x, y), radius))

        return shapes
