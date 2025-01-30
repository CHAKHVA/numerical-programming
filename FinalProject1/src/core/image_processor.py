import cv2
import numpy as np

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

    def _compute_gradient(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute image gradients using Sobel operators"""
        # Sobel kernels
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # Apply kernels
        gradient_x = np.zeros_like(image, dtype=float)
        gradient_y = np.zeros_like(image, dtype=float)

        # Convolution
        rows, cols = image.shape
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                gradient_x[i, j] = np.sum(
                    image[i - 1 : i + 2, j - 1 : j + 2] * kernel_x
                )
                gradient_y[i, j] = np.sum(
                    image[i - 1 : i + 2, j - 1 : j + 2] * kernel_y
                )

        # Calculate magnitude and direction
        magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        direction = np.arctan2(gradient_y, gradient_x)

        return magnitude, direction, gradient_x

    def _non_maximum_suppression(
        self, magnitude: np.ndarray, direction: np.ndarray
    ) -> np.ndarray:
        """Apply non-maximum suppression to gradient magnitude"""
        rows, cols = magnitude.shape
        suppressed = np.zeros_like(magnitude)

        # Convert angles from radians to degrees
        direction = direction * 180 / np.pi
        direction[direction < 0] += 180

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                # Get neighboring pixels based on gradient direction
                if (0 <= direction[i, j] < 22.5) or (157.5 <= direction[i, j] <= 180):
                    neighbors = [magnitude[i, j - 1], magnitude[i, j + 1]]
                elif 22.5 <= direction[i, j] < 67.5:
                    neighbors = [magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]]
                elif 67.5 <= direction[i, j] < 112.5:
                    neighbors = [magnitude[i - 1, j], magnitude[i + 1, j]]
                else:  # 112.5 <= direction[i, j] < 157.5
                    neighbors = [magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]]

                # Keep pixel if it's a local maximum
                if magnitude[i, j] >= max(neighbors):
                    suppressed[i, j] = magnitude[i, j]

        return suppressed

    def _double_threshold(
        self, image: np.ndarray, low_threshold: float, high_threshold: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply double thresholding to identify strong and weak edges"""
        strong_edges = np.zeros_like(image)
        weak_edges = np.zeros_like(image)

        # Strong edges
        strong_indices = image >= high_threshold
        strong_edges[strong_indices] = 255

        # Weak edges
        weak_indices = (image >= low_threshold) & (image < high_threshold)
        weak_edges[weak_indices] = 128

        return strong_edges, weak_edges

    def _hysteresis(
        self, strong_edges: np.ndarray, weak_edges: np.ndarray
    ) -> np.ndarray:
        """Apply hysteresis to connect edges"""
        rows, cols = strong_edges.shape
        output = strong_edges.copy()

        # 8-connected neighborhood offsets
        dx = [-1, -1, -1, 0, 0, 1, 1, 1]
        dy = [-1, 0, 1, -1, 1, -1, 0, 1]

        # Iterate until no more changes
        while True:
            prev = output.copy()

            for i in range(1, rows - 1):
                for j in range(1, cols - 1):
                    if weak_edges[i, j] == 128:
                        # Check if any strong edge in neighborhood
                        for k in range(8):
                            ni, nj = i + dx[k], j + dy[k]
                            if output[ni, nj] == 255:
                                output[i, j] = 255
                                break

            if np.array_equal(prev, output):
                break

        return output

    def _canny_edge_detection(
        self, image: np.ndarray, low_threshold: float, high_threshold: float
    ) -> np.ndarray:
        """Custom implementation of Canny edge detection"""
        # 1. Compute gradients
        magnitude, direction, _ = self._compute_gradient(image)

        # 2. Non-maximum suppression
        suppressed = self._non_maximum_suppression(magnitude, direction)

        # 3. Double thresholding
        strong_edges, weak_edges = self._double_threshold(
            suppressed, low_threshold, high_threshold
        )

        # 4. Edge tracking by hysteresis
        edges = self._hysteresis(strong_edges, weak_edges)

        return edges.astype(np.uint8)

    def detect_shapes(self) -> list[Shape]:
        """Detect shapes in the image"""
        self.load_image()

        # Process image
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA)

        # Use custom Canny edge detection
        edges = self._canny_edge_detection(
            blurred, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD
        )

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
