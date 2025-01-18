from typing import Any

import numpy as np


class CannyEdgeDetector:
    def __init__(self, low_threshold: float = 0.1, high_threshold: float = 0.3,
                 gaussian_kernel_size: int = 5, gaussian_sigma: float = 1.4):
        # Validate parameters
        if not 0 <= low_threshold <= high_threshold <= 1:
            raise ValueError("Thresholds must be in range [0,1] and low_threshold <= high_threshold")
        if gaussian_kernel_size % 2 == 0:
            raise ValueError("Gaussian kernel size must be odd")
        if gaussian_sigma <= 0:
            raise ValueError("Gaussian sigma must be positive")

        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.gaussian_kernel_size = gaussian_kernel_size
        self.gaussian_sigma = gaussian_sigma

    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert RGB image to grayscale using luminosity method."""
        return np.dot(image[..., :3], [0.299, 0.587, 0.114])

    def create_gaussian_kernel(self) -> np.ndarray:
        """Generate 2D Gaussian kernel."""
        size = self.gaussian_kernel_size
        # Create coordinate grids
        x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
        # Apply Gaussian formula
        gaussian = np.exp(-((x ** 2 + y ** 2) / (2.0 * self.gaussian_sigma ** 2)))
        # Normalize kernel
        return gaussian / gaussian.sum()

    def apply_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur to reduce noise."""
        kernel = self.create_gaussian_kernel()
        return self.convolve2d(image, kernel)

    def compute_gradients(self, image: np.ndarray) -> tuple[Any, Any, np.ndarray, np.ndarray]:
        """Compute gradients using Sobel operators."""
        # Sobel operators
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # Compute gradients
        gradient_x = self.convolve2d(image, sobel_x)
        gradient_y = self.convolve2d(image, sobel_y)

        # Compute magnitude with normalization
        gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        gradient_magnitude = gradient_magnitude / gradient_magnitude.max() * 255

        # Compute direction (in degrees for easier thresholding)
        gradient_direction = np.arctan2(gradient_y, gradient_x) * 180 / np.pi

        return gradient_magnitude, gradient_direction, gradient_x, gradient_y

    @staticmethod
    def non_maximum_suppression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Apply non-maximum suppression to thin edges."""
        height, width = magnitude.shape
        result = np.zeros_like(magnitude)

        # Convert angles to degrees and shift to positive values
        angle = np.degrees(direction) % 180

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                # Get neighbors based on gradient direction
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    neighbors = [magnitude[i, j - 1], magnitude[i, j + 1]]
                elif 22.5 <= angle[i, j] < 67.5:
                    neighbors = [magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]]
                elif 67.5 <= angle[i, j] < 112.5:
                    neighbors = [magnitude[i - 1, j], magnitude[i + 1, j]]
                else:
                    neighbors = [magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]]

                # Suppress non-maximum pixels
                if magnitude[i, j] >= max(neighbors):
                    result[i, j] = magnitude[i, j]

        return result

    def double_threshold(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply double threshold to classify edges."""
        # Compute thresholds
        high = image.max() * self.high_threshold
        low = high * self.low_threshold

        strong_edges = (image >= high)
        weak_edges = (image >= low) & (image < high)

        return strong_edges, weak_edges

    @staticmethod
    def edge_tracking(strong_edges: np.ndarray, weak_edges: np.ndarray) -> np.ndarray:
        """Track edges using hysteresis."""
        height, width = strong_edges.shape
        result = strong_edges.copy()

        # 8-connected neighbors
        dx = [-1, -1, -1, 0, 0, 1, 1, 1]
        dy = [-1, 0, 1, -1, 1, -1, 0, 1]

        # Track weak edges connected to strong edges
        while True:
            changed = False
            for i in range(1, height - 1):
                for j in range(1, width - 1):
                    if weak_edges[i, j] and not result[i, j]:
                        # Check if connected to strong edge
                        for k in range(8):
                            if result[i + dx[k], j + dy[k]]:
                                result[i, j] = True
                                changed = True
                                break
            if not changed:
                break

        return result

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Apply Canny edge detection pipeline."""
        # Convert to grayscale
        gray = self.to_grayscale(image)

        # Apply Gaussian blur
        blurred = self.apply_gaussian_blur(gray)

        # Compute gradients
        magnitude, direction, _, _ = self.compute_gradients(blurred)

        # Apply non-maximum suppression
        suppressed = self.non_maximum_suppression(magnitude, direction)

        # Apply double threshold
        strong_edges, weak_edges = self.double_threshold(suppressed)

        # Track edges by hysteresis
        edges = self.edge_tracking(strong_edges, weak_edges)

        return edges

    @staticmethod
    def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Apply 2D convolution."""
        kernel_height, kernel_width = kernel.shape
        pad_height = kernel_height // 2
        pad_width = kernel_width // 2

        # Pad image
        padded = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='reflect')

        # Create view of sliding windows
        windows = np.lib.stride_tricks.sliding_window_view(
            padded, (kernel_height, kernel_width)
        )

        return np.sum(windows * kernel, axis=(2, 3))

