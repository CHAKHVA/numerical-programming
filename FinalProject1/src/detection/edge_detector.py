import numpy as np


class CannyEdgeDetector:
    def __init__(
        self,
        low_threshold: float = 0.08,
        high_threshold: float = 0.25,
        gaussian_kernel_size: int = 5,
        gaussian_sigma: float = 1.4,
    ):
        # Validate parameters
        if not 0 <= low_threshold <= high_threshold <= 1:
            raise ValueError(
                "Thresholds must be in range [0,1] and low_threshold <= high_threshold"
            )
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
        return np.dot(image[..., :3], [0.299, 0.587, 0.114])

    def create_gaussian_kernel(self) -> np.ndarray:
        size = self.gaussian_kernel_size
        # Create coordinate grids
        x, y = np.mgrid[-size // 2 + 1 : size // 2 + 1, -size // 2 + 1 : size // 2 + 1]
        # Apply Gaussian formula
        gaussian = np.exp(-((x**2 + y**2) / (2.0 * self.gaussian_sigma**2)))
        # Normalize kernel
        return gaussian / gaussian.sum()

    def apply_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        kernel = self.create_gaussian_kernel()
        return self.convolve2d(image, kernel)

    def compute_gradients(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Separable Sobel kernels
        sobel_x_1d = np.array([1, 0, -1])
        sobel_y_1d = np.array([1, 2, 1])

        # Apply separable convolution for x gradient
        temp_x = self.convolve2d(image, sobel_x_1d.reshape(1, -1))
        gradient_x = self.convolve2d(temp_x, sobel_y_1d.reshape(-1, 1))

        # Apply separable convolution for y gradient
        temp_y = self.convolve2d(image, sobel_y_1d.reshape(1, -1))
        gradient_y = self.convolve2d(temp_y, sobel_x_1d.reshape(-1, 1))

        # Compute magnitude and normalize
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        gradient_magnitude = gradient_magnitude / gradient_magnitude.max() * 255

        # Compute direction in degrees
        gradient_direction = np.arctan2(gradient_y, gradient_x) * 180 / np.pi

        return gradient_magnitude, gradient_direction, gradient_x, gradient_y

    @staticmethod
    def non_maximum_suppression(
        magnitude: np.ndarray, direction: np.ndarray
    ) -> np.ndarray:
        height, width = magnitude.shape
        angle = np.degrees(direction) % 180

        # Pad magnitude array
        pad_mag = np.pad(magnitude, ((1, 1), (1, 1)), mode="constant")
        result = np.zeros_like(magnitude)

        # Pre-compute angle masks
        angle_0 = (angle < 22.5) | (angle >= 157.5)
        angle_45 = (angle >= 22.5) & (angle < 67.5)
        angle_90 = (angle >= 67.5) & (angle < 112.5)
        angle_135 = (angle >= 112.5) & (angle < 157.5)

        # Create indices for all pixels
        y_indices, x_indices = np.mgrid[1 : height + 1, 1 : width + 1]

        # Apply suppression for each direction
        if angle_0.any():
            mask = angle_0[y_indices - 1, x_indices - 1]
            result[mask] = (
                pad_mag[y_indices, x_indices][mask]
                >= np.maximum(
                    pad_mag[y_indices, x_indices - 1][mask],
                    pad_mag[y_indices, x_indices + 1][mask],
                )
            ) * magnitude[mask]

        if angle_45.any():
            mask = angle_45[y_indices - 1, x_indices - 1]
            result[mask] = (
                pad_mag[y_indices, x_indices][mask]
                >= np.maximum(
                    pad_mag[y_indices - 1, x_indices - 1][mask],
                    pad_mag[y_indices + 1, x_indices + 1][mask],
                )
            ) * magnitude[mask]

        if angle_90.any():
            mask = angle_90[y_indices - 1, x_indices - 1]
            result[mask] = (
                pad_mag[y_indices, x_indices][mask]
                >= np.maximum(
                    pad_mag[y_indices - 1, x_indices][mask],
                    pad_mag[y_indices + 1, x_indices][mask],
                )
            ) * magnitude[mask]

        if angle_135.any():
            mask = angle_135[y_indices - 1, x_indices - 1]
            result[mask] = (
                pad_mag[y_indices, x_indices][mask]
                >= np.maximum(
                    pad_mag[y_indices - 1, x_indices + 1][mask],
                    pad_mag[y_indices + 1, x_indices - 1][mask],
                )
            ) * magnitude[mask]

        return result

    def double_threshold(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Compute thresholds
        high = image.max() * self.high_threshold
        low = high * self.low_threshold

        strong_edges = image >= high
        weak_edges = (image >= low) & (image < high)

        return strong_edges, weak_edges

    @staticmethod
    def edge_tracking(strong_edges: np.ndarray, weak_edges: np.ndarray) -> np.ndarray:
        result = strong_edges.copy()

        # Create padded array for easier neighbor checking
        padded_result = np.pad(result, ((1, 1), (1, 1)), mode="constant")
        padded_weak = np.pad(weak_edges, ((1, 1), (1, 1)), mode="constant")

        # Keep track of weak edge positions
        weak_y, weak_x = np.where(padded_weak[1:-1, 1:-1])
        weak_y += 1
        weak_x += 1

        while True:
            changed = False
            # Create 3x3 window sum for each weak edge position
            neighbors_sum = np.sum(
                [
                    padded_result[weak_y + dy, weak_x + dx]
                    for dy in [-1, 0, 1]
                    for dx in [-1, 0, 1]
                ],
                axis=0,
            )

            # Find weak edges with strong neighbors
            to_activate = (neighbors_sum > 0) & ~padded_result[weak_y, weak_x]

            if to_activate.any():
                changed = True
                padded_result[weak_y[to_activate], weak_x[to_activate]] = True

            if not changed:
                break

        return padded_result[1:-1, 1:-1]

    def detect(self, image: np.ndarray) -> np.ndarray:
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
        kernel_height, kernel_width = kernel.shape
        pad_height = kernel_height // 2
        pad_width = kernel_width // 2

        # Pad image
        padded = np.pad(
            image, ((pad_height, pad_height), (pad_width, pad_width)), mode="reflect"
        )

        # Create view of sliding windows
        windows = np.lib.stride_tricks.sliding_window_view(
            padded, (kernel_height, kernel_width)
        )

        return np.sum(windows * kernel, axis=(2, 3))
