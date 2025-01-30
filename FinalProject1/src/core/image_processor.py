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
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.image = None
        self.height = 0
        self.width = 0
        # Add parameters for filtering shapes
        self.min_shape_area = 100  # Minimum area to consider a shape
        self.max_shape_area = 100000  # Maximum area to consider a shape
        self.min_contour_distance = 20  # Minimum distance between contour centers

    def load_image(self) -> None:
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise ValueError(f"Could not load image at {self.image_path}")
        self.height, self.width = self.image.shape[:2]
        # Automatically set area thresholds based on image size
        self.min_shape_area = (self.width * self.height) * 0.001  # 0.1% of image area
        self.max_shape_area = (self.width * self.height) * 0.1  # 10% of image area

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image to reduce noise and enhance features"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)

        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        return enhanced

    def create_gaussian_kernel(self, size: int, sigma: float) -> np.ndarray:
        """Optimized Gaussian kernel creation using vectorization"""
        if sigma <= 0 or size % 2 == 0:
            raise ValueError("Invalid Gaussian parameters")

        ax = np.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, size)
        xx, yy = np.meshgrid(ax, ax)

        # Vectorized computation
        kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
        return kernel / np.sum(kernel)

    def custom_convolution(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Optimized convolution using numpy operations"""
        if len(image.shape) == 3:
            height, width, channels = image.shape
        else:
            height, width = image.shape
            channels = 1

        k_height, k_width = kernel.shape
        pad_h, pad_w = k_height // 2, k_width // 2

        # Create padded image
        if channels == 1:
            padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
        else:
            padded = np.pad(
                image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="constant"
            )

        # Initialize output
        output = np.zeros_like(image, dtype=np.float32)

        # Optimized convolution using numpy operations
        for i in range(height):
            for j in range(width):
                if channels == 1:
                    window = padded[i : i + k_height, j : j + k_width]
                    output[i, j] = np.sum(window * kernel)
                else:
                    for c in range(channels):
                        window = padded[i : i + k_height, j : j + k_width, c]
                        output[i, j, c] = np.sum(window * kernel)

        return output

    def filter_contours(self, contours: list[np.ndarray]) -> list[np.ndarray]:
        """Filter contours based on area and proximity"""
        filtered_contours = []
        centers = []

        for contour in contours:
            # Calculate contour area
            area = cv2.contourArea(contour)
            if area < self.min_shape_area or area > self.max_shape_area:
                continue

            # Calculate center
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Check distance from existing centers
            too_close = False
            for existing_center in centers:
                dist = np.sqrt(
                    (cx - existing_center[0]) ** 2 + (cy - existing_center[1]) ** 2
                )
                if dist < self.min_contour_distance:
                    too_close = True
                    break

            if not too_close:
                filtered_contours.append(contour)
                centers.append((cx, cy))

        return filtered_contours

    def detect_shapes(self) -> list[Shape]:
        """Improved shape detection with filtering"""
        self.load_image()

        # Preprocess image
        processed = self.preprocess_image(self.image)

        # Apply edge detection
        edges = cv2.Canny(processed, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)

        # Find and filter contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        filtered_contours = self.filter_contours(contours)

        shapes = []
        for contour in filtered_contours:
            # Approximate the contour
            epsilon = CONTOUR_EPSILON_FACTOR * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) >= ELLIPSE_MIN_POINTS and len(contour) >= 5:
                try:
                    # Fit ellipse using OpenCV's method for better accuracy
                    (x, y), (major_axis, minor_axis), angle = cv2.fitEllipse(contour)
                    y = self.height - y  # Convert to mathematical coordinates
                    # Only add if the aspect ratio is reasonable
                    if 0.5 <= major_axis / minor_axis <= 2.0:
                        shapes.append(
                            Shape("ellipse", (x, y), (major_axis / 2, minor_axis / 2))
                        )
                except cv2.error:
                    continue
            else:
                # Fit circle only if the contour is roughly circular
                (x, y), radius = cv2.minEnclosingCircle(contour)
                area = cv2.contourArea(contour)
                circular_area = np.pi * radius * radius
                # Check if the area matches a circle (within 20% tolerance)
                if 0.8 <= area / circular_area <= 1.2:
                    y = self.height - y
                    shapes.append(Shape("circle", (x, y), radius))

        return shapes
