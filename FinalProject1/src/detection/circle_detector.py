import numpy as np
import cv2


class HoughCircleDetector:
    def __init__(self, min_radius: int = 40, max_radius: int = 50,
                 threshold: int = 15, step_radius: int = 1, step_theta: int = 5):
        """
        Initialize Hough Circle Detector with optimized default parameters.
        """
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.threshold = threshold
        self.step_radius = step_radius
        self.step_theta = step_theta

    def detect(self, edge_image: np.ndarray) -> list[tuple[int, int, int]]:
        """
        Detect circles in edge image using improved Hough transform.
        """
        height, width = edge_image.shape
        edge_points = np.where(edge_image > 0)

        # Initialize parameters
        radii = np.arange(self.min_radius, self.max_radius + 1, self.step_radius)
        theta = np.linspace(0, 2 * np.pi, int(360 / self.step_theta))

        # Create 3D accumulator array
        acc = np.zeros((len(radii), height, width))

        # Precompute sine and cosine values
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Vote in accumulator
        for y, x in zip(*edge_points):
            for r_idx, radius in enumerate(radii):
                # Calculate potential center coordinates
                x_centers = (x - radius * cos_theta).astype(int)
                y_centers = (y - radius * sin_theta).astype(int)

                # Filter valid coordinates
                valid_points = (x_centers >= 0) & (x_centers < width) & \
                               (y_centers >= 0) & (y_centers < height)

                # Accumulate votes
                x_valid = x_centers[valid_points]
                y_valid = y_centers[valid_points]
                acc[r_idx, y_valid, x_valid] += 1

        circles = []
        # Find peaks in accumulator
        for r_idx, radius in enumerate(radii):
            acc_layer = acc[r_idx]

            # Apply Gaussian smoothing to reduce noise
            acc_layer = cv2.GaussianBlur(acc_layer, (5, 5), 1.0)

            while True:
                max_val = acc_layer.max()
                if max_val < self.threshold:
                    break

                # Find circle center
                y_center, x_center = np.unravel_index(acc_layer.argmax(), acc_layer.shape)

                # Add circle to results
                circles.append((int(x_center), int(y_center), int(radius)))

                # Suppress neighborhood to avoid multiple detections
                y_idx, x_idx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
                mask_region = x_idx ** 2 + y_idx ** 2 <= (radius * 1.1) ** 2

                # Calculate valid suppression region
                y_start = max(0, y_center - radius)
                y_end = min(height, y_center + radius + 1)
                x_start = max(0, x_center - radius)
                x_end = min(width, x_center + radius + 1)

                # Suppress region
                mask_height = y_end - y_start
                mask_width = x_end - x_start
                mask = mask_region[:mask_height, :mask_width]
                acc_layer[y_start:y_end, x_start:x_end][mask] = 0

        # Post-process to remove overlapping circles
        circles = self._remove_overlapping_circles(circles)

        return circles

    def _remove_overlapping_circles(self, circles: list[tuple[int, int, int]],
                                    overlap_threshold: float = 0.5) -> list[tuple[int, int, int]]:
        """Remove overlapping circles based on center distance and radius."""
        if not circles:
            return circles

        # Sort circles by accumulator value (assuming stronger circles are detected first)
        circles = sorted(circles, key=lambda x: x[2], reverse=True)
        filtered_circles = []

        for circle in circles:
            x1, y1, r1 = circle
            overlap = False

            for filtered_circle in filtered_circles:
                x2, y2, r2 = filtered_circle
                # Calculate center distance
                distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                # Check overlap
                if distance < (r1 + r2) * overlap_threshold:
                    overlap = True
                    break

            if not overlap:
                filtered_circles.append(circle)

        return filtered_circles

    @staticmethod
    def draw_circles(image: np.ndarray, circles: list[tuple[int, int, int]]) -> np.ndarray:
        """Draw detected circles with centers and radius."""
        result = image.copy()
        for x, y, r in circles:
            # Draw circle outline
            cv2.circle(result, (x, y), r, (0, 255, 0), 2)
            # Draw center point
            cv2.circle(result, (x, y), 2, (0, 0, 255), 3)
            # Draw radius line
            cv2.line(result, (x, y), (x + r, y), (255, 0, 0), 1)
        return result