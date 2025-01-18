import cv2
import numpy as np
from src.edge_detector import CannyEdgeDetector

def test_canny_detector(image_path: str, params: dict):
    """Test Canny Edge Detector with given parameters and show steps."""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    # Create detector with given parameters
    detector = CannyEdgeDetector(
        low_threshold=params['low_threshold'],
        high_threshold=params['high_threshold'],
        gaussian_kernel_size=params['kernel_size'],
        gaussian_sigma=params['sigma']
    )

    # Process each step
    gray = detector.to_grayscale(image)
    blurred = detector.apply_gaussian_blur(gray)
    magnitude, direction, gx, gy = detector.compute_gradients(blurred)
    suppressed = detector.non_maximum_suppression(magnitude, direction)
    strong_edges, weak_edges = detector.double_threshold(suppressed)
    final_edges = detector.edge_tracking(strong_edges, weak_edges)

    # Prepare images for display
    steps = {
        'Original': image,
        'Grayscale': gray.astype(np.uint8),
        'Gaussian Blur': blurred.astype(np.uint8),
        'Gradient Magnitude': (magnitude / magnitude.max() * 255).astype(np.uint8),
        'Non-maximum Suppression': (suppressed / suppressed.max() * 255).astype(np.uint8),
        'Strong Edges': strong_edges.astype(np.uint8) * 255,
        'Weak Edges': weak_edges.astype(np.uint8) * 255,
        'Final Edges': final_edges.astype(np.uint8) * 255
    }

    return steps