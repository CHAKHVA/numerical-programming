import numpy as np
import cv2
from time import time


class MotionDetector:
    def __init__(self, edge_high=90, background_alpha=0.01, movement_threshold=25, eps=30, min_samples=5):
        self.edge_high = edge_high
        self.background_alpha = background_alpha
        self.movement_threshold = movement_threshold
        self.background = None

        self.eps = eps
        self.min_samples = min_samples

    def gaussian_blur(self, image):
        kernel = np.array([
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1]
        ]) / 256

        height, width = image.shape
        result = np.zeros_like(image, dtype=np.float32)

        pad_size = 2
        padded = np.pad(image, pad_size, mode='edge')

        for y in range(height):
            for x in range(width):
                window = padded[y:y + 5, x:x + 5]
                result[y, x] = np.sum(window * kernel)

        return result.astype(np.uint8)

    def sobel(self, image):
        kernel_x = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]])

        kernel_y = np.array([[-1, -2, -1],
                             [0, 0, 0],
                             [1, 2, 1]])

        height, width = image.shape
        edges = np.zeros_like(image, dtype=np.float32)

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                window = image[y - 1:y + 2, x - 1:x + 2]
                gx = np.sum(window * kernel_x)
                gy = np.sum(window * kernel_y)
                edges[y, x] = np.sqrt(gx ** 2 + gy ** 2)

        return edges

    def dbscan(self, points, eps, min_samples):
        if len(points) == 0:
            return np.array([])

        n_points = len(points)
        labels = np.full(n_points, -1)
        cluster_label = 0

        def get_neighbors(point_idx):
            distances = np.sqrt(np.sum((points - points[point_idx]) ** 2, axis=1))
            return np.where(distances <= eps)[0]

        def expand_cluster(point_idx, neighbors, cluster_label):
            labels[point_idx] = cluster_label
            i = 0
            while i < len(neighbors):
                neighbor_idx = neighbors[i]
                if labels[neighbor_idx] == -1:
                    labels[neighbor_idx] = cluster_label
                    new_neighbors = get_neighbors(neighbor_idx)
                    if len(new_neighbors) >= min_samples:
                        neighbors = np.union1d(neighbors, new_neighbors)
                i += 1

        for point_idx in range(n_points):
            if labels[point_idx] != -1:
                continue

            neighbors = get_neighbors(point_idx)
            if len(neighbors) < min_samples:
                labels[point_idx] = -1  # Noise
            else:
                expand_cluster(point_idx, neighbors, cluster_label)
                cluster_label += 1

        return labels

    def detect_motion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blurred = self.gaussian_blur(gray)

        if self.background is None:
            self.background = blurred.astype(np.float32)
            return None, 0, []

        cv2.accumulateWeighted(blurred, self.background, self.background_alpha)
        background = self.background.astype(np.uint8)

        diff = cv2.absdiff(blurred, background)

        _, thresh = cv2.threshold(diff, self.movement_threshold, 255, cv2.THRESH_BINARY)

        edges = self.sobel(thresh)

        edge_mask = (edges > self.edge_high).astype(np.uint8) * 255

        points = np.column_stack(np.where(edge_mask > 0))

        if len(points) > 0:
            labels = self.dbscan(points, self.eps, self.min_samples)
            num_clusters = len(np.unique(labels[labels != -1]))

            centroids = []
            for label in np.unique(labels):
                if label != -1:
                    cluster_points = points[labels == label]
                    centroid = np.mean(cluster_points, axis=0)
                    centroids.append(centroid)
        else:
            num_clusters = 0
            centroids = []

        return edge_mask, num_clusters, centroids

    def estimate_speed(self, prev_centroids, curr_centroids, time_diff):
        if not prev_centroids or not curr_centroids or time_diff == 0:
            return []

        speeds = []
        for prev in prev_centroids:
            min_dist = float('inf')
            for curr in curr_centroids:
                dist = np.sqrt((prev[0] - curr[0]) ** 2 + (prev[1] - curr[1]) ** 2)
                min_dist = min(min_dist, dist)

            if min_dist != float('inf'):
                pixels_per_meter = 50
                speed = (min_dist / pixels_per_meter) / time_diff
                speeds.append(speed)

        return speeds


def process_video(video_path, frame_skip=2):
    cap = cv2.VideoCapture(video_path)
    detector = MotionDetector()

    prev_centroids = None
    prev_time = time()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:  # Skip frames to speed up processing
            continue

        current_time = time()
        time_diff = current_time - prev_time

        motion_mask, num_objects, curr_centroids = detector.detect_motion(frame)

        if motion_mask is not None:
            speeds = detector.estimate_speed(prev_centroids, curr_centroids, time_diff)

            for centroid in curr_centroids:
                cv2.circle(frame, (int(centroid[1]), int(centroid[0])), 5, (0, 255, 0), -1)

            cv2.putText(frame, f"Objects: {num_objects}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if speeds:
                avg_speed = sum(speeds) / len(speeds)
                cv2.putText(frame, f"Avg Speed: {avg_speed:.2f} m/s", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Motion Detection', frame)
            cv2.imshow('Motion Mask', motion_mask)

            prev_centroids = curr_centroids
            prev_time = current_time

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

process_video("video.mp4")