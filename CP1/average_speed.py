import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import math


# Step 1: Video Frame Extraction
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    if not cap.isOpened():
        print("Error: Could not open video.")
        return frames

    frame_skip = int(cap.get(cv2.CAP_PROP_FPS) // 2)  # Skip frames to speed up processing
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames


# Step 2: Grayscale Conversion
def convert_to_grayscale(frames):
    grayscale_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    return grayscale_frames


# Step 3: Background Subtraction
def background_subtraction(frames, alpha=0.05):
    background = np.float32(frames[0])
    foreground_masks = []

    for frame in frames:
        gray_frame = np.float32(frame)
        gray_background = cv2.convertScaleAbs(background)
        foreground_mask = cv2.absdiff(cv2.convertScaleAbs(gray_frame), gray_background)
        cv2.accumulateWeighted(gray_frame, background, alpha)
        _, fg_mask = cv2.threshold(foreground_mask, 50, 255, cv2.THRESH_BINARY)
        foreground_masks.append(fg_mask)

    return foreground_masks


# Step 4: Convert Foreground Mask to Points
def mask_to_points(foreground_masks):
    all_points = []
    for mask in foreground_masks:
        points = np.column_stack(np.where(mask > 0))
        all_points.append(points)
    return all_points


# Step 5: Object Detection with DBSCAN
def detect_objects_with_dbscan(points, eps=10, min_samples=5):
    labels_list = []
    for frame_points in points:
        if len(frame_points) == 0:
            labels_list.append([])
            continue
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(frame_points)
        labels_list.append(labels)
    return labels_list


# Step 6: Centroid Calculation
def calculate_centroids(points, labels_list):
    centroids_list = []
    for i, labels in enumerate(labels_list):
        centroids = []
        if len(labels) == 0:
            centroids_list.append(centroids)
            continue
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:  # Ignore noise
                continue
            cluster_points = points[i][labels == label]
            centroid = np.mean(cluster_points, axis=0)
            centroids.append(tuple(centroid))
        centroids_list.append(centroids)
    return centroids_list


# Step 7: Object Tracking Across Frames
def track_objects(centroids_list, max_distance=20):
    tracked_objects = []
    previous_centroids = []

    for centroids in centroids_list:
        if not previous_centroids:
            tracked_objects.append(centroids)
        else:
            current_tracked = []
            for centroid in centroids:
                min_dist = float('inf')
                matched_centroid = None
                for prev_centroid in previous_centroids:
                    dist = math.sqrt((centroid[0] - prev_centroid[0]) ** 2 + (centroid[1] - prev_centroid[1]) ** 2)
                    if dist < min_dist and dist < max_distance:
                        min_dist = dist
                        matched_centroid = prev_centroid
                if matched_centroid:
                    current_tracked.append(centroid)
            tracked_objects.append(current_tracked)
        previous_centroids = centroids

    return tracked_objects


# Step 8: Speed Calculation of Tracked Objects
def calculate_speed(tracked_objects, frame_rate=30):
    object_speeds = []
    for i in range(1, len(tracked_objects)):
        frame_speeds = []
        for j, centroid in enumerate(tracked_objects[i]):
            if j < len(tracked_objects[i - 1]):
                prev_centroid = tracked_objects[i - 1][j]
                displacement = math.sqrt((centroid[0] - prev_centroid[0]) ** 2 + (centroid[1] - prev_centroid[1]) ** 2)
                speed = displacement * frame_rate
                frame_speeds.append(speed)
        object_speeds.append(frame_speeds)
    return object_speeds


# Step 9: Visualization
def visualize_detection(frames, centroids_list):
    for i, frame in enumerate(frames):
        for centroid in centroids_list[i]:
            cv2.circle(frame, (int(centroid[1]), int(centroid[0])), 5, (0, 0, 255), -1)
        cv2.imshow('Detected Objects', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


# Main Function
def main(video_path):
    frames = extract_frames(video_path)
    if not frames:
        return
    grayscale_frames = convert_to_grayscale(frames)
    foreground_masks = background_subtraction(grayscale_frames)
    points = mask_to_points(foreground_masks)
    labels_list = detect_objects_with_dbscan(points)
    centroids_list = calculate_centroids(points, labels_list)
    tracked_objects = track_objects(centroids_list)
    object_speeds = calculate_speed(tracked_objects)
    visualize_detection(frames, centroids_list)


if __name__ == "__main__":
    video_path = "video.mp4"
    main(video_path)
