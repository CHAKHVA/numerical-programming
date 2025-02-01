import cv2
import matplotlib.pyplot as plt
import numpy as np


def detect_ball(frame):
    """
    Detect a blue ball in a frame and return its position and radius.
    """
    # Preprocess frame
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # HSV range for blue ball
    lower_bound = np.array([100, 50, 50])
    upper_bound = np.array([130, 255, 255])

    # Create and clean up mask
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the ball among contours
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:  # Minimum area threshold
            continue

        perimeter = cv2.arcLength(cnt, True)
        circularity = (
            (4 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0
        )

        if circularity > 0.7:  # Circularity threshold
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            return (int(x), int(y)), int(radius)

    return None, None


def extract_ball_positions(video_path, display=False):
    """
    Extract ball positions from video.
    Returns: List of (frame_number, x, y) tuples
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []

    positions = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        pos, radius = detect_ball(frame)

        if pos is not None:
            positions.append((frame_count, pos[0], pos[1]))

            if display:
                # Draw detection for visualization
                result = frame.copy()
                cv2.circle(result, pos, radius, (0, 255, 0), 2)
                cv2.circle(result, pos, 2, (0, 0, 255), -1)
                cv2.putText(
                    result,
                    f"Frame: {frame_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow("Ball Detection", result)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    if display:
        cv2.destroyAllWindows()

    return positions


def plot_trajectory(positions):
    """
    Plot the detected ball trajectory.
    """
    if not positions:
        print("No ball positions detected!")
        return

    positions = np.array(positions)

    plt.figure(figsize=(12, 8))

    plt.scatter(
        positions[:, 1],
        positions[:, 2],
        c=positions[:, 0],
        cmap="viridis",
        s=50,
        alpha=0.6,
    )
    plt.colorbar(label="Frame number")

    plt.plot(positions[:, 1], positions[:, 2], "b-", alpha=0.3)

    plt.title("Ball Trajectory")
    plt.xlabel("X Position (pixels)")
    plt.ylabel("Y Position (pixels)")
    plt.grid(True)
    plt.gca().invert_yaxis()

    plt.show()


if __name__ == "__main__":
    video_path = "test.mp4"

    # Extract positions with visual feedback
    positions = extract_ball_positions(video_path, display=True)

    # Plot the trajectory
    plot_trajectory(positions)

    # Print extracted positions
    print("\nExtracted positions (frame, x, y):")
    for frame, x, y in positions:
        print(f"Frame {frame}: ({x}, {y})")
