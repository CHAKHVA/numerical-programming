"""Ball detection and position extraction from video."""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from constants import (
    BALL_COLOR_LOWER,
    BALL_COLOR_UPPER,
    GAUSSIAN_KERNEL,
    MIN_BALL_AREA,
    MIN_CIRCULARITY,
    MORPH_KERNEL,
)


class BallDetector:
    @staticmethod
    def detect_ball(frame) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
        """Detect ball in a single frame."""
        blurred = cv2.GaussianBlur(frame, GAUSSIAN_KERNEL, 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        lower_bound = np.array(BALL_COLOR_LOWER)
        upper_bound = np.array(BALL_COLOR_UPPER)

        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        kernel = np.ones(MORPH_KERNEL, np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_BALL_AREA:
                continue

            perimeter = cv2.arcLength(cnt, True)
            circularity = (
                (4 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0
            )

            if circularity > MIN_CIRCULARITY:
                ((x, y), radius) = cv2.minEnclosingCircle(cnt)
                return (int(x), int(y)), int(radius)

        return None, None

    @classmethod
    def extract_positions(
        cls, video_path: str, display: bool = False
    ) -> List[Tuple[int, int]]:
        """Extract ball positions from video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        positions = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            pos, radius = cls.detect_ball(frame)

            if pos is not None:
                positions.append(pos)

                if display:
                    cv2.circle(frame, pos, radius, (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"Frame: {frame_count}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                    cv2.imshow("Detection", frame)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        cap.release()
        if display:
            cv2.destroyAllWindows()

        return positions
        return positions
        return positions
        cap.release()
        if display:
            cv2.destroyAllWindows()

        return positions
        return positions
        return positions
