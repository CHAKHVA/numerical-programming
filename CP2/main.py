import cv2
import numpy as np


class BallMotionODE:
    def __init__(self):
        self.g = 9.81

    def derivatives(self, state, mass, drag_coef):
        x, y, vx, vy = state
        v = np.sqrt(vx * vx + vy * vy)

        dx_dt = vx
        dy_dt = vy
        dvx_dt = -(drag_coef / mass) * v * vx
        dvy_dt = -self.g - (drag_coef / mass) * v * vy

        return np.array([dx_dt, dy_dt, dvx_dt, dvy_dt])

    def rk4_step(self, state, dt, mass, drag_coef):
        k1 = self.derivatives(state, mass, drag_coef)
        k2 = self.derivatives(state + dt * k1 / 2, mass, drag_coef)
        k3 = self.derivatives(state + dt * k2 / 2, mass, drag_coef)
        k4 = self.derivatives(state + dt * k3, mass, drag_coef)

        return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


class BallTracker:
    def __init__(self):
        self.ball_color_lower = np.array([5, 50, 50])
        self.ball_color_upper = np.array([25, 255, 255])

        self.min_radius = 5
        self.max_radius = 50

        self.pixels_per_meter = 100

        self.min_area = 50
        self.min_circularity = 0.5

    def detect_ball(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.ball_color_lower, self.ball_color_upper)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, None

        best_contour = None
        best_circularity = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)

            if circularity > best_circularity:
                best_circularity = circularity
                best_contour = contour

        if best_contour is None or best_circularity < self.min_circularity:
            return None, None

        ((x, y), radius) = cv2.minEnclosingCircle(best_contour)

        if radius < self.min_radius or radius > self.max_radius:
            return None, None

        return (int(x), int(y)), int(radius)


class MotionAnalyzer:
    def __init__(self):
        self.ode_solver = BallMotionODE()
        self.positions = []
        self.timestamps = []
        self.window_size = 20

    def calculate_parameters(self, new_position, new_time):
        self.positions.append(new_position)
        self.timestamps.append(new_time)

        if len(self.positions) > self.window_size:
            self.positions.pop(0)
            self.timestamps.pop(0)

        if len(self.positions) < 5:
            return None

        positions = np.array(self.positions)
        times = np.array(self.timestamps)

        velocities = np.zeros((len(positions) - 1, 2))
        for i in range(len(positions) - 1):
            dt = times[i + 1] - times[i]
            if dt > 0:
                velocities[i] = (positions[i + 1] - positions[i]) / dt
            else:
                velocities[i] = velocities[i - 1] if i > 0 else np.zeros(2)

        accelerations = np.zeros((len(velocities) - 1, 2))
        for i in range(len(velocities) - 1):
            dt = times[i + 1] - times[i]
            if dt > 0:
                accelerations[i] = (velocities[i + 1] - velocities[i]) / dt
            else:
                accelerations[i] = accelerations[i - 1] if i > 0 else np.zeros(2)

        velocity = velocities[-1] if len(velocities) > 0 else np.zeros(2)
        speed = np.linalg.norm(velocity)

        peak_indices = np.where(np.abs(velocities[:-1, 1]) < 0.1)[0]
        if len(peak_indices) > 0 and len(accelerations) > peak_indices[0]:
            peak_acc = np.abs(accelerations[peak_indices[0], 1])
            if peak_acc > 0:
                mass = self.ode_solver.g / peak_acc
                mass = np.clip(mass, 0.2, 1.0)
            else:
                mass = 0.6
        else:
            mass = 0.6

        if len(velocities) > 5 and speed > 0.1:
            v_start = np.linalg.norm(velocities[0])
            v_end = speed
            t_elapsed = times[-1] - times[len(times) - len(velocities)]
            if t_elapsed > 0:
                deceleration = (v_start - v_end) / t_elapsed
                drag_coef = abs(mass * deceleration / (speed * speed))
                drag_coef = min(1.0, max(0.1, drag_coef))
            else:
                drag_coef = 0.1
        else:
            drag_coef = 0.1

        return {
            "position": positions[-1],
            "velocity": velocity,
            "speed": speed,
            "mass": mass,
            "drag_coefficient": drag_coef,
        }


class VideoProcessor:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.tracker = BallTracker()
        self.analyzer = MotionAnalyzer()

    def add_parameter_overlay(self, frame, params):
        if params is None:
            return frame

        text_lines = [
            f"Speed: {params['speed']:.2f} m/s",
            f"Mass: {params['mass']:.3f} kg",
            f"Drag Coef: {params['drag_coefficient']:.3f}",
            f"Vx: {params['velocity'][0]:.2f} m/s",
            f"Vy: {params['velocity'][1]:.2f} m/s",
        ]

        for i, line in enumerate(text_lines):
            cv2.putText(
                frame,
                line,
                (10, 30 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        return frame

    def process_video(self):
        time = 0

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            center, radius = self.tracker.detect_ball(frame)

            if center is not None:
                position = np.array(center) / self.tracker.pixels_per_meter

                params = self.analyzer.calculate_parameters(position, time)

                cv2.circle(frame, center, radius, (0, 255, 0), 2)
                cv2.circle(frame, center, 2, (0, 0, 255), -1)

                frame = self.add_parameter_overlay(frame, params)

            cv2.imshow("Ball Motion Analysis", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            time += 1.0 / self.fps

        self.cap.release()
        cv2.destroyAllWindows()


def main():
    video_path = "basketball_throw.mp4"
    try:
        processor = VideoProcessor(video_path)
        processor.process_video()
    except Exception as e:
        print(f"Error processing video: {e}")


if __name__ == "__main__":
    main()
