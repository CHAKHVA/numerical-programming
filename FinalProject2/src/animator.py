"""Animation of ball trajectories."""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from constants import (
    ANIMATION_INTERVAL,
    ARROW_SCALE,
    POINT_SIZE,
    TRAIL_ALPHA,
    VELOCITY_SCALE,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
)


class TrajectoryAnimator:
    @staticmethod
    def create_animation(
        original_trajectory: np.ndarray,
        intercept_trajectory: np.ndarray,
        original_velocities: np.ndarray,
        intercept_velocities: np.ndarray,
        method: str = "RK4",
    ):
        """Create matplotlib animation showing both trajectories."""
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(0, VIDEO_WIDTH)
        ax.set_ylim(VIDEO_HEIGHT, 0)  # Inverted y-axis to match video coordinates
        ax.grid(True)
        ax.set_title(f"Ball Trajectory Simulation ({method})")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")

        # Initialize plot elements
        (original_trail,) = ax.plot(
            [], [], "b-", alpha=TRAIL_ALPHA, label="Original Ball Trail"
        )
        (intercept_trail,) = ax.plot(
            [], [], "r-", alpha=TRAIL_ALPHA, label="Interceptor Trail"
        )
        (original_point,) = ax.plot(
            [], [], "bo", markersize=POINT_SIZE, label="Original Ball"
        )
        (intercept_point,) = ax.plot(
            [], [], "ro", markersize=POINT_SIZE, label="Interceptor"
        )

        original_arrow = ax.quiver([], [], [], [], color="blue", scale=ARROW_SCALE)
        intercept_arrow = ax.quiver([], [], [], [], color="red", scale=ARROW_SCALE)

        time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes)
        velocity_text = ax.text(0.02, 0.94, "", transform=ax.transAxes)

        def init():
            """Initialize animation."""
            original_trail.set_data([], [])
            intercept_trail.set_data([], [])
            original_point.set_data([], [])
            intercept_point.set_data([], [])
            time_text.set_text("")
            velocity_text.set_text("")
            ax.legend()
            return (
                original_trail,
                intercept_trail,
                original_point,
                intercept_point,
                time_text,
            )

        def animate(frame):
            """Animation function called for each frame."""
            # Update original ball
            if frame < len(original_trajectory):
                orig_trail_x = original_trajectory[: frame + 1, 0]
                orig_trail_y = original_trajectory[: frame + 1, 1]
                original_trail.set_data(orig_trail_x, orig_trail_y)
                original_point.set_data(
                    [original_trajectory[frame, 0]], [original_trajectory[frame, 1]]
                )

                original_arrow.set_offsets(original_trajectory[frame])
                original_arrow.set_UVC(
                    original_velocities[frame, 0] * VELOCITY_SCALE,
                    original_velocities[frame, 1] * VELOCITY_SCALE,
                )

                v_mag = np.linalg.norm(original_velocities[frame])
                velocity_text.set_text(f"Velocity: {v_mag:.1f} pixels/s")

            # Update interceptor
            if frame < len(intercept_trajectory):
                int_trail_x = intercept_trajectory[: frame + 1, 0]
                int_trail_y = intercept_trajectory[: frame + 1, 1]
                intercept_trail.set_data(int_trail_x, int_trail_y)
                intercept_point.set_data(
                    [intercept_trajectory[frame, 0]], [intercept_trajectory[frame, 1]]
                )

                intercept_arrow.set_offsets(intercept_trajectory[frame])
                intercept_arrow.set_UVC(
                    intercept_velocities[frame, 0] * VELOCITY_SCALE,
                    intercept_velocities[frame, 1] * VELOCITY_SCALE,
                )

            time_text.set_text(f"Frame: {frame}")

            return (
                original_trail,
                intercept_trail,
                original_point,
                intercept_point,
                original_arrow,
                intercept_arrow,
                time_text,
                velocity_text,
            )

        frames = max(len(original_trajectory), len(intercept_trajectory))
        anim = animation.FuncAnimation(
            fig,
            animate,
            init_func=init,
            frames=frames,
            interval=ANIMATION_INTERVAL,
            blit=True,
        )

        plt.show()
