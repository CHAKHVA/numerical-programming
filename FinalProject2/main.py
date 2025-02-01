"""Main execution file for ball trajectory simulation."""

import numpy as np

from animator import TrajectoryAnimator
from ball_detector import BallDetector
from constants import DEFAULT_VIDEO_PATH, SHOOTER_X_RANGE, SHOOTER_Y_RANGE
from trajectory_predictor import TrajectoryPredictor


def main():
    # Extract ball positions from video
    positions = BallDetector.extract_positions(DEFAULT_VIDEO_PATH, display=True)

    # Create predictors for both methods
    predictor_rk4 = TrajectoryPredictor(solver_type="rk4")
    predictor_euler = TrajectoryPredictor(solver_type="euler")

    # Reconstruct trajectories
    points_rk4, velocities_rk4 = predictor_rk4.reconstruct_trajectory(positions)
    points_euler, velocities_euler = predictor_euler.reconstruct_trajectory(positions)

    # Choose random shooter position
    shooter_pos = (
        np.random.uniform(SHOOTER_X_RANGE[0], SHOOTER_X_RANGE[1]),
        np.random.uniform(SHOOTER_Y_RANGE[0], SHOOTER_Y_RANGE[1]),
    )

    # Simulate interceptors
    intercept_traj_rk4, intercept_vel_rk4 = predictor_rk4.shoot_interceptor(
        points_rk4, velocities_rk4, shooter_pos
    )
    intercept_traj_euler, intercept_vel_euler = predictor_euler.shoot_interceptor(
        points_euler, velocities_euler, shooter_pos
    )

    # Create animations
    print("\nClose the RK4 animation window to see the Euler method animation.")
    TrajectoryAnimator.create_animation(
        points_rk4, intercept_traj_rk4, velocities_rk4, intercept_vel_rk4, method="RK4"
    )
    TrajectoryAnimator.create_animation(
        points_euler,
        intercept_traj_euler,
        velocities_euler,
        intercept_vel_euler,
        method="Euler",
    )


if __name__ == "__main__":
    main()
