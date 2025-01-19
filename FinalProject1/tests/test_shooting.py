import numpy as np
import matplotlib.pyplot as plt
from src.trajectory.params import TrajectoryParams
from src.trajectory.calculator import TrajectoryCalculator
from src.trajectory.shooting import ShootingMethod


def test_trajectory_and_shooting(start, target):
    # Test case parameters
    start_pos = start  # Starting at origin
    target_pos = target  # Target at (100, 50)

    # Create shooting method instance and find parameters
    shooter = ShootingMethod(start_pos, target_pos)
    v0, theta = shooter.find_parameters()

    print(f"Found parameters:")
    print(f"Initial velocity: {v0:.2f} m/s")
    print(f"Angle: {np.degrees(theta):.2f} degrees")

    # Create trajectory with found parameters
    params = TrajectoryParams(
        start_pos=start_pos,
        target_pos=target_pos,
        initial_velocity=v0,
        angle=theta,
        gravity=9.81,
        time_step=0.01,
        max_time=10.0,
        tolerance=0.1
    )

    calculator = TrajectoryCalculator(params)
    x, y = calculator.calculate_trajectory()
    hits_target = calculator.hits_target()

    # Plot results
    plt.figure(figsize=(12, 8))

    # Plot trajectory
    plt.plot(x, y, 'b-', label='Trajectory')

    # Plot start and target points
    plt.plot(start_pos[0], start_pos[1], 'go', markersize=10, label='Start')
    plt.plot(target_pos[0], target_pos[1], 'ro', markersize=10, label='Target')

    # Draw target tolerance circle
    target_circle = plt.Circle(target_pos, params.tolerance, color='r', fill=False)
    plt.gca().add_patch(target_circle)

    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.title(f'Projectile Motion\nv0={v0:.2f} m/s, θ={np.degrees(theta):.2f}°\nHits target: {hits_target}')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')

    plt.show()


if __name__ == "__main__":
    test_cases = [
        ((0, 0), (100, 50)),  # Upward shot
        ((0, 0), (100, 0)),  # Flat shot
        ((0, 50), (100, 0)),  # Downward shot
        ((0, 0), (50, 100)),  # High shot
    ]

    for start, target in test_cases:
        print(f"\nTesting shot from {start} to {target}")
        test_trajectory_and_shooting(start, target)