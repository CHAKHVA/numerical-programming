from dataclasses import dataclass


@dataclass
class TrajectoryParams:
    """Parameters for trajectory calculation."""
    start_pos: tuple[float, float]
    target_pos: tuple[float, float]
    initial_velocity: float
    angle: float
    gravity: float = 9.81
    time_step: float = 0.01
    max_time: float = 10.0
    tolerance: float = 0.1
