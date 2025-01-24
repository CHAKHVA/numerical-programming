from dataclasses import dataclass


@dataclass
class TrajectoryParams:
    start_pos: tuple[float, float]
    target_pos: tuple[float, float]
    initial_velocity: float
    angle: float
    gravity: float = 9.81
    time_step: float = 0.01
    max_time: float = 2.0
    tolerance: float = 0.01
