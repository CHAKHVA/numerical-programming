from dataclasses import dataclass


@dataclass
class Shape:
    """Class to represent a target shape"""

    type: str
    position: tuple[float, float]
    radius: tuple[float, float] | float

    def __str__(self):
        x, y = self.position
        return f"{self.type} at ({x:.1f}, {y:.1f})"
