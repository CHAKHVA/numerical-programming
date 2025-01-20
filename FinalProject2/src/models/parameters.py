from dataclasses import dataclass


@dataclass
class SturmLiouvilleParameters:
    """Parameters for the Sturm-Liouville problem."""

    m: float
    N: int
    num_eigenvalues: int = 8
    epsilon: float = 1e-10
