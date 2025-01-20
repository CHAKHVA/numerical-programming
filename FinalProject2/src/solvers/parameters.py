from dataclasses import dataclass


@dataclass
class SturmLiouvilleParameters:
    m: float
    N: int
    num_eigenvalues: int = 8
    epsilon: float = 1e-10
