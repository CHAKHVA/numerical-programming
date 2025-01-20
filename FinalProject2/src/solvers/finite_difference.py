import math

import numpy as np

from .parameters import SturmLiouvilleParameters


class FiniteDifference:
    def __init__(self, params: SturmLiouvilleParameters):
        self.params = params
        self.h = (math.pi / 2) / (params.N - 1)
        self.x = np.linspace(0, math.pi / 2, params.N)

    def _safe_sin(self, x: float) -> float:
        if abs(x) < self.params.epsilon:
            return x + self.params.epsilon
        return math.sin(x)

    def _compute_coefficients(self, x: float) -> tuple[float, float, float]:
        cos_x = math.cos(x)
        sin_x = self._safe_sin(x)

        # Coefficient of u''
        p = -0.5 * cos_x**4

        # Coefficient of u'
        q = -0.5 * cos_x**3 * math.cos(2 * x) / sin_x

        # Weight function w(x)
        w = sin_x

        return p, q, w

    def build_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        N = self.params.N
        A = np.zeros((N - 2, N - 2))
        W = np.eye(N - 2)

        # Build matrices for interior points
        for i in range(N - 2):
            x = self.x[i + 1]
            p, q, w = self._compute_coefficients(x)

            # Central difference coefficients
            if i > 0:
                A[i, i - 1] = p / self.h**2 - q / (2 * self.h)
            A[i, i] = -2 * p / self.h**2 + w
            if i < N - 3:
                A[i, i + 1] = p / self.h**2 + q / (2 * self.h)

        return A, W
