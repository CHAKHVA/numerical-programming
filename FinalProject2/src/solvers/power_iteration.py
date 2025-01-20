from typing import List, Tuple

import numpy as np


class PowerIterationSolver:
    """Improved power iteration solver with shift-and-invert strategy."""

    def __init__(
        self, A: np.ndarray, W: np.ndarray, max_iter: int = 1000, tol: float = 1e-12
    ):
        self.A = A
        self.W = W
        self.max_iter = max_iter
        self.tol = tol

    def _shift_invert_iteration(
        self, sigma: float, v0: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Shift-and-invert iteration with improved stability."""
        A_shifted = self.A - sigma * self.W
        v = v0.copy()

        for _ in range(self.max_iter):
            # Solve linear system
            try:
                y = np.linalg.solve(A_shifted, self.W @ v)
            except np.linalg.LinAlgError:
                # Add small regularization if matrix is singular
                A_reg = A_shifted + np.eye(len(A_shifted)) * 1e-10
                y = np.linalg.solve(A_reg, self.W @ v)

            # Normalize
            y_norm = np.sqrt(np.sum(y * y))
            if y_norm < self.tol:
                break
            y = y / y_norm

            # Check convergence
            if np.allclose(y, v, rtol=self.tol) or np.allclose(y, -v, rtol=self.tol):
                break

            v = y

        # Compute Rayleigh quotient
        lambda_ = (v @ (self.A @ v)) / (v @ (self.W @ v))
        return lambda_, v

    def find_eigenvalues(
        self, num_eigenvalues: int
    ) -> Tuple[List[float], List[np.ndarray]]:
        """Find multiple eigenvalues using shifted power iteration."""
        eigenvalues = []
        eigenvectors = []
        n = len(self.A)

        # Use different shifts to find different eigenvalues
        shifts = np.linspace(1, 100, num_eigenvalues)  # Adjusted shift range

        for i in range(num_eigenvalues):
            # Generate random initial vector
            v0 = np.random.rand(n)

            # Make v0 orthogonal to previous eigenvectors
            for v in eigenvectors:
                v0 = v0 - np.dot(v0, v) * v

            # Normalize v0
            v0 = v0 / np.sqrt(np.sum(v0 * v0))

            # Find eigenvalue and eigenvector
            lambda_, v = self._shift_invert_iteration(shifts[i], v0)

            eigenvalues.append(lambda_)
            eigenvectors.append(v)

        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)
        eigenvalues = [eigenvalues[i] for i in idx]
        eigenvectors = [eigenvectors[i] for i in idx]

        return eigenvalues, eigenvectors
