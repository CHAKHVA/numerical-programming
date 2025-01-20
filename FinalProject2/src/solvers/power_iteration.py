import numpy as np


class PowerIterationSolver:
    def __init__(
        self, A: np.ndarray, W: np.ndarray, max_iter: int = 1000, tol: float = 1e-12
    ):
        self.A = A
        self.W = W
        self.max_iter = max_iter
        self.tol = tol

    def _shift_invert_iteration(
        self, sigma: float, v0: np.ndarray
    ) -> tuple[float, np.ndarray]:
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
    ) -> tuple[list[float], list[np.ndarray]]:
        eigenvalues = []
        eigenvectors = []
        n = len(self.A)

        # Use different shifts to find different eigenvalues
        # Improved shift strategy using geometric progression
        shifts = np.geomspace(
            1, 200, num_eigenvalues
        )  # Changed from linear to geometric spacing

        for i in range(num_eigenvalues):
            # Generate random initial vector
            v0 = np.random.rand(n)

            # Make v0 orthogonal to previous eigenvectors
            for v in eigenvectors:
                v0 = v0 - (v @ (self.W @ v0)) * v  # Modified to use W-orthogonality

            # Normalize v0
            v0 = v0 / np.sqrt(v0 @ (self.W @ v0))  # W-normalized

            # Find eigenvalue and eigenvector
            lambda_, v = self._shift_invert_iteration(shifts[i], v0)

            eigenvalues.append(lambda_)
            eigenvectors.append(v)

            # Check orthogonality with previous eigenvectors
            if i > 0:
                self._check_orthogonality(v, eigenvectors[:-1])

        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)
        eigenvalues = [eigenvalues[i] for i in idx]
        eigenvectors = [eigenvectors[i] for i in idx]

        return eigenvalues, eigenvectors

    def _check_orthogonality(
        self, v: np.ndarray, previous_vectors: list[np.ndarray]
    ) -> None:
        for i, prev_v in enumerate(previous_vectors):
            inner_product = abs(v @ (self.W @ prev_v))
            if inner_product > 1e-8:
                print(
                    f"Warning: Eigenvector {len(previous_vectors) + 1} not orthogonal to eigenvector {i + 1}"
                )
                print(f"Inner product: {inner_product:.2e}")
