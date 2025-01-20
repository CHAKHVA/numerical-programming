import numpy as np
import pytest

from src.solvers.power_iteration import PowerIterationSolver


def test_power_iteration_simple_case():
    # Create a simple test matrix with known eigenvalues
    A = np.array([[2, 0], [0, 1]])
    W = np.eye(2)

    solver = PowerIterationSolver(A, W)
    eigenvalues, eigenvectors = solver.find_eigenvalues(2)

    assert len(eigenvalues) == 2
    assert len(eigenvectors) == 2
    assert eigenvalues[0] == pytest.approx(1.0, rel=1e-6)
    assert eigenvalues[1] == pytest.approx(2.0, rel=1e-6)


def test_orthogonality():
    A = np.array([[2, 1], [1, 2]])
    W = np.eye(2)

    solver = PowerIterationSolver(A, W)
    _, eigenvectors = solver.find_eigenvalues(2)

    # Check if eigenvectors are orthogonal
    dot_product = np.abs(np.dot(eigenvectors[0], eigenvectors[1]))
    assert dot_product == pytest.approx(0.0, abs=1e-6)
