import numpy as np
import pytest

from src.models.parameters import SturmLiouvilleParameters
from src.solvers.finite_difference import FiniteDifference
from src.solvers.power_iteration import PowerIterationSolver


def test_full_solution():
    # Test the full solution pipeline
    params = SturmLiouvilleParameters(m=2, N=20, num_eigenvalues=3)

    # Create finite difference discretization
    fd = FiniteDifference(params)
    A, W = fd.build_matrices()

    # Solve eigenvalue problem
    solver = PowerIterationSolver(A, W)
    eigenvalues, eigenvectors = solver.find_eigenvalues(params.num_eigenvalues)

    # Basic checks
    assert len(eigenvalues) == 3
    assert all(isinstance(ev, float) for ev in eigenvalues)
    assert all(ev > 0 for ev in eigenvalues)  # Physical constraint
    assert len(eigenvectors) == 3
