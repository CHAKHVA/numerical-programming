import numpy as np
import pytest

from src.models.parameters import SturmLiouvilleParameters
from src.solvers.finite_difference import FiniteDifference


def test_finite_difference_initialization():
    params = SturmLiouvilleParameters(m=2, N=10)
    fd = FiniteDifference(params)

    assert fd.h == pytest.approx(np.pi / 18)
    assert len(fd.x) == 10


def test_coefficient_computation():
    params = SturmLiouvilleParameters(m=2, N=10)
    fd = FiniteDifference(params)

    p, q, w = fd._compute_coefficients(np.pi / 4)

    assert isinstance(p, float)
    assert isinstance(q, float)
    assert isinstance(w, float)


def test_matrix_dimensions():
    params = SturmLiouvilleParameters(m=2, N=10)
    fd = FiniteDifference(params)

    A, W = fd.build_matrices()

    assert A.shape == (8, 8)
    assert W.shape == (8, 8)
