"""pytest unit tests for kernel functionality."""
from typing import Callable
from pytest import approx, mark
from scipy.integrate import quad, dblquad, tplquad
import numpy as np

from sarracen.kernels import BaseKernel
from sarracen.kernels import CubicSplineKernel
from sarracen.kernels import QuarticSplineKernel
from sarracen.kernels import QuinticSplineKernel


def single_kernel(x: float, kernel: BaseKernel) -> float:
    return kernel.w(np.abs(x), 1)


def double_kernel(y: float, x: float, kernel: BaseKernel) -> float:
    # Utility function for double integrals in test_normalization
    return kernel.w(np.sqrt(x ** 2 + y ** 2), 2)


def triple_kernel(z: float, y: float, x: float, kernel: BaseKernel) -> float:
    # Utility function for triple integrals in test_normalization
    return kernel.w(np.sqrt(x ** 2 + y ** 2 + z ** 2), 3)


def double_column(y: float, x: float, column_func: Callable) -> float:
    return column_func(np.sqrt(x ** 2 + y ** 2), 0)


def test_cubicspline() -> None:
    kernel = CubicSplineKernel()

    # testing kernel values at q = 0
    # which should be equal to the normalization constants
    assert kernel.w(0, 1) == 2 / 3
    assert kernel.w(0, 2) == 10 / (7 * np.pi)
    assert kernel.w(0, 3) == 1 / np.pi

    # testing kernel values at q = 1
    assert kernel.w(1, 1) == 1 / 6
    assert kernel.w(1, 2) == 5 / (14 * np.pi)
    assert kernel.w(1, 3) == 1 / (4 * np.pi)

    # testing kernel values at q = 2
    assert kernel.w(2, 2) == 0
    assert kernel.w(10, 3) == 0


def test_quarticspline() -> None:
    kernel = QuarticSplineKernel()

    # unlike the cubic spline, these will NOT
    # be equal to the normalization constants.
    # kernel.w(0, d) = norm_d * 230/16
    assert kernel.w(0, 1) == approx(115 / 192)
    assert kernel.w(0, 2) == approx(1380 / (1199 * np.pi))
    assert kernel.w(0, 3) == approx(23 / (32 * np.pi))

    assert kernel.w(1, 1) == approx(19 / 96)
    assert kernel.w(1, 2) == approx(456 / (1199 * np.pi))
    assert kernel.w(1, 3) == approx(19 / (80 * np.pi))

    # these are equivalent to the normalization constants
    assert kernel.w(1.5, 1) == approx(1 / 24)
    assert kernel.w(1.5, 2) == approx(96 / (1199 * np.pi))
    assert kernel.w(1.5, 3) == approx(1 / (20 * np.pi))

    assert kernel.w(2.5, 2) == 0
    assert kernel.w(10, 3) == 0


def test_quinticspline() -> None:
    kernel = QuinticSplineKernel()

    # again, unlike the cubic spline, these will NOT
    # be equal to the normalization constants.
    # kernel.w(0, d) = norm_d * 66
    assert kernel.w(0, 1) == approx(11 / 20)
    assert kernel.w(0, 2) == approx(231 / (239 * np.pi))
    assert kernel.w(0, 3) == approx(11 / (20 * np.pi))

    assert kernel.w(1, 1) == approx(13 / 60)
    assert kernel.w(1, 2) == approx(91 / (239 * np.pi))
    assert kernel.w(1, 3) == approx(13 / (60 * np.pi))

    # these are equivalent to the normalization constants
    assert kernel.w(2, 1) == approx(1 / 120)
    assert kernel.w(2, 2) == approx(7 / (478 * np.pi))
    assert kernel.w(2, 3) == approx(1 / (120 * np.pi))

    assert kernel.w(3, 2) == 0
    assert kernel.w(10, 3) == 0


@mark.parametrize("kernel",
                  [CubicSplineKernel(),
                   QuarticSplineKernel(),
                   QuinticSplineKernel()])
def test_normalization(kernel: BaseKernel) -> None:

    # Since the three integrals below are only performed in positive space,
    # the resulting normalized values will not be equal to 1, rather 1/(2^dim).
    # This value represents the proportion of space in 1,2, and 3 dimensions
    # that has all positive coordinates.

    norm = quad(single_kernel, -kernel.get_radius(),
                kernel.get_radius(), tuple([kernel]))[0]
    assert approx(norm) == 1

    norm = dblquad(double_kernel, -kernel.get_radius(),
                   kernel.get_radius(), -kernel.get_radius(),
                   kernel.get_radius(), tuple([kernel]))[0]
    assert approx(norm) == 1

    norm = tplquad(triple_kernel, -kernel.get_radius(),
                   kernel.get_radius(), -kernel.get_radius(),
                   kernel.get_radius(), -kernel.get_radius(),
                   kernel.get_radius(), tuple([kernel]))[0]
    assert approx(norm) == 1


def test_cubic_column() -> None:
    kernel = CubicSplineKernel()
    column_kernel = kernel.get_column_kernel(10000)
    pts = np.linspace(0, kernel.get_radius(), 10000)
    # at q = 0, this integral is solvable analytically
    assert np.interp(0, pts, column_kernel) == approx(3 / (2 * np.pi))
    # numerically calculated values
    assert np.interp(0.5, pts, column_kernel) == approx(0.33875339978)
    assert np.interp(1, pts, column_kernel) == approx(0.111036060968)
    assert np.interp(1.5, pts, column_kernel) == approx(0.0114423169642)
    assert np.interp(2, pts, column_kernel) == 0
    assert np.interp(5, pts, column_kernel) == 0


def test_quartic_column() -> None:
    kernel = QuarticSplineKernel()
    column_kernel = kernel.get_column_kernel(10000)
    pts = np.linspace(0, kernel.get_radius(), 10000)

    # at q = 0, this integral is solvable analytically
    assert np.interp(0, pts, column_kernel) == approx(6 / (5 * np.pi))
    # numerically calculated values
    assert np.interp(0.5, pts, column_kernel) == approx(0.288815941868)
    assert np.interp(1, pts, column_kernel) == approx(0.120120735858)
    assert np.interp(1.5, pts, column_kernel) == approx(0.0233911861393)
    assert np.interp(2, pts, column_kernel) == approx(0.00116251851966)
    assert np.interp(2.5, pts, column_kernel) == 0
    assert np.interp(5, pts, column_kernel) == 0


def test_quintic_column() -> None:
    kernel = QuinticSplineKernel()
    column_kernel = kernel.get_column_kernel(10000)
    pts = np.linspace(0, kernel.get_radius(), 10000)

    # at q = 0, this integral is solvable analytically
    assert np.interp(0, pts, column_kernel) == approx(1 / np.pi)
    # numerically calculated values
    assert np.interp(0.5, pts, column_kernel) == approx(0.251567608959)
    assert np.interp(1, pts, column_kernel) == approx(0.121333261458)
    assert np.interp(1.5, pts, column_kernel) == approx(0.0328632154395)
    assert np.interp(2, pts, column_kernel) == approx(0.00403036583315)
    assert np.interp(2.5, pts, column_kernel) == approx(0.0000979416858548,
                                                        rel=1e-4)
    assert np.interp(3, pts, column_kernel) == 0
    assert np.interp(5, pts, column_kernel) == 0


@mark.parametrize("kernel",
                  [CubicSplineKernel(),
                   QuarticSplineKernel(),
                   QuinticSplineKernel()])
def test_normalized_column(kernel: BaseKernel) -> None:
    norm = dblquad(double_column, -kernel.get_radius(), kernel.get_radius(),
                   -kernel.get_radius(), kernel.get_radius(),
                   tuple([kernel.get_column_kernel_func(10000)]))[0]
    assert approx(norm) == 1


@mark.parametrize("kernel",
                  [CubicSplineKernel(),
                   QuarticSplineKernel(),
                   QuinticSplineKernel()])
def test_oob(kernel: BaseKernel) -> None:
    for dimensions in range(1, 3):
        assert kernel.w(-1, dimensions) == 0
        assert kernel.w(kernel.get_radius() + 1, dimensions) == 0

    column_func = kernel.get_column_kernel_func(1000)
    assert column_func(-1, 0) == column_func(0, 0)
    assert approx(column_func(kernel.get_radius() + 1, 0)) == 0
