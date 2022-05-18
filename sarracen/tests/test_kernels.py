from pytest import approx
from scipy.integrate import quad, dblquad, tplquad
import numpy as np

from sarracen.kernels import CubicSplineKernel, QuarticSplineKernel, QuinticSplineKernel


def single_kernel(x, kernel):
    return kernel.w(x, 1)


def double_kernel(y, x, kernel):
    # Utility function for double integrals in test_normalization
    return kernel.w(np.sqrt(x ** 2 + y ** 2), 2)


def triple_kernel(z, y, x, kernel):
    # Utility function for triple integrals in test_normalization
    return kernel.w(np.sqrt(x ** 2 + y ** 2 + z ** 2), 3)


class TestKernels:
    def test_cubicspline(self):
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

    def test_quarticspline(self):
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

    def test_quinticspline(self):
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

    def test_normalization(self):

        # Since the three integrals below are only performed in positive space, the
        # resulting normalized values will not be equal to 1, rather 1/(2^dim). This
        # value represents the proportion of space in 1,2, and 3 dimensions that
        # has all positive coordinates.

        for kernel in [CubicSplineKernel(), QuarticSplineKernel(), QuinticSplineKernel()]:
            norm = quad(single_kernel, 0, kernel.get_radius(), kernel)[0]
            assert approx(norm) == 0.5  # positive space -> a half of 1D space

        for kernel in [CubicSplineKernel(), QuarticSplineKernel(), QuinticSplineKernel()]:
            norm = dblquad(double_kernel, 0, kernel.get_radius(), 0, kernel.get_radius(), [kernel])[0]
            assert approx(norm) == 0.25  # positive space -> a fourth of 2D space

        for kernel in [CubicSplineKernel(), QuarticSplineKernel(), QuinticSplineKernel()]:
            norm = tplquad(triple_kernel, 0, kernel.get_radius(), 0, kernel.get_radius(), 0, kernel.get_radius(), [kernel])[0]
            assert approx(norm) == 0.125  # positive space -> an eight of 3D space
