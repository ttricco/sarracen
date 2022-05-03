from sarracen.kernels import CubicSplineKernel
from math import pi


class TestKernels:
    def test_cubicspline(self):
        kernel = CubicSplineKernel()

        # testing kernel values at q = 0
        # which should be equal to the normalization constants
        assert kernel.value(0, 1) == 2/3
        assert kernel.value(0, 2) == 10 / (7*pi)
        assert kernel.value(0, 3) == 1 / pi

        # testing kernel values at q = 1
        assert kernel.value(1, 1) == 1/6
        assert kernel.value(1, 2) == 5 / (14 * pi)
        assert kernel.value(1, 3) == 1 / (4 * pi)

        # testing kernel values at q = 2
        assert kernel.value(2, 2) == 0
        assert kernel.value(10, 3) == 0
