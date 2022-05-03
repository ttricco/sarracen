from sarracen.kernels import CubicSplineKernel
import numpy as np


class TestKernels:
    def test_cubicspline(self):
        kernel = CubicSplineKernel()

        # testing kernel values at q = 0
        # which should be equal to the normalization constants
        assert kernel.w(0, 1) == 2/3
        assert kernel.w(0, 2) == 10 / (7*np.pi)
        assert kernel.w(0, 3) == 1 / np.pi

        # testing kernel values at q = 1
        assert kernel.w(1, 1) == 1/6
        assert kernel.w(1, 2) == 5 / (14 * np.pi)
        assert kernel.w(1, 3) == 1 / (4 * np.pi)

        # testing kernel values at q = 2
        assert kernel.w(2, 2) == 0
        assert kernel.w(10, 3) == 0
