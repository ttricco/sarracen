import numpy as np

from sarracen.kernels import BaseKernel


class CubicSplineKernel(BaseKernel):
    """
    An implementation of the Cubic Spline kernel, in 1, 2, and 3 dimensions.
    """

    def __init__(self, ndims: int):
        norm = 2 / 3 if (ndims == 1) else \
                10 / (7 * np.pi) if (ndims == 2) else \
                1 / np.pi

        super().__init__(2.0, ndims, norm, self._weight)

    @staticmethod
    def _weight(q):
        if q < 1:
            return 1 - (3. / 2.) * q ** 2 + (3. / 4.) * q ** 3
        elif q < 2:
            return (1. / 4.) * (2 - q) ** 3
        else:
            return 0
