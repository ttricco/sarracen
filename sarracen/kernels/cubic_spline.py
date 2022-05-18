import numpy as np
from numba import jit

from sarracen.kernels import BaseKernel


class CubicSplineKernel(BaseKernel):
    """
    An implementation of the Cubic Spline kernel, in 1, 2, and 3 dimensions.
    """

    @staticmethod
    def get_radius() -> float:
        return 2

    @staticmethod
    @jit(fastmath=True)
    def w(q: float, ndim: int):
        norm = 2 / 3 if (ndim == 1) else 10 / (7 * np.pi) if (ndim == 2) else 1 / np.pi

        return norm * ((1 - (3. / 2.) * q ** 2 + (3. / 4.) * q ** 3) * (q < 1) + (1. / 4.) * (2 - q) ** 3 * (q < 2) * (q >= 1))
