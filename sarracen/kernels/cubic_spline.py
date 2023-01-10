import numpy as np
from numba import jit

from ..kernels import BaseKernel


class CubicSplineKernel(BaseKernel):
    """An implementation of the Cubic Spline kernel"""

    @staticmethod
    def get_radius() -> float:
        return 2

    @staticmethod
    @jit(fastmath=True)
    def w(q: float, ndim: int):
        norm = 2 / 3 if (ndim == 1) else 10 / (7 * np.pi) if (ndim == 2) else 1 / np.pi

        return norm * ((1 - (3. / 2.) * q ** 2 + (3. / 4.) * q ** 3) * (0 <= q) * (q < 1)
                       + (1. / 4.) * (2 - q) ** 3 * (1 <= q) * (q < 2))
