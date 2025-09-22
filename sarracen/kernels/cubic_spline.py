import numpy as np
from numba import njit

from ..kernels import BaseKernel


class CubicSplineKernel(BaseKernel):
    """An implementation of the Cubic Spline kernel"""

    @staticmethod
    def get_radius() -> float:
        return 2

    @staticmethod
    @njit(fastmath=True)
    def w(q: float, ndim: int) -> float:
        norm = 2 / 3 if (ndim == 1) \
            else 10 / (7 * np.pi) if (ndim == 2) \
            else 1 / np.pi

        return norm * ((1 - 1.5 * q**2 + 0.75 * q**3) * (0 <= q) * (q < 1)
                       + 0.25 * (2 - q)**3 * (1 <= q) * (q < 2))
