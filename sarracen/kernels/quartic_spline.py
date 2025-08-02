import numpy as np
from numba import njit

from ..kernels import BaseKernel


class QuarticSplineKernel(BaseKernel):
    """An implementation of the Quartic Spline kernel."""

    @staticmethod
    def get_radius() -> float:
        return 2.5

    @staticmethod
    @njit(fastmath=True)
    def w(q: float, ndim: int) -> float:
        norm = 1 / 24 if (ndim == 1) \
            else 96 / (1199 * np.pi) if (ndim == 2) \
            else 1 / (20 * np.pi)

        return norm * ((2.5 - q)**4 * (q < 2.5)
                       - 5 * (1.5 - q)**4 * (q < 1.5)
                       + 10 * (0.5 - q)**4 * (q < 0.5)) * (0 <= q)
