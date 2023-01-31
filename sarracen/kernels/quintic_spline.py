import numpy as np
from numba import jit

from ..kernels import BaseKernel


class QuinticSplineKernel(BaseKernel):
    """An implementation of the Quintic Spline kernel."""

    @staticmethod
    def get_radius() -> float:
        return 3

    @staticmethod
    @jit(fastmath=True)
    def w(q: float, ndim: int):
        norm = 1 / 120 if (ndim == 1) else \
            7 / (478 * np.pi) if (ndim == 2) else \
            1 / (120 * np.pi)

        return norm * ((3 - q) ** 5 * (q < 3)
                       - 6 * (2 - q) ** 5 * (q < 2)
                       + 15 * (1 - q) ** 5 * (q < 1)) * (0 <= q)
