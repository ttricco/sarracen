import numpy as np

from sarracen.kernels import BaseKernel


class QuinticSplineKernel(BaseKernel):
    """
    An implementation of the Quintic Spline kernel, in 1, 2, and 3 dimensions.
    """
    def __init__(self):
        super().__init__(3.0, 1/120, 7./(478*np.pi), 1./(120*np.pi), self._weight)

    @staticmethod
    def _weight(q):
        if q < 1:
            return (3 - q)**5 - 6*(2 - q)**5 + 15*(1 - q)**5
        elif q < 2:
            return (3 - q)**5 - 6*(2 - q)**5
        elif q < 3:
            return (3 - q)**5
        else:
            return 0
