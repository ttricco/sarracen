from typing import Callable

import numpy as np
from numba import jit
from scipy.integrate import quad


class BaseKernel:
    """A generic kernel used for data interpolation."""
    @staticmethod
    def get_radius() -> float:
        return 1

    @staticmethod
    def weight(q: float, dim: int) -> float:
        """
        The dimensionless part of this kernel at a specific value of q.
        :param q: The value of q to evaluate this kernel at.
        :return:
        """

        return 1

    def get_column_kernel(self, samples):
        """
        Generate a 2D column kernel approximation, by integrating a given 3D kernel over the z-axis.
        :param kernel: The 3D kernel to integrate over.
        :param samples: The number of samples to take of the integral.
        :return: A ndarray of length (samples), containing the kernel approximation.
        """
        results = []
        for sample in np.linspace(0, self.get_radius(), samples):
            results.append(2 * quad(self._int_func,
                                    a=0,
                                    b=np.sqrt(self.get_radius() ** 2 - sample ** 2),
                                    args=(sample, self.weight))[0])

        return np.array(results)

    # Internal function for performing the integral in _get_column_kernel()
    @staticmethod
    @jit(fastmath=True)
    def _int_func(q, a, wfunc):
        return wfunc(np.sqrt(q ** 2 + a ** 2), 3)
