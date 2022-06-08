from typing import Callable

import numpy as np
from numba import jit, njit
from numba.core.registry import CPUDispatcher
from scipy.integrate import quad


class BaseKernel:

    def __init__(self):
        self._ckernel_func_cache = None
        self._ckernel_func_samples = 0

    """A generic kernel used for data interpolation."""
    @staticmethod
    def get_radius() -> float:
        """Get the smoothing radius of this kernel."""
        return 1

    @staticmethod
    def w(q: float, dim: int) -> float:
        """ Get the normalized weight of this kernel.

        Parameters
        ----------
        q : float
            The value to evaluate this kernel at.
        dim : {1, 2, 3}
            The number of dimensions to normalize the kernel value for.

        Returns
        -------
        float
            The normalized kernel weight at `q`.
        """

        return 1

    def get_column_kernel(self, samples: int) -> np.ndarray:
        """ Generate a 2D column kernel approximation, by integrating a given 3D kernel over the z-axis.

        Parameters
        ----------
        samples: int
            Number of sample points to calculate when approximating the kernel.

        Returns
        -------
            A ndarray of length (samples), containing the kernel approximation.

        Examples
        --------
        Use np.linspace and np.interp to use this column kernel approximation:
            np.interp(q, np.linspace(0, kernel.get_radius(), samples), column_kernel)
        """
        results = []
        for sample in np.linspace(0, self.get_radius(), samples):
            results.append(2 * quad(self._int_func,
                                    a=0,
                                    b=np.sqrt(self.get_radius() ** 2 - sample ** 2),
                                    args=(sample, self.w))[0])

        return np.array(results)

    def get_column_kernel_func(self, samples) -> CPUDispatcher:
        if self._ckernel_func_cache is not None and self._ckernel_func_samples == samples:
            return self._ckernel_func_cache
        column_kernel = self.get_column_kernel(samples)
        radius = self.get_radius()

        @njit(fastmath=True)
        def func(q, dim):
            return np.interp(q, np.linspace(0, radius, samples), column_kernel)

        self._ckernel_func_cache = func
        self._ckernel_func_samples = samples
        return func

    # Internal function for performing the integral in _get_column_kernel()
    @staticmethod
    @njit(fastmath=True)
    def _int_func(q, a, wfunc):
        return wfunc(np.sqrt(q ** 2 + a ** 2), 3)
