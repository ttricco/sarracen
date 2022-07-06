import math

import numpy as np
from numba import njit, prange


class BaseKernel:
    """A generic kernel used for data interpolation."""

    def __init__(self):
        self._ckernel_func_cache = None
        self._column_cache = None

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

    def get_column_kernel(self, samples: int = 1000) -> np.ndarray:
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
        if samples == 1000 and self._column_cache is not None:
            return self._column_cache

        c_kernel = BaseKernel._int_func(self.get_radius(), samples, self.w)

        if samples == 1000:
            self._column_cache = c_kernel

        return c_kernel

    def get_column_kernel_func(self, samples):
        """ Generate a numba-accelerated column kernel function.

        Creates a numba-accelerated function for column kernel weights. This function
        can be utilized similarly to kernel.w.

        Parameters
        ----------
        samples: int
            Number of sample points to calculate when approximating the kernel.

        Returns
        -------
        A numba-accelerated weight function.
        """
        if self._ckernel_func_cache is not None and samples == 1000:
            return self._ckernel_func_cache
        column_kernel = self.get_column_kernel(samples)
        radius = self.get_radius()

        @njit(fastmath=True)
        def func(q, dim):
            # using np.linspace() would break compatibility with the GPU backend,
            # so the calculation here is performed manually.
            wab_index = q * (samples - 1) / radius
            index = min(max(0, int(math.floor(wab_index))), samples - 1)
            index1 = min(max(0, int(math.ceil(wab_index))), samples - 1)
            t = wab_index - index
            return column_kernel[index] * (1 - t) + column_kernel[index1] * t

        return func

    # Internal function for performing the integral in _get_column_kernel()
    @staticmethod
    @njit(fastmath=True, parallel=True)
    def _int_func(radius, samples, wfunc):
        result = np.zeros(samples)

        for i in prange(samples):
            q_xy = radius * i / (samples - 1)
            bounds = np.sqrt(radius ** 2 - q_xy ** 2)
            q_z = np.linspace(0, bounds, samples)
            q = np.sqrt(q_xy ** 2 + q_z ** 2)
            y = wfunc(q, 3)
            result[i] = 2 * np.trapz(y, x=q_z)

        return result
