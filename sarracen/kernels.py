import numpy as np


class Kernel:
    """A generic kernel used for data interpolation."""
    def __init__(self, radkernel, cnormk1D, cnormk2D, cnormk3D, wfunc):
        self._radkernel = radkernel
        self._cnormk1D = cnormk1D
        self._cnormk2D = cnormk2D
        self._cnormk3D = cnormk3D
        self._wfunc = wfunc

    def w(self, q, dimension):
        """
        The dimensionless part of this kernel at a specific value of q.
        :param q: The value of q to evaluate this kernel at.
        :param dimension: The number of dimensions (for normalization)
        :return:
        """
        norm = self._cnormk1D if (dimension == 1) else \
            self._cnormk2D if (dimension == 2) else \
            self._cnormk3D

        return norm * self._wfunc(q)


class CubicSplineKernel(Kernel):
    """
    An implementation of the Cubic Spline kernel, in 1, 2, and 3 dimensions.
    """
    def __init__(self):
        super().__init__(2.0, 2./3., 10./(7.*np.pi), 1./np.pi, self._weight)

    @staticmethod
    def _weight(q):
        if q < 1:
            return 1 - (3./2.) * q**2 + (3./4.) * q**3
        elif q < 2:
            return (1./4.) * (2 - q)**3
        else:
            return 0
