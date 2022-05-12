from typing import Callable


class BaseKernel:
    """A generic kernel used for data interpolation."""
    def __init__(self, radkernel: float, ndims: int, cnormk: float, wfunc: Callable):
        if ndims < 1 or ndims > 3:
            raise ValueError('Invalid Number of Dimensions!')

        self._radkernel = radkernel
        self._ndims = ndims
        self._cnormk = cnormk
        self._wfunc = wfunc

    def w(self, q: float) -> float:
        """
        The dimensionless part of this kernel at a specific value of q.
        :param q: The value of q to evaluate this kernel at.
        :return:
        """

        return self._cnormk * self._wfunc(q)

    @property
    def radkernel(self):
        return self._radkernel

    @radkernel.getter
    def radkernel(self):
        return self._radkernel

    @property
    def cnormk(self):
        return self._cnormk

    @cnormk.getter
    def cnormk(self):
        return self._cnormk

    @property
    def ndims(self):
        return self._ndims

    @ndims.getter
    def ndims(self):
        return self._ndims
