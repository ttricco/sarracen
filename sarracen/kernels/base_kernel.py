class BaseKernel:
    """A generic kernel used for data interpolation."""
    def __init__(self, radkernel, cnormk1d, cnormk2d, cnormk3d, wfunc):
        self._radkernel = radkernel
        self._cnormk1d = cnormk1d
        self._cnormk2d = cnormk2d
        self._cnormk3d = cnormk3d
        self._wfunc = wfunc

    def w(self, q, dimension):
        """
        The dimensionless part of this kernel at a specific value of q.
        :param q: The value of q to evaluate this kernel at.
        :param dimension: The number of dimensions (for normalization).
        :return:
        """
        norm = self._cnormk1d if (dimension == 1) else \
            self._cnormk2d if (dimension == 2) else \
            self._cnormk3d

        return norm * self._wfunc(q)

    @property
    def radkernel(self):
        return self._radkernel

    @radkernel.getter
    def radkernel(self):
        return self._radkernel

    @property
    def cnormk1d(self):
        return self._cnormk1d

    @cnormk1d.getter
    def cnormk1d(self):
        return self._cnormk1d

    @property
    def cnormk2d(self):
        return self._cnormk2d

    @cnormk2d.getter
    def cnormk2d(self):
        return self._cnormk2d

    @property
    def cnormk3d(self):
        return self._cnormk3d

    @cnormk3d.getter
    def cnormk3d(self):
        return self._cnormk3d
