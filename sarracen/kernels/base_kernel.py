from typing import Callable


class BaseKernel:
    """A generic kernel used for data interpolation."""
    @staticmethod
    def get_radius() -> float:
        return 1

    @staticmethod
    def w(q: float, dim: int) -> float:
        """
        The dimensionless part of this kernel at a specific value of q.
        :param q: The value of q to evaluate this kernel at.
        :return:
        """

        return 1
