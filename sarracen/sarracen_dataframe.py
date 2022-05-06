from pandas import DataFrame, Series
import numpy as np

from sarracen.render import render
from sarracen.kernels import CubicSplineKernel, BaseKernel

class SarracenDataFrame(DataFrame):
    _metadata = ['_params', '_units', '_xcol', '_ycol']

    def __init__(self, data=None, params=None, *args, **kwargs):

        # call pandas DataFrame contructor
        super().__init__(data, *args, **kwargs)

        self._params = None
        self.params = params

        self._units = None
        self.units = Series([np.nan for i in range(len(self.columns))])

        # First look for 'x', then 'rx', and then fallback to the first column.
        if 'x' in data.columns:
            self._xcol = 'x'
        elif 'rx' in data.columns:
            self._xcol = 'rx'
        else:
            self._xcol = data.columns[0]

        # First look for 'y', then 'ry', and then fallback to the second column.
        if 'y' in data.columns:
            self._ycol = 'y'
        elif 'ry' in data.columns:
            self._ycol = 'ry'
        else:
            self._ycol = data.columns[1]

    def render(self,
               target: str,
               x: str = None,
               y: str = None,
               kernel: BaseKernel = CubicSplineKernel(2),
               xmin: float = None,
               ymin: float = None,
               xmax: float = None,
               ymax: float = None,
               pixcountx: int = 256,
               pixcounty: int = None) -> ('Figure', 'Axes'):
        """
        Render the data within this dataframe to a 2D matplotlib object, using 2D SPH Interpolation of the target
        variable.
        :param target: The variable to interpolate over. [Required]
        :param x: The positional x variable.
        :param y: The positional y variable.
        :param kernel: The smoothing kernel to use for interpolation.
        :param xmin: The minimum bound in the x-direction.
        :param ymin: The minimum bound in the y-direction.
        :param xmax: The maximum bound in the x-direction.
        :param ymax: The maximum bound in the y-direction.
        :param pixcountx: The number of pixels in the x-direction.
        :param pixcounty: The number of pixels in the y-direction.
        :return: The completed plot.
        """

        return render(self, target, x, y, kernel, xmin, ymin, xmax, ymax, pixcountx, pixcounty)

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, new_params):
        if new_params is None:
            self._params = None
            return
        if not type(new_params) is dict:
            raise TypeError("Parameters not a dictionary")
        self._params = new_params

    @params.getter
    def params(self):
        return self._params

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, new_units):
        self._units = new_units

    @units.getter
    def units(self):
        return self._units

    @property
    def xcol(self):
        return self._xcol

    @xcol.getter
    def xcol(self):
        return self._xcol

    @property
    def ycol(self):
        return self._ycol

    @ycol.getter
    def ycol(self):
        return self._ycol
