from typing import Union

from matplotlib.colors import Colormap
from pandas import DataFrame, Series
import numpy as np

from sarracen.render import render
from sarracen.kernels import CubicSplineKernel, BaseKernel


def _snap(value: float):
    """
    Return a number snapped to the nearest integer, with 1e-4 tolerance.
    :param value: The number to snap.
    :return: An integer if a close integer is detected, otherwise return 'value'.
    """
    if np.isclose(value, np.rint(value), atol=1e-4):
        return np.rint(value)
    else:
        return value


class SarracenDataFrame(DataFrame):
    _metadata = ['_params', '_units', '_xcol', '_ycol', '_zcol', '_xmin', '_ymin', '_zmin', '_xmax', '_ymax', '_zmax']

    def __init__(self, data=None, params=None, *args, **kwargs):

        # call pandas DataFrame contructor
        super().__init__(data, *args, **kwargs)

        if params is None:
            params = dict()
        self._params = None
        self.params = params

        self._units = None
        self.units = Series([np.nan for i in range(len(self.columns))])

        if 'massoftype' in self.params:
            self['m'] = self.params['massoftype']

        self._identify_spacial_columns()
        self._identify_spacial_bounds()

    def _identify_spacial_columns(self):
        """
        Identify which columns in this dataframe correspond to positional data.
        """
        # First look for 'x', then 'rx', and then fallback to the first column.
        if 'x' in self.columns:
            self._xcol = 'x'
        elif 'rx' in self.columns:
            self._xcol = 'rx'
        else:
            self._xcol = self.columns[0]

        # First look for 'y', then 'ry', and then fallback to the second column.
        if 'y' in self.columns:
            self._ycol = 'y'
        elif 'ry' in self.columns:
            self._ycol = 'ry'
        else:
            self._ycol = self.columns[1]

        # First look for 'z', then 'rz', and then assume that data is 2 dimensional
        if 'z' in self.columns:
            self._zcol = 'z'
        elif 'rz' in self.columns:
            self._zcol = 'rz'
        else:
            self._zcol = None

    def _identify_spacial_bounds(self):
        """
        Identify the maximum and minimum values of the positional data, snapped
        to the nearest integer.
        Must be called after _identify_spacial_columns()
        """
        # snap the positional bounds to the nearest integer.
        self._xmin = _snap(self.loc[:, self._xcol].min())
        self._ymin = _snap(self.loc[:, self._ycol].min())
        self._xmax = _snap(self.loc[:, self._xcol].max())
        self._ymax = _snap(self.loc[:, self._ycol].max())

        if self._zcol is not None:
            self._zmin = _snap(self.loc[:, self._zcol].min())
            self._zmax = _snap(self.loc[:, self._zcol].max())

    def derive_density(self):
        """
        Create a new column in this dataframe 'rho', derived from
        the existing columns 'hfact', 'h', and 'm'.
        """
        if not {'h', 'm'}.issubset(self.columns) or 'hfact' not in self.params:
            raise ValueError('Density cannot be derived from the columns in this SarracenDataFrame.')

        self['rho'] = (self.params['hfact'] / self['h']) ** (self.get_dim()) * self['m']

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
               pixcounty: int = None,
               cmap: Union[str, Colormap] = 'RdBu') -> ('Figure', 'Axes'):
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
        :param cmap: The color map to use for plotting this data.
        :return: The completed plot.
        """

        return render(self, target, x, y, kernel, xmin, ymin, xmax, ymax, pixcountx, pixcounty, cmap)

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

    @property
    def zcol(self):
        return self._zcol

    @zcol.getter
    def zcol(self):
        return self._zcol

    @property
    def xmin(self):
        return self._xmin

    @xmin.getter
    def xmin(self):
        return self._xmin

    @property
    def ymin(self):
        return self._ymin

    @ymin.getter
    def ymin(self):
        return self._ymin

    @property
    def zmin(self):
        return self._ymin

    @zmin.getter
    def zmin(self):
        return self._zmin

    @property
    def xmax(self):
        return self._xmax

    @xmax.getter
    def xmax(self):
        return self._xmax

    @property
    def ymax(self):
        return self._ymax

    @ymin.getter
    def ymax(self):
        return self._ymax

    @property
    def zmax(self):
        return self._zmax

    @zmin.getter
    def zmax(self):
        return self._zmax

    def get_dim(self):
        """
        Get the dimensionality of the data in this dataframe.
        :return: The number of dimensions.
        """
        return 3 if self._zcol is not None else 2
