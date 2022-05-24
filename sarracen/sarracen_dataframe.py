from typing import Union

from matplotlib.colors import Colormap
from pandas import DataFrame, Series
import numpy as np

from sarracen.render import render_2d, render_1d_cross, render_3d, render_3d_cross
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

        # First look for 'z', then 'rz', and then assume that data is 2 dimensional.
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

    def create_mass_column(self):
        """
        Create a new column in this dataframe 'm', which is copied
        from the 'massoftype' parameter.
        Intended for use with Phantom data dumps.
        :return:
        """
        if 'massoftype' not in self.params:
            raise ValueError("'massoftype' column does not exist in this SarracenDataFrame.")

        self['m'] = self.params['massoftype']

    def derive_density(self):
        """
        Create a new column in this dataframe 'rho', derived from
        the existing columns 'hfact', 'h', and 'm'.
        Intended for use with Phantom data dumps.
        """
        if not {'h', 'm'}.issubset(self.columns) or 'hfact' not in self.params:
            raise ValueError('Density cannot be derived from the columns in this SarracenDataFrame.')

        self['rho'] = (self.params['hfact'] / self['h']) ** (self.get_dim()) * self['m']

    def render_3d(self,
               target: str,
               x: str = None,
               y: str = None,
               kernel: BaseKernel = CubicSplineKernel(),
               xmin: float = None,
               ymin: float = None,
               xmax: float = None,
               ymax: float = None,
               pixcountx: int = 256,
               pixcounty: int = None,
               cmap: Union[str, Colormap] = 'RdBu',
               int_samples: int = 1000) -> ('Figure', 'Axes'):
        """
        Render the data within this dataframe to a 2D matplotlib object, using 3D -> 2D column interpolation of the
        target variable.
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
        :param int_samples: The number of samples to use when approximating the kernel column integral.
        :return: The completed plot.
        """

        return render_3d(self, target, x, y, kernel, xmin, ymin, xmax, ymax, pixcountx, pixcounty, cmap, int_samples)

    def render_2d(self,
               target: str,
               x: str = None,
               y: str = None,
               kernel: BaseKernel = CubicSplineKernel(),
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

        return render_2d(self, target, x, y, kernel, xmin, ymin, xmax, ymax, pixcountx, pixcounty, cmap)

    def render_1d_cross(self,
                       target: str,
                       x: str = None,
                       y: str = None,
                       kernel: BaseKernel = CubicSplineKernel(),
                       x1: float = None,
                       y1: float = None,
                       x2: float = None,
                       y2: float = None,
                       pixcount: int = 500) -> ('Figure', 'Axes'):
        """
        Render the data within this SarracenDataFrame to a 1D matplotlib object, by taking a 1D SPH
        cross-section of the target variable along a given line.
        :param data: The SarracenDataFrame to render. [Required]
        :param target: The variable to interpolate over. [Required]
        :param x: The positional x variable.
        :param y: The positional y variable.
        :param kernel: The kernel to use for smoothing the target data.
        :param x1: The starting x-coordinate of the cross-section line. (in particle data space)
        :param y1: The starting y-coordinate of the cross-section line. (in particle data space)
        :param x2: The ending x-coordinate of the cross-section line. (in particle data space)
        :param y2: The ending y-coordinate of the cross-section line. (in particle data space)
        :param pixcount: The number of pixels in the output over the entire cross-sectional line.
        :return: The completed plot.
        """
        return render_1d_cross(self, target, x, y, kernel, x1, y1, x2, y2, pixcount)

    def render_3d_cross(self,
                        target: str,
                        zslice: float = None,
                        x: str = None,
                        y: str = None,
                        z: str = None,
                        kernel: BaseKernel = CubicSplineKernel(),
                        xmin: float = None,
                        ymin: float = None,
                        xmax: float = None,
                        ymax: float = None,
                        pixcountx: int = 480,
                        pixcounty: int = None,
                        cmap: Union[str, Colormap] = 'RdBu') -> ('Figure', 'Axes'):
        """ Render 3D particle data inside this DataFrame to a 2D grid, using a 3D cross-section.

        Render the data within this SarracenDataFrame to a 2D matplotlib object, using a 3D -> 2D
        cross-section of the target variable. The cross-section is taken of the 3D data at a specific
        value of z, and the contributions of particles near the plane are interpolated to a 2D grid.

        Parameters
        ----------
        target: str
            The column label of the target smoothing data.
        zslice: float
            The z-axis value to take the cross-section at.
        x: str
            The column label of the x-directional axis.
        y: str
            The column label of the y-directional axis.
        z: str
            The column label of the z-directional axis.
        kernel: BaseKernel
            The kernel to use for smoothing the target data.
        xmin: float, optional
            The minimum bound in the x-direction. (in particle data space)
        ymin: float, optional
            The minimum bound in the y-direction. (in particle data space)
        xmax: float, optional
            The maximum bound in the x-direction. (in particle data space)
        ymax: float, optional
            The maximum bound in the y-direction. (in particle data space)
        pixcountx: int, optional
            The number of pixels in the output image in the x-direction.
        pixcounty: int, optional
            The number of pixels in the output image in the y-direction.
        cmap: str or Colormap, optional
            The color map to use when plotting this data.

        Returns
        -------
        Figure
            The resulting matplotlib figure, containing the 3d-cross section and
            a color bar indicating the magnitude of the target variable.
        Axes
            The resulting matplotlib axes, which contains the 3d-cross section image.

        Raises
        -------
        ValueError
           If `pixwidthx`, `pixwidthy`, `pixcountx`, or `pixcounty` are less than or equal to zero.
        """
        return render_3d_cross(self, target, zslice, x, y, z, kernel, xmin, ymin, xmax, ymax, pixcountx, pixcounty, cmap)

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
