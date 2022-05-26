from typing import Union

from matplotlib.colors import Colormap
from pandas import DataFrame, Series
import numpy as np

from sarracen.render import render_2d, render_2d_cross, render_3d, render_3d_cross
from sarracen.kernels import CubicSplineKernel, BaseKernel


class SarracenDataFrame(DataFrame):
    _metadata = ['_params', '_units', '_xcol', '_ycol', '_zcol', '_hcol', '_mcol', '_rhocol', '_kernel']

    def __init__(self, data=None, params=None, *args, **kwargs):

        # call pandas DataFrame constructor
        super().__init__(data, *args, **kwargs)

        if params is None:
            params = dict()
        self._params = None
        self.params = params

        self._units = None
        self.units = Series([np.nan for _ in range(len(self.columns))])

        self._xcol, self._ycol, self._zcol, self._hcol, self._mcol, self._rhocol = None, None, None, None, None, None
        self._identify_special_columns()

        self._kernel = CubicSplineKernel()

    def _identify_special_columns(self):
        """
        Identify which columns in this dataframe correspond to important data columns commonly used in
        analysis functions.
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

        # Look for the keyword 'h' in the data.
        if 'h' in self.columns:
            self._hcol = 'h'

        # Look for the keyword 'm' or 'mass' in the data.
        if 'm' in self.columns:
            self._mcol = 'm'
        elif 'mass' in self.columns:
            self._mcol = 'mass'

        # Look for the keyword 'rho' or 'density' in the data.
        if 'rho' in self.columns:
            self._rhocol = 'rho'
        elif 'density' in self.columns:
            self._rhocol = 'density'

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
        self._mcol = 'm'

    def derive_density(self):
        """
        Create a new column in this dataframe 'rho', derived from
        the existing columns 'hfact', 'h', and 'm'.
        Intended for use with Phantom data dumps.
        """
        if not {'h', 'm'}.issubset(self.columns) or 'hfact' not in self.params:
            raise ValueError('Density cannot be derived from the columns in this SarracenDataFrame.')

        self['rho'] = (self.params['hfact'] / self['h']) ** (self.get_dim()) * self['m']
        self._rhocol = 'rho'

    def render_2d(self,
                  target: str,
                  x: str = None,
                  y: str = None,
                  kernel: BaseKernel = None,
                  x_pixels: int = None,
                  y_pixels: int = None,
                  x_min: float = None,
                  x_max: float = None,
                  y_min: float = None,
                  y_max: float = None,
                  colormap: Union[str, Colormap] = 'RdBu') -> ('Figure', 'Axes'):
        """
        Render the data within this dataframe to a 2D matplotlib object, using 2D SPH Interpolation of the target
        variable.
        :param target: The variable to interpolate over. [Required]
        :param x: The positional x variable.
        :param y: The positional y variable.
        :param kernel: The smoothing kernel to use for interpolation.
        :param x_min: The minimum bound in the x-direction.
        :param y_min: The minimum bound in the y-direction.
        :param x_max: The maximum bound in the x-direction.
        :param y_max: The maximum bound in the y-direction.
        :param x_pixels: The number of pixels in the x-direction.
        :param y_pixels: The number of pixels in the y-direction.
        :param colormap: The color map to use for plotting this data.
        :return: The completed plot.
        """

        return render_2d(self, target, x, y, kernel, x_pixels, y_pixels, x_min, x_max, y_min, y_max, colormap)

    def render_2d_cross(self,
                        target: str,
                        x: str = None,
                        y: str = None,
                        kernel: BaseKernel = None,
                        pixels: int = 512,
                        x1: float = None,
                        y1: float = None,
                        x2: float = None,
                        y2: float = None) -> ('Figure', 'Axes'):
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
        return render_2d_cross(self, target, x, y, kernel, pixels, x1, x2, y1, y2)

    def render_3d(self,
                  target: str,
                  x: str = None,
                  y: str = None,
                  kernel: BaseKernel = None,
                  int_samples: int = 1000,
                  x_pixels: int = None,
                  y_pixels: int = None,
                  x_min: float = None,
                  x_max: float = None,
                  y_min: float = None,
                  y_max: float = None,
                  colormap: Union[str, Colormap] = 'RdBu') -> ('Figure', 'Axes'):
        """
        Render the data within this dataframe to a 2D matplotlib object, using 3D -> 2D column interpolation of the
        target variable.
        :param target: The variable to interpolate over. [Required]
        :param x: The positional x variable.
        :param y: The positional y variable.
        :param kernel: The smoothing kernel to use for interpolation.
        :param x_min: The minimum bound in the x-direction.
        :param y_min: The minimum bound in the y-direction.
        :param x_max: The maximum bound in the x-direction.
        :param y_max: The maximum bound in the y-direction.
        :param x_pixels: The number of pixels in the x-direction.
        :param y_pixels: The number of pixels in the y-direction.
        :param colormap: The color map to use for plotting this data.
        :param int_samples: The number of samples to use when approximating the kernel column integral.
        :return: The completed plot.
        """

        return render_3d(self, target, x, y, kernel, int_samples, x_pixels, y_pixels, x_min, x_max, y_min, y_max,
                         colormap)

    def render_3d_cross(self,
                        target: str,
                        z_slice: float = None,
                        x: str = None,
                        y: str = None,
                        z: str = None,
                        kernel: BaseKernel = None,
                        x_pixels: int = None,
                        y_pixels: int = None,
                        x_min: float = None,
                        x_max: float = None,
                        y_min: float = None,
                        y_max: float = None,
                        colormap: Union[str, Colormap] = 'RdBu') -> ('Figure', 'Axes'):
        """ Render 3D particle data inside this DataFrame to a 2D grid, using a 3D cross-section.

        Render the data within this SarracenDataFrame to a 2D matplotlib object, using a 3D -> 2D
        cross-section of the target variable. The cross-section is taken of the 3D data at a specific
        value of z, and the contributions of particles near the plane are interpolated to a 2D grid.

        Parameters
        ----------
        target: str
            The column label of the target smoothing data.
        z_slice: float
            The z-axis value to take the cross-section at.
        x: str
            The column label of the x-directional axis.
        y: str
            The column label of the y-directional axis.
        z: str
            The column label of the z-directional axis.
        kernel: BaseKernel
            The kernel to use for smoothing the target data.
        x_min: float, optional
            The minimum bound in the x-direction. (in particle data space)
        y_min: float, optional
            The minimum bound in the y-direction. (in particle data space)
        x_max: float, optional
            The maximum bound in the x-direction. (in particle data space)
        y_max: float, optional
            The maximum bound in the y-direction. (in particle data space)
        x_pixels: int, optional
            The number of pixels in the output image in the x-direction.
        y_pixels: int, optional
            The number of pixels in the output image in the y-direction.
        colormap: str or Colormap, optional
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
        return render_3d_cross(self, target, z_slice, x, y, z, kernel, x_pixels, y_pixels, x_min, x_max, y_min, y_max,
                               colormap)

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

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, new_units):
        self._units = new_units

    @property
    def xcol(self):
        return self._xcol

    @xcol.setter
    def xcol(self, new_col):
        if new_col in self:
            self._xcol = new_col

    @property
    def ycol(self):
        return self._ycol

    @ycol.setter
    def ycol(self, new_col):
        if new_col in self:
            self._ycol = new_col

    @property
    def zcol(self):
        return self._zcol

    @zcol.setter
    def zcol(self, new_col):
        if new_col in self:
            self._zcol = new_col

    @property
    def hcol(self):
        return self._hcol

    @hcol.setter
    def hcol(self, new_col):
        if new_col in self:
            self._hcol = new_col

    @property
    def mcol(self):
        return self._mcol

    @mcol.setter
    def mcol(self, new_col):
        if new_col in self:
            self._mcol = new_col

    @property
    def rhocol(self):
        return self._rhocol

    @rhocol.setter
    def rhocol(self, new_col):
        if new_col in self:
            self._rhocol = new_col

    @property
    def kernel(self):
        return self._kernel

    @kernel.setter
    def kernel(self, new_kernel):
        if isinstance(new_kernel, BaseKernel):
            self._kernel = new_kernel

    def get_dim(self):
        """
        Get the dimensionality of the data in this dataframe.
        :return: The number of dimensions.
        """
        return 3 if self._zcol is not None else 2
