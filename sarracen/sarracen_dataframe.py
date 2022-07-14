from typing import Union, Callable, Tuple

from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from pandas import DataFrame, Series
import numpy as np

from sarracen.render import render_2d, render_2d_cross, render_3d, render_3d_cross, streamlines, arrowplot
from sarracen.kernels import CubicSplineKernel, BaseKernel


def _copy_doc(copy_func: Callable) -> Callable:
    """Copy documentation from another function to this function."""
    def wrapper(func: Callable) -> Callable:
        func.__doc__ = copy_func.__doc__
        return func
    return wrapper


class SarracenDataFrame(DataFrame):
    """ A pandas DataFrame which contains relevant data for SPH data visualizations.

    This is an extended version of the pandas DataFrame class, which contains several
    derived parameters used in the `render.py` and `interpolate.py` modules. The labels
    of columns containing x, y, and z directional data, and the labels of columns containing
    mass, density, and smoothing length information are all stored. As well, the kernel
    used for all interpolation operations, and the units for each data column.

    See Also
    --------
    readers : Functions for creating SarracenDataFrame objects from exported SPH data.
    """
    _metadata = ['_params', '_units', '_xcol', '_ycol', '_zcol', '_hcol', '_mcol', '_rhocol', '_kernel']

    def __init__(self, data=None, params=None, *args, **kwargs):
        """ Create a new `SarracenDataFrame`, and automatically detect important columns.

        Parameters
        ----------
        data : ndarray (structured or homogeneous), Iterable, DataFrame, or dict.
            Raw particle data passed to the DataFrame super-initializer.
        params : dict
            Miscellaneous dataset-level parameters.
        *args : tuple
            Additional arguments to the DataFrame super-initializer.
        **kwargs : dict, optional
            Additional keyword arguments to the DataFrame super-initializer.
        """
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
        self._backend = 'cpu'

    @property
    def _constructor(self):
        return SarracenDataFrame

    def _identify_special_columns(self):
        """ Identify special columns commonly used in analysis functions.

        Identify which columns in this dataset correspond to important data columns commonly used in
        analysis functions. The columns which contain x, y, and z positional values are detected and
        set to the `xcol`, `ycol`, and `zcol` values. As well, the columns containing smoothing length,
        mass, and density information are identified and set to the `hcol`, `mcol`, and `rhocol`.

        If the x or y columns cannot be found, they are set to be the first two columns by default.
        If the z, smoothing length, mass, or density columns cannot be sound, the corresponding column
        label is set to `None`.
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
        """ Create a new column 'm', copied from the 'massoftype' dataset parameter.

        Intended for use with Phantom data dumps.

        Raises
        ------
        KeyError
            If the 'massoftype' column does not exist in `params`.
        """
        if 'massoftype' not in self.params:
            raise KeyError("'massoftype' column does not exist in this SarracenDataFrame.")

        self['m'] = self.params['massoftype']
        self._mcol = 'm'

    def derive_density(self):
        """ Create a new column 'rho', derived from columns 'hfact', 'h', and 'm'.

        Intended for use with Phantom data dumps.

        Raises
        ------
        KeyError
            If the `hcol` and `mcol` columns do not exist, or 'hfact' does not exist in `params`.
        """
        if not {self.hcol, self.mcol}.issubset(self.columns) or 'hfact' not in self.params:
            raise KeyError('Density cannot be derived from the columns in this SarracenDataFrame.')

        self['rho'] = (self.params['hfact'] / self['h']) ** (self.get_dim()) * self['m']
        self._rhocol = 'rho'

    @_copy_doc(render_2d)
    def render_2d(self, target: str, x: str = None, y: str = None, kernel: BaseKernel = None, x_pixels: int = None,
                  y_pixels: int = None, x_min: float = None, x_max: float = None, y_min: float = None,
                  y_max: float = None, cmap: Union[str, Colormap] = 'RdBu', cbar: bool = True, cbar_kws: dict = {},
                  cbar_ax: Axes = None, ax: Axes = None, exact: bool = None, backend: str = None, **kwargs) -> Axes:

        return render_2d(self, target, x, y, kernel, x_pixels, y_pixels, x_min, x_max, y_min, y_max, cmap, cbar,
                         cbar_kws, cbar_ax, ax, exact, backend, **kwargs)

    @_copy_doc(render_2d_cross)
    def render_2d_cross(self, target: str, x: str = None, y: str = None, kernel: BaseKernel = None, pixels: int = 512,
                        x1: float = None, y1: float = None, x2: float = None, y2: float = None, ax: Axes = None,
                        backend: str = None, **kwargs) -> Axes:

        return render_2d_cross(self, target, x, y, kernel, pixels, x1, x2, y1, y2, ax, backend, **kwargs)

    @_copy_doc(render_3d)
    def render_3d(self, target: str, x: str = None, y: str = None, z: str = None, kernel: BaseKernel = None,
                  int_samples: int = 1000, rotation: np.ndarray = None, rot_origin: np.ndarray = None,
                  x_pixels: int = None, y_pixels: int = None, x_min: float = None, x_max: float = None,
                  y_min: float = None, y_max: float = None, cmap: Union[str, Colormap] = 'RdBu', cbar: bool = True,
                  cbar_kws: dict = {}, cbar_ax: Axes = None, ax: Axes = None, exact: bool = None, backend: str = None,
                  **kwargs) -> Axes:

        return render_3d(self, target, x, y, z, kernel, int_samples, rotation, rot_origin, x_pixels, y_pixels, x_min,
                         x_max, y_min, y_max, cmap, cbar, cbar_kws, cbar_ax, ax, exact, backend, **kwargs)

    @_copy_doc(render_3d_cross)
    def render_3d_cross(self, target: str, z_slice: float = None, x: str = None, y: str = None, z: str = None,
                        kernel: BaseKernel = None, rotation: np.ndarray = None, rot_origin: np.ndarray = None,
                        x_pixels: int = None, y_pixels: int = None, x_min: float = None, x_max: float = None,
                        y_min: float = None, y_max: float = None, cmap: Union[str, Colormap] = 'RdBu',
                        cbar: bool = True, cbar_kws: dict = {}, cbar_ax: Axes = None, ax: Axes = None,
                        backend: str = None, **kwargs) -> Axes:

        return render_3d_cross(self, target, z_slice, x, y, z, kernel, rotation, rot_origin, x_pixels, y_pixels, x_min,
                               x_max, y_min, y_max, cmap, cbar, cbar_kws, cbar_ax, ax, backend, **kwargs)

    @_copy_doc(streamlines)
    def streamlines(self, target: Union[Tuple[str, str], Tuple[str, str, str]], z_slice: int = None, x: str = None,
                    y: str = None, z: str = None, kernel: BaseKernel = None, integral_samples: int = 1000,
                    rotation: np.ndarray = None, rot_origin: np.ndarray = None, x_pixels: int = None,
                    y_pixels: int = None, x_min: float = None, x_max: float = None, y_min: float = None,
                    y_max: float = None, ax: Axes = None, backend: str='cpu', **kwargs) -> Axes:
        return streamlines(self, target, z_slice, x, y, z, kernel, integral_samples, rotation, rot_origin, x_pixels,
                           y_pixels, x_min, x_max, y_min, y_max, ax, backend, **kwargs)

    @_copy_doc(arrowplot)
    def arrowplot(self, target: Union[Tuple[str, str], Tuple[str, str, str]], z_slice: int = None, x: str = None,
                  y: str = None, z: str = None, kernel: BaseKernel = None, integral_samples: int = 1000,
                  rotation: np.ndarray = None, rot_origin: np.ndarray = None, x_arrows: int = None,
                  y_arrows: int = None, x_min: float = None, x_max: float = None, y_min: float = None,
                  y_max: float = None, ax: Axes = None, backend: str='cpu', **kwargs) -> Axes:
        return arrowplot(self, target, z_slice, x, y, z, kernel, integral_samples, rotation, rot_origin, x_arrows,
                         y_arrows, x_min, x_max, y_min, y_max, ax, backend, **kwargs)

    @property
    def params(self):
        """dict: Miscellaneous dataset-level parameters.

        Raises
        ------
        TypeError
            If `params` is set to a non-dictionary or non-None object.
        """
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
        """Series: Units for each column of this dataset."""
        return self._units

    @units.setter
    def units(self, new_units: Series):
        self._units = new_units

    @property
    def xcol(self):
        """str : Label of the column which contains x-positional data.

        If this is set to a column which does not exist in the dataset, the column
        label will remain set to the old value.
        """
        return self._xcol

    @xcol.setter
    def xcol(self, new_col: str):
        if new_col in self:
            self._xcol = new_col

    @property
    def ycol(self):
        """str : Label of the column which contains y-positional data.

        If this is set to a column which does not exist in the dataset, the column
        label will remain set to the old value.
        """
        return self._ycol

    @ycol.setter
    def ycol(self, new_col: str):
        if new_col in self:
            self._ycol = new_col

    @property
    def zcol(self):
        """str : Label of the column which contains z-positional data.

        If this is set to a column which does not exist in the dataset, the column
        label will remain set to the old value.
        """
        return self._zcol

    @zcol.setter
    def zcol(self, new_col: str):
        if new_col in self:
            self._zcol = new_col

    @property
    def hcol(self):
        """str : Label of the column which contains smoothing length data.

        If this is set to a column which does not exist in the dataset, the column
        label will remain set to the old value.
        """
        return self._hcol

    @hcol.setter
    def hcol(self, new_col: str):
        if new_col in self:
            self._hcol = new_col

    @property
    def mcol(self):
        """str : Label of the column which contains particle mass data.

        If this is set to a column which does not exist in the dataset, the column
        label will remain set to the old value.
        """
        return self._mcol

    @mcol.setter
    def mcol(self, new_col: str):
        if new_col in self:
            self._mcol = new_col

    @property
    def rhocol(self):
        """str : Label of the column which contains particle density data.

        If this is set to a column which does not exist in the dataset, the column
        label will remain set to the old value.
        """
        return self._rhocol

    @rhocol.setter
    def rhocol(self, new_col: str):
        if new_col in self:
            self._rhocol = new_col

    @property
    def kernel(self):
        """BaseKernel : The default kernel to use for interpolation operations with this dataset.

        If this is set to an object which is not a BaseKernel, the kernel will remain set as
        the old value.
        """
        return self._kernel

    @kernel.setter
    def kernel(self, new_kernel: BaseKernel):
        if isinstance(new_kernel, BaseKernel):
            self._kernel = new_kernel

    @property
    def backend(self):
        """['cpu', 'gpu'] : The default backend to use for interpolation operations with this dataset.

        'cpu' - Best for small datasets, or cases where a GPU is not available.
        'gpu' - Best for large datasets, with a CUDA-enabled GPU.
        """
        return self._backend

    @backend.setter
    def backend(self, new_backend: str):
        self._backend = new_backend

    def get_dim(self):
        """ Get the dimensionality of the data in this dataframe.

        Returns
        -------
        int
            The number of positional dimensions.
        """
        return 3 if self._zcol is not None else 2
