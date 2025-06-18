from typing import Union, Callable, Tuple

from matplotlib.axes import Axes
from matplotlib.colors import Colormap
import pandas as pd
from pandas import DataFrame, Series
from numba import cuda
import numpy as np
from scipy.spatial.transform import Rotation

from .render import streamlines, arrowplot, render, lineplot
from .interpolate import interpolate_2d, interpolate_3d_grid
from .kernels import CubicSplineKernel, BaseKernel


def _copy_doc(copy_func: Callable) -> Callable:
    """Copy documentation from another function to this function."""
    def wrapper(func: Callable) -> Callable:
        func.__doc__ = copy_func.__doc__
        return func
    return wrapper


class SarracenDataFrame(DataFrame):
    """
    A SarracenDataFrame is a pandas DataFrame with support for SPH data.

    A SarracenDataFrame is a subclass of the pandas DataFrame class designed to
    hold SPH particle data. Global simulation values are stored in ``params``,
    which is a standard Python dictionary.

    Interpolation and rendering functionality requires (at a minimum) particle
    positions, smoothing lengths and masses. SarracenDataFrames will attempt to
    identify columns which hold these data. For uniform, constant mass
    particles, the particle mass can be specified in the ``params`` dictionary.

    """

    _internal_names = pd.DataFrame._internal_names + ['_xcol', '_ycol',
                                                      '_zcol', '_mcol',
                                                      '_rhocol', '_hcol',
                                                      '_vxcol', '_vycol',
                                                      '_vzcol']
    _internal_names_set = set(_internal_names)

    _metadata = ['_params', '_units', '_kernel']

    def __init__(self, data=None, params=None, *args, **kwargs):
        """
        Construct a SarracenDataFrame from a NumPy array, dictionary, DataFrame
        or Iterable object.

        Parameters
        ----------
        data : ndarray, Iterable, DataFrame, or dict.
            Raw particle data which is passed to the pandas DataFrame
            constructor. Data can be specified in a dictionary, NumPy array or
            another DataFrame.
        params : dict, optional
            Global parameters from the simulation (time, hfact, etc). If
            constant, uniform mass particles are used, then the key ``mass``
            stores the particle mass (rather than specifying per particle).
        *args : tuple, optional
            Additional arguments to pass to the pandas DataFrame constructor.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the pandas DataFrame
            constructor.

        See Also
        --------
        :func:`read_csv` : Read data from a comma separated values (csv) file.
        :func:`read_phantom` : Read data from the Phantom SPH code.

        Examples
        --------
        Constructing using a Python dictionary.

        >>> particles = {'x': [1.0, 2.0, 3.0], 'y': [2.0, 2.0, 2.0], 'h': [3.0, 3.5, 4.0]}
        >>> sdf = sarracen.SarracenDataFrame(particles)
        >>> sdf
            x     y     h
        0   1.0   2.0   3.0
        1   2.0   2.0   3.5
        2   3.0   2.0   4.0

        Constructing using a two-dimensional NumPy array.

        >>> particles = np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 3.5], [3.0, 2.0, 4.0]])
        >>> sdf = sarracen.SarracenDataFrame(particles, columns=['x', 'y', 'h'])
        >>> sdf
            x     y     h
        0   1.0   2.0   3.0
        1   2.0   2.0   3.5
        2   3.0   2.0   4.0

        Constant mass particles can specify mass in the ``params`` dictionary,
        rather than per particle.

        >>> particles = {'x': [1.0, 2.0, 3.0], 'y': [2.0, 2.0, 2.0], 'h': [3.0, 3.5, 4.0]}
        >>> params = {'mass': 0.2, 'hfact': 1.2}
        >>> sdf = sarracen.SarracenDataFrame(particles, params)
        >>> sdf.params
        {'mass': 0.2, 'hfact': 1.2}

        """

        # call pandas DataFrame constructor
        super().__init__(data, *args, **kwargs)

        if params is None:
            params = dict()
        self._params = None
        self.params = params

        self._units = None
        self.units = Series([np.nan for _ in range(len(self.columns))])

        self._xcol, self._ycol, self._zcol = None, None, None
        self._hcol, self._mcol, self._rhocol = None, None, None
        self._vxcol, self._vycol, self._vzcol = None, None, None

        self._identify_special_columns()

        self._kernel = CubicSplineKernel()
        self._backend = 'gpu' if cuda.is_available() else 'cpu'

    @property
    def _constructor(self):
        return SarracenDataFrame

    def _identify_special_columns(self):
        """
        Identify special columns commonly used in analysis functions.

        Identify which columns in this dataset correspond to important data
        columns commonly used in analysis functions. The columns which contain
        x, y, and z positional values are detected and set to the `xcol`,
        `ycol`, and `zcol` values. As well, the columns containing smoothing
        length, mass, and density information are identified and set to the
        `hcol`, `mcol`, and `rhocol`.

        If the x or y columns cannot be found, they are set to be the first two
        columns by default. If the z, smoothing length, mass, or density
        columns cannot be sound, the corresponding column label is set to
        `None`.
        """
        # First look for 'x', then 'rx', and then default to the first column.
        if 'x' in self.columns:
            self.xcol = 'x'
        elif 'rx' in self.columns:
            self.xcol = 'rx'
        elif len(self.columns) > 0:
            self.xcol = self.columns[0]

        # First look for 'y', then 'ry', and then default to the second column.
        if 'y' in self.columns:
            self.ycol = 'y'
        elif 'ry' in self.columns:
            self.ycol = 'ry'
        elif len(self.columns) > 1:
            self.ycol = self.columns[1]

        # First look for 'z', then 'rz', and then assume data is 2-dimensional.
        if 'z' in self.columns:
            self.zcol = 'z'
        elif 'rz' in self.columns:
            self.zcol = 'rz'

        # Look for the keyword 'h' in the data.
        if 'h' in self.columns:
            self.hcol = 'h'

        # Look for the keyword 'm' or 'mass' in the data.
        if 'm' in self.columns:
            self.mcol = 'm'
        elif 'mass' in self.columns:
            self.mcol = 'mass'

        # Look for the keyword 'rho' or 'density' in the data.
        if 'rho' in self.columns:
            self.rhocol = 'rho'
        elif 'density' in self.columns:
            self.rhocol = 'density'

        # Look for the keyword 'rho' or 'density' in the data.
        if 'vx' in self.columns:
            self.vxcol = 'vx'
        if 'vy' in self.columns:
            self.vycol = 'vy'
        if 'vz' in self.columns:
            self.vzcol = 'vz'

    def create_mass_column(self):
        """
        Create a new column 'm', copied from the 'massoftype' parameter.

        Intended for use with Phantom data dumps.

        Raises
        ------
        KeyError
            If the 'massoftype' column does not exist in `params`.
        """
        if 'mass' not in self.params:
            raise KeyError("'mass' value does not exist in this "
                           "SarracenDataFrame.")

        self['m'] = self.params['mass']
        self.mcol = 'm'

    def calc_density(self):
        """
        Create a new column 'rho' that contains particle densities.

        Density for each particle is calculated according to

            .. math::

                \\rho = m \\left( \\frac{h_{\\rm fact}}{h}
                                \\right)^{n_{\\rm dim}}

        where :math:`m` is the particle mass, :math:`h` is the smoothing
        length, and :math:`h_{\\rm fact}` defines the ratio of smoothing length
        to particle spacing. Smoothing lengths are taken from the smoothing
        length column, particle masses from the mass column if present, or
        params if not, and hfact from params.

        Raises
        ------
        KeyError
            If the `hcol` column does not exist, there is no `mcol` column or
            `mass` in params, or if `hfact` does not exist in `params`.
        """
        if not {self.hcol}.issubset(self.columns):
            raise KeyError('Missing smoothing length data in this '
                           'SarracenDataFrame')
        if 'hfact' not in self.params:
            raise KeyError('hfact missing from params in this '
                           'SarracenDataFrame.')
        if self.mcol not in self.columns and 'mass' not in self.params:
            raise KeyError('Missing particle mass data in this '
                           'SarracenDataFrame.')

        # prioritize using mass per particle, if present
        if {self.mcol}.issubset(self.columns):
            mass = self[self.mcol]
        else:
            mass = self.params['mass']

        hfact = self.params['hfact']
        self['rho'] = mass * (hfact / self[self.hcol])**self.get_dim()
        self.rhocol = 'rho'

    def centre_of_mass(self):
        """
        Returns the centre of mass of the data.
        """
        if {self.mcol}.issubset(self.columns):
            mass = self[self.mcol]
        else:
            mass = self.params['mass']

        com_x = (self[self.xcol] * mass).sum()
        com_y = (self[self.ycol] * mass).sum()
        com_z = (self[self.zcol] * mass).sum()

        if isinstance(mass, pd.Series):
            mass = 1.0 / mass.sum()
        else:
            mass = 1.0 / (len(self) * mass)

        return [com_x * mass, com_y * mass, com_z * mass]

    def classify_sink(self, sdf_sinks):
        """
        Creates column calculating the energy of particles
        relative to each sink.
        Creates a new column 'sink' that classifies particles by the sink they
        are bound to, or -1 if they are not bound to any sink.
        """
        # calculate the energy of particles relative to each sink
        v = np.array([self[self.vxcol], self[self.vycol], self[self.vzcol]]).T
        for index, sink in sdf_sinks.iterrows():

            # kinetic energy
            v_sink = np.array([sink[sdf_sinks.vxcol],
                               sink[sdf_sinks.vycol],
                               sink[sdf_sinks.vzcol]])
            v_rel = np.linalg.norm(v-v_sink, axis=1)
            E_k = 0.5 * self[self.mcol] * v_rel**2

            # potential energy
            rel_pos = self[[self.xcol, self.ycol, self.zcol]] - sdf_sinks[
                [sdf_sinks.xcol, sdf_sinks.ycol, sdf_sinks.zcol]].loc[index]
            r = np.linalg.norm(rel_pos, axis=1)
            E_pot = -sink[sdf_sinks.mcol]*self[self.mcol] / r

            self[f"E_{index}"] = E_k + E_pot

        # classify particles by sink they are bound to
        energies = np.array([[self[f"E_{index}"]]
                            for index in sdf_sinks.index])
        self["sink"] = np.argmin(energies, axis=0)[0, :]

        # identify particles not bound to any sink
        unbound = pd.Series((energies.min(axis=0) > 0)[0, :], index=self.index)
        self.loc[unbound[unbound].index, "sink"] = -1


    @_copy_doc(render)
    def render(self,
               target: str,
               x: str = None,
               y: str = None,
               z: str = None,
               xsec: float = None,
               kernel: BaseKernel = None,
               x_pixels: int = None,
               y_pixels: int = None,
               xlim: Tuple[float, float] = None,
               ylim: Tuple[float, float] = None,
               cmap: Union[str, Colormap] = 'gist_heat',
               cbar: bool = True,
               cbar_kws: dict = {},
               cbar_ax: Axes = None,
               ax: Axes = None,
               exact: bool = None,
               backend: str = None,
               integral_samples: int = 1000,
               rotation: Union[np.ndarray, list, Rotation] = None,
               rot_origin: Union[np.ndarray, list, str] = None,
               log_scale: bool = None,
               symlog_scale: bool = None,
               dens_weight: bool = None,
               normalize: bool = False,
               hmin: bool = False,
               **kwargs) -> Axes:
        return render(self, target, x, y, z, xsec, kernel, x_pixels, y_pixels,
                      xlim, ylim, cmap, cbar, cbar_kws, cbar_ax, ax, exact,
                      backend, integral_samples, rotation, rot_origin,
                      log_scale, symlog_scale, dens_weight, normalize, hmin,
                      **kwargs)

    @_copy_doc(lineplot)
    def lineplot(self,
                 target: str,
                 x: str = None,
                 y: str = None,
                 z: str = None,
                 kernel: BaseKernel = None,
                 pixels: int = 512,
                 xlim: Tuple[float, float] = None,
                 ylim: Tuple[float, float] = None,
                 zlim: Tuple[float, float] = None,
                 ax: Axes = None,
                 backend: str = None,
                 log_scale: bool = False,
                 dens_weight: bool = None,
                 normalize: bool = False,
                 hmin: bool = False,
                 **kwargs):
        return lineplot(self, target, x, y, z, kernel, pixels, xlim, ylim,
                        zlim,  ax, backend, log_scale, dens_weight, normalize,
                        hmin, **kwargs)

    @_copy_doc(streamlines)
    def streamlines(self,
                    target: Union[Tuple[str, str], Tuple[str, str, str]],
                    x: str = None,
                    y: str = None,
                    z: str = None,
                    xsec: int = None,
                    kernel: BaseKernel = None,
                    integral_samples: int = 1000,
                    rotation: Union[np.ndarray, list, Rotation] = None,
                    rot_origin: Union[np.ndarray, list, str] = None,
                    x_pixels: int = None,
                    y_pixels: int = None,
                    xlim: Tuple[float, float] = None,
                    ylim: Tuple[float, float] = None,
                    ax: Axes = None,
                    exact: bool = None,
                    backend: str = None,
                    dens_weight: bool = False,
                    normalize: bool = False,
                    hmin: bool = False,
                    **kwargs) -> Axes:
        return streamlines(self, target, x, y, z, xsec, kernel,
                           integral_samples, rotation, rot_origin, x_pixels,
                           y_pixels, xlim, ylim, ax, exact, backend,
                           dens_weight, normalize, hmin, **kwargs)

    @_copy_doc(arrowplot)
    def arrowplot(self,
                  target: Union[Tuple[str, str], Tuple[str, str, str]],
                  x: str = None,
                  y: str = None,
                  z: str = None,
                  xsec: int = None,
                  kernel: BaseKernel = None,
                  integral_samples: int = 1000,
                  rotation: Union[np.ndarray, list, Rotation] = None,
                  rot_origin: Union[np.ndarray, list, str] = None,
                  x_arrows: int = None,
                  y_arrows: int = None,
                  xlim: Tuple[float, float] = None,
                  ylim: Tuple[float, float] = None,
                  ax: Axes = None,
                  qkey: bool = True,
                  qkey_kws: dict = None,
                  exact: bool = None,
                  backend: str = None,
                  dens_weight: bool = None,
                  normalize: bool = False,
                  hmin: bool = False,
                  **kwargs) -> Axes:
        return arrowplot(self, target, x, y, z, xsec, kernel, integral_samples,
                         rotation, rot_origin, x_arrows, y_arrows, xlim, ylim,
                         ax, qkey, qkey_kws, exact, backend, dens_weight,
                         normalize, hmin, **kwargs)

    def sph_interpolate(self,
                        target: str,
                        x: str = None,
                        y: str = None,
                        z: str = None,
                        kernel: BaseKernel = None,
                        rotation: Union[np.ndarray, list, Rotation] = None,
                        rot_origin: Union[np.ndarray, list, str] = None,
                        x_pixels: int = None,
                        y_pixels: int = None,
                        z_pixels: int = None,
                        xlim: Tuple[float, float] = None,
                        ylim: Tuple[float, float] = None,
                        zlim: Tuple[float, float] = None,
                        exact: bool = None,
                        backend: str = 'cpu',
                        dens_weight: bool = False,
                        normalize: bool = False,
                        hmin: bool = False) -> np.ndarray:
        """
        Interpolate this data to a 2D or 3D grid, depending on the
        dimensionality of the data.

        Parameters
        ----------
        target: str
            The column label of the target data.
        x, y, z: str
            The column labels of the directional data to interpolate over.
            Defaults to the x, y, and z columns detected in `data`.
        kernel: BaseKernel
            The kernel to use for smoothing the target data. Defaults to the
            kernel specified in `data`.
        rotation: array_like or SciPy Rotation, optional
            The rotation to apply to the data before interpolation. If defined
            as an array, the order of rotations is [z, y, x] in degrees. Only
            applies to 3D datasets.
        rot_origin: array_like or ['com', 'midpoint'], optional
            Point of rotation of the data. Only applies to 3D datasets. If
            array_like, then the [x, y, z] coordinates specify the point around
            which the data is rotated. If 'com', then data is rotated around
            the centre of mass. If 'midpoint', then data is rotated around the
            midpoint, that is, min + max / 2. Defaults to the midpoint.
        x_pixels, y_pixels, z_pixels: int, optional
            Number of pixels in the output image in the x, y & z directions.
            Default values are chosen to keep a consistent aspect ratio.
        xlim, ylim, zlim: tuple of float, optional
            The minimum and maximum values to use in interpolation, in particle
            data space. Defaults to the minimum and maximum values of `x`, `y`
            and `z`.
        exact: bool
            Whether to use exact interpolation of the data. Only applies to
            2D datasets.
        backend: ['cpu', 'gpu']
            The computation backend to use when interpolating this data.
            Defaults to 'gpu' if CUDA is enabled, otherwise 'cpu' is used. A
            manually specified backend in `data` will override the default.
        dens_weight: bool
            If True, the target will be multiplied by density. Defaults to
            False.
        normalize: bool
            If True, will normalize the interpolation. Defaults to False (this
            may change in future versions).
        hmin: bool
            If True, a minimum smoothing length of 0.5 * pixel size will be
            imposed. This ensures each particle contributes to at least one
            grid cell / pixel. Defaults to False (this may change in a future
            verison).

        Returns
        -------
        ndarray (n-Dimensional)
            The interpolated output image, in a multi-dimensional numpy array.
            The number of dimensions match the dimensions of the data.
            Dimensions are structured in reverse order, where (x, y, z) ->
            [z, y, x].

        Raises
        -------
        ValueError
            If `x_pixels`, `y_pixels` or `z_pixels` are less than or equal to
            zero, or if the specified `x`, `y` and `z` minimum and maximum
            values result in an invalid region, or if `data` is not 2 or
            3-dimensional.
        KeyError
            If `target`, `x`, `y`, `z`, mass, density, or smoothing length
            columns do not exist in `data`.
        """
        if self.get_dim() == 2:
            if xlim is None:
                xlim = (None, None)
            if ylim is None:
                ylim = (None, None)
            return interpolate_2d(self, target, x, y, kernel, x_pixels,
                                  y_pixels, xlim, ylim, exact, backend,
                                  dens_weight, normalize, hmin)
        elif self.get_dim() == 3:
            return interpolate_3d_grid(self, target, x, y, z, kernel, rotation,
                                       rot_origin, x_pixels, y_pixels,
                                       z_pixels, xlim, ylim, zlim, backend,
                                       dens_weight, normalize, hmin)

    @property
    def params(self):
        """
        dict: Miscellaneous dataset-level parameters.

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
        """
        str : Label of the column which contains x-positional data.

        If this is set to a column which does not exist in the dataset, the
        column label will remain set to the old value.
        """
        return self._xcol

    @xcol.setter
    def xcol(self, new_col: str):
        if new_col in self or new_col is None:
            self._xcol = new_col

    @property
    def ycol(self):
        """
        str : Label of the column which contains y-positional data.

        If this is set to a column which does not exist in the dataset, the
        column label will remain set to the old value.
        """
        return self._ycol

    @ycol.setter
    def ycol(self, new_col: str):
        if new_col in self or new_col is None:
            self._ycol = new_col

    @property
    def zcol(self):
        """
        str : Label of the column which contains z-positional data.

        If this is set to a column which does not exist in the dataset, the
        column label will remain set to the old value.
        """
        return self._zcol

    @zcol.setter
    def zcol(self, new_col: str):
        if new_col in self or new_col is None:
            self._zcol = new_col

    @property
    def hcol(self):
        """
        str : Label of the column which contains smoothing length data.

        If this is set to a column which does not exist in the dataset, the
        column label will remain set to the old value.
        """
        return self._hcol

    @hcol.setter
    def hcol(self, new_col: str):
        if new_col in self or new_col is None:
            self._hcol = new_col

    @property
    def mcol(self):
        """
        str : Label of the column which contains particle mass data.

        If this is set to a column which does not exist in the dataset, the
        column label will remain set to the old value.
        """
        return self._mcol

    @mcol.setter
    def mcol(self, new_col: str):
        if new_col in self or new_col is None:
            self._mcol = new_col

    @property
    def rhocol(self):
        """
        str : Label of the column which contains particle density data.

        If this is set to a column which does not exist in the dataset, the
        column label will remain set to the old value.
        """
        return self._rhocol

    @rhocol.setter
    def rhocol(self, new_col: str):
        if new_col in self or new_col is None:
            self._rhocol = new_col

    @property
    def vxcol(self):
        """
        str : Label of the column which contains the x-component of the
        velocity.

        If this is set to a column which does not exist in the dataset, the
        column label will remain set to the old value.
        """
        return self._vxcol

    @vxcol.setter
    def vxcol(self, new_col: str):
        if new_col in self or new_col is None:
            self._vxcol = new_col

    @property
    def vycol(self):
        """
        str : Label of the column which contains the y-component of the
        velocity.

        If this is set to a column which does not exist in the dataset, the
        column label will remain set to the old value.
        """
        return self._vycol

    @vycol.setter
    def vycol(self, new_col: str):
        if new_col in self or new_col is None:
            self._vycol = new_col

    @property
    def vzcol(self):
        """
        str : Label of the column which contains the z-component of the
        velocity.

        If this is set to a column which does not exist in the dataset, the
        column label will remain set to the old value.
        """
        return self._vzcol

    @vzcol.setter
    def vzcol(self, new_col: str):
        if new_col in self or new_col is None:
            self._vzcol = new_col

    @property
    def kernel(self):
        """
        BaseKernel : The default kernel to use for interpolation operations
        with this dataset.

        If this is set to an object which is not a BaseKernel, the kernel will
        remain set as the old value.
        """
        return self._kernel

    @kernel.setter
    def kernel(self, new_kernel: BaseKernel):
        if isinstance(new_kernel, BaseKernel):
            self._kernel = new_kernel

    @property
    def backend(self):
        """
        ['cpu', 'gpu'] : The default backend to use for interpolation
        operations with this dataset.

        'cpu' - Best for small datasets, or cases where a GPU is not available.
        'gpu' - Best for large datasets, with a CUDA-enabled GPU.
        """
        return self._backend

    @backend.setter
    def backend(self, new_backend: str):
        self._backend = new_backend

    def get_dim(self):
        """
        Get the dimensionality of the data in this dataframe.

        Returns
        -------
        int
            The number of positional dimensions.
        """
        return 3 if self.zcol is not None else 2
