from typing import Any, Type, Union, Callable, Tuple, Optional, Dict

from matplotlib.axes import Axes
from matplotlib.colors import Colormap
import pandas as pd
from pandas import DataFrame, Series
from numba import cuda
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import griddata

from .render import streamlines, arrowplot, render, lineplot
from .interpolate import interpolate_2d, interpolate_3d_grid
from .kernels import CubicSplineKernel, BaseKernel

from sklearn.neighbors import KDTree

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
                                                      '_vzcol',
                                                      '_dustfracscol']
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

        self._params = dict(params or {})

        self._units = None
        self.units = Series([np.nan for _ in range(len(self.columns))])

        self._xcol, self._ycol, self._zcol = None, None, None
        self._hcol, self._mcol, self._rhocol = None, None, None
        self._vxcol, self._vycol, self._vzcol = None, None, None
        self._dustfracscol = []

        self._identify_special_columns()

        self._kernel = CubicSplineKernel()
        self._backend = 'gpu' if cuda.is_available() else 'cpu'

    @property
    def _constructor(self) -> Type:
        return SarracenDataFrame

    def _identify_special_columns(self) -> None:
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

        # Look for the keyword 'dustfrac' in the data.
        for column in self.columns:
            if column.startswith('dustfrac'):
                self.dustfracscol.append(column)

    def create_mass_column(self) -> None:
        """
        Create a new column 'm', copied from the 'massoftype' parameter.

        Intended for use with Phantom data dumps.

        Raises
        ------
        KeyError
            If the 'massoftype' column does not exist in `params`.
        """
        if self.params is None or 'mass' not in self.params:
            raise KeyError("'mass' value does not exist in this "
                           "SarracenDataFrame.")

        self['m'] = self.params['mass']
        self.mcol = 'm'

    def calc_density(self) -> None:
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
        if self.params is None or 'hfact' not in self.params:
            raise KeyError('hfact missing from params in this '
                           'SarracenDataFrame.')
        if self.mcol not in self.columns and (self.params is None or
                                              'mass' not in self.params):
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

    def calc_one_fluid_quantities(self) -> None:
        """
        Create new columns that contain the densities of gas, dust (total),
        each dust grain size and the dust-to gas ratio in one-fluid
        (a.k.a. dust-as-mixture) simulations.

        Raises
        -------
        KeyError
            If the `dustfrac` column does not exist.
        ValueError
            If `ndustsmall` is zero or `ndustlarge` is non zero.
        """
        if not {self.dustfracscol[0]}.issubset(self.columns):
            raise KeyError('Missing dust fraction data in this '
                           'SarracenDataFrame')
        # use self[self.dustfrac], check if columns exist
        if not {self.rhocol}.issubset(self.columns):
            self.calc_density()

        if (int(self.params['ndustsmall']) == 0 or
                int(self.params['ndustlarge']) != 0):
            raise ValueError('Not a one-fluid-only dump.')
        else:
            if int(self.params['ndustsmall']) == 1:
                self['rho_g'] = self['rho'] * self['dustfrac']
                self['rho_d'] = self['rho'] * (1 - self['dustfrac'])
                self['dtg'] = self['rho_d'] / self['rho_g']
            else:
                self['dustfrac_total'] = self[self.dustfracscol].sum(axis=1)
                self['rho_g'] = self['rho'] * (1 - self['dustfrac_total'])
                self['rho_d_total'] = self['rho'] * self['dustfrac_total']
                self['rho_d'] = self['rho'] * self[self.dustfracscol[0]]
                for i in range(1, int(self.params['ndustsmall'])):
                    self[f'rho_d_{i+1}'] = self['rho'] * \
                                                    self[self.dustfracscol[i]]
                self['dtg'] = self['dustfrac_total'] / (1 -
                                                        self['dustfrac_total'])

    def centre_of_mass(self) -> list:
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
            inv_mass = 1.0 / mass.sum()
        else:
            inv_mass = 1.0 / (len(self) * mass)

        return [com_x * inv_mass, com_y * inv_mass, com_z * inv_mass]

    def classify_sink(self, sdf_sinks: Type) -> None:
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
            if {self.mcol}.issubset(self.columns):
                mass = self[self.mcol]
            else:
                if self.params is None or 'mass' not in self.params:
                    raise KeyError("'mass' value does not exist in this "
                                   "SarracenDataFrame.")
                mass = self.params['mass']
            E_k = 0.5 * mass * v_rel**2

            # potential energy
            rel_pos = self[[self.xcol, self.ycol, self.zcol]] - sdf_sinks[
                [sdf_sinks.xcol, sdf_sinks.ycol, sdf_sinks.zcol]].loc[index]
            r = np.linalg.norm(rel_pos, axis=1)
            E_pot = -sink[sdf_sinks.mcol] * mass / r

            self[f"E_{index}"] = E_k + E_pot

        # classify particles by sink they are bound to
        energies = np.array([[self[f"E_{index}"]]
                            for index in sdf_sinks.index])
        self["sink"] = np.argmin(energies, axis=0)[0, :]

        # identify particles not bound to any sink
        unbound = pd.Series((energies.min(axis=0) > 0)[0, :], index=self.index)
        self.loc[unbound[unbound].index, "sink"] = -1

    #method to interpolate gas density for dust particles
    def interpolate_gas_density(self):
        """Linearly interpolates gas density onto dust particles using a grid method.
        """
        if "rho_gas" not in self: # speed up by only performing interpolation if not already done

            self.calc_density()
            self["rho_gas"] = self["rho"]

            gas_df = self[self["itype"]==1]
            gas_pos = (gas_df["x"],gas_df["y"],gas_df["z"])
            gasvals = gas_df["rho"]

            dust_df = self[self["itype"]==7]
            dust_pos = (dust_df["x"],dust_df["y"],dust_df["z"])
            dustvals = griddata(gas_pos,gasvals,dust_pos,method="nearest")

            self.loc[self["itype"]==7,"rho_gas"] = dustvals

    def calc_stokes(self,c_s,G=1):
        """"Calculates Stokes number of two-fluid dust particles."""
        if "St" not in self:
            self.interpolate_gas_density()
            self["r"] = (self[self.xcol]**2 + self[self.ycol]**2)**0.5
            omega_kep = (self.sinks["m"][0]*G/self["r"]**3)**0.5
            t_s = self.params["graindens"] * self.params["grainsize"]/(self["rho_gas"]+self["rho"])/c_s
            self["St"] = t_s * omega_kep
            self.loc[self["itype"]==1,"St"] = 0

    @_copy_doc(render)
    def render(self,
               target: str,
               x: Union[str, None] = None,
               y: Union[str, None] = None,
               z: Union[str, None] = None,
               xsec: Union[float, None] = None,
               kernel: Union[BaseKernel, None] = None,
               x_pixels: Union[int, None] = None,
               y_pixels: Union[int, None] = None,
               xlim: Union[Tuple[float, float], None] = None,
               ylim: Union[Tuple[float, float], None] = None,
               cmap: Union[str, Colormap] = 'gist_heat',
               cbar: bool = True,
               cbar_kws: dict = {},
               cbar_ax: Union[Axes, None] = None,
               ax: Union[Axes, None] = None,
               exact: bool = False,
               backend: Union[str, None] = None,
               integral_samples: int = 1000,
               rotation: Union[np.ndarray, list, Rotation, None] = None,
               rot_origin: Union[np.ndarray, list, str, None] = None,
               log_scale: bool = False,
               symlog_scale: bool = False,
               dens_weight: Union[bool, None] = None,
               normalize: bool = False,
               hmin: bool = False,
               **kwargs: Any) -> Axes:
        return render(self, target, x, y, z, xsec, kernel, x_pixels, y_pixels,
                      xlim, ylim, cmap, cbar, cbar_kws, cbar_ax, ax, exact,
                      backend, integral_samples, rotation, rot_origin,
                      log_scale, symlog_scale, dens_weight, normalize, hmin,
                      **kwargs)

    @_copy_doc(lineplot)
    def lineplot(self,
                 target: str,
                 x: Union[str, None] = None,
                 y: Union[str, None] = None,
                 z: Union[str, None] = None,
                 kernel: Union[BaseKernel, None] = None,
                 pixels: int = 512,
                 xlim: Union[Tuple[float, float], None] = None,
                 ylim: Union[Tuple[float, float], None] = None,
                 zlim: Union[Tuple[float, float], None] = None,
                 ax: Union[Axes, None] = None,
                 backend: Union[str, None] = None,
                 log_scale: bool = False,
                 dens_weight: bool = False,
                 normalize: bool = False,
                 hmin: bool = False,
                 **kwargs: Any) -> Axes:
        return lineplot(self, target, x, y, z, kernel, pixels, xlim, ylim,
                        zlim,  ax, backend, log_scale, dens_weight, normalize,
                        hmin, **kwargs)

    @_copy_doc(streamlines)
    def streamlines(self,
                    target: Union[Tuple[str, str], Tuple[str, str, str]],
                    x: Union[str, None] = None,
                    y: Union[str, None] = None,
                    z: Union[str, None] = None,
                    xsec: Union[int, None] = None,
                    kernel: Union[BaseKernel, None] = None,
                    integral_samples: int = 1000,
                    rotation: Union[np.ndarray, list, Rotation, None] = None,
                    rot_origin: Union[np.ndarray, list, str, None] = None,
                    x_pixels: Union[int, None] = None,
                    y_pixels: Union[int, None] = None,
                    xlim: Union[Tuple[float, float], None] = None,
                    ylim: Union[Tuple[float, float], None] = None,
                    ax: Union[Axes, None] = None,
                    exact: bool = False,
                    backend: Union[str, None] = None,
                    dens_weight: bool = False,
                    normalize: bool = False,
                    hmin: bool = False,
                    **kwargs: Any) -> Axes:
        return streamlines(self, target, x, y, z, xsec, kernel,
                           integral_samples, rotation, rot_origin, x_pixels,
                           y_pixels, xlim, ylim, ax, exact, backend,
                           dens_weight, normalize, hmin, **kwargs)

    @_copy_doc(arrowplot)
    def arrowplot(self,
                  target: Union[Tuple[str, str], Tuple[str, str, str]],
                  x: Union[str, None] = None,
                  y: Union[str, None] = None,
                  z: Union[str, None] = None,
                  xsec: Union[int, None] = None,
                  kernel: Union[BaseKernel, None] = None,
                  integral_samples: int = 1000,
                  rotation: Union[np.ndarray, list, Rotation, None] = None,
                  rot_origin: Union[np.ndarray, list, str, None] = None,
                  x_arrows: Union[int, None] = None,
                  y_arrows: Union[int, None] = None,
                  xlim: Union[Tuple[float, float], None] = None,
                  ylim: Union[Tuple[float, float], None] = None,
                  ax: Union[Axes, None] = None,
                  qkey: bool = True,
                  qkey_kws: Union[dict, None] = None,
                  exact: bool = False,
                  backend: Union[str, None] = None,
                  dens_weight: Union[bool, None] = None,
                  normalize: bool = False,
                  hmin: bool = False,
                  **kwargs: Any) -> Axes:
        return arrowplot(self, target, x, y, z, xsec, kernel, integral_samples,
                         rotation, rot_origin, x_arrows, y_arrows, xlim, ylim,
                         ax, qkey, qkey_kws, exact, backend, dens_weight,
                         normalize, hmin, **kwargs)

    def sph_interpolate(self,
                        target: str,
                        x: Union[str, None] = None,
                        y: Union[str, None] = None,
                        z: Union[str, None] = None,
                        kernel: Union[BaseKernel, None] = None,
                        rotation: Union[np.ndarray, list,
                                        Rotation, None] = None,
                        rot_origin: Union[np.ndarray, list, str, None] = None,
                        x_pixels: Union[int, None] = None,
                        y_pixels: Union[int, None] = None,
                        z_pixels: Union[int, None] = None,
                        xlim: Optional[Tuple[Optional[float],
                                             Optional[float]]] = None,
                        ylim: Optional[Tuple[Optional[float],
                                             Optional[float]]] = None,
                        zlim: Union[Tuple[float, float], None] = None,
                        exact: bool = False,
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
        raise ValueError('Invalid number of dimensions.')

    @property
    def params(self) -> Dict[str, Any]:
        """
        dict: Miscellaneous dataset-level parameters.

        Raises
        ------
        TypeError
            If `params` is set to a non-dictionary or non-None object.
        """
        return self._params

    @params.setter
    def params(self, new_params: Union[Dict[str, Any], None]) -> None:
        if new_params is not None and not isinstance(new_params, dict):
            raise TypeError("Parameters not a dictionary")
        self._params = dict(new_params or {})

    @property
    def units(self) -> Series:
        """Series: Units for each column of this dataset."""
        return self._units

    @units.setter
    def units(self, new_units: Series) -> None:
        self._units = new_units

    @property
    def xcol(self) -> str:
        """
        str : Label of the column which contains x-positional data.

        If this is set to a column which does not exist in the dataset, the
        column label will remain set to the old value.
        """
        return self._xcol

    @xcol.setter
    def xcol(self, new_col: Union[str, None]) -> None:
        if new_col in self or new_col is None:
            self._xcol = new_col

    @property
    def ycol(self) -> str:
        """
        str : Label of the column which contains y-positional data.

        If this is set to a column which does not exist in the dataset, the
        column label will remain set to the old value.
        """
        return self._ycol

    @ycol.setter
    def ycol(self, new_col: Union[str, None]) -> None:
        if new_col in self or new_col is None:
            self._ycol = new_col

    @property
    def zcol(self) -> str:
        """
        str : Label of the column which contains z-positional data.

        If this is set to a column which does not exist in the dataset, the
        column label will remain set to the old value.
        """
        return self._zcol

    @zcol.setter
    def zcol(self, new_col: Union[str, None]) -> None:
        if new_col in self or new_col is None:
            self._zcol = new_col

    @property
    def hcol(self) -> str:
        """
        str : Label of the column which contains smoothing length data.

        If this is set to a column which does not exist in the dataset, the
        column label will remain set to the old value.
        """
        return self._hcol

    @hcol.setter
    def hcol(self, new_col: Union[str, None]) -> None:
        if new_col in self or new_col is None:
            self._hcol = new_col

    @property
    def mcol(self) -> str:
        """
        str : Label of the column which contains particle mass data.

        If this is set to a column which does not exist in the dataset, the
        column label will remain set to the old value.
        """
        return self._mcol

    @mcol.setter
    def mcol(self, new_col: Union[str, None]) -> None:
        if new_col in self or new_col is None:
            self._mcol = new_col

    @property
    def rhocol(self) -> str:
        """
        str : Label of the column which contains particle density data.

        If this is set to a column which does not exist in the dataset, the
        column label will remain set to the old value.
        """
        return self._rhocol

    @rhocol.setter
    def rhocol(self, new_col: Union[str, None]) -> None:
        if new_col in self or new_col is None:
            self._rhocol = new_col

    @property
    def vxcol(self) -> str:
        """
        str : Label of the column which contains the x-component of the
        velocity.

        If this is set to a column which does not exist in the dataset, the
        column label will remain set to the old value.
        """
        return self._vxcol

    @vxcol.setter
    def vxcol(self, new_col: Union[str, None]) -> None:
        if new_col in self or new_col is None:
            self._vxcol = new_col

    @property
    def vycol(self) -> str:
        """
        str : Label of the column which contains the y-component of the
        velocity.

        If this is set to a column which does not exist in the dataset, the
        column label will remain set to the old value.
        """
        return self._vycol

    @vycol.setter
    def vycol(self, new_col: Union[str, None]) -> None:
        if new_col in self or new_col is None:
            self._vycol = new_col

    @property
    def vzcol(self) -> str:
        """
        str : Label of the column which contains the z-component of the
        velocity.

        If this is set to a column which does not exist in the dataset, the
        column label will remain set to the old value.
        """
        return self._vzcol

    @vzcol.setter
    def vzcol(self, new_col: Union[str, None]) -> None:
        if new_col in self or new_col is None:
            self._vzcol = new_col

    @property
    def dustfracscol(self):
        return self._dustfracscol

    @dustfracscol.setter
    def dustfracscol(self, new_col: str) -> None:
        if new_col in self or new_col is None:
            self._dustfracscol = new_col

    @property
    def kernel(self) -> BaseKernel:
        """
        BaseKernel : The default kernel to use for interpolation operations
        with this dataset.

        If this is set to an object which is not a BaseKernel, the kernel will
        remain set as the old value.
        """
        return self._kernel

    @kernel.setter
    def kernel(self, new_kernel: BaseKernel) -> None:
        if isinstance(new_kernel, BaseKernel):
            self._kernel = new_kernel

    @property
    def backend(self) -> str:
        """
        ['cpu', 'gpu'] : The default backend to use for interpolation
        operations with this dataset.

        'cpu' - Best for small datasets, or cases where a GPU is not available.
        'gpu' - Best for large datasets, with a CUDA-enabled GPU.
        """
        return self._backend

    @backend.setter
    def backend(self, new_backend: str) -> None:
        self._backend = new_backend

    def get_dim(self) -> int:
        """
        Get the dimensionality of the data in this dataframe.

        Returns
        -------
        int
            The number of positional dimensions.
        """
        return 3 if self.zcol is not None else 2

def _default_vxyz(data, vx, vy, vz):
    """
    Utility function to determine the vx, vy, and vz columns to use.

    Parameters
    ----------
    data: SarracenDataFrame
        The particle dataset to use.
    vx, vy, vz: str
        The vx, vy, and vz directional column labels.

    Returns
    -------
    vx, vy, vz: str
        The directional column labels to use.
    """
    if vx is None:
        vx = data.vxcol if not vy == data.vxcol and not vz == data.vxcol else \
             data.vycol if not vy == data.vycol and not vz == data.vycol else data.vzcol
    if vy is None:
        vy = data.vycol if not vx == data.vycol and not vz == data.vycol else \
             data.vxcol if not vx == data.vxcol and not vz == data.vxcol else data.vzcol
    if vz is None:
        vz = data.vzcol if not vx == data.vzcol and not vy == data.vzcol else \
             data.vycol if not vx == data.vycol and not vy == data.vycol else data.vxcol

    return vx, vy, vz

def stoppingtime(rho_dust, rho_gas, v_gas, v_dust, rho_grain, sgrain, gamma, c_s):
    return np.sqrt(np.pi * gamma * 0.125) * rho_grain * sgrain / (rho_dust + rho_gas) / np.sqrt(1 + 0.0703125 * np.pi * (np.linalg.norm(v_gas - v_dust))**2 / c_s**2)

def Stokes_number(data_dust: 'SarracenDataFrame',
                  data_gas: 'SarracenDataFrame',
                  rho_grid: float,
                  sgrain: float,
                  c_s: float,
                  kernel: BaseKernel = None,
                  x_dust: str = None,
                  y_dust: str = None,
                  z_dust: str = None,
                  vx_dust: str = None,
                  vy_dust: str = None,
                  vz_dust: str = None,
                  x_gas: str = None,
                  y_gas: str = None,
                  z_gas: str = None,
                  vx_gas: str = None,
                  vy_gas: str = None,
                  vz_gas: str = None,
                  backend: str = None):
    
    _check_dimension(data_dust, 3)
    _check_dimension(data_gas, 3)

    x_dust, y_dust, z_dust = _default_xyz(data_dust, x_dust, y_dust, z_dust)
    x_gas, y_gas, z_gas = _default_xyz(data_gas, x_gas, y_gas, z_gas)
    vx_dust, vy_dust, vz_dust = _default_vxyz(data_dust, vx_dust, vy_dust, vz_dust)
    vx_gas, vy_gas, vz_gas = _default_vxyz(data_gas, vx_gas, vy_gas, vz_gas)

    _verify_columns(data_dust, [x_dust, y_dust, z_dust, 'h', vx_dust, vy_dust, vz_dust])
    _verify_columns(data_gas, [x_gas, y_gas, z_gas, 'h', vx_gas, vy_gas, vz_gas])

    x_dust_data = data_dust[x_dust].values
    y_dust_data = data_dust[y_dust].values
    z_dust_data = data_dust[z_dust].values
    h_dust_data = data_dust['h'].values
    vx_dust_data = data_dust[vx_dust].values
    vy_dust_data = data_dust[vy_dust].values
    vz_dust_data = data_dust[vz_dust].values
    rho_dust_data = _get_density(data_dust)

    x_gas_data = data_gas[x_gas].values
    y_gas_data = data_gas[y_gas].values
    z_gas_data = data_gas[z_gas].values
    h_gas_data = data_gas['h'].values
    vx_gas_data = data_gas[vx_gas].values
    vy_gas_data = data_gas[vy_gas].values
    vz_gas_data = data_gas[vz_gas].values
    rho_gas_data = _get_density(data_gas)

    # Combine coordinates
    gas_positions = np.vstack((x_gas_data, y_gas_data, z_gas_data)).T
    dust_positions = np.vstack((x_dust_data, y_dust_data, z_dust_data)).T

    gas_velocity = np.vstack((vx_gas_data, vy_gas_data, vz_gas_data)).T
    dust_velocity = np.vstack((vx_dust_data, vy_dust_data, vz_dust_data)).T

    tree = KDTree(gas_positions, leaf_size=10)
    
    # Search using gas particle positions and smoothing lengths
    neighbours = tree.query_radius(dust_positions, r=2 * h_gas_data)

    # Initialize structure to hold gas neighbours of each dust particle
    dust_neighbours = [np.array([], dtype=np.int64) for _ in range(len(dust_positions))]

    # Loop over each gas-dust neighbour pair
    for gas_particle, gas_neighbours in enumerate(neighbours):
        for dust_neighbour in gas_neighbours:
            dust_neighbours[dust_neighbour] = np.append(dust_neighbours[dust_neighbour], gas_particle)

    dust_number = len(x_dust_data)
    dim = 3
    rhog_on_dust = np.zeros(dust_number)
    vx_on_dust = np.zeros(dust_number)
    vy_on_dust = np.zeros(dust_number)
    vz_on_dust = np.zeros(dust_number)

    for ind, array in enumerate(dust_neighbours):
        vx_neighb = 0
        vy_neighb = 0
        vz_neighb = 0    
        neighbor_rho = 0
        
        rho_dust = rho_dust_data[ind]
        r_dust = dust_positions[ind]
        v_dust = dust_velocity[ind]

        for j in array:
            rho_gas = rho_gas_data[j]
            r_gas = gas_positions[j]
            h_gas = h_gas_data[j]
            vx_gas = vx_gas_data[j]
            vy_gas = vy_gas_data[j]
            vz_gas = vz_gas_data[j]

            q = np.linalg.norm(r_gas - r_dust) / h_gas
            normalized_weight = kernel.w(q, dim)

            neighbor_rho += rho_gas * normalized_weight
            vx_neighb += vx_gas * normalized_weight
            vy_neighb += vy_gas * normalized_weight
            vz_neighb += vz_gas * normalized_weight
            
        rhog_on_dust[ind] = neighbor_rho 
        vx_on_dust[ind] = vx_neighb
        vy_on_dust[ind] = vy_neighb
        vz_on_dust[ind] = vz_neighb

    gas_velocity_on_dust = np.vstack((vx_on_dust, vy_on_dust, vz_on_dust)).T

    tstop = stoppingtime(rho_dust_data, rhog_on_dust, gas_velocity_on_dust, dust_velocity, rho_grain, sgrain, gamma, c_s)
    stokes_number = tstop * c_s * rhog_on_dust * (1/3) / data_gas.params['hfact'] * data_gas.params['mass']**(1/3)

    return stokes_number
