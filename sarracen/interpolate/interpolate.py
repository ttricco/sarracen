"""
Contains several interpolation functions which produce interpolated 2D or 1D
arrays of SPH data.
"""
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from ..interpolate import BaseBackend, CPUBackend, GPUBackend
from ..kernels import BaseKernel

from typing import Tuple, Union, Optional, Type, Literal
import warnings


def _default_xy(data: 'SarracenDataFrame',  # noqa: F821
                x: Union[str, None],
                y: Union[str, None]) -> Tuple[str, str]:
    """
    Utility function to determine the x & y columns to use during 2D
    interpolation.

    Parameters
    ----------
    data: SarracenDataFrame
        The particle dataset to interpolate over.
    x, y: str
        The x and y directional column labels passed to the interpolation
        function.

    Returns
    -------
    x, y: str
        The directional column labels to use in interpolation.
    """
    if x is None:
        x = data.xcol if not y == data.xcol else data.ycol
    if y is None:
        y = data.ycol if not x == data.ycol else data.xcol

    return x, y


def _default_xyz(data: 'SarracenDataFrame',  # noqa: F821
                 x: Union[str, None],
                 y: Union[str, None],
                 z: Union[str, None]) -> Tuple[str, str, str]:
    """
    Utility function to determine the x, y and z columns to use during 3-D
    interpolation.

    Parameters
    ----------
    data: SarracenDataFrame
        The particle dataset to interpolate over.
    x, y, z: str
        The x, y and z directional column labels passed to the interpolation
        function.

    Returns
    -------
    x, y, z: str
        The directional column labels to use in interpolation.
    """
    xcol = data.xcol
    ycol = data.ycol
    zcol = data.zcol

    if x is None:
        x = xcol if not y == xcol and not z == xcol else \
            ycol if not y == ycol and not z == ycol else zcol
    if y is None:
        y = ycol if not x == ycol and not z == ycol else \
            xcol if not x == xcol and not z == xcol else zcol
    if z is None:
        z = zcol if not x == zcol and not y == zcol else \
            ycol if not x == ycol and not y == ycol else xcol

    return x, y, z


def _default_bounds(data: 'SarracenDataFrame',  # noqa: F821
                    x: str,
                    y: str,
                    xlim: Union[Tuple[Union[float, None], Union[float, None]],
                                None],
                    ylim: Union[Tuple[Union[float, None], Union[float, None]],
                                None]) -> Tuple[Tuple[float, float],
                                                Tuple[float, float]]:
    """
    Utility function to determine the 2-dimensional boundaries to use in 2D
    interpolation.

    Parameters
    ----------
    data: SarracenDataFrame
        The particle dataset to interpolate over.
    x, y: str
        The directional column labels that will be used in interpolation.
    xlim, ylim: tuple of float
        The minimum and maximum values passed to the interpolation function, in
        particle data space.

    Returns
    -------
    xlim, ylim: tuple of float
        The minimum and maximum values to use in interpolation, in particle
        data space. Defaults to the maximum and minimum values of `x` and `y`,
        snapped to the nearest integer.
    """
    # boundaries of the plot default to the max & min values of the data.
    x_min = xlim[0] if xlim is not None and xlim[0] is not None else None
    y_min = ylim[0] if ylim is not None and ylim[0] is not None else None
    x_max = xlim[1] if xlim is not None and xlim[1] is not None else None
    y_max = ylim[1] if ylim is not None and ylim[1] is not None else None

    x_min = data.loc[:, x].min() if x_min is None else x_min
    y_min = data.loc[:, y].min() if y_min is None else y_min
    x_max = data.loc[:, x].max() if x_max is None else x_max
    y_max = data.loc[:, y].max() if y_max is None else y_max

    return (x_min, x_max), (y_min, y_max)


def _set_pixels(x_pixels: Union[int, None],
                y_pixels: Union[int, None],
                xlim: Tuple[float, float],
                ylim: Tuple[float, float]) -> Tuple[int, int]:
    """
    Utility function to determine the number of pixels to interpolate over in
    2D interpolation.

    Parameters
    ----------
    x_pixels, y_pixels: int
        The number of pixels in the x & y directions passed to the
        interpolation function.
    xlim, ylim: tuple of float
        The minimum and maximum values to use in interpolation, in particle
        data space.

    Returns
    -------
    x_pixels, y_pixels: int
        The number of pixels in the x & y directions to use in 2D
        interpolation.
    """
    # set # of pixels to maintain an aspect ratio that is the same as the
    # underlying bounds of the data.

    dx = xlim[1] - xlim[0]
    dy = ylim[1] - ylim[0]

    if y_pixels is None:
        if x_pixels is None:
            x_pixels = 512
        y_pixels = int(np.rint(x_pixels * (dy / dx)))
    elif x_pixels is None:
        x_pixels = int(np.rint(y_pixels * (dx / dy)))

    return x_pixels, y_pixels


def _verify_columns(data: 'SarracenDataFrame',  # noqa: F821
                    x: str,
                    y: str) -> None:
    """
    Verify that columns required for 2D interpolation exist in `data`.

    Parameters
    ----------
    data: SarracenDataFrame
        The particle dataset to interpolate over.
    x, y: str
        The directional column labels that will be used in interpolation.

    Raises
    -------
    KeyError
        If `target`, `x`, `y`, mass, density, or smoothing length columns do
        not exist in `data`.
    """
    if x not in data.columns:
        raise KeyError(f"x-directional column '{x}' does not exist in the "
                       f"provided dataset.")
    if y not in data.columns:
        raise KeyError(f"y-directional column '{y}' does not exist in the "
                       f"provided dataset.")
    if data.hcol is None:
        raise KeyError("Smoothing length column does not exist in the "
                       "provided dataset.")
    if data.mcol is None and 'mass' not in data.params:
        raise KeyError("Missing particle mass data in this "
                       "SarracenDataFrame.")
    if data.rhocol is None and 'hfact' not in data.params:
        raise KeyError("Density cannot be derived from the columns in "
                       "this SarracenDataFrame.")


def _check_boundaries(x_pixels: int,
                      y_pixels: int,
                      xlim: Tuple[float, float],
                      ylim: Tuple[float, float]) -> None:
    """
    Verify that the pixel count and boundaries of a 2D plot describe a valid
    region.

    Parameters
    ----------
    x_pixels, y_pixels: int
        The number of pixels in the x & y directions passed to the
        interpolation function.
    xlim, ylim: tuple of float
        The minimum and maximum values to use in interpolation, in particle
        data space.

    Raises
    ------
    ValueError
        If `x_pixels` or `y_pixels` are less than or equal to zero, or if the
        specified `x` and `y` minimum and maximum values result in an invalid
        region.
    """
    if xlim[1] - xlim[0] <= 0:
        raise ValueError("`xlim` max must be greater than min!")
    if ylim[1] - ylim[0] <= 0:
        raise ValueError("`ylim` max must be greater than min!")
    if x_pixels <= 0:
        raise ValueError("`x_pixels` must be greater than zero!")
    if y_pixels <= 0:
        raise ValueError("`y_pixels` must be greater than zero!")


def _check_dimension(data: 'SarracenDataFrame',  # noqa: F821
                     dim: Literal[2, 3]) -> None:
    """
    Verify that a given dataset describes data with a required number of
    dimensions.

    Parameters
    ----------
    data: SarracenDataFrame
        The particle dataset to interpolate over.
    dim: [2, 3]
        The number of required dimensions.

    Returns
    -------
    ValueError
        If the dataset is not `dim`-dimensional or `dim` is not 2 or 3.
    """
    if dim not in [2, 3]:
        raise ValueError("`dim` must be 2 or 3.")
    if data.get_dim() != dim:
        raise ValueError(f"Dataset is not {dim}-dimensional.")


def _rotate_data(data: 'SarracenDataFrame',  # noqa: F821
                 x: str,
                 y: str,
                 z: str,
                 rotation: Union[np.ndarray, list, Rotation, None],
                 rot_origin: Union[np.ndarray, list, pd.Series,
                                   str, None]) -> Tuple[np.ndarray,
                                                        np.ndarray,
                                                        np.ndarray]:
    """
    Rotate vector data in a particle dataset.

    Parameters
    ----------
    data: SarracenDataFrame
        The particle dataset to interpolate over.
    x, y, z: str
        Directional column labels containing each dimension of the vector data.
    rotation: array_like or SciPy Rotation
        The rotation to apply to the vector data. If defined as an array, the
        order of rotations is [z, y, x] in degrees
    rot_origin: array_like or ['com', 'midpoint']
        Point of rotation of the data. Only applies to 3D datasets. If
        array_like, then the [x, y, z] coordinates specify the point around
        which the data is rotated. If 'com', then data is rotated around the
        centre of mass. If 'midpoint', then data is rotated around the
        midpoint, that is, min + max / 2. Defaults to the midpoint.

    Returns
    -------
    x_data, y_data, z_data: ndarray
        The rotated x, y, and z directional data.
    """
    x_data = data[x].to_numpy()
    y_data = data[y].to_numpy()
    z_data = data[z].to_numpy()
    if rotation is not None:
        if not isinstance(rotation, Rotation):
            rotation_obj = Rotation.from_euler('zyx',
                                               rotation,
                                               degrees=True)
        else:
            rotation_obj = rotation

        vectors = data[[x, y, z]].to_numpy()

        # warn whenever rotation is applied
        msg = ("The default rotation point is currently the midpoint of the "
               "x/y/z bounds, but will change to [x, y, z] = [0, 0, 0] in "
               "Sarracen version 1.3.0.")
        warnings.warn(msg, DeprecationWarning, stacklevel=6)

        if rot_origin is None:
            # rot_origin = [0, 0, 0]
            rot_origin_arr = (vectors.min(0) + vectors.max(0)) / 2
        elif rot_origin == 'com':
            rot_origin_arr = data.centre_of_mass()
        elif rot_origin == 'midpoint':
            rot_origin_arr = (vectors.min(0) + vectors.max(0)) / 2
        elif not isinstance(rot_origin, (list, pd.Series, np.ndarray)):
            raise ValueError("rot_origin should be an [x, y, z] point or "
                             "'com' or 'midpoint'")
        elif len(rot_origin) != 3:
            raise ValueError("rot_origin should specify [x, y, z] point.")
        else:
            rot_origin_arr = rot_origin
        vectors = vectors - rot_origin_arr
        vectors = rotation_obj.apply(vectors)
        vectors = vectors + rot_origin_arr

        x_data = vectors[:, 0]
        y_data = vectors[:, 1]
        z_data = vectors[:, 2]

    return x_data, y_data, z_data


def _rotate_xyz(data: 'SarracenDataFrame',  # noqa: F821
                x: str,
                y: str,
                z: str,
                rotation: Union[np.ndarray, list, Rotation, None],
                rot_origin: Union[np.ndarray, list,
                                  str, None]) -> Tuple[np.ndarray,
                                                       np.ndarray,
                                                       np.ndarray]:
    """
    Rotate positional data in a particle dataset.

    Differs from _rotate_data() in that the returned data values are shuffled
    to ensure that the rotation is always applied to the global x, y, and z
    columns of the dataset, no matter the order of x, y, and z provided to
    this function.

    Parameters
    ----------
    data: SarracenDataFrame
        The particle dataset to interpolate over.
    x, y, z: str
        Directional column labels containing the positional column labels
    rotation: array_like or SciPy Rotation, optional
        The rotation to apply to the data. If defined as an array, the
        order of rotations is [z, y, x] in degrees
    rot_origin: array_like or ['com', 'midpoint'], optional
        Point of rotation of the data. Only applies to 3D datasets. If
        array_like, then the [x, y, z] coordinates specify the point around
        which the data is rotated. If 'com', then data is rotated around the
        centre of mass. If 'midpoint', then data is rotated around the
        midpoint, that is, min + max / 2. Defaults to the midpoint.

    Returns
    -------
    x_data, y_data, z_data: ndarray
        The rotated x, y, and z directional data.
    """
    rotated_x, rotated_y, rotated_z = _rotate_data(data, data.xcol, data.ycol,
                                                   data.zcol, rotation,
                                                   rot_origin)
    x_data = rotated_x if x == data.xcol else \
        rotated_y if x == data.ycol else \
        rotated_z if x == data.zcol else data[x]
    y_data = rotated_x if y == data.xcol else \
        rotated_y if y == data.ycol else \
        rotated_z if y == data.zcol else data[y]
    z_data = rotated_x if z == data.xcol else \
        rotated_y if z == data.ycol else \
        rotated_z if z == data.zcol else data[z]

    return x_data, y_data, z_data


def _corotate(corotation: Union[np.ndarray, list],
              rotation: Union[np.ndarray, list,
                              Rotation, None]) -> Tuple[Union[np.ndarray,
                                                              list,
                                                              Rotation],
                                                        Union[np.ndarray,
                                                              list]]:
    """
    Calculates the rotation matrix for a corotating frame.

    Parameters
    ----------
    corotation: array_like
        The x, y, z coordinates of two locations which determines the
        corotating frame. Each coordinate is also array_like.
    rotation: array_like or SciPy Rotation, optional
        An additional rotation to apply to the corotating frame.

    Returns
    -------
    rotation: array_like or SciPy Rotation
        The rotation to apply to the data before interpolation.
    rot_origin: array_like
        Point of rotation of the data.

    """
    corotation[1][0] -= corotation[0][0]
    corotation[1][1] -= corotation[0][1]
    corotation[1][2] -= corotation[0][2]

    rot_origin = corotation[0]
    angle = -np.arctan2(corotation[1][1], corotation[1][0])
    if rotation is None:
        rotation = np.array([angle * 180 / np.pi, 0, 0])
    else:
        if isinstance(rotation, Rotation):
            rotation = rotation.as_rotvec(degrees=True)
        rotation = np.array([angle * 180/np.pi + rotation[0],
                             rotation[1],
                             rotation[2]])
        rotation = Rotation.from_euler('zyx', rotation, degrees=True)

    return rotation, rot_origin


def _get_mass(data: 'SarracenDataFrame') -> Union[np.ndarray,  # noqa: F821
                                                  float]:
    if data.mcol is None:
        return data.params['mass']

    return data[data.mcol].to_numpy()


def _get_density(data: 'SarracenDataFrame') -> np.ndarray:  # noqa: F821
    if data.rhocol is None:
        hfact = data.params['hfact']
        mass = _get_mass(data)
        return ((hfact / data[data.hcol])**(data.get_dim()) * mass).to_numpy()

    return data[data.rhocol].to_numpy()


def _get_weight(data: 'SarracenDataFrame',  # noqa: F821
                target: Union[str, np.ndarray],
                dens_weight: bool) -> np.ndarray:

    if type(target) is str:
        if target == 'rho':
            target_data = _get_density(data)
        else:
            if target not in data.columns:
                raise KeyError(f"Target column '{target}' does not exist in "
                               f"provided dataset.")

            target_data = data[target].to_numpy()
    elif type(target) is np.ndarray:
        target_data = target
    else:
        raise KeyError(f"Target must be of type str or ndarray. "
                       f"Found: '{type(target)}'")

    mass_data = _get_mass(data)
    if dens_weight:
        return target_data * mass_data
    else:
        rho_data = _get_density(data)
        return target_data * mass_data / rho_data


def _get_smoothing_lengths(data: 'SarracenDataFrame',  # noqa: F821
                           hmin: bool,
                           x_pixels: int,
                           y_pixels: int,
                           xlim: Tuple[float, float],
                           ylim: Tuple[float, float]) -> np.ndarray:
    """ Return smoothing lengths, imposing a min length if hmin is True. """

    if hmin:
        pix_size = (xlim[1] - xlim[0]) / x_pixels
        pix_size = np.maximum(pix_size, (ylim[1] - ylim[0]) / y_pixels)
        h_data = np.maximum(data[data.hcol].to_numpy(), 0.5 * pix_size)
    else:
        h_data = data[data.hcol].to_numpy()

    return h_data


def interpolate_2d(data: 'SarracenDataFrame',  # noqa: F821
                   target: str,
                   x: Union[str, None] = None,
                   y: Union[str, None] = None,
                   kernel: Union[BaseKernel, None] = None,
                   x_pixels: Union[int, None] = None,
                   y_pixels: Union[int, None] = None,
                   xlim: Optional[Tuple[Optional[float],
                                        Optional[float]]] = None,
                   ylim: Optional[Tuple[Optional[float],
                                        Optional[float]]] = None,
                   exact: bool = False,
                   backend: Union[str, None] = None,
                   dens_weight: bool = False,
                   normalize: bool = True,
                   hmin: bool = False) -> np.ndarray:
    """
    Interpolate particle data across two directional axes to a 2D grid of
    pixels.

    Interpolate the data within a SarracenDataFrame to a 2D grid, by
    interpolating the values of a target variable. The contributions of all
    particles near the interpolation area are summed and stored to a 2D grid.

    Parameters
    ----------
    data : SarracenDataFrame
        Particle data, in a SarracenDataFrame.
    target: str
        Column label of the target smoothing data.
    x, y: str, optional
        Column labels of the directional axes. Defaults to the x & y columns
        detected in `data`.
    kernel: BaseKernel, optional
        Kernel to use for smoothing the target data. Defaults to the kernel
        specified in `data`.
    x_pixels, y_pixels: int, optional
        Number of pixels in the output image in the x & y directions. Default
        values are chosen to keep a consistent aspect ratio.
    xlim, ylim: tuple of float, optional
        The minimum and maximum values to use in interpolation, in particle
        data space. Defaults to the minimum and maximum values of `x` and `y`.
    exact: bool
        Whether to use exact interpolation of the data.
    backend: ['cpu', 'gpu']
        The computation backend to use when interpolating this data. Defaults
        to 'gpu' if CUDA is enabled, otherwise 'cpu' is used. A manually
        specified backend in `data` will override the default.
    dens_weight: bool, optional
        If True, the target will be multiplied by density. Defaults to False.
    normalize: bool, optional
        If True, will normalize the interpolation. Defaults to False (this may
        change in future versions).
    hmin: bool, optional
        If True, a minimum smoothing length of 0.5 * pixel size will be
        imposed. This ensures each particle contributes to at least one grid
        cell / pixel. Defaults to False (this may change in a future verison).

    Returns
    -------
    ndarray (2-Dimensional)
        The interpolated output image, in a 2-dimensional numpy array.
        Dimensions are structured in reverse order, where (x, y) -> [y, x].

    Raises
    -------
    ValueError
        If `x_pixels` or `y_pixels` are less than or equal to zero, or if the
        specified `x` and `y` minimum and maximum values result in an invalid
        region, or if `data` is not 2-dimensional.
    KeyError
        If `target`, `x`, `y`, mass, density, or smoothing length columns do
        not exist in `data`.
    """
    _check_dimension(data, 2)
    x, y = _default_xy(data, x, y)
    _verify_columns(data, x, y)

    xlim, ylim = _default_bounds(data, x, y, xlim, ylim)
    x_pixels, y_pixels = _set_pixels(x_pixels, y_pixels, xlim, ylim)
    _check_boundaries(x_pixels, y_pixels, xlim, ylim)
    w_data = _get_weight(data, target, dens_weight)

    kernel = kernel if kernel is not None else data.kernel
    backend = backend if backend is not None else data.backend

    h_data = _get_smoothing_lengths(data, hmin, x_pixels, y_pixels, xlim, ylim)

    grid = get_backend(backend)\
        .interpolate_2d_render(data[x].to_numpy(), data[y].to_numpy(),
                               w_data, h_data, kernel.w, kernel.get_radius(),
                               x_pixels, y_pixels, xlim[0], xlim[1],
                               ylim[0], ylim[1], exact)

    if normalize:
        w_norm = _get_weight(data, np.array([1] * len(w_data)), dens_weight)
        norm_grid = get_backend(backend)\
            .interpolate_2d_render(data[x].to_numpy(), data[y].to_numpy(),
                                   w_norm, h_data, kernel.w,
                                   kernel.get_radius(), x_pixels, y_pixels,
                                   xlim[0], xlim[1], ylim[0], ylim[1], exact)
        grid = np.nan_to_num(grid / norm_grid)

    return grid


def interpolate_2d_vec(data: 'SarracenDataFrame',  # noqa: F821
                       target_x: str,
                       target_y: str,
                       x: Union[str, None] = None,
                       y: Union[str, None] = None,
                       kernel: Union[BaseKernel, None] = None,
                       x_pixels: Union[int, None] = None,
                       y_pixels: Union[int, None] = None,
                       xlim: Optional[Tuple[Optional[float],
                                            Optional[float]]] = None,
                       ylim: Optional[Tuple[Optional[float],
                                            Optional[float]]] = None,
                       exact: bool = False,
                       backend: Union[str, None] = None,
                       dens_weight: bool = False,
                       normalize: bool = True,
                       hmin: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate vector particle data across two directional axes to a 2D grid
    of particles.

    Interpolate the data within a SarracenDataFrame to a 2D grid, by
    interpolating the values of a target vector. The contributions of all
    vectors near the interpolation area are summed and stored to a 2D grid.

    Parameters
    ----------
    data : SarracenDataFrame
        Particle data, in a SarracenDataFrame.
    target_x, target_y: str
        Column labels of the target vector.
    x, y: str, optional
        Column labels of the directional axes. Defaults to the x & y columns
        detected in `data`.
    kernel: BaseKernel, optional
        Kernel to use for smoothing the target data. Defaults to the kernel
        specified in `data`.
    x_pixels, y_pixels: int, optional
        Number of pixels in the output image in the x & y directions. Default
        values are chosen to keep a consistent aspect ratio.
    xlim, ylim: tuple of float, optional
        The minimum and maximum values to use in interpolation, in particle
        data space. Defaults to the minimum and maximum values of `x` and `y`.
    exact: bool
        Whether to use exact interpolation of the data.
    backend: ['cpu', 'gpu']
        The computation backend to use when interpolating this data. Defaults
        to 'gpu' if CUDA is enabled, otherwise 'cpu' is used. A manually
        specified backend in `data` will override the default.
    dens_weight: bool, optional
        If True, the target will be multiplied by density. Defaults to False.
    normalize: bool, optional
        If True, will normalize the interpolation. Defaults to False (this may
        change in future versions).
    hmin: bool, optional
        If True, a minimum smoothing length of 0.5 * pixel size will be
        imposed. This ensures each particle contributes to at least one grid
        cell / pixel. Defaults to False (this may change in a future verison).

    Returns
    -------
    output_x, output_y: ndarray (2-Dimensional)
        The interpolated output images, in a 2-dimensional numpy arrays.
        Dimensions are structured in reverse order, where (x, y) -> [y, x].

    Raises
    -------
    ValueError
        If `x_pixels` or `y_pixels` are less than or equal to zero, or if the
        specified `x` and `y` minimum and maximum values result in an invalid
        region, or if `data` is not 2-dimensional.
    KeyError
        If `target_x`, `target_y`, `x`, `y`, mass, density, or smoothing
        length columns do not exist in `data`.
    """
    _check_dimension(data, 2)
    x, y = _default_xy(data, x, y)
    _verify_columns(data, x, y)

    xlim, ylim = _default_bounds(data, x, y, xlim, ylim)
    x_pixels, y_pixels = _set_pixels(x_pixels, y_pixels, xlim, ylim)
    _check_boundaries(x_pixels, y_pixels, xlim, ylim)

    wx_data = _get_weight(data, target_x, dens_weight)
    wy_data = _get_weight(data, target_y, dens_weight)

    kernel = kernel if kernel is not None else data.kernel
    backend = backend if backend is not None else data.backend

    h_data = _get_smoothing_lengths(data, hmin, x_pixels, y_pixels, xlim, ylim)

    gridx, gridy = get_backend(backend)\
        .interpolate_2d_render_vec(data[x].to_numpy(), data[y].to_numpy(),
                                   wx_data, wy_data, h_data, kernel.w,
                                   kernel.get_radius(), x_pixels, y_pixels,
                                   xlim[0], xlim[1], ylim[0], ylim[1], exact)

    if normalize:
        wx_norm = _get_weight(data, np.array([1] * len(wx_data)), dens_weight)
        wy_norm = _get_weight(data, np.array([1] * len(wy_data)), dens_weight)
        norm_gridx, norm_gridy = get_backend(backend)\
            .interpolate_2d_render_vec(data[x].to_numpy(), data[y].to_numpy(),
                                       wx_norm, wy_norm, h_data, kernel.w,
                                       kernel.get_radius(), x_pixels, y_pixels,
                                       xlim[0], xlim[1], ylim[0], ylim[1],
                                       exact)
        gridx = np.nan_to_num(gridx / norm_gridx)
        gridy = np.nan_to_num(gridy / norm_gridy)

    return (gridx, gridy)


def interpolate_2d_line(data: 'SarracenDataFrame',  # noqa: F821
                        target: str,
                        x: Union[str, None] = None,
                        y: Union[str, None] = None,
                        kernel: Union[BaseKernel, None] = None,
                        pixels: Union[int, None] = None,
                        xlim: Union[Tuple[float, float], None] = None,
                        ylim: Union[Tuple[float, float], None] = None,
                        backend: Union[str, None] = None,
                        dens_weight: bool = False,
                        normalize: bool = True,
                        hmin: bool = False) -> np.ndarray:
    """
    Interpolate particle data across two directional axes to a 1D cross-section
    line.

    Interpolate the data within a SarracenDataFrame to a 1D line, by
    interpolating the values of a target variable. The contributions of all
    particles near the specified line are summed and stored to a 1-dimensional
    array.

    Parameters
    ----------
    data : SarracenDataFrame
        Particle data, in a SarracenDataFrame.
    target: str
         Column label of the target smoothing data.
    x, y: str, optional
        Column labels of the directional axes. Defaults to the x & y columns
        detected in `data`.
    kernel: BaseKernel, optional
        Kernel to use for smoothing the target data. Defaults to the kernel
        specified in `data`.
    pixels: int, optional
        Number of points in the resulting line plot in the x-direction.
    xlim, ylim: tuple of float, optional
        Starting and ending coordinates of the cross-section line (in particle
        data space). Defaults to the minimum and maximum values of `x` and `y`.
    backend: ['cpu', 'gpu']
        The computation backend to use when interpolating this data. Defaults
        to 'gpu' if CUDA is enabled, otherwise 'cpu' is used. A manually
        specified backend in `data` will override the default.
    dens_weight: bool, optional
        If True, the target will be multiplied by density. Defaults to False.
    normalize: bool, optional
        If True, will normalize the interpolation. Defaults to True.
    hmin: bool, optional
        If True, a minimum smoothing length of 0.5 * pixel size will be
        imposed. This ensures each particle contributes to at least one grid
         cell / pixel. Defaults to False (this may change in a future verison).

    Returns
    -------
    np.ndarray (1-Dimensional)
        The resulting interpolated output.

    Raises
    -------
    ValueError
        If `x_pixels` or `y_pixels` are less than or equal to zero, or if the
        specified `xlim` and `ylim` values are all the same (indicating a
        zero-length cross-section), or if `data` is not 2-dimensional.
    KeyError
        If `target`, `x`, `y`, mass, density, or smoothing length columns do
        not exist in `data`.
    """
    _check_dimension(data, 2)
    x, y = _default_xy(data, x, y)
    _verify_columns(data, x, y)

    w_data = _get_weight(data, target, dens_weight)

    if isinstance(xlim, float) or isinstance(xlim, int):
        xlim = xlim, xlim
    if isinstance(ylim, float) or isinstance(ylim, int):
        ylim = ylim, ylim

    xlim, ylim = _default_bounds(data, x, y, xlim, ylim)
    if xlim[0] == xlim[1] and ylim[0] == ylim[1]:
        raise ValueError('Zero length cross section!')

    kernel = kernel if kernel is not None else data.kernel
    backend = backend if backend is not None else data.backend
    pixels = pixels if pixels is not None else 512

    if pixels <= 0:
        raise ValueError('pixcount must be greater than zero!')

    if hmin:
        pix_size = np.sqrt((xlim[1] - xlim[0])**2
                           + (ylim[1] - ylim[0])**2) / pixels
        h_data = np.maximum(data[data.hcol].to_numpy(), 0.5 * pix_size)
    else:
        h_data = data[data.hcol].to_numpy()

    grid = get_backend(backend) \
        .interpolate_2d_line(data[x].to_numpy(), data[y].to_numpy(),
                             w_data, h_data, kernel.w, kernel.get_radius(),
                             pixels, xlim[0], xlim[1], ylim[0], ylim[1])

    if normalize:
        w_norm = _get_weight(data, np.array([1] * len(w_data)), dens_weight)
        norm_grid = get_backend(backend) \
            .interpolate_2d_line(data[x].to_numpy(), data[y].to_numpy(),
                                 w_norm, h_data, kernel.w,
                                 kernel.get_radius(), pixels, xlim[0],
                                 xlim[1], ylim[0], ylim[1])
        grid = np.nan_to_num(grid / norm_grid)

    return grid


def interpolate_3d_line(data: 'SarracenDataFrame',  # noqa: F821
                        target: str,
                        x: Union[str, None] = None,
                        y: Union[str, None] = None,
                        z: Union[str, None] = None,
                        kernel: Union[BaseKernel, None] = None,
                        pixels: Union[int, None] = None,
                        xlim: Union[Tuple[float, float], None] = None,
                        ylim: Union[Tuple[float, float], None] = None,
                        zlim: Union[Tuple[float, float], None] = None,
                        backend: Union[str, None] = None,
                        dens_weight: bool = False,
                        normalize: bool = True,
                        hmin: bool = False) -> np.ndarray:
    """
    Interpolate vector particle data across three directional axes to a 1D
    line.

    Interpolate the data within a SarracenDataFrame to a 1D line, by
    interpolating the values of a target variable. The contributions of all
    particles near the interpolation line are summed and stored to a 1D array.

    Parameters
    ----------
    data : SarracenDataFrame
            Particle data, in a SarracenDataFrame.
    target: str
        Column label of the target variable.
    x, y, z: str, optional
        Column labels of the directional axes. Defaults to the x, y & z columns
        detected in `data`.
    kernel: BaseKernel, optional
        Kernel to use for smoothing the target data. Defaults to the kernel
        specified in `data`.
    pixels: int, optional
        Number of pixels in the output image in the x & y directions. Default
        values are chosen to keep a consistent aspect ratio.
    xlim, ylim, zlim: tuple of float, optional
        Starting and ending coordinates of the cross-section line (in particle
        data space). Defaults to the minimum and maximum values of `x`, `y`,
        and `z`.
    backend: ['cpu', 'gpu']
        The computation backend to use when interpolating this data. Defaults
        to 'gpu' if CUDA is enabled, otherwise 'cpu' is used. A manually
        specified backend in `data` will override the default.
    dens_weight: bool, optional
       If True, the target will be multiplied by density. Defaults to False.
    normalize: bool, optional
        If True, will normalize the interpolation. Defaults to True.
    hmin: bool, optional
        If True, a minimum smoothing length of 0.5 * pixel size will be
        imposed. This ensures each particle contributes to at least one grid
        cell / pixel. Defaults to False (this may change in a future verison).

    Returns
    -------
    output: ndarray (1-Dimensional)
        The interpolated output line.

    Raises
    -------
    ValueError
        If `pixels` are less than or equal to zero, or if the specified `x`,
        `y`, and `z` minimum and maximum values result in a zero area
        cross-section, or if `data` is not 3-dimensional.
    KeyError
        If `target`, `x`, `y`, mass, density, or smoothing length data does
        not exist in `data`.
    """
    _check_dimension(data, 3)
    x, y, z = _default_xyz(data, x, y, z)
    _verify_columns(data, x, y)

    w_data = _get_weight(data, target, dens_weight)

    if isinstance(xlim, float) or isinstance(xlim, int):
        xlim = xlim, xlim
    if isinstance(ylim, float) or isinstance(ylim, int):
        ylim = ylim, ylim
    if isinstance(zlim, float) or isinstance(zlim, int):
        zlim = zlim, zlim

    z1 = data.loc[:, z].min() if zlim is None or zlim[0] is None else zlim[0]
    z2 = data.loc[:, z].min() if zlim is None or zlim[1] is None else zlim[1]
    zlim = z1, z2

    xlim, ylim = _default_bounds(data, x, y, xlim, ylim)
    if ylim[1] == ylim[0] and xlim[1] == xlim[0] and zlim[1] == zlim[0]:
        raise ValueError('Zero length cross section!')

    kernel = kernel if kernel is not None else data.kernel
    backend = backend if backend is not None else data.backend
    pixels = pixels if pixels is not None else 512

    if pixels <= 0:
        raise ValueError('pixcount must be greater than zero!')

    if hmin:
        pix_size = np.sqrt((xlim[1] - xlim[0])**2
                           + (ylim[1] - ylim[0])**2
                           + (zlim[1] - zlim[0])**2) / pixels
        h_data = np.maximum(data[data.hcol].to_numpy(), 0.5 * pix_size)
    else:
        h_data = data[data.hcol].to_numpy()

    grid = get_backend(backend) \
        .interpolate_3d_line(data[x].to_numpy(), data[y].to_numpy(),
                             data[z].to_numpy(), w_data, h_data, kernel.w,
                             kernel.get_radius(), pixels, xlim[0], xlim[1],
                             ylim[0], ylim[1], zlim[0], zlim[1])

    if normalize:
        w_norm = _get_weight(data, np.array([1] * len(w_data)), dens_weight)
        norm_grid = get_backend(backend) \
            .interpolate_3d_line(data[x].to_numpy(), data[y].to_numpy(),
                                 data[z].to_numpy(), w_norm, h_data, kernel.w,
                                 kernel.get_radius(), pixels, xlim[0], xlim[1],
                                 ylim[0], ylim[1], zlim[0], zlim[1])
        grid = np.nan_to_num(grid / norm_grid)

    return grid


def interpolate_3d_proj(data: 'SarracenDataFrame',  # noqa: F821
                        target: str,
                        x: Union[str, None] = None,
                        y: Union[str, None] = None,
                        kernel: Union[BaseKernel, None] = None,
                        integral_samples: int = 1000,
                        corotation: Union[np.ndarray, list, None] = None,
                        rotation: Union[np.ndarray, list,
                                        Rotation, None] = None,
                        rot_origin: Union[np.ndarray, list, str, None] = None,
                        x_pixels: Union[int, None] = None,
                        y_pixels: Union[int, None] = None,
                        xlim: Union[Tuple[float, float], None] = None,
                        ylim: Union[Tuple[float, float], None] = None,
                        exact: bool = False,
                        backend: Union[str, None] = None,
                        dens_weight: Union[bool, None] = None,
                        normalize: bool = True,
                        hmin: bool = False) -> np.ndarray:
    """
    Interpolate 3D particle data to a 2D grid of pixels.

    Interpolates three-dimensional particle data in a SarracenDataFrame. The
    data is interpolated to a 2D grid of pixels, by summing contributions in
    columns which span the z-axis.

    Parameters
    ----------
    data : SarracenDataFrame
        Particle data, in a SarracenDataFrame.
    target: str
        Column label of the target smoothing data.
    x, y: str, optional
        Column labels of the directional axes. Defaults to the x & y columns
        detected in `data`.
    kernel: BaseKernel, optional
        Kernel to use for smoothing the target data. Defaults to the kernel
        specified in `data`.
    integral_samples: int, optional
        Number of sample points to take when approximating the 2D column
        kernel.
    corotation: array_like
        The x, y, z coordinates of two locations which determines the
        corotating frame.
    rotation: array_like or SciPy Rotation, optional
        The rotation to apply to the data before interpolation. If defined as
        an array, the order of rotations is [z, y, x] in degrees.
    rot_origin: array_like or ['com', 'midpoint'], optional
        Point of rotation of the data. Only applies to 3D datasets. If
        array_like, then the [x, y, z] coordinates specify the point around
        which the data is rotated. If 'com', then data is rotated around the
        centre of mass. If 'midpoint', then data is rotated around the
        midpoint, that is, min + max / 2. Defaults to the midpoint.
    x_pixels, y_pixels: int, optional
        Number of pixels in the output image in the x & y directions. Default
        values are chosen to keep a consistent aspect ratio.
    xlim, ylim: tuple of float, optional
        The minimum and maximum values to use in interpolation, in particle
        data space. Defaults to the minimum and maximum values of `x` and `y`.
    exact: bool
        Whether to use exact interpolation of the data.
    backend: ['cpu', 'gpu']
        The computation backend to use when interpolating this data. Defaults
        to 'gpu' if CUDA is enabled, otherwise 'cpu' is used. A manually
        specified backend in `data` will override the default.
    dens_weight: bool, optional
        If True, the target will be multiplied by density. Defaults to True for
        column-integrated views, when the target is not density, and False for
        everything else.
    normalize: bool, optional
        If True, will normalize the interpolation. Defaults to True.
    hmin: bool, optional
        If True, a minimum smoothing length of 0.5 * pixel size will be
        imposed. This ensures each particle contributes to at least one grid
        cell / pixel. Defaults to False (this may change in a future verison).

    Returns
    -------
    ndarray (2-Dimensional)
        The interpolated output image, in a 2-dimensional numpy array.
        Dimensions are structured in reverse order, where (x, y) -> [y, x].

    Raises
    -------
    ValueError
        If `x_pixels` or `y_pixels` are less than or equal to zero, or if the
        specified `x` and `y` minimum and maximums result in an invalid region,
        or if the provided data is not 3-dimensional.
    KeyError
        If `target`, `x`, `y`, mass, density, or smoothing length columns do
        not exist in `data`.

    Notes
    -----
    Since the direction of integration is assumed to be straight across the
    z-axis, the z-axis column is not required for this type of interpolation.
    """
    _check_dimension(data, 3)
    x, y, z = _default_xyz(data, x, y, None)
    _verify_columns(data, x, y)

    if dens_weight is None:
        dens_weight = (target != 'rho')

    w_data = _get_weight(data, target, dens_weight)

    xlim, ylim = _default_bounds(data, x, y, xlim, ylim)
    x_pixels, y_pixels = _set_pixels(x_pixels, y_pixels, xlim, ylim)
    _check_boundaries(x_pixels, y_pixels, xlim, ylim)

    if corotation is not None:
        rotation, rot_origin = _corotate(corotation, rotation)

    x_data, y_data, z_data = _rotate_xyz(data, x, y, z, rotation, rot_origin)
    kernel = kernel if kernel is not None else data.kernel
    backend = backend if backend is not None else data.backend

    weight_function = kernel.get_column_kernel_func(integral_samples)

    h_data = _get_smoothing_lengths(data, hmin, x_pixels, y_pixels, xlim, ylim)

    grid = get_backend(backend) \
        .interpolate_3d_projection(x_data, y_data, w_data, h_data,
                                   weight_function, kernel.get_radius(),
                                   x_pixels, y_pixels,
                                   xlim[0], xlim[1], ylim[0], ylim[1], exact)

    if normalize:
        w_norm = _get_weight(data, np.array([1] * len(w_data)), dens_weight)
        norm_grid = get_backend(backend) \
            .interpolate_3d_projection(x_data, y_data, w_norm, h_data,
                                       weight_function, kernel.get_radius(),
                                       x_pixels, y_pixels, xlim[0], xlim[1],
                                       ylim[0], ylim[1], exact)
        grid = np.nan_to_num(grid / norm_grid)

    return grid


def interpolate_3d_vec(data: 'SarracenDataFrame',  # noqa: F821
                       target_x: str,
                       target_y: str,
                       target_z: str,
                       x: Union[str, None] = None,
                       y: Union[str, None] = None,
                       kernel: Union[BaseKernel, None] = None,
                       integral_samples: int = 1000,
                       rotation: Union[np.ndarray, list,
                                       Rotation, None] = None,
                       rot_origin: Union[np.ndarray, list, str, None] = None,
                       x_pixels: Union[int, None] = None,
                       y_pixels: Union[int, None] = None,
                       xlim: Optional[Tuple[Optional[float],
                                            Optional[float]]] = None,
                       ylim: Optional[Tuple[Optional[float],
                                            Optional[float]]] = None,
                       exact: bool = False,
                       backend: Union[str, None] = None,
                       dens_weight: bool = False,
                       normalize: bool = True,
                       hmin: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate 3D vector particle data to a 2D grid of pixels.

    Interpolates three-dimensional vector particle data in a SarracenDataFrame.
    The data is interpolated to a 2D grid of pixels, by summing contributions
    in columns which span the z-axis.

    Parameters
    ----------
    data : SarracenDataFrame
        Particle data, in a SarracenDataFrame.
    target_x, target_y, target_z: str
        Column labels of the target vector.
    x, y: str, optional
        Column labels of the directional axes. Defaults to the x & y columns
        detected in `data`.
    kernel: BaseKernel, optional
        Kernel to use for smoothing the target data. Defaults to the kernel
        specified in `data`.
    integral_samples: int, optional
        Number of sample points to take when approximating the 2D column
        kernel.
    rotation: array_like or SciPy Rotation, optional
        The rotation to apply to the data before interpolation. If defined as
        an array, the order of rotations is [z, y, x] in degrees.
    rot_origin: array_like or ['com', 'midpoint'], optional
        Point of rotation of the data. Only applies to 3D datasets. If
        array_like, then the [x, y, z] coordinates specify the point around
        which the data is rotated. If 'com', then data is rotated around the
        centre of mass. If 'midpoint', then data is rotated around the
        midpoint, that is, min + max / 2. Defaults to the midpoint.
    x_pixels, y_pixels: int, optional
        Number of pixels in the output image in the x & y directions. Default
        values are chosen to keep a consistent aspect ratio.
    xlim, ylim: tuple of float, optional
        The minimum and maximum values to use in interpolation, in particle
        data space. Defaults to the minimum and maximum values of `x` and `y`.
    exact: bool
        Whether to use exact interpolation of the data.
    backend: ['cpu', 'gpu']
        The computation backend to use when interpolating this data. Defaults
        to 'gpu' if CUDA is enabled, otherwise 'cpu' is used. A manually
        specified backend in `data` will override the default.
    dens_weight: bool, optional
        If True, the target will be multiplied by density. Defaults to False.
    normalize: bool, optional
        If True, will normalize the interpolation. Defaults to True.
    hmin: bool, optional
        If True, a minimum smoothing length of 0.5 * pixel size will be
        imposed. This ensures each particle contributes to at least one grid
        cell / pixel. Defaults to False (this may change in a future verison).

    Returns
    -------
    output_x, output_y: ndarray (2-Dimensional)
        The interpolated output images. Dimensions are structured in reverse
        order, where (x, y) -> [y, x].

    Raises
    -------
    ValueError
        If `x_pixels` or `y_pixels` are less than or equal to zero, or if the
        specified `x` and `y` minimum and maximums result in an invalid region,
        or if the provided data is not 3-dimensional.
    KeyError
        If `target_x`, `target_y`, `x`, `y`, mass, density, or smoothing
        length columns do not exist in `data`.

    Notes
    -----
    Since the direction of integration is assumed to be straight across the
    z-axis, the z-axis column is not required for this type of interpolation.
    """

    _check_dimension(data, 3)
    x, y, z = _default_xyz(data, x, y, None)
    _verify_columns(data, x, y)

    xlim, ylim = _default_bounds(data, x, y, xlim, ylim)
    x_pixels, y_pixels = _set_pixels(x_pixels, y_pixels, xlim, ylim)
    _check_boundaries(x_pixels, y_pixels, xlim, ylim)

    x_data, y_data, _ = _rotate_xyz(data, x, y, z, rotation, rot_origin)
    if target_z not in data.columns:
        raise KeyError(f"z-directional target column '{target_z}' does not "
                       f"exist in the provided dataset.")
    target_x_data, target_y_data, _ = _rotate_data(data, target_x, target_y,
                                                   target_z, rotation,
                                                   rot_origin)

    wx_data = _get_weight(data, target_x_data, dens_weight)
    wy_data = _get_weight(data, target_y_data, dens_weight)
    h_data = _get_smoothing_lengths(data, hmin, x_pixels, y_pixels, xlim, ylim)

    kernel = kernel if kernel is not None else data.kernel
    backend = backend if backend is not None else data.backend

    weight_function = kernel.get_column_kernel_func(integral_samples)
    gridx, gridy = get_backend(backend) \
        .interpolate_3d_projection_vec(x_data, y_data, wx_data, wy_data,
                                       h_data, weight_function,
                                       kernel.get_radius(), x_pixels, y_pixels,
                                       xlim[0], xlim[1], ylim[0], ylim[1],
                                       exact)
    if normalize:
        wx_norm = _get_weight(data, np.array([1] * len(wx_data)), dens_weight)
        wy_norm = _get_weight(data, np.array([1] * len(wy_data)), dens_weight)
        norm_gridx, norm_gridy = get_backend(backend) \
            .interpolate_3d_projection_vec(x_data, y_data, wx_norm, wy_norm,
                                           h_data, weight_function,
                                           kernel.get_radius(), x_pixels,
                                           y_pixels, xlim[0], xlim[1],
                                           ylim[0], ylim[1], exact)
        gridx = np.nan_to_num(gridx / norm_gridx)
        gridy = np.nan_to_num(gridy / norm_gridy)

    return (gridx, gridy)


def interpolate_3d_cross(data: 'SarracenDataFrame',  # noqa: F821
                         target: str,
                         x: Union[str, None] = None,
                         y: Union[str, None] = None,
                         z: Union[str, None] = None,
                         z_slice: Union[float, None] = None,
                         kernel: Union[BaseKernel, None] = None,
                         corotation: Union[np.ndarray, list, None] = None,
                         rotation: Union[np.ndarray, list,
                                         Rotation, None] = None,
                         rot_origin: Union[np.ndarray, list, str, None] = None,
                         x_pixels: Union[int, None] = None,
                         y_pixels: Union[int, None] = None,
                         xlim: Union[Tuple[float, float], None] = None,
                         ylim: Union[Tuple[float, float], None] = None,
                         backend: Union[str, None] = None,
                         dens_weight: bool = False,
                         normalize: bool = True,
                         hmin: bool = False) -> np.ndarray:
    """
    Interpolate 3D particle data to a 2D grid, using a 3D cross-section.

    Interpolates particle data in a SarracenDataFrame across three directional
    axes to a 2D grid of pixels. A cross-section is taken of the 3D data at a
    specific value of z, and the contributions of particles near the plane are
    interpolated to a 2D grid.

    Parameters
    ----------
    data : SarracenDataFrame
        The particle data to interpolate over.
    target: str
        The column label of the target smoothing data.
    z_slice: float
        The z-axis value to take the cross-section at. Defaults to the midpoint
        of the z-directional data.
    x, y, z: str, optional
        The column labels of the directional data to interpolate over. Defaults
        to the x, y, and z columns
        detected in `data`.
    kernel: BaseKernel
        The kernel to use for smoothing the target data. Defaults to the kernel
        specified in `data`.
    corotation: array_like
        The x, y, z coordinates of two locations which determines the
        corotating frame.
    rotation: array_like or SciPy Rotation, optional
        The rotation to apply to the data before interpolation. If defined as
        an array, the order of rotations is [z, y, x] in degrees.
    rot_origin: array_like or ['com', 'midpoint'], optional
        Point of rotation of the data. Only applies to 3D datasets. If
        array_like, then the [x, y, z] coordinates specify the point around
        which the data is rotated. If 'com', then data is rotated around the
        centre of mass. If 'midpoint', then data is rotated around the
        midpoint, that is, min + max / 2. Defaults to the midpoint.
    x_pixels, y_pixels: int, optional
        Number of pixels in the output image in the x & y directions. Default
        values are chosen to keep a consistent aspect ratio.
    xlim, ylim: tuple of float, optional
        The minimum and maximum values to use in interpolation, in particle
        data space. Defaults to the minimum and maximum values of `x` and `y`.
    backend: ['cpu', 'gpu']
        The computation backend to use when interpolating this data. Defaults
        to 'gpu' if CUDA is enabled, otherwise 'cpu' is used. A manually
        specified backend in `data` will override the default.
    dens_weight: bool, optional
        If True, the target will be multiplied by density. Defaults to False.
    normalize: bool, optional
        If True, will normalize the interpolation. Defaults to True.
    hmin: bool, optional
        If True, a minimum smoothing length of 0.5 * pixel size will be
        imposed. This ensures each particle contributes to at least one grid
        cell / pixel. Defaults to False (this may change in a future verison).

    Returns
    -------
    ndarray (2-Dimensional)
        The interpolated output image, in a 2-dimensional numpy array.
        Dimensions are structured in reverse order, where (x, y) -> [y, x].

    Raises
    -------
    ValueError
        If `pixwidthx`, `pixwidthy`, `pixcountx`, or `pixcounty` are less than
        or equal to zero, or if the specified `x` and `y` minimum and maximums
        result in an invalid region, or if the provided data is not
        3-dimensional.
    KeyError
        If `target`, `x`, `y`, `z`, mass, density, or smoothing length columns
        do not exist in `data`.
    """
    _check_dimension(data, 3)

    # x and y columns default to the variables from the SarracenDataFrame.
    x, y, z = _default_xyz(data, x, y, z)
    _verify_columns(data, x, y)

    # set default slice to be through the data's average z-value.
    if z_slice is None:
        z_slice = data.loc[:, z].mean()

    w_data = _get_weight(data, target, dens_weight)

    # boundaries of the plot default to the max & min values of the data.
    xlim, ylim = _default_bounds(data, x, y, xlim, ylim)
    x_pixels, y_pixels = _set_pixels(x_pixels, y_pixels, xlim, ylim)
    _check_boundaries(x_pixels, y_pixels, xlim, ylim)

    kernel = kernel if kernel is not None else data.kernel
    backend = backend if backend is not None else data.backend

    if corotation is not None:
        rotation, rot_origin = _corotate(corotation, rotation)

    x_data, y_data, z_data = _rotate_xyz(data, x, y, z, rotation, rot_origin)
    h_data = _get_smoothing_lengths(data, hmin, x_pixels, y_pixels, xlim, ylim)

    grid = get_backend(backend) \
        .interpolate_3d_cross(x_data, y_data, z_data, z_slice, w_data, h_data,
                              kernel.w, kernel.get_radius(), x_pixels,
                              y_pixels, xlim[0], xlim[1], ylim[0], ylim[1])

    if normalize:
        w_norm = _get_weight(data, np.array([1] * len(w_data)), dens_weight)
        norm_grid = get_backend(backend) \
            .interpolate_3d_cross(x_data, y_data, z_data, z_slice, w_norm,
                                  h_data, kernel.w, kernel.get_radius(),
                                  x_pixels, y_pixels,
                                  xlim[0], xlim[1], ylim[0], ylim[1])
        grid = np.nan_to_num(grid / norm_grid)

    return grid


def interpolate_3d_cross_vec(data: 'SarracenDataFrame',  # noqa: F821
                             target_x: str,
                             target_y: str,
                             target_z: str,
                             z_slice: Union[float, None] = None,
                             x: Union[str, None] = None,
                             y: Union[str, None] = None,
                             z: Union[str, None] = None,
                             kernel: Union[BaseKernel, None] = None,
                             rotation: Union[np.ndarray, list, Rotation,
                                             None] = None,
                             rot_origin: Union[np.ndarray, list,
                                               str, None] = None,
                             x_pixels: Union[int, None] = None,
                             y_pixels: Union[int, None] = None,
                             xlim: Optional[Tuple[Optional[float],
                                                  Optional[float]]] = None,
                             ylim: Optional[Tuple[Optional[float],
                                                  Optional[float]]] = None,
                             backend: Union[str, None] = None,
                             dens_weight: bool = False,
                             normalize: bool = True,
                             hmin: bool = False) -> Tuple[np.ndarray,
                                                          np.ndarray]:
    """
    Interpolate 3D vector particle data to a 2D grid, using a 3D cross-section.

    Interpolates vector particle data in a SarracenDataFrame across three
    directional axes to a 2D grid of pixels. A cross-section is taken of the
    3D data at a specific value of z, and the contributions of vectors near
    the plane are interpolated to a 2D grid.

    Parameters
    ----------
    data : SarracenDataFrame
        The particle data to interpolate over.
    target_x, target_y, target_z: str
        The column labels of the target vector.
    z_slice: float
        The z-axis value to take the cross-section at. Defaults to the midpoint
        of the z-directional data.
    x, y, z: str, optional
        The column labels of the directional data to interpolate over. Defaults
        to the x, y, and z columns detected in `data`.
    kernel: BaseKernel
        The kernel to use for smoothing the target data. Defaults to the kernel
        specified in `data`.
    rotation: array_like or SciPy Rotation, optional
        The rotation to apply to the data before interpolation. If defined as
        an array, the order of rotations is [z, y, x] in degrees.
    rot_origin: array_like or ['com', 'midpoint'], optional
        Point of rotation of the data. Only applies to 3D datasets. If
        array_like, then the [x, y, z] coordinates specify the point around
        which the data is rotated. If 'com', then data is rotated around the
        centre of mass. If 'midpoint', then data is rotated around the
        midpoint, that is, min + max / 2. Defaults to the midpoint.
    x_pixels, y_pixels: int, optional
        Number of pixels in the output image in the x & y directions. Default
        values are chosen to keep a consistent aspect ratio.
    xlim, ylim: float, optional
        The minimum and maximum values to use in interpolation, in particle
        data space. Defaults to the minimum and maximum values of `x` and `y`.
    backend: ['cpu', 'gpu']
        The computation backend to use when interpolating this data. Defaults
        to 'gpu' if CUDA is enabled, otherwise 'cpu' is used. A manually
        specified backend in `data` will override the default.
    dens_weight: bool, optional
        If True, the target will be multiplied by density. Defaults to False.
    normalize: bool, optional
        If True, will normalize the interpolation. Defaults to True.
    hmin: bool, optional
        If True, a minimum smoothing length of 0.5 * pixel size will be
        imposed. This ensures each particle contributes to at least one grid
        cell / pixel. Defaults to False (this may change in a future verison).

    Returns
    -------
    output_x, output_y: ndarray (2-Dimensional)
        The interpolated output images. Dimensions are structured in reverse
        order, where (x, y) -> [y, x].

    Raises
    -------
    ValueError
        If `pixwidthx`, `pixwidthy`, `pixcountx`, or `pixcounty` are less than
        or equal to zero, or if the specified `x` and `y` minimum and maximums
        result in an invalid region, or if the provided data is not
        3-dimensional.
    KeyError
        If `target_x`, `target_y`, `target_z`, `x`, `y`, `z`, mass, density,
        or smoothing length columns do not exist in `data`.
    """

    _check_dimension(data, 3)
    x, y, z = _default_xyz(data, x, y, z)
    _verify_columns(data, x, y)

    # set default slice to be through the data's average z-value.
    if z_slice is None:
        z_slice = data.loc[:, z].mean()

    # boundaries of the plot default to the max & min values of the data.
    xlim, ylim = _default_bounds(data, x, y, xlim, ylim)
    x_pixels, y_pixels = _set_pixels(x_pixels, y_pixels, xlim, ylim)
    _check_boundaries(x_pixels, y_pixels, xlim, ylim)

    x_data, y_data, z_data = _rotate_xyz(data, x, y, z, rotation, rot_origin)
    target_x_data, target_y_data, _ = _rotate_data(data, target_x, target_y,
                                                   target_z, rotation,
                                                   rot_origin)

    wx_data = _get_weight(data, target_x_data, dens_weight)
    wy_data = _get_weight(data, target_y_data, dens_weight)
    h_data = _get_smoothing_lengths(data, hmin, x_pixels, y_pixels, xlim, ylim)

    kernel = kernel if kernel is not None else data.kernel
    backend = backend if backend is not None else data.backend

    gridx, gridy = get_backend(backend) \
        .interpolate_3d_cross_vec(x_data, y_data, z_data, z_slice, wx_data,
                                  wy_data, h_data, kernel.w,
                                  kernel.get_radius(), x_pixels, y_pixels,
                                  xlim[0], xlim[1], ylim[0], ylim[1])

    if normalize:
        wx_norm = _get_weight(data, np.array([1] * len(wx_data)), dens_weight)
        wy_norm = _get_weight(data, np.array([1] * len(wy_data)), dens_weight)
        norm_gridx, norm_gridy = get_backend(backend) \
            .interpolate_3d_cross_vec(x_data, y_data, z_data, z_slice, wx_norm,
                                      wy_norm, h_data, kernel.w,
                                      kernel.get_radius(), x_pixels, y_pixels,
                                      xlim[0], xlim[1], ylim[0], ylim[1])
        gridx = np.nan_to_num(gridx / norm_gridx)
        gridy = np.nan_to_num(gridy / norm_gridy)

    return (gridx, gridy)


def interpolate_3d_grid(data: 'SarracenDataFrame',  # noqa: F821
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
                        backend: Union[str, None] = None,
                        dens_weight: bool = False,
                        normalize: bool = True,
                        hmin: bool = False) -> np.ndarray:
    """
    Interpolate 3D particle data to a 3D grid of pixels

    Interpolates particle data in a SarracenDataFrame across three directional
    axes to a 3D grid of pixels. The contributions of all particles near each
    3D cell are summed and stored in the 3D grid.

    Parameters
    ----------
    data : SarracenDataFrame
        The particle data to interpolate over.
    target: str
        The column label of the target data.
    x, y, z: str, optional
        The column labels of the directional data to interpolate over. Defaults
        to the x, y, and z columns detected in `data`.
    kernel: BaseKernel
        The kernel to use for smoothing the target data. Defaults to the kernel
        specified in `data`.
    rotation: array_like or SciPy Rotation, optional
        The rotation to apply to the data before interpolation. If defined as
        an array, the order of rotations is [z, y, x] in degrees.
    rot_origin: array_like or ['com', 'midpoint'], optional
        Point of rotation of the data. Only applies to 3D datasets. If
        array_like, then the [x, y, z] coordinates specify the point around
        which the data is rotated. If 'com', then data is rotated around the
        centre of mass. If 'midpoint', then data is rotated around the
        midpoint, that is, min + max / 2. Defaults to the midpoint.
    x_pixels, y_pixels, z_pixels: int, optional
        Number of pixels in the output image in the x, y & z directions.
        Default values are chosen to keep a consistent aspect ratio.
    xlim, ylim, zlim: tuple of float, optional
        The minimum and maximum values to use in interpolation, in particle
        data space. Defaults to the minimum and maximum values of `x`, `y`
        and `z`.
    backend: ['cpu', 'gpu']
        The computation backend to use when interpolating this data. Defaults
        to 'gpu' if CUDA is enabled, otherwise 'cpu' is used. A manually
        specified backend in `data` will override the default.
    dens_weight: bool, optional
        If True, the target will be multiplied by density. Defaults to False.
    normalize: bool, optional
        If True, will normalize the interpolation. Defaults to True.
    hmin: bool, optional
        If True, a minimum smoothing length of 0.5 * pixel size will be
        imposed. This ensures each particle contributes to at least one grid
        cell / pixel. Defaults to False (this may change in a future verison).

    Returns
    -------
    ndarray (3-Dimensional)
        The interpolated output image, in a 3-dimensional numpy array.
        Dimensions are structured in reverse order, where (x, y, z) ->
        [z, y, x].

    Raises
    -------
    ValueError
        If `x_pixels`, `y_pixels` or `z_pixels` are less than or equal to zero,
        or if the specified `x`, `y` and `z` minimum and maximum values result
        in an invalid region, or if `data` is not 3-dimensional.
    KeyError
        If `target`, `x`, `y`, `z`, mass, density, or smoothing length columns
        do not exist in `data`.
    """
    _check_dimension(data, 3)
    x, y, z = _default_xyz(data, x, y, z)
    _verify_columns(data, x, y)

    w_data = _get_weight(data, target, dens_weight)

    if not xlim:
        xlim = (None, None)
    if not ylim:
        ylim = (None, None)
    xlim, ylim = _default_bounds(data, x, y, xlim, ylim)
    zlim = zlim if zlim else (data.loc[:, z].min(), data.loc[:, z].max())

    x_pixels, y_pixels = _set_pixels(x_pixels, y_pixels, xlim, ylim)
    if z_pixels is None:
        dz = zlim[1] - zlim[0]
        dx = xlim[1] - xlim[0]
        z_pixels = int(np.rint(x_pixels * (dz / dx)))

    _check_boundaries(x_pixels, y_pixels, xlim, ylim)
    if zlim[1] - zlim[0] <= 0:
        raise ValueError("`z_max` must be greater than `z_min`!")
    if z_pixels <= 0:
        raise ValueError("`z_pixels` must be greater than zero!")

    kernel = kernel if kernel is not None else data.kernel
    backend = backend if backend is not None else data.backend

    x_data, y_data, z_data = _rotate_xyz(data, x, y, data.zcol,
                                         rotation, rot_origin)
    h_data = _get_smoothing_lengths(data, hmin, x_pixels, y_pixels,
                                    xlim, ylim)

    grid = get_backend(backend)\
        .interpolate_3d_grid(x_data, y_data, z_data, w_data, h_data, kernel.w,
                             kernel.get_radius(), x_pixels, y_pixels, z_pixels,
                             xlim[0], xlim[1], ylim[0], ylim[1],
                             zlim[0], zlim[1])

    if normalize:
        w_norm = _get_weight(data, np.array([1] * len(w_data)), dens_weight)
        norm_grid = get_backend(backend)\
            .interpolate_3d_grid(x_data, y_data, z_data, w_norm, h_data,
                                 kernel.w, kernel.get_radius(), x_pixels,
                                 y_pixels, z_pixels, xlim[0], xlim[1], ylim[0],
                                 ylim[1], zlim[0], zlim[1])
        grid = np.nan_to_num(grid / norm_grid)

    return grid


def get_backend(code: str) -> Type[BaseBackend]:
    """
    Get the interpolation backend associated with a string code.

    Parameters
    ----------
    code: str
        The code associated with the particular backend. At the moment, 'cpu'
        for the CPU backend, and 'gpu' for the GPU backend are supported.

    Returns
    -------
    CPUBackend: The backend to use for interpolation.
    """
    if code == 'cpu':
        return CPUBackend
    if code == 'gpu':
        return GPUBackend
    raise ValueError("Invalid backend")
