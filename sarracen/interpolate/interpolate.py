"""
Contains several interpolation functions which produce interpolated 2D or 1D arrays of SPH data.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from ..interpolate import BaseBackend, CPUBackend, GPUBackend
from ..kernels import BaseKernel

from typing import Tuple

def _snap(value: float):
    """ Snap a number to the nearest integer

    Return a number which is rounded to the nearest integer,
    with a 1e-4 absolute range of tolerance.

    Parameters
    ----------
    value: float
        The number to snap.

    Returns
    -------
    float: An integer (in float form) if a close integer is detected, otherwise return `value`.
    """
    if np.isclose(value, np.rint(value), atol=1e-4):
        return np.rint(value)
    else:
        return value


def _default_xy(data, x, y):
    """ Utility function to determine the x & y columns to use during 2D interpolation.

    Parameters
    ----------
    data: SarracenDataFrame
        The particle dataset to interpolate over.
    x, y: str
        The x and y directional column labels passed to the interpolation function.

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


def _default_xyz(data, x, y, z):
    """ Utility function to determine the x, y and z columns to use during 3-D interpolation.

    Parameters
    ----------
    data: SarracenDataFrame
        The particle dataset to interpolate over.
    x, y, z: str
        The x, y and z directional column labels passed to the interpolation function.

    Returns
    -------
    x, y, z: str
        The directional column labels to use in interpolation.
    """
    if x is None:
        x = data.xcol if not y == data.xcol and not z == data.xcol else \
            data.ycol if not y == data.ycol and not z == data.ycol else data.zcol
    if y is None:
        y = data.ycol if not x == data.ycol and not z == data.ycol else \
            data.xcol if not x == data.xcol and not z == data.xcol else data.zcol
    if z is None:
        z = data.zcol if not x == data.zcol and not y == data.zcol else \
            data.ycol if not x == data.ycol and not y == data.ycol else data.xcol

    return x, y, z


def _snap_boundaries(data, x, y, xlim, ylim) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Utility function to determine the 2-dimensional boundaries to use in 2D interpolation.

    Parameters
    ----------
    data: SarracenDataFrame
        The particle dataset to interpolate over.
    x, y: str
        The directional column labels that will be used in interpolation.
    xlim, ylim: tuple of float
        The minimum and maximum values passed to the interpolation function, in particle data space.

    Returns
    -------
    xlim, ylim: tuple of float
        The minimum and maximum values to use in interpolation, in particle data space. Defaults
        to the maximum and minimum values of `x` and `y`, snapped to the nearest integer.
    """
    # boundaries of the plot default to the maximum & minimum values of the data.
    x_min = xlim[0] if xlim is not None and xlim[0] is not None else None
    y_min = ylim[0] if ylim is not None and ylim[0] is not None else None
    x_max = xlim[1] if xlim is not None and xlim[1] is not None else None
    y_max = ylim[1] if ylim is not None and ylim[1] is not None else None

    x_min = _snap(data.loc[:, x].min()) if x_min is None else x_min
    y_min = _snap(data.loc[:, y].min()) if y_min is None else y_min
    x_max = _snap(data.loc[:, x].max()) if x_max is None else x_max
    y_max = _snap(data.loc[:, y].max()) if y_max is None else y_max

    return (x_min, x_max), (y_min, y_max)


def _set_pixels(x_pixels: int, y_pixels: int, xlim: Tuple[float, float], ylim: Tuple[float, float]) -> Tuple[int, int]:
    """Utility function to determine the number of pixels to interpolate over in 2D interpolation.

    Parameters
    ----------
    x_pixels, y_pixels: int
        The number of pixels in the x & y directions passed to the interpolation function.
    xlim, ylim: tuple of float
        The minimum and maximum values to use in interpolation, in particle data space.

    Returns
    -------
    x_pixels, y_pixels: int
        The number of pixels in the x & y directions to use in 2D interpolation.
    """
    # set # of pixels to maintain an aspect ratio that is the same as the underlying bounds of the data.
    if x_pixels is None and y_pixels is None:
        x_pixels = 512
    if x_pixels is None:
        x_pixels = int(np.rint(y_pixels * ((xlim[1] - xlim[0]) / (ylim[1] - ylim[0]))))
    if y_pixels is None:
        y_pixels = int(np.rint(x_pixels * ((ylim[1] - ylim[0]) / (xlim[1] - xlim[0]))))

    return x_pixels, y_pixels


def _verify_columns(data, target, x, y):
    """ Verify that columns required for 2D interpolation exist in `data`.

    Parameters
    ----------
    data: SarracenDataFrame
        The particle dataset to interpolate over.
    target:
        Column label of the target variable to interpolate over.
    x, y: str
        The directional column labels that will be used in interpolation.

    Raises
    -------
    KeyError
        If `target`, `x`, `y`, mass, density, or smoothing length columns do not
        exist in `data`.
    """
    if x not in data.columns:
        raise KeyError(f"x-directional column '{x}' does not exist in the provided dataset.")
    if y not in data.columns:
        raise KeyError(f"y-directional column '{y}' does not exist in the provided dataset.")
    if target not in data.columns:
        raise KeyError(f"Target column '{target}' does not exist in provided dataset.")
    if data.hcol is None:
        raise KeyError("Smoothing length column does not exist in the provided dataset.")


def _check_boundaries(x_pixels: int, y_pixels: int, xlim: Tuple[float, float], ylim: Tuple[float, float]):
    """ Verify that the pixel count and boundaries of a 2D plot describe a valid region.

    Parameters
    ----------
    x_pixels, y_pixels: int
        The number of pixels in the x & y directions passed to the interpolation function.
    xlim, ylim: tuple of float
        The minimum and maximum values to use in interpolation, in particle data space.

    Raises
    ------
    ValueError
        If `x_pixels` or `y_pixels` are less than or equal to zero, or
        if the specified `x` and `y` minimum and maximum values result in an invalid region.
    """
    if xlim[1] - xlim[0] <= 0:
        raise ValueError("`xlim` max must be greater than min!")
    if ylim[1] - xlim[0] <= 0:
        raise ValueError("`ylim` max must be greater than min!")
    if x_pixels <= 0:
        raise ValueError("`x_pixels` must be greater than zero!")
    if y_pixels <= 0:
        raise ValueError("`y_pixels` must be greater than zero!")


def _check_dimension(data, dim):
    """ Verify that a given dataset describes data with a required number of dimensions.

    Parameters
    ----------
    data: SarracenDataFrame
        The particle dataset to interpolate over.
    dim: [2, 3]
        The number of required dimensions.

    Returns
    -------
    ValueError
        If the dataset is not `dim`-dimensional.
    """
    if data.get_dim() != dim:
        raise TypeError(f"Dataset is not {dim}-dimensional.")


def _rotate_data(data, x, y, z, rotation, origin):
    """ Rotate vector data in a particle dataset.

    Parameters
    ----------
    data: SarracenDataFrame
        The particle dataset to interpolate over.
    x, y, z: str
        Directional column labels containing each dimension of the vector data.
    rotation: array_like or Rotation, optional
        The rotation to apply to the vector data. If defined as an array, the
        order of rotations is [z, y, x] in degrees
    origin: array_like, optional
        Point of rotation of the data, in [x, y, z] form.

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
            rotation = Rotation.from_euler('zyx', rotation, degrees=True)

        vectors = data[[x, y, z]].to_numpy()
        if origin is None:
            origin = (vectors.min(0) + vectors.max(0)) / 2

        vectors = vectors - origin
        vectors = rotation.apply(vectors)
        vectors = vectors + origin

        x_data = vectors[:, 0]
        y_data = vectors[:, 1]
        z_data = vectors[:, 2]

    return x_data, y_data, z_data


def _rotate_xyz(data, x, y, z, rotation, origin):
    """ Rotate positional data in a particle dataset.

        Differs from _rotate_data() in that the returned data values are shuffled to ensure that
        the rotation is always applied to the global x, y, and z columns of the dataset, no matter
        the order of x, y, and z provided to this function.

        Parameters
        ----------
        data: SarracenDataFrame
            The particle dataset to interpolate over.
        x, y, z: str
            Directional column labels containing the positional column labels
        rotation: array_like or Rotation, optional
            The rotation to apply to the data. If defined as an array, the
            order of rotations is [z, y, x] in degrees
        origin: array_like, optional
            Point of rotation of the data, in [x, y, z] form.

        Returns
        -------
        x_data, y_data, z_data: ndarray
            The rotated x, y, and z directional data.
        """
    rotated_x, rotated_y, rotated_z = _rotate_data(data, data.xcol, data.ycol, data.zcol, rotation, origin)
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


def _get_mass(data: 'SarracenDataFrame'):
    if data._mcol == None:
        if 'mass' not in data.params:
            raise KeyError("'mass' column does not exist in this SarracenDataFrame.")
        return data.params['mass']

    return data[data._mcol].to_numpy()


def _get_density(data: 'SarracenDataFrame'):
    if data._rhocol == None:
        if not {data.hcol}.issubset(data.columns) or 'hfact' not in data.params:
            raise KeyError('Density cannot be derived from the columns in this SarracenDataFrame.')

        return ((data.params['hfact'] / data['h']) ** (data.get_dim()) * _get_mass(data)).to_numpy()

    return data[data._rhocol].to_numpy()


def interpolate_2d(data: 'SarracenDataFrame', target: str, x: str = None, y: str = None, kernel: BaseKernel = None,
                   x_pixels: int = None, y_pixels: int = None, xlim: Tuple[float, float] = None,
                   ylim: Tuple[float, float] = None, exact: bool = False, backend: str = None) -> np.ndarray:
    """ Interpolate particle data across two directional axes to a 2D grid of pixels.

    Interpolate the data within a SarracenDataFrame to a 2D grid, by interpolating the values
    of a target variable. The contributions of all particles near the interpolation area are
    summed and stored to a 2D grid.

    Parameters
    ----------
    data : SarracenDataFrame
        Particle data, in a SarracenDataFrame.
    target: str
        Column label of the target smoothing data.
    x, y: str
        Column labels of the directional axes. Defaults to the x & y columns detected in `data`.
    kernel: BaseKernel, optional
        Kernel to use for smoothing the target data. Defaults to the kernel specified in `data`.
    x_pixels, y_pixels: int, optional
        Number of pixels in the output image in the x & y directions. Default values are chosen to keep
        a consistent aspect ratio.
    xlim, ylim: tuple of float, optional
        The minimum and maximum values to use in interpolation, in particle data space. Defaults
        to the minimum and maximum values of `x` and `y`.
    exact: bool
        Whether to use exact interpolation of the data.
    backend: ['cpu', 'gpu']
        The computation backend to use when interpolating this data. Defaults to the backend specified in `data`.

    Returns
    -------
    ndarray (2-Dimensional)
        The interpolated output image, in a 2-dimensional numpy array. Dimensions are
        structured in reverse order, where (x, y) -> [y, x].

    Raises
    -------
    ValueError
        If `x_pixels` or `y_pixels` are less than or equal to zero, or
        if the specified `x` and `y` minimum and maximum values result in an invalid region, or
        if `data` is not 2-dimensional.
    KeyError
        If `target`, `x`, `y`, mass, density, or smoothing length columns do not
        exist in `data`.
    """
    _check_dimension(data, 2)
    x, y = _default_xy(data, x, y)

    mass_data = _get_mass(data)
    rho_data = _get_density(data)

    if target == 'rho':
        target_data = rho_data
    else:
        _verify_columns(data, x, y, target)
        target_data = data[target].to_numpy()

    w_data = target_data * mass_data / rho_data

    xlim, ylim = _snap_boundaries(data, x, y, xlim, ylim)
    x_pixels, y_pixels = _set_pixels(x_pixels, y_pixels, xlim, ylim)
    _check_boundaries(x_pixels, y_pixels, xlim, ylim)

    kernel = kernel if kernel is not None else data.kernel
    backend = backend if backend is not None else data.backend

    return get_backend(backend). \
        interpolate_2d_render(data[x].to_numpy(), data[y].to_numpy(), w_data, data[data._hcol].to_numpy(), kernel.w,
                              kernel.get_radius(), x_pixels, y_pixels, xlim[0], xlim[1], ylim[0], ylim[1], exact)


def interpolate_2d_vec(data: 'SarracenDataFrame', target_x: str, target_y: str, x: str = None, y: str = None,
                       kernel: BaseKernel = None, x_pixels: int = None, y_pixels: int = None,
                       xlim: Tuple[float, float] = None, ylim: Tuple[float, float] = None, exact: bool = False,
                       backend: str = None):
    """ Interpolate vector particle data across two directional axes to a 2D grid of particles.

    Interpolate the data within a SarracenDataFrame to a 2D grid, by interpolating the values
    of a target vector. The contributions of all vectors near the interpolation area are
    summed and stored to a 2D grid.

    Parameters
    ----------
    data : SarracenDataFrame
        Particle data, in a SarracenDataFrame.
    target_x, target_y: str
        Column labels of the target vector.
    x, y: str
        Column labels of the directional axes. Defaults to the x & y columns detected in `data`.
    kernel: BaseKernel, optional
        Kernel to use for smoothing the target data. Defaults to the kernel specified in `data`.
    x_pixels, y_pixels: int, optional
        Number of pixels in the output image in the x & y directions. Default values are chosen to keep
        a consistent aspect ratio.
    xlim, ylim: tuple of float, optional
        The minimum and maximum values to use in interpolation, in particle data space. Defaults
        to the minimum and maximum values of `x` and `y`.
    exact: bool
        Whether to use exact interpolation of the data.
    backend: ['cpu', 'gpu']
        The computation backend to use when interpolating this data. Defaults to the backend specified in `data`.

    Returns
    -------
    output_x, output_y: ndarray (2-Dimensional)
        The interpolated output images, in a 2-dimensional numpy arrays. Dimensions are
        structured in reverse order, where (x, y) -> [y, x].

    Raises
    -------
    ValueError
        If `x_pixels` or `y_pixels` are less than or equal to zero, or
        if the specified `x` and `y` minimum and maximum values result in an invalid region, or
        if `data` is not 2-dimensional.
    KeyError
        If `target_x`, `target_y`, `x`, `y`, mass, density, or smoothing length columns do not
        exist in `data`.
    """
    _check_dimension(data, 2)
    x, y = _default_xy(data, x, y)
    _verify_columns(data, x, y, target_x)
    _verify_columns(data, x, y, target_y)

    xlim, ylim = _snap_boundaries(data, x, y, xlim, ylim)
    x_pixels, y_pixels = _set_pixels(x_pixels, y_pixels, xlim, ylim)
    _check_boundaries(x_pixels, y_pixels, xlim, ylim)

    mass_data = _get_mass(data)
    rho_data = _get_density(data)
    wx_data = data[target_x].to_numpy() * mass_data / rho_data
    wy_data = data[target_y].to_numpy() * mass_data / rho_data

    kernel = kernel if kernel is not None else data.kernel
    backend = backend if backend is not None else data.backend

    return get_backend(backend).\
        interpolate_2d_render_vec(data[x].to_numpy(), data[y].to_numpy(), wx_data, wy_data, data['h'].to_numpy(),
                                  kernel.w, kernel.get_radius(), x_pixels, y_pixels, xlim[0], xlim[1], ylim[0], ylim[1],
                                  exact)


def interpolate_2d_line(data: 'SarracenDataFrame', target: str, x: str = None, y: str = None,
                         kernel: BaseKernel = None, pixels: int = None, xlim: Tuple[float, float] = None,
                         ylim: Tuple[float, float] = None, backend: str = None) -> np.ndarray:
    """ Interpolate particle data across two directional axes to a 1D cross-section line.

    Interpolate the data within a SarracenDataFrame to a 1D line, by interpolating the values
    of a target variable. The contributions of all particles near the specified line are
    summed and stored to a 1-dimensional array.

    Parameters
    ----------
    data : SarracenDataFrame
        Particle data, in a SarracenDataFrame.
    target: str
         Column label of the target smoothing data.
    x, y: str
        Column labels of the directional axes. Defaults to the x & y columns detected in `data`.
    kernel: BaseKernel, optional
        Kernel to use for smoothing the target data. Defaults to the kernel specified in `data`.
    pixels: int, optional
        Number of points in the resulting line plot in the x-direction.
    xlim, ylim: tuple of float, optional
        Starting and ending coordinates of the cross-section line (in particle data space). Defaults to
        the minimum and maximum values of `x` and `y`.
    backend: ['cpu', 'gpu']
        The computation backend to use when interpolating this data. Defaults to the backend specified in `data`.

    Returns
    -------
    np.ndarray (1-Dimensional)
        The resulting interpolated output.

    Raises
    -------
    ValueError
        If `x_pixels` or `y_pixels` are less than or equal to zero, or
        if the specified `xlim` and `ylim` values are all the same (indicating a zero-length cross-section), or
        if `data` is not 2-dimensional.
    KeyError
        If `target`, `x`, `y`, mass, density, or smoothing length columns do not
        exist in `data`.
    """
    _check_dimension(data, 2)
    x, y = _default_xy(data, x, y)

    mass_data = _get_mass(data)
    rho_data = _get_density(data)

    if target == 'rho':
        target_data = rho_data
    else:
        _verify_columns(data, x, y, target)
        target_data = data[target].to_numpy()

    w_data = target_data * mass_data / rho_data

    if isinstance(xlim, float) or isinstance(xlim, int):
        xlim = xlim, xlim
    if isinstance(ylim, float) or isinstance(ylim, int):
        ylim = ylim, ylim

    xlim, ylim = _snap_boundaries(data, x, y, xlim, ylim)
    if xlim[0] == xlim[1] and ylim[0] == ylim[1]:
        raise ValueError('Zero length cross section!')

    kernel = kernel if kernel is not None else data.kernel
    backend = backend if backend is not None else data.backend
    pixels = pixels if pixels is not None else 512

    if pixels <= 0:
        raise ValueError('pixcount must be greater than zero!')

    return get_backend(backend)\
        .interpolate_2d_cross(data[x].to_numpy(), data[y].to_numpy(), w_data, data['h'].to_numpy(), kernel.w,
                              kernel.get_radius(), pixels, xlim[0], xlim[1], ylim[0], ylim[1])


def interpolate_3d_line(data: 'SarracenDataFrame', target: str, x: str = None, y: str = None, z: str = None,
                        kernel: BaseKernel = None, pixels: int = None, xlim: Tuple[float, float] = None,
                        ylim: Tuple[float, float] = None, zlim: Tuple[float, float] = None, backend: str = None):
    """ Interpolate vector particle data across three directional axes to a 1D line.

    Interpolate the data within a SarracenDataFrame to a 1D line, by interpolating the values
    of a target variable. The contributions of all particles near the interpolation line are
    summed and stored to a 1D array.

    Parameters
    ----------
    data : SarracenDataFrame
            Particle data, in a SarracenDataFrame.
    target: str
        Column label of the target variable.
    x, y, z: str
        Column labels of the directional axes. Defaults to the x, y & z columns detected in `data`.
    kernel: BaseKernel, optional
        Kernel to use for smoothing the target data. Defaults to the kernel specified in `data`.
    pixels: int, optional
        Number of pixels in the output image in the x & y directions. Default values are chosen to keep
        a consistent aspect ratio.
    xlim, ylim, zlim: tuple of float, optional
        Starting and ending coordinates of the cross-section line (in particle data space). Defaults to
        the minimum and maximum values of `x`, `y`, and `z`.
    backend: ['cpu', 'gpu']
        The computation backend to use when interpolating this data. Defaults to the backend specified in `data`.

    Returns
    -------
    output: ndarray (1-Dimensional)
        The interpolated output line.

    Raises
    -------
    ValueError
        If `pixels` are less than or equal to zero, or
        if the specified `x`, `y`, and `z` minimum and maximum values result in a zero area cross-section, or
        if `data` is not 3-dimensional.
    KeyError
        If `target`, `x`, `y`, mass, density, or smoothing length data does not exist in `data`.
    """
    _check_dimension(data, 3)
    x, y, z = _default_xyz(data, x, y, z)

    mass_data = _get_mass(data)
    rho_data = _get_density(data)

    if target == 'rho':
        target_data = rho_data
    else:
        _verify_columns(data, x, y, target)
        target_data = data[target].to_numpy()

    w_data = target_data * mass_data / rho_data

    if isinstance(xlim, float) or isinstance(xlim, int):
        xlim = xlim, xlim
    if isinstance(ylim, float) or isinstance(ylim, int):
        ylim = ylim, ylim
    if isinstance(zlim, float) or isinstance(zlim, int):
        zlim = zlim, zlim

    if zlim is None or zlim[0] is None:
        z1 = _snap(data.loc[:, z].min())
    else:
        z1 = zlim[0]
    if zlim is None or zlim[1] is None:
        z2 = _snap(data.loc[:, z].max())
    else:
        z2 = zlim[1]
    zlim = z1, z2

    xlim, ylim = _snap_boundaries(data, x, y, xlim, ylim)
    if ylim[1] == ylim[0] and xlim[1] == xlim[0] and zlim[1] == zlim[0]:
        raise ValueError('Zero length cross section!')

    kernel = kernel if kernel is not None else data.kernel
    backend = backend if backend is not None else data.backend

    if pixels <= 0:
        raise ValueError('pixcount must be greater than zero!')

    return get_backend(backend) \
        .interpolate_3d_line(data[x].to_numpy(), data[y].to_numpy(), data[z].to_numpy(), w_data, data['h'].to_numpy(),
                             kernel.w, kernel.get_radius(), pixels, xlim[0], xlim[1], ylim[0], ylim[1], zlim[0],
                             zlim[1])


def interpolate_3d(data: 'SarracenDataFrame', target: str, x: str = None, y: str = None, kernel: BaseKernel = None,
                   integral_samples: int = 1000, rotation: np.ndarray = None, origin: np.ndarray = None,
                   x_pixels: int = None, y_pixels: int = None, xlim: Tuple[float, float] = None,
                   ylim: Tuple[float, float] = None, exact: bool = False, backend: str = None):
    """ Interpolate 3D particle data to a 2D grid of pixels.

    Interpolates three-dimensional particle data in a SarracenDataFrame. The data
    is interpolated to a 2D grid of pixels, by summing contributions in columns which
    span the z-axis.

    Parameters
    ----------
    data : SarracenDataFrame
        Particle data, in a SarracenDataFrame.
    target: str
        Column label of the target smoothing data.
    x, y: str
        Column labels of the directional axes. Defaults to the x & y columns detected in `data`.
    kernel: BaseKernel, optional
        Kernel to use for smoothing the target data. Defaults to the kernel specified in `data`.
    integral_samples: int, optional
        Number of sample points to take when approximating the 2D column kernel.
    rotation: array_like or Rotation, optional
        The rotation to apply to the data before interpolation. If defined as an array, the
        order of rotations is [z, y, x] in degrees.
    origin: array_like, optional
        Point of rotation of the data, in [x, y, z] form. Defaults to the centre
        point of the bounds of the data.
    x_pixels, y_pixels: int, optional
        Number of pixels in the output image in the x & y directions. Default values are chosen to keep
        a consistent aspect ratio.
    xlim, ylim: tuple of float, optional
        The minimum and maximum values to use in interpolation, in particle data space. Defaults
        to the minimum and maximum values of `x` and `y`.
    exact: bool
        Whether to use exact interpolation of the data.
    backend: ['cpu', 'gpu']
        The computation backend to use when interpolating this data. Defaults to the backend specified in `data`.

    Returns
    -------
    ndarray (2-Dimensional)
        The interpolated output image, in a 2-dimensional numpy array. Dimensions are
        structured in reverse order, where (x, y) -> [y, x].

    Raises
    -------
    ValueError
        If `x_pixels` or `y_pixels` are less than or equal to zero, or
        if the specified `x` and `y` minimum and maximums result in an invalid region, or
        if the provided data is not 3-dimensional.
    KeyError
        If `target`, `x`, `y`, mass, density, or smoothing length columns do not
        exist in `data`.

    Notes
    -----
    Since the direction of integration is assumed to be straight across the z-axis, the z-axis column
    is not required for this type of interpolation.
    """
    _check_dimension(data, 3)
    x, y, z = _default_xyz(data, x, y, None)

    mass_data = _get_mass(data)
    rho_data = _get_density(data)

    if target == 'rho':
        target_data = rho_data
    else:
        _verify_columns(data, x, y, target)
        target_data = data[target].to_numpy()

    w_data = target_data * mass_data / rho_data

    xlim, ylim = _snap_boundaries(data, x, y, xlim, ylim)
    x_pixels, y_pixels = _set_pixels(x_pixels, y_pixels, xlim, ylim)
    _check_boundaries(x_pixels, y_pixels, xlim, ylim)

    x_data, y_data, z_data = _rotate_xyz(data, x, y, z, rotation, origin)
    kernel = kernel if kernel is not None else data.kernel
    backend = backend if backend is not None else data.backend

    weight_function = kernel.get_column_kernel_func(integral_samples)

    return get_backend(backend). \
        interpolate_3d_projection(x_data, y_data, z_data, w_data, data['h'].to_numpy(), weight_function,
                                  kernel.get_radius(), x_pixels, y_pixels, xlim[0], xlim[1], ylim[0], ylim[1], exact)


def interpolate_3d_vec(data: 'SarracenDataFrame', target_x: str, target_y: str, target_z: str, x: str = None,
                       y: str = None, kernel: BaseKernel = None, integral_samples: int = 1000,
                       rotation: np.ndarray = None, origin: np.ndarray = None, x_pixels: int = None,
                       y_pixels: int = None, xlim: Tuple[float, float] = None, ylim: Tuple[float, float] = None,
                       exact: bool = False, backend: str = None):
    """ Interpolate 3D vector particle data to a 2D grid of pixels.

        Interpolates three-dimensional vector particle data in a SarracenDataFrame. The data
        is interpolated to a 2D grid of pixels, by summing contributions in columns which
        span the z-axis.

        Parameters
        ----------
        data : SarracenDataFrame
            Particle data, in a SarracenDataFrame.
        target_x, target_y, target_z: str
            Column labels of the target vector.
        x, y: str
            Column labels of the directional axes. Defaults to the x & y columns detected in `data`.
        kernel: BaseKernel, optional
            Kernel to use for smoothing the target data. Defaults to the kernel specified in `data`.
        integral_samples: int, optional
            Number of sample points to take when approximating the 2D column kernel.
        rotation: array_like or Rotation, optional
            The rotation to apply to the data before interpolation. If defined as an array, the
            order of rotations is [z, y, x] in degrees.
        origin: array_like, optional
            Point of rotation of the data, in [x, y, z] form. Defaults to the centre
            point of the bounds of the data.
        x_pixels, y_pixels: int, optional
            Number of pixels in the output image in the x & y directions. Default values are chosen to keep
            a consistent aspect ratio.
        xlim, ylim: tuple of float, optional
            The minimum and maximum values to use in interpolation, in particle data space. Defaults
            to the minimum and maximum values of `x` and `y`.
        exact: bool
            Whether to use exact interpolation of the data.
        backend: ['cpu', 'gpu']
            The computation backend to use when interpolating this data. Defaults to the backend specified in `data`.

        Returns
        -------
        output_x, output_y: ndarray (2-Dimensional)
            The interpolated output images. Dimensions are structured in reverse order, where (x, y) -> [y, x].

        Raises
        -------
        ValueError
            If `x_pixels` or `y_pixels` are less than or equal to zero, or
            if the specified `x` and `y` minimum and maximums result in an invalid region, or
            if the provided data is not 3-dimensional.
        KeyError
            If `target_x`, `target_y`, `x`, `y`, mass, density, or smoothing length columns do not
            exist in `data`.

        Notes
        -----
        Since the direction of integration is assumed to be straight across the z-axis, the z-axis column
        is not required for this type of interpolation.
        """
    _check_dimension(data, 3)
    x, y, z = _default_xyz(data, x, y, None)
    _verify_columns(data, x, y, target_x)
    _verify_columns(data, x, y, target_y)
    _verify_columns(data, x, y, target_z)

    xlim, ylim = _snap_boundaries(data, x, y, xlim, ylim)
    x_pixels, y_pixels = _set_pixels(x_pixels, y_pixels, xlim, ylim)
    _check_boundaries(x_pixels, y_pixels, xlim, ylim)

    x_data, y_data, _ = _rotate_xyz(data, x, y, z, rotation, origin)
    if target_z not in data.columns:
        raise KeyError(f"z-directional target column '{target_z}' does not exist in the provided dataset.")
    target_x_data, target_y_data, _ = _rotate_data(data, target_x, target_y, target_z, rotation, origin)

    mass_data = _get_mass(data)
    rho_data = _get_density(data)
    wx_data = target_x_data * mass_data / rho_data
    wy_data = target_y_data * mass_data / rho_data

    kernel = kernel if kernel is not None else data.kernel
    backend = backend if backend is not None else data.backend

    weight_function = kernel.get_column_kernel_func(integral_samples)
    return get_backend(backend). \
        interpolate_3d_projection_vec(x_data, y_data, wx_data, wy_data, data['h'].to_numpy(), weight_function,
                                      kernel.get_radius(), x_pixels, y_pixels, xlim[0], xlim[1], ylim[0], ylim[1],
                                      exact)


def interpolate_3d_cross(data: 'SarracenDataFrame', target: str, x: str = None, y: str = None, z: str = None,
                         z_slice: float = None, kernel: BaseKernel = None, rotation: np.ndarray = None,
                         origin: np.ndarray = None, x_pixels: int = None, y_pixels: int = None,
                         xlim: Tuple[float, float] = None, ylim: Tuple[float, float] = None, backend: str = None):
    """ Interpolate 3D particle data to a 2D grid, using a 3D cross-section.

    Interpolates particle data in a SarracenDataFrame across three directional axes to a 2D
    grid of pixels. A cross-section is taken of the 3D data at a specific value of z, and
    the contributions of particles near the plane are interpolated to a 2D grid.

    Parameters
    ----------
    data : SarracenDataFrame
        The particle data to interpolate over.
    target: str
        The column label of the target smoothing data.
    z_slice: float
        The z-axis value to take the cross-section at. Defaults to the midpoint of the z-directional data.
    x, y, z: str
        The column labels of the directional data to interpolate over. Defaults to the x, y, and z columns
        detected in `data`.
    kernel: BaseKernel
        The kernel to use for smoothing the target data. Defaults to the kernel specified in `data`.
    rotation: array_like or Rotation, optional
        The rotation to apply to the data before interpolation. If defined as an array, the
        order of rotations is [z, y, x] in degrees.
    origin: array_like, optional
        Point of rotation of the data, in [x, y, z] form. Defaults to the centre
        point of the bounds of the data.
    x_pixels, y_pixels: int, optional
        Number of pixels in the output image in the x & y directions. Default values are chosen to keep
        a consistent aspect ratio.
    xlim, ylim: tuple of float, optional
        The minimum and maximum values to use in interpolation, in particle data space. Defaults
        to the minimum and maximum values of `x` and `y`.
    backend: ['cpu', 'gpu']
        The computation backend to use when interpolating this data. Defaults to the backend specified in `data`.

    Returns
    -------
    ndarray (2-Dimensional)
        The interpolated output image, in a 2-dimensional numpy array. Dimensions are
        structured in reverse order, where (x, y) -> [y, x].

    Raises
    -------
    ValueError
        If `pixwidthx`, `pixwidthy`, `pixcountx`, or `pixcounty` are less than or equal to zero, or
        if the specified `x` and `y` minimum and maximums result in an invalid region, or
        if the provided data is not 3-dimensional.
    KeyError
        If `target`, `x`, `y`, `z`, mass, density, or smoothing length columns do not
        exist in `data`.
    """
    _check_dimension(data, 3)

    # x & y columns default to the variables determined by the SarracenDataFrame.
    x, y, z = _default_xyz(data, x, y, z)

    # set default slice to be through the data's average z-value.
    if z_slice is None:
        z_slice = _snap(data.loc[:, z].mean())

    mass_data = _get_mass(data)
    rho_data = _get_density(data)

    if target == 'rho':
        target_data = rho_data
    else:
        _verify_columns(data, x, y, target)
        target_data = data[target].to_numpy()

    w_data = target_data * mass_data / rho_data

    # boundaries of the plot default to the maximum & minimum values of the data.
    xlim, ylim = _snap_boundaries(data, x, y, xlim, ylim)
    x_pixels, y_pixels = _set_pixels(x_pixels, y_pixels, xlim, ylim)
    _check_boundaries(x_pixels, y_pixels, xlim, ylim)

    kernel = kernel if kernel is not None else data.kernel
    backend = backend if backend is not None else data.backend

    x_data, y_data, z_data = _rotate_xyz(data, x, y, z, rotation, origin)

    return get_backend(backend) \
        .interpolate_3d_cross(x_data, y_data, z_data, z_slice, w_data, data['h'].to_numpy(), kernel.w,
                              kernel.get_radius(), x_pixels, y_pixels, xlim[0], xlim[1], ylim[0], ylim[1])


def interpolate_3d_cross_vec(data: 'SarracenDataFrame', target_x: str, target_y: str, target_z: str,
                             z_slice: float = None, x: str = None, y: str = None, z: str = None,
                             kernel: BaseKernel = None, rotation: np.ndarray = None, origin: np.ndarray = None,
                             x_pixels: int = None, y_pixels: int = None, xlim: Tuple[float, float] = None,
                             ylim: Tuple[float, float] = None, backend: str = None):
    """ Interpolate 3D vector particle data to a 2D grid, using a 3D cross-section.

        Interpolates vector particle data in a SarracenDataFrame across three directional axes to a 2D
        grid of pixels. A cross-section is taken of the 3D data at a specific value of z, and
        the contributions of vectors near the plane are interpolated to a 2D grid.

        Parameters
        ----------
        data : SarracenDataFrame
            The particle data to interpolate over.
        target_x, target_y, target_z: str
            The column labels of the target vector.
        z_slice: float
            The z-axis value to take the cross-section at. Defaults to the midpoint of the z-directional data.
        x, y, z: str
            The column labels of the directional data to interpolate over. Defaults to the x, y, and z columns
            detected in `data`.
        kernel: BaseKernel
            The kernel to use for smoothing the target data. Defaults to the kernel specified in `data`.
        rotation: array_like or Rotation, optional
            The rotation to apply to the data before interpolation. If defined as an array, the
            order of rotations is [z, y, x] in degrees.
        origin: array_like, optional
            Point of rotation of the data, in [x, y, z] form. Defaults to the centre
            point of the bounds of the data.
        x_pixels, y_pixels: int, optional
            Number of pixels in the output image in the x & y directions. Default values are chosen to keep
            a consistent aspect ratio.
        xlim, ylim: float, optional
            The minimum and maximum values to use in interpolation, in particle data space. Defaults
            to the minimum and maximum values of `x` and `y`.
        backend: ['cpu', 'gpu']
            The computation backend to use when interpolating this data. Defaults to the backend specified in `data`.

        Returns
        -------
        output_x, output_y: ndarray (2-Dimensional)
            The interpolated output images. Dimensions are structured in reverse order, where (x, y) -> [y, x].

        Raises
        -------
        ValueError
            If `pixwidthx`, `pixwidthy`, `pixcountx`, or `pixcounty` are less than or equal to zero, or
            if the specified `x` and `y` minimum and maximums result in an invalid region, or
            if the provided data is not 3-dimensional.
        KeyError
            If `target_x`, `target_y`, `target_z`, `x`, `y`, `z`, mass, density, or smoothing length columns do not
            exist in `data`.
        """
    _check_dimension(data, 3)
    x, y = _default_xy(data, x, y)
    _verify_columns(data, x, y, target_x)
    _verify_columns(data, x, y, target_y)
    _verify_columns(data, x, y, target_z)

    x, y, z = _default_xyz(data, x, y, z)

    # set default slice to be through the data's average z-value.
    if z_slice is None:
        z_slice = _snap(data.loc[:, z].mean())

    # boundaries of the plot default to the maximum & minimum values of the data.
    xlim, ylim = _snap_boundaries(data, x, y, xlim, ylim)
    x_pixels, y_pixels = _set_pixels(x_pixels, y_pixels, xlim, ylim)
    _check_boundaries(x_pixels, y_pixels, xlim, ylim)

    x_data, y_data, z_data = _rotate_xyz(data, x, y, z, rotation, origin)
    target_x_data, target_y_data, _ = _rotate_data(data, target_x, target_y, target_z, rotation, origin)

    mass_data = _get_mass(data)
    rho_data = _get_density(data)
    wx_data = target_x_data * mass_data / rho_data
    wy_data = target_y_data * mass_data / rho_data

    kernel = kernel if kernel is not None else data.kernel
    backend = backend if backend is not None else data.backend

    return get_backend(backend) \
        .interpolate_3d_cross_vec(x_data, y_data, z_data, z_slice, wx_data, wy_data, data['h'].to_numpy(), kernel.w,
                                  kernel.get_radius(), x_pixels, y_pixels, xlim[0], xlim[1], ylim[0], ylim[1])


def interpolate_3d_grid(data: 'SarracenDataFrame', target: str, x: str = None, y: str = None, z: str = None,
                        kernel: BaseKernel = None, rotation: np.ndarray = None, rot_origin: np.ndarray = None,
                        x_pixels: int = None, y_pixels: int = None, z_pixels: int = None,
                        xlim: Tuple[float, float] = None, ylim: Tuple[float, float] = None,
                        zlim: Tuple[float, float] = None, backend: str = None):
    """ Interpolate 3D particle data to a 3D grid of pixels

    Interpolates particle data in a SarracenDataFrame across three directional axes to a 3D
    grid of pixels. The contributions of all particles near each 3D cell are summed and
    stored in the 3D grid.

    Parameters
    ----------
    data : SarracenDataFrame
        The particle data to interpolate over.
    target: str
        The column label of the target data.
    x, y, z: str
        The column labels of the directional data to interpolate over. Defaults to the x, y, and z columns
        detected in `data`.
    kernel: BaseKernel
        The kernel to use for smoothing the target data. Defaults to the kernel specified in `data`.
    rotation: array_like or Rotation, optional
        The rotation to apply to the data before interpolation. If defined as an array, the
        order of rotations is [z, y, x] in degrees.
    rot_origin: array_like, optional
        Point of rotation of the data, in [x, y, z] form. Defaults to the centre
        point of the bounds of the data.
    x_pixels, y_pixels, z_pixels: int, optional
        Number of pixels in the output image in the x, y & z directions. Default values are chosen to keep
        a consistent aspect ratio.
    xlim, ylim, zlim: tuple of float, optional
        The minimum and maximum values to use in interpolation, in particle data space. Defaults
        to the minimum and maximum values of `x`, `y` and `z`.
    backend: ['cpu', 'gpu']
        The computation backend to use when interpolating this data. Defaults to the backend specified in `data`.

    Returns
    -------
    ndarray (3-Dimensional)
        The interpolated output image, in a 3-dimensional numpy array. Dimensions are structured in reverse order,
        where (x, y, z) -> [z, y, x].

    Raises
    -------
    ValueError
        If `x_pixels`, `y_pixels` or `z_pixels` are less than or equal to zero, or
        if the specified `x`, `y` and `z` minimum and maximum values result in an invalid region, or
        if `data` is not 3-dimensional.
    KeyError
        If `target`, `x`, `y`, `z`, mass, density, or smoothing length columns do not
        exist in `data`.
    """
    _check_dimension(data, 3)
    x, y, z = _default_xyz(data, x, y, z)

    mass_data = _get_mass(data)
    rho_data = _get_density(data)

    if target == 'rho':
        target_data = rho_data
    else:
        _verify_columns(data, target, x, y)
        target_data = data[target].to_numpy()

    w_data = target_data * mass_data / rho_data

    if not xlim:
        xlim = (None, None)
    if not ylim:
        ylim = (None, None)
    xlim, ylim = _snap_boundaries(data, x, y, xlim, ylim)
    zlim = zlim if zlim else (_snap(data.loc[:, z].min()), _snap(data.loc[:, z].max()))

    x_pixels, y_pixels = _set_pixels(x_pixels, y_pixels, xlim, ylim)
    z_pixels = int(np.rint(x_pixels * ((zlim[1] - zlim[0]) / (xlim[1] - xlim[0])))) if z_pixels is None else z_pixels
    _check_boundaries(x_pixels, y_pixels, xlim, ylim)
    if zlim[1] - zlim[0] <= 0:
        raise ValueError("`z_max` must be greater than `z_min`!")
    if z_pixels <= 0:
        raise ValueError("`z_pixels` must be greater than zero!")

    kernel = kernel if kernel is not None else data.kernel
    backend = backend if backend is not None else data.backend

    x_data, y_data, z_data = _rotate_xyz(data, x, y, data.zcol, rotation, rot_origin)

    return get_backend(backend) \
        .interpolate_3d_grid(x_data, y_data, z_data, w_data, data['h'].to_numpy(), kernel.w, kernel.get_radius(),
                             x_pixels, y_pixels, z_pixels, xlim[0], xlim[1], ylim[0], ylim[1], zlim[0], zlim[1])


def get_backend(code: str) -> BaseBackend:
    """ Get the interpolation backend assocated with a string code.

    Parameters
    ----------
    code: str
        The code assocated with the particular backend. At the moment, 'cpu' for the CPU backend, and 'gpu' for
        the GPU backend are supported.

    Returns
    -------
    CPUBackend: The backend to use for interpolation.
    """
    if code == 'cpu':
        return CPUBackend
    if code == 'gpu':
        return GPUBackend
    raise ValueError("Invalid code!")
