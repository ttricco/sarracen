"""
Contains several interpolation functions which produce interpolated 2D or 1D arrays of SPH data.
"""
import math

import numpy as np
from numba import prange, njit, cuda
from scipy.spatial.transform import Rotation as R

from sarracen._atomic_operations import atomic_add
from sarracen.kernels import BaseKernel


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
    """Utility function to determine the x & y columns to use during interpolation.

    Parameters
    ----------
    data: SarracenDataFrame
        The particle dataset to interpolate over.
    x, y: str
        The x and y directional column labels passed to the interpolation function.

    Returns
    -------
    x, y: str
        The directional column labels to use in interpolation. Defaults to the x-column detected in `data`
    """
    if x is None:
        x = data.xcol
    if y is None:
        y = data.ycol

    return x, y


def _snap_boundaries(data, x, y, x_min, x_max, y_min, y_max):
    """Utility function to determine the 2-dimensional boundaries to use in 2D interpolation.

    Parameters
    ----------
    data: SarracenDataFrame
        The particle dataset to interpolate over.
    x, y: str
        The directional column labels that will be used in interpolation.
    x_min, x_max, y_min, y_max: float
        The minimum and maximum values passed to the interpolation function, in particle data space.

    Returns
    -------
    x_min, x_max, y_min, y_max: float
        The minimum and maximum values to use in interpolation, in particle data space. Defaults
        to the maximum and minimum values of `x` and `y`, snapped to the nearest integer.
    """
    # boundaries of the plot default to the maximum & minimum values of the data.
    if x_min is None:
        x_min = _snap(data.loc[:, x].min())
    if y_min is None:
        y_min = _snap(data.loc[:, y].min())
    if x_max is None:
        x_max = _snap(data.loc[:, x].max())
    if y_max is None:
        y_max = _snap(data.loc[:, y].max())

    return x_min, x_max, y_min, y_max


def _set_pixels(x_pixels, y_pixels, x_min, x_max, y_min, y_max):
    """Utility function to determine the number of pixels to interpolate over in 2D interpolation.

    Parameters
    ----------
    x_pixels, y_pixels: int
        The number of pixels in the x & y directions passed to the interpolation function.
    x_min, x_max, y_min, y_max: float
        The minimum and maximum values to use in interpolation, in particle data space.

    Returns
    -------
    x_pixels, y_pixels
        The number of pixels in the x & y directions to use in 2D interpolation.
    """
    # set # of pixels to maintain an aspect ratio that is the same as the underlying bounds of the data.
    if x_pixels is None and y_pixels is None:
        x_pixels = 512
    if x_pixels is None:
        x_pixels = int(np.rint(y_pixels * ((x_max - x_min) / (y_max - y_min))))
    if y_pixels is None:
        y_pixels = int(np.rint(x_pixels * ((y_max - y_min) / (x_max - x_min))))

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
        raise KeyError(f"x-directional column '{y}' does not exist in the provided dataset.")
    if target not in data.columns:
        raise KeyError(f"Target column '{target}' does not exist in provided dataset.")
    if data.mcol is None:
        raise KeyError("Mass column does not exist in the provided dataset, please create it with "
                       "sdf.create_mass_column().")
    if data.rhocol is None:
        raise KeyError("Density column does not exist in the provided dataset, please create it with"
                       "sdf.derive_density().")
    if data.hcol is None:
        raise KeyError("Smoothing length column does not exist in the provided dataset.")


def _check_boundaries(x_pixels, y_pixels, x_min, x_max, y_min, y_max):
    """ Verify that the pixel count and boundaries of a 2D plot describe a valid region.

    Parameters
    ----------
    x_pixels, y_pixels: int
        The number of pixels in the x & y directions passed to the interpolation function.
    x_min, x_max, y_min, y_max: float
        The minimum and maximum values to use in interpolation, in particle data space.

    Raises
    ------
    ValueError
        If `x_pixels` or `y_pixels` are less than or equal to zero, or
        if the specified `x` and `y` minimum and maximum values result in an invalid region.
    """
    if x_max - x_min <= 0:
        raise ValueError("`xmax` must be greater than `xmin`!")
    if y_max - y_min <= 0:
        raise ValueError("`ymax` must be greater than `ymin`!")
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
        raise ValueError(f"Dataset is not {dim}-dimensional.")


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
        if not isinstance(rotation, R):
            rotation = R.from_euler('zyx', rotation, degrees=True)

        vectors = data[[data.xcol, data.ycol, data.zcol]].to_numpy()
        if origin is None:
            origin = (vectors[:, 0].min() + vectors[:, 0].max()) / 2

        vectors = vectors - origin
        vectors = rotation.apply(vectors)
        vectors = vectors + origin

        x_data = vectors[:, 0] if x == data.xcol else \
            vectors[:, 1] if x == data.ycol else \
            vectors[:, 2] if x == data.zcol else x_data
        y_data = vectors[:, 0] if y == data.xcol else \
            vectors[:, 1] if y == data.ycol else \
            vectors[:, 2] if y == data.zcol else y_data
        z_data = vectors[:, 0] if z == data.xcol else \
            vectors[:, 1] if z == data.ycol else \
            vectors[:, 2] if z == data.zcol else z_data

    return x_data, y_data, z_data


def interpolate_2d(data: 'SarracenDataFrame', target: str, x: str = None, y: str = None, kernel: BaseKernel = None,
                   x_pixels: int = None, y_pixels: int = None,  x_min: float = None, x_max: float = None,
                   y_min: float = None, y_max: float = None, backend: str = None) -> np.ndarray:
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
    x_min, x_max, y_min, y_max: float, optional
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
        If `x_pixels` or `y_pixels` are less than or equal to zero, or
        if the specified `x` and `y` minimum and maximum values result in an invalid region, or
        if `data` is not 2-dimensional.
    KeyError
        If `target`, `x`, `y`, mass, density, or smoothing length columns do not
        exist in `data`.
    """
    x, y = _default_xy(data, x, y)
    _verify_columns(data, x, y, target)

    x_min, x_max, y_min, y_max = _snap_boundaries(data, x, y, x_min, x_max, y_min, y_max)
    x_pixels, y_pixels = _set_pixels(x_pixels, y_pixels, x_min, x_max, y_min, y_max)
    _check_boundaries(x_pixels, y_pixels, x_min, x_max, y_min, y_max)

    kernel = kernel if kernel is not None else data.kernel
    backend = backend if backend is not None else data.backend
    _check_dimension(data, 2)

    return _fast_2d(data[target].to_numpy(), 0, data[x].to_numpy(), data[y].to_numpy(), np.zeros(len(data)),
                    data['m'].to_numpy(), data['rho'].to_numpy(), data['h'].to_numpy(), kernel.w,
                    kernel.get_radius(), x_pixels, y_pixels, x_min, x_max, y_min, y_max, 2, backend)


def interpolate_2d_cross(data: 'SarracenDataFrame', target: str, x: str = None, y: str = None,
                         kernel: BaseKernel = None, pixels: int = None, x1: float = None, x2: float = None,
                         y1: float = None, y2: float = None, backend: str = None) -> np.ndarray:
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
    x1, x2, y1, y2: float, optional
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
        if the specified `x1`, `x2`, `y1`, and `y2` are all the same (indicating a zero-length cross-section), or
        if `data` is not 2-dimensional.
    KeyError
        If `target`, `x`, `y`, mass, density, or smoothing length columns do not
        exist in `data`.
    """
    x, y = _default_xy(data, x, y)
    _verify_columns(data, x, y, target)

    x1, x2, y1, y2 = _snap_boundaries(data, x, y, x1, x2, y1, y2)
    if y2 == y1 and x2 == x1:
        raise ValueError('Zero length cross section!')

    kernel = kernel if kernel is not None else data.kernel
    backend = backend if backend is not None else data.backend
    _check_dimension(data, 2)

    if pixels <= 0:
        raise ValueError('pixcount must be greater than zero!')

    return _fast_2d_cross(data[target].to_numpy(), data[x].to_numpy(), data[y].to_numpy(), data['m'].to_numpy(),
                          data['rho'].to_numpy(), data['h'].to_numpy(), kernel.w, kernel.get_radius(), pixels, x1,
                          x2, y1, y2, backend)


def interpolate_3d(data: 'SarracenDataFrame', target: str, x: str = None, y: str = None, kernel: BaseKernel = None,
                   integral_samples: int = 1000, rotation: np.ndarray = None, origin: np.ndarray = None,
                   x_pixels: int = None, y_pixels: int = None, x_min: float = None, x_max: float = None,
                   y_min: float = None, y_max: float = None, backend: str = None):
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
    x_min, x_max, y_min, y_max: float, optional
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
    x, y = _default_xy(data, x, y)
    _verify_columns(data, x, y, target)

    x_min, x_max, y_min, y_max = _snap_boundaries(data, x, y, x_min, x_max, y_min, y_max)
    x_pixels, y_pixels = _set_pixels(x_pixels, y_pixels, x_min, x_max, y_min, y_max)
    _check_boundaries(x_pixels, y_pixels, x_min, x_max, y_min, y_max)

    x_data, y_data, _ = _rotate_data(data, x, y, data.zcol, rotation, origin)
    kernel = kernel if kernel is not None else data.kernel
    backend = backend if backend is not None else data.backend
    _check_dimension(data, 3)

    weight_function = kernel.get_column_kernel_func(integral_samples)

    return _fast_2d(data[target].to_numpy(), 0, x_data, y_data, np.zeros(len(data)), data['m'].to_numpy(),
                    data['rho'].to_numpy(), data['h'].to_numpy(), weight_function,
                    kernel.get_radius(), x_pixels, y_pixels, x_min, x_max, y_min, y_max, 2, backend)


def interpolate_3d_cross(data: 'SarracenDataFrame', target: str, z_slice: float = None, x: str = None, y: str = None,
                         z: str = None, kernel: BaseKernel = None, rotation: np.ndarray = None,
                         origin: np.ndarray = None, x_pixels: int = None, y_pixels: int = None, x_min: float = None,
                         x_max: float = None, y_min: float = None, y_max: float = None, backend: str = None):
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
    x_min, x_max, y_min, y_max: float, optional
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
    # x & y columns default to the variables determined by the SarracenDataFrame.
    x, y = _default_xy(data, x, y)
    _verify_columns(data, target, x, y)

    if z is None:
        z = data.zcol
    if z not in data.columns:
        raise KeyError(f"z-directional column '{z}' does not exist in the provided dataset.")

    # set default slice to be through the data's average z-value.
    if z_slice is None:
        z_slice = _snap(data.loc[:, z].mean())

    # boundaries of the plot default to the maximum & minimum values of the data.
    x_min, x_max, y_min, y_max = _snap_boundaries(data, x, y, x_min, x_max, y_min, y_max)
    x_pixels, y_pixels = _set_pixels(x_pixels, y_pixels, x_min, x_max, y_min, y_max)
    _check_boundaries(x_pixels, y_pixels, x_min, x_max, y_min, y_max)

    kernel = kernel if kernel is not None else data.kernel
    backend = backend if backend is not None else data.backend

    _check_dimension(data, 3)

    x_data, y_data, z_data = _rotate_data(data, x, y, z, rotation, origin)

    return _fast_2d(data[target].to_numpy(), z_slice, x_data, y_data, z_data, data['m'].to_numpy(),
                    data['rho'].to_numpy(), data['h'].to_numpy(), kernel.w, kernel.get_radius(), x_pixels, y_pixels,
                    x_min, x_max, y_min, y_max, 3, backend)


def _fast_2d(target, z_slice, x_data, y_data, z_data, mass_data, rho_data, h_data, weight_function, kernel_radius,
             x_pixels, y_pixels, x_min, x_max, y_min, y_max, n_dims, backend):
    if backend == 'cpu':
        return _fast_2d_cpu(target, z_slice, x_data, y_data, z_data, mass_data, rho_data, h_data, weight_function,
                            kernel_radius, x_pixels, y_pixels, x_min, x_max, y_min, y_max, n_dims)
    elif backend == 'gpu':
        return _fast_2d_gpu(target, z_slice, x_data, y_data, z_data, mass_data, rho_data, h_data, weight_function,
                            kernel_radius, x_pixels, y_pixels, x_min, x_max, y_min, y_max, n_dims)


def _fast_2d_cross(target, x_data, y_data, mass_data, rho_data, h_data, weight_function, kernel_radius, pixels, x1, x2,
                   y1, y2, backend):
    if backend == 'cpu':
        return _fast_2d_cross_cpu(target, x_data, y_data, mass_data, rho_data, h_data, weight_function,
                                  kernel_radius, pixels, x1, x2, y1, y2)
    elif backend == 'gpu':
        return _fast_2d_cross_gpu(target, x_data, y_data, mass_data, rho_data, h_data, weight_function,
                                  kernel_radius, pixels, x1, x2, y1, y2)


# Underlying CPU numba-compiled code for interpolation to a 2D grid. Used in interpolation of 2D data,
# and column integration / cross-sections of 3D data.
@njit(parallel=True, fastmath=True)
def _fast_2d_cpu(target, z_slice, x_data, y_data, z_data, mass_data, rho_data, h_data, weight_function, kernel_radius,
                 x_pixels, y_pixels, x_min, x_max, y_min, y_max, n_dims):
    image = np.zeros((y_pixels, x_pixels))
    pixwidthx = (x_max - x_min) / x_pixels
    pixwidthy = (y_max - y_min) / y_pixels
    if not n_dims == 2:
        dz = np.float64(z_slice) - z_data
    else:
        dz = np.zeros(target.size)

    term = (target * mass_data / (rho_data * h_data ** n_dims))

    # iterate through the indexes of non-filtered particles
    for i in prange(term.size):
        if np.abs(dz[i]) >= kernel_radius * h_data[i]:
            continue

        # determine maximum and minimum pixels that this particle contributes to
        ipixmin = int(np.rint((x_data[i] - kernel_radius * h_data[i] - x_min) / pixwidthx))
        jpixmin = int(np.rint((y_data[i] - kernel_radius * h_data[i] - y_min) / pixwidthy))
        ipixmax = int(np.rint((x_data[i] + kernel_radius * h_data[i] - x_min) / pixwidthx))
        jpixmax = int(np.rint((y_data[i] + kernel_radius * h_data[i] - y_min) / pixwidthy))

        if ipixmax < 0 or ipixmin > x_pixels or jpixmax < 0 or jpixmin > y_pixels:
            continue
        if ipixmin < 0:
            ipixmin = 0
        if ipixmax > x_pixels:
            ipixmax = x_pixels
        if jpixmin < 0:
            jpixmin = 0
        if jpixmax > y_pixels:
            jpixmax = y_pixels

        # precalculate differences in the x-direction (optimization)
        dx2i = ((x_min + (np.arange(ipixmin, ipixmax) + 0.5) * pixwidthx - x_data[i]) ** 2) \
               * (1 / (h_data[i] ** 2)) + ((dz[i] ** 2) * (1 / h_data[i] ** 2))

        # determine differences in the y-direction
        ypix = y_min + (np.arange(jpixmin, jpixmax) + 0.5) * pixwidthy
        dy = ypix - y_data[i]
        dy2 = dy * dy * (1 / (h_data[i] ** 2))

        # calculate contributions at pixels i, j due to particle at x, y
        q2 = dx2i + dy2.reshape(len(dy2), 1)

        for jpix in prange(jpixmax - jpixmin):
            for ipix in prange(ipixmax - ipixmin):
                if np.sqrt(q2[jpix][ipix]) > kernel_radius:
                    continue
                wab = weight_function(np.sqrt(q2[jpix][ipix]), n_dims)
                atomic_add(image, (jpix + jpixmin, ipix + ipixmin), term[i] * wab)

    return image


# For the GPU, the numba code is compiled using a factory function approach. This is required
# since a CUDA numba kernel cannot easily take weight_function as an argument.
def _fast_2d_gpu(target, z_slice, x_data, y_data, z_data, mass_data, rho_data, h_data, weight_function, kernel_radius,
                 x_pixels, y_pixels, x_min, x_max, y_min, y_max, n_dims):
    # Underlying GPU numba-compiled code for interpolation to a 2D grid. Used in interpolation of 2D data,
    # and column integration / cross-sections of 3D data.
    @cuda.jit(fastmath=True)
    def _2d_func(target, z_slice, x_data, y_data, z_data, mass_data, rho_data, h_data, kernel_radius,
                     x_pixels, y_pixels, x_min, x_max, y_min, y_max, n_dims, image):
        pixwidthx = (x_max - x_min) / x_pixels
        pixwidthy = (y_max - y_min) / y_pixels

        i = cuda.grid(1)
        if i < len(target):
            if not n_dims == 2:
                dz = np.float64(z_slice) - z_data[i]
            else:
                dz = 0

            term = (target[i] * mass_data[i] / (rho_data[i] * h_data[i] ** n_dims))

            if abs(dz) >= kernel_radius * h_data[i]:
                return

            # determine maximum and minimum pixels that this particle contributes to
            ipixmin = round((x_data[i] - kernel_radius * h_data[i] - x_min) / pixwidthx)
            jpixmin = round((y_data[i] - kernel_radius * h_data[i] - y_min) / pixwidthy)
            ipixmax = round((x_data[i] + kernel_radius * h_data[i] - x_min) / pixwidthx)
            jpixmax = round((y_data[i] + kernel_radius * h_data[i] - y_min) / pixwidthy)

            if ipixmax < 0 or ipixmin > x_pixels or jpixmax < 0 or jpixmin > y_pixels:
                return
            if ipixmin < 0:
                ipixmin = 0
            if ipixmax > x_pixels:
                ipixmax = x_pixels
            if jpixmin < 0:
                jpixmin = 0
            if jpixmax > y_pixels:
                jpixmax = y_pixels

            # calculate contributions to all nearby pixels
            for jpix in range(jpixmax - jpixmin):
                for ipix in range(ipixmax - ipixmin):
                    # determine difference in the x-direction
                    xpix = x_min + ((ipix + ipixmin) + 0.5) * pixwidthx
                    dx = xpix - x_data[i]
                    dx2 = dx * dx * (1 / (h_data[i] ** 2))

                    # determine difference in the y-direction
                    ypix = y_min + ((jpix + jpixmin) + 0.5) * pixwidthy
                    dy = ypix - y_data[i]
                    dy2 = dy * dy * (1 / (h_data[i] ** 2))

                    dz2 = ((dz ** 2) * (1 / h_data[i] ** 2))

                    # calculate contributions at pixels i, j due to particle at x, y
                    q = math.sqrt(dx2 + dy2 + dz2)

                    # add contribution to image
                    if q < kernel_radius:
                        # atomic add protects the summation against race conditions.
                        wab = weight_function(q, n_dims)
                        cuda.atomic.add(image, (jpix + jpixmin, ipix + ipixmin), term * wab)

    threadsperblock = 32
    blockspergrid = (target.size + (threadsperblock - 1)) // threadsperblock

    # transfer relevant data to the GPU
    d_target = cuda.to_device(target)
    d_x = cuda.to_device(x_data)
    d_y = cuda.to_device(y_data)
    d_z = cuda.to_device(z_data)
    d_m = cuda.to_device(mass_data)
    d_rho = cuda.to_device(rho_data)
    d_h = cuda.to_device(h_data)
    # CUDA kernels have no return values, so the image data must be
    # allocated on the device beforehand.
    d_image = cuda.to_device(np.zeros((y_pixels, x_pixels)))

    # execute the newly compiled CUDA kernel.
    _2d_func[blockspergrid, threadsperblock](d_target, z_slice, d_x, d_y, d_z, d_m, d_rho, d_h, kernel_radius, x_pixels,
                                             y_pixels, x_min, x_max, y_min, y_max, n_dims, d_image)

    return d_image.copy_to_host()


# Underlying CPU numba-compiled code for 2D->1D cross-sections.
@njit(parallel=True, fastmath=True)
def _fast_2d_cross_cpu(target, x_data, y_data, mass_data, rho_data, h_data, weight_function, kernel_radius, pixels, x1,
                       x2, y1, y2):
    # determine the slope of the cross-section line
    gradient = 0
    if not x2 - x1 == 0:
        gradient = (y2 - y1) / (x2 - x1)
    yint = y2 - gradient * x2

    # determine the fraction of the line that one pixel represents
    xlength = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    pixwidth = xlength / pixels
    xpixwidth = (x2 - x1) / pixels

    term = target * mass_data / (rho_data * h_data ** 2)

    # the intersections between the line and a particle's 'smoothing circle' are
    # found by solving a quadratic equation with the below values of a, b, and c.
    # if the determinant is negative, the particle does not contribute to the
    # cross-section, and can be removed.
    aa = 1 + gradient ** 2
    bb = 2 * gradient * (yint - y_data) - 2 * x_data
    cc = x_data ** 2 + y_data ** 2 - 2 * yint * y_data + yint ** 2 - (kernel_radius * h_data) ** 2
    det = bb ** 2 - 4 * aa * cc

    # create a filter for particles that do not contribute to the cross-section
    filter_det = det >= 0
    det = np.sqrt(det)
    cc = None

    output = np.zeros(pixels)

    # the starting and ending x coordinates of the lines intersections with a particle's smoothing circle
    xstart = ((-bb[filter_det] - det[filter_det]) / (2 * aa)).clip(a_min=x1, a_max=x2)
    xend = ((-bb[filter_det] + det[filter_det]) / (2 * aa)).clip(a_min=x1, a_max=x2)
    bb, det = None, None

    # the start and end distances which lie within a particle's smoothing circle.
    rstart = np.sqrt((xstart - x1) ** 2 + ((gradient * xstart + yint) - y1) ** 2)
    rend = np.sqrt((xend - x1) ** 2 + (((gradient * xend + yint) - y1) ** 2))
    xstart, xend = None, None

    # the maximum and minimum pixels that each particle contributes to.
    ipixmin = np.rint(rstart / pixwidth).clip(a_min=0, a_max=pixels)
    ipixmax = np.rint(rend / pixwidth).clip(a_min=0, a_max=pixels)
    rstart, rend = None, None

    # iterate through the indices of all non-filtered particles
    for i in prange(len(x_data[filter_det])):
        # determine contributions to all affected pixels for this particle
        xpix = x1 + (np.arange(int(ipixmin[i]), int(ipixmax[i])) + 0.5) * xpixwidth
        ypix = gradient * xpix + yint
        dy = ypix - y_data[filter_det][i]
        dx = xpix - x_data[filter_det][i]

        q2 = (dx * dx + dy * dy) * (1 / (h_data[filter_det][i] * h_data[filter_det][i]))
        wab = weight_function(np.sqrt(q2), 2)

        # add contributions to output total
        for ipix in prange(int(ipixmax[i]) - int(ipixmin[i])):
            atomic_add(output, ipix + int(ipixmin[i]), term[i] * wab[ipix])

    return output


# For the GPU, the numba code is compiled using a factory function approach. This is required
# since a CUDA numba kernel cannot easily take weight_function as an argument.
def _fast_2d_cross_gpu(target, x_data, y_data, mass_data, rho_data, h_data, weight_function, kernel_radius, pixels, x1,
                       x2, y1, y2):
    # determine the slope of the cross-section line
    gradient = 0
    if not x2 - x1 == 0:
        gradient = (y2 - y1) / (x2 - x1)
    yint = y2 - gradient * x2

    # determine the fraction of the line that one pixel represents
    xlength = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    pixwidth = xlength / pixels
    xpixwidth = (x2 - x1) / pixels
    aa = 1 + gradient ** 2

    # Underlying GPU numba-compiled code for 2D->1D cross-sections
    @cuda.jit(fastmath=True)
    def _2d_func(target, x_data, y_data, mass_data, rho_data, h_data, kernel_radius, pixels, x1, x2, y1, y2, image):
        i = cuda.grid(1)
        if i < target.size:
            term = target[i] * mass_data[i] / (rho_data[i] * h_data[i] ** 2)

            # the intersections between the line and a particle's 'smoothing circle' are
            # found by solving a quadratic equation with the below values of a, b, and c.
            # if the determinant is negative, the particle does not contribute to the
            # cross-section, and can be removed.
            bb = 2 * gradient * (yint - y_data[i]) - 2 * x_data[i]
            cc = x_data[i] ** 2 + y_data[i] ** 2 - 2 * yint * y_data[i] + yint ** 2 - (kernel_radius * h_data[i]) ** 2
            det = bb ** 2 - 4 * aa * cc

            # create a filter for particles that do not contribute to the cross-section
            if det < 0:
                return

            det = math.sqrt(det)

            # the starting and ending x coordinates of the lines intersections with a particle's smoothing circle
            xstart = min(max(x1, (-bb - det) / (2 * aa)), x2)
            xend = min(max(x1, (-bb + det) / (2 * aa)), x2)

            # the start and end distances which lie within a particle's smoothing circle.
            rstart = math.sqrt((xstart - x1) ** 2 + ((gradient * xstart + yint) - y1) ** 2)
            rend = math.sqrt((xend - x1) ** 2 + (((gradient * xend + yint) - y1) ** 2))

            # the maximum and minimum pixels that each particle contributes to.
            ipixmin = min(max(0, round(rstart / pixwidth)), pixels)
            ipixmax = min(max(0, round(rend / pixwidth)), pixels)

            # iterate through all affected pixels
            for ipix in range(ipixmin, ipixmax):
                # determine contributions to all affected pixels for this particle
                xpix = x1 + (ipix + 0.5) * xpixwidth
                ypix = gradient * xpix + yint
                dy = ypix - y_data[i]
                dx = xpix - x_data[i]

                q2 = (dx * dx + dy * dy) * (1 / (h_data[i] * h_data[i]))
                wab = weight_function(math.sqrt(q2), 2)

                # add contributions to output total.
                cuda.atomic.add(image, ipix, wab * term)

    threadsperblock = 32
    blockspergrid = (target.size + (threadsperblock - 1)) // threadsperblock

    # transfer relevant data to the GPU
    d_target = cuda.to_device(target)
    d_x = cuda.to_device(x_data)
    d_y = cuda.to_device(y_data)
    d_m = cuda.to_device(mass_data)
    d_rho = cuda.to_device(rho_data)
    d_h = cuda.to_device(h_data)

    # CUDA kernels have no return values, so the image data must be
    # allocated on the device beforehand.
    d_image = cuda.to_device(np.zeros(pixels))

    # execute the newly compiled GPU kernel
    _2d_func[blockspergrid, threadsperblock](d_target, d_x, d_y, d_m, d_rho, d_h, kernel_radius, pixels, x1, x2, y1, y2,
                                             d_image)

    return d_image.copy_to_host()
