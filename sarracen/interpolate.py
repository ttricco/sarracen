"""
Contains several interpolation functions which produce interpolated 2D or 1D arrays of SPH data.
"""
import numpy as np
from numba import prange, njit
from scipy.spatial.transform import Rotation as R, Rotation

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
    # x & y columns default to the variables determined by the SarracenDataFrame.
    if x is None:
        x = data.xcol
    if y is None:
        y = data.ycol

    return x, y


def _snap_boundaries(data, x, y, x_min, x_max, y_min, y_max):
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


def _set_pixels(data, x_pixels, y_pixels, x_min, x_max, y_min, y_max):
    # set # of pixels to maintain an aspect ratio that is the same as the underlying bounds of the data.
    if x_pixels is None and y_pixels is None:
        x_pixels = 512
    if x_pixels is None:
        x_pixels = int(np.rint(y_pixels * ((x_max - x_min) / (y_max - y_min))))
    if y_pixels is None:
        y_pixels = int(np.rint(x_pixels * ((y_max - y_min) / (x_max - x_min))))

    return x_pixels, y_pixels


def _verify_columns(data, target, x, y):
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


def _default_kernel(data, kernel):
    if kernel is None:
        return data.kernel
    return kernel


def _check_boundaries(x_pixels, y_pixels, x_min, x_max, y_min, y_max):
    if x_max - x_min <= 0:
        raise ValueError("`xmax` must be greater than `xmin`!")
    if y_max - y_min <= 0:
        raise ValueError("`ymax` must be greater than `ymin`!")
    if x_pixels <= 0:
        raise ValueError("`x_pixels` must be greater than zero!")
    if y_pixels <= 0:
        raise ValueError("`y_pixels` must be greater than zero!")


def _check_dimension(data, dim):
    if data.get_dim() != dim:
        raise ValueError(f"Dataset is not {dim}-dimensional.")


def _rotate_data(data, x, y, z, rotation, origin):
    x_data = data[x].to_numpy()
    y_data = data[y].to_numpy()
    z_data = data[z].to_numpy()
    if rotation is not None:
        if not isinstance(rotation, Rotation):
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


def interpolate_2d(data: 'SarracenDataFrame',
                   target: str,
                   x: str = None,
                   y: str = None,
                   kernel: BaseKernel = None,
                   x_pixels: int = None,
                   y_pixels: int = None,
                   x_min: float = None,
                   x_max: float = None,
                   y_min: float = None,
                   y_max: float = None) -> np.ndarray:
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
    x: str
        Column label of the x-directional axis.
    y: str
        Column label of the y-directional axis.
    kernel: BaseKernel
        Kernel to use for smoothing the target data.
    x_pixels: int, optional
        Number of pixels in the output image in the x-direction.
    y_pixels: int, optional
        Number of pixels in the output image in the y-direction.
    x_min: float, optional
        Minimum bound in the x-direction (in particle data space).
    x_max: float, optional
        Maximum bound in the x-direction (in particle data space).
    y_min: float, optional
        Minimum bound in the y-direction (in particle data space).
    y_max: float, optional
        Maximum bound in the y-direction (in particle data space).

    Returns
    -------
    np.ndarray (2-Dimensional)
        The resulting interpolated grid.

    Raises
    -------
    ValueError
        If `x_pixels` or `y_pixels` are less than or equal to zero, or
        if the specified `x` and `y` minimum and maximum values result in an invalid region.
    KeyError
        If `target`, `x`, `y`, mass, density, or smoothing length columns do not
        exist in `data`.
    """
    x, y = _default_xy(data, x, y)
    _verify_columns(data, x, y, target)

    x_min, x_max, y_min, y_max = _snap_boundaries(data, x, y, x_min, x_max, y_min, y_max)
    x_pixels, y_pixels = _set_pixels(data, x_pixels, y_pixels, x_min, x_max, y_min, y_max)
    _check_boundaries(x_pixels, y_pixels, x_min, x_max, y_min, y_max)

    kernel = _default_kernel(data, kernel)
    _check_dimension(data, 2)

    return _fast_2d(data[target].to_numpy(), data[x].to_numpy(), data[y].to_numpy(), data['m'].to_numpy(),
                    data['rho'].to_numpy(), data['h'].to_numpy(), kernel.w, kernel.get_radius(), x_pixels,
                    y_pixels, x_min, x_max, y_min, y_max)


def interpolate_2d_cross(data: 'SarracenDataFrame',
                         target: str,
                         x: str = None,
                         y: str = None,
                         kernel: BaseKernel = None,
                         pixels: int = None,
                         x1: float = None,
                         x2: float = None,
                         y1: float = None,
                         y2: float = None) -> np.ndarray:
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
    x: str
        Column label of the x-directional axis.
    y: str
        Column label of the y-directional axis.
    kernel: BaseKernel
        Kernel to use for smoothing the target data.
    pixels: int, optional
        Number of points in the resulting line plot in the x-direction.
    x1: float, optional
        Starting x-coordinate of the line (in particle data space).
    x2: float, optional
        Ending x-coordinate of the line (in particle data space).
    y1: float, optional
        Starting y-coordinate of the line (in particle data space).
    y2: float, optional
        Ending y-coordinate of the line (in particle data space).

    Returns
    -------
    np.ndarray (1-Dimensional)
        The resulting interpolated output.

    Raises
    -------
    ValueError
        If `x_pixels` or `y_pixels` are less than or equal to zero, or
        if the specified `x1`, `x2`, `y1`, and `y2` are all the same
        (indicating a zero-length cross-section).
    KeyError
        If `target`, `x`, `y`, mass, density, or smoothing length columns do not
        exist in `data`.
    """
    x, y = _default_xy(data, x, y)
    _verify_columns(data, x, y, target)

    x1, x2, y1, y2 = _snap_boundaries(data, x, y, x1, x2, y1, y2)
    if y2 == y1 and x2 == x1:
        raise ValueError('Zero length cross section!')

    kernel = _default_kernel(data, kernel)
    _check_dimension(data, 2)

    if pixels <= 0:
        raise ValueError('pixcount must be greater than zero!')

    return _fast_2d_cross(data[target].to_numpy(), data[x].to_numpy(), data[y].to_numpy(), data['m'].to_numpy(),
                          data['rho'].to_numpy(), data['h'].to_numpy(), kernel.w, kernel.get_radius(), pixels, x1,
                          x2, y1, y2)


def interpolate_3d(data: 'SarracenDataFrame',
                   target: str,
                   x: str = None,
                   y: str = None,
                   kernel: BaseKernel = None,
                   integral_samples: int = 1000,
                   rotation: np.ndarray = None,
                   origin: np.ndarray = None,
                   x_pixels: int = None,
                   y_pixels: int = None,
                   x_min: float = None,
                   x_max: float = None,
                   y_min: float = None,
                   y_max: float = None):
    """ Interpolate 3D particle data to a 2D grid of pixels.

    Interpolates three-dimensional particle data in a SarracenDataFrame. The data
    is interpolated to a 2D grid of pixels, by summing contributions in columns which
    span the z-axis.

    Parameters
    ----------
    data : SarracenDataFrame
        Particle data, in a SarracenDataFrame.
    x: str
        Column label of the x-directional axis.
    y: str
        Column label of the y-directional axis.
    target: str
        Column label of the target smoothing data.
    kernel: BaseKernel
        Kernel to use for smoothing the target data.
    integral_samples: int, optional
        Number of sample points to take when approximating the 2D column kernel.
    rotation: array_like or Rotation, optional
        The rotation to apply to the data before interpolation. If defined as an array, the
        order of rotations is [z, y, x] in degrees
    origin: array_like, optional
        Point of rotation of the data, in [x, y, z] form. Defaults to the centre
        point of the bounds of the data.
    x_pixels: int, optional
        Number of pixels in the output image in the x-direction.
    y_pixels: int, optional
        Number of pixels in the output image in the y-direction.
    x_min: float, optional
        Minimum bound in the x-direction (in particle data space).
    x_max: float, optional
        Maximum bound in the x-direction (in particle data space).
    y_min: float, optional
        Minimum bound in the y-direction (in particle data space).
    y_max: float, optional
        Maximum bound in the y-direction (in particle data space).

    Returns
    -------
    ndarray
        The interpolated output image, in a 2-dimensional numpy array.

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
    x_pixels, y_pixels = _set_pixels(data, x_pixels, y_pixels, x_min, x_max, y_min, y_max)
    _check_boundaries(x_pixels, y_pixels, x_min, x_max, y_min, y_max)

    x_data, y_data, _ = _rotate_data(data, x, y, data.zcol, rotation, origin)
    kernel = _default_kernel(data, kernel)
    _check_dimension(data, 3)

    return _fast_3d(data[target].to_numpy(), x_data, y_data, data['m'].to_numpy(),
                    data['rho'].to_numpy(), data['h'].to_numpy(), kernel.get_column_kernel(integral_samples),
                    kernel.get_radius(), integral_samples, x_pixels, y_pixels, x_min, x_max, y_min, y_max)


def interpolate_3d_cross(data: 'SarracenDataFrame',
                         target: str,
                         z_slice: float = None,
                         x: str = None,
                         y: str = None,
                         z: str = None,
                         kernel: BaseKernel = None,
                         rotation: np.ndarray = None,
                         origin: np.ndarray = None,
                         x_pixels: int = None,
                         y_pixels: int = None,
                         x_min: float = None,
                         x_max: float = None,
                         y_min: float = None,
                         y_max: float = None):
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
        The z-axis value to take the cross-section at.
    x: str
        The column label of the x-directional data to interpolate over.
    y: str
        The column label of the y-directional data to interpolate over.
    z: str
        The column label of the z-directional data to interpolate over.
    kernel: BaseKernel
        The kernel to use for smoothing the target data.
    rotation: array_like or Rotation, optional
        The rotation to apply to the data before interpolation. If defined as an array, the
        order of rotations is [z, y, x] in degrees.
    origin: array_like, optional
        Point of rotation of the data, in [x, y, z] form. Defaults to the centre
        point of the bounds of the data.
    x_pixels: int, optional
        Number of pixels in the output image in the x-direction.
    y_pixels: int, optional
        Number of pixels in the output image in the y-direction.
    x_min: float, optional
        Minimum bound in the x-direction (in particle data space).
    x_max: float, optional
        Maximum bound in the x-direction (in particle data space).
    y_min: float, optional
        Minimum bound in the y-direction (in particle data space).
    y_max: float, optional
        Maximum bound in the y-direction (in particle data space).

    Returns
    -------
    ndarray
        The interpolated output image, in a 2-dimensional numpy array. Dimensions are
        structured in reverse order, where (x, y) -> [y][x]

    Raises
    -------
    ValueError
        If `pixwidthx`, `pixwidthy`, `pixcountx`, or `pixcounty` are less than or equal to zero, or
        if the specified `x` and `y` minimum and maximums result in an invalid region, or
        if the provided data is not 3-dimensional.
    KeyError
        If `target`, `x`, `y`, mass, density, or smoothing length columns do not
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
    x_pixels, y_pixels = _set_pixels(data, x_pixels, y_pixels, x_min, x_max, y_min, y_max)
    _check_boundaries(x_pixels, y_pixels, x_min, x_max, y_min, y_max)

    kernel = _default_kernel(data, kernel)

    _check_dimension(data, 3)

    x_data, y_data, z_data = _rotate_data(data, x, y, z, rotation, origin)

    return _fast_3d_cross(data[target].to_numpy(), z_slice, x_data, y_data, z_data,
                          data['m'].to_numpy(), data['rho'].to_numpy(), data['h'].to_numpy(), kernel.w,
                          kernel.get_radius(), x_pixels, y_pixels, x_min, x_max, y_min, y_max)


# Underlying numba-compiled code for 2D interpolation
@njit(parallel=True, fastmath=True)
def _fast_2d(target, x_data, y_data, mass_data, rho_data, h_data, weight_function, kernel_radius, x_pixels, y_pixels,
             x_min, x_max, y_min, y_max):
    image = np.zeros((y_pixels, x_pixels))
    pixwidthx = (x_max - x_min) / x_pixels
    pixwidthy = (y_max - y_min) / y_pixels

    term = (target * mass_data / (rho_data * h_data ** 2))

    # determine maximum and minimum pixels that each particle contributes to
    ipixmin = np.rint((x_data - kernel_radius * h_data - x_min) / pixwidthx) \
        .clip(a_min=0, a_max=x_pixels)
    jpixmin = np.rint((y_data - kernel_radius * h_data - y_min) / pixwidthy) \
        .clip(a_min=0, a_max=y_pixels)
    ipixmax = np.rint((x_data + kernel_radius * h_data - x_min) / pixwidthx) \
        .clip(a_min=0, a_max=x_pixels)
    jpixmax = np.rint((y_data + kernel_radius * h_data - y_min) / pixwidthy) \
        .clip(a_min=0, a_max=y_pixels)

    # iterate through the indexes of non-filtered particles
    for i in prange(len(term)):
        # precalculate differences in the x-direction (optimization)
        dx2i = ((x_min + (np.arange(int(ipixmin[i]), int(ipixmax[i])) + 0.5) * pixwidthx - x_data[i]) ** 2) \
               * (1 / (h_data[i] ** 2))

        # determine differences in the y-direction
        ypix = y_min + (np.arange(int(jpixmin[i]), int(jpixmax[i])) + 0.5) * pixwidthy
        dy = ypix - y_data[i]
        dy2 = dy * dy * (1 / (h_data[i] ** 2))

        # calculate contributions at pixels i, j due to particle at x, y
        q2 = dx2i + dy2.reshape(len(dy2), 1)
        wab = weight_function(np.sqrt(q2), 2)

        # add contributions to image
        image[int(jpixmin[i]):int(jpixmax[i]), int(ipixmin[i]):int(ipixmax[i])] += (wab * term[i])

    return image


# Underlying numba-compiled code for 2D->1D cross-sections
@njit(parallel=True, fastmath=True)
def _fast_2d_cross(target, x_data, y_data, mass_data, rho_data, h_data, weight_function, kernel_radius, pixels, x1, x2,
                   y1, y2):
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

        # add contributions to output total, transformed by minimum/maximum pixels
        output[int(ipixmin[i]):int(ipixmax[i])] += (wab * term[filter_det][i])

    return output


# Underlying numba-compiled code for 3D column interpolation.
@njit(parallel=True, fastmath=True)
def _fast_3d(target, x_data, y_data, mass_data, rho_data, h_data, integrated_kernel, kernel_rad, int_samples, x_pixels,
             y_pixels, x_min, x_max, y_min, y_max):
    image = np.zeros((y_pixels, x_pixels))
    pixwidthx = (x_max - x_min) / x_pixels
    pixwidthy = (y_max - y_min) / y_pixels

    term = target * mass_data / (rho_data * h_data ** 2)

    # determine maximum and minimum pixels that each particle contributes to
    ipixmin = np.rint((x_data - kernel_rad * h_data - x_min) / pixwidthx).clip(a_min=0, a_max=x_pixels)
    jpixmin = np.rint((y_data - kernel_rad * h_data - y_min) / pixwidthy).clip(a_min=0, a_max=y_pixels)
    ipixmax = np.rint((x_data + kernel_rad * h_data - x_min) / pixwidthx).clip(a_min=0, a_max=x_pixels)
    jpixmax = np.rint((y_data + kernel_rad * h_data - y_min) / pixwidthy).clip(a_min=0, a_max=y_pixels)

    # iterate through the indexes of non-filtered particles
    for i in prange(len(term)):
        # precalculate differences in the x-direction (optimization)
        dx2i = ((x_min + (np.arange(int(ipixmin[i]), int(ipixmax[i])) + 0.5) * pixwidthx - x_data[i]) ** 2) \
               * (1 / (h_data[i] ** 2))

        # determine differences in the y-direction
        ypix = y_min + (np.arange(int(jpixmin[i]), int(jpixmax[i])) + 0.5) * pixwidthy
        dy = ypix - y_data[i]
        dy2 = dy * dy * (1 / (h_data[i] ** 2))

        # calculate contributions at pixels i, j due to particle at x, y
        q2 = dx2i + dy2.reshape(len(dy2), 1)
        wab = np.interp(np.sqrt(q2), np.linspace(0, kernel_rad, int_samples), integrated_kernel)

        # add contributions to image
        image[int(jpixmin[i]):int(jpixmax[i]), int(ipixmin[i]):int(ipixmax[i])] += (wab * term[i])

    return image


# Underlying numba-compiled code for 3D->2D cross-sections
@njit(parallel=True, fastmath=True)
def _fast_3d_cross(target, z_slice, x_data, y_data, z_data, mass_data, rho_data, h_data, weight_function, kernel_radius,
                   x_pixels, y_pixels, x_min, x_max, y_min, y_max):
    # Filter out particles that do not contribute to this cross-section slice
    term = target * mass_data / (rho_data * h_data ** 3)
    pixwidthx = (x_max - x_min) / x_pixels
    pixwidthy = (y_max - y_min) / y_pixels
    dz = z_slice - z_data

    filter_distance = np.abs(dz) < kernel_radius * h_data

    ipixmin = np.rint((x_data[filter_distance] - kernel_radius * h_data[filter_distance] - x_min) / pixwidthx).clip(
        a_min=0,
        a_max=x_pixels)
    jpixmin = np.rint((y_data[filter_distance] - kernel_radius * h_data[filter_distance] - y_min) / pixwidthy).clip(
        a_min=0,
        a_max=y_pixels)
    ipixmax = np.rint((x_data[filter_distance] + kernel_radius * h_data[filter_distance] - x_min) / pixwidthx).clip(
        a_min=0,
        a_max=x_pixels)
    jpixmax = np.rint((y_data[filter_distance] + kernel_radius * h_data[filter_distance] - y_min) / pixwidthy).clip(
        a_min=0,
        a_max=y_pixels)

    image = np.zeros((y_pixels, x_pixels))

    for i in prange(len(x_data[filter_distance])):
        # precalculate differences in the x-direction
        dx2i = (((x_min + (np.arange(int(ipixmin[i]), int(ipixmax[i])) + 0.5)
                  * pixwidthx - x_data[filter_distance][i]) ** 2)
                * (1 / (h_data[filter_distance][i] ** 2))) + (
                       (dz[filter_distance][i] ** 2) * (1 / h_data[filter_distance][i] ** 2))

        ypix = y_min + (np.arange(int(jpixmin[i]), int(jpixmax[i])) + 0.5) * pixwidthy
        dy = ypix - y_data[filter_distance][i]
        dy2 = dy * dy * (1 / (h_data[filter_distance][i] ** 2))

        q2 = dx2i + dy2.reshape(len(dy2), 1)
        contribution = (term[filter_distance][i] * weight_function(np.sqrt(q2), 3))
        image[int(jpixmin[i]):int(jpixmax[i]), int(ipixmin[i]):int(ipixmax[i])] += contribution

    return image
