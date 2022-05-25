import numpy as np
from numba import prange, njit

from sarracen.kernels import BaseKernel


def interpolate_2d(data: 'SarracenDataFrame',
                   target: str,
                   x: str,
                   y: str,
                   kernel: BaseKernel,
                   x_pixels: int = 512,
                   y_pixels: int = 512,
                   x_min: float = 0,
                   x_max: float = 1,
                   y_min: float = 0,
                   y_max: float = 1):
    """
    Interpolates particle data in a SarracenDataFrame across two directional axes to a 2D
    grid of pixels.

    :param data: The particle data, in a SarracenDataFrame.
    :param x: The column label of the x-directional axis.
    :param y: The column label of the y-directional axis.
    :param target: The column label of the target smoothing data.
    :param kernel: The kernel to use for smoothing the target data.
    :param pixwidthx: The width that each pixel represents in particle data space.
    :param pixwidthy: The height that each pixel represents in particle data space.
    :param xmin: The starting x-coordinate (in particle data space).
    :param ymin: The starting y-coordinate (in particle data space).
    :param pixcountx: The number of pixels in the output image in the x-direction.
    :param pixcounty: The number of pixels in the output image in the y-direction.
    :return: The output image, in a 2-dimensional numpy array.
    """
    if x_max - x_min <= 0:
        raise ValueError("`xmax` must be greater than `xmin`!")
    if y_max - y_min <= 0:
        raise ValueError("`ymax` must be greater than `ymin`!")
    if x_pixels <= 0:
        raise ValueError("`x_pixels` must be greater than zero!")
    if y_pixels <= 0:
        raise ValueError("`y_pixels` must be greater than zero!")

    return _fast_2d(data[target].to_numpy(), data[x].to_numpy(), data[y].to_numpy(), data['m'].to_numpy(),
                    data['rho'].to_numpy(), data['h'].to_numpy(), kernel.weight, kernel.get_radius(), x_pixels, y_pixels, x_min, x_max, y_min,
                    y_max)


# Underlying numba-compiled code for 2D interpolation
@njit(parallel=True, fastmath=True)
def _fast_2d(target, x_data, y_data, mass_data, rho_data, h_data, weight_function, kernel_radius, x_pixels, y_pixels, x_min, x_max, y_min, y_max):
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


def interpolate_2d_cross(data: 'SarracenDataFrame',
                         target: str,
                         x: str,
                         y: str,
                         kernel: BaseKernel,
                         pixels: int = 512,
                         x1: float = 0,
                         x2: float = 1,
                         y1: float = 0,
                         y2: float = 1) -> np.ndarray:
    """
    Interpolates particle data in a SarracenDataFrame across two directional axes to a 1D
    cross-section line.

    :param data: The particle data, in a SarracenDataFrame.
    :param x: The column label of the x-directional axis.
    :param y: The column label of the y-directional axis.
    :param target: The column label of the target smoothing data.
    :param kernel: The kernel to use for smoothing the target data.
    :param x1: The starting x-coordinate of the cross-section line. (in particle data space)
    :param y1: The starting y-coordinate of the cross-section line. (in particle data space)
    :param x2: The ending x-coordinate of the cross-section line. (in particle data space)
    :param y2: The ending y-coordinate of the cross-section line. (in particle data space)
    :return: The interpolated output, in a 1-dimensional numpy array.
    """
    if y2 == y1 and x2 == x1:
        raise ValueError('Zero length cross section!')

    if pixels <= 0:
        raise ValueError('pixcount must be greater than zero!')

    return _fast_2d_cross(data[target].to_numpy(), data[x].to_numpy(), data[y].to_numpy(), data['m'].to_numpy(),
                          data['rho'].to_numpy(), data['h'].to_numpy(), kernel.weight, kernel.get_radius(), pixels, x1,
                          x2, y1, y2)


# Underlying numba-compiled code for 2D->1D cross-sections
@njit(parallel=True, fastmath=True)
def _fast_2d_cross(target, x_data, y_data, mass_data, rho_data, h_data, weight_function, kernel_radius, pixels, x1, x2, y1, y2):
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


def interpolate_3d(data: 'SarracenDataFrame',
                   target: str,
                   x: str,
                   y: str,
                   kernel: BaseKernel,
                   integral_samples: int = 1000,
                   x_pixels: int = 512,
                   y_pixels: int = 512,
                   x_min: float = 0,
                   x_max: float = 1,
                   y_min: float = 0,
                   y_max: float = 1):
    """ Interpolate 3D particle data to a 2D grid of pixels.

    Interpolates three-dimensional particle data in a SarracenDataFrame. The data
    is interpolated to a 2D grid of pixels, by summing contributions in columns which
    span the z-axis.

    Parameters
    ----------
    data : SarracenDataFrame
        The particle data, in a SarracenDataFrame.
    x: str
        The column label of the x-directional axis.
    y: str
        The column label of the y-directional axis.
    target: str
        The column label of the target smoothing data.
    kernel: BaseKernel
        The kernel to use for smoothing the target data.
    pixwidthx: float
        The width that each pixel represents in particle data space.
    pixwidthy: float
        The height that each pixel represents in particle data space.
    xmin: float, optional
        The starting x-coordinate (in particle data space).
    ymin: float, optional
        The starting y-coordinate (in particle data space).
    x_pixels: int, optional
        The number of pixels in the output image in the x-direction.
    y_pixels: int, optional
        The number of pixels in the output image in the y-direction.
    int_samples: int, optional
        The number of sample points to take when approximating the 2D column kernel.

    Returns
    -------
    ndarray
        The interpolated output image, in a 2-dimensional numpy array.

    Raises
    -------
    ValueError
        If `pixwidthx`, `pixwidthy`, `pixcountx`, or `pixcounty` are less than or equal to zero.
    """
    if x_max - x_min <= 0:
        raise ValueError("`x_max` must be greater than `x_min`!")
    if y_max - y_min <= 0:
        raise ValueError("`y_max` must be greater than `y_min`!")
    if x_pixels <= 0:
        raise ValueError("`x_pixels` must be greater than zero!")
    if y_pixels <= 0:
        raise ValueError("`y_pixels` must be greater than zero!")

    return _fast_3d(data[target].to_numpy(), data[x].to_numpy(), data[y].to_numpy(), data['m'].to_numpy(),
                    data['rho'].to_numpy(), data['h'].to_numpy(), kernel.get_column_kernel(integral_samples),
                    kernel.get_radius(), integral_samples, x_pixels, y_pixels, x_min, x_max, y_min, y_max)


# Underlying numba-compiled code for 3D column interpolation.
@njit(parallel=True, fastmath=True)
def _fast_3d(target, x_data, y_data, mass_data, rho_data, h_data, integrated_kernel, kernel_rad, int_samples, x_pixels, y_pixels, x_min, x_max, y_min, y_max):
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


def interpolate_3d_cross(data: 'SarracenDataFrame',
                         target: str,
                         z_slice: float,
                         x: str,
                         y: str,
                         z: str,
                         kernel: BaseKernel,
                         x_pixels: int = 512,
                         y_pixels: int = 512,
                         x_min: float = 0,
                         x_max: float = 1,
                         y_min: float = 0,
                         y_max: float = 1):
    """ Interpolate 3D particle data to a 2D grid, using a 3D cross-section.

    Interpolates particle data in a SarracenDataFrame across three directional axes to a 2D
    grid of pixels. A cross-section is taken of the 3D data at a specific value of z, and
    the contributions of particles near the plane are interpolated to a 2D grid.

    Parameters
    ----------
    data : SarracenDataFrame
        The particle data to interpolate over.
    x: str
        The column label of the x-directional axis.
    y: str
        The column label of the y-directional axis.
    z: str
        The column label of the z-directional axis.
    target: str
        The column label of the target smoothing data.
    kernel: BaseKernel
        The kernel to use for smoothing the target data.
    zslice: float
        The z-axis value to take the cross-section at.
    pixwidthx: float
        The width that each pixel represents in particle data space.
    xmax: float
        The height that each pixel represents in particle data space.
    xmin: float, optional
        The starting x-coordinate (in particle data space).
    ymin: float, optional
        The starting y-coordinate (in particle data space).
    pixcountx: int, optional
        The number of pixels in the output image in the x-direction.
    pixcounty: int, optional
        The number of pixels in the output image in the y-direction.

    Returns
    -------
    ndarray
        The interpolated output image, in a 2-dimensional numpy array. Dimensions are
        structured in reverse order, where (x, y) -> [y][x]

    Raises
    -------
    ValueError
        If `pixwidthx`, `pixwidthy`, `pixcountx`, or `pixcounty` are less than or equal to zero.
    """
    if x_max - x_min <= 0:
        raise ValueError("`x_max` must be greater than `x_min`!")
    if y_max - y_min <= 0:
        raise ValueError("`y_max` must be greater than `y_min`!")
    if x_pixels <= 0:
        raise ValueError("`x_pixels` must be greater than zero!")
    if y_pixels <= 0:
        raise ValueError("`y_pixels` must be greater than zero!")

    return _fast_3d_cross(data[target].to_numpy(), z_slice, data[x].to_numpy(), data[y].to_numpy(), data[z].to_numpy(),
                          data['m'].to_numpy(), data['rho'].to_numpy(), data['h'].to_numpy(), kernel.weight,
                          kernel.get_radius(), x_pixels, y_pixels, x_min, x_max, y_min, y_max)


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

    ipixmin = np.rint((x_data[filter_distance] - kernel_radius * h_data[filter_distance] - x_min) / pixwidthx).clip(a_min=0,
                                                                                                                    a_max=x_pixels)
    jpixmin = np.rint((y_data[filter_distance] - kernel_radius * h_data[filter_distance] - y_min) / pixwidthy).clip(a_min=0,
                                                                                                                    a_max=y_pixels)
    ipixmax = np.rint((x_data[filter_distance] + kernel_radius * h_data[filter_distance] - x_min) / pixwidthx).clip(a_min=0,
                                                                                                                    a_max=x_pixels)
    jpixmax = np.rint((y_data[filter_distance] + kernel_radius * h_data[filter_distance] - y_min) / pixwidthy).clip(a_min=0,
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
