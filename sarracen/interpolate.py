import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame

from sarracen import SarracenDataFrame
from sarracen.kernels import BaseKernel


def interpolate2D(data: SarracenDataFrame,
                  x: str,
                  y: str,
                  target: str,
                  kernel: BaseKernel,
                  pixwidthx: float,
                  pixwidthy: float,
                  xmin: float = 0,
                  ymin: float = 0,
                  pixcountx: int = 480,
                  pixcounty: int = 480):
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
    if pixwidthx <= 0:
        raise ValueError("pixwidthx must be greater than zero!")
    if pixwidthy <= 0:
        raise ValueError("pixwidthy must be greater than zero!")
    if pixcountx <= 0:
        raise ValueError("pixcountx must be greater than zero!")
    if pixcounty <= 0:
        raise ValueError("pixcounty must be greater than zero!")

    if kernel.ndims != 2:
        raise ValueError("Kernel must be two-dimensional!")

    image = np.zeros((pixcounty, pixcountx))

    # iterate through all particles
    for i, particle in data.iterrows():
        # dimensionless weight
        # w_i = m_i / (rho_i * (h_i) ** 2)
        weight = particle['m'] / (particle['rho'] * particle['h'] ** 2)

        # skip particles with 0 weight
        if weight <= 0:
            continue

        # kernel radius scaled by the particle's 'h' value
        radkern = kernel.radkernel * particle['h']
        term = weight * particle[target]
        hi1 = 1 / particle['h']
        hi21 = hi1 ** 2

        part_x = particle[x]
        part_y = particle[y]

        # determine the min/max x&y coordinates affected by this particle
        ipixmin = int(np.rint((part_x - radkern - xmin) / pixwidthx))
        jpixmin = int(np.rint((part_y - radkern - ymin) / pixwidthy))
        ipixmax = int(np.rint((part_x + radkern - xmin) / pixwidthx))
        jpixmax = int(np.rint((part_y + radkern - ymin) / pixwidthy))

        # ensure that the min/max x&y coordinates remain within the bounds of the image
        if ipixmin < 0:
            ipixmin = 0
        if ipixmax > pixcountx:
            ipixmax = pixcountx
        if jpixmin < 0:
            jpixmin = 0
        if jpixmax > pixcounty:
            jpixmax = pixcounty

        # precalculate differences in the x-direction (optimization)
        dx2i = np.zeros(pixcountx)
        for ipix in range(ipixmin, ipixmax):
            dx2i[ipix] = ((xmin + (ipix + 0.5) * pixwidthx - part_x) ** 2) * hi21

        # traverse horizontally through affected pixels
        for jpix in range(jpixmin, jpixmax):
            # determine differences in the y-direction
            ypix = ymin + (jpix + 0.5) * pixwidthy
            dy = ypix - part_y
            dy2 = dy * dy * hi21

            for ipix in range(ipixmin, ipixmax):
                # calculate contribution at i, j due to particle at x, y
                q2 = dx2i[ipix] + dy2
                wab = kernel.w(np.sqrt(q2))

                # add contribution to image
                image[jpix][ipix] += term * wab

    return image


def interpolate2DCross(data: SarracenDataFrame,
                       x: str,
                       y: str,
                       target: str,
                       kernel: BaseKernel,
                       x1: float = 0,
                       y1: float = 0,
                       x2: float = 1,
                       y2: float = 1,
                       pixcount: int = 500) -> ndarray:
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
    :param pixcount: The number of pixels in the output over the entire cross-sectional line.
    :return: The interpolated output, in a 1-dimensional numpy array.
    """
    if np.isclose(y2, y1) and np.isclose(x2, x1):
        raise ValueError('Zero length cross section!')

    if pixcount <= 0:
        raise ValueError('pixcount must be greater than zero!')

    if kernel.ndims != 2:
        raise ValueError("Kernel must be two-dimensional!")

    output = np.zeros(pixcount)

    # determine the slope of the cross-section line
    gradient = 0
    if not np.isclose(x2, x1):
        gradient = (y2 - y1) / (x2 - x1)
    yint = y2 - gradient * x2

    # determine the fraction of the line that one pixel represents
    xlength = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    pixwidth = xlength / pixcount
    xpixwidth = (x2 - x1) / pixcount

    # copy necessary data columns into a new dataframe for vectorized operations
    parts = pd.DataFrame()
    parts['x'] = data[x]
    parts['y'] = data[y]
    parts['weight'] = data['m'] / (data['rho'] * data['h'] ** 2)
    parts['h'] = data['h']
    parts['term'] = parts['weight'] * data[target]

    # filter out particles with 0 weight
    parts = parts[parts['weight'] > 0]
    parts = parts.drop('weight', axis=1)

    # the intersections between the line and a particle's 'smoothing circle' are
    # found by solving a quadratic equation with the below values of a, b, and c.
    # if the determinant is negative, the particle does not contribute to the
    # cross-section, and can be removed.
    aa = 1 + gradient**2
    parts['bb'] = 2 * gradient * (yint - parts[y]) - 2 * parts[x]
    parts['cc'] = parts[x] ** 2 + parts[y] ** 2 - 2 * yint * parts[y] + yint ** 2 - (kernel.radkernel * parts['h']) ** 2
    parts['det'] = parts['bb'] ** 2 - 4 * aa * parts['cc']
    parts = parts[parts['det'] >= 0]
    parts['det'] = np.sqrt(parts['det'])
    parts = parts.drop(['cc'], axis=1)

    # the starting and ending x coordinates of the lines intersections with a particle's smoothing circle
    parts['xstart'] = ((-parts['bb'] - parts['det']) / (2 * aa)).clip(lower=x1, upper=x2)
    parts['xend'] = ((-parts['bb'] + parts['det']) / (2 * aa)).clip(lower=x1, upper=x2)
    parts = parts.drop(['det', 'bb'], axis=1)

    # the start and end distances which lie within a particle's smoothing circle.
    parts['rstart'] = np.sqrt((parts['xstart']-x1)**2 + ((gradient * parts['xstart'] + yint)-y1)**2)
    parts['rend'] = np.sqrt((parts['xend']-x1)**2 + (((gradient * parts['xend'] + yint)-y1)**2))
    parts = parts.drop(['xstart', 'xend'], axis=1)

    # the maximum and minimum pixels that each particle contributes to.
    parts['ipixmin'] = np.rint(parts['rstart']/pixwidth).clip(lower=0, upper=pixcount)
    parts['ipixmax'] = np.rint(parts['rend']/pixwidth).clip(lower=0, upper=pixcount)
    parts = parts.drop(['rstart', 'rend'], axis=1)

    # iterate through all particles.
    for part in parts[['ipixmin', 'ipixmax', 'x', 'y', 'h', 'term']].itertuples():
        # iterate through pixels contributed to.
        for ipix in range(int(part.ipixmin), int(part.ipixmax)):
            # determine the x & y differences between the particle and this pixel.
            xpix = x1 + (ipix+0.5)*xpixwidth
            ypix = gradient*xpix + yint
            dy = ypix - part.y
            dx = xpix - part.x

            # determine the particles contribution to this pixel, and add it to the output.
            q2 = (dx*dx + dy*dy)*(1 / (part.h*part.h))
            wab = kernel.w(np.sqrt(q2))
            output[ipix] += part.term * wab

    return output
