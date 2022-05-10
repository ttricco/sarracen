import numpy as np

from sarracen.kernels import BaseKernel


def interpolate2D(data: 'SarracenDataFrame',
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

    # clone the necessary data columns into a new dataframe for vectorized operations
    parts = data.copy()[['m', 'rho', 'h', target, x, y]]
    parts['weight'] = parts['m'] / (parts['rho'] * parts['h'] ** 2)

    # filter out particles with 0 weight
    parts = parts[parts['weight'] > 0]
    parts['radkern'] = kernel.radkernel * parts['h']
    parts['term'] = parts['weight'] * parts[target]
    parts['hi21'] = (1 / parts['h']) ** 2

    # determine maximum and minimum pixels that each particle contributes to
    parts['ipixmin'] = np.rint((parts[x] - parts['radkern'] - xmin) / pixwidthx).clip(lower=0, upper=pixcountx)
    parts['jpixmin'] = np.rint((parts[y] - parts['radkern'] - ymin) / pixwidthy).clip(lower=0, upper=pixcounty)
    parts['ipixmax'] = np.rint((parts[x] + parts['radkern'] - xmin) / pixwidthx).clip(lower=0, upper=pixcountx)
    parts['jpixmax'] = np.rint((parts[y] + parts['radkern'] - ymin) / pixwidthy).clip(lower=0, upper=pixcounty)

    # iterate through all pixels
    for part in parts[['ipixmin', 'ipixmax', x, 'hi21', 'jpixmin', 'jpixmax', y, 'term']].itertuples():
        # precalculate differences in the x-direction (optimization)
        dx2i = np.zeros(pixcountx)
        for ipix in range(int(part[1]), int(part[2])):
            dx2i[ipix] = ((xmin + (ipix + 0.5) * pixwidthx - part[3]) ** 2) * part[4]

        # traverse horizontally through affected pixels
        for jpix in range(int(part[5]), int(part[6])):
            # determine differences in the y-direction
            ypix = ymin + (jpix + 0.5) * pixwidthy
            dy = ypix - part[7]
            dy2 = dy * dy * part[4]

            for ipix in range(int(part[1]), int(part[2])):
                # calculate contribution at i, j due to particle at x, y
                q2 = dx2i[ipix] + dy2
                wab = kernel.w(np.sqrt(q2))

                # add contribution to image
                image[jpix][ipix] += part[8] * wab

    return image
