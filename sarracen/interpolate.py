import numpy as np

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
                       x1: float,
                       y1: float,
                       x2: float,
                       y2: float,
                       pixcount: int):
    if np.isclose(y2, y1) and np.isclose(x2, x1):
        raise ValueError('Zero length cross section!')

    if pixcount <= 0:
        raise ValueError('pixcount must be greater than zero!')

    if kernel.ndims != 2:
        raise ValueError("Kernel must be two-dimensional!")

    output = np.zeros(pixcount)

    gradient = 0
    if not np.isclose(x2, x1):
        gradient = (y2 - y1) / (x2 - x1)
    yint = y2 - gradient * x2

    xlength = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    pixwidth = xlength / pixcount
    xpixwidth = (x2 - x1) / pixcount

    particles = data.copy()[['m', 'rho', 'h', target, x, y, ]]
    particles['weight'] = particles['m'] / (particles['rho'] * particles['h'] ** 2)
    particles = particles[particles['weight'] > 0]
    particles['radkern'] = kernel.radkernel * particles['h']
    particles['term'] = particles['weight'] * particles[target]
    particles['hi1'] = 1 / particles['h']

    aa = 1 + gradient**2
    particles['bb'] = 2 * gradient * (yint - particles[y]) - 2 * particles[x]
    particles['cc'] = particles[x] ** 2 - particles[y] ** 2 - 2 * yint * particles[y] + yint ** 2 - particles['radkern'] ** 2
    particles['determinant'] = particles['bb'] ** 2 - 4 * aa * particles['cc']

    particles = particles[particles['determinant'] > 0]
    particles['det'] = np.sqrt(particles['determinant'])
    particles['xstart'] = ((-particles['bb'] - particles['det']) / (2 * aa)).clip(lower=x1, upper=x2)
    particles['xend'] = ((-particles['bb'] + particles['det']) / (2 * aa)).clip(lower=y1, upper=y2)
    particles['ystart'] = gradient * particles['xstart'] + yint
    particles['yend'] = gradient * particles['xend'] + yint

    particles['rstart'] = np.sqrt((particles['xstart']-x1)**2 + (particles['ystart']-y1)**2)
    particles['rend'] = np.sqrt((particles['xend']-x1)**2 + ((particles['yend']-y1)**2))
    particles['ipixmin'] = np.rint(particles['rstart']/pixwidth).clip(lower=0, upper=pixcount)
    particles['ipixmax'] = np.rint(particles['rend']/pixwidth).clip(lower=0, upper=pixcount)

    # iterate through all particles
    for part in particles[['ipixmin', 'ipixmax', x, y, 'hi1', 'term']].itertuples():
        for ipix in range(int(part[1]), int(part[2])):
            xpix = x1 + (ipix+0.5)*xpixwidth
            ypix = gradient*xpix + yint
            dy = ypix - part[4]
            dx = xpix - part[3]
            q2 = (dx*dx + dy*dy)*part[5]*part[5]

            wab = kernel.w(np.sqrt(q2))
            output[ipix] += part[6] * wab

    return output
