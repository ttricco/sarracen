import numpy as np

from sarracen import SarracenDataFrame
from sarracen.kernels import Kernel


def interpolate2D(data: SarracenDataFrame,
                  x: str,
                  y: str,
                  target: str,
                  kernel: Kernel,
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
        raise ValueError(f"pixcountx must be greater than zero!")
    if pixcounty <= 0:
        raise ValueError(f"pixcounty must be greater than zero!")

    image = np.zeros((pixcounty, pixcountx))

    # iterate through all pixels
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
                wab = kernel.w(np.sqrt(q2), 2)

                # add contribution to image
                image[jpix][ipix] += term * wab

    return image
