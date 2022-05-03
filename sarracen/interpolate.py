import numpy as np


def interpolate2D(data, x, y, target, kernel, xmin, ymin, pixwidthx, pixwidthy, pixcountx, pixcounty):
    """
    Interpolates particle data in a SarracenDataFrame across two directional axes to a 2D
    grid of pixels.

    :param data: The particle data, in a SarracenDataFrame.
    :param x: The column label of the x-directional axis.
    :param y: The column label of the y-directional axis.
    :param target: The column label of the target smoothing data.
    :param kernel: The kernel to use for smoothing the target data.
    :param xmin: The starting x-coordinate (in particle data space)
    :param ymin: The starting y-coordinate (in particle data space)
    :param pixcountx: The number of pixels in the output image in the x-direction
    :param pixcounty: The number of pixels in the output image in the y-direction
    :param pixwidthx: The width that each pixel represents in particle data space.
    :param pixwidthy: The height that each pixel represents in particle data space
    :return: The output image, in a 2-dimensional numpy array.
    """
    image = np.zeros((pixcountx, pixcounty))

    # iterate through all pixels
    for i, particle in data.iterrows():
        # dimensionless weight
        # w_i = m_i / (rho_i * (h_i) ** 2)
        weight = particle['m'] / (particle['rho'] * particle['h'] ** 2)

        # normalize the weight using the kernel normalization constant
        termnorm = kernel.cnormk2D * weight
        # skip particles with 0 weight
        if termnorm <= 0: continue

        # kernel radius scaled by the particle's 'h' value
        radkern = kernel.radkernel * particle['h']
        term = termnorm * particle[target]
        hi1 = 1 / particle['h']
        hi21 = hi1 ** 2

        part_x = particle[x]
        part_y = particle[y]

        # determine the min/max x&y coordinates affected by this particle
        ipixmin = int((part_x - radkern - xmin)/np.abs(pixwidthx))
        jpixmin = int((part_y - radkern - ymin)/np.abs(pixwidthy))
        ipixmax = int((part_x + radkern - xmin)/np.abs(pixwidthx))
        jpixmax = int((part_y + radkern - ymin)/np.abs(pixwidthy))

        # ensure that the min/max x&y coordinates remain within the bounds of the image
        if ipixmin < 0: ipixmin = 0
        if ipixmax > pixcountx: ipixmax = pixcountx
        if jpixmin < 0: jpixmin = 0
        if jpixmax > pixcounty: jpixmax = pixcounty

        # precalculate derivatives in the x-direction (optimization)
        dx2i = np.zeros(pixcountx)
        for ipix in range(ipixmin, ipixmax):
            dx2i[ipix] = ((xmin + (ipix - 0.5) * pixwidthx - part_x) ** 2) * hi21

        # traverse horizontally through affected pixels
        for jpix in range(jpixmin, jpixmax):
            # determine derivatives in the y-direction
            ypix = ymin + (jpix - 0.5) * pixwidthy
            dy = ypix - part_y
            dy2 = dy * dy * hi21

            for ipix in range(ipixmin, ipixmax):
                # calculate contribution at i, j due to particle at x, y
                q2 = dx2i[ipix] + dy2
                wab = kernel.w(q2, 2)

                # add contribution to image
                image[ipix][jpix] += term * wab

    return image
