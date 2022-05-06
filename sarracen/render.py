import numpy as np
from matplotlib import pyplot as plt

from sarracen.interpolate import interpolate2D
from sarracen.kernels import BaseKernel, CubicSplineKernel


def snap(value: float):
    """
    Return a number snapped to the nearest integer, with 1e-4 tolerance.
    :param value: The number to snap.
    :return: An integer if a close integer is detected, otherwise return 'value'.
    """
    if np.isclose(value, np.rint(value), atol=1e-4):
        return np.rint(value)
    else:
        return value


def render(data: 'SarracenDataFrame',
           target: str,
           x: str = None,
           y: str = None,
           kernel: BaseKernel = CubicSplineKernel(2),
           xmin: float = None,
           ymin: float = None,
           xmax: float = None,
           ymax: float = None,
           pixcountx: int = 256,
           pixcounty: int = None) -> ('Figure', 'Axes'):
    """
    Render the data within a SarracenDataFrame to a 2D matplotlib object, using 2D SPH Interpolation
    of the target variable.
    :param data: The SarracenDataFrame to render. [Required]
    :param target: The variable to interpolate over. [Required]
    :param x: The positional x variable.
    :param y: The positional y variable.
    :param kernel: The smoothing kernel to use for interpolation.
    :param xmin: The minimum bound in the x-direction.
    :param ymin: The minimum bound in the y-direction.
    :param xmax: The maximum bound in the x-direction.
    :param ymax: The maximum bound in the y-direction.
    :param pixcountx: The number of pixels in the x-direction.
    :param pixcounty: The number of pixels in the y-direction.
    :return: The completed plot.
    """
    # x & y columns default to the variables determined by the SarracenDataFrame.
    if x is None:
        x = data.xcol
    if y is None:
        y = data.ycol

    # snap the bounds of the plot to the nearest integer.
    if xmin is None:
        xmin = snap(data.loc[:, x].min())
    if ymin is None:
        ymin = snap(data.loc[:, y].min())
    if xmax is None:
        xmax = snap(data.loc[:, x].max())
    if ymax is None:
        ymax = snap(data.loc[:, y].max())
    # set pixcounty to maintain an aspect ratio that is the same as the underlying bounds of the data.
    if pixcounty is None:
        pixcounty = int(np.rint(pixcountx * ((ymax - ymin) / (xmax - xmin))))

    pixwidthx = (xmax - xmin) / pixcountx
    pixwidthy = (ymax - ymin) / pixcounty
    image = interpolate2D(data, x, y, target, kernel, pixwidthx, pixwidthy, xmin, ymin, pixcountx, pixcounty)

    # this figsize approximation seems to work well enough in most cases
    fig, ax = plt.subplots(figsize=(4, 3 * ((ymax - ymin) / (xmax - xmin))))
    img = ax.imshow(image, cmap='RdBu', origin='lower', extent=[xmin, xmax, ymin, ymax])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    cbar = fig.colorbar(img, ax=ax)
    cbar.ax.set_ylabel(target)

    return fig, ax
