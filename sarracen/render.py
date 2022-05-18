from typing import Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
import seaborn as sns

from sarracen.interpolate import interpolate2DCross, interpolate1DCross
from sarracen.kernels import BaseKernel, CubicSplineKernel


def render_2d(data: 'SarracenDataFrame',
              target: str,
              x: str = None,
              y: str = None,
              kernel: BaseKernel = CubicSplineKernel(),
              xmin: float = None,
              ymin: float = None,
              xmax: float = None,
              ymax: float = None,
              pixcountx: int = 256,
              pixcounty: int = None,
              cmap: Union[str, Colormap] = 'RdBu') -> ('Figure', 'Axes'):
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
    :param cmap: The color map to use for plotting this data.
    :return: The completed plot.
    """
    # x & y columns default to the variables determined by the SarracenDataFrame.
    if x is None:
        x = data.xcol
    if y is None:
        y = data.ycol

    # plot bounds default to variable determined in SarracenDataFrame
    if xmin is None:
        xmin = data.xmin
    if ymin is None:
        ymin = data.ymin
    if xmax is None:
        xmax = data.xmax
    if ymax is None:
        ymax = data.ymax

    # set pixcounty to maintain an aspect ratio that is the same as the underlying bounds of the data.
    if pixcounty is None:
        pixcounty = int(np.rint(pixcountx * ((ymax - ymin) / (xmax - xmin))))

    pixwidthx = (xmax - xmin) / pixcountx
    pixwidthy = (ymax - ymin) / pixcounty
    image = interpolate2DCross(data, x, y, target, kernel, pixwidthx, pixwidthy, xmin, ymin, pixcountx, pixcounty)

    # ensure the plot size maintains the aspect ratio of the underlying bounds of the data
    fig, ax = plt.subplots(figsize=(6.4, 4.8 * ((ymax - ymin) / (xmax - xmin))))
    img = ax.imshow(image, cmap=cmap, origin='lower', extent=[xmin, xmax, ymin, ymax])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    cbar = fig.colorbar(img, ax=ax)
    cbar.ax.set_ylabel(target)

    return fig, ax


def render_1d_cross(data: 'SarracenDataFrame',
                    target: str,
                    x: str = None,
                    y: str = None,
                    kernel: BaseKernel = CubicSplineKernel(),
                    x1: float = None,
                    y1: float = None,
                    x2: float = None,
                    y2: float = None,
                    pixcount: int = 256) -> ('Figure', 'Axes'):
    """
    Render the data within a SarracenDataFrame to a 1D matplotlib object, by taking a 1D SPH
    cross-section of the target variable along a given line.
    :param data: The SarracenDataFrame to render. [Required]
    :param target: The variable to interpolate over. [Required]
    :param x: The positional x variable.
    :param y: The positional y variable.
    :param kernel: The kernel to use for smoothing the target data.
    :param x1: The starting x-coordinate of the cross-section line. (in particle data space)
    :param y1: The starting y-coordinate of the cross-section line. (in particle data space)
    :param x2: The ending x-coordinate of the cross-section line. (in particle data space)
    :param y2: The ending y-coordinate of the cross-section line. (in particle data space)
    :param pixcount: The number of pixels in the output over the entire cross-sectional line.
    :return: The completed plot.
    """
    # x & y columns default to the variables determined by the SarracenDataFrame.
    if x is None:
        x = data.xcol
    if y is None:
        y = data.ycol

    # plot bounds default to variable determined in SarracenDataFrame
    if x1 is None:
        x1 = data.xmin
    if y1 is None:
        y1 = data.ymin
    if x2 is None:
        x2 = data.xmax
    if y2 is None:
        y2 = data.ymax

    output = interpolate1DCross(data, x, y, target, kernel, x1, y1, x2, y2, pixcount)

    fig, ax = plt.subplots()
    ax.margins(x=0, y=0)
    sns.lineplot(x=np.linspace(0, np.sqrt((x2 - x1) ** 2+ (y2 - y1) ** 2), pixcount), y=output, ax=ax)
    ax.set_xlabel(f'cross-section ({x}, {y})')
    ax.set_ylabel(target)

    return fig, ax
