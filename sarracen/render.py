from typing import Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
import seaborn as sns

from sarracen.interpolate import interpolate_2d_cross, interpolate_2d, interpolate_3d, interpolate_3d_cross
from sarracen.kernels import BaseKernel, CubicSplineKernel


def render_2d(data: 'SarracenDataFrame',
              target: str,
              x: str = None,
              y: str = None,
              kernel: BaseKernel = CubicSplineKernel(),
              x_pixels: int = None,
              y_pixels: int = None,
              x_min: float = None,
              y_min: float = None,
              x_max: float = None,
              y_max: float = None,
              colormap: Union[str, Colormap] = 'RdBu') -> ('Figure', 'Axes'):
    """
    Render the data within a SarracenDataFrame to a 2D matplotlib object, using 2D SPH Interpolation
    of the target variable.
    :param data: The SarracenDataFrame to render. [Required]
    :param target: The variable to interpolate over. [Required]
    :param x: The positional x variable.
    :param y: The positional y variable.
    :param kernel: The smoothing kernel to use for interpolation.
    :param x_min: The minimum bound in the x-direction.
    :param y_min: The minimum bound in the y-direction.
    :param x_max: The maximum bound in the x-direction.
    :param y_max: The maximum bound in the y-direction.
    :param x_pixels: The number of pixels in the x-direction.
    :param y_pixels: The number of pixels in the y-direction.
    :param colormap: The color map to use for plotting this data.
    :return: The completed plot.
    """
    # x & y columns default to the variables determined by the SarracenDataFrame.
    if x is None:
        x = data.xcol
    if y is None:
        y = data.ycol

    # plot bounds default to variable determined in SarracenDataFrame
    if x_min is None:
        x_min = data.xmin
    if y_min is None:
        y_min = data.ymin
    if x_max is None:
        x_max = data.xmax
    if y_max is None:
        y_max = data.ymax

    # set # of pixels to maintain an aspect ratio that is the same as the underlying bounds of the data.
    if x_pixels is None and y_pixels is None:
        x_pixels = 512
    if x_pixels is None:
        x_pixels = int(np.rint(y_pixels * ((x_max - x_min) / (y_max - y_min))))
    if y_pixels is None:
        y_pixels = int(np.rint(x_pixels * ((y_max - y_min) / (x_max - x_min))))

    image = interpolate_2d(data, target, x, y, kernel, x_pixels, y_pixels, x_min, x_max, y_min, y_max)

    # ensure the plot size maintains the aspect ratio of the underlying bounds of the data
    fig, ax = plt.subplots(figsize=(6.4, 4.8 * ((y_max - y_min) / (x_max - x_min))))
    img = ax.imshow(image, cmap=colormap, origin='lower', extent=[x_min, x_max, y_min, y_max])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    cbar = fig.colorbar(img, ax=ax)
    cbar.ax.set_ylabel(target)

    return fig, ax


def render_2d_cross(data: 'SarracenDataFrame',
                    target: str,
                    x: str = None,
                    y: str = None,
                    kernel: BaseKernel = CubicSplineKernel(),
                    pixels: int = 512,
                    x1: float = None,
                    x2: float = None,
                    y1: float = None,
                    y2: float = None) -> ('Figure', 'Axes'):
    """
    Render two-dimensional data inside a SarracenDataFrame to a 1D matplotlib object, by taking
    a 1D SPH cross-section of the target variable along a given line.
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

    output = interpolate_2d_cross(data, target, x, y, kernel, pixels, x1, x2, y1, y2)

    fig, ax = plt.subplots()
    ax.margins(x=0, y=0)
    sns.lineplot(x=np.linspace(0, np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2), pixels), y=output, ax=ax)
    ax.set_xlabel(f'cross-section ({x}, {y})')
    ax.set_ylabel(target)

    return fig, ax


def render_3d(data: 'SarracenDataFrame',
              target: str,
              x: str = None,
              y: str = None,
              kernel: BaseKernel = CubicSplineKernel(),
              integral_samples: int = 1000,
              x_pixels: int = None,
              y_pixels: int = None,
              x_min: float = None,
              y_min: float = None,
              x_max: float = None,
              y_max: float = None,
              colormap: Union[str, Colormap] = 'RdBu') -> ('Figure', 'Axes'):
    """
    Render the data within a SarracenDataFrame to a 2D matplotlib object, using 3D -> 2D column interpolation of the
    target variable.
    :param data: The SarracenDataFrame to render. [Required]
    :param target: The variable to interpolate over. [Required]
    :param x: The positional x variable.
    :param y: The positional y variable.
    :param kernel: The smoothing kernel to use for interpolation.
    :param x_min: The minimum bound in the x-direction.
    :param y_min: The minimum bound in the y-direction.
    :param x_max: The maximum bound in the x-direction.
    :param y_max: The maximum bound in the y-direction.
    :param x_pixels: The number of pixels in the x-direction.
    :param y_pixels: The number of pixels in the y-direction.
    :param colormap: The color map to use for plotting this data.
    :param integral_samples: The number of samples to use when approximating the kernel column integral.
    :return: The completed plot.
    """
    # x & y columns default to the variables determined by the SarracenDataFrame.
    if x is None:
        x = data.xcol
    if y is None:
        y = data.ycol

    # plot bounds default to variables determined in SarracenDataFrame
    if x_min is None:
        x_min = data.xmin
    if y_min is None:
        y_min = data.ymin
    if x_max is None:
        x_max = data.xmax
    if y_max is None:
        y_max = data.ymax

    # set # of pixels to maintain an aspect ratio that is the same as the underlying bounds of the data.
    if x_pixels is None and y_pixels is None:
        x_pixels = 512
    if x_pixels is None:
        x_pixels = int(np.rint(y_pixels * ((x_max - x_min) / (y_max - y_min))))
    if y_pixels is None:
        y_pixels = int(np.rint(x_pixels * ((y_max - y_min) / (x_max - x_min))))

    img = interpolate_3d(data, target, x, y, kernel, integral_samples, x_pixels, y_pixels, x_min, x_max, y_min, y_max)

    # ensure the plot size maintains the aspect ratio of the underlying bounds of the data
    fig, ax = plt.subplots(figsize=(6.4, 4.8 * ((y_max - y_min) / (x_max - x_min))))
    graphic = ax.imshow(img, cmap=colormap, origin='lower', extent=[x_min, x_max, y_min, y_max])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    cbar = fig.colorbar(graphic, ax=ax)
    cbar.ax.set_ylabel(f"column {target}")

    return fig, ax


def render_3d_cross(data: 'SarracenDataFrame',
                    target: str,
                    z_slice: float = None,
                    x: str = None,
                    y: str = None,
                    z: str = None,
                    kernel: BaseKernel = CubicSplineKernel(),
                    x_pixels: int = None,
                    y_pixels: int = None,
                    x_min: float = None,
                    y_min: float = None,
                    x_max: float = None,
                    y_max: float = None,
                    colormap: Union[str, Colormap] = 'RdBu') -> tuple['Figure', 'Axes']:
    """ Render 3D particle data to a 2D grid, using a 3D cross-section.

    Render the data within a SarracenDataFrame to a 2D matplotlib object, using a 3D -> 2D
    cross-section of the target variable. The cross-section is taken of the 3D data at a specific
    value of z, and the contributions of particles near the plane are interpolated to a 2D grid.

    Parameters
    ----------
    data : SarracenDataFrame
        The particle data, in a SarracenDataFrame.
    target: str
        The column label of the target smoothing data.
    z_slice: float
        The z-axis value to take the cross-section at.
    x: str
        The column label of the x-directional axis.
    y: str
        The column label of the y-directional axis.
    z: str
        The column label of the z-directional axis.
    kernel: BaseKernel
        The kernel to use for smoothing the target data.
    x_min: float, optional
        The minimum bound in the x-direction. (in particle data space)
    y_min: float, optional
        The minimum bound in the y-direction. (in particle data space)
    x_max: float, optional
        The maximum bound in the x-direction. (in particle data space)
    y_max: float, optional
        The maximum bound in the y-direction. (in particle data space)
    pixcountx: int, optional
        The number of pixels in the output image in the x-direction.
    pixcounty: int, optional
        The number of pixels in the output image in the y-direction.
    colormap: str or Colormap, optional
        The color map to use when plotting this data.

    Returns
    -------
    Figure
        The resulting matplotlib figure, containing the 3d-cross section and
        a color bar indicating the magnitude of the target variable.
    Axes
        The resulting matplotlib axes, which contains the 3d-cross section image.

    Raises
    -------
    ValueError
        If `pixwidthx`, `pixwidthy`, `pixcountx`, or `pixcounty` are less than or equal to zero.
    """
    # x & y columns default to the variables determined by the SarracenDataFrame.
    if x is None:
        x = data.xcol
    if y is None:
        y = data.ycol
    if z is None:
        z = data.zcol

    # plot bounds default to variables determined in SarracenDataFrame
    if x_min is None:
        x_min = data.xmin
    if y_min is None:
        y_min = data.ymin
    if x_max is None:
        x_max = data.xmax
    if y_max is None:
        y_max = data.ymax

    # set default slice to be through the middle of the data's z-axis.
    if z_slice is None:
        z_slice = (data.zmin + data.zmax) / 2

    # set # of pixels to maintain an aspect ratio that is the same as the underlying bounds of the data.
    if x_pixels is None and y_pixels is None:
        x_pixels = 512
    if x_pixels is None:
        x_pixels = int(np.rint(y_pixels * ((x_max - x_min) / (y_max - y_min))))
    if y_pixels is None:
        y_pixels = int(np.rint(x_pixels * ((y_max - y_min) / (x_max - x_min))))

    img = interpolate_3d_cross(data, target, z_slice, x, y, z, kernel, x_pixels, y_pixels, x_min, x_max, y_min, y_max)

    # ensure the plot size maintains the aspect ratio of the underlying bounds of the data
    fig, ax = plt.subplots(figsize=(6.4, 4.8 * ((y_max - y_min) / (x_max - x_min))))
    graphic = ax.imshow(img, cmap=colormap, origin='lower', extent=[x_min, x_max, y_min, y_max])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    cbar = fig.colorbar(graphic, ax=ax)
    cbar.ax.set_ylabel(f"{target}")

    return fig, ax
