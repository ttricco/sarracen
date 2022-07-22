"""
Provides several rendering functions which produce matplotlib plots of SPH data.
These functions act as interfaces to interpolation functions within interpolate.py.

These functions can be accessed directly, for example:
    render_2d(data, target)
Or, they can be accessed through a `SarracenDataFrame` object, for example:
    data.render_2d(target)
"""

from typing import Union, Tuple

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, LogNorm

from sarracen.interpolate import interpolate_2d_cross, interpolate_2d, interpolate_3d, interpolate_3d_cross, \
    interpolate_3d_vec, interpolate_3d_cross_vec, interpolate_2d_vec
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


def _default_axes(data, x, y):
    """Utility function to determine the x & y columns to use for rendering.

    Parameters
    ----------
    data: SarracenDataFrame
        The particle dataset to render.
    x, y: str
        The x and y directional column labels passed to the render function.

    Returns
    -------
    x, y: str
        The directional column labels to use for rendering. Defaults to the x-column detected in `data`
    """
    if x is None:
        x = data.xcol
    if y is None:
        y = data.ycol

    return x, y


def _default_bounds(data, x, y, x_min, x_max, y_min, y_max):
    """Utility function to determine the 2-dimensional boundaries to use in 2D rendering.

    Parameters
    ----------
    data: SarracenDataFrame
        The particle dataset to render.
    x, y: str
        The directional column labels that will be used for rendering.
    x_min, x_max, y_min, y_max: float
        The minimum and maximum values passed to the render function, in particle data space.

    Returns
    -------
    x_min, x_max, y_min, y_max: float
        The minimum and maximum values to use for rendering, in particle data space. Defaults
        to the maximum and minimum values of `x` and `y`, snapped to the nearest integer.
    """
    if x_min is None:
        x_min = _snap(data.loc[:, x].min())
    if y_min is None:
        y_min = _snap(data.loc[:, y].min())
    if x_max is None:
        x_max = _snap(data.loc[:, x].max())
    if y_max is None:
        y_max = _snap(data.loc[:, y].max())

    return x_min, x_max, y_min, y_max


def _set_pixels(x_pixels, y_pixels, x_min, x_max, y_min, y_max, default):
    """Utility function to determine the number of pixels to interpolate over in 2D interpolation.
    Parameters
    ----------
    x_pixels, y_pixels: int
        The number of pixels in the x & y directions passed to the interpolation function.
    x_min, x_max, y_min, y_max: float
        The minimum and maximum values to use in interpolation, in particle data space.
    Returns
    -------
    x_pixels, y_pixels
        The number of pixels in the x & y directions to use in 2D interpolation.
    """
    # set # of pixels to maintain an aspect ratio that is the same as the underlying bounds of the data.
    if x_pixels is None and y_pixels is None:
        x_pixels = default
    if x_pixels is None:
        x_pixels = int(np.rint(y_pixels * ((x_max - x_min) / (y_max - y_min))))
    if y_pixels is None:
        y_pixels = int(np.rint(x_pixels * ((y_max - y_min) / (x_max - x_min))))

    return x_pixels, y_pixels


def render_2d(data: 'SarracenDataFrame', target: str, x: str = None, y: str = None, kernel: BaseKernel = None,
              x_pixels: int = None, y_pixels: int = None, x_min: float = None, x_max: float = None, y_min: float = None,
              y_max: float = None, cmap: Union[str, Colormap] = 'RdBu', cbar: bool = True, cbar_kws: dict = {},
              cbar_ax: Axes = None, ax: Axes = None, exact: bool = None, backend: str = None, log_scale: bool = False,
              **kwargs) -> Axes:
    """ Render 2D particle data to a 2D grid, using SPH rendering of a target variable.

    Render the data within a SarracenDataFrame to a 2D matplotlib object, by rendering the values
    of a target variable. The contributions of all particles near the rendered area are summed and
    stored to a 2D grid.

    Parameters
    ----------
    data : SarracenDataFrame
        Particle data, in a SarracenDataFrame.
    target: str
        Column label of the target smoothing data.
    x, y: str, optional
        Column label of the x & y directional axes. Defaults to the columns detected in `data`.
    kernel: BaseKernel, optional
        Kernel to use for smoothing the target data. Defaults to the kernel specified in `data`.
    x_pixels, y_pixels: int, optional
        Number of pixels in the output image in the x & y directions. Default values are chosen to keep
        a consistent aspect ratio.
    x_min, x_max, y_min, y_max: float, optional
        The minimum and maximum values to use in interpolation, in particle data space. Defaults
        to the minimum and maximum values of `x` and `y`.
    cmap: str or Colormap, optional
        The color map to use when plotting this data.
    cbar: bool, optional
        True if a colorbar should be drawn.
    cbar_kws: dict, optional
        Keyword arguments to pass to matplotlib.figure.Figure.colorbar()
    cbar_ax: Axes
        Axes to draw the colorbar in, if not provided then space will be taken from the main Axes.
    ax: Axes
        The main axes in which to draw the rendered image.
    exact: bool
        Whether to use exact interpolation of the data. Defaults to False.
    backend: ['cpu', 'gpu']
        The computation backend to use when rendering this data. Defaults to the backend specified in `data`.
    log_scale: bool
        Whether to use a logarithmic scale for color coding.
    kwargs: other keyword arguments
        Keyword arguments to pass to matplotlib.axes.Axes.imshow().

    Returns
    -------
    Axes
        The resulting matplotlib axes, which contain the 2d rendered image.

    Raises
    -------
    ValueError
        If `x_pixels` or `y_pixels` are less than or equal to zero, or
        if the specified `x` and `y` minimum and maximums result in an invalid region.
    KeyError
        If `target`, `x`, `y`, mass, density, or smoothing length columns do not
        exist in `data`.
    """
    image = interpolate_2d(data, target, x, y, kernel, x_pixels, y_pixels, x_min, x_max, y_min, y_max, exact, backend)

    if ax is None:
        ax = plt.gca()

    x, y = _default_axes(data, x, y)
    x_min, x_max, y_min, y_max = _default_bounds(data, x, y, x_min, x_max, y_min, y_max)

    kwargs.setdefault("origin", 'lower')
    kwargs.setdefault("extent", [x_min, x_max, y_min, y_max])
    if log_scale:
        kwargs.setdefault("norm", LogNorm(clip=True))

    graphic = ax.imshow(image, cmap=cmap, **kwargs)
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    if cbar:
        colorbar = ax.figure.colorbar(graphic, cbar_ax, ax, **cbar_kws)
        label = target
        if log_scale:
            label = f"log ({label})"
        colorbar.ax.set_ylabel(label)

    return ax


def streamlines(data: 'SarracenDataFrame', target: Union[Tuple[str, str], Tuple[str, str, str]], z_slice: int = None,
                x: str = None, y: str = None, z: str = None, kernel: BaseKernel = None, integral_samples: int = 1000,
                rotation: np.ndarray = None, origin: np.ndarray = None, x_pixels: int = None, y_pixels: int = None,
                x_min: float = None, x_max: float = None, y_min: float = None, y_max: float = None, ax: Axes = None,
                exact: bool = None, backend: str = None, **kwargs) -> Axes:
    """ Create an SPH interpolated streamline plot of a target vector.

    Render the data within a SarracenDataFrame to a 2D matplotlib object, by rendering the values
    of a target vector. The contributions of all particles near the rendered area are summed and
    stored to a 2D grid for the x & y axes of the target vector. This data is then used to create
    a streamline plot using ax.streamlines().

    Parameters
    ----------
    data : SarracenDataFrame
        Particle data, in a SarracenDataFrame.
    target: str tuple of shape (2) or (3).
        Column label of the target vector. Shape must match the # of dimensions in `data`.
    z_slice: float
        The z to take a cross-section at. If none, column interpolation is performed.
    x, y, z: str, optional
        Column label of the x, y & z directional axes. Defaults to the columns detected in `data`.
    kernel: BaseKernel, optional
        Kernel to use for smoothing the target data. Defaults to the kernel specified in `data`.
    integral_samples: int, optional
        If using column interpolation, the number of sample points to take when approximating the 2D column kernel.
    rotation: array_like or Rotation, optional
        The rotation to apply to the data before interpolation. If defined as an array, the
        order of rotations is [z, y, x] in degrees.
    origin: array_like, optional
        Point of rotation of the data, in [x, y, z] form. Defaults to the centre
        point of the bounds of the data.
    x_pixels, y_pixels: int, optional
        Number of interpolation samples to pass to ax.streamlines(). Default values are chosen to keep
        a consistent aspect ratio.
    x_min, x_max, y_min, y_max: float, optional
        The minimum and maximum values to use in interpolation, in particle data space. Defaults
        to the minimum and maximum values of `x` and `y`.
    ax: Axes
        The main axes in which to draw the rendered image.
    exact: bool
        Whether to use exact interpolation of the data. For cross-sections this is ignored. Defaults to False.
    backend: ['cpu', 'gpu']
        The computation backend to use when rendering this data. Defaults to the backend specified in `data`.
    kwargs: other keyword arguments
        Keyword arguments to pass to ax.streamlines()

    Returns
    -------
    Axes
        The resulting matplotlib axes, which contains the streamline plot.

    Raises
    ------
    ValueError
        If `x_pixels` or `y_pixels` are less than or equal to zero, or
        if the specified `x` and `y` minimum and maximums result in an invalid region, or
        if the number of dimensions in the target vector does not match the data, or
        if `data` is not 2 or 3 dimensional.
    KeyError
        If `target`, `x`, `y`, `z` (for 3-dimensional data), mass, density, or smoothing length columns do not
        exist in `data`.
    """
    # Choose between the various interpolation functions available, based on initial data passed to this function.
    if data.get_dim() == 2:
        if not len(target) == 2:
            raise ValueError('Target vector is not 2-dimensional.')
        img = interpolate_2d_vec(data, target[0], target[1], x, y, kernel, x_pixels, y_pixels, x_min, x_max, y_min,
                                 y_max, exact, backend)
    elif data.get_dim() == 3:
        if not len(target) == 3:
            raise ValueError('Target vector is not 3-dimensional.')
        if z_slice is None:
            img = interpolate_3d_vec(data, target[0], target[1], target[2], x, y, kernel, integral_samples, rotation,
                                     origin, x_pixels, y_pixels, x_min, x_max, y_min, y_max, exact, backend)
        else:
            img = interpolate_3d_cross_vec(data, target[0], target[1], target[2], z_slice, x, y, z, kernel, rotation,
                                           origin, x_pixels, y_pixels, x_min, x_max, y_min, y_max, backend)
    else:
        raise ValueError('`data` is not a valid number of dimensions.')

    if ax is None:
        ax = plt.gca()

    x, y = _default_axes(data, x, y)
    x_min, x_max, y_min, y_max = _default_bounds(data, x, y, x_min, x_max, y_min, y_max)

    kwargs.setdefault("color", 'black')
    ax.streamplot(np.linspace(x_min, x_max, np.size(img[0], 1)), np.linspace(y_min, y_max, np.size(img[0], 0)),
                  img[0], img[1], **kwargs)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # remove the x & y ticks if the data is rotated, since these no longer have physical
    # relevance to the displayed data.
    if rotation is not None:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_xlabel(x)
        ax.set_ylabel(y)

    return ax


def arrowplot(data: 'SarracenDataFrame', target: Union[Tuple[str, str], Tuple[str, str, str]], z_slice: int = None,
              x: str = None, y: str = None, z: str = None, kernel: BaseKernel = None, integral_samples: int = 1000,
              rotation: np.ndarray = None, origin: np.ndarray = None, x_arrows: int = None, y_arrows: int = None,
              x_min: float = None, x_max: float = None, y_min: float = None, y_max: float = None, ax: Axes = None,
              exact: bool = None, backend: str = None, **kwargs) -> Axes:
    """ Create an SPH interpolated vector field plot of a target vector.

    Render the data within a SarracenDataFrame to a 2D matplotlib object, by rendering the values
    of a target vector. The contributions of all particles near the rendered area are summed and
    stored to a 2D grid for the x & y axes of the target vector. This data is then used to create
    an arrow plot using ax.quiver().

    Parameters
    ----------
    data : SarracenDataFrame
        Particle data, in a SarracenDataFrame.
    target: str tuple of shape (2) or (3).
        Column label of the target vector. Shape must match the # of dimensions in `data`.
    z_slice: float
        The z to take a cross-section at. If none, column interpolation is performed.
    x, y, z: str, optional
        Column label of the x, y & z directional axes. Defaults to the columns detected in `data`.
    kernel: BaseKernel, optional
        Kernel to use for smoothing the target data. Defaults to the kernel specified in `data`.
    integral_samples: int, optional
        If using column interpolation, the number of sample points to take when approximating the 2D column kernel.
    rotation: array_like or Rotation, optional
        The rotation to apply to the data before interpolation. If defined as an array, the order of rotations
        is [z, y, x] in degrees.
    origin: array_like, optional
        Point of rotation of the data, in [x, y, z] form. Defaults to the centre point of the bounds of the data.
    x_arrows, y_arrows: int, optional
        Number of arrows in the output image in the x & y directions. Default values are chosen to keep
        a consistent aspect ratio.
    x_min, x_max, y_min, y_max: float, optional
        The minimum and maximum values to use in interpolation, in particle data space. Defaults
        to the minimum and maximum values of `x` and `y`.
    ax: Axes
        The main axes in which to draw the rendered image.
    exact: bool
        Whether to use exact interpolation of the data. For cross-sections this is ignored. Defaults to False.
    backend: ['cpu', 'gpu']
        The computation backend to use when rendering this data. Defaults to the backend specified in `data`.
    kwargs: other keyword arguments
        Keyword arguments to pass to ax.quiver()

    Returns
    -------
    Axes
        The resulting matplotlib axes, which contains the arrow plot.

    Raises
    ------
    ValueError
        If `x_arrows` or `y_arrows` are less than or equal to zero, or
        if the specified `x` and `y` minimum and maximums result in an invalid region, or
        if the number of dimensions in the target vector does not match the data, or
        if `data` is not 2 or 3 dimensional.
    KeyError
        If `target`, `x`, `y`, `z` (for 3-dimensional data), mass, density, or smoothing length columns do not
        exist in `data`.
    """
    x, y = _default_axes(data, x, y)
    x_min, x_max, y_min, y_max = _default_bounds(data, x, y, x_min, x_max, y_min, y_max)
    x_arrows, y_arrows = _set_pixels(x_arrows, y_arrows, x_min, x_max, y_min, y_max, 20)

    if data.get_dim() == 2:
        if not len(target) == 2:
            raise ValueError('Target vector is not 2-dimensional.')
        img = interpolate_2d_vec(data, target[0], target[1], x, y, kernel, x_arrows, y_arrows, x_min, x_max, y_min,
                                 y_max, exact, backend)
    elif data.get_dim() == 3:
        if not len(target) == 3:
            raise ValueError('Target vector is not 3-dimensional.')
        if z_slice is None:
            img = interpolate_3d_vec(data, target[0], target[1], target[2], x, y, kernel, integral_samples, rotation,
                                     origin, x_arrows, y_arrows, x_min, x_max, y_min, y_max, exact, backend)
        else:
            if exact:
                raise UserWarning("Exact interpolation is not supported for 3D cross-sections.")

            img = interpolate_3d_cross_vec(data, target[0], target[1], target[2], z_slice, x, y, z, kernel, rotation,
                                           origin, x_arrows, y_arrows, x_min, x_max, y_min, y_max, backend)
    else:
        raise ValueError('`data` is not a valid number of dimensions.')

    if ax is None:
        ax = plt.gca()

    x, y = _default_axes(data, x, y)
    x_min, x_max, y_min, y_max = _default_bounds(data, x, y, x_min, x_max, y_min, y_max)

    kwargs.setdefault("angles", 'uv')
    kwargs.setdefault("pivot", 'mid')
    ax.quiver(np.linspace(x_min, x_max, np.size(img[0], 1)), np.linspace(y_min, y_max, np.size(img[0], 0)), img[0],
              img[1], **kwargs)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # remove the x & y ticks if the data is rotated, since these no longer have physical
    # relevance to the displayed data.
    if rotation is not None:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_xlabel(x)
        ax.set_ylabel(y)

    return ax


def render_2d_cross(data: 'SarracenDataFrame', target: str, x: str = None, y: str = None, kernel: BaseKernel = None,
                    pixels: int = 512, x1: float = None, x2: float = None, y1: float = None, y2: float = None,
                    ax: Axes = None, backend: str = None, log_scale: bool = False, **kwargs) -> Axes:
    """ Render 2D particle data to a 1D line, using a 2D cross-section.

    Render the data within a SarracenDataFrame to a seaborn-generated line plot, by taking
    a 2D->1D cross section of a target variable. The contributions of all particles near the
    cross-section line are summed and stored in a 1D array.

    Parameters
    ----------
    data : SarracenDataFrame
        Particle data, in a SarracenDataFrame.
    target: str
        Column label of the target smoothing data.
    x, y: str
        Column labels of the directional axes. Defaults to the x & y columns detected in `data`.
    kernel: BaseKernel, optional
        Kernel to use for smoothing the target data. Defaults to the kernel specified in `data`.
    pixels: int, optional
        Number of points in the resulting line plot in the x-direction.
    x1, x2, y1, y2: float, optional
        Starting and ending coordinates of the cross-section line (in particle data space). Defaults to
        the minimum and maximum values of `x` and `y`.
    ax: Axes
        The main axes in which to draw the rendered image.
    backend: ['cpu', 'gpu']
        The computation backend to use when rendering this data. Defaults to the backend specified in `data`.
    log_scale: bool
        Whether to use a logarithmic scale in the y-axis.
    kwargs: other keyword arguments
        Keyword arguments to pass to seaborn.lineplot().

    Returns
    -------
    Axes
        The resulting matplotlib axes, which contains the 1d cross-sectional plot.

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
    output = interpolate_2d_cross(data, target, x, y, kernel, pixels, x1, x2, y1, y2, backend)

    if ax is None:
        ax = plt.gca()

    x, y = _default_axes(data, x, y)
    x1, x2, y1, y2 = _default_bounds(data, x, y, x1, x2, y1, y2)

    lineplot = sns.lineplot(x=np.linspace(0, np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2), pixels), y=output, ax=ax,
                            **kwargs)

    if log_scale:
        lineplot.set(yscale='log')

    ax.margins(x=0, y=0)
    ax.set_xlabel(f'cross-section ({x}, {y})')

    label = target
    if log_scale:
        label = f"log ({label})"
    ax.set_ylabel(label)

    return ax


def render_3d(data: 'SarracenDataFrame', target: str, x: str = None, y: str = None, kernel: BaseKernel = None,
              integral_samples: int = 1000, rotation: np.ndarray = None, rot_origin: np.ndarray = None,
              x_pixels: int = None, y_pixels: int = None, x_min: float = None, x_max: float = None, y_min: float = None,
              y_max: float = None, cmap: Union[str, Colormap] = 'RdBu', cbar: bool = True, cbar_kws: dict = {},
              cbar_ax: Axes = None, ax: Axes = None, exact: bool = None, backend: str = None, log_scale: bool = False,
              **kwargs) -> Axes:
    """ Render 3D particle data to a 2D grid, using SPH column rendering of a target variable.

    Render the data within a SarracenDataFrame to a 2D matplotlib object, by rendering the values
    of a target variable. The contributions of all particles near columns across the z-axis are
    summed and stored in a to a 2D grid.

    Parameters
    ----------
    data : SarracenDataFrame
        Particle data, in a SarracenDataFrame.
    target: str
        Column label of the target smoothing data.
    x, y: str
        Column labels of the directional axes. Defaults to the x & y columns detected in `data`.
    kernel: BaseKernel, optional
        Kernel to use for smoothing the target data. Defaults to the kernel specified in `data`.
    integral_samples: int, optional
        The number of sample points to take when approximating the 2D column kernel.
    rotation: array_like or Rotation, optional
        The rotation to apply to the data before interpolation. If defined as an array, the
        order of rotations is [z, y, x] in degrees.
    rot_origin: array_like, optional
        Point of rotation of the data, in [x, y, z] form. Defaults to the centre
        point of the bounds of the data.
    x_pixels, y_pixels: int, optional
        Number of pixels in the output image in the x & y directions. Default values are chosen to keep
        a consistent aspect ratio.
    x_min, x_max, y_min, y_max: float, optional
        The minimum and maximum values to render over, in particle data space. Defaults
        to the minimum and maximum values of `x` and `y`.
    cmap: str or Colormap, optional
        The color map to use when plotting this data.
    cbar: bool, optional
        True if a colorbar should be drawn.
    cbar_kws: dict, optional
        Keyword arguments to pass to matplotlib.figure.Figure.colorbar()
    cbar_ax: Axes
        Axes to draw the colorbar in, if not provided then space will be taken from the main Axes.
    ax: Axes
        The main axes in which to draw the rendered image.
    exact: bool
        Whether to use exact interpolation of the data. Defaults to False.
    backend: ['cpu', 'gpu']
        The computation backend to use when rendering this data. Defaults to the backend specified in `data`.
    log_scale: bool
        Whether to use a logarithmic scale for color coding.
    kwargs: other keyword arguments
        Keyword arguments to pass to matplotlib.axes.Axes.imshow().

    Returns
    -------
    Axes
        The resulting matplotlib axes, which contains the 2d rendered image.

    Raises
    -------
    ValueError
        If `x_pixels` or `y_pixels` are less than or equal to zero, or
        if the specified `x` and `y` minimum and maximums result in an invalid region, or
        if the provided data is not 3-dimensional.
    KeyError
        If `target`, `x`, `y`, mass, density, or smoothing length columns do not
        exist in `data`.
    """
    image = interpolate_3d(data, target, x, y, kernel, integral_samples, rotation, rot_origin, x_pixels, y_pixels,
                           x_min, x_max, y_min, y_max, exact, backend)

    if ax is None:
        ax = plt.gca()

    x, y = _default_axes(data, x, y)
    x_min, x_max, y_min, y_max = _default_bounds(data, x, y, x_min, x_max, y_min, y_max)

    kwargs.setdefault("origin", 'lower')
    kwargs.setdefault("extent", [x_min, x_max, y_min, y_max])
    if log_scale:
        kwargs.setdefault("norm", LogNorm(clip=True))

    graphic = ax.imshow(image, cmap=cmap, **kwargs)

    # remove the x & y ticks if the data is rotated, since these no longer have physical
    # relevance to the displayed data.
    if rotation is not None:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_xlabel(x)
        ax.set_ylabel(y)

    if cbar:
        colorbar = ax.figure.colorbar(graphic, cbar_ax, ax, **cbar_kws)
        label = f"column {target}"
        if log_scale:
            label = f"log ({label})"
        colorbar.ax.set_ylabel(label)

    return ax


def render_3d_cross(data: 'SarracenDataFrame', target: str, z_slice: float = None, x: str = None, y: str = None,
                    z: str = None, kernel: BaseKernel = None, rotation: np.ndarray = None,
                    rot_origin: np.ndarray = None, x_pixels: int = None, y_pixels: int = None, x_min: float = None,
                    x_max: float = None, y_min: float = None, y_max: float = None, cmap: Union[str, Colormap] = 'RdBu',
                    cbar: bool = True, cbar_kws: dict = {}, cbar_ax: Axes = None, ax: Axes = None, backend: str = None,
                    log_scale: bool = False, **kwargs) -> Axes:
    """ Render 3D particle data to a 2D grid, using a 3D cross-section.

    Render the data within a SarracenDataFrame to a 2D matplotlib object, using a 3D -> 2D
    cross-section of the target variable. The cross-section is taken of the 3D data at a specific
    value of z, and the contributions of particles near the plane are interpolated to a 2D grid.

    Parameters
    ----------
    data : SarracenDataFrame
        The particle data to render.
    target: str
        The column label of the target smoothing data.
    z_slice: float
        The z-axis value to take the cross-section at. Defaults to the midpoint of the z-directional data.
    x, y, z: str
        The column labels of the directional data to render. Defaults to the x, y, and z columns
        detected in `data`.
    kernel: BaseKernel
        The kernel to use for smoothing the target data. Defaults to the kernel specified in `data`.
    rotation: array_like or Rotation, optional
        The rotation to apply to the data before rendering. If defined as an array, the
        order of rotations is [z, y, x] in degrees.
    rot_origin: array_like, optional
        Point of rotation of the data, in [x, y, z] form. Defaults to the centre
        point of the bounds of the data.
    x_pixels, y_pixels: int, optional
        Number of pixels in the output image in the x & y directions. Default values are chosen to keep
        a consistent aspect ratio.
    x_min, x_max, y_min, y_max: float, optional
        The minimum and maximum values to render over, in particle data space. Defaults
        to the minimum and maximum values of `x` and `y`.
    cmap: str or Colormap, optional
        The color map to use when plotting this data.
    cbar: bool, optional
        True if a colorbar should be drawn.
    cbar_kws: dict, optional
        Keyword arguments to pass to matplotlib.figure.Figure.colorbar()
    cbar_ax: Axes
        Axes to draw the colorbar in, if not provided then space will be taken from the main Axes.
    ax: Axes
        The main axes in which to draw the rendered image.
    backend: ['cpu', 'gpu']
        The computation backend to use when rendering this data. Defaults to the backend specified in `data`.
    log_scale: bool
        Whether to use a logarithmic scale for color coding.
    kwargs: other keyword arguments
        Keyword arguments to pass to matplotlib.axes.Axes.imshow().

    Returns
    -------
    Axes
        The resulting matplotlib axes object, containing the 3d cross-section image.

    Raises
    -------
    ValueError
        If `x_pixels` or `y_pixels` are less than or equal to zero, or
        if the specified `x` and `y` minimum and maximums result in an invalid region, or
        if the provided data is not 3-dimensional.
    KeyError
        If `target`, `x`, `y`, `z`, mass, density, or smoothing length columns do not
        exist in `data`.
    """
    image = interpolate_3d_cross(data, target, z_slice, x, y, z, kernel, rotation, rot_origin, x_pixels, y_pixels, x_min,
                                 x_max, y_min, y_max, backend)

    if ax is None:
        ax = plt.gca()

    x, y = _default_axes(data, x, y)
    x_min, x_max, y_min, y_max = _default_bounds(data, x, y, x_min, x_max, y_min, y_max)

    kwargs.setdefault("origin", 'lower')
    kwargs.setdefault("extent", [x_min, x_max, y_min, y_max])
    if log_scale:
        kwargs.setdefault("norm", LogNorm(clip=True))

    graphic = ax.imshow(image, cmap=cmap, **kwargs)

    # remove the x & y ticks if the data is rotated, since these no longer have physical
    # relevance to the displayed data.
    if rotation is not None:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_xlabel(x)
        ax.set_ylabel(y)

    if cbar:
        colorbar = ax.figure.colorbar(graphic, cbar_ax, ax, **cbar_kws)
        label = target
        if log_scale:
            label = f"log ({label})"
        colorbar.ax.set_ylabel(label)

    return ax
