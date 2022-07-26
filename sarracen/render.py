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


def _default_bounds(data, x, y, xlim, ylim) -> tuple[tuple[float, float], tuple[float, float]]:
    """Utility function to determine the 2-dimensional boundaries to use in 2D rendering.

    Parameters
    ----------
    data: SarracenDataFrame
        The particle dataset to render.
    x, y: str
        The directional column labels that will be used for rendering.
    xlim, ylim: float
        The minimum and maximum values passed to the render function, in particle data space.

    Returns
    -------
    xlim, ylim: tuple of float
        The minimum and maximum values to use for rendering, in particle data space. Defaults
        to the maximum and minimum values of `x` and `y`, snapped to the nearest integer.
    """
    # boundaries of the plot default to the maximum & minimum values of the data.
    x_min = xlim[0] if xlim is not None and xlim[0] is not None else None
    y_min = ylim[0] if ylim is not None and ylim[0] is not None else None
    x_max = xlim[1] if xlim is not None and xlim[1] is not None else None
    y_max = ylim[1] if ylim is not None and ylim[1] is not None else None

    x_min = _snap(data.loc[:, x].min()) if x_min is None else x_min
    y_min = _snap(data.loc[:, y].min()) if y_min is None else y_min
    x_max = _snap(data.loc[:, x].max()) if x_max is None else x_max
    y_max = _snap(data.loc[:, y].max()) if y_max is None else y_max

    return (x_min, x_max), (y_min, y_max)


def _set_pixels(x_pixels, y_pixels, xlim, ylim, default):
    """Utility function to determine the number of pixels to interpolate over in 2D interpolation.
    Parameters
    ----------
    x_pixels, y_pixels: int
        The number of pixels in the x & y directions passed to the interpolation function.
    xlim, ylim: tuple of float
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
        x_pixels = int(np.rint(y_pixels * ((xlim[1] - xlim[0]) / (ylim[1] - ylim[0]))))
    if y_pixels is None:
        y_pixels = int(np.rint(x_pixels * ((ylim[1] - ylim[0]) / (xlim[1] - xlim[0]))))

    return x_pixels, y_pixels


def render(data: 'SarracenDataFrame', target: str, x: str = None, y: str = None, z: str = None,
           xsec: Union[float, bool] = None, kernel: BaseKernel = None, x_pixels: int = None, y_pixels: int = None,
           xlim: tuple[float, float] = None, ylim: tuple[float, float] = None, cmap: Union[str, Colormap] = 'RdBu',
           cbar: bool = True, cbar_kws: dict = {}, cbar_ax: Axes = None, ax: Axes = None, exact: bool = None,
           backend: str = None, integral_samples: int = 1000, rotation: np.ndarray = None,
           rot_origin: np.ndarray = None, log_scale: bool = False, **kwargs) -> Axes:
    """ Render a scalar SPH target variable to a grid or line plot.

    Parameters
    ----------
    data : SarracenDataFrame
        Particle data, in a SarracenDataFrame.
    target: str
        Column label of the target variable.
    xsec: float or bool, optional.
        For 2D datasets, whether to perform a cross-section (no cross-section by default). For a 3D dataset, the z to
        take a cross-section at. If none, column interpolation is performed.
    x, y, z: str, optional
        Column labels of the x, y & z directional axes. Defaults to the columns detected in `data`.
    kernel: BaseKernel, optional
        Kernel to use for smoothing the target data. Defaults to the kernel specified in `data`.
    x_pixels, y_pixels: int, optional
        Number of pixels present in the final image. For 2D cross-sections, `x_pixels` sets the number of samples in
        the final line plot.
    xlim, ylim: tuple of float, optional
        The starting and ending corners of the final 2D image, or the starting and ending points of the cross-sectional
        line.
    cmap: str or Colormap, optional
        The color map to use when plotting a 2D image. Not applicable to 2D cross-sections.
    cbar: bool, optional
        True if a colorbar should be drawn. Not applicable to 2D cross-sections.
    cbar_kws: dict, optional
        Keyword arguments to pass to matplotlib.figure.Figure.colorbar(). Not applicable to 2D cross-sections.
    cbar_ax: Axes
        Axes to draw the colorbar in, if not provided then space will be taken from the main Axes.
        Not applicable to 2D cross-sections.
    ax: Axes
        The main axes in which to draw the rendered image.
    exact: bool
        Whether to use exact interpolation of the data. For cross-sections this is ignored. Defaults to False.
    backend: ['cpu', 'gpu']
        The computation backend to use when rendering this data. Defaults to the backend specified in `data`.
    integral_samples: int, optional
        If using column interpolation, the number of sample points to take when approximating the 2D column kernel.
    rotation: array_like or Rotation, optional
        The rotation to apply to the data before interpolation. If defined as an array, the
        order of rotations is [z, y, x] in degrees. Only applies to 3D datasets.
    rot_origin: array_like, optional
        Point of rotation of the data, in [x, y, z] form. Defaults to the centre
        point of the bounds of the data. Only applies to 3D datasets.
    log_scale: bool
        Whether to use a logarithmic scale for color coding.
    kwargs: other keyword arguments
        Keyword arguments to pass to ax.imshow or sns.lineplot.

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
    if data.get_dim() == 2:
        if xsec is not None:
            image = interpolate_2d_cross(data, target, x, y, kernel, x_pixels, xlim, ylim, backend)
        else:
            image = interpolate_2d(data, target, x, y, kernel, x_pixels, y_pixels, xlim, ylim, exact, backend)
    else:
        if xsec is not None:
            image = interpolate_3d_cross(data, target, x, y, z, None if xsec is True else xsec, kernel, rotation,
                                         rot_origin, x_pixels, y_pixels, xlim, ylim, backend)
        else:
            image = interpolate_3d(data, target, x, y, kernel, integral_samples, rotation, rot_origin, x_pixels,
                                   y_pixels, xlim, ylim, exact, backend)

    if ax is None:
        ax = plt.gca()

    x, y = _default_axes(data, x, y)
    xlim, ylim = _default_bounds(data, x, y, xlim, ylim)

    if image.ndim == 1:
        lineplot = sns.lineplot(x=np.linspace(0, np.sqrt((xlim[1] - xlim[0]) ** 2 + (ylim[1] - ylim[0]) ** 2),
                                              image.size), y=image, ax=ax, **kwargs)

        if log_scale:
            lineplot.set(yscale='log')

        ax.margins(x=0, y=0)
        ax.set_xlabel(f'cross-section ({x}, {y})')

        label = target
        if log_scale:
            label = f"log ({label})"
        ax.set_ylabel(label)
    else:
        kwargs.setdefault("origin", 'lower')
        kwargs.setdefault("extent", [xlim[0], xlim[1], ylim[0], ylim[1]])
        if log_scale:
            kwargs.setdefault("norm", LogNorm(clip=True))

        graphic = ax.imshow(image, cmap=cmap, **kwargs)
        if rotation is not None and data.get_dim() == 3:
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_xlabel(x)
            ax.set_ylabel(y)

        if cbar:
            colorbar = ax.figure.colorbar(graphic, cbar_ax, ax, **cbar_kws)
            label = target
            if data.get_dim() == 3 and xsec is None:
                label = f"column {label}"
            if log_scale:
                label = f"log ({label})"
            colorbar.ax.set_ylabel(label)

    return ax


def streamlines(data: 'SarracenDataFrame', target: Union[Tuple[str, str], Tuple[str, str, str]], x: str = None,
                y: str = None, z: str = None, z_slice: int = None, kernel: BaseKernel = None,
                integral_samples: int = 1000, rotation: np.ndarray = None, rot_origin: np.ndarray = None,
                x_pixels: int = None, y_pixels: int = None, xlim: tuple[float, float] = None,
                ylim: tuple[float, float] = None, ax: Axes = None, exact: bool = None, backend: str = None,
                **kwargs) -> Axes:
    """ Create an SPH interpolated streamline plot of a target vector.

    Render the data within a SarracenDataFrame to a 2D matplotlib object, by rendering the values
    of a target vector. The contributions of all particles near the rendered area are summed and
    stored to a 2D grid for the x & y axes of the target vector. This data is then used to create
    a streamline plot using ax.streamlines().

    Parameters
    ----------
    data: SarracenDataFrame
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
    rot_origin: array_like, optional
        Point of rotation of the data, in [x, y, z] form. Defaults to the centre
        point of the bounds of the data.
    x_pixels, y_pixels: int, optional
        Number of interpolation samples to pass to ax.streamlines(). Default values are chosen to keep
        a consistent aspect ratio.
    xlim, ylim: float, optional
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
        img = interpolate_2d_vec(data, target[0], target[1], x, y, kernel, x_pixels, y_pixels, xlim, ylim, exact,
                                 backend)
    elif data.get_dim() == 3:
        if not len(target) == 3:
            raise ValueError('Target vector is not 3-dimensional.')
        if z_slice is None:
            img = interpolate_3d_vec(data, target[0], target[1], target[2], x, y, kernel, integral_samples, rotation,
                                     rot_origin, x_pixels, y_pixels, xlim, ylim, exact, backend)
        else:
            img = interpolate_3d_cross_vec(data, target[0], target[1], target[2], z_slice, x, y, z, kernel, rotation,
                                           rot_origin, x_pixels, y_pixels, xlim, ylim, backend)
    else:
        raise ValueError('`data` is not a valid number of dimensions.')

    if ax is None:
        ax = plt.gca()

    x, y = _default_axes(data, x, y)
    xlim, ylim = _default_bounds(data, x, y, xlim, ylim)

    kwargs.setdefault("color", 'black')
    ax.streamplot(np.linspace(xlim[0], xlim[1], np.size(img[0], 1)), np.linspace(ylim[0], ylim[1], np.size(img[0], 0)),
                  img[0], img[1], **kwargs)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # remove the x & y ticks if the data is rotated, since these no longer have physical
    # relevance to the displayed data.
    if rotation is not None:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_xlabel(x)
        ax.set_ylabel(y)

    return ax


def arrowplot(data: 'SarracenDataFrame', target: Union[Tuple[str, str], Tuple[str, str, str]], x: str = None,
              y: str = None, z: str = None, z_slice: int = None, kernel: BaseKernel = None,
              integral_samples: int = 1000, rotation: np.ndarray = None, rot_origin: np.ndarray = None,
              x_arrows: int = None, y_arrows: int = None, xlim: tuple[float, float] = None,
              ylim: tuple[float, float] = None, ax: Axes = None, exact: bool = None, backend: str = None,
              **kwargs) -> Axes:
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
    rot_origin: array_like, optional
        Point of rotation of the data, in [x, y, z] form. Defaults to the centre point of the bounds of the data.
    x_arrows, y_arrows: int, optional
        Number of arrows in the output image in the x & y directions. Default values are chosen to keep
        a consistent aspect ratio.
    xlim, ylim: tuple of float, optional
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
    xlim, ylim = _default_bounds(data, x, y, xlim, ylim)
    x_arrows, y_arrows = _set_pixels(x_arrows, y_arrows, xlim, ylim, 20)

    if data.get_dim() == 2:
        if not len(target) == 2:
            raise ValueError('Target vector is not 2-dimensional.')
        img = interpolate_2d_vec(data, target[0], target[1], x, y, kernel, x_arrows, y_arrows, xlim, ylim, exact,
                                 backend)
    elif data.get_dim() == 3:
        if not len(target) == 3:
            raise ValueError('Target vector is not 3-dimensional.')
        if z_slice is None:
            img = interpolate_3d_vec(data, target[0], target[1], target[2], x, y, kernel, integral_samples, rotation,
                                     rot_origin, x_arrows, y_arrows, xlim, ylim, exact, backend)
        else:
            if exact:
                raise UserWarning("Exact interpolation is not supported for 3D cross-sections.")

            img = interpolate_3d_cross_vec(data, target[0], target[1], target[2], z_slice, x, y, z, kernel, rotation,
                                           rot_origin, x_arrows, y_arrows, xlim, ylim, backend)
    else:
        raise ValueError('`data` is not a valid number of dimensions.')

    if ax is None:
        ax = plt.gca()

    kwargs.setdefault("angles", 'uv')
    kwargs.setdefault("pivot", 'mid')
    ax.quiver(np.linspace(xlim[0], xlim[1], np.size(img[0], 1)), np.linspace(ylim[0], ylim[1], np.size(img[0], 0)),
              img[0], img[1], **kwargs)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # remove the x & y ticks if the data is rotated, since these no longer have physical
    # relevance to the displayed data.
    if rotation is not None:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_xlabel(x)
        ax.set_ylabel(y)

    return ax
