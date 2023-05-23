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

from .interpolate import interpolate_2d_line, interpolate_2d, interpolate_3d_proj, interpolate_3d_cross, \
    interpolate_3d_vec, interpolate_3d_cross_vec, interpolate_2d_vec, interpolate_3d_line
from .kernels import BaseKernel

from typing import Tuple

def _snap(value: float):
    """
    Snap a number to the nearest integer

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
    """
    Utility function to determine the x & y columns to use for rendering.

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


def _default_bounds(data, x, y, xlim, ylim) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Utility function to determine the 2-dimensional boundaries to use in 2D rendering.

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
    """
    Utility function to determine the number of pixels to interpolate over in 2D interpolation.
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
           xlim: Tuple[float, float] = None, ylim: Tuple[float, float] = None, cmap: Union[str, Colormap] = 'gist_heat',
           cbar: bool = True, cbar_kws: dict = {}, cbar_ax: Axes = None, ax: Axes = None, exact: bool = None,
           backend: str = None, integral_samples: int = 1000, rotation: np.ndarray = None,
           rot_origin: np.ndarray = None, log_scale: bool = False, dens_weight: bool = None, normalize: bool = True,
           hmin: bool = False, **kwargs) -> Axes:
    """
    Render a scalar SPH target variable to a grid plot.

    Parameters
    ----------
    data : SarracenDataFrame
        Particle data, in a SarracenDataFrame.
    target: str
        Column label of the target variable.
    x, y, z: str, optional
        Column labels of the x, y & z directional axes. Defaults to the columns detected in `data`.
    xsec: float, optional.
        For a 3D dataset, the z value to take a cross-section at. If none, column interpolation is performed.
    kernel: BaseKernel, optional
        Kernel to use for smoothing the target data. Defaults to the kernel specified in `data`.
    x_pixels, y_pixels: int, optional
        Number of pixels present in the final image.
    xlim, ylim: tuple of float, optional
        The starting and ending corners of the final 2D image.
    vmin, vmax: float, optional
        Lower and upper limits of the range of values for the colour bar.
    cmap: str or Colormap, optional
        The color map to use when plotting a 2D image.
    cbar: bool, optional
        True if a colorbar should be drawn.
    cbar_kws: dict, optional
        Keyword arguments to pass to matplotlib.figure.Figure.colorbar().
    cbar_ax: Axes
        Axes to draw the colorbar in, if not provided then space will be taken from the main Axes.
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
    dens_weight: bool
        If True, will plot the target mutliplied by the density. Defaults to True for column-integrated views,
        when the target is not density, and False for everything else.
    normalize: bool
        If True, will normalize the interpolation. Defaults to False (this may change in future versions).
    hmin: bool
        If True, a minimum smoothing length of 0.5 * pixel size will be imposed. This ensures each particle
        contributes to at least one grid cell / pixel. Defaults to False (this may change in a future verison).
    kwargs: other keyword arguments
        Keyword arguments to pass to ax.imshow.

    Returns
    -------
    Axes
        The resulting matplotlib axes, which contain the 2d rendered image.

    Raises
    ------
    ValueError
        If `x_pixels` or `y_pixels` are less than or equal to zero, or
        if the specified `x` and `y` minimum and maximums result in an invalid region.
    KeyError
        If `target`, `x`, `y`, mass, density, or smoothing length columns do not
        exist in `data`.

    Notes
    -----
    The standard render will interpolate the target quantity, :math:`A`, from the particles to a pixel
    grid using the following equation:

        .. math::

            A_{pixel} = \\sum_b \\frac{m_b}{\\rho_b} A_b W_{ab}(h_b)

    where :math:`m` is the mass, :math:`\\rho` is the density, and :math:`W` is the smoothing kernel with
    smoothing length, :math:`h`.

    Normalized interpolation divides the above summation by an interpolation of a constant scalar field
    equal to 1:

        .. math::

            A_{pixel} = \\frac{\\sum_b \\frac{m_b}{\\rho_b} A_b W_{ab}(h_b)}{\\sum_b \\frac{m_b}{\\rho_b} W_{ab}(h_b)}

    In theory, the denominator will be equal to 1 and dividing by 1 has no impact. In practice, the
    particle arrangement and the smoothing kernel affects the quality of interpolation. Normalizing by
    this approximation of 1 helps to account for this.

    For when to use normalized interpolation, the advice given by Splash is recommended: in general use
    it for smoother renderings, but avoid when there are free surfaces, as it can cause them to be
    over-exaggerated.

    Density-weighted interpolation will interpolate the quantity :math:`\\rho A`, that is, the target
    :math:`A` multiplied by the density, :math:`\\rho`. If normalize=True, then density-weighted
    interpolation will be normalized by the density.

    Column-integrated views of 3D data (i.e., xsec=None) will calculate the following:

        .. math::

            A_{pixel} = \\sum_b \\frac{m_b}{\\rho_b} A_b \int W_{ab}(h_b) dz ,

    which uses the integral of the kernel along the chosen line of sight.
    """
    if data.get_dim() == 2:
        interpolation_type = '2d'
        if dens_weight is None:
            dens_weight = False
    else:
        if xsec is not None:
            interpolation_type = '3d_cross'
            if dens_weight is None:
                dens_weight = False
        else:
            interpolation_type = '3d'

    if interpolation_type == '2d':
        img = interpolate_2d(data, target, x, y, kernel, x_pixels, y_pixels, xlim, ylim, exact, backend, dens_weight,
                             normalize, hmin)
    elif interpolation_type == '3d_cross':
        img = interpolate_3d_cross(data, target, x, y, z, xsec, kernel, rotation,
                                   rot_origin, x_pixels, y_pixels, xlim, ylim, backend, dens_weight, normalize, hmin)
    elif interpolation_type == '3d':
        img = interpolate_3d_proj(data, target, x, y, kernel, integral_samples, rotation, rot_origin, x_pixels,
                             y_pixels, xlim, ylim, exact, backend, dens_weight, normalize, hmin)
    else:
        raise ValueError('`data` is not a valid number of dimensions.')

    if ax is None:
        ax = plt.gca()

    x, y = _default_axes(data, x, y)
    xlim, ylim = _default_bounds(data, x, y, xlim, ylim)

    kwargs.setdefault("origin", 'lower')
    kwargs.setdefault("extent", [xlim[0], xlim[1], ylim[0], ylim[1]])
    if log_scale:
        kwargs.setdefault("norm", LogNorm(clip=True, vmin=kwargs.get('vmin'), vmax=kwargs.get('vmax')))
        kwargs.pop("vmin", None)
        kwargs.pop("vmax", None)

    graphic = ax.imshow(img, cmap=cmap, **kwargs)
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


def lineplot(data: 'SarracenDataFrame', target: str, x: str = None, y: str = None, z: str = None,
             kernel: BaseKernel = None, pixels: int = 512, xlim: Tuple[float, float] = None,
             ylim: Tuple[float, float] = None, zlim: Tuple[float, float] = None, ax: Axes = None, backend: str = None,
             log_scale: bool = False, dens_weight: bool = False, normalize: bool = True, hmin: bool = False, **kwargs):
    """
    Render a scalar SPH target variable to line plot.

    Parameters
    ----------
    data : SarracenDataFrame
        Particle data, in a SarracenDataFrame.
    target: str
        Column label of the target variable.
    x, y, z: str, optional
        Column labels of the x, y & z directional axes. Defaults to the columns detected in `data`.
    kernel: BaseKernel, optional
        Kernel to use for smoothing the target data. Defaults to the kernel specified in `data`.
    pixels: int, optional
        Number of samples taken across the x axis in the final plot.
    xlim, ylim, zlim: tuple of float, optional
        Coordinates of the two points that make up the cross-sectional line.
    ax: Axes
        The main axes in which to draw the final plot.
    backend: ['cpu', 'gpu']
        The computation backend to use when performing the underlying interpolation. Defaults to the backend
        specified in `data`.
    log_scale: bool
        Whether to use a logarithmic scale for color coding.
    dens_weight: bool
        If True, will plot the target mutliplied by the density. Defaults to False.
    normalize: bool
        If True, will normalize the interpolation. Defaults to False (this may change in future versions).
    hmin: bool
        If True, a minimum smoothing length of 0.5 * pixel size will be imposed. This ensures each particle
        contributes to at least one grid cell / pixel. Defaults to False (this may change in a future verison).
    kwargs: other keyword arguments
        Keyword arguments to pass to sns.lineplot.

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
        img = interpolate_2d_line(data, target, x, y, kernel, pixels, xlim, ylim, backend, dens_weight, normalize,
                                  hmin)
    else:
        img = interpolate_3d_line(data, target, x, y, z, kernel, pixels, xlim, ylim, zlim, backend, dens_weight,
                                  normalize, hmin)

    if isinstance(xlim, float) or isinstance(xlim, int):
        xlim = xlim, xlim
    if isinstance(ylim, float) or isinstance(ylim, int):
        ylim = ylim, ylim

    x, y = _default_axes(data, x, y)
    xlim, ylim = _default_bounds(data, x, y, xlim, ylim)

    if data.get_dim() == 2:
        upper_lim = np.sqrt((xlim[1] - xlim[0]) ** 2 + (ylim[1] - ylim[0]) ** 2)
    else:
        if z is None:
            z = data.zcol
        if z not in data.columns:
            raise KeyError(f"z-directional column '{z}' does not exist in the provided dataset.")

        if isinstance(zlim, float) or isinstance(zlim, int):
            zlim = zlim, zlim
        if zlim is None or zlim[0] is None:
            z1 = _snap(data.loc[:, z].min())
        else:
            z1 = zlim[0]
        if zlim is None or zlim[1] is None:
            z2 = _snap(data.loc[:, z].max())
        else:
            z2 = zlim[1]
        zlim = z2, z1

        upper_lim = np.sqrt((xlim[1] - xlim[0]) ** 2 + (ylim[1] - ylim[0]) ** 2 + (zlim[1] - zlim[0]) ** 2)

    ax = sns.lineplot(x=np.linspace(0, upper_lim, img.size), y=img, ax=ax, **kwargs)

    if log_scale:
        ax.set(yscale='log')

    ax.margins(x=0, y=0)

    ax.set_xlabel('cross-section ' + (f'({x}, {y})' if data.get_dim() == 2 else f'({x}, {y}, {z})'))

    label = target
    if log_scale:
        label = f"log ({label})"
    ax.set_ylabel(label)

    return ax


def streamlines(data: 'SarracenDataFrame', target: Union[Tuple[str, str], Tuple[str, str, str]], x: str = None,
                y: str = None, z: str = None, xsec: int = None, kernel: BaseKernel = None,
                integral_samples: int = 1000, rotation: np.ndarray = None, rot_origin: np.ndarray = None,
                x_pixels: int = None, y_pixels: int = None, xlim: Tuple[float, float] = None,
                ylim: Tuple[float, float] = None, ax: Axes = None, exact: bool = None, backend: str = None,
                dens_weight: bool = None, normalize: bool = True, hmin: bool = False, **kwargs) -> Axes:
    """
    Create an SPH interpolated streamline plot of a target vector.

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
    xsec: float
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
    dens_weight: bool
        If True, will plot the target mutliplied by the density. Defaults to True for column-integrated views
        and False for everything else.
    normalize: bool
        If True, will normalize the interpolation. Defaults to False (this may change in future versions).
    hmin: bool
        If True, a minimum smoothing length of 0.5 * pixel size will be imposed. This ensures each particle
        contributes to at least one grid cell / pixel. Defaults to False (this may change in a future verison).
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
    interpolation_type = None

    if data.get_dim() == 2:
        if not len(target) == 2:
            raise ValueError('Target vector is not 2-dimensional.')
        interpolation_type = '2d'
    elif data.get_dim() == 3:
        if not len(target) == 3:
            raise ValueError('Target vector is not 3-dimensional.')
        if xsec is not None:
            interpolation_type = '3d_cross'
        else:
            interpolation_type = '3d'

    if interpolation_type == '2d':
        img = interpolate_2d_vec(data, target[0], target[1], x, y, kernel, x_pixels, y_pixels, xlim, ylim, exact,
                                 backend, dens_weight, normalize, hmin)
    elif interpolation_type == '3d_cross':
        img = interpolate_3d_cross_vec(data, target[0], target[1], target[2], xsec, x, y, z, kernel, rotation,
                                       rot_origin, x_pixels, y_pixels, xlim, ylim, backend, dens_weight, normalize,
                                       hmin)
    elif interpolation_type == '3d':
        img = interpolate_3d_vec(data, target[0], target[1], target[2], x, y, kernel, integral_samples, rotation,
                                 rot_origin, x_pixels, y_pixels, xlim, ylim, exact, backend, dens_weight, normalize,
                                 hmin)
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
              y: str = None, z: str = None, xsec: float = None, kernel: BaseKernel = None,
              integral_samples: int = 1000, rotation: np.ndarray = None, rot_origin: np.ndarray = None,
              x_arrows: int = None, y_arrows: int = None, xlim: Tuple[float, float] = None,
              ylim: Tuple[float, float] = None, ax: Axes = None, qkey: bool = True, qkey_kws=None, exact: bool = None,
              backend: str = None, dens_weight: bool = None, normalize: bool = True, hmin: bool = False,
              **kwargs) -> Axes:
    """
    Create an SPH interpolated vector field plot of a target vector.

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
    xsec: float
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
    qkey: bool
        Whether to include a quiver key on the final plot.
    qkey_kws: dict
        Keywords to pass through to ax.quiver.
    exact: bool
        Whether to use exact interpolation of the data. For cross-sections this is ignored. Defaults to False.
    backend: ['cpu', 'gpu']
        The computation backend to use when rendering this data. Defaults to the backend specified in `data`.
    dens_weight: bool
        If True, will plot the target mutliplied by the density. Defaults to True for column-integrated views
        and False for everything else.
    normalize: bool
        If True, will normalize the interpolation. Defaults to False (this may change in future versions).
    hmin: bool
        If True, a minimum smoothing length of 0.5 * pixel size will be imposed. This ensures each particle
        contributes to at least one grid cell / pixel. Defaults to False (this may change in a future verison).
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

    interpolation_type = None

    if data.get_dim() == 2:
        if not len(target) == 2:
            raise ValueError('Target vector is not 2-dimensional.')
        interpolation_type = '2d'
    elif data.get_dim() == 3:
        if not len(target) == 3:
            raise ValueError('Target vector is not 3-dimensional.')
        if xsec is not None:
            interpolation_type = '3d_cross'
        else:
            interpolation_type = '3d'

    if interpolation_type == '2d':
        img = interpolate_2d_vec(data, target[0], target[1], x, y, kernel, x_arrows, y_arrows, xlim, ylim, exact,
                                 backend, dens_weight, normalize, hmin)
    elif interpolation_type == '3d_cross':
        img = interpolate_3d_cross_vec(data, target[0], target[1], target[2], xsec, x, y, z, kernel, rotation,
                                       rot_origin, x_arrows, y_arrows, xlim, ylim, backend, dens_weight, normalize,
                                       hmin)
    elif interpolation_type == '3d':
        img = interpolate_3d_vec(data, target[0], target[1], target[2], x, y, kernel, integral_samples, rotation,
                                 rot_origin, x_arrows, y_arrows, xlim, ylim, exact, backend, dens_weight, normalize,
                                 hmin)
    else:
        raise ValueError('`data` is not a valid number of dimensions.')


    if ax is None:
        ax = plt.gca()

    kwargs.setdefault("angles", 'uv')
    kwargs.setdefault("pivot", 'mid')
    ax.quiver(np.linspace(xlim[0], xlim[1], np.size(img[0], 1)), np.linspace(ylim[0], ylim[1], np.size(img[0], 0)),
              img[0], img[1], **kwargs)
    Q = ax.quiver(np.linspace(xlim[0], xlim[1], np.size(img[0], 1)), np.linspace(ylim[0], ylim[1], np.size(img[0], 0)),
                  img[0], img[1], **kwargs)

    if qkey:
        if qkey_kws is None:
            qkey_kws = dict()
        # approximately equivalent to the top right of the plot.
        qkey_kws.setdefault('X', 0.85)
        qkey_kws.setdefault('Y', 1.02)

        # find a reasonable default value for the quiver key length.
        key_length = float(np.format_float_positional(np.mean(np.sqrt(img[0] ** 2 + img[1] ** 2)), precision=1,
                                                      unique=False, fractional=False, trim='k'))
        qkey_kws.setdefault('U', key_length)
        qkey_kws.setdefault('label', f"= {qkey_kws['U']}")

        qkey_kws.setdefault('labelpos', 'E')
        qkey_kws.setdefault('coordinates', 'axes')

        ax.quiverkey(Q, **qkey_kws)

    # remove the x & y ticks if the data is rotated, since these no longer have physical
    # relevance to the displayed data.
    if rotation is not None:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_xlabel(x)
        ax.set_ylabel(y)

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
