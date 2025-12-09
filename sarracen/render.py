"""
Provides several rendering functions which produce matplotlib plots of SPH
data. These functions act as interfaces to interpolation functions within
interpolate.py.

These functions can be accessed directly, for example:
    render_2d(data, target)
Or, they can be accessed through a `SarracenDataFrame` object, for example:
    data.render_2d(target)
"""

from typing import Any, Union, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, LogNorm, SymLogNorm

from .interpolate import interpolate_2d_line, interpolate_2d, \
    interpolate_3d_proj, interpolate_3d_cross, interpolate_3d_vec, \
    interpolate_3d_cross_vec, interpolate_2d_vec, interpolate_3d_line
from .kernels import BaseKernel


def _default_axes(data: 'SarracenDataFrame',  # noqa: F821
                  x: Union[str, None],
                  y: Union[str, None]) -> Tuple[str, str]:
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
        The directional column labels to use for rendering. Defaults to the
        x-column detected in `data`
    """
    if x is None:
        x = data.xcol
    if y is None:
        y = data.ycol

    return x, y


def _rotate_data(data: 'SarracenDataFrame',  # noqa: F821
                 x_data: np.ndarray,
                 y_data: np.ndarray,
                 z_data: np.ndarray,
                 rotation: Union[np.ndarray, list, Rotation, None],
                 rot_origin: Union[np.ndarray, list, pd.Series,
                                   str, None]) -> Tuple[np.ndarray,
                                                        np.ndarray,
                                                        np.ndarray]:
    """
    Rotate vector data in a particle dataset.

    Parameters
    ----------
    data: SarracenDataFrame
        The particle dataset to interpolate over.
    x_data, y_data, z_data: ndarray
        The directional vector data.
    rotation: array_like or SciPy Rotation
        The rotation to apply to the vector data. If defined as an array, the
        order of rotations is [z, y, x] in degrees
    rot_origin: array_like or ['com', 'midpoint']
        Point of rotation of the data. Only applies to 3D datasets. If
        array_like, then the [x, y, z] coordinates specify the point around
        which the data is rotated. If 'com', then data is rotated around the
        centre of mass. If 'midpoint', then data is rotated around the
        midpoint, that is, min + max / 2. Defaults to the midpoint.

    Returns
    -------
    x_data, y_data, z_data: ndarray
        The rotated x, y, and z directional data.
    """
    vectors = np.array([x_data, y_data, z_data]).transpose()
    if rotation is not None:
        if not isinstance(rotation, Rotation):
            rotation_obj = Rotation.from_euler('zyx',
                                               rotation,
                                               degrees=True)
        else:
            rotation_obj = rotation

        if rot_origin is None:
            # rot_origin = [0, 0, 0]
            rot_origin_arr = (vectors.min(0) + vectors.max(0)) / 2
        elif rot_origin == 'com':
            rot_origin_arr = data.centre_of_mass()
        elif rot_origin == 'midpoint':
            rot_origin_arr = (vectors.min(0) + vectors.max(0)) / 2
        elif not isinstance(rot_origin, (list, pd.Series, np.ndarray)):
            raise ValueError("rot_origin should be an [x, y, z] point or "
                             "'com' or 'midpoint'")
        elif len(rot_origin) != 3:
            raise ValueError("rot_origin should specify [x, y, z] point.")
        else:
            rot_origin_arr = rot_origin
        vectors = vectors - rot_origin_arr
        vectors = rotation_obj.apply(vectors)
        vectors = vectors + rot_origin_arr

    return vectors[:, 0], vectors[:, 1], vectors[:, 2]


def _default_bounding_box(data: 'SarracenDataFrame',  # noqa: F821
                          x: str,
                          y: str,
                          xlim: Union[Tuple[float, float], None],
                          ylim: Union[Tuple[float, float], None],
                          z_slice: Union[float, None] = None) -> np.ndarray:
    # boundaries of the plot default to the max & min values of the data.
    x_min = xlim[0] if xlim is not None and xlim[0] is not None else None
    y_min = ylim[0] if ylim is not None and ylim[0] is not None else None
    x_max = xlim[1] if xlim is not None and xlim[1] is not None else None
    y_max = ylim[1] if ylim is not None and ylim[1] is not None else None

    x_min = data.loc[:, x].min() if x_min is None else x_min
    y_min = data.loc[:, y].min() if y_min is None else y_min
    x_max = data.loc[:, x].max() if x_max is None else x_max
    y_max = data.loc[:, y].max() if y_max is None else y_max

    z_slice = 0 if z_slice is None else z_slice

    corners = [[x_min, y_min], [x_min, y_max], [x_max, y_min], [x_max, y_max]]
    for i in range(len(corners)):
        corners[i].append(z_slice)

    return np.array(corners).transpose()


def _default_bounds(data: 'SarracenDataFrame',  # noqa: F821
                    x_data: np.ndarray,
                    y_data: np.ndarray,
                    xlim: Union[Tuple[float, float], None],
                    ylim: Union[Tuple[float, float], None],
                    ) -> Tuple[Tuple[float, float],
                               Tuple[float, float]]:
    """
    Utility function to determine the 2-dimensional boundaries to use in 2D
    rendering.

    Parameters
    ----------
    data: SarracenDataFrame
        The particle dataset to render.
    x, y: str
        The directional column labels that will be used for rendering.
    xlim, ylim: tuple of float
        The minimum and maximum values passed to the render function, in
        particle data space.

    Returns
    -------
    xlim, ylim: tuple of float
        The minimum and maximum values to use for rendering, in particle data
        space. Defaults to the maximum and minimum values of `x` and `y`,
        snapped to the nearest integer.
    """
    # boundaries of the plot default to the max & min values of the data.
    x_min = xlim[0] if xlim is not None and xlim[0] is not None else None
    y_min = ylim[0] if ylim is not None and ylim[0] is not None else None
    x_max = xlim[1] if xlim is not None and xlim[1] is not None else None
    y_max = ylim[1] if ylim is not None and ylim[1] is not None else None

    x_min = min(x_data) if x_min is None else x_min
    y_min = min(y_data) if y_min is None else y_min
    x_max = max(x_data) if x_max is None else x_max
    y_max = max(y_data) if y_max is None else y_max

    return (x_min, x_max), (y_min, y_max)


def _set_pixels(x_pixels: Union[int, None],
                y_pixels: Union[int, None],
                xlim: Tuple[float, float],
                ylim: Tuple[float, float],
                default: int) -> Tuple[int, int]:
    """
    Utility function to determine the number of pixels to interpolate over in
    2D interpolation.

    Parameters
    ----------
    x_pixels, y_pixels: int
        The number of pixels in the x & y directions passed to the
        interpolation function.
    xlim, ylim: tuple of float
        The minimum and maximum values to use in interpolation, in particle
        data space.

    Returns
    -------
    x_pixels, y_pixels: int
        The number of pixels in the x & y directions to use in 2D
        interpolation.
    """
    # set # of pixels to maintain an aspect ratio that is the same as the
    # underlying bounds of the data.

    dx = xlim[1] - xlim[0]
    dy = ylim[1] - ylim[0]

    if y_pixels is None:
        if x_pixels is None:
            x_pixels = default
        y_pixels = int(np.rint(x_pixels * (dy / dx)))
    elif x_pixels is None:
        x_pixels = int(np.rint(y_pixels * (dx / dy)))

    return x_pixels, y_pixels


def render(data: 'SarracenDataFrame',  # noqa: F821
           target: str,
           x: Union[str, None] = None,
           y: Union[str, None] = None,
           z: Union[str, None] = None,
           xsec: Union[float, None] = None,
           kernel: Union[BaseKernel, None] = None,
           x_pixels: Union[int, None] = None,
           y_pixels: Union[int, None] = None,
           xlim: Union[Tuple[float, float], None] = None,
           ylim: Union[Tuple[float, float], None] = None,
           cmap: Union[str, Colormap] = 'gist_heat',
           cbar: bool = True,
           cbar_kws: dict = {},
           cbar_ax: Union[Axes, None] = None,
           ax: Union[Axes, None] = None,
           exact: bool = False,
           backend: Union[str, None] = None,
           integral_samples: int = 1000,
           rotation: Union[np.ndarray, list, Rotation, None] = None,
           rot_origin: Union[np.ndarray, list, str, None] = None,
           log_scale: bool = False,
           symlog_scale: bool = False,
           dens_weight: Union[bool, None] = None,
           normalize: bool = True,
           hmin: bool = False,
           corotation: Union[np.ndarray, list, None] = None,
           **kwargs: Any) -> Axes:
    """
    Render a scalar SPH target variable to a grid plot.

    Parameters
    ----------
    data : SarracenDataFrame
        Particle data, in a SarracenDataFrame.
    target: str
        Column label of the target variable.
    x, y, z: str, optional
        Column labels of the x, y & z directional axes. Defaults to the columns
        detected in `data`.
    xsec: float, optional.
        For a 3D dataset, the z value to take a cross-section at. If none,
        column interpolation is performed.
    kernel: BaseKernel, optional
        Kernel to use for smoothing the target data. Defaults to the kernel
        specified in `data`.
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
    cbar_ax: Axes, optional
        Axes to draw the colorbar in, if not provided then space will be taken
        from the main Axes.
    ax: Axes, optional
        The main axes in which to draw the rendered image.
    exact: bool, optional
        Whether to use exact interpolation of the data. For cross-sections this
        is ignored. Defaults to False.
    backend: ['cpu', 'gpu'], optional
        The computation backend to use when interpolating this data. Defaults
        to 'gpu' if CUDA is enabled, otherwise 'cpu' is used. A manually
        specified backend in `data` will override the default.
    integral_samples: int, optional
        If using column interpolation, the number of sample points to take when
        approximating the 2D column kernel.
    rotation: array_like or SciPy Rotation, optional
        The rotation to apply to the data before interpolation. If defined as
        an array, the order of rotations is [z, y, x] in degrees. Only applies
        to 3D datasets.
    rot_origin: array_like or ['com', 'midpoint'], optional
        Point of rotation of the data. Only applies to 3D datasets. If
        array_like, then the [x, y, z] coordinates specify the point around
        which the data is rotated. If 'com', then data is rotated around the
        centre of mass. If 'midpoint', then data is rotated around the
        midpoint, that is, min + max / 2. Defaults to the midpoint.
    log_scale: bool, optional
        Whether to use a logarithmic scale for color coding.
    symlog_scale: bool, optional
        Whether to use a symmetrical logarithmic scale for color coding (i.e.,
        allows positive and negative values). Optionally add "linthresh" and
        "linscale" to kwargs to set the linear region and the scaling of linear
        values, respectively (defaults to 1e-9 and 1, respectively). Only works
        if log_scale == True.
    dens_weight: bool, optional
        If True, will plot the target multiplied by the density. Defaults to
        True for column-integrated views and False for everything else.
    normalize: bool, optional
        If True, will normalize the interpolation. Defaults to True.
    hmin: bool, optional
        If True, a minimum smoothing length of 0.5 * pixel size will be
        imposed. This ensures each particle contributes to at least one grid
        cell / pixel. Defaults to False (this may change in a future verison).
    corotation: list, optional
        Moves particles to the co-rotating frame of two location. corotation
        contains two lists which correspond to the two x, y, z coordinates.
    kwargs: other keyword arguments
        Keyword arguments to pass to ax.imshow.

    Returns
    -------
    Axes
        The resulting matplotlib axes, which contain the 2d rendered image.

    Raises
    ------
    ValueError
        If `x_pixels` or `y_pixels` are less than or equal to zero, or if the
        specified `x` and `y` minimum and maximums result in an invalid region.
    KeyError
        If `target`, `x`, `y`, mass, density, or smoothing length columns do
        not exist in `data`.

    Notes
    -----
    The standard render will interpolate the target quantity, :math:`A`, from
    the particles to a pixel grid using the following equation:

        .. math::

            A_{pixel} = \\sum_b \\frac{m_b}{\\rho_b} A_b W_{ab}(h_b)

    where :math:`m` is the mass, :math:`\\rho` is the density, and :math:`W` is
    the smoothing kernel with smoothing length, :math:`h`.

    Normalized interpolation divides the above summation by an interpolation of
    a constant scalar field equal to 1:

        .. math::

            A_{pixel} = \\frac{\\sum_b \\frac{m_b}{\\rho_b} A_b W_{ab}(h_b)}
                              {\\sum_b \\frac{m_b}{\\rho_b} W_{ab}(h_b)}

    In theory, the denominator will be equal to 1 and dividing by 1 has no
    impact. In practice, the particle arrangement and the smoothing kernel
    affects the quality of interpolation. Normalizing by this approximation of
    1 helps to account for this.

    For when to use normalized interpolation, the advice given by Splash is
    recommended: in general use it for smoother renderings, but avoid when
    there are free surfaces, as it can cause them to be over-exaggerated.

    Density-weighted interpolation will interpolate the quantity
    :math:`\\rho A`, that is, the target :math:`A` multiplied by the density,
    :math:`\\rho`. If normalize=True, then density-weighted interpolation will
    be normalized by the density.

    Column-integrated views of 3D data (i.e., xsec=None) will calculate the
    following:

        .. math::

            A_{pixel} = \\sum_b \\frac{m_b}{\\rho_b} A_b \\int W_{ab}(h_b) dz ,

    which uses the integral of the kernel along the chosen line of sight.

    Exact rendering calculates the volume integral of the kernel through each
    pixel using the method of Petkova et al (2018) [1]_. It only works for the
    cubic spline kernel.

    References
    ----------
    .. [1] M. A. Petkova, G. Laibe & I. A. Bonnell, "Fast and accurate Voronoi
       density gridding from Lagrangian hydrodynamics data," J. Comput. Phys.,
       353, 15, 300-315 (2018). `doi:10.1016/j.jcp.2017.10.024
       <https://doi.org/10.1016/j.jcp.2017.10.024>`_

    """
    if data.get_dim() == 2:
        if dens_weight is None:
            dens_weight = False
        img = interpolate_2d(data, target, x, y, kernel, x_pixels, y_pixels,
                             xlim, ylim, exact, backend, dens_weight,
                             normalize, hmin)
    elif data.get_dim() == 3:
        if xsec is not None:
            if dens_weight is None:
                dens_weight = False
            img = interpolate_3d_cross(data, target, x, y, z, xsec, kernel,
                                       corotation, rotation, rot_origin,
                                       x_pixels, y_pixels, xlim, ylim, backend,
                                       dens_weight, normalize, hmin)
        else:
            img = interpolate_3d_proj(data, target, x, y, kernel,
                                      integral_samples, corotation, rotation,
                                      rot_origin, x_pixels, y_pixels, xlim,
                                      ylim, exact, backend, dens_weight,
                                      normalize, hmin)
    else:
        raise ValueError('`data` is not a valid number of dimensions.')

    if ax is None:
        ax = plt.gca()

    x, y = _default_axes(data, x, y)
    corners = _default_bounding_box(data, x, y, xlim, ylim, xsec)
    rotated_corners = _rotate_data(data, corners[0], corners[1], corners[2],
                                   rotation, rot_origin)
    xlim, ylim = _default_bounds(data, rotated_corners[0], rotated_corners[1],
                                 xlim, ylim)

    kwargs.setdefault("origin", 'lower')
    kwargs.setdefault("extent", [xlim[0], xlim[1], ylim[0], ylim[1]])
    if log_scale:
        # By default, a log scale plot will only cover 4 levels of magnitude.
        vminref = 10 ** (np.log10(kwargs.get("vmax", img.max())) - 4)
        vmin = kwargs.get('vmin', max(vminref, img.min()))

        if symlog_scale:
            kwargs.setdefault("norm",
                              SymLogNorm(kwargs.pop("linthresh", 1e-9),
                                         linscale=kwargs.pop("linscale", 1.),
                                         vmin=vmin,
                                         vmax=kwargs.get('vmax')))
        else:
            kwargs.setdefault("norm", LogNorm(clip=True,
                                              vmin=vmin,
                                              vmax=kwargs.get('vmax')))
        kwargs.pop("vmin", None)
        kwargs.pop("vmax", None)

    graphic = ax.imshow(img, cmap=cmap, **kwargs)
    if rotation is not None and data.get_dim() == 3:
        if corotation is not None:
            ax.set_xlabel(x)
            ax.set_ylabel(y)
    else:
        ax.set_xlabel(x)
        ax.set_ylabel(y)

    if cbar:
        colorbar = ax.figure.colorbar(graphic, cbar_ax, ax, **cbar_kws)
        if 'label' not in cbar_kws:
            label = target
            if data.get_dim() == 3 and xsec is None:
                label = f"column {label}"
            if log_scale:
                label = f"log ({label})"
            colorbar.ax.set_ylabel(label)

    return ax


def lineplot(data: 'SarracenDataFrame',  # noqa: F821
             target: str,
             x: Union[str, None] = None,
             y: Union[str, None] = None,
             z: Union[str, None] = None,
             kernel: Union[BaseKernel, None] = None,
             pixels: int = 512,
             xlim: Union[Tuple[float, float], None] = None,
             ylim: Union[Tuple[float, float], None] = None,
             zlim: Union[Tuple[float, float], None] = None,
             ax: Union[Axes, None] = None,
             backend: Union[str, None] = None,
             log_scale: bool = False,
             dens_weight: bool = False,
             normalize: bool = True,
             hmin: bool = False,
             **kwargs: Any) -> Axes:
    """
    Render a scalar SPH target variable to line plot.

    Parameters
    ----------
    data : SarracenDataFrame
        Particle data, in a SarracenDataFrame.
    target: str
        Column label of the target variable.
    x, y, z: str, optional
        Column labels of the x, y & z directional axes. Defaults to the columns
        detected in `data`.
    kernel: BaseKernel, optional
        Kernel to use for smoothing the target data. Defaults to the kernel
        specified in `data`.
    pixels: int, optional
        Number of samples taken across the x axis in the final plot.
    xlim, ylim, zlim: tuple of float, optional
        Coordinates of the two points that make up the cross-sectional line.
    ax: Axes, optional
        The main axes in which to draw the final plot.
    backend: ['cpu', 'gpu'], optional
        The computation backend to use when interpolating this data. Defaults
        to 'gpu' if CUDA is enabled, otherwise 'cpu' is used. A manually
        specified backend in `data` will override the default.
    log_scale: bool, optional
        Whether to use a logarithmic scale for color coding.
    dens_weight: bool, optional
        If True, will plot the target mutliplied by the density. Defaults to
        False.
    normalize: bool, optional
        If True, will normalize the interpolation. Defaults to False (this may
        change in future versions).
    hmin: bool, optional
        If True, a minimum smoothing length of 0.5 * pixel size will be
        imposed. This ensures each particle contributes to at least one grid
        cell / pixel. Defaults to False (this may change in a future verison).
    kwargs: other keyword arguments
        Keyword arguments to pass to sns.lineplot.

    Returns
    -------
    Axes
        The resulting matplotlib axes, which contain the 2d rendered image.

    Raises
    -------
    ValueError
        If `x_pixels` or `y_pixels` are less than or equal to zero, or if the
        specified `x` and `y` minimum and maximums result in an invalid region.
    KeyError
        If `target`, `x`, `y`, mass, density, or smoothing length columns do
        not exist in `data`.
    """

    if data.get_dim() == 2:
        img = interpolate_2d_line(data, target, x, y, kernel, pixels, xlim,
                                  ylim, backend, dens_weight, normalize, hmin)
    else:
        img = interpolate_3d_line(data, target, x, y, z, kernel, pixels, xlim,
                                  ylim, zlim, backend, dens_weight, normalize,
                                  hmin)

    if isinstance(xlim, float) or isinstance(xlim, int):
        xlim = xlim, xlim
    if isinstance(ylim, float) or isinstance(ylim, int):
        ylim = ylim, ylim

    x, y = _default_axes(data, x, y)
    xlim, ylim = _default_bounds(data, data.loc[:, x], data.loc[:, y],
                                 xlim, ylim)

    if data.get_dim() == 2:
        upper_lim = np.sqrt((xlim[1] - xlim[0])**2 + (ylim[1] - ylim[0])**2)
    else:
        if z is None:
            z = data.zcol
        if z not in data.columns:
            raise KeyError(f"z-directional column '{z}' does not exist in the "
                           f"provided dataset.")

        if isinstance(zlim, float) or isinstance(zlim, int):
            zlim = zlim, zlim
        zmin = data.loc[:, z].min()
        z1 = zmin if zlim is None or zlim[0] is None else zlim[0]
        z2 = zmin if zlim is None or zlim[1] is None else zlim[1]
        zlim = z2, z1

        upper_lim = np.sqrt((xlim[1] - xlim[0])**2
                            + (ylim[1] - ylim[0])**2
                            + (zlim[1] - zlim[0])**2)

    ax = sns.lineplot(x=np.linspace(0, upper_lim, img.size),
                      y=img,
                      ax=ax, **kwargs)

    if log_scale:
        ax.set(yscale='log')

    ax.margins(x=0, y=0)

    label = f'({x}, {y})' if data.get_dim() == 2 else f'({x}, {y}, {z})'
    ax.set_xlabel('cross-section ' + label)

    label = target
    if log_scale:
        label = f"log ({label})"
    ax.set_ylabel(label)

    return ax


def streamlines(data: 'SarracenDataFrame',  # noqa: F821
                target: Union[Tuple[str, str], Tuple[str, str, str]],
                x: Union[str, None] = None,
                y: Union[str, None] = None,
                z: Union[str, None] = None,
                xsec: Union[float, None] = None,
                kernel: Union[BaseKernel, None] = None,
                integral_samples: int = 1000,
                rotation: Union[np.ndarray, list, Rotation, None] = None,
                rot_origin: Union[np.ndarray, list, str, None] = None,
                x_pixels: Union[int, None] = None,
                y_pixels: Union[int, None] = None,
                xlim: Union[Tuple[float, float], None] = None,
                ylim: Union[Tuple[float, float], None] = None,
                ax: Union[Axes, None] = None,
                exact: bool = False,
                backend: Union[str, None] = None,
                dens_weight: Union[bool, None] = None,
                normalize: bool = True,
                hmin: bool = False,
                **kwargs: Any) -> Axes:
    """
    Create an SPH interpolated streamline plot of a target vector.

    Render the data within a SarracenDataFrame to a 2D matplotlib object, by
    rendering the values of a target vector. The contributions of all particles
    near the rendered area are summed and stored to a 2D grid for the x & y
    axes of the target vector. This data is then used to create a streamline
    plot using ax.streamlines().

    Parameters
    ----------
    data: SarracenDataFrame
        Particle data, in a SarracenDataFrame.
    target: str tuple of shape (2) or (3).
        Column label of the target vector. Shape must match the # of dimensions
        in `data`.
    x, y, z: str, optional
        Column label of the x, y & z directional axes. Defaults to the columns
        detected in `data`.
    xsec: float, optional
        The z to take a cross-section at. If none, column interpolation is
        performed.
    kernel: BaseKernel, optional
        Kernel to use for smoothing the target data. Defaults to the kernel
        specified in `data`.
    integral_samples: int, optional
        If using column interpolation, the number of sample points to take when
        approximating the 2D column kernel.
    rotation: array_like or SciPy Rotation, optional
        The rotation to apply to the data before interpolation. If defined as
        an array, the order of rotations is [z, y, x] in degrees.
    rot_origin: array_like or ['com', 'midpoint'], optional
        Point of rotation of the data. Only applies to 3D datasets. If
        array_like, then the [x, y, z] coordinates specify the point around
        which the data is rotated. If 'com', then data is rotated around the
        centre of mass. If 'midpoint', then data is rotated around the
        midpoint, that is, min + max / 2. Defaults to the midpoint.
    x_pixels, y_pixels: int, optional
        Number of interpolation samples to pass to ax.streamlines(). Default
        values are chosen to keep a consistent aspect ratio.
    xlim, ylim: float, optional
        The minimum and maximum values to use in interpolation, in particle
        data space. Defaults to the minimum and maximum values of `x` and `y`.
    ax: Axes, optional
        The main axes in which to draw the rendered image.
    exact: bool, optional
        Whether to use exact interpolation of the data. For cross-sections
        this is ignored. Defaults to False.
    backend: ['cpu', 'gpu'], optional
        The computation backend to use when interpolating this data. Defaults
        to 'gpu' if CUDA is enabled, otherwise 'cpu' is used. A manually
        specified backend in `data` will override the default.
    dens_weight: bool, optional
        If True, will plot the target mutliplied by the density. Defaults to
        True for column-integrated views and False for everything else.
    normalize: bool, optional
        If True, will normalize the interpolation. Defaults to False (this may
        change in future versions).
    hmin: bool, optional
        If True, a minimum smoothing length of 0.5 * pixel size will be
        imposed. This ensures each particle contributes to at least one grid
        cell / pixel. Defaults to False (this may change in a future verison).
    kwargs: other keyword arguments
        Keyword arguments to pass to ax.streamlines()

    Returns
    -------
    Axes
        The resulting matplotlib axes, which contains the streamline plot.

    Raises
    ------
    ValueError
        If `x_pixels` or `y_pixels` are less than or equal to zero, or if the
        specified `x` and `y` minimum and maximums result in an invalid region,
        or if the number of dimensions in the target vector does not match the
        data, or if `data` is not 2 or 3 dimensional.
    KeyError
        If `target`, `x`, `y`, `z` (for 3-dimensional data), mass, density, or
        smoothing length columns do not exist in `data`.
    """
    # Choose between the various interpolation functions available, based on
    # initial data passed to this function.
    if data.get_dim() == 2:
        if not len(target) == 2:
            raise ValueError('Target vector is not 2-dimensional.')
        if dens_weight is None:
            dens_weight = False
        img = interpolate_2d_vec(data, target[0], target[1], x, y, kernel,
                                 x_pixels, y_pixels, xlim, ylim, exact,
                                 backend, dens_weight, normalize, hmin)
    elif data.get_dim() == 3:
        if not len(target) == 3:
            raise ValueError('Target vector is not 3-dimensional.')
        if xsec is not None:
            if dens_weight is None:
                dens_weight = False
            img = interpolate_3d_cross_vec(data, target[0], target[1],
                                           target[2], xsec, x, y, z, kernel,
                                           rotation, rot_origin, x_pixels,
                                           y_pixels, xlim, ylim, backend,
                                           dens_weight, normalize, hmin)
        else:
            if dens_weight is None:
                dens_weight = True
            img = interpolate_3d_vec(data, target[0], target[1], target[2], x,
                                     y, kernel, integral_samples, rotation,
                                     rot_origin, x_pixels, y_pixels, xlim,
                                     ylim, exact, backend, dens_weight,
                                     normalize, hmin)
    else:
        raise ValueError('`data` is not a valid number of dimensions.')

    if ax is None:
        ax = plt.gca()

    x, y = _default_axes(data, x, y)
    corners = _default_bounding_box(data, x, y, xlim, ylim, xsec)
    rotated_corners = _rotate_data(data, corners[0], corners[1], corners[2],
                                   rotation, rot_origin)
    xlim, ylim = _default_bounds(data, rotated_corners[0], rotated_corners[1],
                                 xlim, ylim)

    kwargs.setdefault("color", 'black')
    ax.streamplot(np.linspace(xlim[0], xlim[1], np.size(img[0], 1)),
                  np.linspace(ylim[0], ylim[1], np.size(img[0], 0)),
                  img[0], img[1], **kwargs)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # remove the x & y labels if the data is rotated, since these no longer
    # have physical relevance to the displayed data.
    if rotation is None:
        ax.set_xlabel(x)
        ax.set_ylabel(y)

    return ax


def arrowplot(data: 'SarracenDataFrame',  # noqa: F821
              target: Union[Tuple[str, str], Tuple[str, str, str]],
              x: Union[str, None] = None,
              y: Union[str, None] = None,
              z: Union[str, None] = None,
              xsec: Union[float, None] = None,
              kernel: Union[BaseKernel, None] = None,
              integral_samples: int = 1000,
              rotation: Union[np.ndarray, list, Rotation, None] = None,
              rot_origin: Union[np.ndarray, list, str, None] = None,
              x_arrows: Union[int, None] = None,
              y_arrows: Union[int, None] = None,
              xlim: Union[Tuple[float, float], None] = None,
              ylim: Union[Tuple[float, float], None] = None,
              ax: Union[Axes, None] = None,
              qkey: bool = True,
              qkey_kws: Union[dict, None] = None,
              exact: bool = False,
              backend: Union[str, None] = None,
              dens_weight: Union[bool, None] = None,
              normalize: bool = True,
              hmin: bool = False,
              **kwargs: Any) -> Axes:
    """
    Create an SPH interpolated vector field plot of a target vector.

    Render the data within a SarracenDataFrame to a 2D matplotlib object, by
    rendering the values of a target vector. The contributions of all particles
    near the rendered area are summed and stored to a 2D grid for the x & y
    axes of the target vector. This data is then used to create an arrow plot
    using ax.quiver().

    Parameters
    ----------
    data : SarracenDataFrame
        Particle data, in a SarracenDataFrame.
    target: str tuple of shape (2) or (3).
        Column label of the target vector. Shape must match the # of dimensions
        in `data`.
    x, y, z: str, optional
        Column label of the x, y & z directional axes. Defaults to the columns
        detected in `data`.
    xsec: float, optional
        The z to take a cross-section at. If none, column interpolation is
        performed.
    kernel: BaseKernel, optional
        Kernel to use for smoothing the target data. Defaults to the kernel
        specified in `data`.
    integral_samples: int, optional
        If using column interpolation, the number of sample points to take when
        approximating the 2D column kernel.
    rotation: array_like or SciPy Rotation, optional
        The rotation to apply to the data before interpolation. If defined as
        an array, the order of rotations is [z, y, x] in degrees.
    rot_origin: array_like or ['com', 'midpoint'], optional
        Point of rotation of the data. Only applies to 3D datasets. If
        array_like, then the [x, y, z] coordinates specify the point around
        which the data is rotated. If 'com', then data is rotated around the
        centre of mass. If 'midpoint', then data is rotated around the
        midpoint, that is, min + max / 2. Defaults to the midpoint.
    x_arrows, y_arrows: int, optional
        Number of arrows in the output image in the x & y directions. Default
        values are chosen to keep a consistent aspect ratio.
    xlim, ylim: tuple of float, optional
        The minimum and maximum values to use in interpolation, in particle
        data space. Defaults to the minimum and maximum values of `x` and `y`.
    ax: Axes, optional
        The main axes in which to draw the rendered image.
    qkey: bool, optional
        Whether to include a quiver key on the final plot.
    qkey_kws: dict, optional
        Keywords to pass through to ax.quiver.
    exact: bool, optional
        Whether to use exact interpolation of the data. For cross-sections this
        is ignored. Defaults to False.
    backend: ['cpu', 'gpu'], optional
        The computation backend to use when interpolating this data. Defaults
        to 'gpu' if CUDA is enabled, otherwise 'cpu' is used. A manually
        specified backend in `data` will override the default.
    dens_weight: bool, optional
        If True, will plot the target mutliplied by the density. Defaults to
        True for column-integrated views and False for everything else.
    normalize: bool, optional
        If True, will normalize the interpolation. Defaults to False (this may
        change in future versions).
    hmin: bool, optional
        If True, a minimum smoothing length of 0.5 * pixel size will be
        imposed. This ensures each particle contributes to at least one grid
        cell / pixel. Defaults to False (this may change in a future verison).
    kwargs: other keyword arguments
        Keyword arguments to pass to ax.quiver()

    Returns
    -------
    Axes
        The resulting matplotlib axes, which contains the arrow plot.

    Raises
    ------
    ValueError
        If `x_arrows` or `y_arrows` are less than or equal to zero, or if the
        specified `x` and `y` minimum and maximums result in an invalid region,
        or if the number of dimensions in the target vector does not match the
        data, or if `data` is not 2 or 3-dimensional.
    KeyError
        If `target`, `x`, `y`, `z` (for 3-dimensional data), mass, density, or
        smoothing length columns do not exist in `data`.
    """
    x, y = _default_axes(data, x, y)
    corners = _default_bounding_box(data, x, y, xlim, ylim, xsec)
    rotated_corners = _rotate_data(data, corners[0], corners[1], corners[2],
                                   rotation, rot_origin)
    xlim, ylim = _default_bounds(data, rotated_corners[0], rotated_corners[1],
                                 xlim, ylim)
    x_arrows, y_arrows = _set_pixels(x_arrows, y_arrows, xlim, ylim, 20)

    if data.get_dim() == 2:
        if not len(target) == 2:
            raise ValueError('Target vector is not 2-dimensional.')
        if dens_weight is None:
            dens_weight = False
        img = interpolate_2d_vec(data, target[0], target[1], x, y, kernel,
                                 x_arrows, y_arrows, xlim, ylim, exact,
                                 backend, dens_weight, normalize, hmin)
    elif data.get_dim() == 3:
        if not len(target) == 3:
            raise ValueError('Target vector is not 3-dimensional.')
        if xsec is not None:
            if dens_weight is None:
                dens_weight = False
            img = interpolate_3d_cross_vec(data, target[0], target[1],
                                           target[2], xsec, x, y, z, kernel,
                                           rotation, rot_origin, x_arrows,
                                           y_arrows, xlim, ylim, backend,
                                           dens_weight, normalize, hmin)
        else:
            if dens_weight is None:
                dens_weight = True
            img = interpolate_3d_vec(data, target[0], target[1], target[2], x,
                                     y, kernel, integral_samples, rotation,
                                     rot_origin, x_arrows, y_arrows, xlim,
                                     ylim, exact, backend, dens_weight,
                                     normalize, hmin)
    else:
        raise ValueError('`data` is not a valid number of dimensions.')

    if ax is None:
        ax = plt.gca()

    kwargs.setdefault("angles", 'uv')
    kwargs.setdefault("pivot", 'mid')
    ax.quiver(np.linspace(xlim[0], xlim[1], np.size(img[0], 1)),
              np.linspace(ylim[0], ylim[1], np.size(img[0], 0)),
              img[0], img[1], **kwargs)
    Q = ax.quiver(np.linspace(xlim[0], xlim[1], np.size(img[0], 1)),
                  np.linspace(ylim[0], ylim[1], np.size(img[0], 0)),
                  img[0], img[1], **kwargs)

    if qkey:
        if qkey_kws is None:
            qkey_kws = dict()
        # approximately equivalent to the top right of the plot.
        qkey_kws.setdefault('X', 0.85)
        qkey_kws.setdefault('Y', 1.02)

        # find a reasonable default value for the quiver key length.
        key_length = np.mean(np.sqrt(img[0] ** 2 + img[1] ** 2))
        key_length = float(np.format_float_positional(key_length,
                                                      precision=1,
                                                      unique=False,
                                                      fractional=False,
                                                      trim='k'))
        qkey_kws.setdefault('U', key_length)
        qkey_kws.setdefault('label', f"= {qkey_kws['U']}")

        qkey_kws.setdefault('labelpos', 'E')
        qkey_kws.setdefault('coordinates', 'axes')

        ax.quiverkey(Q, **qkey_kws)

    # remove the x & y labels if the data is rotated, since these no longer
    # have physical relevance to the displayed data.
    if rotation is None:
        ax.set_xlabel(x)
        ax.set_ylabel(y)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return ax
