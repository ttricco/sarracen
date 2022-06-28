"""pytest unit tests for render.py functions."""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy.testing import assert_allclose, assert_array_equal
from pytest import approx

from sarracen import SarracenDataFrame, interpolate_2d, interpolate_2d_cross, interpolate_3d, interpolate_3d_cross
from sarracen.kernels import CubicSplineKernel
from sarracen.render import render_2d, render_2d_cross, render_3d, render_3d_cross


def test_interpolation_passthrough():
    df = pd.DataFrame({'x': [3, 6], 'y': [5, 1], 'P': [1, 1], 'h': [1, 1], 'rho': [1, 1], 'm': [1, 1]})
    sdf = SarracenDataFrame(df)

    fig, ax = plt.subplots()
    render_2d(sdf, 'P', ax=ax)

    assert_array_equal(ax.images[0].get_array().filled(0), interpolate_2d(sdf, 'P'))

    fig, ax = plt.subplots()
    render_2d_cross(sdf, 'P', x1=3, x2=6, y1=5, y2=1, ax=ax)

    assert_array_equal(ax.lines[0].get_ydata(), interpolate_2d_cross(sdf, 'P', x1=3, x2=6, y1=5, y2=1))

    df = pd.DataFrame({'x': [3, 6], 'y': [5, 1], 'z': [2, 1], 'P': [1, 1], 'h': [1, 1], 'rho': [1, 1], 'm': [1, 1]})
    sdf = SarracenDataFrame(df)

    fig, ax = plt.subplots()
    render_3d(sdf, 'P', ax=ax)

    assert_array_equal(ax.images[0].get_array().filled(0), interpolate_3d(sdf, 'P'))

    fig, ax = plt.subplots()
    render_3d_cross(sdf, 'P', ax=ax)

    assert_array_equal(ax.images[0].get_array().filled(0), interpolate_3d_cross(sdf, 'P'))


def test_cmap():
    df = pd.DataFrame({'x': [3, 6], 'y': [5, 1], 'P': [1, 1], 'h': [1, 1], 'rho': [1, 1], 'm': [1, 1]})
    sdf = SarracenDataFrame(df)

    fig, ax = plt.subplots()
    render_2d(sdf, 'P', ax=ax, cmap='magma')

    assert ax.images[0].cmap.name == 'magma'

    df = pd.DataFrame({'x': [3, 6], 'y': [5, 1], 'z': [2, 1], 'P': [1, 1], 'h': [1, 1], 'rho': [1, 1], 'm': [1, 1]})
    sdf = SarracenDataFrame(df)

    fig, ax = plt.subplots()
    render_3d(sdf, 'P', cmap='magma', ax=ax)

    assert ax.images[0].cmap.name == 'magma'

    fig, ax = plt.subplots()
    render_3d_cross(sdf, 'P', cmap='magma', ax=ax)

    assert ax.images[0].cmap.name == 'magma'


def test_cbar_exclusion():
    df_2 = pd.DataFrame({'x': [3, 6], 'y': [5, 1], 'P': [1, 1], 'h': [1, 1], 'rho': [1, 1], 'm': [1, 1]})
    sdf_2 = SarracenDataFrame(df_2)
    df_3 = pd.DataFrame({'x': [3, 6], 'y': [5, 1], 'z': [2, 1], 'P': [1, 1], 'h': [1, 1], 'rho': [1, 1], 'm': [1, 1]})
    sdf_3 = SarracenDataFrame(df_3)

    for func in [render_2d, render_3d, render_3d_cross]:
        fig, ax = plt.subplots()
        func(sdf_2 if func is render_2d else sdf_3, 'P', ax=ax, cbar=True)

        assert ax.images[-1].colorbar is not None

        fig, ax = plt.subplots()
        func(sdf_2 if func is render_2d else sdf_3, 'P', ax=ax, cbar=False)

        assert ax.images[-1].colorbar is None


def test_cbar_keywords():
    df_2 = pd.DataFrame({'x': [3, 6], 'y': [5, 1], 'P': [1, 1], 'h': [1, 1], 'rho': [1, 1], 'm': [1, 1]})
    sdf_2 = SarracenDataFrame(df_2)
    df_3 = pd.DataFrame({'x': [3, 6], 'y': [5, 1], 'z': [2, 1], 'P': [1, 1], 'h': [1, 1], 'rho': [1, 1], 'm': [1, 1]})
    sdf_3 = SarracenDataFrame(df_3)

    for func in [render_2d, render_3d, render_3d_cross]:
        fig, ax = plt.subplots()
        func(sdf_2 if func is render_2d else sdf_3, 'P', ax=ax, cbar_kws={'orientation': 'horizontal'})

        assert ax.images[-1].colorbar.orientation == 'horizontal'


def test_kwargs():
    df_2 = pd.DataFrame({'x': [3, 6], 'y': [5, 1], 'P': [1, 1], 'h': [1, 1], 'rho': [1, 1], 'm': [1, 1]})
    sdf_2 = SarracenDataFrame(df_2)
    df_3 = pd.DataFrame({'x': [3, 6], 'y': [5, 1], 'z': [2, 1], 'P': [1, 1], 'h': [1, 1], 'rho': [1, 1], 'm': [1, 1]})
    sdf_3 = SarracenDataFrame(df_3)

    for func in [render_2d, render_3d, render_3d_cross]:
        fig, ax = plt.subplots()
        func(sdf_2 if func is render_2d else sdf_3, 'P', ax=ax, origin='upper')

        assert ax.images[0].origin == 'upper'

    fig, ax = plt.subplots()
    render_2d_cross(sdf_2, 'P', x1=3, x2=6, y1=5, y2=1, ax=ax, linestyle='--')

    assert ax.lines[0].get_linestyle() == '--'


def test_2d_plot():
    df = pd.DataFrame({'x': [3, 6],
                       'y': [5, 1],
                       'P': [1, 1],
                       'h': [1, 1],
                       'rho': [1, 1],
                       'm': [1, 1]})
    sdf = SarracenDataFrame(df)

    fig, ax = plt.subplots()
    sdf.render_2d('P', ax=ax)

    assert isinstance(ax, Axes)

    assert ax.get_xlabel() == 'x'
    assert ax.get_ylabel() == 'y'
    # the colorbar is contained in a second axes object inside the figure
    assert ax.figure.axes[1].get_ylabel() == 'P'

    assert ax.get_xlim() == (3, 6)
    assert ax.get_ylim() == (1, 5)

    # aspect ratio of data max & min is 4/3,
    # pixel count => (512, 683)
    # pixel width => (3/512, 4/638)
    # both particles are in corners
    # therefore closest pixel is => sqrt((3/1024)**2, (2/683)**2)
    # use default kernel to determine the max pressure value
    assert ax.figure.axes[1].get_ylim() == (0, CubicSplineKernel().w(np.sqrt((3 / 1024) ** 2 + (2 / 683) ** 2), 2))


def test_2d_cross_plot():
    df = pd.DataFrame({'rx': [0, 5],
                       'P': [1, 1],
                       'h': [1, 1],
                       'rho': [1, 1],
                       'y': [0, 4],
                       'm': [1, 1]})
    sdf = SarracenDataFrame(df)

    fig, ax = plt.subplots()
    sdf.render_2d_cross('P', ax=ax)

    assert isinstance(ax, Axes)

    assert ax.get_xlabel() == 'cross-section (rx, y)'
    assert ax.get_ylabel() == 'P'

    # cross section from (0, 0) -> (5, 4), therefore x goes from 0 -> sqrt(5**2 + 4**2)
    assert ax.get_xlim() == (0, np.sqrt(41))
    # 512 pixels across (by default), and both particles are in corners
    # therefore closest pixel to a particle is sqrt(41)/1024 units away
    # use default kernel to determine the max pressure value
    assert ax.get_ylim() == (0, approx(CubicSplineKernel().w(np.sqrt(41) / 1024, 2)))


def test_3d_plot():
    df = pd.DataFrame({'rx': [0, 5],
                       'P': [1, 1],
                       'h': [1, 1],
                       'rz': [3, 1],
                       'rho': [1, 1],
                       'y': [0, 4],
                       'm': [1, 1]})
    sdf = SarracenDataFrame(df)

    fig, ax = plt.subplots()
    sdf.render_3d('P', int_samples=10000, x_pixels=300, ax=ax)

    assert isinstance(ax, Axes)

    assert ax.get_xlabel() == 'rx'
    assert ax.get_ylabel() == 'y'
    # the colorbar is contained in a second axes object inside the figure
    assert ax.figure.axes[1].get_ylabel() == 'column P'

    assert ax.get_xlim() == (0, 5)
    assert ax.get_ylim() == (0, 4)
    # particle width is 1/60
    # therefore closest pixel has q=sqrt(2*(1/120)^2)
    # F(q) ~= 0.477372919027 (calculated externally)
    assert ax.figure.axes[1].get_ylim() == (0, approx(0.477372919027))


def test_3d_cross_plot():
    df = pd.DataFrame({'rx': [0, 2.5, 5],
                       'P': [1, 1, 1],
                       'h': [1, 1, 1],
                       'rz': [3, 2, 1],
                       'rho': [1, 1, 1],
                       'y': [0, 2, 4],
                       'm': [1, 1, 1]})
    sdf = SarracenDataFrame(df)

    # setting the pixel count to an odd number ensures that the middle particle at (2.5, 2, 2) is at the
    # exact same position as the centre pixel.
    fig, ax = plt.subplots()
    sdf.render_3d_cross('P', x_pixels=489, ax=ax)

    assert isinstance(ax, Axes)

    assert ax.get_xlabel() == 'rx'
    assert ax.get_ylabel() == 'y'
    # the colorbar is contained in a second axes object inside the figure
    assert ax.figure.axes[1].get_ylabel() == 'P'

    assert ax.get_xlim() == (0, 5)
    assert ax.get_ylim() == (0, 4)

    # closest pixel/particle pair comes from the particle at (2.5, 2, 2), with r = 0.
    assert ax.figure.axes[1].get_ylim() == (0, approx(CubicSplineKernel().w(0, 3)))


def test_render_passthrough():
    # Basic tests that both sdf.render() and render(sdf) return the same plots

    # 2D dataset
    df = pd.DataFrame({'x': [3, 6],
                       'y': [5, 1],
                       'P': [1, 1],
                       'h': [1, 1],
                       'rho': [1, 1],
                       'm': [1, 1]})
    sdf = SarracenDataFrame(df)

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    ax1 = sdf.render_2d('P', ax=ax1)
    ax2 = render_2d(sdf, 'P', ax=ax2)

    assert repr(ax1) == repr(ax2)

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    ax1 = sdf.render_2d_cross('P', ax=ax1)
    ax2 = render_2d_cross(sdf, 'P', ax=ax2)

    assert repr(ax1) == repr(ax2)

    # 3D dataset
    df = pd.DataFrame({'x': [3, 6],
                       'y': [5, 1],
                       'z': [2, 1],
                       'P': [1, 1],
                       'h': [1, 1],
                       'rho': [1, 1],
                       'm': [1, 1]})
    sdf = SarracenDataFrame(df)

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    ax1 = sdf.render_3d('P', ax=ax1)
    ax2 = render_3d(sdf, 'P', ax=ax2)

    assert repr(ax1) == repr(ax2)

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    ax1 = sdf.render_3d_cross('P', ax=ax1)
    ax2 = render_3d_cross(sdf, 'P', ax=ax2)

    assert repr(ax1) == repr(ax2)


def test_snap():
    df = pd.DataFrame({'x': [0.0001, 5.2],
                       'y': [3.00004, 0.1],
                       'P': [1, 1],
                       'h': [1, 1],
                       'rho': [1, 1],
                       'm': [1, 1]})
    sdf = SarracenDataFrame(df)

    fig, ax = plt.subplots()
    sdf.render_2d('P', ax=ax)

    # 0.0001 -> 0.0, 5.2 -> 5.2
    assert ax.get_xlim() == (0.0, 5.2)
    # 0.1 -> 0.1, 3.00004 -> 3.0
    assert ax.get_ylim() == (0.1, 3.0)
