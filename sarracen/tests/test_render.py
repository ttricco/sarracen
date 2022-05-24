import pandas as pd
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pytest import approx

from sarracen import SarracenDataFrame
from sarracen.kernels import CubicSplineKernel
from sarracen.render import render_2d, render_1d_cross, render_3d, render_3d_cross


def test_2d_plot():
    df = pd.DataFrame({'x': [3, 6],
                       'y': [5, 1],
                       'P': [1, 1],
                       'h': [1, 1],
                       'rho': [1, 1],
                       'm': [1, 1]})
    sdf = SarracenDataFrame(df)

    fig, ax = sdf.render_2d('P')

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    assert ax.get_xlabel() == 'x'
    assert ax.get_ylabel() == 'y'
    # the colorbar is contained in a second axes object inside the figure
    assert fig.axes[1].get_ylabel() == 'P'

    assert ax.get_xlim() == (3, 6)
    assert ax.get_ylim() == (1, 5)

    # aspect ratio of data max & min is 4/3,
    # pixel count => (256, 341)
    # pixel width => (3/256, 4/341)
    # both particles are in corners
    # therefore closest pixel is => sqrt((3/512)**2, (2/341)**2)
    # use default kernel to determine the max pressure value
    assert fig.axes[1].get_ylim() == (0, CubicSplineKernel().w(np.sqrt((3/512)**2 + (2/341)**2), 2))


def test_1dcross_plot():
    df = pd.DataFrame({'rx': [0, 5],
                       'P': [1, 1],
                       'h': [1, 1],
                       'rho': [1, 1],
                       'y': [0, 4],
                       'm': [1, 1]})
    sdf = SarracenDataFrame(df)

    fig, ax = sdf.render_1d_cross('P')

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    assert ax.get_xlabel() == 'cross-section (rx, y)'
    assert ax.get_ylabel() == 'P'

    # cross section from (0, 0) -> (5, 4), therefore x goes from 0 -> sqrt(5**2 + 4**2)
    assert ax.get_xlim() == (0, np.sqrt(41))
    # 1000 pixels across, and both particles are in corners
    # therefore closest pixel to a particle is sqrt(41)/1000 units away
    # use default kernel to determine the max pressure value
    assert ax.get_ylim() == (0, CubicSplineKernel().w(np.sqrt(41) / 1000, 2))


def test_3d_plot():
    df = pd.DataFrame({'rx': [0, 5],
                       'P': [1, 1],
                       'h': [1, 1],
                       'rz': [3, 1],
                       'rho': [1, 1],
                       'y': [0, 4],
                       'm': [1, 1]})
    sdf = SarracenDataFrame(df)

    fig, ax = sdf.render_3d('P', pixcountx=300, int_samples=10000)

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    assert ax.get_xlabel() == 'rx'
    assert ax.get_ylabel() == 'y'
    # the colorbar is contained in a second axes object inside the figure
    assert fig.axes[1].get_ylabel() == 'column P'

    assert ax.get_xlim() == (0, 5)
    assert ax.get_ylim() == (0, 4)
    # particle width is 1/60
    # therefore closest pixel has q=sqrt(2*(1/120)^2)
    # F(q) ~= 0.477372919027 (calculated externally)
    assert fig.axes[1].get_ylim() == (0, approx(0.477372919027))


def test_3dcross_plot():
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
    fig, ax = sdf.render_3d_cross('P', pixcountx=489)

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    assert ax.get_xlabel() == 'rx'
    assert ax.get_ylabel() == 'y'
    # the colorbar is contained in a second axes object inside the figure
    assert fig.axes[1].get_ylabel() == 'P'

    assert ax.get_xlim() == (0, 5)
    assert ax.get_ylim() == (0, 4)

    # closest pixel/particle pair comes from the particle at (2.5, 2, 2), with r = 0.
    assert fig.axes[1].get_ylim() == (0, approx(CubicSplineKernel().w(0, 3)))


def test_render_passthrough():
    # Basic tests that both sdf.render() and render(sdf) return the same plots
    df = pd.DataFrame({'x': [3, 6],
                       'y': [5, 1],
                       'P': [1, 1],
                       'h': [1, 1],
                       'z': [6, 3],
                       'rho': [1, 1],
                       'm': [1, 1]})
    sdf = SarracenDataFrame(df)

    fig1, ax1 = sdf.render_2d('P')
    fig2, ax2 = render_2d(sdf, 'P')

    assert repr(fig1) == repr(fig2)
    assert repr(ax1) == repr(ax2)

    fig1, ax1 = sdf.render_1d_cross('P')
    fig2, ax2 = render_1d_cross(sdf, 'P')

    assert repr(fig1) == repr(fig2)
    assert repr(ax1) == repr(ax2)

    fig1, ax1 = sdf.render_3d('P')
    fig2, ax2 = render_3d(sdf, 'P')

    assert repr(fig1) == repr(fig2)
    assert repr(ax1) == repr(ax2)

    fig1, ax1 = sdf.render_3d_cross('P')
    fig2, ax2 = render_3d_cross(sdf, 'P')

    assert repr(fig1) == repr(fig2)
    assert repr(ax1) == repr(ax2)
