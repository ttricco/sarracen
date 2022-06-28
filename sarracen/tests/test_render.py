"""pytest unit tests for render.py functions."""
import pandas as pd
from matplotlib import pyplot as plt
from numpy.testing import assert_array_equal

from sarracen import SarracenDataFrame, interpolate_2d, interpolate_2d_cross, interpolate_3d, interpolate_3d_cross
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


def test_rotated_ticks():
    df = pd.DataFrame({'x': [3, 6], 'y': [5, 1], 'z': [2, 1], 'P': [1, 1], 'h': [1, 1], 'rho': [1, 1], 'm': [1, 1]})
    sdf = SarracenDataFrame(df)

    for func in [render_3d, render_3d_cross]:
        fig, ax = plt.subplots()
        func(sdf, 'P', rotation=[34, 23, 50], ax=ax)

        assert ax.get_xticks().size == 0
        assert ax.get_yticks().size == 0


def test_plot_labels():
    df_2 = pd.DataFrame({'x': [3, 6], 'y': [5, 1], 'P': [1, 1], 'h': [1, 1], 'rho': [1, 1], 'm': [1, 1]})
    sdf_2 = SarracenDataFrame(df_2)
    df_3 = pd.DataFrame({'x': [3, 6], 'y': [5, 1], 'z': [2, 1], 'P': [1, 1], 'h': [1, 1], 'rho': [1, 1], 'm': [1, 1]})
    sdf_3 = SarracenDataFrame(df_3)

    for func in [render_2d, render_3d, render_3d_cross]:
        fig, ax = plt.subplots()
        func(sdf_2 if func is render_2d else sdf_3, 'P', ax=ax)

        assert ax.get_xlabel() == 'x'
        assert ax.get_ylabel() == 'y'
        assert ax.figure.axes[1].get_ylabel() == ('column ' if func is render_3d else '') + 'P'

        fig, ax = plt.subplots()
        func(sdf_2 if func is render_2d else sdf_3, 'rho', x='y', y='x', ax=ax)

        assert ax.get_xlabel() == 'y'
        assert ax.get_ylabel() == 'x'
        assert ax.figure.axes[1].get_ylabel() == ('column ' if func is render_3d else '') + 'rho'

    fig, ax = plt.subplots()
    render_2d_cross(sdf_2, 'P', ax=ax)

    assert ax.get_xlabel() == 'cross-section (x, y)'
    assert ax.get_ylabel() == 'P'

    fig, ax = plt.subplots()
    render_2d_cross(sdf_2, 'rho', x='y', y='x', ax=ax)

    assert ax.get_xlabel() == 'cross-section (y, x)'
    assert ax.get_ylabel() == 'rho'


def test_plot_bounds():
    df_2 = pd.DataFrame({'x': [6, 3], 'y': [5, 1], 'P': [1, 1], 'h': [1, 1], 'rho': [1, 1], 'm': [1, 1]})
    sdf_2 = SarracenDataFrame(df_2)
    df_3 = pd.DataFrame({'x': [6, 3], 'y': [5, 1], 'z': [0, 0], 'P': [1, 1], 'h': [1, 1], 'rho': [1, 1], 'm': [1, 1]})
    sdf_3 = SarracenDataFrame(df_3)

    for func in [render_2d, render_2d_cross, render_3d, render_3d_cross]:
        fig, ax = plt.subplots()
        func(sdf_2 if func is render_2d or func is render_2d_cross else sdf_3, 'P', ax=ax)

        if func is render_2d_cross:
            assert ax.get_xlim() == (0, 5)

            # 512 pixels across (by default), and both particles are in corners
            # therefore closest pixel to a particle is sqrt(41)/1024 units away
            # use default kernel to determine the max pressure value
            assert ax.get_ylim() == (0, interpolate_2d_cross(sdf_2, 'P').max())
        else:
            assert ax.get_xlim() == (3, 6)
            assert ax.get_ylim() == (1, 5)

        if func is render_2d:
            assert ax.figure.axes[1].get_ylim() == (0, interpolate_2d(sdf_2, 'P').max())
        elif func is render_3d:
            assert ax.figure.axes[1].get_ylim() == (0, interpolate_3d(sdf_3, 'P').max())
        elif func is render_3d_cross:
            assert ax.figure.axes[1].get_ylim() == (0, interpolate_3d_cross(sdf_3, 'P').max())


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
