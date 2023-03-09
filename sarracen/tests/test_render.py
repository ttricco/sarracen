"""pytest unit tests for render.py functions."""
import pandas as pd
from matplotlib import pyplot as plt
from numba import cuda
from numpy.testing import assert_array_equal
from pytest import mark

from sarracen import SarracenDataFrame, interpolate_2d, interpolate_2d_line, interpolate_3d_proj, interpolate_3d_cross
from sarracen.render import render, streamlines, arrowplot, lineplot

backends = ['cpu']
if cuda.is_available():
    backends.append('gpu')

@mark.parametrize("backend", backends)
def test_interpolation_passthrough(backend):
    """
    Verify that each rendering function uses the proper underlying interpolation function.
    """
    df = pd.DataFrame({'x': [3, 6], 'y': [5, 1], 'P': [1, 1], 'Ax': [3, 2], 'Ay': [2, 1], 'h': [1, 1], 'rho': [1, 1],
                       'm': [1, 1]})
    sdf = SarracenDataFrame(df)
    sdf.backend = backend

    fig, ax = plt.subplots()
    render(sdf, 'P', ax=ax)
    assert_array_equal(ax.images[0].get_array(), interpolate_2d(sdf, 'P'))
    plt.close(fig)

    fig, ax = plt.subplots()
    lineplot(sdf, 'P', xlim=(3, 6), ylim=(1, 5), ax=ax)
    assert_array_equal(ax.lines[0].get_ydata(), interpolate_2d_line(sdf, 'P', xlim=(3, 6), ylim=(1, 5)))
    plt.close(fig)

    df = pd.DataFrame({'x': [3, 6], 'y': [5, 1], 'z': [2, 1], 'P': [1, 1], 'h': [1, 1], 'rho': [1, 1], 'm': [1, 1]})
    sdf = SarracenDataFrame(df)
    sdf.backend = backend

    fig, ax = plt.subplots()
    render(sdf, 'P', ax=ax)
    assert_array_equal(ax.images[0].get_array(), interpolate_3d_proj(sdf, 'P'))
    plt.close(fig)

    fig, ax = plt.subplots()
    render(sdf, 'P', xsec=1.5, ax=ax)
    assert_array_equal(ax.images[0].get_array(), interpolate_3d_cross(sdf, 'P'))
    plt.close(fig)


@mark.parametrize("backend", backends)
def test_cmap(backend):
    """
    Verify that each rendering function uses the provided color map.
    """
    df = pd.DataFrame({'x': [3, 6], 'y': [5, 1], 'P': [1, 1], 'h': [1, 1], 'rho': [1, 1], 'm': [1, 1]})
    sdf = SarracenDataFrame(df)
    sdf.backend = backend

    fig, ax = plt.subplots()
    render(sdf, 'P', cmap='magma', ax=ax)
    assert ax.images[0].cmap.name == 'magma'
    plt.close(fig)

    df = pd.DataFrame({'x': [3, 6], 'y': [5, 1], 'z': [2, 1], 'P': [1, 1], 'h': [1, 1], 'rho': [1, 1], 'm': [1, 1]})
    sdf = SarracenDataFrame(df)
    sdf.backend = backend

    fig, ax = plt.subplots()
    render(sdf, 'P', cmap='magma', ax=ax)
    assert ax.images[0].cmap.name == 'magma'
    plt.close(fig)

    fig, ax = plt.subplots()
    render(sdf, 'P', xsec=1.5, cmap='magma', ax=ax)
    assert ax.images[0].cmap.name == 'magma'
    plt.close(fig)


@mark.parametrize("backend", backends)
def test_cbar_exclusion(backend):
    """
    Verify that each rendering function respects the cbar argument.
    """
    df_2 = pd.DataFrame({'x': [3, 6], 'y': [5, 1], 'P': [1, 1], 'h': [1, 1], 'rho': [1, 1], 'm': [1, 1]})
    sdf_2 = SarracenDataFrame(df_2)
    sdf_2.backend = backend
    df_3 = pd.DataFrame({'x': [3, 6], 'y': [5, 1], 'z': [2, 1], 'P': [1, 1], 'h': [1, 1], 'rho': [1, 1], 'm': [1, 1]})
    sdf_3 = SarracenDataFrame(df_3)
    sdf_3.backend = backend

    for args in [{'data': sdf_2, 'xsec': None}, {'data': sdf_3, 'xsec': None}, {'data': sdf_3, 'xsec': 1.5}]:
        fig, ax = plt.subplots()
        render(args['data'], 'P', xsec=args['xsec'], cbar=True, ax=ax)
        assert ax.images[-1].colorbar is not None
        plt.close(fig)

        fig, ax = plt.subplots()
        render(args['data'], 'P', xsec=args['xsec'], cbar=False, ax=ax)
        assert ax.images[-1].colorbar is None
        plt.close(fig)


@mark.parametrize("backend", backends)
def test_cbar_keywords(backend):
    """
    Verify that each rendering function respects the passed keywords for the colorbar.
    """
    df_2 = pd.DataFrame({'x': [3, 6], 'y': [5, 1], 'P': [1, 1], 'h': [1, 1], 'rho': [1, 1], 'm': [1, 1]})
    sdf_2 = SarracenDataFrame(df_2)
    sdf_2.backend = backend
    df_3 = pd.DataFrame({'x': [3, 6], 'y': [5, 1], 'z': [2, 1], 'P': [1, 1], 'h': [1, 1], 'rho': [1, 1], 'm': [1, 1]})
    sdf_3 = SarracenDataFrame(df_3)
    sdf_3.backend = backend

    for args in [{'data': sdf_2, 'xsec': None}, {'data': sdf_3, 'xsec': None}, {'data': sdf_3, 'xsec': 1.5}]:
        fig, ax = plt.subplots()
        render(args['data'], 'P', xsec=args['xsec'], cbar_kws={'orientation': 'horizontal'}, ax=ax)
        assert ax.images[-1].colorbar.orientation == 'horizontal'
        plt.close(fig)


@mark.parametrize("backend", backends)
def test_kwargs(backend):
    """
    Verify that each rendering function respects passed keyword arguments.
    """
    df_2 = pd.DataFrame({'x': [3, 6], 'y': [5, 1], 'P': [1, 1], 'h': [1, 1], 'rho': [1, 1], 'm': [1, 1], 'Ax': [1, 1],
                         'Ay': [1, 1]})
    sdf_2 = SarracenDataFrame(df_2)
    sdf_2.backend = backend
    df_3 = pd.DataFrame({'x': [3, 6], 'y': [5, 1], 'z': [2, 1], 'P': [1, 1], 'h': [1, 1], 'rho': [1, 1], 'm': [1, 1]})
    sdf_3 = SarracenDataFrame(df_3)
    sdf_3.backend = backend

    for args in [{'data': sdf_2, 'xsec': None}, {'data': sdf_3, 'xsec': None}, {'data': sdf_3, 'xsec': 1.5}]:
        fig, ax = plt.subplots()
        render(args['data'], 'P', xsec=args['xsec'], ax=ax, origin='upper')
        assert ax.images[0].origin == 'upper'
        plt.close(fig)

    fig, ax = plt.subplots()
    streamlines(sdf_2, ('Ax', 'Ay'), ax=ax, zorder=5)
    assert ax.patches[0].zorder == 5
    plt.close(fig)

    fig, ax = plt.subplots()
    arrowplot(sdf_2, ('Ax', 'Ay'), ax=ax, zorder=5)
    assert ax.collections[0].zorder == 5
    plt.close(fig)

    fig, ax = plt.subplots()
    lineplot(sdf_2, 'P', xlim=(3, 6), ylim=(1, 5), ax=ax, linestyle='--')
    assert ax.lines[0].get_linestyle() == '--'
    plt.close(fig)


@mark.parametrize("backend", backends)
def test_rotated_ticks(backend):
    """
    A rotated plot should have no x & y ticks.
    """
    df = pd.DataFrame({'x': [3, 6], 'y': [5, 1], 'z': [2, 1], 'P': [1, 1], 'h': [1, 1], 'rho': [1, 1], 'm': [1, 1],
                       'Ax': [1, 1], 'Ay': [1, 1], 'Az': [1, 1]})
    sdf = SarracenDataFrame(df)
    sdf.backend = backend

    for xsec in [None, 1.5]:
        fig, ax = plt.subplots()
        render(sdf, 'P', xsec=xsec, ax=ax, rotation=[34, 23, 50])

        assert ax.get_xticks().size == 0
        assert ax.get_yticks().size == 0
        plt.close(fig)


    for func in [arrowplot, streamlines]:
        fig, ax = plt.subplots()
        func(sdf, ('Ax', 'Ay', 'Az'), rotation=[34, 23, 50], ax=ax)

        assert ax.get_xticks().size == 0
        assert ax.get_yticks().size == 0
        plt.close(fig)



@mark.parametrize("backend", backends)
def test_plot_labels(backend):
    """
    Verify that plot labels for each rendering function are correct.
    """
    df_2 = pd.DataFrame({'x': [3, 6], 'y': [5, 1], 'P': [1, 1], 'h': [1, 1], 'rho': [1, 1], 'm': [1, 1], 'Ax': [1, 1],
                         'Ay': [1, 1]})
    sdf_2 = SarracenDataFrame(df_2)
    sdf_2.backend = backend

    df_3 = pd.DataFrame({'x': [3, 6], 'y': [5, 1], 'z': [2, 1], 'P': [1, 1], 'h': [1, 1], 'rho': [1, 1], 'm': [1, 1],
                         'Ax': [1, 1], 'Ay': [1, 1], 'Az': [1, 1]})
    sdf_3 = SarracenDataFrame(df_3)
    sdf_3.backend = backend

    for args in [{'data': sdf_2, 'xsec': None}, {'data': sdf_3, 'xsec': None}, {'data': sdf_3, 'xsec': 0}]:
        fig, ax = plt.subplots()
        render(args['data'], 'P', xsec=args['xsec'], ax=ax)

        assert ax.get_xlabel() == 'x'
        assert ax.get_ylabel() == 'y'
        assert ax.figure.axes[1].get_ylabel() == \
               ('column ' if args['data'] is sdf_3 and args['xsec'] is None else '') + 'P'
        plt.close(fig)

        fig, ax = plt.subplots()
        render(args['data'], 'rho', x='y', y='x', xsec=args['xsec'], ax=ax)

        assert ax.get_xlabel() == 'y'
        assert ax.get_ylabel() == 'x'
        assert ax.figure.axes[1].get_ylabel() == ('column ' if args['data'] is sdf_3 and args['xsec'] is None else '')\
               + 'rho'
        plt.close(fig)

    for func in [streamlines, arrowplot]:
        fig, ax = plt.subplots()
        func(sdf_2, ('Ax', 'Ay'), ax=ax)

        assert ax.get_xlabel() == 'x'
        assert ax.get_ylabel() == 'y'
        plt.close(fig)

        fig, ax = plt.subplots()
        func(sdf_3, ('Ax', 'Ay', 'Az'), x='y', y='x', ax=ax)

        assert ax.get_xlabel() == 'y'
        assert ax.get_ylabel() == 'x'
        plt.close(fig)

    fig, ax = plt.subplots()
    lineplot(sdf_2, 'P', ax=ax)

    assert ax.get_xlabel() == 'cross-section (x, y)'
    assert ax.get_ylabel() == 'P'
    plt.close(fig)

    fig, ax = plt.subplots()
    lineplot(sdf_2, 'rho', x='y', y='x', ax=ax)

    assert ax.get_xlabel() == 'cross-section (y, x)'
    assert ax.get_ylabel() == 'rho'
    plt.close(fig)


@mark.parametrize("backend", backends)
def test_plot_bounds(backend):
    """
    Verify that plot bounds are set correctly for each rendering function.
    """
    df_2 = pd.DataFrame({'x': [6, 3], 'y': [5, 1], 'P': [1, 1], 'h': [1, 1], 'rho': [1, 1], 'm': [1, 1]})
    sdf_2 = SarracenDataFrame(df_2)
    sdf_2.backend = backend

    df_3 = pd.DataFrame({'x': [6, 3], 'y': [5, 1], 'z': [2, 1], 'P': [1, 1], 'h': [1, 1], 'rho': [1, 1], 'm': [1, 1],
                         'Ax': [1, 1], 'Ay': [1, 1], 'Az': [1, 1]})
    sdf_3 = SarracenDataFrame(df_3)
    sdf_3.backend = backend

    for args in [{'data': sdf_2, 'xsec': None}, {'data': sdf_3, 'xsec': None}, {'data': sdf_3, 'xsec': 1.5}]:
        fig, ax = plt.subplots()
        render(args['data'], 'P', xsec=args['xsec'], ax=ax)

        assert ax.get_xlim() == (3, 6)
        assert ax.get_ylim() == (1, 5)

        if args['data'] is sdf_2:
            assert ax.figure.axes[1].get_ylim() == (0, interpolate_2d(sdf_2, 'P').max())
        else:
            if args['xsec']:
                assert ax.figure.axes[1].get_ylim() == (0, interpolate_3d_cross(sdf_3, 'P').max())
            else:
                assert ax.figure.axes[1].get_ylim() == (0, interpolate_3d_proj(sdf_3, 'P').max())
        plt.close(fig)

    fig, ax = plt.subplots()
    lineplot(sdf_2, 'P', ax=ax)

    assert ax.get_xlim() == (0, 5)
    # 512 pixels across (by default), and both particles are in corners
    # therefore closest pixel to a particle is sqrt(41)/1024 units away
    # use default kernel to determine the max pressure value
    assert ax.get_ylim() == (0, interpolate_2d_line(sdf_2, 'P').max())
    plt.close(fig)

    for func in [arrowplot, streamlines]:
        fig, ax = plt.subplots()
        func(sdf_3, ('Ax', 'Ay', 'Az'), ax=ax)

        assert ax.get_xlim() == (3, 6)
        assert ax.get_ylim() == (1, 5)
        plt.close(fig)


@mark.parametrize("backend", backends)
def test_snap(backend):
    df = pd.DataFrame({'x': [0.0001, 5.2],
                       'y': [3.00004, 0.1],
                       'P': [1, 1],
                       'h': [1, 1],
                       'rho': [1, 1],
                       'm': [1, 1]})
    sdf = SarracenDataFrame(df)
    sdf.backend = backend

    fig, ax = plt.subplots()
    sdf.render('P', ax=ax)

    # 0.0001 -> 0.0, 5.2 -> 5.2
    assert ax.get_xlim() == (0.0, 5.2)
    # 0.1 -> 0.1, 3.00004 -> 3.0
    assert ax.get_ylim() == (0.1, 3.0)
    plt.close(fig)
