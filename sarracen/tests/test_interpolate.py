import pandas as pd
import numpy as np
from pytest import approx

from sarracen import SarracenDataFrame
from sarracen.kernels import CubicSplineKernel, QuarticSplineKernel
from sarracen.interpolate import interpolate2DCross, interpolate1DCross, interpolate3DCross, interpolate3D


def test_interpolate2d():
    df = pd.DataFrame({'x': [0],
                       'y': [0],
                       'P': [1],
                       'h': [1],
                       'rho': [1],
                       'm': [1]})
    sdf = SarracenDataFrame(df, params=dict())

    image = interpolate2DCross(sdf, 'x', 'y', 'P', CubicSplineKernel(), 0.1, 0.1, -2, -2, 40, 40)

    assert image[0][0] == 0
    assert image[20][0] == approx(CubicSplineKernel().w(np.sqrt((-1.95) ** 2 + 0.05 ** 2), 2), rel=1e-8)
    assert image[20][20] == approx(CubicSplineKernel().w(np.sqrt(0.05 ** 2 + 0.05 ** 2), 2), rel=1e-8)
    assert image[12][17] == approx(CubicSplineKernel().w(np.sqrt(0.75 ** 2 + 0.25 ** 2), 2), rel=1e-8)


def test_interpolate1dcross():
    df = df = pd.DataFrame({'x': [0],
                            'y': [0],
                            'P': [1],
                            'h': [1],
                            'rho': [1],
                            'm': [1]})
    sdf = SarracenDataFrame(df, params=dict())

    # first, test a cross-section at y=0
    output = interpolate1DCross(sdf, 'x', 'y', 'P', CubicSplineKernel(), -2, 0, 2, 0, 40)

    assert output[0] == approx(CubicSplineKernel().w(np.sqrt((-1.95) ** 2), 2), rel=1e-8)
    assert output[20] == approx(CubicSplineKernel().w(np.sqrt(0.05 ** 2), 2), rel=1e-8)
    assert output[17] == approx(CubicSplineKernel().w(np.sqrt(0.25 ** 2), 2), rel=1e-8)

    # next, test a cross-section where x=y
    output = interpolate1DCross(sdf, 'x', 'y', 'P', CubicSplineKernel(), -2, -2, 2, 2, 40)

    assert output[0] == approx(CubicSplineKernel().w(np.sqrt(2 * (1.95 ** 2)), 2), rel=1e-8)
    assert output[20] == approx(CubicSplineKernel().w(np.sqrt(2 * (0.05 ** 2)), 2), rel=1e-8)
    assert output[17] == approx(CubicSplineKernel().w(np.sqrt(2 * (0.25 ** 2)), 2), rel=1e-8)


def test_interpolate3dcross():
    df = df = pd.DataFrame({'x': [0],
                            'y': [0],
                            'z': [0],
                            'P': [1],
                            'h': [1],
                            'rho': [1],
                            'm': [1]})
    sdf = SarracenDataFrame(df, params=dict())

    # first, test a cross-section at z=0
    image = interpolate3DCross(sdf, 'x', 'y', 'z', 'P', CubicSplineKernel(), 0, 0.1, 0.1, -2, -2, 40, 40)

    # should be exactly the same as for a 2D rendering, except q values are now taken from the 3D kernel.
    assert image[0][0] == 0
    assert image[20][0] == approx(CubicSplineKernel().w(np.sqrt((-1.95) ** 2 + 0.05 ** 2), 3), rel=1e-8)
    assert image[20][20] == approx(CubicSplineKernel().w(np.sqrt(0.05 ** 2 + 0.05 ** 2), 3), rel=1e-8)
    assert image[12][17] == approx(CubicSplineKernel().w(np.sqrt(0.75 ** 2 + 0.25 ** 2), 3), rel=1e-8)

    # next, test a cross-section at z=0.5
    image = interpolate3DCross(sdf, 'x', 'y', 'z', 'P', CubicSplineKernel(), 0.5, 0.1, 0.1, -2, -2, 40, 40)

    assert image[0][0] == 0
    assert image[20][0] == approx(CubicSplineKernel().w(np.sqrt((-1.95) ** 2 + 0.05 ** 2 + (0.5 ** 2)), 3), rel=1e-8)
    assert image[20][20] == approx(CubicSplineKernel().w(np.sqrt(2 * (0.05 ** 2) + (0.5 ** 2)), 3), rel=1e-8)
    assert image[12][17] == approx(CubicSplineKernel().w(np.sqrt(0.75 ** 2 + 0.25 ** 2 + (0.5 ** 2)), 3), rel=1e-8)


def test_interpolate3d():
    df = pd.DataFrame({'x': [0.25],
                       'y': [0.25],
                       'z': [0],
                       'P': [0.5],
                       'h': [0.125],
                       'rho': [0.5],
                       'm': [0.01],
                       'A': [3]})
    sdf = SarracenDataFrame(df, params=dict())
    kernel = QuarticSplineKernel()

    image = interpolate3D(sdf, 'x', 'y', 'A', kernel, 0.05, 0.05, 0, 0, 10, 10, 10000)
    column_kernel = kernel.get_column_kernel(10000)

    # w = 0.01 / (0.5 * 0.125^3) = 10.24
    w = (sdf['m'] / (sdf['rho'] * sdf['h'] ** 3))[0]

    assert image[0][0] == 0
    # 10.24 * 0.125 * 3 * F(sqrt(0.025^2 + 0.225^2)/0.125) ~= 3.84 * 0.000409322579272
    F = np.interp(np.sqrt(0.025 ** 2 + 0.225 ** 2) / df['h'][0], np.linspace(0, kernel.get_radius(), 10000),
                  column_kernel)
    assert image[0][4] == approx(w * sdf['h'][0] * sdf['A'][0] * F)
    # 10.24 * 0.125 * 3 * F(sqrt(0.025^2 + 0.025^2)/0.125) ~= 3.84 * 0.427916515256
    F = np.interp(np.sqrt(2 * (0.025 ** 2)) / df['h'][0], np.linspace(0, kernel.get_radius(), 10000),
                  column_kernel)
    assert image[5][5] == approx(w * sdf['h'][0] * sdf['A'][0] * F)
