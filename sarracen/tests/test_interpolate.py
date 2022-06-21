"""
pytest unit tests for interpolate.py functions.
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pytest import approx

from sarracen import SarracenDataFrame
from sarracen.kernels import CubicSplineKernel, QuarticSplineKernel, QuinticSplineKernel
from sarracen.interpolate import interpolate_2d, interpolate_2d_cross, interpolate_3d_cross, interpolate_3d, \
    interpolate_2d_vec, interpolate_3d_vec, interpolate_3d_cross_vec


def test_interpolate_2d():
    df = pd.DataFrame({'x': [0],
                       'y': [0],
                       'P': [1],
                       'h': [1],
                       'rho': [1],
                       'm': [1]})
    sdf = SarracenDataFrame(df, params=dict())

    image = interpolate_2d(sdf, 'P', 'x', 'y', CubicSplineKernel(), 40, 40, -2, 2, -2, 2)

    assert image[0][0] == 0
    assert image[20][0] == approx(CubicSplineKernel().w(np.sqrt((-1.95) ** 2 + 0.05 ** 2), 2), rel=1e-8)
    assert image[20][20] == approx(CubicSplineKernel().w(np.sqrt(0.05 ** 2 + 0.05 ** 2), 2), rel=1e-8)
    assert image[12][17] == approx(CubicSplineKernel().w(np.sqrt(0.75 ** 2 + 0.25 ** 2), 2), rel=1e-8)

    # next, use a dataset where rho != 0, h != 0, m != 0.
    df = pd.DataFrame({'y': [0],
                       'x': [1],
                       'A': [4],
                       'h': [0.9],
                       'rho': [0.4],
                       'm': [0.03]})
    sdf = SarracenDataFrame(df, params=dict())
    kernel = QuinticSplineKernel()
    w = sdf['m'][0] / (sdf['rho'][0] * sdf['h'] ** 2)

    image = interpolate_2d(sdf, 'A', 'x', 'y', kernel, 10, 15, 0, 2, 0, 3)

    assert image[14][8] == 0
    assert image[14][5] == approx(w * sdf['A'][0] * kernel.w(np.sqrt(0.1 ** 2 + 2.9 ** 2) / sdf['h'][0], 2), rel=1e-8)
    assert image[5][1] == approx(w * sdf['A'][0] * kernel.w(np.sqrt(0.7 ** 2 + 1.1 ** 2) / sdf['h'][0], 2), rel=1e-8)
    assert image[0][4] == approx(w * sdf['A'][0] * kernel.w(np.sqrt(2 * (0.1 ** 2)) / sdf['h'][0], 2), rel=1e-8)
    assert image[0][0] == approx(image[0][9], rel=1e-8)


def test_interpolate_2d_vec():
    df = pd.DataFrame({'x': [0],
                       'y': [0],
                       'Px': [1],
                       'Py': [2],
                       'Pz': [3],
                       'h': [0.35],
                       'rho': [0.3],
                       'm': [0.1]})
    sdf = SarracenDataFrame(df, params=dict())
    w = sdf['m'][0] / (sdf['rho'][0] * sdf['h'] ** 2)
    kernel = QuarticSplineKernel()
    sdf.kernel = kernel

    image = interpolate_2d_vec(sdf, 'Px', 'Py', x_pixels=20, y_pixels=20, x_min=0, y_min=0, x_max=1, y_max=1)

    # X-dimension of vector field
    assert image[0][19][19] == 0
    assert image[0][0][0] == approx(w * sdf['Px'][0] * kernel.w(np.sqrt(0.025 ** 2 + 0.025 ** 2) / sdf['h'][0], 2))
    assert image[0][15][12] == approx(w * sdf['Px'][0] * kernel.w(np.sqrt(0.775 ** 2 + 0.625 ** 2) / sdf['h'][0], 2))

    # Y-dimension of vector field
    assert image[1][19][19] == 0
    assert image[1][0][0] == approx(w * sdf['Py'][0] * kernel.w(np.sqrt(0.025 ** 2 + 0.025 ** 2) / sdf['h'][0], 2))
    assert image[1][15][12] == approx(w * sdf['Py'][0] * kernel.w(np.sqrt(0.775 ** 2 + 0.625 ** 2) / sdf['h'][0], 2))

    # Result of interpolate_2d_vec should be equivalent to the result of interpolate_2d performed on both x & y.
    assert np.array_equal(image[0], interpolate_2d(sdf, 'Px', x_pixels=20, y_pixels=20, x_min=0, y_min=0, x_max=1,
                                                   y_max=1))
    assert np.array_equal(image[1], interpolate_2d(sdf, 'Py', x_pixels=20, y_pixels=20, x_min=0, y_min=0, x_max=1,
                                                   y_max=1))


def test_interpolate_2d_cross():
    df = pd.DataFrame({'x': [0],
                       'y': [0],
                       'P': [1],
                       'h': [1],
                       'rho': [1],
                       'm': [1]})
    sdf = SarracenDataFrame(df, params=dict())

    # first, test a cross-section at y=0
    output = interpolate_2d_cross(sdf, 'P', 'x', 'y', CubicSplineKernel(), 40, -2, 2, 0, 0)

    assert output[0] == approx(CubicSplineKernel().w(np.sqrt((-1.95) ** 2), 2), rel=1e-8)
    assert output[20] == approx(CubicSplineKernel().w(np.sqrt(0.05 ** 2), 2), rel=1e-8)
    assert output[17] == approx(CubicSplineKernel().w(np.sqrt(0.25 ** 2), 2), rel=1e-8)

    # next, test a cross-section where x=y
    output = interpolate_2d_cross(sdf, 'P', 'x', 'y', CubicSplineKernel(), 40, -2, 2, -2, 2)

    assert output[0] == approx(CubicSplineKernel().w(np.sqrt(2 * (1.95 ** 2)), 2), rel=1e-8)
    assert output[20] == approx(CubicSplineKernel().w(np.sqrt(2 * (0.05 ** 2)), 2), rel=1e-8)
    assert output[17] == approx(CubicSplineKernel().w(np.sqrt(2 * (0.25 ** 2)), 2), rel=1e-8)

    # lastly, use a dataset where rho != 0, h != 0, m != 0.
    df = pd.DataFrame({'y': [0],
                       'x': [1],
                       'A': [2.1],
                       'h': [3],
                       'rho': [0.8],
                       'm': [0.05]})
    sdf = SarracenDataFrame(df, params=dict())
    kernel = QuarticSplineKernel()
    w = sdf['m'][0] / (sdf['rho'][0] * sdf['h'][0] ** 2)

    output = interpolate_2d_cross(sdf, 'A', 'x', 'y', kernel, 20, -3, 2, 3, -3)
    # delta_x = 5, delta_y = -6
    # therefore, the change in difference between pixels is dx=5/20, dy=-6/20

    # pixels are evenly spaced across the line, so the first pixel starts at (-3 + 5/40, 3 - 6/40)
    # difference from particle at (1, 0) -> (-3 + 5/40 - 1, 3 - 6/40)
    assert output[0] == approx(w * sdf['A'][0] * kernel.w(np.sqrt(3.875 ** 2 + 2.85 ** 2) / sdf['h'][0], 2), rel=1e-8)
    # (-3.875 + 10 * (5/20), 2.85 - 10 * (6/20))
    assert output[10] == approx(w * sdf['A'][0] * kernel.w(np.sqrt(1.375 ** 2 + 0.15 ** 2) / sdf['h'][0], 2), rel=1e-8)
    # (-3.875 + 19 * (5/20), 2.85 - 19 * (6/20))
    assert output[19] == approx(w * sdf['A'][0] * kernel.w(np.sqrt(0.875 ** 2 + 2.85 ** 2) / sdf['h'][0], 2), rel=1e-8)


def test_interpolate_3d():
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

    image = interpolate_3d(sdf, 'A', 'x', 'y', kernel, 10000, None, None, 10, 10, 0, 0.5, 0, 0.5)
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

    df = pd.DataFrame({'y': [0],
                       'x': [2],
                       'A': [3.1],
                       'h': [1.5],
                       'z': [-0.5],
                       'rho': [0.21],
                       'm': [0.15]})
    sdf = SarracenDataFrame(df, params=dict())
    kernel = QuinticSplineKernel()

    image = interpolate_3d(sdf, 'A', 'x', 'y', kernel, 10000, None, None, 20, 15, 0, 2, 0, 0.9)
    column_kernel = kernel.get_column_kernel(10000)

    # w = 0.15 / (0.21 * 1.5^3) = 0.21164
    w = (sdf['m'] / (sdf['rho'] * sdf['h'] ** 3))[0]

    # r = sqrt(dx ^ 2 + dy ^ 2) = sqrt((2 - (0.1 * 0.5))^2 + (0 + (0.06 * 0.5))^2)
    F = np.interp(np.sqrt(1.95 ** 2 + 0.03 ** 2) / df['h'][0], np.linspace(0, kernel.get_radius(), 10000),
                  column_kernel)
    assert image[0][0] == approx(w * sdf['h'][0] * sdf['A'][0] * F)

    # r = sqrt((2 - (0.1 * 19.5)) ^ 2 + (0 + (0.06 * 0.5)) ^ 2)
    F = np.interp(np.sqrt(0.05 ** 2 + 0.03 ** 2) / df['h'][0], np.linspace(0, kernel.get_radius(), 10000),
                  column_kernel)
    assert image[0][19] == approx(w * sdf['h'][0] * sdf['A'][0] * F)

    # r = sqrt((2 - (0.1 * 14.5)) ^ 2 + (0 + (0.06 * 9.5)) ^ 2)
    F = np.interp(np.sqrt(0.55 ** 2 + 0.57 ** 2) / df['h'][0], np.linspace(0, kernel.get_radius(), 10000),
                  column_kernel)
    assert image[9][14] == approx(w * sdf['h'][0] * sdf['A'][0] * F)

    # Next, use a dataset with two particles.
    df = pd.DataFrame({'y': [0, 0],
                       'x': [0, 0],
                       'A': [4.5, 3.5],
                       'h': [1.6, 1.5],
                       'z': [1, -1],
                       'rho': [0.55, 0.75],
                       'm': [0.3, 0.2]})

    sdf = SarracenDataFrame(df)
    kernel = CubicSplineKernel()

    # With no rotation, both particles are directly on top of each other.
    image = interpolate_3d(sdf, 'A', 'x', 'y', kernel, 10000, None, None, 25, 25, -0.5, 0.5, -0.5, 0.5)
    column_kernel = kernel.get_column_kernel(10000)
    w = (sdf['m'] / (sdf['rho'] * sdf['h'] ** 3))

    F = column_kernel[0]
    assert image[12][12] == approx((w * sdf['h'] * sdf['A'] * F).sum())

    F = np.interp(np.sqrt(0.48 ** 2) / df['h'], np.linspace(0, kernel.get_radius(), 10000),
                  column_kernel)
    assert image[12][0] == approx((w * sdf['h'] * sdf['A'] * F).sum())

    F = np.interp(np.sqrt(0.4 ** 2 + 0.44 ** 2) / df['h'], np.linspace(0, kernel.get_radius(), 10000),
                  column_kernel)
    assert image[1][22] == approx((w * sdf['h'] * sdf['A'] * F).sum())

    # With this rotation, both particles are now at opposite corners, at x & y distances of (1/sqrt(2)) from the centre.
    image = interpolate_3d(sdf, 'A', 'x', 'y', kernel, 10000, [0, 45, 270], None, 25, 25, -0.5, 0.5, -0.5, 0.5)

    # At the centre square, both particles are 1 unit away.
    F = np.interp(1 / df['h'], np.linspace(0, kernel.get_radius(), 10000),
                  column_kernel)
    assert image[12][12] == approx((w * sdf['h'] * sdf['A'] * F).sum())

    # At a grid corner, each particle is 1/sqrt(2) + 0.48 away from the square in one direction, and
    # 1/sqrt(2) - 0.48 away in the other direction.
    F = np.interp(np.sqrt(((1 / np.sqrt(2)) + 0.48) ** 2 + ((1 / np.sqrt(2)) - 0.48) ** 2) / df['h'],
                  np.linspace(0, kernel.get_radius(), 10000), column_kernel)
    assert image[0][24] == approx((w * sdf['h'] * sdf['A'] * F).sum())

    # At the top square of the grid, one particle is closer than the other.
    F1 = np.interp(np.sqrt((1/2) + ((1 / np.sqrt(2)) + 0.48) ** 2) / df['h'][0],
                   np.linspace(0, kernel.get_radius(), 10000), column_kernel)
    F2 = np.interp(np.sqrt((1/2) + ((1 / np.sqrt(2)) - 0.48) ** 2) / df['h'][1],
                   np.linspace(0, kernel.get_radius(), 10000), column_kernel)
    assert image[12][0] == approx(w[0] * sdf['h'][0] * sdf['A'][0] * F1 + w[1] * sdf['h'][1] * sdf['A'][1] * F2)


def test_interpolate_3d_vec():
    df = pd.DataFrame({'x': [0], 'y': [0], 'z': [1], 'Px': [1], 'Py': [2], 'Pz': [3], 'h': [0.35], 'rho': [0.3], 'm': [0.1]})
    sdf = SarracenDataFrame(df, params=dict())
    w = sdf['m'][0] / (sdf['rho'][0] * sdf['h'][0] ** 2)
    kernel = QuarticSplineKernel()
    sdf.kernel = kernel

    image = interpolate_3d_vec(sdf, 'Px', 'Py', 'Pz', integral_samples=1000, x_pixels=20, y_pixels=20, x_min=0, y_min=0,
                               x_max=1, y_max=1)
    column_function = kernel.get_column_kernel_func(1000)

    # X-dimension of vector field
    assert image[0][19][19] == 0
    assert image[0][0][0] == approx(w * sdf['Px'][0] * column_function(np.sqrt(0.025 ** 2 + 0.025 ** 2) / sdf['h'][0], 0))
    assert image[0][15][12] == approx(w * sdf['Px'][0] * column_function(np.sqrt(0.775 ** 2 + 0.625 ** 2) / sdf['h'][0], 0))

    # Y-dimension of vector field
    assert image[1][19][19] == 0
    assert image[1][0][0] == approx(w * sdf['Py'][0] * column_function(np.sqrt(0.025 ** 2 + 0.025 ** 2) / sdf['h'][0], 0))
    assert image[1][15][12] == approx(w * sdf['Py'][0] * column_function(np.sqrt(0.775 ** 2 + 0.625 ** 2) / sdf['h'][0], 0))

    # Result of interpolate_3d_vec should be equivalent to the result of interpolate_3d performed on both x & y.
    assert np.array_equal(image[0], interpolate_3d(sdf, 'Px', integral_samples=1000, x_pixels=20, y_pixels=20, x_min=0,
                                                   y_min=0, x_max=1, y_max=1))
    assert np.array_equal(image[1], interpolate_3d(sdf, 'Py', integral_samples=1000, x_pixels=20, y_pixels=20, x_min=0,
                                                   y_min=0, x_max=1, y_max=1))

    # After this rotation, the particle will be at (1/sqrt(2), 1/sqrt(2), 0)
    # The target vector will be (4/sqrt(2), 2/sqrt(2), -2)
    image = interpolate_3d_vec(sdf, 'Px', 'Py', 'Pz', integral_samples=1000, x_pixels=20, y_pixels=20, x_min=0, y_min=0,
                               x_max=1, y_max=1, rotation=[0, 45, -90], origin=[0, 0, 0])

    # X-dimension of vector field
    assert image[0][0][0] == 0
    assert image[0][19][19] == approx(w * (4 / np.sqrt(2)) * column_function((np.sqrt(2) - 1 - np.sqrt(2) * 0.025) / sdf['h'][0], 0))

    # Y-dimension of vector field
    assert image[1][0][0] == 0
    assert image[1][19][19] == approx(w * (2 / np.sqrt(2)) * column_function((np.sqrt(2) - 1 - np.sqrt(2) * 0.025) / sdf['h'][0], 0))


def test_interpolate_3d_cross():
    df = pd.DataFrame({'x': [0],
                       'y': [0],
                       'z': [0],
                       'P': [1],
                       'h': [1],
                       'rho': [1],
                       'm': [1]})
    sdf = SarracenDataFrame(df, params=dict())

    # first, test a cross-section at z=0
    image = interpolate_3d_cross(sdf, 'P', 0, 'x', 'y', 'z', CubicSplineKernel(), None, None, 40, 40, -2, 2, -2,
                                 2)

    # should be exactly the same as for a 2D rendering, except q values are now taken from the 3D kernel.
    assert image[0][0] == 0
    assert image[20][0] == approx(CubicSplineKernel().w(np.sqrt((-1.95) ** 2 + 0.05 ** 2), 3), rel=1e-8)
    assert image[20][20] == approx(CubicSplineKernel().w(np.sqrt(0.05 ** 2 + 0.05 ** 2), 3), rel=1e-8)
    assert image[12][17] == approx(CubicSplineKernel().w(np.sqrt(0.75 ** 2 + 0.25 ** 2), 3), rel=1e-8)

    # next, test a cross-section at z=0.5
    image = interpolate_3d_cross(sdf, 'P', 0.5, 'x', 'y', 'z', CubicSplineKernel(), None, None, 40, 40, -2, 2, -2,
                                 2)

    assert image[0][0] == 0
    assert image[20][0] == approx(CubicSplineKernel().w(np.sqrt((-1.95) ** 2 + 0.05 ** 2 + (0.5 ** 2)), 3), rel=1e-8)
    assert image[20][20] == approx(CubicSplineKernel().w(np.sqrt(2 * (0.05 ** 2) + (0.5 ** 2)), 3), rel=1e-8)
    assert image[12][17] == approx(CubicSplineKernel().w(np.sqrt(0.75 ** 2 + 0.25 ** 2 + (0.5 ** 2)), 3), rel=1e-8)

    # next, use a dataset where rho != 0, h != 0, m != 0.
    df = pd.DataFrame({'x': [0],
                            'y': [0],
                            'z': [-1],
                            'A': [4],
                            'h': [2],
                            'rho': [0.5],
                            'm': [0.1]})
    sdf = SarracenDataFrame(df, params=dict())

    w = sdf['m'][0] / (sdf['rho'][0] * sdf['h'][0] ** 3)
    kernel = QuarticSplineKernel()
    image = interpolate_3d_cross(sdf, 'A', 0, 'x', 'y', 'z', kernel, None, None, 15, 11, -0.75, 0.75, -0.825,
                                 0.825)

    # r = sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    assert image[0][0] == approx(w * sdf['A'][0] * kernel.w(np.sqrt(0.7 ** 2 + 0.75 ** 2 + 1) / sdf['h'][0], 3))
    assert image[4][0] == approx(w * sdf['A'][0] * kernel.w(np.sqrt(0.7 ** 2 + 0.15 ** 2 + 1) / sdf['h'][0], 3))
    assert image[7][10] == approx(w * sdf['A'][0] * kernel.w(np.sqrt(0.3 ** 2 + 0.3 ** 2 + 1) / sdf['h'][0], 3))

    # now, use a dataset with multiple particles.
    df = pd.DataFrame({'y': [0, 0],
                       'x': [0, 0],
                       'A': [4.5, 3.5],
                       'h': [1.6, 1.5],
                       'z': [1, -1],
                       'rho': [0.55, 0.75],
                       'm': [0.3, 0.2]})

    sdf = SarracenDataFrame(df)
    kernel = QuarticSplineKernel()
    w = (sdf['m'] / (sdf['rho'] * sdf['h'] ** 3))

    # With no rotation, both particles are directly on top of each other.
    image = interpolate_3d_cross(sdf, 'A', 0, 'x', 'y', 'z', kernel, None, None, 25, 25, -0.5, 0.5, -0.5, 0.5)

    # Each particle is one unit away from the centre point in the z-direction.
    assert image[12][12] == approx((w * sdf['A'] * kernel.w(1 / sdf['h'], 3)).sum())

    F = kernel.w(np.sqrt(2 * (0.48 ** 2) + 1) / sdf['h'], 3)
    assert image[24][0] == approx((w * sdf['A'] * F).sum())

    # With this rotation, both particles are now at opposite corners, at x & y distances of (1/sqrt(2)) from the centre,
    # and a z-value of z=0.
    image = interpolate_3d_cross(sdf, 'A', 0, 'x', 'y', 'z', kernel, [0, 45, 270], None, 25, 25, -0.5, 0.5, -0.5, 0.5)

    # Same as above (no rotation), since the particles are still 1 unit away from the centre point.
    assert image[12][12] == approx((w * sdf['A'] * kernel.w(1 / sdf['h'], 3)).sum())

    # At a grid corner, each particle is 1/sqrt(2) + 0.48 away from the square in one direction, and
    # 1/sqrt(2) - 0.48 away in the other direction.
    F = kernel.w(np.sqrt(((1 / np.sqrt(2)) + 0.48) ** 2 + ((1 / np.sqrt(2)) - 0.48) ** 2) / df['h'], 3)
    assert image[0][24] == approx((w * sdf['A'] * F).sum())

    # At the top grid square, each particle is at a different distance.
    F1 = kernel.w(np.sqrt((1/2) + ((1 / np.sqrt(2)) + 0.48) ** 2) / df['h'][0], 3)
    F2 = kernel.w(np.sqrt((1/2) + ((1 / np.sqrt(2)) - 0.48) ** 2) / df['h'][1], 3)
    assert image[12][0] == approx(w[0] * sdf['A'][0] * F1 + w[1] * sdf['A'][1] * F2)


def test_interpolate_3d_cross_vec():
    df = pd.DataFrame({'x': [0], 'y': [0], 'z': [1], 'Px': [1], 'Py': [2], 'Pz': [3], 'h': [0.35], 'rho': [0.3], 'm': [0.1]})
    sdf = SarracenDataFrame(df, params=dict())
    w = sdf['m'][0] / (sdf['rho'][0] * sdf['h'][0] ** 3)
    kernel = QuarticSplineKernel()
    sdf.kernel = kernel

    image = interpolate_3d_cross_vec(sdf, 'Px', 'Py', 'Pz', z_slice=0, x_pixels=20, y_pixels=20, x_min=0, y_min=0,
                                     x_max=1, y_max=1)

    # X-dimension of vector field
    assert image[0][19][19] == 0
    assert image[0][0][0] == approx(w * sdf['Px'][0] * kernel.w(np.sqrt(0.025 ** 2 + 0.025 ** 2 + 1) / sdf['h'][0], 3))
    assert image[0][15][12] == approx(w * sdf['Px'][0] * kernel.w(np.sqrt(0.775 ** 2 + 0.625 ** 2 + 1) / sdf['h'][0], 3))

    # Y-dimension of vector field
    assert image[1][19][19] == 0
    assert image[1][0][0] == approx(w * sdf['Py'][0] * kernel.w(np.sqrt(0.025 ** 2 + 0.025 ** 2 + 1) / sdf['h'][0], 3))
    assert image[1][15][12] == approx(
        w * sdf['Py'][0] * kernel.w(np.sqrt(0.775 ** 2 + 0.625 ** 2 + 1) / sdf['h'][0], 3))

    # Result of interpolate_3d_cross_vec should be equivalent to the result of interpolate_3d_cross performed on both x & y.
    assert np.array_equal(image[0], interpolate_3d_cross(sdf, 'Px', z_slice=0, x_pixels=20, y_pixels=20, x_min=0,
                                                   y_min=0, x_max=1, y_max=1))
    assert np.array_equal(image[1], interpolate_3d_cross(sdf, 'Py', z_slice=0, x_pixels=20,
                                                         y_pixels=20, x_min=0, y_min=0, x_max=1, y_max=1))

    # After this rotation, the particle will be at (1/sqrt(2), 1/sqrt(2), 0)
    # The target vector will be (4/sqrt(2), 2/sqrt(2), -2)
    image = interpolate_3d_cross_vec(sdf, 'Px', 'Py', 'Pz', z_slice=0, x_pixels=20, y_pixels=20, x_min=0, y_min=0,
                               x_max=1, y_max=1, rotation=[0, 45, -90], origin=[0, 0, 0])

    # X-dimension of vector field
    assert image[0][0][0] == 0
    assert image[0][19][19] == approx(
        w * (4 / np.sqrt(2)) * kernel.w((np.sqrt(2) - 1 - np.sqrt(2) * 0.025) / sdf['h'][0], 3))

    # Y-dimension of vector field
    assert image[1][0][0] == 0
    assert image[1][19][19] == approx(
        w * (2 / np.sqrt(2)) * kernel.w((np.sqrt(2) - 1 - np.sqrt(2) * 0.025) / sdf['h'][0], 3))


def test_race_conditions():
    df = pd.concat([pd.DataFrame({'x': [0],
                       'y': [0],
                       'z': [-1],
                       'A': [4],
                       'h': [2],
                       'rho': [0.5],
                       'm': [0.1]})] * 10000, ignore_index=True)
    sdf = SarracenDataFrame(df, params=dict())

    w = sdf['m'][0] / (sdf['rho'][0] * sdf['h'][0] ** 3)
    kernel = QuarticSplineKernel()
    image = interpolate_3d(sdf, 'A', 'x', 'y', kernel, 1000, None, None, 15, 15, -0.5, 0.5, -0.5, 0.5)
    column_kernel = kernel.get_column_kernel(1000)

    assert image[7][7] == approx(w * sdf['h'][0] * sdf['A'][0] * column_kernel[0] * 10000, rel=1e-9)
