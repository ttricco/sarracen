"""
pytest unit tests for interpolate.py functions.
"""
import pandas as pd
import numpy as np
from numpy.testing import assert_allclose
from pytest import approx, raises
from scipy.spatial.transform import Rotation

from sarracen import SarracenDataFrame
from sarracen.kernels import CubicSplineKernel, QuarticSplineKernel, QuinticSplineKernel
from sarracen.interpolate import interpolate_2d, interpolate_2d_cross, interpolate_3d_cross, interpolate_3d


def test_single_particle():
    df = pd.DataFrame({'x': [0], 'y': [0], 'A': [4], 'h': [0.9], 'rho': [0.4], 'm': [0.03]})
    sdf = SarracenDataFrame(df, params=dict())
    kernel = CubicSplineKernel()
    sdf.kernel = kernel
    w = sdf['m'] / (sdf['rho'] * sdf['h'] ** 2)

    # A mapping of pixel indices to x / y values in particle space.
    real = -kernel.get_radius() + (np.arange(0, 25) + 0.5) * (2 * kernel.get_radius() / 25)

    image = interpolate_2d(sdf, 'A', x_pixels=25,  y_pixels=25, x_min=-kernel.get_radius(), x_max=kernel.get_radius(),
                           y_min=-kernel.get_radius(), y_max=kernel.get_radius())

    for y in range(25):
        for x in range(25):
            assert image[y][x] == approx(w[0] * sdf['A'][0] * kernel.w(np.sqrt(real[x] ** 2 + real[y] ** 2) / sdf['h'][0], 2))

    image = interpolate_2d_cross(sdf, 'A', pixels=25, x1=-kernel.get_radius(), x2=kernel.get_radius(),
                                 y1=-kernel.get_radius(), y2=kernel.get_radius())

    for x in range(25):
        assert image[x] == approx(w[0] * sdf['A'][0] * kernel.w(np.sqrt(2) * np.abs(real[x]) / sdf['h'][0], 2))

    sdf['z'] = -0.5
    sdf.zcol = 'z'

    image = interpolate_3d(sdf, 'A', x_pixels=25, y_pixels=25, x_min=-kernel.get_radius(), x_max=kernel.get_radius(),
                           y_min=-kernel.get_radius(), y_max=kernel.get_radius())
    column_func = kernel.get_column_kernel_func(1000)

    for y in range(25):
        for x in range(25):
            assert image[y][x] == approx(w[0] * sdf['A'][0] * column_func(np.sqrt(real[x] ** 2 + real[y] ** 2) / sdf['h'][0], 2))

    w = sdf['m'] / (sdf['rho'] * sdf['h'] ** 3)
    image = interpolate_3d_cross(sdf, 'A', 0, x_pixels=25, y_pixels=25, x_min=-kernel.get_radius(),
                                 x_max=kernel.get_radius(), y_min=-kernel.get_radius(), y_max=kernel.get_radius())

    for y in range(25):
        for x in range(25):
            assert image[y][x] == approx(w[0] * sdf['A'][0] * kernel.w(np.sqrt(real[x] ** 2 + real[y] ** 2 + 0.5 ** 2) / sdf['h'][0], 3))


def test_single_repeated_particle():
    repetitions = 10000

    df = pd.concat([pd.DataFrame({'x': [0], 'y': [0], 'A': [4], 'h': [0.9], 'rho': [0.4], 'm': [0.03]})] * repetitions,
                   ignore_index=True)
    sdf = SarracenDataFrame(df, params=dict())
    kernel = CubicSplineKernel()
    sdf.kernel = kernel
    w = sdf['m'] / (sdf['rho'] * sdf['h'] ** 2)

    # A mapping of pixel indices to x / y values in particle space.
    real = -kernel.get_radius() + (np.arange(0, 25) + 0.5) * (2 * kernel.get_radius() / 25)

    image = interpolate_2d(sdf, 'A', x_pixels=25,  y_pixels=25, x_min=-kernel.get_radius(), x_max=kernel.get_radius(),
                           y_min=-kernel.get_radius(), y_max=kernel.get_radius())

    for y in range(25):
        for x in range(25):
            assert image[y][x] == approx(repetitions * w[0] * sdf['A'][0] * kernel.w(np.sqrt(real[x] ** 2 + real[y] ** 2) / sdf['h'][0], 2))

    image = interpolate_2d_cross(sdf, 'A', pixels=25, x1=-kernel.get_radius(), x2=kernel.get_radius(),
                                 y1=-kernel.get_radius(), y2=kernel.get_radius())

    for x in range(25):
        assert image[x] == approx(repetitions * w[0] * sdf['A'][0] * kernel.w(np.sqrt(2) * np.abs(real[x]) / sdf['h'][0], 2))

    sdf['z'] = -0.5
    sdf.zcol = 'z'

    image = interpolate_3d(sdf, 'A', x_pixels=25, y_pixels=25, x_min=-kernel.get_radius(), x_max=kernel.get_radius(),
                           y_min=-kernel.get_radius(), y_max=kernel.get_radius())
    column_func = kernel.get_column_kernel_func(1000)

    for y in range(25):
        for x in range(25):
            assert image[y][x] == approx(repetitions * w[0] * sdf['A'][0] * column_func(np.sqrt(real[x] ** 2 + real[y] ** 2) / sdf['h'][0], 2))

    w = sdf['m'] / (sdf['rho'] * sdf['h'] ** 3)
    image = interpolate_3d_cross(sdf, 'A', 0, x_pixels=25, y_pixels=25, x_min=-kernel.get_radius(),
                                 x_max=kernel.get_radius(), y_min=-kernel.get_radius(), y_max=kernel.get_radius())

    for y in range(25):
        for x in range(25):
            assert image[y][x] == approx(repetitions * w[0] * sdf['A'][0] * kernel.w(np.sqrt(real[x] ** 2 + real[y] ** 2 + 0.5 ** 2) / sdf['h'][0], 3))


def test_dimension_check():
    df = pd.DataFrame({'x': [0, 1], 'y': [0, 1], 'P': [1, 1], 'h': [1, 1], 'rho': [1, 1], 'm': [1, 1]})
    sdf = SarracenDataFrame(df, params=dict())

    with raises(TypeError):
        interpolate_3d(sdf, 'P')

    with raises(TypeError):
        interpolate_3d_cross(sdf, 'P', 0.5)

    df = pd.DataFrame({'x': [0, 1], 'y': [0, 1], 'z': [0, 1], 'P': [1, 1], 'h': [1, 1], 'rho': [1, 1], 'm': [1, 1]})
    sdf = SarracenDataFrame(df, params=dict())

    with raises(TypeError):
        interpolate_2d(sdf, 'P')

    with raises(TypeError):
        interpolate_2d_cross(sdf, 'P')


def test_3d_xsec_equivalency():
    df = pd.DataFrame({'x': [0], 'y': [0], 'z': [0], 'A': [4], 'h': [0.9], 'rho': [0.4], 'm': [0.03]})
    sdf = SarracenDataFrame(df, params=dict())
    kernel = CubicSplineKernel()
    sdf.kernel = kernel

    column_image = interpolate_3d(sdf, 'A', x_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1)

    xsec_image = np.zeros((50, 50))
    samples = 10000
    for z in np.linspace(-kernel.get_radius() * sdf['h'][0], kernel.get_radius() * sdf['h'][0], samples):
        xsec_image += interpolate_3d_cross(sdf, 'A', z, x_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1)
    xsec_image *= kernel.get_radius() * sdf['h'][0] * 2 / samples

    assert_allclose(xsec_image, column_image, rtol=1e-4, atol=1e-5)


def test_2d_xsec_equivalency():
    df = pd.DataFrame({'x': [0], 'y': [0], 'A': [4], 'h': [0.9], 'rho': [0.4], 'm': [0.03]})
    sdf = SarracenDataFrame(df, params=dict())
    kernel = CubicSplineKernel()
    sdf.kernel = kernel

    true_image = interpolate_2d(sdf, 'A', x_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1)

    # A mapping of pixel indices to x & y values in particle space.
    real = -1 + (np.arange(0, 50) + 0.5) * (2 / 50)

    reconstructed_image = np.zeros((50, 50))
    for y in range(50):
        reconstructed_image[y, :] = interpolate_2d_cross(sdf, 'A', pixels=50, x1=-1, x2=1, y1=real[y], y2=real[y])

    assert_allclose(reconstructed_image, true_image)


def test_corner_particles():
    df = pd.DataFrame({'x': [-1, 1], 'y': [-1, 1], 'A': [2, 1.5], 'h': [1.1, 1.3], 'rho': [0.55, 0.45], 'm': [0.04, 0.05]})
    sdf = SarracenDataFrame(df, params=dict())
    kernel = CubicSplineKernel()
    sdf.kernel = kernel

    w = sdf['m'] / (sdf['rho'] * sdf['h'] ** 2)

    # A mapping of pixel indices to x / y values in particle space.
    real = (np.arange(0, 25) + 0.5) * (2 / 25)

    image = interpolate_2d(sdf, 'A', x_pixels=25,  y_pixels=25, x_min=-1, x_max=1,
                           y_min=-1, y_max=1)

    for y in range(25):
        for x in range(25):
            assert approx(w[0] * sdf['A'][0] * kernel.w(np.sqrt(real[x] ** 2 + real[y] ** 2) / sdf['h'][0], 2)
                   + w[1] * sdf['A'][1] * kernel.w(np.sqrt(real[24 - x] ** 2 + real[24 - y] ** 2) / sdf['h'][1], 2)) == image[y][x]


def test_image_transpose():
    df = pd.DataFrame({'x': [-1, 1], 'y': [1, -1], 'A': [2, 1.5], 'h': [1.1, 1.3], 'rho': [0.55, 0.45], 'm': [0.04, 0.05]})
    sdf = SarracenDataFrame(df, params=dict())

    image1 = interpolate_2d(sdf, 'A', x_pixels=50,  y_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1)
    image2 = interpolate_2d(sdf, 'A', x='y', y='x', x_pixels=50,  y_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1)

    assert_allclose(image1, image2.T)

    df = pd.DataFrame({'x': [-1, 1], 'y': [1, -1], 'z': [-1, 1], 'A': [2, 1.5], 'h': [1.1, 1.3], 'rho': [0.55, 0.45], 'm': [0.04, 0.05]})
    sdf = SarracenDataFrame(df, params=dict())

    image1 = interpolate_3d(sdf, 'A', x_pixels=50,  y_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1)
    image2 = interpolate_3d(sdf, 'A', x='y', y='x', x_pixels=50,  y_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1)

    assert_allclose(image1, image2.T)

    image1 = interpolate_3d_cross(sdf, 'A', 0, x_pixels=50,  y_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1)
    image2 = interpolate_3d_cross(sdf, 'A', 0, x='y', y='x', x_pixels=50,  y_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1)

    assert_allclose(image1, image2.T)


def test_default_kernel():
    df_2 = pd.DataFrame({'x': [0], 'y': [0], 'A': [1], 'h': [1], 'rho': [1], 'm': [1]})
    sdf_2 = SarracenDataFrame(df_2, params=dict())
    df_3 = pd.DataFrame({'x': [0], 'y': [0], 'z': [0], 'A': [1], 'h': [1], 'rho': [1], 'm': [1]})
    sdf_3 = SarracenDataFrame(df_3, params=dict())

    kernel = QuarticSplineKernel()
    sdf_2.kernel = kernel
    sdf_3.kernel = kernel

    w = sdf_2['m'] / (sdf_2['rho'] * sdf_2['h'] ** 2)

    image = interpolate_2d(sdf_2, 'A', x_pixels=1, y_pixels=1, x_min=-1, x_max=1, y_min=-1, y_max=1)
    assert image == kernel.w(0, 2)

    image = interpolate_2d_cross(sdf_2, 'A', pixels=1, x1=-1, x2=1, y1=-1, y2=1)
    assert image == kernel.w(0, 2)

    image = interpolate_3d(sdf_3, 'A', x_pixels=1, y_pixels=1, x_min=-1, x_max=1, y_min=-1, y_max=1)
    assert image == kernel.get_column_kernel()[0]

    image = interpolate_3d_cross(sdf_3, 'A', x_pixels=1, y_pixels=1, x_min=-1, x_max=1, y_min=-1, y_max=1)
    assert image == kernel.w(0, 3)

    kernel = QuinticSplineKernel()
    image = interpolate_2d(sdf_2, 'A', x_pixels=1, y_pixels=1, x_min=-1, x_max=1, y_min=-1, y_max=1, kernel=kernel)
    assert image == kernel.w(0, 2)\

    image = interpolate_2d_cross(sdf_2, 'A', pixels=1, x1=-1, x2=1, y1=-1, y2=1, kernel=kernel)
    assert image == kernel.w(0, 2)

    image = interpolate_3d(sdf_3, 'A', x_pixels=1, y_pixels=1, x_min=-1, x_max=1, y_min=-1, y_max=1, kernel=kernel)
    assert image == kernel.get_column_kernel_func(1000)(0, 0)

    image = interpolate_3d_cross(sdf_3, 'A', x_pixels=1, y_pixels=1, x_min=-1, x_max=1, y_min=-1, y_max=1, kernel=kernel)
    assert image == kernel.w(0, 3)


def test_column_samples():
    df_3 = pd.DataFrame({'x': [0], 'y': [0], 'z': [0], 'A': [1], 'h': [1], 'rho': [1], 'm': [1]})
    sdf_3 = SarracenDataFrame(df_3, params=dict())
    kernel = QuinticSplineKernel()
    sdf_3.kernel = kernel

    image = interpolate_3d(sdf_3, 'A', x_pixels=1, y_pixels=1, x_min=-1, x_max=1, y_min=-1, y_max=1, integral_samples=2)
    assert image == kernel.get_column_kernel(2)[0]


def test_pixel_arguments():
    df_2 = pd.DataFrame({'x': [-2, 4], 'y': [3, 8], 'A': [1, 1], 'h': [1, 1], 'rho': [1, 1], 'm': [1, 1]})
    sdf_2 = SarracenDataFrame(df_2, params=dict())
    df_3 = pd.DataFrame({'x': [-2, 4], 'y': [3, 8], 'z': [7, -2], 'A': [1, 1], 'h': [1, 1], 'rho': [1, 1], 'm': [1, 1]})
    sdf_3 = SarracenDataFrame(df_3, params=dict())

    ratio_xy = (df_2['x'][1] - df_2['x'][0]) / (df_2['y'][1] - df_2['y'][0])
    ratio_xz = (df_2['x'][1] - df_2['x'][0]) / (df_3['z'][0] - df_3['z'][1])
    ratio_yz = (df_2['y'][1] - df_2['y'][0]) / (df_3['z'][0] - df_3['z'][1])

    # Basic aspect ratio tests
    image = interpolate_2d(sdf_2, 'A')
    assert image.shape[0] / image.shape[1] == approx(1 / ratio_xy, rel=1e-2)

    image = interpolate_3d(sdf_3, 'A')
    assert image.shape[0] / image.shape[1] == approx(1 / ratio_xy, rel=1e-2)

    image = interpolate_3d_cross(sdf_3, 'A', 0)
    assert image.shape[0] / image.shape[1] == approx(1 / ratio_xy, rel=1e-2)

    # Swapped axes
    image = interpolate_2d(sdf_2, 'A', x='y', y='x')
    assert image.shape[0] / image.shape[1] == approx(ratio_xy, rel=1e-2)

    image = interpolate_3d(sdf_3, 'A', x='y', y='x')
    assert image.shape[0] / image.shape[1] == approx(ratio_xy, rel=1e-2)

    image = interpolate_3d(sdf_3, 'A', y='z')
    assert image.shape[0] / image.shape[1] == approx(1 / ratio_xz, rel=1e-2)

    image = interpolate_3d(sdf_3, 'A', x='z', y='x')
    assert image.shape[0] / image.shape[1] == approx(ratio_xz, rel=1e-2)

    image = interpolate_3d(sdf_3, 'A', x='y', y='z')
    assert image.shape[0] / image.shape[1] == approx(1 / ratio_yz, rel=1e-2)

    image = interpolate_3d(sdf_3, 'A', x='z', y='y')
    assert image.shape[0] / image.shape[1] == approx(ratio_yz, rel=1e-2)

    image = interpolate_3d_cross(sdf_3, 'A', 0, x='y', y='x')
    assert image.shape[0] / image.shape[1] == approx(ratio_xy, rel=1e-2)

    image = interpolate_3d_cross(sdf_3, 'A', 0, y='z')
    assert image.shape[0] / image.shape[1] == approx(1 / ratio_xz, rel=1e-2)

    image = interpolate_3d_cross(sdf_3, 'A', 0, x='z', y='x')
    assert image.shape[0] / image.shape[1] == approx(ratio_xz, rel=1e-2)

    image = interpolate_3d_cross(sdf_3, 'A', 0, x='y', y='z')
    assert image.shape[0] / image.shape[1] == approx(1 / ratio_yz, rel=1e-2)

    image = interpolate_3d_cross(sdf_3, 'A', 0, x='z', y='y')
    assert image.shape[0] / image.shape[1] == approx(ratio_yz, rel=1e-2)

    default_pixels = 20
    # One defined axis
    image = interpolate_2d(sdf_2, 'A', x_pixels=default_pixels)
    assert image.shape == (round(default_pixels / ratio_xy), default_pixels)

    image = interpolate_3d(sdf_3, 'A', x_pixels=default_pixels)
    assert image.shape == (round(default_pixels / ratio_xy), default_pixels)

    image = interpolate_3d_cross(sdf_3, 'A', x_pixels=default_pixels)
    assert image.shape == (round(default_pixels / ratio_xy), default_pixels)

    image = interpolate_2d(sdf_2, 'A', y_pixels=default_pixels)
    assert image.shape == (default_pixels, round(default_pixels * ratio_xy))

    image = interpolate_3d(sdf_3, 'A', y_pixels=default_pixels)
    assert image.shape == (default_pixels, round(default_pixels * ratio_xy))

    image = interpolate_3d_cross(sdf_3, 'A', y_pixels=default_pixels)
    assert image.shape == (default_pixels, round(default_pixels * ratio_xy))

    # One defined axis + swapped axes
    image = interpolate_2d(sdf_2, 'A', x='y', y='x', x_pixels=default_pixels)
    assert image.shape == (round(default_pixels * ratio_xy), default_pixels)

    image = interpolate_3d(sdf_3, 'A', x='y', y='x', x_pixels=default_pixels)
    assert image.shape == (round(default_pixels * ratio_xy), default_pixels)

    image = interpolate_3d(sdf_3, 'A', y='z', x_pixels=default_pixels)
    assert image.shape == (round(default_pixels / ratio_xz), default_pixels)

    image = interpolate_3d(sdf_3, 'A', x='z', y='x', x_pixels=default_pixels)
    assert image.shape == (round(default_pixels * ratio_xz), default_pixels)

    image = interpolate_3d(sdf_3, 'A', x='y', y='z', x_pixels=default_pixels)
    assert image.shape == (round(default_pixels / ratio_yz), default_pixels)

    image = interpolate_3d(sdf_3, 'A', x='z', y='y', x_pixels=default_pixels)
    assert image.shape == (round(default_pixels * ratio_yz), default_pixels)

    image = interpolate_3d_cross(sdf_3, 'A', 0, x='y', y='x', x_pixels=default_pixels)
    assert image.shape == (round(default_pixels * ratio_xy), default_pixels)

    image = interpolate_3d_cross(sdf_3, 'A', 0, y='z', x_pixels=default_pixels)
    assert image.shape == (round(default_pixels / ratio_xz), default_pixels)

    image = interpolate_3d_cross(sdf_3, 'A', 0, x='z', y='x', x_pixels=default_pixels)
    assert image.shape == (round(default_pixels * ratio_xz), default_pixels)

    image = interpolate_3d_cross(sdf_3, 'A', 0, x='y', y='z', x_pixels=default_pixels)
    assert image.shape == (round(default_pixels / ratio_yz), default_pixels)

    image = interpolate_3d_cross(sdf_3, 'A', 0, x='z', y='y', x_pixels=default_pixels)
    assert image.shape == (round(default_pixels * ratio_yz), default_pixels)

    image = interpolate_2d(sdf_2, 'A', x='y', y='x', y_pixels=default_pixels)
    assert image.shape == (default_pixels, round(default_pixels / ratio_xy))

    image = interpolate_3d(sdf_3, 'A', x='y', y='x', y_pixels=default_pixels)
    assert image.shape == (default_pixels, round(default_pixels / ratio_xy))

    image = interpolate_3d(sdf_3, 'A', y='z', y_pixels=default_pixels)
    assert image.shape == (default_pixels, round(default_pixels * ratio_xz))

    image = interpolate_3d(sdf_3, 'A', x='z', y='x', y_pixels=default_pixels)
    assert image.shape == (default_pixels, round(default_pixels / ratio_xz))

    image = interpolate_3d(sdf_3, 'A', x='y', y='z', y_pixels=default_pixels)
    assert image.shape == (default_pixels, round(default_pixels * ratio_yz))

    image = interpolate_3d(sdf_3, 'A', x='z', y='y', y_pixels=default_pixels)
    assert image.shape == (default_pixels, round(default_pixels / ratio_yz))

    image = interpolate_3d_cross(sdf_3, 'A', 0, x='y', y='x', y_pixels=default_pixels)
    assert image.shape == (default_pixels, round(default_pixels / ratio_xy))

    image = interpolate_3d_cross(sdf_3, 'A', 0, y='z', y_pixels=default_pixels)
    assert image.shape == (default_pixels, round(default_pixels * ratio_xz))

    image = interpolate_3d_cross(sdf_3, 'A', 0, x='z', y='x', y_pixels=default_pixels)
    assert image.shape == (default_pixels, round(default_pixels / ratio_xz))

    image = interpolate_3d_cross(sdf_3, 'A', 0, x='y', y='z', y_pixels=default_pixels)
    assert image.shape == (default_pixels, round(default_pixels * ratio_yz))

    image = interpolate_3d_cross(sdf_3, 'A', 0, x='z', y='y', y_pixels=default_pixels)
    assert image.shape == (default_pixels, round(default_pixels / ratio_yz))

    x_pixels, y_pixels = 20, 30
    # Two defined axes
    image = interpolate_2d(sdf_2, 'A', x_pixels=x_pixels, y_pixels=y_pixels)
    assert image.shape == (y_pixels, x_pixels)

    image = interpolate_3d(sdf_3, 'A', x_pixels=x_pixels, y_pixels=y_pixels)
    assert image.shape == (y_pixels, x_pixels)

    image = interpolate_3d_cross(sdf_3, 'A', 0, x_pixels=x_pixels, y_pixels=y_pixels)
    assert image.shape == (y_pixels, x_pixels)


def test_irregular_bounds():
    df = pd.DataFrame({'x': [0], 'y': [0], 'A': [4], 'h': [0.9], 'rho': [0.4], 'm': [0.03]})
    sdf = SarracenDataFrame(df, params=dict())
    kernel = CubicSplineKernel()
    sdf.kernel = kernel
    w = sdf['m'] / (sdf['rho'] * sdf['h'] ** 2)

    # A mapping of pixel indices to x / y values in particle space.
    real_x = -kernel.get_radius() + (np.arange(0, 50) + 0.5) * (2 * kernel.get_radius() / 50)
    real_y = -kernel.get_radius() + (np.arange(0, 25) + 0.5) * (2 * kernel.get_radius() / 25)

    image = interpolate_2d(sdf, 'A', x_pixels=50, y_pixels=25, x_min=-kernel.get_radius(), x_max=kernel.get_radius(),
                           y_min=-kernel.get_radius(), y_max=kernel.get_radius())

    for y in range(25):
        for x in range(50):
            assert image[y][x] == approx(
                w[0] * sdf['A'][0] * kernel.w(np.sqrt(real_x[x] ** 2 + real_y[y] ** 2) / sdf['h'][0], 2))

    sdf['z'] = -0.5
    sdf.zcol = 'z'

    image = interpolate_3d(sdf, 'A', x_pixels=50, y_pixels=25, x_min=-kernel.get_radius(), x_max=kernel.get_radius(),
                           y_min=-kernel.get_radius(), y_max=kernel.get_radius())
    column_func = kernel.get_column_kernel_func(1000)

    for y in range(25):
        for x in range(50):
            assert image[y][x] == approx(
                w[0] * sdf['A'][0] * column_func(np.sqrt(real_x[x] ** 2 + real_y[y] ** 2) / sdf['h'][0], 2))

    w = sdf['m'] / (sdf['rho'] * sdf['h'] ** 3)
    image = interpolate_3d_cross(sdf, 'A', 0, x_pixels=50, y_pixels=25, x_min=-kernel.get_radius(),
                                 x_max=kernel.get_radius(), y_min=-kernel.get_radius(), y_max=kernel.get_radius())

    for y in range(25):
        for x in range(50):
            assert image[y][x] == approx(
                w[0] * sdf['A'][0] * kernel.w(np.sqrt(real_x[x] ** 2 + real_y[y] ** 2 + 0.5 ** 2) / sdf['h'][0], 3))


def test_oob_particles():
    df_2 = pd.DataFrame({'x': [0], 'y': [0], 'A': [4], 'h': [0.9], 'rho': [0.4], 'm': [0.03]})
    sdf_2 = SarracenDataFrame(df_2, params=dict())
    df_3 = pd.DataFrame({'x': [0], 'y': [0], 'z': [-0.5], 'A': [4], 'h': [0.9], 'rho': [0.4], 'm': [0.03]})
    sdf_3 = SarracenDataFrame(df_3, params=dict())

    kernel = CubicSplineKernel()
    sdf_2.kernel = kernel
    sdf_3.kernel = kernel
    w = sdf_2['m'] / (sdf_2['rho'] * sdf_2['h'] ** 2)

    # A mapping of pixel indices to x / y values in particle space.
    real_x = 1 + (np.arange(0, 25) + 0.5) * (1 / 25)
    real_y = 1 + (np.arange(0, 25) + 0.5) * (1 / 25)

    image = interpolate_2d(sdf_2, 'A', x_pixels=25, y_pixels=25, x_min=1, x_max=2, y_min=1, y_max=2)

    for y in range(25):
        for x in range(25):
            assert image[y][x] == approx(
                w[0] * sdf_2['A'][0] * kernel.w(np.sqrt(real_x[x] ** 2 + real_y[y] ** 2) / sdf_2['h'][0], 2))

    image = interpolate_3d(sdf_3, 'A', x_pixels=25, y_pixels=25, x_min=1, x_max=2, y_min=1, y_max=2)
    column_func = kernel.get_column_kernel_func(1000)

    for y in range(25):
        for x in range(25):
            assert image[y][x] == approx(
                w[0] * sdf_3['A'][0] * column_func(np.sqrt(real_x[x] ** 2 + real_y[y] ** 2) / sdf_3['h'][0], 2))

    w = sdf_3['m'] / (sdf_3['rho'] * sdf_3['h'] ** 3)
    image = interpolate_3d_cross(sdf_3, 'A', 0, x_pixels=25, y_pixels=25, x_min=1, x_max=2, y_min=1, y_max=2)

    for y in range(25):
        for x in range(25):
            assert image[y][x] == approx(
                w[0] * sdf_3['A'][0] * kernel.w(np.sqrt(real_x[x] ** 2 + real_y[y] ** 2 + 0.5 ** 2) / sdf_3['h'][0], 3))


def test_nonstandard_rotation():
    df = pd.DataFrame({'x': [1], 'y': [1], 'z': [1], 'A': [4], 'h': [0.9], 'rho': [0.4], 'm': [0.03]})
    sdf = SarracenDataFrame(df, params=dict())
    kernel = CubicSplineKernel()
    sdf.kernel = kernel

    column_kernel = kernel.get_column_kernel_func(1000)

    rot_z, rot_y, rot_x = 129, 34, 50

    image = interpolate_3d(sdf, 'A', x_pixels=50, y_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1, rotation=[rot_z, rot_y, rot_x], origin=[0, 0, 0])

    w = sdf['m'] / (sdf['rho'] * sdf['h'] ** 2)

    pos_z1 = 1
    pos_y1 = np.sin(rot_z / (180 / np.pi)) + np.cos(rot_z / (180 / np.pi))
    pos_x1 = np.cos(rot_z / (180 / np.pi)) - np.sin(rot_z / (180 / np.pi))

    pos_x2 = pos_x1 * np.cos(rot_y / (180 / np.pi)) + np.sin(rot_y / (180 / np.pi))
    pos_y2 = pos_y1
    pos_z2 = pos_x1 * -np.sin(rot_y / (180 / np.pi)) + np.cos(rot_y / (180 / np.pi))

    pos_x3 = pos_x2
    pos_y3 = pos_y2 * np.cos(rot_x / (180 / np.pi)) - pos_z2 * np.sin(rot_x / (180 / np.pi))
    pos_z3 = pos_y2 * np.sin(rot_x / (180 / np.pi)) + pos_z2 * np.cos(rot_x / (180 / np.pi))

    real = -1 + (np.arange(0, 50) + 0.5) * (1 / 25)

    for y in range(50):
        for x in range(50):
            assert image[y][x] == approx(w[0] * sdf['A'][0] * column_kernel(
                np.sqrt((pos_x3 - real[x]) ** 2 + (pos_y3 - real[y]) ** 2) / sdf['h'][0], 3))


def test_scipy_rotation_equivalency():
    df = pd.DataFrame({'x': [1], 'y': [1], 'z': [1], 'A': [4], 'h': [0.9], 'rho': [0.4], 'm': [0.03]})
    sdf = SarracenDataFrame(df, params=dict())
    kernel = CubicSplineKernel()
    sdf.kernel = kernel

    column_kernel = kernel.get_column_kernel_func(1000)
    rot_z, rot_y, rot_x = 67, -34, 91

    image1 = interpolate_3d(sdf, 'A', x_pixels=50, y_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1, rotation=[rot_z, rot_y, rot_x], origin=[0, 0, 0])
    image2 = interpolate_3d(sdf, 'A', x_pixels=50, y_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1, rotation=Rotation.from_euler('zyx', [rot_z, rot_y, rot_x], degrees=True), origin=[0, 0, 0])

    assert_allclose(image1, image2)


def test_quaternion_rotation():
    df = pd.DataFrame({'x': [1], 'y': [1], 'z': [1], 'A': [4], 'h': [0.9], 'rho': [0.4], 'm': [0.03]})
    sdf = SarracenDataFrame(df, params=dict())
    kernel = CubicSplineKernel()
    sdf.kernel = kernel

    column_kernel = kernel.get_column_kernel_func(1000)
    q = [5/np.sqrt(99), 3/np.sqrt(99), 8/np.sqrt(99), 1/np.sqrt(99)]
    quat = Rotation.from_quat(q)
    image = interpolate_3d(sdf, 'A', x_pixels=50, y_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1, rotation=quat, origin=[0, 0, 0])

    w = sdf['m'] / (sdf['rho'] * sdf['h'] ** 2)

    pos = quat.apply([1, 1, 1])
    real = -1 + (np.arange(0, 50) + 0.5) * (1 / 25)

    for y in range(50):
        for x in range(50):
            assert image[y][x] == approx(w[0] * sdf['A'][0] * column_kernel(
                np.sqrt((pos[0] - real[x]) ** 2 + (pos[1] - real[y]) ** 2) / sdf['h'][0], 3))


def test_rotation_stability():
    df = pd.DataFrame({'x': [1], 'y': [1], 'z': [1], 'A': [4], 'h': [0.9], 'rho': [0.4], 'm': [0.03]})
    sdf = SarracenDataFrame(df, params=dict())
    kernel = CubicSplineKernel()
    sdf.kernel = kernel
    column_kernel = kernel.get_column_kernel_func(1000)

    real = -1 + (np.arange(0, 50) + 0.5) * (1 / 25)
    pixel_x, pixel_y = 12, 30

    image = interpolate_3d(sdf, 'A', x_pixels=50, y_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1)
    image_rot = interpolate_3d(sdf, 'A', x_pixels=50, y_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1,
                               rotation=[237, 0, 0], origin=[real[pixel_x], real[pixel_y], 0])

    w = sdf['m'] / (sdf['rho'] * sdf['h'] ** 2)

    assert image[pixel_y][pixel_x] == approx(image_rot[pixel_y][pixel_x])


def test_axes_rotation_separation():
    df = pd.DataFrame({'x': [-1, 1], 'y': [1, -1], 'z': [1, -1], 'A': [2, 1.5], 'h': [1.1, 1.3], 'rho': [0.55, 0.45], 'm': [0.04, 0.05]})
    sdf = SarracenDataFrame(df, params=dict())

    image1 = interpolate_3d(sdf, 'A', x_pixels=50,  y_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1, rotation=[234, 90, 48])
    image2 = interpolate_3d(sdf, 'A', x='y', y='x', x_pixels=50,  y_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1, rotation=[234, 90, 48])

    assert_allclose(image1, image2.T)

    image1 = interpolate_3d_cross(sdf, 'A', 0, x_pixels=50,  y_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1, rotation=[234, 90, 48])
    image2 = interpolate_3d_cross(sdf, 'A', 0, x='y', y='x', x_pixels=50,  y_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1, rotation=[234, 90, 48])

    assert_allclose(image1, image2.T)


def test_axes_rotation_equivalency():
    df = pd.DataFrame({'x': [-1, 1], 'y': [1, -1], 'z': [1, -1], 'A': [2, 1.5], 'h': [1.1, 1.3], 'rho': [0.55, 0.45],
                       'm': [0.04, 0.05]})
    sdf = SarracenDataFrame(df, params=dict())

    x, y, z = 'x', 'y', 'z'
    flip_x, flip_y, flip_z = False, False, False
    for i_z in range(4):
        for i_y in range(4):
            for i_x in range(4):
                rot_x, rot_y, rot_z = i_x * 90, i_y * 90, i_z * 90
                image1 = interpolate_3d(sdf, 'A', x_pixels=50, y_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1,
                                        rotation=[rot_z, rot_y, rot_x], origin=[0, 0, 0])
                image2 = interpolate_3d(sdf, 'A', x=x, y=y, x_pixels=50, y_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1)
                image2 = image2 if not flip_x else np.flip(image2, 1)
                image2 = image2 if not flip_y else np.flip(image2, 0)
                assert_allclose(image1, image2)
                y, z = z, y
                flip_y, flip_z = not flip_z, flip_y
            x, z = z, x
            flip_x, flip_z = flip_z, not flip_x
        x, y = y, x
        flip_x, flip_y = not flip_y, flip_x
