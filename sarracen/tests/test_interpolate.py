"""
pytest unit tests for interpolate.py functions.
"""
import pandas as pd
import numpy as np
from numpy.testing import assert_allclose
from pytest import approx, raises

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

