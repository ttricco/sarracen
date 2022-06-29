"""
pytest unit tests for interpolate.py functions.
"""
import pandas as pd
import numpy as np
from numba import cuda
from numpy.testing import assert_allclose
from pytest import approx, raises, mark
from scipy.spatial.transform import Rotation

from sarracen import SarracenDataFrame
from sarracen.kernels import CubicSplineKernel, QuarticSplineKernel, QuinticSplineKernel
from sarracen.interpolate import interpolate_2d, interpolate_2d_cross, interpolate_3d_cross, interpolate_3d, \
    interpolate_2d_vec, interpolate_3d_vec, interpolate_3d_cross_vec

backends = ['cpu']
if cuda.is_available():
    backends.append('gpu')


@mark.parametrize("backend", backends)
def test_single_particle(backend):
    df = pd.DataFrame({'x': [0], 'y': [0], 'A': [4], 'B': [5], 'h': [0.9], 'rho': [0.4], 'm': [0.03]})
    sdf = SarracenDataFrame(df, params=dict())
    kernel = CubicSplineKernel()
    sdf.kernel = kernel
    sdf.backend = backend
    w = sdf['m'] / (sdf['rho'] * sdf['h'] ** 2)

    # A mapping of pixel indices to x / y values in particle space.
    real = -kernel.get_radius() + (np.arange(0, 25) + 0.5) * (2 * kernel.get_radius() / 25)

    image = interpolate_2d(sdf, 'A', x_pixels=25,  y_pixels=25, x_min=-kernel.get_radius(), x_max=kernel.get_radius(),
                           y_min=-kernel.get_radius(), y_max=kernel.get_radius())

    for y in range(25):
        for x in range(25):
            assert image[y][x] ==\
                   approx(w[0] * sdf['A'][0] * kernel.w(np.sqrt(real[x] ** 2 + real[y] ** 2) / sdf['h'][0], 2))

    image = interpolate_2d_vec(sdf, 'A', 'B', x_pixels=25, y_pixels=25, x_min=-kernel.get_radius(),
                               x_max=kernel.get_radius(), y_min=-kernel.get_radius(), y_max=kernel.get_radius())

    for y in range(25):
        for x in range(25):
            assert image[0][y][x] ==\
                   approx(w[0] * sdf['A'][0] * kernel.w(np.sqrt(real[x] ** 2 + real[y] ** 2) / sdf['h'][0], 2))
            assert image[1][y][x] == \
                   approx(w[0] * sdf['B'][0] * kernel.w(np.sqrt(real[x] ** 2 + real[y] ** 2) / sdf['h'][0], 2))

    image = interpolate_2d_cross(sdf, 'A', pixels=25, x1=-kernel.get_radius(), x2=kernel.get_radius(),
                                 y1=-kernel.get_radius(), y2=kernel.get_radius())

    for x in range(25):
        assert image[x] == approx(w[0] * sdf['A'][0] * kernel.w(np.sqrt(2) * np.abs(real[x]) / sdf['h'][0], 2))

    sdf['z'] = -0.5
    sdf['C'] = 10
    sdf.zcol = 'z'

    column_func = kernel.get_column_kernel_func(1000)

    image = interpolate_3d(sdf, 'A', x_pixels=25, y_pixels=25, x_min=-kernel.get_radius(), x_max=kernel.get_radius(),
                           y_min=-kernel.get_radius(), y_max=kernel.get_radius())
    for y in range(25):
        for x in range(25):
            assert image[y][x] ==\
                   approx(w[0] * sdf['A'][0] * column_func(np.sqrt(real[x] ** 2 + real[y] ** 2) / sdf['h'][0], 2))

    image = interpolate_3d_vec(sdf, 'A', 'B', 'C', x_pixels=25, y_pixels=25, x_min=-kernel.get_radius(),
                               x_max=kernel.get_radius(), y_min=-kernel.get_radius(), y_max=kernel.get_radius())
    for y in range(25):
        for x in range(25):
            assert image[0][y][x] == \
                   approx(w[0] * sdf['A'][0] * column_func(np.sqrt(real[x] ** 2 + real[y] ** 2) / sdf['h'][0], 2))
            assert image[1][y][x] == \
                   approx(w[0] * sdf['B'][0] * column_func(np.sqrt(real[x] ** 2 + real[y] ** 2) / sdf['h'][0], 2))

    w = sdf['m'] / (sdf['rho'] * sdf['h'] ** 3)

    image = interpolate_3d_cross(sdf, 'A', 0, x_pixels=25, y_pixels=25, x_min=-kernel.get_radius(),
                                 x_max=kernel.get_radius(), y_min=-kernel.get_radius(), y_max=kernel.get_radius())
    for y in range(25):
        for x in range(25):
            assert image[y][x] == approx(w[0] * sdf['A'][0] *
                                         kernel.w(np.sqrt(real[x] ** 2 + real[y] ** 2 + 0.5 ** 2) / sdf['h'][0], 3))

    image = interpolate_3d_cross_vec(sdf, 'A', 'B', 'C', 0, x_pixels=25, y_pixels=25, x_min=-kernel.get_radius(),
                                     x_max=kernel.get_radius(), y_min=-kernel.get_radius(), y_max=kernel.get_radius())
    for y in range(25):
        for x in range(25):
            assert image[0][y][x] == approx(w[0] * sdf['A'][0] *
                                            kernel.w(np.sqrt(real[x] ** 2 + real[y] ** 2 + 0.5 ** 2) / sdf['h'][0], 3))
            assert image[1][y][x] == approx(w[0] * sdf['B'][0] *
                                            kernel.w(np.sqrt(real[x] ** 2 + real[y] ** 2 + 0.5 ** 2) / sdf['h'][0], 3))


@mark.parametrize("backend", backends)
def test_single_repeated_particle(backend):
    repetitions = 10000

    df = pd.concat([pd.DataFrame({'x': [0], 'y': [0], 'A': [4], 'B': [5], 'h': [0.9], 'rho': [0.4],
                                  'm': [0.03]})] * repetitions, ignore_index=True)
    sdf = SarracenDataFrame(df, params=dict())
    kernel = CubicSplineKernel()
    sdf.kernel = kernel
    sdf.backend = backend
    w = repetitions * sdf['m'] / (sdf['rho'] * sdf['h'] ** 2)

    # A mapping of pixel indices to x / y values in particle space.
    real = -kernel.get_radius() + (np.arange(0, 25) + 0.5) * (2 * kernel.get_radius() / 25)

    image = interpolate_2d(sdf, 'A', x_pixels=25, y_pixels=25, x_min=-kernel.get_radius(), x_max=kernel.get_radius(),
                           y_min=-kernel.get_radius(), y_max=kernel.get_radius())

    for y in range(25):
        for x in range(25):
            assert image[y][x] == \
                   approx(w[0] * sdf['A'][0] * kernel.w(np.sqrt(real[x] ** 2 + real[y] ** 2) / sdf['h'][0], 2))

    image = interpolate_2d_vec(sdf, 'A', 'B', x_pixels=25, y_pixels=25, x_min=-kernel.get_radius(),
                               x_max=kernel.get_radius(), y_min=-kernel.get_radius(), y_max=kernel.get_radius())

    for y in range(25):
        for x in range(25):
            assert image[0][y][x] == \
                   approx(w[0] * sdf['A'][0] * kernel.w(np.sqrt(real[x] ** 2 + real[y] ** 2) / sdf['h'][0], 2))
            assert image[1][y][x] == \
                   approx(w[0] * sdf['B'][0] * kernel.w(np.sqrt(real[x] ** 2 + real[y] ** 2) / sdf['h'][0], 2))

    image = interpolate_2d_cross(sdf, 'A', pixels=25, x1=-kernel.get_radius(), x2=kernel.get_radius(),
                                 y1=-kernel.get_radius(), y2=kernel.get_radius())

    for x in range(25):
        assert image[x] == approx(w[0] * sdf['A'][0] * kernel.w(np.sqrt(2) * np.abs(real[x]) / sdf['h'][0], 2))

    sdf['z'] = -0.5
    sdf['C'] = 10
    sdf.zcol = 'z'

    column_func = kernel.get_column_kernel_func(1000)

    image = interpolate_3d(sdf, 'A', x_pixels=25, y_pixels=25, x_min=-kernel.get_radius(), x_max=kernel.get_radius(),
                           y_min=-kernel.get_radius(), y_max=kernel.get_radius())
    for y in range(25):
        for x in range(25):
            assert image[y][x] == \
                   approx(w[0] * sdf['A'][0] * column_func(np.sqrt(real[x] ** 2 + real[y] ** 2) / sdf['h'][0], 2))

    image = interpolate_3d_vec(sdf, 'A', 'B', 'C', x_pixels=25, y_pixels=25, x_min=-kernel.get_radius(),
                               x_max=kernel.get_radius(), y_min=-kernel.get_radius(), y_max=kernel.get_radius())
    for y in range(25):
        for x in range(25):
            assert image[0][y][x] == \
                   approx(w[0] * sdf['A'][0] * column_func(np.sqrt(real[x] ** 2 + real[y] ** 2) / sdf['h'][0], 2))
            assert image[1][y][x] == \
                   approx(w[0] * sdf['B'][0] * column_func(np.sqrt(real[x] ** 2 + real[y] ** 2) / sdf['h'][0], 2))

    w = repetitions * sdf['m'] / (sdf['rho'] * sdf['h'] ** 3)

    image = interpolate_3d_cross(sdf, 'A', 0, x_pixels=25, y_pixels=25, x_min=-kernel.get_radius(),
                                 x_max=kernel.get_radius(), y_min=-kernel.get_radius(), y_max=kernel.get_radius())
    for y in range(25):
        for x in range(25):
            assert image[y][x] == approx(w[0] * sdf['A'][0] *
                                         kernel.w(np.sqrt(real[x] ** 2 + real[y] ** 2 + 0.5 ** 2) / sdf['h'][0], 3))

    image = interpolate_3d_cross_vec(sdf, 'A', 'B', 'C', 0, x_pixels=25, y_pixels=25, x_min=-kernel.get_radius(),
                                     x_max=kernel.get_radius(), y_min=-kernel.get_radius(), y_max=kernel.get_radius())
    for y in range(25):
        for x in range(25):
            assert image[0][y][x] == approx(w[0] * sdf['A'][0] *
                                            kernel.w(np.sqrt(real[x] ** 2 + real[y] ** 2 + 0.5 ** 2) / sdf['h'][0], 3))
            assert image[1][y][x] == approx(w[0] * sdf['B'][0] *
                                            kernel.w(np.sqrt(real[x] ** 2 + real[y] ** 2 + 0.5 ** 2) / sdf['h'][0], 3))


@mark.parametrize("backend", backends)
def test_dimension_check(backend):
    df = pd.DataFrame({'x': [0, 1], 'y': [0, 1], 'P': [1, 1], 'Ax': [1, 1], 'Ay': [1, 1], 'Az': [1, 1], 'h': [1, 1],
                       'rho': [1, 1], 'm': [1, 1]})
    sdf = SarracenDataFrame(df, params=dict())
    sdf.backend = backend

    with raises(TypeError):
        interpolate_3d(sdf, 'P')

    with raises(TypeError):
        interpolate_3d_cross(sdf, 'P')

    with raises(TypeError):
        interpolate_3d_vec(sdf, 'Ax', 'Ay', 'Az')

    with raises(TypeError):
        interpolate_3d_cross_vec(sdf, 'Ax', 'Ay', 'Az')

    df = pd.DataFrame({'x': [0, 1], 'y': [0, 1], 'z': [0, 1], 'P': [1, 1], 'Ax': [1, 1], 'Ay': [1, 1], 'Az': [1, 1],
                       'h': [1, 1], 'rho': [1, 1], 'm': [1, 1]})
    sdf = SarracenDataFrame(df, params=dict())

    with raises(TypeError):
        interpolate_2d(sdf, 'P')

    with raises(TypeError):
        interpolate_2d_cross(sdf, 'P')

    with raises(TypeError):
        interpolate_2d_vec(sdf, 'Ax', 'Ay')


@mark.parametrize("backend", backends)
def test_3d_xsec_equivalency(backend):
    df = pd.DataFrame({'x': [0], 'y': [0], 'z': [0], 'A': [4], 'B': [6], 'C': [2], 'h': [0.9], 'rho': [0.4],
                       'm': [0.03]})
    sdf = SarracenDataFrame(df, params=dict())
    kernel = CubicSplineKernel()
    sdf.kernel = kernel
    sdf.backend = backend

    samples = 250

    column_image = interpolate_3d(sdf, 'A', x_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1)
    xsec_image = np.zeros((50, 50))
    column_image_vec = interpolate_3d_vec(sdf, 'A', 'B', 'C', x_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1)
    xsec_image_vec = [np.zeros((50, 50)), np.zeros((50, 50))]

    for z in np.linspace(0, kernel.get_radius() * sdf['h'][0], samples):
        xsec_image += interpolate_3d_cross(sdf, 'A', z, x_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1)

        vec_sample = interpolate_3d_cross_vec(sdf, 'A', 'B', 'C', z, x_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1)
        xsec_image_vec[0] += vec_sample[0]
        xsec_image_vec[1] += vec_sample[1]

    xsec_image *= kernel.get_radius() * sdf['h'][0] * 2 / samples
    xsec_image_vec[0] *= kernel.get_radius() * sdf['h'][0] * 2 / samples
    xsec_image_vec[1] *= kernel.get_radius() * sdf['h'][0] * 2 / samples

    assert_allclose(xsec_image, column_image, rtol=1e-3, atol=1e-4)
    assert_allclose(xsec_image[0], column_image_vec[0], rtol=1e-3, atol=1e-4)
    assert_allclose(xsec_image[1], column_image_vec[1], rtol=1e-3, atol=1e-4)


@mark.parametrize("backend", backends)
def test_2d_xsec_equivalency(backend):
    df = pd.DataFrame({'x': [0], 'y': [0], 'A': [4], 'h': [0.9], 'rho': [0.4], 'm': [0.03]})
    sdf = SarracenDataFrame(df, params=dict())
    kernel = CubicSplineKernel()
    sdf.kernel = kernel
    sdf.backend = backend

    true_image = interpolate_2d(sdf, 'A', x_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1)

    # A mapping of pixel indices to x & y values in particle space.
    real = -1 + (np.arange(0, 50) + 0.5) * (2 / 50)

    reconstructed_image = np.zeros((50, 50))
    for y in range(50):
        reconstructed_image[y, :] = interpolate_2d_cross(sdf, 'A', pixels=50, x1=-1, x2=1, y1=real[y], y2=real[y])

    assert_allclose(reconstructed_image, true_image)


@mark.parametrize("backend", backends)
def test_corner_particles(backend):
    df_2 = pd.DataFrame({'x': [-1, 1], 'y': [-1, 1], 'A': [2, 1.5], 'B': [5, 2.3], 'h': [1.1, 1.3], 'rho': [0.55, 0.45],
                         'm': [0.04, 0.05]})
    sdf_2 = SarracenDataFrame(df_2, params=dict())
    df_3 = pd.DataFrame({'x': [-1, 1], 'y': [-1, 1], 'z': [-1, 1], 'A': [2, 1.5], 'B': [2, 1], 'C': [7, 8],
                         'h': [1.1, 1.3], 'rho': [0.55, 0.45], 'm': [0.04, 0.05]})
    sdf_3 = SarracenDataFrame(df_3, params=dict())

    kernel = CubicSplineKernel()
    sdf_2.kernel = kernel
    sdf_2.backend = backend
    sdf_3.kernel = kernel
    sdf_3.backend = backend

    w = sdf_2['m'] / (sdf_2['rho'] * sdf_2['h'] ** 2)

    # A mapping of pixel indices to x / y values in particle space.
    real = (np.arange(0, 25) + 0.5) * (2 / 25)

    image = interpolate_2d(sdf_2, 'A', x_pixels=25,  y_pixels=25, x_min=-1, x_max=1, y_min=-1, y_max=1)
    image_vec = interpolate_2d_vec(sdf_2, 'A', 'B', x_pixels=25,  y_pixels=25, x_min=-1, x_max=1, y_min=-1, y_max=1)
    for y in range(25):
        for x in range(25):
            assert approx(w[0] * sdf_2['A'][0] * kernel.w(np.sqrt(real[x] ** 2 + real[y] ** 2) / sdf_2['h'][0], 2)
                          + w[1] * sdf_2['A'][1] * kernel.w(np.sqrt(real[24 - x] ** 2 + real[24 - y] ** 2)
                                                            / sdf_2['h'][1], 2)) == image[y][x]

            assert approx(w[0] * sdf_2['A'][0] * kernel.w(np.sqrt(real[x] ** 2 + real[y] ** 2) / sdf_2['h'][0], 2)
                          + w[1] * sdf_2['A'][1] * kernel.w(np.sqrt(real[24 - x] ** 2 + real[24 - y] ** 2)
                                                            / sdf_2['h'][1], 2)) == image_vec[0][y][x]

            assert approx(w[0] * sdf_2['B'][0] * kernel.w(np.sqrt(real[x] ** 2 + real[y] ** 2) / sdf_2['h'][0], 2)
                          + w[1] * sdf_2['B'][1] * kernel.w(np.sqrt(real[24 - x] ** 2 + real[24 - y] ** 2)
                                                            / sdf_2['h'][1], 2)) == image_vec[1][y][x]

    image = interpolate_2d_cross(sdf_2, 'A', pixels=25, x1=-1, x2=1, y1=-1, y2=1)
    for x in range(25):
        assert approx(w[0] * sdf_2['A'][0] * kernel.w(np.sqrt(real[x] ** 2 + real[x] ** 2) / sdf_2['h'][0], 2)
                      + w[1] * sdf_2['A'][1] * kernel.w(np.sqrt(real[24 - x] ** 2 + real[24 - x] ** 2)
                                                        / sdf_2['h'][1], 2)) == image[x]

    c_kernel = kernel.get_column_kernel_func(1000)

    image = interpolate_3d(sdf_3, 'A', x_pixels=25, y_pixels=25, x_min=-1, x_max=1, y_min=-1, y_max=1)
    image_vec = interpolate_3d_vec(sdf_3, 'A', 'B', 'C', x_pixels=25, y_pixels=25, x_min=-1, x_max=1, y_min=-1, y_max=1)
    for y in range(25):
        for x in range(25):
            assert approx(w[0] * sdf_3['A'][0] * c_kernel(np.sqrt(real[x] ** 2 + real[y] ** 2) / sdf_3['h'][0], 2)
                          + w[1] * sdf_3['A'][1] * c_kernel(np.sqrt(real[24 - x] ** 2 + real[24 - y] ** 2)
                                                            / sdf_3['h'][1], 2)) == image[y][x]

            assert approx(w[0] * sdf_3['A'][0] * c_kernel(np.sqrt(real[x] ** 2 + real[y] ** 2) / sdf_3['h'][0], 2)
                          + w[1] * sdf_3['A'][1] * c_kernel(np.sqrt(real[24 - x] ** 2 + real[24 - y] ** 2)
                                                            / sdf_3['h'][1], 2)) == image_vec[0][y][x]

            assert approx(w[0] * sdf_3['B'][0] * c_kernel(np.sqrt(real[x] ** 2 + real[y] ** 2) / sdf_3['h'][0], 2)
                          + w[1] * sdf_3['B'][1] * c_kernel(np.sqrt(real[24 - x] ** 2 + real[24 - y] ** 2)
                                                            / sdf_3['h'][1], 2)) == image_vec[1][y][x]

    w = sdf_3['m'] / (sdf_3['rho'] * sdf_3['h'] ** 3)

    image = interpolate_3d_cross(sdf_3, 'A', 0, x_pixels=25, y_pixels=25, x_min=-1, x_max=1, y_min=-1, y_max=1)
    image_vec = interpolate_3d_cross_vec(sdf_3, 'A', 'B', 'C', 0, x_pixels=25, y_pixels=25, x_min=-1, x_max=1, y_min=-1,
                                         y_max=1)
    for y in range(25):
        for x in range(25):
            assert approx(w[0] * sdf_3['A'][0] * kernel.w(np.sqrt(real[x] ** 2 + real[y] ** 2 + 1) / sdf_3['h'][0], 3)
                          + w[1] * sdf_3['A'][1] * kernel.w(np.sqrt(real[24 - x] ** 2 + real[24 - y] ** 2 + 1)
                                                            / sdf_3['h'][1], 3)) == image[y][x]

            assert approx(w[0] * sdf_3['A'][0] * kernel.w(np.sqrt(real[x] ** 2 + real[y] ** 2 + 1) / sdf_3['h'][0], 3)
                          + w[1] * sdf_3['A'][1] * kernel.w(np.sqrt(real[24 - x] ** 2 + real[24 - y] ** 2 + 1)
                                                            / sdf_3['h'][1], 3)) == image_vec[0][y][x]

            assert approx(w[0] * sdf_3['B'][0] * kernel.w(np.sqrt(real[x] ** 2 + real[y] ** 2 + 1) / sdf_3['h'][0], 3)
                          + w[1] * sdf_3['B'][1] * kernel.w(np.sqrt(real[24 - x] ** 2 + real[24 - y] ** 2 + 1)
                                                            / sdf_3['h'][1], 3)) == image_vec[1][y][x]


@mark.parametrize("backend", backends)
def test_image_transpose(backend):
    df = pd.DataFrame({'x': [-1, 1], 'y': [1, -1], 'A': [2, 1.5], 'B': [5, 4], 'h': [1.1, 1.3], 'rho': [0.55, 0.45],
                       'm': [0.04, 0.05]})
    sdf = SarracenDataFrame(df, params=dict())
    sdf.backend = backend

    image1 = interpolate_2d(sdf, 'A', x_pixels=50,  y_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1)
    image2 = interpolate_2d(sdf, 'A', x='y', y='x', x_pixels=50,  y_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1)
    assert_allclose(image1, image2.T)

    image1 = interpolate_2d_vec(sdf, 'A', 'B', x_pixels=50, y_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1)
    image2 = interpolate_2d_vec(sdf, 'A', 'B', x='y', y='x', x_pixels=50, y_pixels=50, x_min=-1, x_max=1, y_min=-1,
                                y_max=1)
    assert_allclose(image1[0], image2[0].T)
    assert_allclose(image1[1], image2[1].T)

    df = pd.DataFrame({'x': [-1, 1], 'y': [1, -1], 'z': [-1, 1], 'A': [2, 1.5], 'B': [5, 4], 'C': [2.5, 3],
                       'h': [1.1, 1.3], 'rho': [0.55, 0.45], 'm': [0.04, 0.05]})
    sdf = SarracenDataFrame(df, params=dict())

    image1 = interpolate_3d(sdf, 'A', x_pixels=50,  y_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1)
    image2 = interpolate_3d(sdf, 'A', x='y', y='x', x_pixels=50,  y_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1)
    assert_allclose(image1, image2.T)

    image1 = interpolate_3d_vec(sdf, 'A', 'B', 'C', x_pixels=50, y_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1)
    image2 = interpolate_3d_vec(sdf, 'A', 'B', 'C', x='y', y='x', x_pixels=50, y_pixels=50, x_min=-1, x_max=1, y_min=-1,
                                y_max=1)
    assert_allclose(image1[0], image2[0].T)
    assert_allclose(image1[1], image2[1].T)

    image1 = interpolate_3d_cross(sdf, 'A', 0, x_pixels=50,  y_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1)
    image2 = interpolate_3d_cross(sdf, 'A', 0, x='y', y='x', x_pixels=50,  y_pixels=50, x_min=-1, x_max=1, y_min=-1,
                                  y_max=1)
    assert_allclose(image1, image2.T)

    image1 = interpolate_3d_cross_vec(sdf, 'A', 'B', 'C', 0, x_pixels=50, y_pixels=50, x_min=-1, x_max=1, y_min=-1,
                                      y_max=1)
    image2 = interpolate_3d_cross_vec(sdf, 'A', 'B', 'C', 0, x='y', y='x', x_pixels=50, y_pixels=50, x_min=-1, x_max=1,
                                  y_min=-1, y_max=1)
    assert_allclose(image1[0], image2[0].T)
    assert_allclose(image1[1], image2[1].T)


@mark.parametrize("backend", backends)
def test_default_kernel(backend):
    df_2 = pd.DataFrame({'x': [0], 'y': [0], 'A': [1], 'B': [1], 'h': [1], 'rho': [1], 'm': [1]})
    sdf_2 = SarracenDataFrame(df_2, params=dict())
    df_3 = pd.DataFrame({'x': [0], 'y': [0], 'z': [0], 'A': [1], 'B': [1], 'C': [1], 'h': [1], 'rho': [1], 'm': [1]})
    sdf_3 = SarracenDataFrame(df_3, params=dict())

    kernel = QuarticSplineKernel()
    sdf_2.kernel = kernel
    sdf_3.kernel = kernel
    sdf_2.backend = backend
    sdf_3.backend = backend

    w = sdf_2['m'] / (sdf_2['rho'] * sdf_2['h'] ** 2)

    image = interpolate_2d(sdf_2, 'A', x_pixels=1, y_pixels=1, x_min=-1, x_max=1, y_min=-1, y_max=1)
    assert image == kernel.w(0, 2)

    image = interpolate_2d_vec(sdf_2, 'A', 'B', x_pixels=1, y_pixels=1, x_min=-1, x_max=1, y_min=-1, y_max=1)
    assert image[0] == kernel.w(0, 2)
    assert image[1] == kernel.w(0, 2)

    image = interpolate_2d_cross(sdf_2, 'A', pixels=1, x1=-1, x2=1, y1=-1, y2=1)
    assert image == kernel.w(0, 2)

    image = interpolate_3d(sdf_3, 'A', x_pixels=1, y_pixels=1, x_min=-1, x_max=1, y_min=-1, y_max=1)
    assert image == kernel.get_column_kernel()[0]

    image = interpolate_3d_vec(sdf_3, 'A', 'B', 'C', x_pixels=1, y_pixels=1, x_min=-1, x_max=1, y_min=-1, y_max=1)
    assert image[0] == kernel.get_column_kernel()[0]
    assert image[1] == kernel.get_column_kernel()[0]

    image = interpolate_3d_cross(sdf_3, 'A', x_pixels=1, y_pixels=1, x_min=-1, x_max=1, y_min=-1, y_max=1)
    assert image == kernel.w(0, 3)

    image = interpolate_3d_cross_vec(sdf_3, 'A', 'B', 'C', x_pixels=1, y_pixels=1, x_min=-1, x_max=1, y_min=-1, y_max=1)
    assert image[0] == kernel.w(0, 3)
    assert image[1] == kernel.w(0, 3)

    kernel = QuinticSplineKernel()
    image = interpolate_2d(sdf_2, 'A', x_pixels=1, y_pixels=1, x_min=-1, x_max=1, y_min=-1, y_max=1, kernel=kernel)
    assert image == kernel.w(0, 2)

    image = interpolate_2d_vec(sdf_2, 'A', 'B', x_pixels=1, y_pixels=1, x_min=-1, x_max=1, y_min=-1, y_max=1,
                               kernel=kernel)
    assert image[0] == kernel.w(0, 2)
    assert image[1] == kernel.w(0, 2)

    image = interpolate_2d_cross(sdf_2, 'A', pixels=1, x1=-1, x2=1, y1=-1, y2=1,
                                 kernel=kernel)
    assert image == kernel.w(0, 2)

    image = interpolate_3d(sdf_3, 'A', x_pixels=1, y_pixels=1, x_min=-1, x_max=1, y_min=-1, y_max=1,
                           kernel=kernel)
    assert image == kernel.get_column_kernel()[0]

    image = interpolate_3d_vec(sdf_3, 'A', 'B', 'C', x_pixels=1, y_pixels=1, x_min=-1, x_max=1, y_min=-1, y_max=1,
                               kernel=kernel)
    assert image[0] == kernel.get_column_kernel()[0]
    assert image[1] == kernel.get_column_kernel()[0]

    image = interpolate_3d_cross(sdf_3, 'A', x_pixels=1, y_pixels=1, x_min=-1, x_max=1, y_min=-1, y_max=1,
                                 kernel=kernel)
    assert image == kernel.w(0, 3)

    image = interpolate_3d_cross_vec(sdf_3, 'A', 'B', 'C', x_pixels=1, y_pixels=1, x_min=-1, x_max=1, y_min=-1, y_max=1,
                                     kernel=kernel)
    assert image[0] == kernel.w(0, 3)
    assert image[1] == kernel.w(0, 3)


@mark.parametrize("backend", backends)
def test_column_samples(backend):
    df_3 = pd.DataFrame({'x': [0], 'y': [0], 'z': [0], 'A': [1], 'h': [1], 'rho': [1], 'm': [1]})
    sdf_3 = SarracenDataFrame(df_3, params=dict())
    kernel = QuinticSplineKernel()
    sdf_3.kernel = kernel
    sdf_3.backend = backend

    image = interpolate_3d(sdf_3, 'A', x_pixels=1, y_pixels=1, x_min=-1, x_max=1, y_min=-1, y_max=1, integral_samples=2)
    assert image == kernel.get_column_kernel(2)[0]


@mark.parametrize("backend", backends)
def test_pixel_arguments(backend):
    df_2 = pd.DataFrame({'x': [-2, 4], 'y': [3, 8], 'A': [1, 1], 'B': [1, 1], 'h': [1, 1], 'rho': [1, 1], 'm': [1, 1]})
    sdf_2 = SarracenDataFrame(df_2, params=dict())
    df_3 = pd.DataFrame({'x': [-2, 4], 'y': [3, 8], 'z': [7, -2], 'A': [1, 1], 'B': [1, 1], 'C': [1, 1], 'h': [1, 1],
                         'rho': [1, 1], 'm': [1, 1]})
    sdf_3 = SarracenDataFrame(df_3, params=dict())

    sdf_2.backend = backend
    sdf_3.backend = backend

    default_pixels = 100

    # Non-vector functions
    for func in [interpolate_2d, interpolate_3d, interpolate_3d_cross]:
        for axes in [('x', 'y'), ('x', 'z'), ('y', 'z'), ('y', 'x'), ('z', 'x'), ('z', 'y')]:
            ratio = np.abs(df_3[axes[1]][1] - df_3[axes[1]][0]) / np.abs(df_3[axes[0]][1] - df_3[axes[0]][0])

            if (axes[0] == 'z' or axes[1] == 'z') and func is interpolate_2d:
                continue

            sdf = sdf_2 if func is interpolate_2d else sdf_3

            image = func(sdf, 'A', x=axes[0], y=axes[1])
            assert image.shape[0] / image.shape[1] == approx(ratio, rel=1e-2)

            image = func(sdf, 'A', x=axes[0], y=axes[1], x_pixels=default_pixels)
            assert image.shape == (round(default_pixels * ratio), default_pixels)

            image = func(sdf, 'A', x=axes[0], y=axes[1], y_pixels=default_pixels)
            assert image.shape == (default_pixels, round(default_pixels / ratio))

            image = func(sdf, 'A', x_pixels=default_pixels * 2, y_pixels=default_pixels)
            assert image.shape == (default_pixels, default_pixels * 2)

    # 3D Vector-based functions
    for func in [interpolate_3d_vec, interpolate_3d_cross_vec]:
        for axes in [('x', 'y'), ('x', 'z'), ('y', 'z'), ('y', 'x'), ('z', 'x'), ('z', 'y')]:
            ratio = np.abs(df_3[axes[1]][1] - df_3[axes[1]][0]) / np.abs(df_3[axes[0]][1] - df_3[axes[0]][0])

            image = func(sdf_3, 'A', 'B', 'C', x=axes[0], y=axes[1])
            assert image[0].shape[0] / image[0].shape[1] == approx(ratio, rel=1e-2)
            assert image[1].shape[0] / image[1].shape[1] == approx(ratio, rel=1e-2)

            image = func(sdf_3, 'A', 'B', 'C', x=axes[0], y=axes[1], x_pixels=default_pixels)
            assert image[0].shape == (round(default_pixels * ratio), default_pixels)
            assert image[1].shape == (round(default_pixels * ratio), default_pixels)

            image = func(sdf_3, 'A', 'B', 'C', x=axes[0], y=axes[1], y_pixels=default_pixels)
            assert image[0].shape == (default_pixels, round(default_pixels / ratio))
            assert image[1].shape == (default_pixels, round(default_pixels / ratio))

            image = func(sdf_3, 'A', 'B', 'C', x_pixels=default_pixels * 2, y_pixels=default_pixels)
            assert image[0].shape == (default_pixels, default_pixels * 2)
            assert image[1].shape == (default_pixels, default_pixels * 2)

    # 2D vector interpolation
    for axes in [('x', 'y'), ('y', 'x')]:
        ratio = np.abs(df_3[axes[1]][1] - df_3[axes[1]][0]) / np.abs(df_3[axes[0]][1] - df_3[axes[0]][0])

        image = interpolate_2d_vec(sdf_2, 'A', 'B', x=axes[0], y=axes[1])
        assert image[0].shape[0] / image[0].shape[1] == approx(ratio, rel=1e-2)
        assert image[1].shape[0] / image[1].shape[1] == approx(ratio, rel=1e-2)

        image = interpolate_2d_vec(sdf_2, 'A', 'B', x=axes[0], y=axes[1], x_pixels=default_pixels)
        assert image[0].shape == (round(default_pixels * ratio), default_pixels)
        assert image[1].shape == (round(default_pixels * ratio), default_pixels)

        image = interpolate_2d_vec(sdf_2, 'A', 'B', x=axes[0], y=axes[1], y_pixels=default_pixels)
        assert image[0].shape == (default_pixels, round(default_pixels / ratio))
        assert image[1].shape == (default_pixels, round(default_pixels / ratio))

        image = interpolate_2d_vec(sdf_2, 'A', 'B', x_pixels=default_pixels * 2, y_pixels=default_pixels)
        assert image[0].shape == (default_pixels, default_pixels * 2)
        assert image[1].shape == (default_pixels, default_pixels * 2)


@mark.parametrize("backend", backends)
def test_irregular_bounds(backend):
    df = pd.DataFrame({'x': [0], 'y': [0], 'A': [4], 'B': [7], 'h': [0.9], 'rho': [0.4], 'm': [0.03]})
    sdf = SarracenDataFrame(df, params=dict())
    kernel = CubicSplineKernel()
    sdf.kernel = kernel
    sdf.backend = backend
    w = sdf['m'] / (sdf['rho'] * sdf['h'] ** 2)

    # A mapping of pixel indices to x / y values in particle space.
    real_x = -kernel.get_radius() + (np.arange(0, 50) + 0.5) * (2 * kernel.get_radius() / 50)
    real_y = -kernel.get_radius() + (np.arange(0, 25) + 0.5) * (2 * kernel.get_radius() / 25)

    image = interpolate_2d(sdf, 'A', x_pixels=50, y_pixels=25, x_min=-kernel.get_radius(), x_max=kernel.get_radius(),
                           y_min=-kernel.get_radius(), y_max=kernel.get_radius())
    image_vec = interpolate_2d_vec(sdf, 'A', 'B', x_pixels=50, y_pixels=25, x_min=-kernel.get_radius(),
                                   x_max=kernel.get_radius(), y_min=-kernel.get_radius(), y_max=kernel.get_radius())

    for y in range(25):
        for x in range(50):
            assert image[y][x] == approx(
                w[0] * sdf['A'][0] * kernel.w(np.sqrt(real_x[x] ** 2 + real_y[y] ** 2) / sdf['h'][0], 2))

            assert image_vec[0][y][x] == approx(
                w[0] * sdf['A'][0] * kernel.w(np.sqrt(real_x[x] ** 2 + real_y[y] ** 2) / sdf['h'][0], 2))

            assert image_vec[1][y][x] == approx(
                w[0] * sdf['B'][0] * kernel.w(np.sqrt(real_x[x] ** 2 + real_y[y] ** 2) / sdf['h'][0], 2))

    sdf['C'] = 5
    sdf['z'] = -0.5
    sdf.zcol = 'z'

    column_func = kernel.get_column_kernel_func(1000)
    image = interpolate_3d(sdf, 'A', x_pixels=50, y_pixels=25, x_min=-kernel.get_radius(), x_max=kernel.get_radius(),
                           y_min=-kernel.get_radius(), y_max=kernel.get_radius())
    image_vec = interpolate_3d_vec(sdf, 'A', 'B', 'C', x_pixels=50, y_pixels=25, x_min=-kernel.get_radius(),
                                   x_max=kernel.get_radius(), y_min=-kernel.get_radius(), y_max=kernel.get_radius())

    for y in range(25):
        for x in range(50):
            assert image[y][x] == approx(
                w[0] * sdf['A'][0] * column_func(np.sqrt(real_x[x] ** 2 + real_y[y] ** 2) / sdf['h'][0], 2))

            assert image_vec[0][y][x] == approx(
                w[0] * sdf['A'][0] * column_func(np.sqrt(real_x[x] ** 2 + real_y[y] ** 2) / sdf['h'][0], 2))

            assert image_vec[1][y][x] == approx(
                w[0] * sdf['B'][0] * column_func(np.sqrt(real_x[x] ** 2 + real_y[y] ** 2) / sdf['h'][0], 2))

    w = sdf['m'] / (sdf['rho'] * sdf['h'] ** 3)
    image = interpolate_3d_cross(sdf, 'A', 0, x_pixels=50, y_pixels=25, x_min=-kernel.get_radius(),
                                 x_max=kernel.get_radius(), y_min=-kernel.get_radius(), y_max=kernel.get_radius())
    image_vec = interpolate_3d_cross_vec(sdf, 'A', 'B', 'C', 0, x_pixels=50, y_pixels=25, x_min=-kernel.get_radius(),
                                     x_max=kernel.get_radius(), y_min=-kernel.get_radius(), y_max=kernel.get_radius())

    for y in range(25):
        for x in range(50):
            assert image[y][x] == approx(
                w[0] * sdf['A'][0] * kernel.w(np.sqrt(real_x[x] ** 2 + real_y[y] ** 2 + 0.5 ** 2) / sdf['h'][0], 3))

            assert image_vec[0][y][x] == approx(
                w[0] * sdf['A'][0] * kernel.w(np.sqrt(real_x[x] ** 2 + real_y[y] ** 2 + 0.5 ** 2) / sdf['h'][0], 3))

            assert image_vec[1][y][x] == approx(
                w[0] * sdf['B'][0] * kernel.w(np.sqrt(real_x[x] ** 2 + real_y[y] ** 2 + 0.5 ** 2) / sdf['h'][0], 3))


@mark.parametrize("backend", backends)
def test_oob_particles(backend):
    df_2 = pd.DataFrame({'x': [0], 'y': [0], 'A': [4], 'B': [3], 'h': [1.9], 'rho': [0.4], 'm': [0.03]})
    sdf_2 = SarracenDataFrame(df_2, params=dict())
    df_3 = pd.DataFrame({'x': [0], 'y': [0], 'z': [-0.5], 'A': [4], 'B': [3], 'C': [2], 'h': [1.9], 'rho': [0.4],
                         'm': [0.03]})
    sdf_3 = SarracenDataFrame(df_3, params=dict())

    kernel = CubicSplineKernel()
    sdf_2.kernel = kernel
    sdf_2.backend = backend
    sdf_3.kernel = kernel
    sdf_3.backend = backend
    w = sdf_2['m'] / (sdf_2['rho'] * sdf_2['h'] ** 2)

    # A mapping of pixel indices to x / y values in particle space.
    real_x = 1 + (np.arange(0, 25) + 0.5) * (1 / 25)
    real_y = 1 + (np.arange(0, 25) + 0.5) * (1 / 25)

    image = interpolate_2d(sdf_2, 'A', x_pixels=25, y_pixels=25, x_min=1, x_max=2, y_min=1, y_max=2)
    image_vec = interpolate_2d_vec(sdf_2, 'A', 'B', x_pixels=25, y_pixels=25, x_min=1, x_max=2, y_min=1, y_max=2)
    line = interpolate_2d_cross(sdf_2, 'A', pixels=25, x1=1, x2=2, y1=1, y2=2)

    for y in range(25):
        assert line[y] == approx(
            w[0] * sdf_2['A'][0] * kernel.w(np.sqrt(real_x[y] ** 2 + real_y[y] ** 2) / sdf_2['h'][0], 2))
        for x in range(25):
            assert image[y][x] == approx(
                w[0] * sdf_2['A'][0] * kernel.w(np.sqrt(real_x[x] ** 2 + real_y[y] ** 2) / sdf_2['h'][0], 2))
            assert image_vec[0][y][x] == approx(
                w[0] * sdf_2['A'][0] * kernel.w(np.sqrt(real_x[x] ** 2 + real_y[y] ** 2) / sdf_2['h'][0], 2))
            assert image_vec[1][y][x] == approx(
                w[0] * sdf_2['B'][0] * kernel.w(np.sqrt(real_x[x] ** 2 + real_y[y] ** 2) / sdf_2['h'][0], 2))

    column_func = kernel.get_column_kernel_func(1000)
    image = interpolate_3d(sdf_3, 'A', x_pixels=25, y_pixels=25, x_min=1, x_max=2, y_min=1, y_max=2)
    image_vec = interpolate_3d_vec(sdf_3, 'A', 'B', 'C', x_pixels=25, y_pixels=25, x_min=1, x_max=2, y_min=1, y_max=2)

    for y in range(25):
        for x in range(25):
            assert image[y][x] == approx(
                w[0] * sdf_3['A'][0] * column_func(np.sqrt(real_x[x] ** 2 + real_y[y] ** 2) / sdf_3['h'][0], 2))
            assert image_vec[0][y][x] == approx(
                w[0] * sdf_3['A'][0] * column_func(np.sqrt(real_x[x] ** 2 + real_y[y] ** 2) / sdf_3['h'][0], 2))
            assert image_vec[1][y][x] == approx(
                w[0] * sdf_3['B'][0] * column_func(np.sqrt(real_x[x] ** 2 + real_y[y] ** 2) / sdf_3['h'][0], 2))

    w = sdf_3['m'] / (sdf_3['rho'] * sdf_3['h'] ** 3)
    image = interpolate_3d_cross(sdf_3, 'A', 0, x_pixels=25, y_pixels=25, x_min=1, x_max=2, y_min=1, y_max=2)
    image_vec = interpolate_3d_cross_vec(sdf_3, 'A', 'B', 'C', 0, x_pixels=25, y_pixels=25, x_min=1, x_max=2, y_min=1,
                                         y_max=2)

    for y in range(25):
        for x in range(25):
            assert image[y][x] == approx(
                w[0] * sdf_3['A'][0] * kernel.w(np.sqrt(real_x[x] ** 2 + real_y[y] ** 2 + 0.5 ** 2) / sdf_3['h'][0], 3))
            assert image_vec[0][y][x] == approx(
                w[0] * sdf_3['A'][0] * kernel.w(np.sqrt(real_x[x] ** 2 + real_y[y] ** 2 + 0.5 ** 2) / sdf_3['h'][0], 3))
            assert image_vec[1][y][x] == approx(
                w[0] * sdf_3['B'][0] * kernel.w(np.sqrt(real_x[x] ** 2 + real_y[y] ** 2 + 0.5 ** 2) / sdf_3['h'][0], 3))


@mark.parametrize("backend", backends)
def test_nonstandard_rotation(backend):
    df = pd.DataFrame({'x': [1], 'y': [1], 'z': [1], 'A': [4], 'h': [0.9], 'rho': [0.4], 'm': [0.03]})
    sdf = SarracenDataFrame(df, params=dict())
    kernel = CubicSplineKernel()
    sdf.kernel = kernel
    sdf.backend = backend

    column_kernel = kernel.get_column_kernel_func(1000)

    rot_z, rot_y, rot_x = 129, 34, 50

    image_col = interpolate_3d(sdf, 'A', x_pixels=50, y_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1,
                               rotation=[rot_z, rot_y, rot_x], origin=[0, 0, 0])
    image_cross = interpolate_3d_cross(sdf, 'A', 0, x_pixels=50, y_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1,
                                       rotation=[rot_z, rot_y, rot_x], origin=[0, 0, 0])

    w_col = sdf['m'] / (sdf['rho'] * sdf['h'] ** 2)
    w_cross = sdf['m'] / (sdf['rho'] * sdf['h'] ** 3)

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
            assert image_col[y][x] == approx(w_col[0] * sdf['A'][0] * column_kernel(
                np.sqrt((pos_x3 - real[x]) ** 2 + (pos_y3 - real[y]) ** 2) / sdf['h'][0], 3))
            assert image_cross[y][x] == approx(w_cross[0] * sdf['A'][0] * kernel.w(
                np.sqrt((pos_x3 - real[x]) ** 2 + (pos_y3 - real[y]) ** 2 + pos_z3 ** 2) / sdf['h'][0], 3))


@mark.parametrize("backend", backends)
def test_scipy_rotation_equivalency(backend):
    df = pd.DataFrame({'x': [1], 'y': [1], 'z': [1], 'A': [4], 'B': [3], 'C': [2], 'h': [0.9], 'rho': [0.4],
                       'm': [0.03]})
    sdf = SarracenDataFrame(df, params=dict())
    kernel = CubicSplineKernel()
    sdf.kernel = kernel
    sdf.backend = backend

    rot_z, rot_y, rot_x = 67, -34, 91

    image1 = interpolate_3d(sdf, 'A', x_pixels=50, y_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1,
                            rotation=[rot_z, rot_y, rot_x], origin=[0, 0, 0])
    image2 = interpolate_3d(sdf, 'A', x_pixels=50, y_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1,
                            rotation=Rotation.from_euler('zyx', [rot_z, rot_y, rot_x], degrees=True), origin=[0, 0, 0])
    assert_allclose(image1, image2)

    image1 = interpolate_3d_cross(sdf, 'A', 0, x_pixels=50, y_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1,
                                  rotation=[rot_z, rot_y, rot_x], origin=[0, 0, 0])
    image2 = interpolate_3d_cross(sdf, 'A', 0, x_pixels=50, y_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1,
                                  rotation=Rotation.from_euler('zyx', [rot_z, rot_y, rot_x], degrees=True),
                                  origin=[0, 0, 0])
    assert_allclose(image1, image2)

    image1 = interpolate_3d_vec(sdf, 'A', 'B', 'C', x_pixels=50, y_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1,
                                  rotation=[rot_z, rot_y, rot_x], origin=[0, 0, 0])
    image2 = interpolate_3d_vec(sdf, 'A', 'B', 'C', x_pixels=50, y_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1,
                                  rotation=Rotation.from_euler('zyx', [rot_z, rot_y, rot_x], degrees=True),
                                  origin=[0, 0, 0])
    assert_allclose(image1[0], image2[0])
    assert_allclose(image1[1], image2[1])

    image1 = interpolate_3d_cross_vec(sdf, 'A', 'B', 'C', 0, x_pixels=50, y_pixels=50, x_min=-1, x_max=1, y_min=-1,
                                      y_max=1, rotation=[rot_z, rot_y, rot_x], origin=[0, 0, 0])
    image2 = interpolate_3d_cross_vec(sdf, 'A', 'B', 'C', 0, x_pixels=50, y_pixels=50, x_min=-1, x_max=1, y_min=-1,
                                      y_max=1, rotation=Rotation.from_euler('zyx', [rot_z, rot_y, rot_x], degrees=True),
                                      origin=[0, 0, 0])
    assert_allclose(image1[0], image2[0])
    assert_allclose(image1[1], image2[1])


@mark.parametrize("backend", backends)
def test_quaternion_rotation(backend):
    df = pd.DataFrame({'x': [1], 'y': [1], 'z': [1], 'A': [4], 'B': [3], 'C': [2], 'h': [1.9], 'rho': [0.4],
                       'm': [0.03]})
    sdf = SarracenDataFrame(df, params=dict())
    kernel = CubicSplineKernel()
    sdf.kernel = kernel
    sdf.backend = backend

    column_kernel = kernel.get_column_kernel_func(1000)
    q = [5/np.sqrt(99), 3/np.sqrt(99), 8/np.sqrt(99), 1/np.sqrt(99)]
    quat = Rotation.from_quat(q)
    image_col = interpolate_3d(sdf, 'A', x_pixels=50, y_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1, rotation=quat,
                               origin=[0, 0, 0])
    image_colvec = interpolate_3d_vec(sdf, 'A', 'B', 'C', x_pixels=50, y_pixels=50, x_min=-1, x_max=1, y_min=-1,
                                      y_max=1, rotation=quat, origin=[0, 0, 0])
    image_cross = interpolate_3d_cross(sdf, 'A', 0, x_pixels=50, y_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1,
                                       rotation=quat, origin=[0, 0, 0])
    image_crossvec = interpolate_3d_cross_vec(sdf, 'A', 'B', 'C', 0, x_pixels=50, y_pixels=50, x_min=-1, x_max=1,
                                              y_min=-1, y_max=1, rotation=quat, origin=[0, 0, 0])

    w_col = sdf['m'] / (sdf['rho'] * sdf['h'] ** 2)
    w_cross = sdf['m'] / (sdf['rho'] * sdf['h'] ** 3)

    pos = quat.apply([1, 1, 1])
    val = quat.apply([sdf['A'][0], sdf['B'][0], sdf['C'][0]])
    real = -1 + (np.arange(0, 50) + 0.5) * (1 / 25)

    for y in range(50):
        for x in range(50):
            assert image_col[y][x] == approx(w_col[0] * sdf['A'][0] * column_kernel(
                np.sqrt((pos[0] - real[x]) ** 2 + (pos[1] - real[y]) ** 2) / sdf['h'][0], 3))

            assert image_colvec[0][y][x] == approx(w_col[0] * val[0] * column_kernel(
                np.sqrt((pos[0] - real[x]) ** 2 + (pos[1] - real[y]) ** 2) / sdf['h'][0], 3))
            assert image_colvec[1][y][x] == approx(w_col[0] * val[1] * column_kernel(
                np.sqrt((pos[0] - real[x]) ** 2 + (pos[1] - real[y]) ** 2) / sdf['h'][0], 3))

            assert image_cross[y][x] == approx(w_cross[0] * sdf['A'][0] * kernel.w(
                np.sqrt((pos[0] - real[x]) ** 2 + (pos[1] - real[y]) ** 2 + pos[2] ** 2) / sdf['h'][0], 3))

            assert image_crossvec[0][y][x] == approx(w_cross[0] * val[0] * kernel.w(
                np.sqrt((pos[0] - real[x]) ** 2 + (pos[1] - real[y]) ** 2 + pos[2] ** 2) / sdf['h'][0], 3))
            assert image_crossvec[1][y][x] == approx(w_cross[0] * val[1] * kernel.w(
                np.sqrt((pos[0] - real[x]) ** 2 + (pos[1] - real[y]) ** 2 + pos[2] ** 2) / sdf['h'][0], 3))


@mark.parametrize("backend", backends)
def test_rotation_stability(backend):
    df = pd.DataFrame({'x': [1, 3], 'y': [1, -1], 'z': [1, -0.5], 'A': [4, 3], 'B': [3, 2], 'C': [1, 1.5],
                       'h': [0.9, 1.4], 'rho': [0.4, 0.6], 'm': [0.03, 0.06]})
    sdf = SarracenDataFrame(df, params=dict())
    kernel = CubicSplineKernel()
    sdf.kernel = kernel
    sdf.backend = backend

    real = -1 + (np.arange(0, 50) + 0.5) * (1 / 25)
    pixel_x, pixel_y = 12, 30

    for func in [interpolate_3d, interpolate_3d_cross]:
        image = func(sdf, 'A', x_pixels=50, y_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1)
        image_rot = func(sdf, 'A', x_pixels=50, y_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1, rotation=[237, 0, 0],
                         origin=[real[pixel_x], real[pixel_y], 0])

        assert image[pixel_y][pixel_x] == approx(image_rot[pixel_y][pixel_x])


@mark.parametrize("backend", backends)
def test_axes_rotation_separation(backend):
    df = pd.DataFrame({'x': [-1, 1], 'y': [1, -1], 'z': [1, -1], 'A': [2, 1.5], 'B': [2, 2], 'C': [4, 3],
                       'h': [1.1, 1.3], 'rho': [0.55, 0.45], 'm': [0.04, 0.05]})
    sdf = SarracenDataFrame(df, params=dict())
    sdf.backend = backend

    image1 = interpolate_3d(sdf, 'A', x_pixels=50,  y_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1,
                            rotation=[234, 90, 48])
    image2 = interpolate_3d(sdf, 'A', x='y', y='x', x_pixels=50,  y_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1,
                            rotation=[234, 90, 48])
    assert_allclose(image1, image2.T)

    image1 = interpolate_3d_vec(sdf, 'A', 'B', 'C', x_pixels=50, y_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1,
                            rotation=[234, 90, 48])
    image2 = interpolate_3d_vec(sdf, 'A', 'B', 'C', x='y', y='x', x_pixels=50, y_pixels=50, x_min=-1, x_max=1, y_min=-1,
                                y_max=1, rotation=[234, 90, 48])
    assert_allclose(image1[0], image2[0].T)
    assert_allclose(image1[1], image2[1].T)

    image1 = interpolate_3d_cross(sdf, 'A', 0, x_pixels=50,  y_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1,
                                  rotation=[234, 90, 48])
    image2 = interpolate_3d_cross(sdf, 'A', 0, x='y', y='x', x_pixels=50,  y_pixels=50, x_min=-1, x_max=1, y_min=-1,
                                  y_max=1,rotation=[234, 90, 48])
    assert_allclose(image1, image2.T)

    image1 = interpolate_3d_cross_vec(sdf, 'A', 'B', 'C', 0, x_pixels=50, y_pixels=50, x_min=-1, x_max=1, y_min=-1,
                                      y_max=1, rotation=[234, 90, 48])
    image2 = interpolate_3d_cross_vec(sdf, 'A', 'B', 'C', 0, x='y', y='x', x_pixels=50, y_pixels=50, x_min=-1, x_max=1,
                                      y_min=-1, y_max=1, rotation=[234, 90, 48])
    assert_allclose(image1[0], image2[0].T)
    assert_allclose(image1[1], image2[1].T)


@mark.parametrize("backend", backends)
def test_axes_rotation_equivalency(backend):
    df = pd.DataFrame({'x': [-1, 1], 'y': [1, -1], 'z': [1, -1], 'A': [2, 1.5], 'B': [2, 2], 'C': [4, 3],
                       'h': [1.1, 1.3], 'rho': [0.55, 0.45], 'm': [0.04, 0.05]})
    sdf = SarracenDataFrame(df, params=dict())
    sdf.backend = backend

    x, y, z = 'x', 'y', 'z'
    flip_x, flip_y, flip_z = False, False, False
    for i_z in range(4):
        for i_y in range(4):
            for i_x in range(4):
                rot_x, rot_y, rot_z = i_x * 90, i_y * 90, i_z * 90

                for func in [interpolate_3d, interpolate_3d_cross]:
                    image1 = func(sdf, 'A', x_pixels=50, y_pixels=50, x_min=-1, x_max=1, y_min=-1, y_max=1,
                                            rotation=[rot_z, rot_y, rot_x], origin=[0, 0, 0])
                    image2 = func(sdf, 'A', x=x, y=y, x_pixels=50, y_pixels=50, x_min=-1, x_max=1, y_min=-1,
                                            y_max=1)
                    image2 = image2 if not flip_x else np.flip(image2, 1)
                    image2 = image2 if not flip_y else np.flip(image2, 0)
                    assert_allclose(image1, image2)

                y, z = z, y
                flip_y, flip_z = not flip_z, flip_y
            x, z = z, x
            flip_x, flip_z = flip_z, not flip_x
        x, y = y, x
        flip_x, flip_y = not flip_y, flip_x


@mark.parametrize("backend", backends)
def test_invalid_region(backend):
    df_2 = pd.DataFrame({'x': [0], 'y': [0], 'A': [4], 'B': [3], 'C': [2.5], 'h': [0.9], 'rho': [0.4], 'm': [0.03]})
    sdf_2 = SarracenDataFrame(df_2, params=dict())
    df_3 = pd.DataFrame({'x': [0], 'y': [0], 'z': [-0.5], 'A': [4], 'B': [3], 'C': [2.5], 'h': [0.9], 'rho': [0.4],
                         'm': [0.03]})
    sdf_3 = SarracenDataFrame(df_3, params=dict())

    sdf_2.backend = backend
    sdf_3.backend = backend

    for b in [(-3, 3, 3, -3, 20, 20), (3, 3, 3, 3, 20, 20), (-3, 3, -3, 3, 0, 0)]:
        with raises(ValueError):
            interpolate_2d(sdf_2, 'A', x_min=b[0], x_max=b[1], y_min=b[2], y_max=b[3], x_pixels=b[4], y_pixels=b[5])
        with raises(ValueError):
            interpolate_2d_vec(sdf_2, 'A', 'B', 'C', x_min=b[0], x_max=b[1], y_min=b[2], y_max=b[3], x_pixels=b[4],
                               y_pixels=b[5])
        # the first case will not fail for this type of interpolation.
        if not b[0] == -3 and not b[3] == -3:
            with raises(ValueError):
                interpolate_2d_cross(sdf_2, 'A', x1=b[0], x2=b[1], y1=b[2], y2=b[3], pixels=b[4])
        with raises(ValueError):
            interpolate_3d(sdf_3, 'A', x_min=b[0], x_max=b[1], y_min=b[2], y_max=b[3], x_pixels=b[4], y_pixels=b[5])
        with raises(ValueError):
            interpolate_3d_vec(sdf_3, 'A', 'B', 'C', x_min=b[0], x_max=b[1], y_min=b[2], y_max=b[3], x_pixels=b[4],
                               y_pixels=b[5])
        with raises(ValueError):
            interpolate_3d_cross(sdf_3, 'A', 0, x_min=b[0], x_max=b[1], y_min=b[2], y_max=b[3], x_pixels=b[4],
                                 y_pixels=b[5])
        with raises(ValueError):
            interpolate_3d_cross_vec(sdf_3, 'A', 'B', 'C', 0, x_min=b[0], x_max=b[1], y_min=b[2], y_max=b[3],
                                     x_pixels=b[4], y_pixels=b[5])


@mark.parametrize("backend", backends)
def test_required_columns(backend):
    df_2 = pd.DataFrame({'x': [-1, 1], 'y': [1, -1], 'A': [2, 1.5], 'B': [5, 4], 'C': [3, 2], 'h': [1.1, 1.3],
                         'rho': [0.55, 0.45], 'm': [0.04, 0.05]})
    sdf_2 = SarracenDataFrame(df_2, params=dict())
    df_3 = pd.DataFrame({'x': [-1, 1], 'y': [1, -1], 'z': [1, -1], 'A': [2, 1.5], 'B': [5, 4], 'C': [3, 2],
                         'h': [1.1, 1.3], 'rho': [0.55, 0.45], 'm': [0.04, 0.05]})
    sdf_3 = SarracenDataFrame(df_3, params=dict())

    sdf_2.backend = backend
    sdf_3.backend = backend

    for column in ['target', 'mass', 'rho', 'h']:
        sdf_dropped = sdf_2.drop(column, axis=1)
        with raises(KeyError):
            interpolate_2d(sdf_dropped, 'A')
        with raises(KeyError):
            interpolate_2d_cross(sdf_dropped, 'A')
        with raises(KeyError):
            interpolate_2d_vec(sdf_dropped, 'A')

        sdf_dropped = sdf_3.drop(column, axis=1)
        with raises(KeyError):
            interpolate_3d(sdf_dropped, 'A')
        with raises(KeyError):
            interpolate_3d_cross(sdf_dropped, 'A')
        with raises(KeyError):
            interpolate_3d_vec(sdf_dropped, 'A', 'B', 'C')
        with raises(KeyError):
            interpolate_3d_cross_vec(sdf_dropped, 'A', 'B', 'C')
