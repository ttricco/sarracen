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
from sarracen.interpolate import interpolate_2d, interpolate_2d_line, interpolate_3d_cross, interpolate_3d_proj, \
    interpolate_2d_vec, interpolate_3d_vec, interpolate_3d_cross_vec, interpolate_3d_grid, interpolate_3d_line


backends = ['cpu']
if cuda.is_available():
    backends.append('gpu')


@mark.parametrize("backend", backends)
def test_single_particle(backend):
    """
    The result of interpolation over a single particle should be equal to scaled kernel
    values at each point of the image.
    """
    df = pd.DataFrame({'x': [0], 'y': [0], 'A': [4], 'B': [5], 'h': [0.9], 'rho': [0.4], 'm': [0.03]})
    sdf = SarracenDataFrame(df, params=dict())
    kernel = CubicSplineKernel()
    sdf.kernel = kernel
    sdf.backend = backend

    # Weight for 2D interpolation & 3D column interpolation.
    w = sdf['m'] / (sdf['rho'] * sdf['h'] ** 2)

    # A mapping of pixel indices to x / y values in particle space.
    real = -kernel.get_radius() + (np.arange(0, 25) + 0.5) * (2 * kernel.get_radius() / 25)

    image = interpolate_2d(sdf, 'A', x_pixels=25,  y_pixels=25, xlim=(-kernel.get_radius(), kernel.get_radius()),
                           ylim=(-kernel.get_radius(), kernel.get_radius()), normalize=False, hmin=False)
    image_vec = interpolate_2d_vec(sdf, 'A', 'B', x_pixels=25, y_pixels=25,
                                   xlim=(-kernel.get_radius(), kernel.get_radius()),
                                   ylim=(-kernel.get_radius(), kernel.get_radius()), normalize=False, hmin=False)
    for y in range(25):
        for x in range(25):
            assert image[y][x] ==\
                   approx(w[0] * sdf['A'][0] * kernel.w(np.sqrt(real[x] ** 2 + real[y] ** 2) / sdf['h'][0], 2))
            assert image_vec[0][y][x] ==\
                   approx(w[0] * sdf['A'][0] * kernel.w(np.sqrt(real[x] ** 2 + real[y] ** 2) / sdf['h'][0], 2))
            assert image_vec[1][y][x] == \
                   approx(w[0] * sdf['B'][0] * kernel.w(np.sqrt(real[x] ** 2 + real[y] ** 2) / sdf['h'][0], 2))

    image = interpolate_2d_line(sdf, 'A', pixels=25, xlim=(-kernel.get_radius(), kernel.get_radius()),
                                 ylim=(-kernel.get_radius(), kernel.get_radius()), normalize=False, hmin=False)
    for x in range(25):
        assert image[x] == approx(w[0] * sdf['A'][0] * kernel.w(np.sqrt(2) * np.abs(real[x]) / sdf['h'][0], 2))

    # Convert the previous 2D dataframe to a 3D dataframe.
    sdf['z'] = -0.5
    sdf['C'] = 10
    sdf.zcol = 'z'

    column_func = kernel.get_column_kernel_func(1000)

    image = interpolate_3d_proj(sdf, 'A', x_pixels=25, y_pixels=25, xlim=(-kernel.get_radius(), kernel.get_radius()),
                           ylim=(-kernel.get_radius(), kernel.get_radius()), dens_weight=False, normalize=False, hmin=False)
    image_vec = interpolate_3d_vec(sdf, 'A', 'B', 'C', x_pixels=25, y_pixels=25,
                                   xlim=(-kernel.get_radius(), kernel.get_radius()),
                                   ylim=(-kernel.get_radius(), kernel.get_radius()), dens_weight=False, normalize=False, hmin=False)
    for y in range(25):
        for x in range(25):
            assert image[y][x] ==\
                   approx(w[0] * sdf['A'][0] * column_func(np.sqrt(real[x] ** 2 + real[y] ** 2) / sdf['h'][0], 2))
            assert image_vec[0][y][x] == \
                   approx(w[0] * sdf['A'][0] * column_func(np.sqrt(real[x] ** 2 + real[y] ** 2) / sdf['h'][0], 2))
            assert image_vec[1][y][x] == \
                   approx(w[0] * sdf['B'][0] * column_func(np.sqrt(real[x] ** 2 + real[y] ** 2) / sdf['h'][0], 2))

    # Weight for 3D cross-sections.
    w = sdf['m'] / (sdf['rho'] * sdf['h'] ** 3)

    image = interpolate_3d_cross(sdf, 'A', z_slice=0, x_pixels=25, y_pixels=25,
                                 xlim=(-kernel.get_radius(), kernel.get_radius()),
                                 ylim=(-kernel.get_radius(), kernel.get_radius()), normalize=False, hmin=False)
    image_vec = interpolate_3d_cross_vec(sdf, 'A', 'B', 'C', 0, x_pixels=25, y_pixels=25,
                                         xlim=(-kernel.get_radius(), kernel.get_radius()),
                                         ylim=(-kernel.get_radius(), kernel.get_radius()), normalize=False, hmin=False)
    for y in range(25):
        for x in range(25):
            assert image[y][x] == approx(w[0] * sdf['A'][0] *
                                         kernel.w(np.sqrt(real[x] ** 2 + real[y] ** 2 + 0.5 ** 2) / sdf['h'][0], 3))
            assert image_vec[0][y][x] == approx(w[0] * sdf['A'][0] *
                                                kernel.w(np.sqrt(real[x] ** 2 + real[y] ** 2 + 0.5 ** 2)
                                                         / sdf['h'][0], 3))
            assert image_vec[1][y][x] == approx(w[0] * sdf['B'][0] *
                                                kernel.w(np.sqrt(real[x] ** 2 + real[y] ** 2 + 0.5 ** 2)
                                                         / sdf['h'][0], 3))

    bounds = (-kernel.get_radius(), kernel.get_radius())
    image = interpolate_3d_grid(sdf, 'A', x_pixels=25, y_pixels=25, z_pixels=25, xlim=bounds, ylim=bounds,
                                zlim=bounds, normalize=False, hmin=False)
    for z in range(25):
        for y in range(25):
            for x in range(25):
                assert image[z][y][x] == approx(w[0] * sdf['A'][0] *
                                                kernel.w(np.sqrt(real[x] ** 2 + real[y] ** 2 + (real[z] + 0.5) ** 2)
                                                         / sdf['h'][0], 3))

    image = interpolate_3d_line(sdf, 'A', pixels=25, xlim=(-kernel.get_radius(), kernel.get_radius()),
                                ylim=(-kernel.get_radius(), kernel.get_radius()),
                                zlim=(-kernel.get_radius(), kernel.get_radius()), normalize=False, hmin=False)
    for x in range(25):
        assert image[x] == approx(w[0] * sdf['A'][0] *
                                  kernel.w(np.sqrt(2 * real[x] ** 2 + (real[x] + 0.5) ** 2) / sdf['h'][0], 3))


@mark.parametrize("backend", backends)
def test_single_repeated_particle(backend):
    """
    The result of interpolation over a single particle repeated several times should be equal to scaled kernel
    values at each point of the image multiplied by the number of particles.

    If this test fails, it is likely that there is a race condition issue in the interpolation implementation.
    """

    repetitions = 10000
    df = pd.concat([pd.DataFrame({'x': [0], 'y': [0], 'A': [4], 'B': [5], 'h': [0.9], 'rho': [0.4],
                                  'm': [0.03]})] * repetitions, ignore_index=True)
    sdf = SarracenDataFrame(df, params=dict())

    kernel = CubicSplineKernel()
    sdf.kernel = kernel
    sdf.backend = backend

    # Multiplying by repetitions here is done for ease of use.
    w = repetitions * sdf['m'] / (sdf['rho'] * sdf['h'] ** 2)

    # A mapping of pixel indices to x / y values in particle space.
    real = -kernel.get_radius() + (np.arange(0, 25) + 0.5) * (2 * kernel.get_radius() / 25)

    image = interpolate_2d(sdf, 'A', x_pixels=25, y_pixels=25, xlim=(-kernel.get_radius(), kernel.get_radius()),
                           ylim=(-kernel.get_radius(), kernel.get_radius()), normalize=False, hmin=False)
    image_vec = interpolate_2d_vec(sdf, 'A', 'B', x_pixels=25, y_pixels=25,
                                   xlim=(-kernel.get_radius(), kernel.get_radius()),
                                   ylim=(-kernel.get_radius(), kernel.get_radius()), normalize=False, hmin=False)
    for y in range(25):
        for x in range(25):
            assert image[y][x] == \
                   approx(w[0] * sdf['A'][0] * kernel.w(np.sqrt(real[x] ** 2 + real[y] ** 2) / sdf['h'][0], 2))
            assert image_vec[0][y][x] == \
                   approx(w[0] * sdf['A'][0] * kernel.w(np.sqrt(real[x] ** 2 + real[y] ** 2) / sdf['h'][0], 2))
            assert image_vec[1][y][x] == \
                   approx(w[0] * sdf['B'][0] * kernel.w(np.sqrt(real[x] ** 2 + real[y] ** 2) / sdf['h'][0], 2))

    image = interpolate_2d_line(sdf, 'A', pixels=25, xlim=(-kernel.get_radius(), kernel.get_radius()),
                                 ylim=(-kernel.get_radius(), kernel.get_radius()), normalize=False, hmin=False)
    for x in range(25):
        assert image[x] == approx(w[0] * sdf['A'][0] * kernel.w(np.sqrt(2) * np.abs(real[x]) / sdf['h'][0], 2))

    # Convert the previous 2D dataframe to a 3D dataframe.
    sdf['z'] = -0.5
    sdf['C'] = 10
    sdf.zcol = 'z'

    column_func = kernel.get_column_kernel_func(1000)

    image = interpolate_3d_proj(sdf, 'A', x_pixels=25, y_pixels=25, xlim=(-kernel.get_radius(), kernel.get_radius()),
                           ylim=(-kernel.get_radius(), kernel.get_radius()), dens_weight=False, normalize=False, hmin=False)
    image_vec = interpolate_3d_vec(sdf, 'A', 'B', 'C', x_pixels=25, y_pixels=25,
                                   xlim=(-kernel.get_radius(), kernel.get_radius()),
                                   ylim=(-kernel.get_radius(), kernel.get_radius()), dens_weight=False, normalize=False, hmin=False)
    for y in range(25):
        for x in range(25):
            assert image[y][x] == \
                   approx(w[0] * sdf['A'][0] * column_func(np.sqrt(real[x] ** 2 + real[y] ** 2) / sdf['h'][0], 2))
            assert image_vec[0][y][x] == \
                   approx(w[0] * sdf['A'][0] * column_func(np.sqrt(real[x] ** 2 + real[y] ** 2) / sdf['h'][0], 2))
            assert image_vec[1][y][x] == \
                   approx(w[0] * sdf['B'][0] * column_func(np.sqrt(real[x] ** 2 + real[y] ** 2) / sdf['h'][0], 2))

    # Weight for 3D cross-sections
    w = repetitions * sdf['m'] / (sdf['rho'] * sdf['h'] ** 3)

    image = interpolate_3d_cross(sdf, 'A', z_slice=0, x_pixels=25, y_pixels=25,
                                 xlim=(-kernel.get_radius(), kernel.get_radius()),
                                 ylim=(-kernel.get_radius(), kernel.get_radius()), normalize=False, hmin=False)
    image_vec = interpolate_3d_cross_vec(sdf, 'A', 'B', 'C', 0, x_pixels=25, y_pixels=25,
                                         xlim=(-kernel.get_radius(), kernel.get_radius()),
                                         ylim=(-kernel.get_radius(), kernel.get_radius()), normalize=False, hmin=False)
    for y in range(25):
        for x in range(25):
            assert image[y][x] == approx(w[0] * sdf['A'][0] *
                                         kernel.w(np.sqrt(real[x] ** 2 + real[y] ** 2 + 0.5 ** 2) / sdf['h'][0], 3))
            assert image_vec[0][y][x] == approx(w[0] * sdf['A'][0] *
                                                kernel.w(np.sqrt(real[x] ** 2 + real[y] ** 2 + 0.5 ** 2)
                                                         / sdf['h'][0], 3))
            assert image_vec[1][y][x] == approx(w[0] * sdf['B'][0] *
                                                kernel.w(np.sqrt(real[x] ** 2 + real[y] ** 2 + 0.5 ** 2)
                                                         / sdf['h'][0], 3))

    bounds = (-kernel.get_radius(), kernel.get_radius())
    image = interpolate_3d_grid(sdf, 'A', x_pixels=25, y_pixels=25, z_pixels=25, xlim=bounds, ylim=bounds, zlim=bounds,
                                normalize=False, hmin=False)
    for z in range(25):
        for y in range(25):
            for x in range(25):
                assert image[z][y][x] == approx(w[0] * sdf['A'][0] *
                                                kernel.w(np.sqrt(real[x] ** 2 + real[y] ** 2 + (real[z] + 0.5) ** 2)
                                                         / sdf['h'][0], 3))

    image = interpolate_3d_line(sdf, 'A', pixels=25, xlim=(-kernel.get_radius(), kernel.get_radius()),
                                ylim=(-kernel.get_radius(), kernel.get_radius()),
                                zlim=(-kernel.get_radius(), kernel.get_radius()), normalize=False, hmin=False)
    for x in range(25):
        assert image[x] == approx(w[0] * sdf['A'][0] *
                                  kernel.w(np.sqrt(2 * real[x] ** 2 + (real[x] + 0.5) ** 2) / sdf['h'][0], 3))


@mark.parametrize("backend", backends)
def test_dimension_check(backend):
    """
    Passing a dataframe with invalid dimensions should raise a TypeError for all interpolation functions.
    """
    # First, test a basic 2D dataframe passed to 3D interpolation functions.
    df = pd.DataFrame({'x': [0, 1], 'y': [0, 1], 'P': [1, 1], 'Ax': [1, 1], 'Ay': [1, 1], 'h': [1, 1],
                       'rho': [1, 1], 'm': [1, 1]})
    sdf = SarracenDataFrame(df, params=dict())
    sdf.backend = backend

    for func in [interpolate_3d_proj, interpolate_3d_cross]:
        with raises(TypeError):
            func(sdf, 'P', normalize=False, hmin=False)
    for func in [interpolate_3d_vec, interpolate_3d_cross_vec, interpolate_3d_grid]:
        with raises(TypeError):
            func(sdf, 'Ax', 'Ay', 'Az', normalize=False, hmin=False)

    # Next, test a basic 3D dataframe passed to 2D interpolation functions.
    df = pd.DataFrame({'x': [0, 1], 'y': [0, 1], 'z': [0, 1], 'P': [1, 1], 'Ax': [1, 1], 'Ay': [1, 1], 'Az': [1, 1],
                       'h': [1, 1], 'rho': [1, 1], 'm': [1, 1]})
    sdf = SarracenDataFrame(df, params=dict())
    sdf.backend = backend

    for func in [interpolate_2d, interpolate_2d_line, interpolate_3d_line]:
        with raises(TypeError):
            func(sdf, 'P', normalize=False, hmin=False)
    with raises(TypeError):
        interpolate_2d_vec(sdf, 'Ax', 'Ay', normalize=False, hmin=False)


@mark.parametrize("backend", backends)
def test_3d_xsec_equivalency(backend):
    """
    A single 3D column integration of a dataframe should be equivalent to the average of several evenly spaced 3D
    cross-sections.
    """
    df = pd.DataFrame({'x': [0], 'y': [0], 'z': [0], 'A': [4], 'B': [6], 'C': [2], 'h': [0.9], 'rho': [0.4],
                       'm': [0.03]})
    sdf = SarracenDataFrame(df, params=dict())
    kernel = CubicSplineKernel()
    sdf.kernel = kernel
    sdf.backend = backend

    samples = 250

    column_image = interpolate_3d_proj(sdf, 'A', x_pixels=50, xlim=(-1, 1), ylim=(-1, 1),
                                       dens_weight=False, normalize=False, hmin=False)
    column_image_vec = interpolate_3d_vec(sdf, 'A', 'B', 'C', x_pixels=50, xlim=(-1, 1), ylim=(-1, 1),
                                          dens_weight=False, normalize=False, hmin=False)

    xsec_image = np.zeros((50, 50))
    xsec_image_vec = [np.zeros((50, 50)), np.zeros((50, 50))]
    for z in np.linspace(0, kernel.get_radius() * sdf['h'][0], samples):
        xsec_image += interpolate_3d_cross(sdf, 'A', z_slice=z, x_pixels=50, xlim=(-1, 1), ylim=(-1, 1),
                                           normalize=False, hmin=False)

        vec_sample = interpolate_3d_cross_vec(sdf, 'A', 'B', 'C', z, x_pixels=50, xlim=(-1, 1), ylim=(-1, 1),
                                              normalize=False, hmin=False)
        xsec_image_vec[0] += vec_sample[0]
        xsec_image_vec[1] += vec_sample[1]

    # Scale each cross-section summation to be equivalent to the column integration.
    xsec_image *= kernel.get_radius() * sdf['h'][0] * 2 / samples
    xsec_image_vec[0] *= kernel.get_radius() * sdf['h'][0] * 2 / samples
    xsec_image_vec[1] *= kernel.get_radius() * sdf['h'][0] * 2 / samples

    # The tolerances are lower here to accommodate for the relatively low sample size. A larger number of samples
    # would result in an unacceptable test time for the GPU backend (which already doesn't perform well with repeated
    # interpolation of just one particle)
    assert_allclose(xsec_image, column_image, rtol=1e-3, atol=1e-4)
    assert_allclose(xsec_image_vec[0], column_image_vec[0], rtol=1e-3, atol=1e-4)
    assert_allclose(xsec_image_vec[1], column_image_vec[1], rtol=1e-3, atol=1e-4)


@mark.parametrize("backend", backends)
def test_2d_xsec_equivalency(backend):
    """
    A single 2D interpolation should be equivalent to several combined 2D cross-sections.
    """
    # This test currently fails on both backends, since a vertical 2D cross-section currently
    # returns zero for an unknown reason.
    df = pd.DataFrame({'x': [0], 'y': [0], 'A': [4], 'h': [0.9], 'rho': [0.4], 'm': [0.03]})
    sdf = SarracenDataFrame(df, params=dict())
    kernel = CubicSplineKernel()
    sdf.kernel = kernel
    sdf.backend = backend

    true_image = interpolate_2d(sdf, 'A', x_pixels=50, xlim=(-1, 1), ylim=(-1, 1), normalize=False, hmin=False)

    # A mapping of pixel indices to x & y values in particle space.
    real = -1 + (np.arange(0, 50) + 0.5) * (2 / 50)

    reconstructed_image = np.zeros((50, 50))
    for y in range(50):
        reconstructed_image[y, :] = interpolate_2d_line(sdf, 'A', pixels=50, xlim=(-1, 1), ylim=(real[y], real[y]), normalize=False, hmin=False)
    assert_allclose(reconstructed_image, true_image)

    # reconstructed_image = np.zeros((50, 50))
    # for x in range(50):
    #     reconstructed_image[:, x] = interpolate_2d_line(sdf, 'A', pixels=50, xlim=(real[x], real[x]), ylim=(-1, 1))
    # assert_allclose(reconstructed_image, true_image)


@mark.parametrize("backend", backends)
def test_corner_particles(backend):
    """
    Interpolation over a dataset with two particles should be equal to the sum of contributions at each point.
    """
    kernel = CubicSplineKernel()

    df_2 = pd.DataFrame({'x': [-1, 1], 'y': [-1, 1], 'A': [2, 1.5], 'B': [5, 2.3], 'h': [1.1, 1.3], 'rho': [0.55, 0.45],
                         'm': [0.04, 0.05]})
    sdf_2 = SarracenDataFrame(df_2, params=dict())
    sdf_2.kernel = kernel
    sdf_2.backend = backend

    df_3 = pd.DataFrame({'x': [-1, 1], 'y': [-1, 1], 'z': [-1, 1], 'A': [2, 1.5], 'B': [2, 1], 'C': [7, 8],
                         'h': [1.1, 1.3], 'rho': [0.55, 0.45], 'm': [0.04, 0.05]})
    sdf_3 = SarracenDataFrame(df_3, params=dict())
    sdf_3.kernel = kernel
    sdf_3.backend = backend

    # Weight for 2D interpolation, and 3D column interpolation.
    w = sdf_2['m'] / (sdf_2['rho'] * sdf_2['h'] ** 2)

    # A mapping of pixel indices to x / y values in particle space.
    real = (np.arange(0, 25) + 0.5) * (2 / 25)

    image = interpolate_2d(sdf_2, 'A', x_pixels=25,  y_pixels=25, normalize=False, hmin=False)
    image_vec = interpolate_2d_vec(sdf_2, 'A', 'B', x_pixels=25,  y_pixels=25, normalize=False, hmin=False)
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

    image = interpolate_2d_line(sdf_2, 'A', pixels=25, normalize=False, hmin=False)
    for x in range(25):
        assert approx(w[0] * sdf_2['A'][0] * kernel.w(np.sqrt(real[x] ** 2 + real[x] ** 2) / sdf_2['h'][0], 2)
                      + w[1] * sdf_2['A'][1] * kernel.w(np.sqrt(real[24 - x] ** 2 + real[24 - x] ** 2)
                                                        / sdf_2['h'][1], 2)) == image[x]

    c_kernel = kernel.get_column_kernel_func(1000)

    image = interpolate_3d_proj(sdf_3, 'A', x_pixels=25, y_pixels=25,
                                dens_weight=False, normalize=False, hmin=False)
    image_vec = interpolate_3d_vec(sdf_3, 'A', 'B', 'C', x_pixels=25, y_pixels=25,
                                   dens_weight=False, normalize=False, hmin=False)
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

    # Weight for 3D cross-section interpolation.
    w = sdf_3['m'] / (sdf_3['rho'] * sdf_3['h'] ** 3)

    image = interpolate_3d_cross(sdf_3, 'A', x_pixels=25, y_pixels=25, normalize=False, hmin=False)
    image_vec = interpolate_3d_cross_vec(sdf_3, 'A', 'B', 'C', x_pixels=25, y_pixels=25, normalize=False, hmin=False)
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

    image = interpolate_3d_grid(sdf_3, 'A', x_pixels=25, y_pixels=25, z_pixels=25, normalize=False, hmin=False)
    for z in range(25):
        for y in range(25):
            for x in range(25):
                assert approx(
                    w[0] * sdf_3['A'][0] * kernel.w(np.sqrt(real[x] ** 2 + real[y] ** 2 + real[z] ** 2)
                                                    / sdf_3['h'][0], 3) +
                    w[1] * sdf_3['A'][1] * kernel.w(np.sqrt(real[24 - x] ** 2 + real[24 - y] ** 2 + real[24 - z] ** 2)
                                                    / sdf_3['h'][1], 3)) == image[z][y][x]


@mark.parametrize("backend", backends)
def test_image_transpose(backend):
    """
    Interpolation with flipped x & y axes should be equivalent to the transpose of regular interpolation.
    """
    df = pd.DataFrame({'x': [-1, 1], 'y': [1, -1], 'A': [2, 1.5], 'B': [5, 4], 'h': [1.1, 1.3], 'rho': [0.55, 0.45],
                       'm': [0.04, 0.05]})
    sdf = SarracenDataFrame(df, params=dict())
    sdf.backend = backend

    image1 = interpolate_2d(sdf, 'A', x_pixels=20,  y_pixels=20, normalize=False, hmin=False)
    image2 = interpolate_2d(sdf, 'A', x='y', y='x', x_pixels=20,  y_pixels=20, normalize=False, hmin=False)
    assert_allclose(image1, image2.T)

    image1 = interpolate_2d_vec(sdf, 'A', 'B', x_pixels=20, y_pixels=20, normalize=False, hmin=False)
    image2 = interpolate_2d_vec(sdf, 'A', 'B', x='y', y='x', x_pixels=20, y_pixels=20, normalize=False, hmin=False)
    assert_allclose(image1[0], image2[0].T)
    assert_allclose(image1[1], image2[1].T)

    df = pd.DataFrame({'x': [-1, 1], 'y': [1, -1], 'z': [-1, 1], 'A': [2, 1.5], 'B': [5, 4], 'C': [2.5, 3],
                       'h': [1.1, 1.3], 'rho': [0.55, 0.45], 'm': [0.04, 0.05]})
    sdf = SarracenDataFrame(df, params=dict())

    image1 = interpolate_3d_proj(sdf, 'A', x_pixels=20,  y_pixels=20, normalize=False, hmin=False)
    image2 = interpolate_3d_proj(sdf, 'A', x='y', y='x', x_pixels=20,  y_pixels=20, normalize=False, hmin=False)
    assert_allclose(image1, image2.T)

    image1 = interpolate_3d_vec(sdf, 'A', 'B', 'C', x_pixels=50, y_pixels=50, normalize=False, hmin=False)
    image2 = interpolate_3d_vec(sdf, 'A', 'B', 'C', x='y', y='x', x_pixels=50, y_pixels=50, normalize=False, hmin=False)
    assert_allclose(image1[0], image2[0].T)
    assert_allclose(image1[1], image2[1].T)

    image1 = interpolate_3d_cross(sdf, 'A', x_pixels=50, y_pixels=50, normalize=False, hmin=False)
    image2 = interpolate_3d_cross(sdf, 'A', x='y', y='x', x_pixels=50, y_pixels=50, normalize=False, hmin=False)
    assert_allclose(image1, image2.T)

    image1 = interpolate_3d_cross_vec(sdf, 'A', 'B', 'C', x_pixels=20, y_pixels=20, normalize=False, hmin=False)
    image2 = interpolate_3d_cross_vec(sdf, 'A', 'B', 'C', x='y', y='x', x_pixels=20, y_pixels=20, normalize=False, hmin=False)
    assert_allclose(image1[0], image2[0].T)
    assert_allclose(image1[1], image2[1].T)

    image1 = interpolate_3d_grid(sdf, 'A', x_pixels=20, y_pixels=20, normalize=False, hmin=False)
    image2 = interpolate_3d_grid(sdf, 'A', x='y', y='x', x_pixels=20, y_pixels=20, normalize=False, hmin=False)
    assert_allclose(image1, image2.transpose(0, 2, 1))


@mark.parametrize("backend", backends)
def test_default_kernel(backend):
    """
    Interpolation should use the kernel supplied to the function. If no kernel is supplied, the kernel attached to the
    dataframe should be used.
    """
    df_2 = pd.DataFrame({'x': [0], 'y': [0], 'A': [1], 'B': [1], 'h': [1], 'rho': [1], 'm': [1]})
    sdf_2 = SarracenDataFrame(df_2, params=dict())
    df_3 = pd.DataFrame({'x': [0], 'y': [0], 'z': [0], 'A': [1], 'B': [1], 'C': [1], 'h': [1], 'rho': [1], 'm': [1]})
    sdf_3 = SarracenDataFrame(df_3, params=dict())

    kernel = QuarticSplineKernel()
    sdf_2.kernel = kernel
    sdf_3.kernel = kernel
    sdf_2.backend = backend
    sdf_3.backend = backend

    # First, test that the dataframe kernel is used in cases with no kernel supplied.

    # Each interpolation is performed over one pixel, offering an easy way to check the kernel used by the function.
    image = interpolate_2d(sdf_2, 'A', x_pixels=1, y_pixels=1, xlim=(-1, 1), ylim=(-1, 1), normalize=False, hmin=False)
    assert image == kernel.w(0, 2)
    image = interpolate_2d_vec(sdf_2, 'A', 'B', x_pixels=1, y_pixels=1, xlim=(-1, 1), ylim=(-1, 1), normalize=False, hmin=False)
    assert image[0] == kernel.w(0, 2)
    assert image[1] == kernel.w(0, 2)

    image = interpolate_2d_line(sdf_2, 'A', pixels=1, xlim=(-1, 1), ylim=(-1, 1), normalize=False, hmin=False)
    assert image == kernel.w(0, 2)

    image = interpolate_3d_proj(sdf_3, 'A', x_pixels=1, y_pixels=1, xlim=(-1, 1), ylim=(-1, 1), normalize=False, hmin=False)
    assert image == kernel.get_column_kernel()[0]
    image = interpolate_3d_vec(sdf_3, 'A', 'B', 'C', x_pixels=1, y_pixels=1, xlim=(-1, 1), ylim=(-1, 1), normalize=False, hmin=False)
    assert image[0] == kernel.get_column_kernel()[0]
    assert image[1] == kernel.get_column_kernel()[0]

    image = interpolate_3d_cross(sdf_3, 'A', x_pixels=1, y_pixels=1, xlim=(-1, 1), ylim=(-1, 1), normalize=False, hmin=False)
    assert image == kernel.w(0, 3)
    image = interpolate_3d_cross_vec(sdf_3, 'A', 'B', 'C', x_pixels=1, y_pixels=1, xlim=(-1, 1), ylim=(-1, 1), normalize=False, hmin=False)
    assert image[0] == kernel.w(0, 3)
    assert image[1] == kernel.w(0, 3)

    image = interpolate_3d_grid(sdf_3, 'A', x_pixels=1, y_pixels=1, z_pixels=1, xlim=(-1, 1), ylim=(-1, 1),
                                zlim=(-1, 1), normalize=False, hmin=False)
    assert image == kernel.w(0, 3)

    image = interpolate_3d_line(sdf_3, 'A', pixels=1, xlim=(-1, 1), ylim=(-1, 1), normalize=False, hmin=False)
    assert image == kernel.w(0, 3)

    # Next, test that the kernel supplied to the function is actually used.
    kernel = QuinticSplineKernel()
    image = interpolate_2d(sdf_2, 'A', x_pixels=1, y_pixels=1, xlim=(-1, 1), ylim=(-1, 1), kernel=kernel, normalize=False, hmin=False)
    assert image == kernel.w(0, 2)
    image = interpolate_2d_vec(sdf_2, 'A', 'B', x_pixels=1, y_pixels=1, xlim=(-1, 1), ylim=(-1, 1), kernel=kernel, normalize=False, hmin=False)
    assert image[0] == kernel.w(0, 2)
    assert image[1] == kernel.w(0, 2)

    image = interpolate_2d_line(sdf_2, 'A', pixels=1, xlim=(-1, 1), ylim=(-1, 1), kernel=kernel, normalize=False, hmin=False)
    assert image == kernel.w(0, 2)

    image = interpolate_3d_proj(sdf_3, 'A', x_pixels=1, y_pixels=1, xlim=(-1, 1), ylim=(-1, 1), kernel=kernel, normalize=False, hmin=False)
    assert image == kernel.get_column_kernel()[0]
    image = interpolate_3d_vec(sdf_3, 'A', 'B', 'C', x_pixels=1, y_pixels=1, xlim=(-1, 1), ylim=(-1, 1), kernel=kernel, normalize=False, hmin=False)
    assert image[0] == kernel.get_column_kernel()[0]
    assert image[1] == kernel.get_column_kernel()[0]

    image = interpolate_3d_cross(sdf_3, 'A', kernel=kernel, x_pixels=1, y_pixels=1, xlim=(-1, 1), ylim=(-1, 1), normalize=False, hmin=False)
    assert image == kernel.w(0, 3)
    image = interpolate_3d_cross_vec(sdf_3, 'A', 'B', 'C', x_pixels=1, y_pixels=1, xlim=(-1, 1), ylim=(-1, 1),
                                     kernel=kernel, normalize=False, hmin=False)
    assert image[0] == kernel.w(0, 3)
    assert image[1] == kernel.w(0, 3)

    image = interpolate_3d_grid(sdf_3, 'A', x_pixels=1, y_pixels=1, z_pixels=1, xlim=(-1, 1), ylim=(-1, 1),
                                zlim=(-1, 1), kernel=kernel, normalize=False, hmin=False)
    assert image == kernel.w(0, 3)

    image = interpolate_3d_line(sdf_3, 'A', pixels=1, xlim=(-1, 1), ylim=(-1, 1), kernel=kernel, normalize=False, hmin=False)
    assert image == kernel.w(0, 3)


@mark.parametrize("backend", backends)
def test_column_samples(backend):
    """
    3D column interpolation should use the number of integral samples supplied as an argument.
    """
    df_3 = pd.DataFrame({'x': [0], 'y': [0], 'z': [0], 'A': [1], 'h': [1], 'rho': [1], 'm': [1]})
    sdf_3 = SarracenDataFrame(df_3, params=dict())
    kernel = QuinticSplineKernel()
    sdf_3.kernel = kernel
    sdf_3.backend = backend

    # 2 samples is used here, since a column kernel with 2 samples will be drastically different than the
    # default kernel of 1000 samples.
    image = interpolate_3d_proj(sdf_3, 'A', x_pixels=1, y_pixels=1, xlim=(-1, 1), ylim=(-1, 1),
                                integral_samples=2, normalize=False, hmin=False)
    assert image == kernel.get_column_kernel(2)[0]


# this test is incredibly slow on the GPU backend (30min+) so it only runs on the CPU
# backend for now.
#@mark.parametrize("backend", backends)
def test_pixel_arguments():
    """
    Default interpolation pixel counts should be selected to preserve the aspect ratio of the data.
    """
    backend = 'cpu'

    df_2 = pd.DataFrame({'x': [-2, 4], 'y': [3, 7], 'A': [1, 1], 'B': [1, 1], 'h': [1, 1], 'rho': [1, 1], 'm': [1, 1]})
    sdf_2 = SarracenDataFrame(df_2, params=dict())
    sdf_2.backend = backend
    df_3 = pd.DataFrame({'x': [-2, 4], 'y': [3, 7], 'z': [6, -2], 'A': [1, 1], 'B': [1, 1], 'C': [1, 1], 'h': [1, 1],
                         'rho': [1, 1], 'm': [1, 1]})
    sdf_3 = SarracenDataFrame(df_3, params=dict())
    sdf_3.backend = backend

    default_pixels = 12

    # 3D grid interpolation
    for axes in [('x', 'y', 'z'), ('x', 'z', 'y'), ('y', 'z', 'x'), ('y', 'x', 'z'), ('z', 'x', 'y'), ('z', 'y', 'x')]:
        ratio01 = np.abs(df_3[axes[0]][1] - df_3[axes[0]][0]) / np.abs(df_3[axes[1]][1] - df_3[axes[1]][0])
        ratio02 = np.abs(df_3[axes[0]][1] - df_3[axes[0]][0]) / np.abs(df_3[axes[2]][1] - df_3[axes[2]][0])
        ratio12 = np.abs(df_3[axes[1]][1] - df_3[axes[1]][0]) / np.abs(df_3[axes[2]][1] - df_3[axes[2]][0])

        image = interpolate_3d_grid(sdf_3, 'A', x=axes[0], y=axes[1], z=axes[2], normalize=False, hmin=False)
        assert image.shape[2] / image.shape[1] == approx(ratio01, rel=1e-2)
        assert image.shape[1] / image.shape[0] == approx(ratio12, rel=1e-2)
        assert image.shape[2] / image.shape[0] == approx(ratio02, rel=1e-2)

        image = interpolate_3d_grid(sdf_3, 'A', x=axes[0], y=axes[1], z=axes[2], x_pixels=default_pixels, normalize=False, hmin=False)
        assert image.shape == (round(default_pixels / ratio02), round(default_pixels / ratio01), default_pixels)

        image = interpolate_3d_grid(sdf_3, 'A', x=axes[0], y=axes[1], z=axes[2], y_pixels=default_pixels, normalize=False, hmin=False)
        assert image.shape == (round(default_pixels / ratio12), default_pixels, round(default_pixels * ratio01))

        image = interpolate_3d_grid(sdf_3, 'A', x=axes[0], y=axes[1], z=axes[2], x_pixels=default_pixels,
                                    y_pixels=default_pixels, z_pixels=default_pixels, normalize=False, hmin=False)
        assert image.shape == (default_pixels, default_pixels, default_pixels)

    # Non-vector functions
    for func in [interpolate_2d, interpolate_3d_proj, interpolate_3d_cross]:
        for axes in [('x', 'y'), ('x', 'z'), ('y', 'z'), ('y', 'x'), ('z', 'x'), ('z', 'y')]:
            # The ratio of distance between particles in the second axis versus the distance between particles in
            # the first axis.
            ratio = np.abs(df_3[axes[1]][1] - df_3[axes[1]][0]) / np.abs(df_3[axes[0]][1] - df_3[axes[0]][0])

            # Avoids passing a z-axis argument to interpolate_2d, which would result in an error.
            if (axes[0] == 'z' or axes[1] == 'z') and func is interpolate_2d:
                continue

            # The dataframe is selected to ensure the correct number of dimensions.
            sdf = sdf_2 if func is interpolate_2d else sdf_3

            # With no pixels specified, the pixels in the image will match the ratio of the data.
            # The loose tolerance here accounts for the integer rounding.
            image = func(sdf, 'A', x=axes[0], y=axes[1], normalize=False, hmin=False)
            assert image.shape[0] / image.shape[1] == approx(ratio, rel=1e-2)

            # With one axis specified, the pixels in the other axis will be selected to match the ratio of the data.
            image = func(sdf, 'A', x=axes[0], y=axes[1], x_pixels=default_pixels, normalize=False, hmin=False)
            assert image.shape == (round(default_pixels * ratio), default_pixels)

            image = func(sdf, 'A', x=axes[0], y=axes[1], y_pixels=default_pixels, normalize=False, hmin=False)
            assert image.shape == (default_pixels, round(default_pixels / ratio))

            # With both axes specified, the pixels will simply match the specified counts.
            image = func(sdf, 'A', x_pixels=default_pixels * 2, y_pixels=default_pixels, normalize=False, hmin=False)
            assert image.shape == (default_pixels, default_pixels * 2)

    # 3D Vector-based functions
    for func in [interpolate_3d_vec, interpolate_3d_cross_vec]:
        for axes in [('x', 'y'), ('x', 'z'), ('y', 'z'), ('y', 'x'), ('z', 'x'), ('z', 'y')]:
            ratio = np.abs(df_3[axes[1]][1] - df_3[axes[1]][0]) / np.abs(df_3[axes[0]][1] - df_3[axes[0]][0])

            # Here, the tests are performed for both vector directions.
            image = func(sdf_3, 'A', 'B', 'C', x=axes[0], y=axes[1], normalize=False, hmin=False)
            assert image[0].shape[0] / image[0].shape[1] == approx(ratio, rel=1e-2)
            assert image[1].shape[0] / image[1].shape[1] == approx(ratio, rel=1e-2)

            image = func(sdf_3, 'A', 'B', 'C', x=axes[0], y=axes[1], x_pixels=default_pixels, normalize=False, hmin=False)
            assert image[0].shape == (round(default_pixels * ratio), default_pixels)
            assert image[1].shape == (round(default_pixels * ratio), default_pixels)

            image = func(sdf_3, 'A', 'B', 'C', x=axes[0], y=axes[1], y_pixels=default_pixels, normalize=False, hmin=False)
            assert image[0].shape == (default_pixels, round(default_pixels / ratio))
            assert image[1].shape == (default_pixels, round(default_pixels / ratio))

            image = func(sdf_3, 'A', 'B', 'C', x_pixels=default_pixels * 2, y_pixels=default_pixels, normalize=False, hmin=False)
            assert image[0].shape == (default_pixels, default_pixels * 2)
            assert image[1].shape == (default_pixels, default_pixels * 2)

    # 2D vector interpolation
    for axes in [('x', 'y'), ('y', 'x')]:
        ratio = np.abs(df_3[axes[1]][1] - df_3[axes[1]][0]) / np.abs(df_3[axes[0]][1] - df_3[axes[0]][0])

        image = interpolate_2d_vec(sdf_2, 'A', 'B', x=axes[0], y=axes[1], normalize=False, hmin=False)
        assert image[0].shape[0] / image[0].shape[1] == approx(ratio, rel=1e-2)
        assert image[1].shape[0] / image[1].shape[1] == approx(ratio, rel=1e-2)

        image = interpolate_2d_vec(sdf_2, 'A', 'B', x=axes[0], y=axes[1], x_pixels=default_pixels, normalize=False, hmin=False)
        assert image[0].shape == (round(default_pixels * ratio), default_pixels)
        assert image[1].shape == (round(default_pixels * ratio), default_pixels)

        image = interpolate_2d_vec(sdf_2, 'A', 'B', x=axes[0], y=axes[1], y_pixels=default_pixels, normalize=False, hmin=False)
        assert image[0].shape == (default_pixels, round(default_pixels / ratio))
        assert image[1].shape == (default_pixels, round(default_pixels / ratio))

        image = interpolate_2d_vec(sdf_2, 'A', 'B', x_pixels=default_pixels * 2, y_pixels=default_pixels, normalize=False, hmin=False)
        assert image[0].shape == (default_pixels, default_pixels * 2)
        assert image[1].shape == (default_pixels, default_pixels * 2)


@mark.parametrize("backend", backends)
def test_irregular_bounds(backend):
    """
    When the aspect ratio of pixels is different than the aspect ratio in particle space, the interpolation functions
    should still correctly interpolate to the skewed grid.
    """
    df = pd.DataFrame({'x': [0], 'y': [0], 'A': [4], 'B': [7], 'h': [0.9], 'rho': [0.4], 'm': [0.03]})
    sdf = SarracenDataFrame(df, params=dict())
    kernel = CubicSplineKernel()
    sdf.kernel = kernel
    sdf.backend = backend

    # Weight for 2D interpolation and 3D column interpolation.
    w = sdf['m'] / (sdf['rho'] * sdf['h'] ** 2)

    # A mapping of pixel indices to x / y values in particle space.
    real_x = -kernel.get_radius() + (np.arange(0, 50) + 0.5) * (2 * kernel.get_radius() / 50)
    real_y = -kernel.get_radius() + (np.arange(0, 25) + 0.5) * (2 * kernel.get_radius() / 25)

    image = interpolate_2d(sdf, 'A', x_pixels=50, y_pixels=25, xlim=(-kernel.get_radius(), kernel.get_radius()),
                           ylim=(-kernel.get_radius(), kernel.get_radius()), normalize=False, hmin=False)
    image_vec = interpolate_2d_vec(sdf, 'A', 'B', x_pixels=50, y_pixels=25,
                                   xlim=(-kernel.get_radius(), kernel.get_radius()),
                                   ylim=(-kernel.get_radius(), kernel.get_radius()), normalize=False, hmin=False)
    for y in range(25):
        for x in range(50):
            assert image[y][x] == approx(
                w[0] * sdf['A'][0] * kernel.w(np.sqrt(real_x[x] ** 2 + real_y[y] ** 2) / sdf['h'][0], 2))
            assert image_vec[0][y][x] == approx(
                w[0] * sdf['A'][0] * kernel.w(np.sqrt(real_x[x] ** 2 + real_y[y] ** 2) / sdf['h'][0], 2))
            assert image_vec[1][y][x] == approx(
                w[0] * sdf['B'][0] * kernel.w(np.sqrt(real_x[x] ** 2 + real_y[y] ** 2) / sdf['h'][0], 2))

    # Convert the existing 2D dataframe to a 3D dataframe.
    sdf['C'] = 5
    sdf['z'] = -0.5
    sdf.zcol = 'z'

    column_func = kernel.get_column_kernel_func(1000)

    image = interpolate_3d_proj(sdf, 'A', x_pixels=50, y_pixels=25, xlim=(-kernel.get_radius(), kernel.get_radius()),
                           ylim=(-kernel.get_radius(), kernel.get_radius()), dens_weight=False, normalize=False, hmin=False)
    image_vec = interpolate_3d_vec(sdf, 'A', 'B', 'C', x_pixels=50, y_pixels=25,
                                   xlim=(-kernel.get_radius(), kernel.get_radius()),
                                   ylim=(-kernel.get_radius(), kernel.get_radius()), dens_weight=False, normalize=False, hmin=False)
    for y in range(25):
        for x in range(50):
            assert image[y][x] == approx(
                w[0] * sdf['A'][0] * column_func(np.sqrt(real_x[x] ** 2 + real_y[y] ** 2) / sdf['h'][0], 2))
            assert image_vec[0][y][x] == approx(
                w[0] * sdf['A'][0] * column_func(np.sqrt(real_x[x] ** 2 + real_y[y] ** 2) / sdf['h'][0], 2))
            assert image_vec[1][y][x] == approx(
                w[0] * sdf['B'][0] * column_func(np.sqrt(real_x[x] ** 2 + real_y[y] ** 2) / sdf['h'][0], 2))

    # Weight for 3D cross-section interpolation.
    w = sdf['m'] / (sdf['rho'] * sdf['h'] ** 3)

    image = interpolate_3d_cross(sdf, 'A', z_slice=0, x_pixels=50, y_pixels=25,
                                 xlim=(-kernel.get_radius(), kernel.get_radius()),
                                 ylim=(-kernel.get_radius(), kernel.get_radius()), normalize=False, hmin=False)
    image_vec = interpolate_3d_cross_vec(sdf, 'A', 'B', 'C', 0, x_pixels=50, y_pixels=25,
                                         xlim=(-kernel.get_radius(), kernel.get_radius()),
                                         ylim=(-kernel.get_radius(), kernel.get_radius()), normalize=False, hmin=False)
    for y in range(25):
        for x in range(50):
            assert image[y][x] == approx(
                w[0] * sdf['A'][0] * kernel.w(np.sqrt(real_x[x] ** 2 + real_y[y] ** 2 + 0.5 ** 2) / sdf['h'][0], 3))
            assert image_vec[0][y][x] == approx(
                w[0] * sdf['A'][0] * kernel.w(np.sqrt(real_x[x] ** 2 + real_y[y] ** 2 + 0.5 ** 2) / sdf['h'][0], 3))
            assert image_vec[1][y][x] == approx(
                w[0] * sdf['B'][0] * kernel.w(np.sqrt(real_x[x] ** 2 + real_y[y] ** 2 + 0.5 ** 2) / sdf['h'][0], 3))

    real_z = -kernel.get_radius() + 0.5 + (np.arange(0, 15) + 0.5) * (2 * kernel.get_radius() / 15)
    limit = -kernel.get_radius(), kernel.get_radius()

    image = interpolate_3d_grid(sdf, 'A', x_pixels=50, y_pixels=25, z_pixels=15, xlim=limit, ylim=limit, zlim=limit, normalize=False, hmin=False)
    for z in range(15):
        for y in range(25):
            for x in range(50):
                assert image[z][y][x] == approx(
                    w[0] * sdf['A'][0] * kernel.w(np.sqrt(real_x[x] ** 2 + real_y[y] ** 2 + real_z[z] ** 2) / sdf['h'][0], 3))


@mark.parametrize("backend", backends)
def test_oob_particles(backend):
    """
    Particles outside the bounds of an interpolation operation should be included in the result.
    """
    kernel = CubicSplineKernel()

    df_2 = pd.DataFrame({'x': [0], 'y': [0], 'A': [4], 'B': [3], 'h': [1.9], 'rho': [0.4], 'm': [0.03]})
    sdf_2 = SarracenDataFrame(df_2, params=dict())
    sdf_2.kernel = kernel
    sdf_2.backend = backend

    df_3 = pd.DataFrame({'x': [0], 'y': [0], 'z': [0.5], 'A': [4], 'B': [3], 'C': [2], 'h': [1.9], 'rho': [0.4],
                         'm': [0.03]})
    sdf_3 = SarracenDataFrame(df_3, params=dict())
    sdf_3.kernel = kernel
    sdf_3.backend = backend

    # Weight for 2D interpolation, and 3D column interpolation.
    w = sdf_2['m'] / (sdf_2['rho'] * sdf_2['h'] ** 2)

    # A mapping of pixel indices to x / y values in particle space.
    real_x = 1 + (np.arange(0, 25) + 0.5) * (1 / 25)
    real_y = 1 + (np.arange(0, 25) + 0.5) * (1 / 25)

    image = interpolate_2d(sdf_2, 'A', x_pixels=25, y_pixels=25, xlim=(1, 2), ylim=(1, 2), normalize=False, hmin=False)
    image_vec = interpolate_2d_vec(sdf_2, 'A', 'B', x_pixels=25, y_pixels=25, xlim=(1, 2), ylim=(1, 2), normalize=False, hmin=False)
    line = interpolate_2d_line(sdf_2, 'A', pixels=25, xlim=(1, 2), ylim=(1, 2), normalize=False, hmin=False)
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

    image = interpolate_3d_proj(sdf_3, 'A', x_pixels=25, y_pixels=25, xlim=(1, 2), ylim=(1, 2),
                                dens_weight=False, normalize=False, hmin=False)
    image_vec = interpolate_3d_vec(sdf_3, 'A', 'B', 'C', x_pixels=25, y_pixels=25, xlim=(1, 2), ylim=(1, 2),
                                   dens_weight=False, normalize=False, hmin=False)
    for y in range(25):
        for x in range(25):
            assert image[y][x] == approx(
                w[0] * sdf_3['A'][0] * column_func(np.sqrt(real_x[x] ** 2 + real_y[y] ** 2) / sdf_3['h'][0], 2))
            assert image_vec[0][y][x] == approx(
                w[0] * sdf_3['A'][0] * column_func(np.sqrt(real_x[x] ** 2 + real_y[y] ** 2) / sdf_3['h'][0], 2))
            assert image_vec[1][y][x] == approx(
                w[0] * sdf_3['B'][0] * column_func(np.sqrt(real_x[x] ** 2 + real_y[y] ** 2) / sdf_3['h'][0], 2))

    # Weight for 3D cross-sections.
    w = sdf_3['m'] / (sdf_3['rho'] * sdf_3['h'] ** 3)

    image = interpolate_3d_cross(sdf_3, 'A', z_slice=0, x_pixels=25, y_pixels=25, xlim=(1, 2), ylim=(1, 2), normalize=False, hmin=False)
    image_vec = interpolate_3d_cross_vec(sdf_3, 'A', 'B', 'C', 0, x_pixels=25, y_pixels=25, xlim=(1, 2), ylim=(1, 2), normalize=False, hmin=False)
    for y in range(25):
        for x in range(25):
            assert image[y][x] == approx(
                w[0] * sdf_3['A'][0] * kernel.w(np.sqrt(real_x[x] ** 2 + real_y[y] ** 2 + 0.5 ** 2) / sdf_3['h'][0], 3))
            assert image_vec[0][y][x] == approx(
                w[0] * sdf_3['A'][0] * kernel.w(np.sqrt(real_x[x] ** 2 + real_y[y] ** 2 + 0.5 ** 2) / sdf_3['h'][0], 3))
            assert image_vec[1][y][x] == approx(
                w[0] * sdf_3['B'][0] * kernel.w(np.sqrt(real_x[x] ** 2 + real_y[y] ** 2 + 0.5 ** 2) / sdf_3['h'][0], 3))

    real_z = 0.5 + (np.arange(0, 25) + 0.5) * (1 / 25)

    image = interpolate_3d_grid(sdf_3, 'A', x_pixels=25, y_pixels=25, z_pixels=25, xlim=(1, 2), ylim=(1, 2),
                                zlim=(1, 2), normalize=False, hmin=False)

    for z in range(25):
        for y in range(25):
            for x in range(25):
                assert image[z][y][x] == approx(
                    w[0] * sdf_3['A'][0] * kernel.w(np.sqrt(real_x[x]**2 + real_y[y]**2 + real_z[z]**2) / sdf_3['h'][0], 3))


def rotate(target, rot_z, rot_y, rot_x):
    """ Perform a rotation of a target vector in three dimensions.

    A helper function for test_nonstandard_rotation()

    Parameters
    ----------
    target: float tuple of shape (3)
    rot_z, rot_y, rot_x: Rotation around each axis (in degrees)

    Returns
    -------
    float tuple of shape (3):
        The rotated vector.
    """
    pos_x1 = target[0] * np.cos(rot_z / (180 / np.pi)) - target[1] * np.sin(rot_z / (180 / np.pi))
    pos_y1 = target[0] * np.sin(rot_z / (180 / np.pi)) + target[1] * np.cos(rot_z / (180 / np.pi))
    pos_z1 = target[2]

    pos_x2 = pos_x1 * np.cos(rot_y / (180 / np.pi)) + pos_z1 * np.sin(rot_y / (180 / np.pi))
    pos_y2 = pos_y1
    pos_z2 = pos_x1 * -np.sin(rot_y / (180 / np.pi)) + pos_z1 * np.cos(rot_y / (180 / np.pi))

    pos_x3 = pos_x2
    pos_y3 = pos_y2 * np.cos(rot_x / (180 / np.pi)) - pos_z2 * np.sin(rot_x / (180 / np.pi))
    pos_z3 = pos_y2 * np.sin(rot_x / (180 / np.pi)) + pos_z2 * np.cos(rot_x / (180 / np.pi))

    return pos_x3, pos_y3, pos_z3


@mark.parametrize("backend", backends)
def test_nonstandard_rotation(backend):
    """
    Interpolation of a rotated dataframe with nonstandard angles should function properly.
    """
    df = pd.DataFrame({'x': [1], 'y': [1], 'z': [1], 'A': [4], 'B': [5], 'C': [6], 'h': [0.9], 'rho': [0.4],
                       'm': [0.03]})
    sdf = SarracenDataFrame(df, params=dict())
    kernel = CubicSplineKernel()
    sdf.kernel = kernel
    sdf.backend = backend

    column_kernel = kernel.get_column_kernel_func(1000)

    rot_z, rot_y, rot_x = 129, 34, 50

    image_col = interpolate_3d_proj(sdf, 'A', x_pixels=50, y_pixels=50, xlim=(-1, 1), ylim=(-1, 1),
                               rotation=[rot_z, rot_y, rot_x], origin=[0, 0, 0], dens_weight=False, normalize=False, hmin=False)
    image_cross = interpolate_3d_cross(sdf, 'A', z_slice=0, rotation=[rot_z, rot_y, rot_x], origin=[0, 0, 0],
                                       x_pixels=50, y_pixels=50, xlim=(-1, 1), ylim=(-1, 1), normalize=False, hmin=False)
    image_colvec = interpolate_3d_vec(sdf, 'A', 'B', 'C', x_pixels=50, y_pixels=50, xlim=(-1, 1), ylim=(-1, 1),
                                      rotation=[rot_z, rot_y, rot_x], origin=[0, 0, 0], dens_weight=False, normalize=False, hmin=False)
    image_crossvec = interpolate_3d_cross_vec(sdf, 'A', 'B', 'C', 0, x_pixels=50, y_pixels=50, xlim=(-1, 1),
                                              ylim=(-1, 1), rotation=[rot_z, rot_y, rot_x], origin=[0, 0, 0], normalize=False, hmin=False)

    w_col = sdf['m'] / (sdf['rho'] * sdf['h'] ** 2)
    w_cross = sdf['m'] / (sdf['rho'] * sdf['h'] ** 3)

    pos_x, pos_y, pos_z = rotate((1, 1, 1), rot_z, rot_y, rot_x)
    target_x, target_y, target_z = rotate((4, 5, 6), rot_z, rot_y, rot_x)

    real = -1 + (np.arange(0, 50) + 0.5) * (1 / 25)

    for y in range(50):
        for x in range(50):
            assert image_col[y][x] == approx(w_col[0] * sdf['A'][0] * column_kernel(
                np.sqrt((pos_x - real[x]) ** 2 + (pos_y - real[y]) ** 2) / sdf['h'][0], 3))
            assert image_colvec[0][y][x] == approx(w_col[0] * target_x * column_kernel(
                np.sqrt((pos_x - real[x]) ** 2 + (pos_y - real[y]) ** 2) / sdf['h'][0], 3))
            assert image_colvec[1][y][x] == approx(w_col[0] * target_y * column_kernel(
                np.sqrt((pos_x - real[x]) ** 2 + (pos_y - real[y]) ** 2) / sdf['h'][0], 3))
            assert image_cross[y][x] == approx(w_cross[0] * sdf['A'][0] * kernel.w(
                np.sqrt((pos_x - real[x]) ** 2 + (pos_y - real[y]) ** 2 + pos_z ** 2) / sdf['h'][0], 3))
            assert image_crossvec[0][y][x] == approx(w_cross[0] * target_x * kernel.w(
                np.sqrt((pos_x - real[x]) ** 2 + (pos_y - real[y]) ** 2 + pos_z ** 2) / sdf['h'][0], 3))
            assert image_crossvec[1][y][x] == approx(w_cross[0] * target_y * kernel.w(
                np.sqrt((pos_x - real[x]) ** 2 + (pos_y - real[y]) ** 2 + pos_z ** 2) / sdf['h'][0], 3))

    image_grid = interpolate_3d_grid(sdf, 'A', x_pixels=50, y_pixels=50, z_pixels=50, xlim=(-1, 1), ylim=(-1, 1),
                                     zlim=(-1, 1), rotation=[rot_z, rot_y, rot_x], rot_origin=[0, 0, 0], normalize=False, hmin=False)

    for z in range(50):
        for y in range(50):
            for x in range(50):
                assert image_grid[z][y][x] == \
                       approx(w_cross[0] * sdf['A'][0] * kernel.w(np.sqrt((pos_x - real[x]) ** 2
                                                                          + (pos_y - real[y]) ** 2
                                                                          + (pos_z - real[z]) ** 2) / sdf['h'][0], 3))


@mark.parametrize("backend", backends)
def test_scipy_rotation_equivalency(backend):
    """
    For interpolation functions, a [z, y, x] rotation defined with degrees should be equivalent
    to the scipy version using from_euler().
    """
    df = pd.DataFrame({'x': [1], 'y': [1], 'z': [1], 'A': [4], 'B': [3], 'C': [2], 'h': [0.9], 'rho': [0.4],
                       'm': [0.03]})
    sdf = SarracenDataFrame(df, params=dict())
    kernel = CubicSplineKernel()
    sdf.kernel = kernel
    sdf.backend = backend

    rot_z, rot_y, rot_x = 67, -34, 91

    image1 = interpolate_3d_proj(sdf, 'A', x_pixels=50, y_pixels=50, xlim=(-1, 1), ylim=(-1, 1),
                            rotation=[rot_z, rot_y, rot_x], origin=[0, 0, 0], normalize=False, hmin=False)
    image2 = interpolate_3d_proj(sdf, 'A', x_pixels=50, y_pixels=50, xlim=(-1, 1), ylim=(-1, 1),
                                 rotation=Rotation.from_euler('zyx', [rot_z, rot_y, rot_x], degrees=True),
                                 origin=[0, 0, 0], normalize=False, hmin=False)
    assert_allclose(image1, image2)

    image1 = interpolate_3d_cross(sdf, 'A', z_slice=0, rotation=[rot_z, rot_y, rot_x], origin=[0, 0, 0], x_pixels=50,
                                  y_pixels=50, xlim=(-1, 1), ylim=(-1, 1), normalize=False, hmin=False)
    image2 = interpolate_3d_cross(sdf, 'A', z_slice=0,
                                  rotation=Rotation.from_euler('zyx', [rot_z, rot_y, rot_x], degrees=True),
                                  origin=[0, 0, 0], x_pixels=50, y_pixels=50, xlim=(-1, 1), ylim=(-1, 1), normalize=False, hmin=False)
    assert_allclose(image1, image2)

    image1 = interpolate_3d_vec(sdf, 'A', 'B', 'C', x_pixels=50, y_pixels=50, xlim=(-1, 1), ylim=(-1, 1),
                                  rotation=[rot_z, rot_y, rot_x], origin=[0, 0, 0], normalize=False, hmin=False)
    image2 = interpolate_3d_vec(sdf, 'A', 'B', 'C', x_pixels=50, y_pixels=50, xlim=(-1, 1), ylim=(-1, 1),
                                  rotation=Rotation.from_euler('zyx', [rot_z, rot_y, rot_x], degrees=True),
                                  origin=[0, 0, 0], normalize=False, hmin=False)
    assert_allclose(image1[0], image2[0])
    assert_allclose(image1[1], image2[1])

    image1 = interpolate_3d_cross_vec(sdf, 'A', 'B', 'C', 0, x_pixels=50, y_pixels=50, xlim=(-1, 1), ylim=(-1, 1),
                                      rotation=[rot_z, rot_y, rot_x], origin=[0, 0, 0], normalize=False, hmin=False)
    image2 = interpolate_3d_cross_vec(sdf, 'A', 'B', 'C', 0, x_pixels=50, y_pixels=50, xlim=(-1, 1), ylim=(-1, 1),
                                      rotation=Rotation.from_euler('zyx', [rot_z, rot_y, rot_x], degrees=True),
                                      origin=[0, 0, 0], normalize=False, hmin=False)
    assert_allclose(image1[0], image2[0])
    assert_allclose(image1[1], image2[1])

    image1 = interpolate_3d_grid(sdf, 'A', x_pixels=50, y_pixels=50, xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1),
                                 rotation=[rot_z, rot_y, rot_x], rot_origin=[0, 0, 0], normalize=False, hmin=False)
    image2 = interpolate_3d_grid(sdf, 'A', x_pixels=50, y_pixels=50, xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1),
                                 rotation=Rotation.from_euler('zyx', [rot_z, rot_y, rot_x], degrees=True),
                                 rot_origin=[0, 0, 0], normalize=False, hmin=False)
    assert_allclose(image1, image2)


@mark.parametrize("backend", backends)
def test_quaternion_rotation(backend):
    """
    An alternate rotation (in this case, a quaternion) defined using scipy should function properly.
    """
    df = pd.DataFrame({'x': [1], 'y': [1], 'z': [1], 'A': [4], 'B': [3], 'C': [2], 'h': [1.9], 'rho': [0.4],
                       'm': [0.03]})
    sdf = SarracenDataFrame(df, params=dict())
    kernel = CubicSplineKernel()
    sdf.kernel = kernel
    sdf.backend = backend

    column_kernel = kernel.get_column_kernel_func(1000)

    quat = Rotation.from_quat([5, 3, 8, 1])
    image_col = interpolate_3d_proj(sdf, 'A', x_pixels=50, y_pixels=50, xlim=(-1, 1), ylim=(-1, 1), rotation=quat,
                               origin=[0, 0, 0], dens_weight=False, normalize=False, hmin=False)
    image_colvec = interpolate_3d_vec(sdf, 'A', 'B', 'C', x_pixels=50, y_pixels=50, xlim=(-1, 1), ylim=(-1, 1),
                                      rotation=quat, origin=[0, 0, 0], dens_weight=False, normalize=False, hmin=False)
    image_cross = interpolate_3d_cross(sdf, 'A', z_slice=0, rotation=quat, origin=[0, 0, 0], x_pixels=50, y_pixels=50,
                                       xlim=(-1, 1), ylim=(-1, 1), normalize=False, hmin=False)
    image_crossvec = interpolate_3d_cross_vec(sdf, 'A', 'B', 'C', 0, x_pixels=50, y_pixels=50, xlim=(-1, 1),
                                              ylim=(-1, 1), rotation=quat, origin=[0, 0, 0], normalize=False, hmin=False)

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

    image_grid = interpolate_3d_grid(sdf, 'A', x_pixels=50, y_pixels=50, z_pixels=50, xlim=(-1, 1), ylim=(-1, 1),
                                     zlim=(-1, 1), rotation=quat, rot_origin=[0, 0, 0], normalize=False, hmin=False)

    for z in range(50):
        for y in range(50):
            for x in range(50):
                assert image_grid[z][y][x] == approx(w_cross[0] * sdf['A'][0] * kernel.w(
                    np.sqrt((pos[0] - real[x]) ** 2 + (pos[1] - real[y]) ** 2 + (pos[2] - real[z]) ** 2)
                    / sdf['h'][0], 3))


@mark.parametrize("backend", backends)
def test_rotation_stability(backend):
    """
    A rotation performed at the same location as a pixel (for 3d column & cross-section interpolation) shouldn't change
    the resulting interpolation value at the pixel.
    """
    df = pd.DataFrame({'x': [1, 3], 'y': [1, -1], 'z': [1, -0.5], 'A': [4, 3], 'B': [3, 2], 'C': [1, 1.5],
                       'h': [0.9, 1.4], 'rho': [0.4, 0.6], 'm': [0.03, 0.06]})
    sdf = SarracenDataFrame(df, params=dict())
    kernel = CubicSplineKernel()
    sdf.kernel = kernel
    sdf.backend = backend

    real = -1 + (np.arange(0, 50) + 0.5) * (1 / 25)
    pixel_x, pixel_y = 12, 30

    for func in [interpolate_3d_proj, interpolate_3d_cross]:
        image = func(sdf, 'A', x_pixels=50, y_pixels=50, xlim=(-1, 1), ylim=(-1, 1), normalize=False, hmin=False)
        image_rot = func(sdf, 'A', x_pixels=50, y_pixels=50, xlim=(-1, 1), ylim=(-1, 1), rotation=[237, 0, 0],
                         origin=[real[pixel_x], real[pixel_y], 0], normalize=False, hmin=False)

        assert image[pixel_y][pixel_x] == approx(image_rot[pixel_y][pixel_x])


@mark.parametrize("backend", backends)
def test_axes_rotation_separation(backend):
    """
    Rotations should be independent of the defined x & y interpolation axes. Similar to test_image_transpose(), but a
    rotation is applied to all interpolations.
    """
    df = pd.DataFrame({'x': [-1, 1], 'y': [1, -1], 'z': [1, -1], 'A': [2, 1.5], 'B': [2, 2], 'C': [4, 3],
                       'h': [1.1, 1.3], 'rho': [0.55, 0.45], 'm': [0.04, 0.05]})
    sdf = SarracenDataFrame(df, params=dict())
    sdf.backend = backend

    image1 = interpolate_3d_proj(sdf, 'A', x_pixels=50,  y_pixels=50, xlim=(-1, 1), ylim=(-1, 1),
                                 rotation=[234, 90, 48], normalize=False, hmin=False)
    image2 = interpolate_3d_proj(sdf, 'A', x='y', y='x', x_pixels=50,  y_pixels=50, xlim=(-1, 1), ylim=(-1, 1),
                                 rotation=[234, 90, 48], normalize=False, hmin=False)
    assert_allclose(image1, image2.T)

    image1 = interpolate_3d_vec(sdf, 'A', 'B', 'C', x_pixels=50, y_pixels=50, xlim=(-1, 1), ylim=(-1, 1),
                                rotation=[234, 90, 48], normalize=False, hmin=False)
    image2 = interpolate_3d_vec(sdf, 'A', 'B', 'C', x='y', y='x', x_pixels=50, y_pixels=50, xlim=(-1, 1), ylim=(-1, 1),
                                rotation=[234, 90, 48], normalize=False, hmin=False)
    assert_allclose(image1[0], image2[0].T)
    assert_allclose(image1[1], image2[1].T)

    image1 = interpolate_3d_cross(sdf, 'A', z_slice=0, rotation=[234, 90, 48], x_pixels=50, y_pixels=50, xlim=(-1, 1),
                                  ylim=(-1, 1), normalize=False, hmin=False)
    image2 = interpolate_3d_cross(sdf, 'A', x='y', y='x', z_slice=0, rotation=[234, 90, 48], x_pixels=50, y_pixels=50,
                                  xlim=(-1, 1), ylim=(-1, 1), normalize=False, hmin=False)
    assert_allclose(image1, image2.T)

    image1 = interpolate_3d_cross_vec(sdf, 'A', 'B', 'C', 0, x_pixels=50, y_pixels=50, xlim=(-1, 1), ylim=(-1, 1),
                                      rotation=[234, 90, 48], normalize=False, hmin=False)
    image2 = interpolate_3d_cross_vec(sdf, 'A', 'B', 'C', 0, x='y', y='x', x_pixels=50, y_pixels=50, xlim=(-1, 1),
                                      ylim=(-1, 1), rotation=[234, 90, 48], normalize=False, hmin=False)
    assert_allclose(image1[0], image2[0].T)
    assert_allclose(image1[1], image2[1].T)

    image1 = interpolate_3d_grid(sdf, 'A', x_pixels=50, y_pixels=50, z_pixels=50, xlim=(-1, 1), ylim=(-1, 1),
                                 zlim=(-1, 1), rotation=[234, 90, 48], normalize=False, hmin=False)
    image2 = interpolate_3d_grid(sdf, 'A', x='y', y='x', x_pixels=50, y_pixels=50, z_pixels=50, xlim=(-1, 1),
                                 ylim=(-1, 1), zlim=(-1, 1), rotation=[234, 90, 48], normalize=False, hmin=False)
    assert_allclose(image1, image2.transpose(0, 2, 1))


@mark.parametrize("backend", backends)
def test_axes_rotation_equivalency(backend):
    """
    A rotated interpolation (at multiples of 90 degrees) should be equivalent to a transformed interpolation with
    different x & y axes. For example, an interpolation rotated by 180 degrees around the z axis should be equivalent
    to the transpose of an unaltered interpolation.
    """
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

                for func in [interpolate_3d_proj, interpolate_3d_cross]:
                    image1 = func(sdf, 'A', x_pixels=50, y_pixels=50, xlim=(-1, 1), ylim=(-1, 1),
                                  rotation=[rot_z, rot_y, rot_x], origin=[0, 0, 0], normalize=False, hmin=False)
                    image2 = func(sdf, 'A', x=x, y=y, x_pixels=50, y_pixels=50, xlim=(-1, 1), ylim=(-1, 1), normalize=False, hmin=False)
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
    """
    Interpolation with invalid bounds should raise a ValueError.
    """
    df_2 = pd.DataFrame({'x': [0], 'y': [0], 'A': [4], 'B': [3], 'C': [2.5], 'h': [0.9], 'rho': [0.4], 'm': [0.03]})
    sdf_2 = SarracenDataFrame(df_2, params=dict())
    df_3 = pd.DataFrame({'x': [0], 'y': [0], 'z': [-0.5], 'A': [4], 'B': [3], 'C': [2.5], 'h': [0.9], 'rho': [0.4],
                         'm': [0.03]})
    sdf_3 = SarracenDataFrame(df_3, params=dict())

    sdf_2.backend = backend
    sdf_3.backend = backend

    for b in [(-3, 3, 3, -3, 20, 20), (3, 3, 3, 3, 20, 20), (-3, 3, -3, 3, 0, 0)]:
        with raises(ValueError):
            interpolate_2d(sdf_2, 'A', xlim=(b[0], b[1]), ylim=(b[2], b[3]), x_pixels=b[4], y_pixels=b[5], normalize=False, hmin=False)
        with raises(ValueError):
            interpolate_2d_vec(sdf_2, 'A', 'B', 'C', xlim=(b[0], b[1]), ylim=(b[2], b[3]), x_pixels=b[4],
                               y_pixels=b[5], normalize=False, hmin=False)
        # the first case will not fail for this type of interpolation.
        if not b[0] == -3 and not b[3] == -3:
            with raises(ValueError):
                interpolate_2d_line(sdf_2, 'A', xlim=(b[0], b[1]), ylim=(b[2], b[3]), pixels=b[4], normalize=False, hmin=False)
        with raises(ValueError):
            interpolate_3d_proj(sdf_3, 'A', xlim=(b[0], b[1]), ylim=(b[2], b[3]), x_pixels=b[4], y_pixels=b[5], normalize=False, hmin=False)
        with raises(ValueError):
            interpolate_3d_vec(sdf_3, 'A', 'B', 'C', xlim=(b[0], b[1]), ylim=(b[2], b[3]), x_pixels=b[4],
                               y_pixels=b[5], normalize=False, hmin=False)
        with raises(ValueError):
            interpolate_3d_cross(sdf_3, 'A', z_slice=0, x_pixels=b[4], y_pixels=b[5], xlim=(b[0], b[1]),
                                 ylim=(b[2], b[3]), normalize=False, hmin=False)
        with raises(ValueError):
            interpolate_3d_cross_vec(sdf_3, 'A', 'B', 'C', 0, xlim=(b[0], b[1]), ylim=(b[2], b[3]), x_pixels=b[4],
                                     y_pixels=b[5], normalize=False, hmin=False)
        with raises(ValueError):
            interpolate_3d_grid(sdf_3, 'A', xlim=(b[0], b[1]), ylim=(b[2], b[3]), zlim=(-3, 3), x_pixels=b[4],
                                y_pixels=b[5], z_pixels=10, normalize=False, hmin=False)


@mark.parametrize("backend", backends)
def test_required_columns(backend):
    """
    Interpolation without one of the required columns will result in a KeyError.
    """
    # This test is currently expected to fail on both backends, since dropping a column from a SarracenDataFrame
    # returns a DataFrame.
    df_2 = pd.DataFrame({'x': [-1, 1], 'y': [1, -1], 'A': [2, 1.5], 'B': [5, 4], 'C': [3, 2], 'h': [1.1, 1.3],
                         'rho': [0.55, 0.45], 'm': [0.04, 0.05]})
    sdf_2 = SarracenDataFrame(df_2, params=dict())
    df_3 = pd.DataFrame({'x': [-1, 1], 'y': [1, -1], 'z': [1, -1], 'A': [2, 1.5], 'B': [5, 4], 'C': [3, 2],
                         'h': [1.1, 1.3], 'rho': [0.55, 0.45], 'm': [0.04, 0.05]})
    sdf_3 = SarracenDataFrame(df_3, params=dict())

    sdf_2.backend = backend
    sdf_3.backend = backend

    for column in ['m', 'h']:
        sdf_dropped = sdf_2.drop(column, axis=1)
        with raises(KeyError):
            interpolate_2d(sdf_dropped, 'A', normalize=False, hmin=False)
        with raises(KeyError):
            interpolate_2d_line(sdf_dropped, 'A', normalize=False, hmin=False)
        with raises(KeyError):
            interpolate_2d_vec(sdf_dropped, 'A', 'B', normalize=False, hmin=False)

        sdf_dropped = sdf_3.drop(column, axis=1)
        with raises(KeyError):
            interpolate_3d_proj(sdf_dropped, 'A', normalize=False, hmin=False)
        with raises(KeyError):
            interpolate_3d_cross(sdf_dropped, 'A', normalize=False, hmin=False)
        with raises(KeyError):
            interpolate_3d_vec(sdf_dropped, 'A', 'B', 'C', normalize=False, hmin=False)
        with raises(KeyError):
            interpolate_3d_cross_vec(sdf_dropped, 'A', 'B', 'C', normalize=False, hmin=False)
        with raises(KeyError):
            interpolate_3d_grid(sdf_dropped, 'A', normalize=False, hmin=False)


@mark.parametrize("backend", backends)
def test_exact_interpolation(backend):
    """
    Exact interpolation over the entire effective area of a kernel should return 1 over the particle bounds, multiplied by the weight.
    """
    df_2 = pd.DataFrame({'x': [0], 'y': [0], 'A': [2], 'h': [1.1], 'rho': [0.55], 'm': [0.04]})
    sdf_2 = SarracenDataFrame(df_2, params=dict())
    sdf_2.backend = backend
    df_3 = pd.DataFrame({'x': [0], 'y': [0], 'z': [1], 'A': [2], 'h': [1.1], 'rho': [0.55], 'm': [0.04]})
    sdf_3 = SarracenDataFrame(df_3, params=dict())
    sdf_3.backend = backend

    kernel = CubicSplineKernel()
    w = sdf_2['m'] * sdf_2['A'] / (sdf_2['rho'] * sdf_2['h'] ** 2)

    bound = kernel.get_radius() * sdf_2['h'][0]
    image = interpolate_2d(sdf_2, 'A', xlim=(-bound, bound), ylim=(-bound, bound), x_pixels=1, exact=True, normalize=False, hmin=False)

    assert image.sum() == approx(w[0] * sdf_2['h'][0] ** 2 / (4 * bound ** 2))

    image = interpolate_3d_proj(sdf_3, 'A', xlim=(-bound, bound), ylim=(-bound, bound), x_pixels=1, exact=True, dens_weight=False, normalize=False, hmin=False)

    assert image.sum() == approx(w[0] * sdf_2['h'][0] ** 2 / (4 * bound ** 2))


@mark.parametrize("backend", backends)
def test_density_weighted(backend):
    """
    Enabling density weighted interpolation will change the resultant image
    """
    df_2 = pd.DataFrame({'x': [0], 'y': [0], 'A': [2], 'B': [3], 'h': [0.5], 'rho': [0.25], 'm': [0.75]})
    sdf_2 = SarracenDataFrame(df_2, params=dict())
    df_3 = pd.DataFrame({'x': [0], 'y': [0], 'z': [0], 'A': [2], 'B': [3], 'C': [4], 'h': [0.5], 'rho': [0.25],
                         'm': [0.75]})
    sdf_3 = SarracenDataFrame(df_3, params=dict())

    kernel = CubicSplineKernel()
    sdf_2.backend = backend
    sdf_3.backend = backend

    for dens_weight in [True, False]:
        if dens_weight:
            weight2d = sdf_2['m'][0] / (sdf_2['h'][0] ** 2)
            weight3d = sdf_2['m'][0] / (sdf_2['h'][0] ** 3)
        else:
            weight2d = sdf_2['m'][0] / (sdf_2['rho'][0] * sdf_2['h'][0] ** 2)
            weight3d = sdf_2['m'][0] / (sdf_2['rho'][0] * sdf_2['h'][0] ** 3)

        image = interpolate_2d(sdf_2, 'A', x_pixels=1, y_pixels=1, xlim=(-1, 1), ylim=(-1, 1), dens_weight=dens_weight, normalize=False, hmin=False)
        assert image == weight2d * sdf_2['A'][0] * kernel.w(0, 2)
        image = interpolate_2d_vec(sdf_2, 'A', 'B', x_pixels=1, y_pixels=1, xlim=(-1, 1), ylim=(-1, 1), dens_weight=dens_weight, normalize=False, hmin=False)
        assert image[0] == weight2d * sdf_2['A'][0] * kernel.w(0, 2)
        assert image[1] == weight2d * sdf_2['B'][0] * kernel.w(0, 2)

        image = interpolate_2d_line(sdf_2, 'A', pixels=1, xlim=(-1, 1), ylim=(-1, 1), dens_weight=dens_weight, normalize=False, hmin=False)
        assert image[0] == weight2d * sdf_2['A'][0] * kernel.w(0, 2)

        image = interpolate_3d_proj(sdf_3, 'A', x_pixels=1, y_pixels=1, xlim=(-1, 1), ylim=(-1, 1), dens_weight=dens_weight, normalize=False, hmin=False)
        assert image[0] == weight2d * sdf_2['A'][0] * kernel.get_column_kernel()[0]
        image = interpolate_3d_vec(sdf_3, 'A', 'B', 'C', x_pixels=1, y_pixels=1, xlim=(-1, 1), ylim=(-1, 1), dens_weight=dens_weight, normalize=False, hmin=False)
        assert image[0] == weight2d * sdf_2['A'][0] * kernel.get_column_kernel()[0]
        assert image[1] == weight2d * sdf_2['B'][0] * kernel.get_column_kernel()[0]

        image = interpolate_3d_cross(sdf_3, 'A', x_pixels=1, y_pixels=1, xlim=(-1, 1), ylim=(-1, 1), dens_weight=dens_weight, normalize=False, hmin=False)
        assert image[0] == weight3d * sdf_2['A'][0] * kernel.w(0, 3)
        image = interpolate_3d_cross_vec(sdf_3, 'A', 'B', 'C', x_pixels=1, y_pixels=1, xlim=(-1, 1), ylim=(-1, 1), dens_weight=dens_weight, normalize=False, hmin=False)
        assert image[0] == weight3d * sdf_2['A'][0] * kernel.w(0, 3)
        assert image[1] == weight3d * sdf_2['B'][0] * kernel.w(0, 3)

        image = interpolate_3d_grid(sdf_3, 'A', x_pixels=1, y_pixels=1, z_pixels=1, xlim=(-1, 1), ylim=(-1, 1),
                                    zlim=(-1, 1), dens_weight=dens_weight, normalize=False, hmin=False)
        assert image[0] == weight3d * sdf_2['A'][0] * kernel.w(0, 3)

        image = interpolate_3d_line(sdf_3, 'A', pixels=1, xlim=(-1, 1), ylim=(-1, 1), dens_weight=dens_weight, normalize=False, hmin=False)
        assert image[0] == weight3d * sdf_2['A'][0] * kernel.w(0, 3)


@mark.parametrize("backend", backends)
def test_normalize_interpolation(backend):
    sdf_2 = SarracenDataFrame({'x': [0], 'y': [0], 'A': [2], 'B': [3], 'h': [0.5], 'rho': [0.25], 'm': [0.75]},
                              params=dict())
    sdf_3 = SarracenDataFrame({'x': [0], 'y': [0], 'z': [0], 'A': [2], 'B': [3], 'C': [4], 'h': [0.5], 'rho': [0.25],
                               'm': [0.75]}, params=dict())

    kernel = CubicSplineKernel()
    sdf_2.backend = backend
    sdf_3.backend = backend

    weight2d = sdf_2['m'][0] / (sdf_2['rho'][0] * sdf_2['h'][0] ** 2) * kernel.w(0, 2)
    weight3d = sdf_2['m'][0] / (sdf_2['rho'][0] * sdf_2['h'][0] ** 3) * kernel.w(0, 3)
    weight3d_column = sdf_2['m'][0] / (sdf_2['rho'][0] * sdf_2['h'][0] ** 2) * kernel.get_column_kernel()[0]

    for normalize in [True, False]:

        norm2d = 1.0
        norm3d = 1.0
        norm3d_column = 1.0
        if normalize:
            norm2d = sdf_2['m'][0] / (sdf_2['rho'][0] * sdf_2['h'][0] ** 2) * kernel.w(0, 2)
            norm3d = sdf_2['m'][0] / (sdf_2['rho'][0] * sdf_2['h'][0] ** 3) * kernel.w(0, 3)
            norm3d_column = sdf_2['m'][0] / (sdf_2['rho'][0] * sdf_2['h'][0] ** 2) * kernel.get_column_kernel()[0]

        image = interpolate_2d(sdf_2, 'A', x_pixels=1, y_pixels=1, xlim=(-1, 1), ylim=(-1, 1),
                               dens_weight=False, normalize=normalize)
        assert image == weight2d * sdf_2['A'][0] / norm2d

        image = interpolate_2d_vec(sdf_2, 'A', 'B', x_pixels=1, y_pixels=1, xlim=(-1, 1), ylim=(-1, 1),
                                   dens_weight=False, normalize=normalize)
        assert image[0] == weight2d * sdf_2['A'][0] / norm2d
        assert image[1] == weight2d * sdf_2['B'][0] / norm2d

        image = interpolate_2d_line(sdf_2, 'A', pixels=1, xlim=(-1, 1), ylim=(-1, 1),
                                    dens_weight=False, normalize=normalize)
        assert image[0] == weight2d * sdf_2['A'][0] / norm2d

        image = interpolate_3d_proj(sdf_3, 'A', x_pixels=1, y_pixels=1, xlim=(-1, 1), ylim=(-1, 1),
                                    dens_weight=False, normalize=normalize)
        assert image[0] == weight3d_column * sdf_2['A'][0] / norm3d_column
        image = interpolate_3d_vec(sdf_3, 'A', 'B', 'C', x_pixels=1, y_pixels=1, xlim=(-1, 1), ylim=(-1, 1),
                                   dens_weight=False, normalize=normalize)
        assert image[0] == weight3d_column * sdf_2['A'][0] / norm3d_column
        assert image[1] == weight3d_column * sdf_2['B'][0] / norm3d_column

        image = interpolate_3d_cross(sdf_3, 'A', x_pixels=1, y_pixels=1, xlim=(-1, 1), ylim=(-1, 1),
                                     dens_weight=False, normalize=normalize)
        assert image[0] == weight3d * sdf_2['A'][0] / norm3d
        image = interpolate_3d_cross_vec(sdf_3, 'A', 'B', 'C', x_pixels=1, y_pixels=1, xlim=(-1, 1), ylim=(-1, 1),
                                         dens_weight=False, normalize=normalize)
        assert image[0] == weight3d * sdf_2['A'][0] / norm3d
        assert image[1] == weight3d * sdf_2['B'][0] / norm3d

        image = interpolate_3d_grid(sdf_3, 'A', x_pixels=1, y_pixels=1, z_pixels=1, xlim=(-1, 1), ylim=(-1, 1),
                                    zlim=(-1, 1), dens_weight=False, normalize=normalize)
        assert image[0] == weight3d * sdf_2['A'][0] / norm3d

        image = interpolate_3d_line(sdf_3, 'A', pixels=1, xlim=(-1, 1), ylim=(-1, 1),
                                    dens_weight=False, normalize=normalize)
        assert image[0] == weight3d * sdf_2['A'][0] / norm3d


@mark.parametrize("backend", backends)
def test_exact_interpolation_culling(backend):
    sdf_2 = SarracenDataFrame({'x': [0], 'y': [0], 'A': [2], 'h': [0.4], 'rho': [0.1], 'm': [1]}, params=dict())
    sdf_2.backend = backend
    sdf_3 = SarracenDataFrame({'x': [0], 'y': [0], 'z': [0], 'A': [2], 'h': [0.4], 'rho': [0.1], 'm': [1]},
                              params=dict())
    sdf_3.backend = backend

    image_2 = sdf_2.sph_interpolate('A', xlim=(-1, 1), ylim=(-1, 1), x_pixels=5, exact=True)
    image_3 = interpolate_3d_proj(sdf_3, 'A', xlim=(-1, 1), ylim=(-1, 1), x_pixels=5, exact=True)

    assert image_2[2, 4] != 0
    assert image_3[2, 4] != 0


@mark.parametrize("backend", backends)
def test_minimum_smoothing_length_2d(backend):
    """ Test that the minimum smoothing length evaluates correctly. """

    pixels = 5
    xlim, ylim = (-1, 1), (-1, 1)
    hmin = 0.5 * (xlim[1] - xlim[0]) / pixels

    sdf_a = SarracenDataFrame(data={'rx': [0.3, -0.1, 0.1, 0.1, 0.05, -0.05, -0.25, -0.2],
                                             'ry': [0.0, 0.1, -0.1, 0.0, -0.05, 0.07, -0.3, -0.2],
                                             'h': [hmin, hmin, 0.3, 0.25, hmin, hmin, 0.2, hmin],
                                             'm': [0.56] * 8},
                                       params={'hfact': 1.2})

    sdf_b = SarracenDataFrame(data={'rx': [0.3, -0.1, 0.1, 0.1, 0.05, -0.05, -0.25, -0.2],
                                             'ry': [0.0, 0.1, -0.1, 0.0, -0.05, 0.07, -0.3, -0.2],
                                             'h': [0.01, 0.01, 0.3, 0.25, 0.01, 0.01, 0.2, 0.01],
                                             'm': [0.56] * 8},
                                       params={'hfact': 1.2})

    sdf_a.backend = backend
    sdf_b.backend = backend

    for interpolate in [interpolate_2d]:
        grid = interpolate(data=sdf_a, target='rho', xlim=xlim, ylim=ylim, x_pixels=pixels, y_pixels=pixels,
                           normalize=False, hmin=False)
        grid_hmin = interpolate(data=sdf_b, target='rho', xlim=xlim, ylim=ylim, x_pixels=pixels, y_pixels=pixels,
                                normalize=False, hmin=True)

        assert (grid == grid_hmin).all()


@mark.parametrize("backend", backends)
def test_minimum_smoothing_length_3d(backend):
    """ Test that the minimum smoothing length evaluates correctly. """

    pixels = 5
    xlim, ylim = (-1, 1), (-1, 1)
    hmin = 0.5 * (xlim[1] - xlim[0]) / pixels

    sdf_a = SarracenDataFrame(data={'rx': [0.3, -0.1, 0.1, 0.1, 0.05, -0.05, -0.25, -0.2],
                                             'ry': [0.0, 0.1, -0.1, 0.0, -0.05, 0.07, -0.3, -0.2],
                                             'rz': [0.1, 0.32, 0.03, -0.3, -0.2, 0.1, -0.06, 0.22],
                                             'h': [hmin, hmin, 0.3, 0.25, hmin, hmin, 0.2, hmin],
                                             'm': [0.56] * 8},
                                       params={'hfact': 1.2})

    sdf_b = SarracenDataFrame(data={'rx': [0.3, -0.1, 0.1, 0.1, 0.05, -0.05, -0.25, -0.2],
                                             'ry': [0.0, 0.1, -0.1, 0.0, -0.05, 0.07, -0.3, -0.2],
                                             'rz': [0.1, 0.32, 0.03, -0.3, -0.2, 0.1, -0.06, 0.22],
                                             'h': [0.01, 0.01, 0.3, 0.25, 0.01, 0.01, 0.2, 0.01],
                                             'm': [0.56] * 8},
                                       params={'hfact': 1.2})

    sdf_a.backend = backend
    sdf_b.backend = backend

    for interpolate in [interpolate_3d_cross, interpolate_3d_proj, interpolate_3d_grid]:
        grid = interpolate(data=sdf_a, target='rho', xlim=xlim, ylim=ylim, x_pixels=pixels, y_pixels=pixels,
                           normalize=False, hmin=False)
        grid_hmin = interpolate(data=sdf_b, target='rho', xlim=xlim, ylim=ylim, x_pixels=pixels, y_pixels=pixels,
                                normalize=False, hmin=True)

        assert (grid == grid_hmin).all()


@mark.parametrize("backend", backends)
def test_minimum_smoothing_length_1d_lines(backend):
    """ Test that the minimum smoothing length evaluates correctly. """

    pixels = 5
    xlim, ylim, zlim = (-1, 1), (-0.5, 0.5), (-0.5, 0.5)

    hmin = 0.5 * np.sqrt((xlim[1] - xlim[0]) ** 2 + (ylim[1] - ylim[0]) ** 2) / pixels

    sdf_a = SarracenDataFrame(data={'rx': [0.3, -0.1, 0.1, 0.1, 0.05, -0.05, -0.25, -0.2],
                                             'ry': [0.0, 0.1, -0.1, 0.0, -0.05, 0.07, -0.3, -0.2],
                                             'h': [hmin, hmin, 0.3, 0.25, hmin, hmin, hmin, hmin],
                                             'm': [0.56] * 8},
                                       params={'hfact': 1.2})

    sdf_b = SarracenDataFrame(data={'rx': [0.3, -0.1, 0.1, 0.1, 0.05, -0.05, -0.25, -0.2],
                                             'ry': [0.0, 0.1, -0.1, 0.0, -0.05, 0.07, -0.3, -0.2],
                                             'h': [0.01, 0.01, 0.3, 0.25, 0.01, 0.01, 0.2, 0.01],
                                             'm': [0.56] * 8},
                                       params={'hfact': 1.2})

    sdf_a.backend = backend
    sdf_b.backend = backend

    grid = interpolate_2d_line(data=sdf_a, target='rho', xlim=xlim, ylim=ylim, pixels=pixels,
                               normalize=False, hmin=False)
    grid_hmin = interpolate_2d_line(data=sdf_b, target='rho', xlim=xlim, ylim=ylim, pixels=pixels,
                                    normalize=False, hmin=True)

    assert (grid == grid_hmin).all()

    hmin = 0.5 * np.sqrt((xlim[1] - xlim[0]) ** 2 + (ylim[1] - ylim[0]) ** 2 + (zlim[1] - zlim[0]) ** 2) / pixels

    sdf_a = SarracenDataFrame(data={'rx': [0.3, -0.1, 0.1, 0.1, 0.05, -0.05, -0.25, -0.2],
                                             'ry': [0.0, 0.1, -0.1, 0.0, -0.05, 0.07, -0.3, -0.2],
                                             'rz': [0.1, 0.32, 0.03, -0.3, -0.2, 0.1, -0.06, 0.22],
                                             'h': [hmin, hmin, 0.3, 0.25, hmin, hmin, hmin, hmin],
                                             'm': [0.56] * 8},
                                       params={'hfact': 1.2})

    sdf_b = SarracenDataFrame(data={'rx': [0.3, -0.1, 0.1, 0.1, 0.05, -0.05, -0.25, -0.2],
                                             'ry': [0.0, 0.1, -0.1, 0.0, -0.05, 0.07, -0.3, -0.2],
                                             'rz': [0.1, 0.32, 0.03, -0.3, -0.2, 0.1, -0.06, 0.22],
                                             'h': [0.01, 0.01, 0.3, 0.25, 0.01, 0.01, 0.2, 0.01],
                                             'm': [0.56] * 8},
                                       params={'hfact': 1.2})

    sdf_a.backend = backend
    sdf_b.backend = backend

    grid = interpolate_3d_line(data=sdf_a, target='rho', xlim=xlim, ylim=ylim, zlim=zlim, pixels=pixels,
                               normalize=False, hmin=False)
    grid_hmin = interpolate_3d_line(data=sdf_b, target='rho', xlim=xlim, ylim=ylim, zlim=zlim, pixels=pixels,
                                    normalize=False, hmin=True)

    assert (grid == grid_hmin).all()

