"""
pytest unit tests for interpolate.py functions.
"""
from typing import Any, Callable, Dict, List

import pandas as pd
import numpy as np
from numba import cuda
from numpy.testing import assert_allclose
from pytest import approx, raises, mark

from sarracen import SarracenDataFrame
from sarracen.kernels import CubicSplineKernel, QuarticSplineKernel, \
    QuinticSplineKernel, BaseKernel
from sarracen.interpolate import interpolate_2d, interpolate_2d_line, \
    interpolate_3d_cross, interpolate_3d_proj, interpolate_2d_vec, \
    interpolate_3d_vec, interpolate_3d_cross_vec, interpolate_3d_grid, \
    interpolate_3d_line

backends = ['cpu']
if cuda.is_available():
    backends.append('gpu')

funcs2d: List[Callable] = [interpolate_2d, interpolate_2d_line]
funcs2dvec: List[Callable] = [interpolate_2d_vec]
funcs3d: List[Callable] = [interpolate_3d_line, interpolate_3d_proj,
                           interpolate_3d_cross, interpolate_3d_grid]
funcs3dvec: List[Callable] = [interpolate_3d_vec, interpolate_3d_cross_vec]

funcscolumn: List[Callable] = [interpolate_3d_proj, interpolate_3d_vec]
funcsline: List[Callable] = [interpolate_2d_line, interpolate_3d_line]

funcs = funcs2d + funcs2dvec + funcs3d + funcs3dvec


@mark.parametrize("backend", backends)
def test_single_particle(backend: str) -> None:
    """
    The result of interpolation over a single particle should be equal to
    scaled kernel values at each point of the image.
    """
    data = {'x': [0], 'y': [0], 'A': [4], 'B': [5],
            'h': [0.9], 'rho': [0.4], 'm': [0.03]}
    sdf = SarracenDataFrame(data, params=dict())
    kernel = CubicSplineKernel()
    kernel_rad = kernel.get_radius()
    bounds = (-kernel_rad, kernel_rad)

    sdf.kernel = kernel
    sdf.backend = backend

    # Weight for 2D interpolation & 3D column interpolation.
    weight = sdf['m'] / (sdf['rho'] * sdf['h'] ** 2)

    # A mapping of pixel indices to x / y values in particle space.
    real = -kernel_rad + (np.arange(0, 25) + 0.5) * (2 * kernel_rad / 25)

    img = interpolate_2d(sdf, 'A',
                         x_pixels=25, y_pixels=25,
                         xlim=bounds, ylim=bounds,
                         normalize=False, hmin=False)
    img_vec = interpolate_2d_vec(sdf, 'A', 'B',
                                 x_pixels=25, y_pixels=25,
                                 xlim=bounds, ylim=bounds,
                                 normalize=False, hmin=False)
    for y in range(25):
        for x in range(25):
            r = np.sqrt(real[x] ** 2 + real[y] ** 2)
            w = kernel.w(r / sdf['h'][0], 2)
            assert img[y][x] == approx(weight[0] * sdf['A'][0] * w)
            assert img_vec[0][y][x] == approx(weight[0] * sdf['A'][0] * w)
            assert img_vec[1][y][x] == approx(weight[0] * sdf['B'][0] * w)

    img = interpolate_2d_line(sdf, 'A',
                              pixels=25,
                              xlim=bounds, ylim=bounds,
                              normalize=False, hmin=False)
    for x in range(25):
        r = np.sqrt(2) * np.abs(real[x])
        w = kernel.w(r / sdf['h'][0], 2)
        assert img[x] == approx(weight[0] * sdf['A'][0] * w)

    # Convert the previous 2D dataframe to a 3D dataframe.
    sdf['z'] = -0.5
    sdf['C'] = 10
    sdf.zcol = 'z'

    column_func = kernel.get_column_kernel_func(1000)

    img = interpolate_3d_proj(sdf, 'A',
                              x_pixels=25, y_pixels=25,
                              xlim=bounds, ylim=bounds,
                              dens_weight=False,
                              normalize=False, hmin=False)
    img_vec = interpolate_3d_vec(sdf, 'A', 'B', 'C',
                                 x_pixels=25, y_pixels=25,
                                 xlim=bounds, ylim=bounds,
                                 dens_weight=False,
                                 normalize=False, hmin=False)
    for y in range(25):
        for x in range(25):
            r = np.sqrt(real[x] ** 2 + real[y] ** 2)
            w = column_func(r / sdf['h'][0], 2)
            assert img[y][x] == approx(weight[0] * sdf['A'][0] * w)
            assert img_vec[0][y][x] == approx(weight[0] * sdf['A'][0] * w)
            assert img_vec[1][y][x] == approx(weight[0] * sdf['B'][0] * w)

    # Weight for 3D cross-sections.
    weight = sdf['m'] / (sdf['rho'] * sdf['h'] ** 3)

    img = interpolate_3d_cross(sdf, 'A',
                               x_pixels=25, y_pixels=25,
                               xlim=bounds, ylim=bounds,
                               z_slice=0,
                               normalize=False, hmin=False)
    img_vec = interpolate_3d_cross_vec(sdf, 'A', 'B', 'C',
                                       x_pixels=25, y_pixels=25,
                                       xlim=bounds, ylim=bounds,
                                       z_slice=0,
                                       normalize=False, hmin=False)
    for y in range(25):
        for x in range(25):
            r = np.sqrt(real[x] ** 2 + real[y] ** 2 + 0.5 ** 2)
            w = kernel.w(r / sdf['h'][0], 3)
            assert img[y][x] == approx(weight[0] * sdf['A'][0] * w)
            assert img_vec[0][y][x] == approx(weight[0] * sdf['A'][0] * w)
            assert img_vec[1][y][x] == approx(weight[0] * sdf['B'][0] * w)

    img = interpolate_3d_grid(sdf, 'A',
                              x_pixels=25, y_pixels=25, z_pixels=25,
                              xlim=bounds, ylim=bounds, zlim=bounds,
                              normalize=False, hmin=False)
    for z in range(25):
        for y in range(25):
            for x in range(25):
                r = np.sqrt(real[x] ** 2 + real[y] ** 2 + (real[z] + 0.5) ** 2)
                w = kernel.w(r / sdf['h'][0], 3)
                assert img[z][y][x] == approx(weight[0] * sdf['A'][0] * w)

    img = interpolate_3d_line(sdf, 'A',
                              pixels=25,
                              xlim=bounds, ylim=bounds, zlim=bounds,
                              normalize=False, hmin=False)
    for x in range(25):
        r = np.sqrt(2 * real[x] ** 2 + (real[x] + 0.5) ** 2)
        w = kernel.w(r / sdf['h'][0], 3)
        assert img[x] == approx(weight[0] * sdf['A'][0] * w)


@mark.parametrize("backend", backends)
def test_single_repeated_particle(backend: str) -> None:
    """
    The result of interpolation over a single particle repeated several times
    should be equal to scaled kernel values at each point of the image
    multiplied by the number of particles.

    If this test fails, it is likely that there is a race condition issue in
    the interpolation implementation.
    """

    repetitions = 10000
    df = pd.concat([pd.DataFrame({'x': [0], 'y': [0], 'A': [4], 'B': [5],
                                  'h': [0.9], 'rho': [0.4],
                                  'm': [0.03]})] * repetitions,
                   ignore_index=True)
    sdf = SarracenDataFrame(df, params=dict())

    kernel = CubicSplineKernel()
    kernel_rad = kernel.get_radius()
    bounds = (-kernel_rad, kernel_rad)

    sdf.kernel = kernel
    sdf.backend = backend

    # Multiplying by repetitions here is done for ease of use.
    weight = repetitions * sdf['m'] / (sdf['rho'] * sdf['h'] ** 2)

    # A mapping of pixel indices to x / y values in particle space.
    real = -kernel_rad + (np.arange(0, 25) + 0.5) * (2 * kernel_rad / 25)

    img = interpolate_2d(sdf, 'A',
                         x_pixels=25, y_pixels=25,
                         xlim=bounds, ylim=bounds,
                         normalize=False, hmin=False)
    img_vec = interpolate_2d_vec(sdf, 'A', 'B',
                                 x_pixels=25, y_pixels=25,
                                 xlim=bounds, ylim=bounds,
                                 normalize=False, hmin=False)
    for y in range(25):
        for x in range(25):
            r = np.sqrt(real[x] ** 2 + real[y] ** 2)
            w = kernel.w(r / sdf['h'][0], 2)
            assert img[y][x] == approx(weight[0] * sdf['A'][0] * w)
            assert img_vec[0][y][x] == approx(weight[0] * sdf['A'][0] * w)
            assert img_vec[1][y][x] == approx(weight[0] * sdf['B'][0] * w)

    img = interpolate_2d_line(sdf, 'A',
                              pixels=25,
                              xlim=bounds, ylim=bounds,
                              normalize=False, hmin=False)
    for x in range(25):
        r = np.sqrt(2) * np.abs(real[x])
        w = kernel.w(r / sdf['h'][0], 2)
        assert img[x] == approx(weight[0] * sdf['A'][0] * w)

    # Convert the previous 2D dataframe to a 3D dataframe.
    sdf['z'] = -0.5
    sdf['C'] = 10
    sdf.zcol = 'z'

    column_func = kernel.get_column_kernel_func(1000)

    img = interpolate_3d_proj(sdf, 'A',
                              x_pixels=25, y_pixels=25,
                              xlim=bounds, ylim=bounds,
                              dens_weight=False,
                              normalize=False, hmin=False)
    img_vec = interpolate_3d_vec(sdf, 'A', 'B', 'C',
                                 x_pixels=25, y_pixels=25,
                                 xlim=bounds, ylim=bounds,
                                 dens_weight=False,
                                 normalize=False, hmin=False)
    for y in range(25):
        for x in range(25):
            r = np.sqrt(real[x] ** 2 + real[y] ** 2)
            w = column_func(r / sdf['h'][0], 2)
            assert img[y][x] == approx(weight[0] * sdf['A'][0] * w)
            assert img_vec[0][y][x] == approx(weight[0] * sdf['A'][0] * w)
            assert img_vec[1][y][x] == approx(weight[0] * sdf['B'][0] * w)

    # Weight for 3D cross-sections
    weight = repetitions * sdf['m'] / (sdf['rho'] * sdf['h'] ** 3)

    img = interpolate_3d_cross(sdf, 'A',
                               x_pixels=25, y_pixels=25,
                               xlim=bounds, ylim=bounds,
                               z_slice=0,
                               normalize=False, hmin=False)
    img_vec = interpolate_3d_cross_vec(sdf, 'A', 'B', 'C',
                                       x_pixels=25, y_pixels=25,
                                       xlim=bounds, ylim=bounds,
                                       z_slice=0,
                                       normalize=False, hmin=False)
    for y in range(25):
        for x in range(25):
            r = np.sqrt(real[x] ** 2 + real[y] ** 2 + 0.5 ** 2)
            w = kernel.w(r / sdf['h'][0], 3)
            assert img[y][x] == approx(weight[0] * sdf['A'][0] * w)
            assert img_vec[0][y][x] == approx(weight[0] * sdf['A'][0] * w)
            assert img_vec[1][y][x] == approx(weight[0] * sdf['B'][0] * w)

    img = interpolate_3d_grid(sdf, 'A',
                              x_pixels=25, y_pixels=25, z_pixels=25,
                              xlim=bounds, ylim=bounds, zlim=bounds,
                              normalize=False, hmin=False)
    for z in range(25):
        for y in range(25):
            for x in range(25):
                r = np.sqrt(real[x] ** 2 + real[y] ** 2 + (real[z] + 0.5) ** 2)
                w = kernel.w(r / sdf['h'][0], 3)
                assert img[z][y][x] == approx(weight[0] * sdf['A'][0] * w)

    img = interpolate_3d_line(sdf, 'A',
                              pixels=25,
                              xlim=bounds, ylim=bounds, zlim=bounds,
                              normalize=False, hmin=False)
    for x in range(25):
        r = np.sqrt(2 * real[x] ** 2 + (real[x] + 0.5) ** 2)
        w = kernel.w(r / sdf['h'][0], 3)
        assert img[x] == approx(weight[0] * sdf['A'][0] * w)


@mark.parametrize("backend", backends)
@mark.parametrize("func", funcs)
def test_dimension_check(backend: str, func: Callable) -> None:
    """
    Passing a dataframe with invalid dimensions should raise a TypeError for
    all interpolation functions.
    """

    data = {'x': [0, 1], 'y': [0, 1], 'P': [1, 1],
            'Ax': [1, 1], 'Ay': [1, 1], 'h': [1, 1],
            'rho': [1, 1], 'm': [1, 1]}
    sdf = SarracenDataFrame(data, params=dict())
    sdf.backend = backend

    # 2D dataframe passed to 3D interpolation functions
    if func in funcs3d:
        with raises(ValueError):
            func(sdf, 'P', normalize=False, hmin=False)
    elif func in funcs3dvec:
        with raises(ValueError):
            func(sdf, 'Ax', 'Ay', 'Az', normalize=False, hmin=False)

    # 3D dataframe passed to 2D interpolation functions
    elif func in (funcs2d + funcs2dvec):
        sdf['z'] = [0, 1]
        sdf['Az'] = [1, 1]
        sdf.zcol = 'z'

        if func in funcs2d:
            with raises(ValueError):
                func(sdf, 'P', normalize=False, hmin=False)
        elif func in funcs2dvec:
            with raises(ValueError):
                func(sdf, 'Ax', 'Ay', normalize=False, hmin=False)


@mark.parametrize("backend", backends)
def test_3d_xsec_equivalency(backend: str) -> None:
    """
    A single 3D column integration of a dataframe should be equivalent to the
    average of several evenly spaced 3D cross-sections.
    """
    data = {'x': [0], 'y': [0], 'z': [0],
            'A': [4], 'B': [6], 'C': [2],
            'h': [0.9], 'rho': [0.4], 'm': [0.03]}
    sdf = SarracenDataFrame(data, params=dict())
    kernel = CubicSplineKernel()
    sdf.kernel = kernel
    sdf.backend = backend

    samples = 250

    column_img = interpolate_3d_proj(sdf, 'A',
                                     x_pixels=50,
                                     xlim=(-1, 1), ylim=(-1, 1),
                                     dens_weight=False,
                                     normalize=False, hmin=False)
    column_img_vec = interpolate_3d_vec(sdf, 'A', 'B', 'C',
                                        x_pixels=50,
                                        xlim=(-1, 1), ylim=(-1, 1),
                                        dens_weight=False,
                                        normalize=False, hmin=False)

    xsec_img = np.zeros((50, 50))
    xsec_img_vec = [np.zeros((50, 50)), np.zeros((50, 50))]
    for z in np.linspace(0, kernel.get_radius() * sdf['h'][0], samples):
        xsec_img += interpolate_3d_cross(sdf, 'A',
                                         x_pixels=50,
                                         xlim=(-1, 1), ylim=(-1, 1),
                                         z_slice=z,
                                         normalize=False, hmin=False)

        vec_sample = interpolate_3d_cross_vec(sdf, 'A', 'B', 'C',
                                              x_pixels=50,
                                              xlim=(-1, 1), ylim=(-1, 1),
                                              z_slice=z,
                                              normalize=False, hmin=False)
        xsec_img_vec[0] += vec_sample[0]
        xsec_img_vec[1] += vec_sample[1]

    # Scale each cross-section sum to be equivalent to the column integration.
    xsec_img *= kernel.get_radius() * sdf['h'][0] * 2 / samples
    xsec_img_vec[0] *= kernel.get_radius() * sdf['h'][0] * 2 / samples
    xsec_img_vec[1] *= kernel.get_radius() * sdf['h'][0] * 2 / samples

    # The tolerances are lower here to accommodate for the relatively low
    # sample size. A larger number of samples would result in an unacceptable
    # test time for the GPU backend (which already doesn't perform well with
    # repeated interpolation of just one particle)
    assert_allclose(xsec_img, column_img, rtol=1e-3, atol=1e-4)
    assert_allclose(xsec_img_vec[0], column_img_vec[0], rtol=1e-3, atol=1e-4)
    assert_allclose(xsec_img_vec[1], column_img_vec[1], rtol=1e-3, atol=1e-4)


@mark.parametrize("backend", backends)
def test_2d_xsec_equivalency(backend: str) -> None:
    """
    A single 2D interpolation should be equivalent to several combined
    2D cross-sections.
    """
    # This test currently fails on both backends, since a vertical 2D
    # cross-section currently returns zero for an unknown reason.
    data = {'x': [0], 'y': [0], 'A': [4],
            'h': [0.9], 'rho': [0.4], 'm': [0.03]}
    sdf = SarracenDataFrame(data, params=dict())
    kernel = CubicSplineKernel()
    sdf.kernel = kernel
    sdf.backend = backend

    true_img = interpolate_2d(sdf, 'A',
                              x_pixels=50,
                              xlim=(-1, 1), ylim=(-1, 1),
                              normalize=False, hmin=False)

    # A mapping of pixel indices to x & y values in particle space.
    real = -1 + (np.arange(0, 50) + 0.5) * (2 / 50)

    recon_img = np.zeros((50, 50))
    for y in range(50):
        recon_img[y, :] = interpolate_2d_line(sdf, 'A',
                                              pixels=50,
                                              xlim=(-1, 1),
                                              ylim=(real[y], real[y]),
                                              normalize=False, hmin=False)
    assert_allclose(recon_img, true_img)

    # reconstructed_img = np.zeros((50, 50))
    # for x in range(50):
    #     reconstructed_img[:, x] = interpolate_2d_line(sdf, 'A',
    #     pixels=50, xlim=(real[x], real[x]), ylim=(-1, 1))
    # assert_allclose(reconstructed_img, true_img)


@mark.parametrize("backend", backends)
def test_corner_particles(backend: str) -> None:
    """
    Interpolation over a dataset with two particles should be equal to the sum
    of contributions at each point.
    """
    kernel = CubicSplineKernel()

    data_2 = {'x': [-1, 1], 'y': [-1, 1],
              'A': [2, 1.5], 'B': [5, 2.3],
              'h': [1.1, 1.3], 'rho': [0.55, 0.45], 'm': [0.04, 0.05]}
    sdf_2 = SarracenDataFrame(data_2, params=dict())
    sdf_2.kernel = kernel
    sdf_2.backend = backend

    data_3 = {'x': [-1, 1], 'y': [-1, 1], 'z': [-1, 1],
              'A': [2, 1.5], 'B': [2, 1], 'C': [7, 8],
              'h': [1.1, 1.3], 'rho': [0.55, 0.45], 'm': [0.04, 0.05]}
    sdf_3 = SarracenDataFrame(data_3, params=dict())
    sdf_3.kernel = kernel
    sdf_3.backend = backend

    # Weight for 2D interpolation, and 3D column interpolation.
    weight = sdf_2['m'] / (sdf_2['rho'] * sdf_2['h'] ** 2)

    # A mapping of pixel indices to x / y values in particle space.
    real = (np.arange(0, 25) + 0.5) * (2 / 25)

    img = interpolate_2d(sdf_2, 'A',
                         x_pixels=25, y_pixels=25,
                         normalize=False, hmin=False)
    img_vec = interpolate_2d_vec(sdf_2, 'A', 'B',
                                 x_pixels=25, y_pixels=25,
                                 normalize=False, hmin=False)
    for y in range(25):
        for x in range(25):
            r = np.sqrt(real[x] ** 2 + real[y] ** 2)
            w0 = kernel.w(r / sdf_2['h'][0], 2)

            r = np.sqrt(real[24 - x] ** 2 + real[24 - y] ** 2)
            w1 = kernel.w(r / sdf_2['h'][1], 2)
            assert img[y][x] == approx(weight[0] * sdf_2['A'][0] * w0
                                       + weight[1] * sdf_2['A'][1] * w1)
            assert img_vec[0][y][x] == approx(weight[0] * sdf_2['A'][0] * w0
                                              + weight[1] * sdf_2['A'][1] * w1)
            assert img_vec[1][y][x] == approx(weight[0] * sdf_2['B'][0] * w0
                                              + weight[1] * sdf_2['B'][1] * w1)

    img = interpolate_2d_line(sdf_2, 'A',
                              pixels=25,
                              normalize=False, hmin=False)
    for x in range(25):
        r = np.sqrt(real[x] ** 2 + real[x] ** 2)
        w0 = kernel.w(r / sdf_2['h'][0], 2)

        r = np.sqrt(real[24 - x] ** 2 + real[24 - x] ** 2)
        w1 = kernel.w(r / sdf_2['h'][1], 2)
        assert img[x] == approx(weight[0] * sdf_2['A'][0] * w0
                                + weight[1] * sdf_2['A'][1] * w1)

    c_kernel = kernel.get_column_kernel_func(1000)

    img = interpolate_3d_proj(sdf_3, 'A',
                              x_pixels=25, y_pixels=25,
                              dens_weight=False,
                              normalize=False, hmin=False)
    img_vec = interpolate_3d_vec(sdf_3, 'A', 'B', 'C',
                                 x_pixels=25, y_pixels=25,
                                 dens_weight=False,
                                 normalize=False, hmin=False)
    for y in range(25):
        for x in range(25):
            r = np.sqrt(real[x] ** 2 + real[y] ** 2)
            w0 = c_kernel(r / sdf_3['h'][0], 2)

            r = np.sqrt(real[24 - x] ** 2 + real[24 - y] ** 2)
            w1 = c_kernel(r / sdf_3['h'][1], 2)
            assert img[y][x] == approx(weight[0] * sdf_3['A'][0] * w0
                                       + weight[1] * sdf_3['A'][1] * w1)
            assert img_vec[0][y][x] == approx(weight[0] * sdf_3['A'][0] * w0
                                              + weight[1] * sdf_3['A'][1] * w1)
            assert img_vec[1][y][x] == approx(weight[0] * sdf_3['B'][0] * w0
                                              + weight[1] * sdf_3['B'][1] * w1)

    # Weight for 3D cross-section interpolation.
    weight = sdf_3['m'] / (sdf_3['rho'] * sdf_3['h'] ** 3)

    img = interpolate_3d_cross(sdf_3, 'A',
                               x_pixels=25, y_pixels=25,
                               normalize=False, hmin=False)
    img_vec = interpolate_3d_cross_vec(sdf_3, 'A', 'B', 'C',
                                       x_pixels=25, y_pixels=25,
                                       normalize=False, hmin=False)
    for y in range(25):
        for x in range(25):
            r = np.sqrt(real[x] ** 2 + real[y] ** 2 + 1)
            w0 = kernel.w(r / sdf_3['h'][0], 3)

            r = np.sqrt(real[24 - x] ** 2 + real[24 - y] ** 2 + 1)
            w1 = kernel.w(r / sdf_3['h'][1], 3)
            assert img[y][x] == approx(weight[0] * sdf_3['A'][0] * w0
                                       + weight[1] * sdf_3['A'][1] * w1)
            assert img_vec[0][y][x] == approx(weight[0] * sdf_3['A'][0] * w0
                                              + weight[1] * sdf_3['A'][1] * w1)
            assert img_vec[1][y][x] == approx(weight[0] * sdf_3['B'][0] * w0
                                              + weight[1] * sdf_3['B'][1] * w1)

    img = interpolate_3d_grid(sdf_3, 'A',
                              x_pixels=25, y_pixels=25, z_pixels=25,
                              normalize=False, hmin=False)
    for z in range(25):
        for y in range(25):
            for x in range(25):
                r = np.sqrt(real[x] ** 2 + real[y] ** 2 + real[z] ** 2)
                w0 = kernel.w(r / sdf_3['h'][0], 3)

                r = np.sqrt(real[24-x]**2 + real[24-y]**2 + real[24-z]**2)
                w1 = kernel.w(r / sdf_3['h'][1], 3)
                assert img[z][y][x] == approx(weight[0] * sdf_3['A'][0] * w0
                                              + weight[1] * sdf_3['A'][1] * w1)


@mark.parametrize("backend", backends)
def test_image_transpose(backend: str) -> None:
    """
    Interpolation with flipped x & y axes should be equivalent to the transpose
    of regular interpolation.
    """
    data = {'x': [-1, 1], 'y': [1, -1], 'A': [2, 1.5], 'B': [5, 4],
            'h': [1.1, 1.3], 'rho': [0.55, 0.45], 'm': [0.04, 0.05]}
    sdf = SarracenDataFrame(data, params=dict())
    sdf.backend = backend

    img1 = interpolate_2d(sdf, 'A',
                          x_pixels=20, y_pixels=20,
                          normalize=False, hmin=False)
    img2 = interpolate_2d(sdf, 'A', x='y', y='x',
                          x_pixels=20, y_pixels=20,
                          normalize=False, hmin=False)
    assert_allclose(img1, img2.T)

    img1_tuple = interpolate_2d_vec(sdf, 'A', 'B',
                                    x_pixels=20, y_pixels=20,
                                    normalize=False, hmin=False)
    img2_tuple = interpolate_2d_vec(sdf, 'A', 'B', x='y', y='x',
                                    x_pixels=20, y_pixels=20,
                                    normalize=False, hmin=False)
    assert_allclose(img1_tuple[0], img2_tuple[0].T)
    assert_allclose(img1_tuple[1], img2_tuple[1].T)

    data = {'x': [-1, 1], 'y': [1, -1], 'z': [-1, 1],
            'A': [2, 1.5], 'B': [5, 4], 'C': [2.5, 3],
            'h': [1.1, 1.3], 'rho': [0.55, 0.45], 'm': [0.04, 0.05]}
    sdf = SarracenDataFrame(data, params=dict())

    img1 = interpolate_3d_proj(sdf, 'A',
                               x_pixels=20, y_pixels=20,
                               normalize=False, hmin=False)
    img2 = interpolate_3d_proj(sdf, 'A',
                               x='y', y='x',
                               x_pixels=20, y_pixels=20,
                               normalize=False, hmin=False)
    assert_allclose(img1, img2.T)

    img1_tuple = interpolate_3d_vec(sdf, 'A', 'B', 'C',
                                    x_pixels=50, y_pixels=50,
                                    normalize=False, hmin=False)
    img2_tuple = interpolate_3d_vec(sdf, 'A', 'B', 'C',
                                    x='y', y='x',
                                    x_pixels=50, y_pixels=50,
                                    normalize=False, hmin=False)
    assert_allclose(img1_tuple[0], img2_tuple[0].T)
    assert_allclose(img1_tuple[1], img2_tuple[1].T)

    img1 = interpolate_3d_cross(sdf, 'A',
                                x_pixels=50, y_pixels=50,
                                normalize=False, hmin=False)
    img2 = interpolate_3d_cross(sdf, 'A',
                                x='y', y='x',
                                x_pixels=50, y_pixels=50,
                                normalize=False, hmin=False)
    assert_allclose(img1, img2.T)

    img1_tuple = interpolate_3d_cross_vec(sdf, 'A', 'B', 'C',
                                          x_pixels=20, y_pixels=20,
                                          normalize=False, hmin=False)
    img2_tuple = interpolate_3d_cross_vec(sdf, 'A', 'B', 'C',
                                          x='y', y='x',
                                          x_pixels=20, y_pixels=20,
                                          normalize=False, hmin=False)
    assert_allclose(img1_tuple[0], img2_tuple[0].T)
    assert_allclose(img1_tuple[1], img2_tuple[1].T)

    img1 = interpolate_3d_grid(sdf, 'A',
                               x_pixels=20, y_pixels=20,
                               normalize=False, hmin=False)
    img2 = interpolate_3d_grid(sdf, 'A',
                               x='y', y='x',
                               x_pixels=20, y_pixels=20,
                               normalize=False, hmin=False)
    assert_allclose(img1, img2.transpose(0, 2, 1))


@mark.parametrize("backend", backends)
@mark.parametrize("use_default_kernel", [True, False])
def test_default_kernel(backend: str, use_default_kernel: bool) -> None:
    """
    Interpolation should use the kernel supplied to the function. If no kernel
    is supplied, the kernel attached to the dataframe should be used.
    """
    data_2 = {'x': [0], 'y': [0], 'A': [1], 'B': [1],
              'h': [1], 'rho': [1], 'm': [1]}
    sdf_2 = SarracenDataFrame(data_2, params=dict())
    data_3 = {'x': [0], 'y': [0], 'z': [0],
              'A': [1], 'B': [1], 'C': [1],
              'h': [1], 'rho': [1], 'm': [1]}
    sdf_3 = SarracenDataFrame(data_3, params=dict())

    sdf_2.backend = backend
    sdf_3.backend = backend

    kwargs: Dict[str, Any] = {'normalize': False}

    if use_default_kernel:
        kernel: BaseKernel = QuarticSplineKernel()
        sdf_2.kernel = kernel
        sdf_3.kernel = kernel
    else:
        kernel = QuinticSplineKernel()
        kwargs['kernel'] = kernel

    # First, test that the dataframe kernel is used when no kernel is supplied.
    # Next, test that the kernel supplied to the function is actually used.

    img = interpolate_2d(sdf_2, 'A',
                         x_pixels=1, y_pixels=1,
                         xlim=(-1, 1), ylim=(-1, 1), **kwargs)
    assert img == kernel.w(0, 2)
    img_tuple = interpolate_2d_vec(sdf_2, 'A', 'B',
                                   x_pixels=1, y_pixels=1,
                                   xlim=(-1, 1), ylim=(-1, 1), **kwargs)
    assert img_tuple[0] == kernel.w(0, 2)
    assert img_tuple[1] == kernel.w(0, 2)

    img = interpolate_2d_line(sdf_2, 'A',
                              pixels=1,
                              xlim=(-1, 1), ylim=(-1, 1), **kwargs)
    assert img == kernel.w(0, 2)

    img = interpolate_3d_proj(sdf_3, 'A',
                              x_pixels=1, y_pixels=1,
                              xlim=(-1, 1), ylim=(-1, 1), **kwargs)
    assert img == kernel.get_column_kernel()[0]
    img_tuple = interpolate_3d_vec(sdf_3, 'A', 'B', 'C',
                                   x_pixels=1, y_pixels=1,
                                   xlim=(-1, 1), ylim=(-1, 1), **kwargs)
    assert img_tuple[0] == kernel.get_column_kernel()[0]
    assert img_tuple[1] == kernel.get_column_kernel()[0]

    img = interpolate_3d_cross(sdf_3, 'A',
                               x_pixels=1, y_pixels=1,
                               xlim=(-1, 1), ylim=(-1, 1), **kwargs)
    assert img == kernel.w(0, 3)
    img_tuple = interpolate_3d_cross_vec(sdf_3, 'A', 'B', 'C',
                                         x_pixels=1, y_pixels=1,
                                         xlim=(-1, 1), ylim=(-1, 1), **kwargs)
    assert img_tuple[0] == kernel.w(0, 3)
    assert img_tuple[1] == kernel.w(0, 3)

    img = interpolate_3d_grid(sdf_3, 'A',
                              x_pixels=1, y_pixels=1, z_pixels=1,
                              xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1),
                              **kwargs)
    assert img == kernel.w(0, 3)

    img = interpolate_3d_line(sdf_3, 'A',
                              pixels=1,
                              xlim=(-1, 1), ylim=(-1, 1), **kwargs)
    assert img == kernel.w(0, 3)


@mark.parametrize("backend", backends)
def test_column_samples(backend: str) -> None:
    """
    3D column interpolation should use the number of integral samples supplied
    as an argument.
    """
    data_3 = {'x': [0], 'y': [0], 'z': [0],
              'A': [1], 'h': [1], 'rho': [1], 'm': [1]}
    sdf_3 = SarracenDataFrame(data_3, params=dict())
    kernel = QuinticSplineKernel()
    sdf_3.kernel = kernel
    sdf_3.backend = backend

    # 2 samples is used here, since a column kernel with 2 samples will be
    # drastically different than the default kernel of 1000 samples.
    img = interpolate_3d_proj(sdf_3, 'A',
                              x_pixels=1, y_pixels=1,
                              xlim=(-1, 1), ylim=(-1, 1),
                              integral_samples=2,
                              normalize=False, hmin=False)
    assert img == kernel.get_column_kernel(2)[0]


# this test is incredibly slow on the GPU backend (30min+) so it only runs on
# the CPU backend for now.
# @mark.parametrize("backend", backends)
def test_pixel_arguments() -> None:
    """
    Default interpolation pixel counts should be selected to preserve the
    aspect ratio of the data.
    """
    backend = 'cpu'

    data_2 = {'x': [-2, 4], 'y': [3, 7], 'A': [1, 1], 'B': [1, 1],
              'h': [1, 1], 'rho': [1, 1], 'm': [1, 1]}
    sdf_2 = SarracenDataFrame(data_2, params=dict())
    sdf_2.backend = backend
    data_3 = {'x': [-2, 4], 'y': [3, 7], 'z': [6, -2],
              'A': [1, 1], 'B': [1, 1], 'C': [1, 1],
              'h': [1, 1], 'rho': [1, 1], 'm': [1, 1]}
    sdf_3 = SarracenDataFrame(data_3, params=dict())
    sdf_3.backend = backend

    default_pixels = 12

    # 3D grid interpolation
    for ax in [('x', 'y', 'z'),
               ('x', 'z', 'y'),
               ('y', 'z', 'x'),
               ('y', 'x', 'z'),
               ('z', 'x', 'y'),
               ('z', 'y', 'x')]:
        diff_0 = np.abs(sdf_3[ax[0]][1] - sdf_3[ax[0]][0])
        diff_1 = np.abs(sdf_3[ax[1]][1] - sdf_3[ax[1]][0])
        diff_2 = np.abs(sdf_3[ax[2]][1] - sdf_3[ax[2]][0])

        ratio01 = diff_0 / diff_1
        ratio02 = diff_0 / diff_2
        ratio12 = diff_1 / diff_2

        img = interpolate_3d_grid(sdf_3, 'A',
                                  x=ax[0], y=ax[1], z=ax[2],
                                  normalize=False, hmin=False)
        assert len(img.shape) == 3
        assert img.shape[2] / img.shape[1] == approx(ratio01, rel=1e-2)
        assert img.shape[1] / img.shape[0] == approx(ratio12, rel=1e-2)
        assert img.shape[2] / img.shape[0] == approx(ratio02, rel=1e-2)

        img = interpolate_3d_grid(sdf_3, 'A',
                                  x=ax[0], y=ax[1], z=ax[2],
                                  x_pixels=default_pixels,
                                  normalize=False, hmin=False)
        assert img.shape == (round(default_pixels / ratio02),
                             round(default_pixels / ratio01), default_pixels)

        img = interpolate_3d_grid(sdf_3, 'A',
                                  x=ax[0], y=ax[1], z=ax[2],
                                  y_pixels=default_pixels,
                                  normalize=False, hmin=False)
        assert img.shape == (round(default_pixels / ratio12),
                             default_pixels,
                             round(default_pixels * ratio01))

        img = interpolate_3d_grid(sdf_3, 'A',
                                  x=ax[0], y=ax[1], z=ax[2],
                                  x_pixels=default_pixels,
                                  y_pixels=default_pixels,
                                  z_pixels=default_pixels,
                                  normalize=False, hmin=False)
        assert img.shape == (default_pixels, default_pixels, default_pixels)

    # Non-vector functions
    functions_list: List[Callable] = [interpolate_2d, interpolate_3d_proj,
                                      interpolate_3d_cross]
    for func in functions_list:
        for axes in [('x', 'y'),
                     ('x', 'z'),
                     ('y', 'z'),
                     ('y', 'x'),
                     ('z', 'x'),
                     ('z', 'y')]:
            # The ratio of distance between particles in the second axis versus
            # the distance between particles in the first axis.
            ratio = np.abs(sdf_3[axes[1]][1] - sdf_3[axes[1]][0]) \
                    / np.abs(sdf_3[axes[0]][1] - sdf_3[axes[0]][0])

            # Avoids passing a z-axis argument to interpolate_2d, which would
            # result in an error.
            if (axes[0] == 'z' or axes[1] == 'z') and func is interpolate_2d:
                continue

            # Dataframe is selected to ensure the correct number of dimensions.
            sdf = sdf_2 if func is interpolate_2d else sdf_3

            # With no pixels specified, the pixels in the image will match the
            # ratio of the data. The loose tolerance here accounts for the
            # integer rounding.
            img = func(sdf, 'A',
                       x=axes[0], y=axes[1],
                       normalize=False, hmin=False)
            assert img.shape[0] / img.shape[1] == approx(ratio, rel=1e-2)

            # With one axis specified, the pixels in the other axis will be
            # selected to match the ratio of the data.
            img = func(sdf, 'A',
                       x=axes[0], y=axes[1],
                       x_pixels=default_pixels,
                       normalize=False, hmin=False)
            assert img.shape == (round(default_pixels * ratio), default_pixels)

            img = func(sdf, 'A',
                       x=axes[0], y=axes[1],
                       y_pixels=default_pixels,
                       normalize=False, hmin=False)
            assert img.shape == (default_pixels, round(default_pixels / ratio))

            # With both axes specified, the pixels will simply match the
            # specified counts.
            img = func(sdf, 'A',
                       x_pixels=default_pixels * 2, y_pixels=default_pixels,
                       normalize=False, hmin=False)
            assert img.shape == (default_pixels, default_pixels * 2)

    # 3D Vector-based functions
    for func in funcs3dvec:
        for axes in [('x', 'y'),
                     ('x', 'z'),
                     ('y', 'z'),
                     ('y', 'x'),
                     ('z', 'x'),
                     ('z', 'y')]:
            ratio = np.abs(sdf_3[axes[1]][1] - sdf_3[axes[1]][0]) \
                    / np.abs(sdf_3[axes[0]][1] - sdf_3[axes[0]][0])

            # Here, the tests are performed for both vector directions.
            img = func(sdf_3, 'A', 'B', 'C',
                       x=axes[0], y=axes[1],
                       normalize=False, hmin=False)
            assert img[0].shape[0] / img[0].shape[1] == approx(ratio, rel=1e-2)
            assert img[1].shape[0] / img[1].shape[1] == approx(ratio, rel=1e-2)

            img = func(sdf_3, 'A', 'B', 'C',
                       x=axes[0], y=axes[1],
                       x_pixels=default_pixels,
                       normalize=False, hmin=False)
            assert img[0].shape == (round(default_pixels * ratio),
                                    default_pixels)
            assert img[1].shape == (round(default_pixels * ratio),
                                    default_pixels)

            img = func(sdf_3, 'A', 'B', 'C',
                       x=axes[0], y=axes[1],
                       y_pixels=default_pixels,
                       normalize=False, hmin=False)
            assert img[0].shape == (default_pixels,
                                    round(default_pixels / ratio))
            assert img[1].shape == (default_pixels,
                                    round(default_pixels / ratio))

            img = func(sdf_3, 'A', 'B', 'C',
                       x_pixels=default_pixels * 2, y_pixels=default_pixels,
                       normalize=False, hmin=False)
            assert img[0].shape == (default_pixels, default_pixels * 2)
            assert img[1].shape == (default_pixels, default_pixels * 2)

    # 2D vector interpolation
    for axes in [('x', 'y'), ('y', 'x')]:
        ratio = np.abs(sdf_3[axes[1]][1] - sdf_3[axes[1]][0]) \
                / np.abs(sdf_3[axes[0]][1] - sdf_3[axes[0]][0])

        img_tuple = interpolate_2d_vec(sdf_2, 'A', 'B',
                                       x=axes[0], y=axes[1],
                                       normalize=False, hmin=False)
        assert img_tuple[0].shape[0] \
               / img_tuple[0].shape[1] == approx(ratio, rel=1e-2)
        assert img_tuple[1].shape[0] \
               / img_tuple[1].shape[1] == approx(ratio, rel=1e-2)

        img_tuple = interpolate_2d_vec(sdf_2, 'A', 'B',
                                       x=axes[0], y=axes[1],
                                       x_pixels=default_pixels,
                                       normalize=False, hmin=False)
        assert img_tuple[0].shape == (round(default_pixels * ratio),
                                      default_pixels)
        assert img_tuple[1].shape == (round(default_pixels * ratio),
                                      default_pixels)

        img_tuple = interpolate_2d_vec(sdf_2, 'A', 'B',
                                       x=axes[0], y=axes[1],
                                       y_pixels=default_pixels,
                                       normalize=False, hmin=False)
        assert img_tuple[0].shape == (default_pixels,
                                      round(default_pixels / ratio))
        assert img_tuple[1].shape == (default_pixels,
                                      round(default_pixels / ratio))

        img_tuple = interpolate_2d_vec(sdf_2, 'A', 'B',
                                       x_pixels=default_pixels * 2,
                                       y_pixels=default_pixels,
                                       normalize=False, hmin=False)
        assert img_tuple[0].shape == (default_pixels, default_pixels * 2)
        assert img_tuple[1].shape == (default_pixels, default_pixels * 2)


@mark.parametrize("backend", backends)
def test_irregular_bounds(backend: str) -> None:
    """
    When the aspect ratio of pixels is different than the aspect ratio in
    particle space, the interpolation functions should still correctly
    interpolate to the skewed grid.
    """
    data = {'x': [0], 'y': [0], 'A': [4], 'B': [7],
            'h': [0.9], 'rho': [0.4], 'm': [0.03]}
    sdf = SarracenDataFrame(data, params=dict())
    kernel = CubicSplineKernel()
    kernel_rad = kernel.get_radius()
    bounds = (-kernel_rad, kernel_rad)
    sdf.kernel = kernel
    sdf.backend = backend

    # Weight for 2D interpolation and 3D column interpolation.
    weight = sdf['m'] / (sdf['rho'] * sdf['h'] ** 2)

    # A mapping of pixel indices to x / y values in particle space.
    real_x = -kernel_rad + (np.arange(0, 50) + 0.5) * (2 * kernel_rad / 50)
    real_y = -kernel_rad + (np.arange(0, 25) + 0.5) * (2 * kernel_rad / 25)

    img = interpolate_2d(sdf, 'A',
                         x_pixels=50, y_pixels=25,
                         xlim=bounds, ylim=bounds,
                         normalize=False, hmin=False)
    img_vec = interpolate_2d_vec(sdf, 'A', 'B',
                                 x_pixels=50, y_pixels=25,
                                 xlim=bounds, ylim=bounds,
                                 normalize=False, hmin=False)
    for y in range(25):
        for x in range(50):
            r = np.sqrt(real_x[x]**2 + real_y[y]**2)
            w = kernel.w(r / sdf['h'][0], 2)
            assert img[y][x] == approx(weight[0] * sdf['A'][0] * w)
            assert img_vec[0][y][x] == approx(weight[0] * sdf['A'][0] * w)
            assert img_vec[1][y][x] == approx(weight[0] * sdf['B'][0] * w)

    # Convert the existing 2D dataframe to a 3D dataframe.
    sdf['C'] = 5
    sdf['z'] = -0.5
    sdf.zcol = 'z'

    column_func = kernel.get_column_kernel_func(1000)

    img = interpolate_3d_proj(sdf, 'A',
                              x_pixels=50, y_pixels=25,
                              xlim=bounds, ylim=bounds,
                              dens_weight=False,
                              normalize=False, hmin=False)
    img_vec = interpolate_3d_vec(sdf, 'A', 'B', 'C',
                                 x_pixels=50, y_pixels=25,
                                 xlim=bounds, ylim=bounds,
                                 dens_weight=False,
                                 normalize=False, hmin=False)
    for y in range(25):
        for x in range(50):
            r = np.sqrt(real_x[x]**2 + real_y[y]**2)
            w = column_func(r / sdf['h'][0], 2)
            assert img[y][x] == approx(weight[0] * sdf['A'][0] * w)
            assert img_vec[0][y][x] == approx(weight[0] * sdf['A'][0] * w)
            assert img_vec[1][y][x] == approx(weight[0] * sdf['B'][0] * w)

    # Weight for 3D cross-section interpolation.
    weight = sdf['m'] / (sdf['rho'] * sdf['h'] ** 3)

    img = interpolate_3d_cross(sdf, 'A',
                               x_pixels=50, y_pixels=25,
                               xlim=bounds, ylim=bounds,
                               z_slice=0,
                               normalize=False, hmin=False)
    img_vec = interpolate_3d_cross_vec(sdf, 'A', 'B', 'C',
                                       x_pixels=50, y_pixels=25,
                                       xlim=bounds, ylim=bounds,
                                       z_slice=0,
                                       normalize=False, hmin=False)
    for y in range(25):
        for x in range(50):
            r = np.sqrt(real_x[x]**2 + real_y[y]**2 + 0.5**2)
            w = kernel.w(r / sdf['h'][0], 3)
            assert img[y][x] == approx(weight[0] * sdf['A'][0] * w)
            assert img_vec[0][y][x] == approx(weight[0] * sdf['A'][0] * w)
            assert img_vec[1][y][x] == approx(weight[0] * sdf['B'][0] * w)

    real_z = -kernel_rad + 0.5 + (np.arange(0, 15) + 0.5) * 2 * kernel_rad / 15

    img = interpolate_3d_grid(sdf, 'A',
                              x_pixels=50, y_pixels=25, z_pixels=15,
                              xlim=bounds, ylim=bounds, zlim=bounds,
                              normalize=False, hmin=False)
    for z in range(15):
        for y in range(25):
            for x in range(50):
                r = np.sqrt(real_x[x]**2 + real_y[y]**2 + real_z[z]**2)
                w = kernel.w(r / sdf['h'][0], 3)
                assert img[z][y][x] == approx(weight[0] * sdf['A'][0] * w)


@mark.parametrize("backend", backends)
def test_oob_particles(backend: str) -> None:
    """
    Particles outside the bounds of an interpolation operation should be
    included in the result.
    """
    kernel = CubicSplineKernel()

    data_2 = {'x': [0], 'y': [0], 'A': [4], 'B': [3],
              'h': [1.9], 'rho': [0.4], 'm': [0.03]}
    sdf_2 = SarracenDataFrame(data_2, params=dict())
    sdf_2.kernel = kernel
    sdf_2.backend = backend

    data_3 = {'x': [0], 'y': [0], 'z': [0.5],
              'A': [4], 'B': [3], 'C': [2],
              'h': [1.9], 'rho': [0.4], 'm': [0.03]}
    sdf_3 = SarracenDataFrame(data_3, params=dict())
    sdf_3.kernel = kernel
    sdf_3.backend = backend

    # Weight for 2D interpolation, and 3D column interpolation.
    weight = sdf_2['m'] / (sdf_2['rho'] * sdf_2['h'] ** 2)

    # A mapping of pixel indices to x / y values in particle space.
    real_x = 1 + (np.arange(0, 25) + 0.5) * (1 / 25)
    real_y = 1 + (np.arange(0, 25) + 0.5) * (1 / 25)

    img = interpolate_2d(sdf_2, 'A',
                         x_pixels=25, y_pixels=25,
                         xlim=(1, 2), ylim=(1, 2),
                         normalize=False, hmin=False)
    img_vec = interpolate_2d_vec(sdf_2, 'A', 'B',
                                 x_pixels=25, y_pixels=25,
                                 xlim=(1, 2), ylim=(1, 2),
                                 normalize=False, hmin=False)
    line = interpolate_2d_line(sdf_2, 'A',
                               pixels=25,
                               xlim=(1, 2), ylim=(1, 2),
                               normalize=False, hmin=False)
    for y in range(25):
        r = np.sqrt(real_x[y]**2 + real_y[y]**2)
        w = kernel.w(r / sdf_2['h'][0], 2)
        assert line[y] == approx(weight[0] * sdf_2['A'][0] * w)
        for x in range(25):
            r = np.sqrt(real_x[x]**2 + real_y[y]**2)
            w = kernel.w(r / sdf_2['h'][0], 2)
            assert img[y][x] == approx(weight[0] * sdf_2['A'][0] * w)
            assert img_vec[0][y][x] == approx(weight[0] * sdf_2['A'][0] * w)
            assert img_vec[1][y][x] == approx(weight[0] * sdf_2['B'][0] * w)

    column_func = kernel.get_column_kernel_func(1000)

    img = interpolate_3d_proj(sdf_3, 'A',
                              x_pixels=25, y_pixels=25,
                              xlim=(1, 2), ylim=(1, 2),
                              dens_weight=False,
                              normalize=False, hmin=False)
    img_vec = interpolate_3d_vec(sdf_3, 'A', 'B', 'C',
                                 x_pixels=25, y_pixels=25,
                                 xlim=(1, 2), ylim=(1, 2),
                                 dens_weight=False,
                                 normalize=False, hmin=False)
    for y in range(25):
        for x in range(25):
            r = np.sqrt(real_x[x]**2 + real_y[y]**2)
            w = column_func(r / sdf_3['h'][0], 2)
            assert img[y][x] == approx(weight[0] * sdf_3['A'][0] * w)
            assert img_vec[0][y][x] == approx(weight[0] * sdf_3['A'][0] * w)
            assert img_vec[1][y][x] == approx(weight[0] * sdf_3['B'][0] * w)

    # Weight for 3D cross-sections.
    weight = sdf_3['m'] / (sdf_3['rho'] * sdf_3['h'] ** 3)

    img = interpolate_3d_cross(sdf_3, 'A',
                               x_pixels=25, y_pixels=25,
                               xlim=(1, 2), ylim=(1, 2),
                               z_slice=0,
                               normalize=False, hmin=False)
    img_vec = interpolate_3d_cross_vec(sdf_3, 'A', 'B', 'C',
                                       x_pixels=25, y_pixels=25,
                                       xlim=(1, 2), ylim=(1, 2),
                                       z_slice=0,
                                       normalize=False, hmin=False)
    for y in range(25):
        for x in range(25):
            r = np.sqrt(real_x[x]**2 + real_y[y]**2 + 0.5**2)
            w = kernel.w(r / sdf_3['h'][0], 3)
            assert img[y][x] == approx(weight[0] * sdf_3['A'][0] * w)
            assert img_vec[0][y][x] == approx(weight[0] * sdf_3['A'][0] * w)
            assert img_vec[1][y][x] == approx(weight[0] * sdf_3['B'][0] * w)

    real_z = 0.5 + (np.arange(0, 25) + 0.5) * (1 / 25)

    img = interpolate_3d_grid(sdf_3, 'A',
                              x_pixels=25, y_pixels=25, z_pixels=25,
                              xlim=(1, 2), ylim=(1, 2), zlim=(1, 2),
                              normalize=False, hmin=False)

    for z in range(25):
        for y in range(25):
            for x in range(25):
                r = np.sqrt(real_x[x]**2 + real_y[y]**2 + real_z[z]**2)
                w = kernel.w(r / sdf_3['h'][0], 3)
                assert img[z][y][x] == approx(weight[0] * sdf_3['A'][0] * w)


@mark.parametrize("backend", backends)
def test_invalid_region(backend: str) -> None:
    """
    Interpolation with invalid bounds should raise a ValueError.
    """
    data_2 = {'x': [0], 'y': [0], 'A': [4], 'B': [3], 'C': [2.5],
              'h': [0.9], 'rho': [0.4], 'm': [0.03]}
    sdf_2 = SarracenDataFrame(data_2, params=dict())
    data_3 = {'x': [0], 'y': [0], 'z': [-0.5],
              'A': [4], 'B': [3], 'C': [2.5],
              'h': [0.9], 'rho': [0.4], 'm': [0.03]}
    sdf_3 = SarracenDataFrame(data_3, params=dict())

    sdf_2.backend = backend
    sdf_3.backend = backend

    for b in [(-3, 3, 3, -3, 20, 20),
              (3, 3, 3, 3, 20, 20),
              (-3, 3, -3, 3, 0, 0)]:
        with raises(ValueError):
            interpolate_2d(sdf_2, 'A',
                           x_pixels=b[4], y_pixels=b[5],
                           xlim=(b[0], b[1]), ylim=(b[2], b[3]),
                           normalize=False, hmin=False)
        with raises(ValueError):
            interpolate_2d_vec(sdf_2, 'A', 'B', 'C',
                               x_pixels=b[4], y_pixels=b[5],
                               xlim=(b[0], b[1]), ylim=(b[2], b[3]),
                               normalize=False, hmin=False)
        # the first case will not fail for this type of interpolation.
        if not b[0] == -3 and not b[3] == -3:
            with raises(ValueError):
                interpolate_2d_line(sdf_2, 'A',
                                    pixels=b[4],
                                    xlim=(b[0], b[1]), ylim=(b[2], b[3]),
                                    normalize=False, hmin=False)
        with raises(ValueError):
            interpolate_3d_proj(sdf_3, 'A',
                                x_pixels=b[4], y_pixels=b[5],
                                xlim=(b[0], b[1]), ylim=(b[2], b[3]),
                                normalize=False, hmin=False)
        with raises(ValueError):
            interpolate_3d_vec(sdf_3, 'A', 'B', 'C',
                               xlim=(b[0], b[1]), ylim=(b[2], b[3]),
                               x_pixels=b[4], y_pixels=b[5],
                               normalize=False, hmin=False)
        with raises(ValueError):
            interpolate_3d_cross(sdf_3, 'A',
                                 x_pixels=b[4], y_pixels=b[5],
                                 xlim=(b[0], b[1]), ylim=(b[2], b[3]),
                                 z_slice=0,
                                 normalize=False, hmin=False)
        with raises(ValueError):
            interpolate_3d_cross_vec(sdf_3, 'A', 'B', 'C',
                                     xlim=(b[0], b[1]), ylim=(b[2], b[3]),
                                     x_pixels=b[4], y_pixels=b[5],
                                     z_slice=0,
                                     normalize=False, hmin=False)
        with raises(ValueError):
            interpolate_3d_grid(sdf_3, 'A',
                                x_pixels=b[4], y_pixels=b[5], z_pixels=10,
                                xlim=(b[0], b[1]),
                                ylim=(b[2], b[3]),
                                zlim=(-3, 3),
                                normalize=False, hmin=False)


@mark.parametrize("backend", backends)
@mark.parametrize("func", funcs)
@mark.parametrize("column", ['m', 'h'])
def test_required_columns(backend: str, func: Callable, column: str) -> None:
    """
    Interpolation without one of the required columns results in a KeyError.
    """
    # This test is currently expected to fail on both backends, since dropping
    # a column from a SarracenDataFrame returns a DataFrame.
    data = {'x': [-1, 1], 'y': [1, -1],
            'A': [2, 1.5], 'B': [5, 4], 'C': [3, 2],
            'h': [1.1, 1.3], 'rho': [0.55, 0.45], 'm': [0.04, 0.05]}

    if func in funcs3d + funcs3dvec:
        data['z'] = [1, -1]

    sdf = SarracenDataFrame(data, params=dict())
    sdf.backend = backend

    kwargs: Dict[str, Any] = dict()
    if func in funcs2d + funcs3d:
        kwargs['target'] = 'A'
    elif func in funcs2dvec:
        kwargs['target_x'] = 'A'
        kwargs['target_y'] = 'B'
    elif func in funcs3dvec:
        kwargs['target_x'] = 'A'
        kwargs['target_y'] = 'B'
        kwargs['target_z'] = 'C'

    sdf_dropped = sdf.drop(column, axis=1)

    with raises(KeyError):
        func(sdf_dropped, **kwargs)


@mark.parametrize("backend", backends)
def test_exact_interpolation(backend: str) -> None:
    """
    Exact interpolation over the entire effective area of a kernel should
    return 1 over the particle bounds, multiplied by the weight.
    """
    data_2 = {'x': [0], 'y': [0], 'A': [2],
              'h': [1.1], 'rho': [0.55], 'm': [0.04]}
    sdf_2 = SarracenDataFrame(data_2, params=dict())
    sdf_2.backend = backend
    data_3 = {'x': [0], 'y': [0], 'z': [1], 'A': [2],
              'h': [1.1], 'rho': [0.55], 'm': [0.04]}
    sdf_3 = SarracenDataFrame(data_3, params=dict())
    sdf_3.backend = backend

    kernel = CubicSplineKernel()
    w = sdf_2['m'] * sdf_2['A'] / (sdf_2['rho'] * sdf_2['h'] ** 2)

    bound = kernel.get_radius() * float(sdf_2['h'][0])
    img = interpolate_2d(sdf_2, 'A',
                         x_pixels=1,
                         xlim=(-bound, bound), ylim=(-bound, bound),
                         exact=True,
                         normalize=False, hmin=False)

    assert img.sum() == approx(w[0] * sdf_2['h'][0] ** 2 / (4 * bound ** 2))

    img = interpolate_3d_proj(sdf_3, 'A',
                              x_pixels=1,
                              xlim=(-bound, bound), ylim=(-bound, bound),
                              exact=True,
                              dens_weight=False,
                              normalize=False, hmin=False)

    assert img.sum() == approx(w[0] * sdf_2['h'][0] ** 2 / (4 * bound ** 2))


@mark.parametrize("backend", backends)
@mark.parametrize("func", funcs)
@mark.parametrize("dens_weight", [True, False])
def test_density_weighted(backend: str,
                          func: Callable,
                          dens_weight: bool) -> None:
    """
    Enabling density weighted interpolation will change the resultant image
    """

    data = {'x': [0], 'y': [0], 'A': [2], 'B': [3],
            'h': [0.5], 'rho': [0.25], 'm': [0.75]}

    if func in funcs3d + funcs3dvec:
        data['z'] = [0]
        data['C'] = [4]

    sdf = SarracenDataFrame(data, params=dict())
    sdf.backend = backend

    if func in funcs2d + funcs2dvec + funcscolumn:
        ndim = 2
    else:
        ndim = 3

    kernel = CubicSplineKernel()
    if func in funcscolumn:
        w = kernel.get_column_kernel()[0]
    else:
        w = kernel.w(0, ndim)

    weight = sdf['m'][0] / (sdf['h'][0] ** ndim)
    if not dens_weight:
        weight = weight / sdf['rho'][0]

    kwargs: Dict[str, Any] = {'xlim': (-1, 1), 'ylim': (-1, 1),
                              'dens_weight': dens_weight,
                              'normalize': False, 'hmin': False}

    if func in funcsline:
        kwargs['pixels'] = 1
    else:
        kwargs['x_pixels'] = 1
        kwargs['y_pixels'] = 1

    if func in [interpolate_3d_grid]:
        kwargs['z_pixels'] = 1
        kwargs['zlim'] = (-1, 1)

    if func in funcs2dvec:
        img = func(sdf, 'A', 'B', **kwargs)
        assert img[0] == weight * sdf['A'][0] * w
        assert img[1] == weight * sdf['B'][0] * w
    elif func in funcs3dvec:
        img = func(sdf, 'A', 'B', 'C', **kwargs)
        assert img[0] == weight * sdf['A'][0] * w
        assert img[1] == weight * sdf['B'][0] * w
    else:
        img = func(sdf, 'A', **kwargs)
        assert img == weight * sdf['A'][0] * w


@mark.parametrize("backend", backends)
@mark.parametrize("normalize", [False, True])
def test_normalize_interpolation(backend: str, normalize: bool) -> None:
    data_2 = {'x': [0], 'y': [0],
              'A': [2], 'B': [3],
              'h': [0.5], 'rho': [0.25], 'm': [0.75]}
    sdf_2 = SarracenDataFrame(data_2, params=dict())

    data_3 = {'x': [0], 'y': [0], 'z': [0],
              'A': [2], 'B': [3], 'C': [4],
              'h': [0.5], 'rho': [0.25], 'm': [0.75]}
    sdf_3 = SarracenDataFrame(data_3, params=dict())

    kernel = CubicSplineKernel()
    sdf_2.backend = backend
    sdf_3.backend = backend

    weight = sdf_2['m'][0] / (sdf_2['rho'][0] * sdf_2['h'][0] ** 2)
    weight2d = weight * kernel.w(0, 2)
    weight3d = weight / sdf_2['h'][0] * kernel.w(0, 3)
    weight3d_column = weight * kernel.get_column_kernel()[0]

    norm2d = 1.0
    norm3d = 1.0
    norm3d_column = 1.0

    if normalize:
        weight = sdf_2['m'][0] / (sdf_2['rho'][0] * sdf_2['h'][0] ** 2)
        norm2d = weight * kernel.w(0, 2)
        norm3d = weight / sdf_2['h'][0] * kernel.w(0, 3)
        norm3d_column = weight * kernel.get_column_kernel()[0]

    kwargs: Dict[str, Any] = {'xlim': (-1, 1), 'ylim': (-1, 1),
                              'dens_weight': False, 'normalize': normalize}

    img = interpolate_2d(sdf_2, 'A',
                         x_pixels=1, y_pixels=1, **kwargs)
    assert img == weight2d * sdf_2['A'][0] / norm2d

    img_tuple = interpolate_2d_vec(sdf_2, 'A', 'B',
                                   x_pixels=1, y_pixels=1, **kwargs)
    assert img_tuple[0] == weight2d * sdf_2['A'][0] / norm2d
    assert img_tuple[1] == weight2d * sdf_2['B'][0] / norm2d

    img = interpolate_2d_line(sdf_2, 'A',
                              pixels=1, **kwargs)
    assert img[0] == weight2d * sdf_2['A'][0] / norm2d

    img = interpolate_3d_proj(sdf_3, 'A',
                              x_pixels=1, y_pixels=1, **kwargs)
    assert img[0] == weight3d_column * sdf_2['A'][0] / norm3d_column

    img_tuple = interpolate_3d_vec(sdf_3, 'A', 'B', 'C',
                                   x_pixels=1, y_pixels=1, **kwargs)
    assert img_tuple[0] == weight3d_column * sdf_2['A'][0] / norm3d_column
    assert img_tuple[1] == weight3d_column * sdf_2['B'][0] / norm3d_column

    img = interpolate_3d_cross(sdf_3, 'A',
                               x_pixels=1, y_pixels=1, **kwargs)
    assert img[0] == weight3d * sdf_2['A'][0] / norm3d

    img_tuple = interpolate_3d_cross_vec(sdf_3, 'A', 'B', 'C',
                                         x_pixels=1, y_pixels=1, **kwargs)
    assert img_tuple[0] == weight3d * sdf_2['A'][0] / norm3d
    assert img_tuple[1] == weight3d * sdf_2['B'][0] / norm3d

    img = interpolate_3d_grid(sdf_3, 'A',
                              x_pixels=1, y_pixels=1, z_pixels=1,
                              zlim=(-1, 1), **kwargs)
    assert img[0] == weight3d * sdf_2['A'][0] / norm3d

    img = interpolate_3d_line(sdf_3, 'A',
                              pixels=1, **kwargs)
    assert img[0] == weight3d * sdf_2['A'][0] / norm3d


@mark.parametrize("backend", backends)
def test_exact_interpolation_culling(backend: str) -> None:
    data_2 = {'x': [0], 'y': [0], 'A': [2],
              'h': [0.4], 'rho': [0.1], 'm': [1]}
    sdf_2 = SarracenDataFrame(data_2, params=dict())
    sdf_2.backend = backend

    data_3 = {'x': [0], 'y': [0], 'z': [0], 'A': [2],
              'h': [0.4], 'rho': [0.1], 'm': [1]}
    sdf_3 = SarracenDataFrame(data_3, params=dict())
    sdf_3.backend = backend

    img_2 = sdf_2.sph_interpolate('A',
                                  x_pixels=5,
                                  xlim=(-1, 1), ylim=(-1, 1),
                                  exact=True)
    img_3 = interpolate_3d_proj(sdf_3, 'A',
                                x_pixels=5,
                                xlim=(-1, 1), ylim=(-1, 1),
                                exact=True)

    assert img_2[2, 4] != 0
    assert img_3[2, 4] != 0


@mark.parametrize("backend", backends)
def test_minimum_smoothing_length_2d(backend: str) -> None:
    """ Test that the minimum smoothing length evaluates correctly. """

    pixels = 5
    xlim, ylim = (-1, 1), (-1, 1)
    hmin = 0.5 * (xlim[1] - xlim[0]) / pixels

    data_a = {'rx': [0.3, -0.1, 0.1, 0.1, 0.05, -0.05, -0.25, -0.2],
              'ry': [0.0, 0.1, -0.1, 0.0, -0.05, 0.07, -0.3, -0.2],
              'h': [hmin, hmin, 0.3, 0.25, hmin, hmin, 0.2, hmin],
              'm': [0.56] * 8}
    sdf_a = SarracenDataFrame(data_a, params={'hfact': 1.2})

    data_b = {'rx': [0.3, -0.1, 0.1, 0.1, 0.05, -0.05, -0.25, -0.2],
              'ry': [0.0, 0.1, -0.1, 0.0, -0.05, 0.07, -0.3, -0.2],
              'h': [0.01, 0.01, 0.3, 0.25, 0.01, 0.01, 0.2, 0.01],
              'm': [0.56] * 8}
    sdf_b = SarracenDataFrame(data_b, params={'hfact': 1.2})

    sdf_a.backend = backend
    sdf_b.backend = backend

    for interpolate in [interpolate_2d]:
        grid = interpolate(sdf_a, 'rho',
                           x_pixels=pixels, y_pixels=pixels,
                           xlim=xlim, ylim=ylim,
                           normalize=False, hmin=False)
        grid_hmin = interpolate(sdf_b, 'rho',
                                x_pixels=pixels, y_pixels=pixels,
                                xlim=xlim, ylim=ylim,
                                normalize=False, hmin=True)

        assert (grid == grid_hmin).all()


@mark.parametrize("backend", backends)
def test_minimum_smoothing_length_3d(backend: str) -> None:
    """ Test that the minimum smoothing length evaluates correctly. """

    pixels = 5
    xlim, ylim = (-1, 1), (-1, 1)
    hmin = 0.5 * (xlim[1] - xlim[0]) / pixels

    data_a = {'rx': [0.3, -0.1, 0.1, 0.1, 0.05, -0.05, -0.25, -0.2],
              'ry': [0.0, 0.1, -0.1, 0.0, -0.05, 0.07, -0.3, -0.2],
              'rz': [0.1, 0.32, 0.03, -0.3, -0.2, 0.1, -0.06, 0.22],
              'h': [hmin, hmin, 0.3, 0.25, hmin, hmin, 0.2, hmin],
              'm': [0.56] * 8}
    sdf_a = SarracenDataFrame(data_a, params={'hfact': 1.2})

    data_b = {'rx': [0.3, -0.1, 0.1, 0.1, 0.05, -0.05, -0.25, -0.2],
              'ry': [0.0, 0.1, -0.1, 0.0, -0.05, 0.07, -0.3, -0.2],
              'rz': [0.1, 0.32, 0.03, -0.3, -0.2, 0.1, -0.06, 0.22],
              'h': [0.01, 0.01, 0.3, 0.25, 0.01, 0.01, 0.2, 0.01],
              'm': [0.56] * 8}
    sdf_b = SarracenDataFrame(data_b, params={'hfact': 1.2})

    sdf_a.backend = backend
    sdf_b.backend = backend

    functions_list: List[Callable] = [interpolate_3d_cross,
                                      interpolate_3d_proj,
                                      interpolate_3d_grid]
    for interpolate in functions_list:
        grid = interpolate(sdf_a, 'rho',
                           x_pixels=pixels, y_pixels=pixels,
                           xlim=xlim, ylim=ylim,
                           normalize=False, hmin=False)
        grid_hmin = interpolate(sdf_b, 'rho',
                                x_pixels=pixels, y_pixels=pixels,
                                xlim=xlim, ylim=ylim,
                                normalize=False, hmin=True)

        assert (grid == grid_hmin).all()


@mark.parametrize("backend", backends)
def test_minimum_smoothing_length_1d_lines(backend: str) -> None:
    """ Test that the minimum smoothing length evaluates correctly. """

    pixels = 5
    xlim, ylim, zlim = (-1, 1), (-0.5, 0.5), (-0.5, 0.5)

    hmin = 0.5 * np.sqrt((xlim[1] - xlim[0])**2
                         + (ylim[1] - ylim[0])**2) / pixels

    data_a = {'rx': [0.3, -0.1, 0.1, 0.1, 0.05, -0.05, -0.25, -0.2],
              'ry': [0.0, 0.1, -0.1, 0.0, -0.05, 0.07, -0.3, -0.2],
              'h': [hmin, hmin, 0.3, 0.25, hmin, hmin, hmin, hmin],
              'm': [0.56] * 8}
    sdf_a = SarracenDataFrame(data_a, params={'hfact': 1.2})

    data_b = {'rx': [0.3, -0.1, 0.1, 0.1, 0.05, -0.05, -0.25, -0.2],
              'ry': [0.0, 0.1, -0.1, 0.0, -0.05, 0.07, -0.3, -0.2],
              'h': [0.01, 0.01, 0.3, 0.25, 0.01, 0.01, 0.2, 0.01],
              'm': [0.56] * 8}
    sdf_b = SarracenDataFrame(data_b, params={'hfact': 1.2})

    sdf_a.backend = backend
    sdf_b.backend = backend

    grid = interpolate_2d_line(sdf_a, 'rho',
                               pixels=pixels,
                               xlim=xlim, ylim=ylim,
                               normalize=False, hmin=False)
    grid_hmin = interpolate_2d_line(sdf_b, 'rho',
                                    pixels=pixels,
                                    xlim=xlim, ylim=ylim,
                                    normalize=False, hmin=True)

    assert (grid == grid_hmin).all()

    hmin = 0.5 * np.sqrt((xlim[1] - xlim[0])**2
                         + (ylim[1] - ylim[0])**2
                         + (zlim[1] - zlim[0])**2) / pixels

    data_a = {'rx': [0.3, -0.1, 0.1, 0.1, 0.05, -0.05, -0.25, -0.2],
              'ry': [0.0, 0.1, -0.1, 0.0, -0.05, 0.07, -0.3, -0.2],
              'rz': [0.1, 0.32, 0.03, -0.3, -0.2, 0.1, -0.06, 0.22],
              'h': [hmin, hmin, 0.3, 0.25, hmin, hmin, hmin, hmin],
              'm': [0.56] * 8}
    sdf_a = SarracenDataFrame(data_a, params={'hfact': 1.2})

    data_b = {'rx': [0.3, -0.1, 0.1, 0.1, 0.05, -0.05, -0.25, -0.2],
              'ry': [0.0, 0.1, -0.1, 0.0, -0.05, 0.07, -0.3, -0.2],
              'rz': [0.1, 0.32, 0.03, -0.3, -0.2, 0.1, -0.06, 0.22],
              'h': [0.01, 0.01, 0.3, 0.25, 0.01, 0.01, 0.2, 0.01],
              'm': [0.56] * 8}
    sdf_b = SarracenDataFrame(data_b, params={'hfact': 1.2})

    sdf_a.backend = backend
    sdf_b.backend = backend

    grid = interpolate_3d_line(sdf_a, 'rho',
                               pixels=pixels,
                               xlim=xlim, ylim=ylim, zlim=zlim,
                               normalize=False, hmin=False)
    grid_hmin = interpolate_3d_line(sdf_b, 'rho',
                                    pixels=pixels,
                                    xlim=xlim, ylim=ylim, zlim=zlim,
                                    normalize=False, hmin=True)

    assert (grid == grid_hmin).all()
