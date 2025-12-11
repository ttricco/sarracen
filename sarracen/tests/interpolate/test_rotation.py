from typing import Callable, List, Tuple

import numpy as np
from numba import cuda
from numpy.testing import assert_allclose
from pytest import approx, mark
from scipy.spatial.transform import Rotation

from sarracen import SarracenDataFrame
from sarracen.kernels import CubicSplineKernel
from sarracen.interpolate import interpolate_3d_cross, interpolate_3d_proj, \
    interpolate_3d_vec, interpolate_3d_cross_vec, interpolate_3d_grid

backends = ['cpu']
if cuda.is_available():
    backends.append('gpu')


def rotate(target: Tuple[float, float, float],
           rot_z: float,
           rot_y: float,
           rot_x: float) -> Tuple[float, float, float]:
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
    r = 180 / np.pi

    pos_x1 = target[0] * np.cos(rot_z / r) - target[1] * np.sin(rot_z / r)
    pos_y1 = target[0] * np.sin(rot_z / r) + target[1] * np.cos(rot_z / r)
    pos_z1 = target[2]

    pos_x2 = pos_x1 * np.cos(rot_y / r) + pos_z1 * np.sin(rot_y / r)
    pos_y2 = pos_y1
    pos_z2 = pos_x1 * -np.sin(rot_y / r) + pos_z1 * np.cos(rot_y / r)

    pos_x3 = pos_x2
    pos_y3 = pos_y2 * np.cos(rot_x / r) - pos_z2 * np.sin(rot_x / r)
    pos_z3 = pos_y2 * np.sin(rot_x / r) + pos_z2 * np.cos(rot_x / r)

    return pos_x3, pos_y3, pos_z3


@mark.parametrize("backend", backends)
def test_nonstandard_rotation(backend: str) -> None:
    """
    Interpolation of a rotated dataframe with nonstandard angles.
    """
    data = {'x': [1], 'y': [1], 'z': [1],
            'A': [4], 'B': [5], 'C': [6],
            'h': [0.9], 'rho': [0.4], 'm': [0.03]}
    sdf = SarracenDataFrame(data, params=dict())
    kernel = CubicSplineKernel()
    sdf.kernel = kernel
    sdf.backend = backend

    column_kernel = kernel.get_column_kernel_func(1000)

    rot_z, rot_y, rot_x = 129, 34, 50

    img_col = interpolate_3d_proj(sdf, 'A',
                                  x_pixels=50, y_pixels=50,
                                  xlim=(-1, 1), ylim=(-1, 1),
                                  rotation=[rot_z, rot_y, rot_x],
                                  rot_origin=[0, 0, 0],
                                  dens_weight=False,
                                  normalize=False, hmin=False)
    img_xsec = interpolate_3d_cross(sdf, 'A',
                                    x_pixels=50, y_pixels=50,
                                    xlim=(-1, 1), ylim=(-1, 1),
                                    z_slice=0,
                                    rotation=[rot_z, rot_y, rot_x],
                                    rot_origin=[0, 0, 0],
                                    normalize=False, hmin=False)
    img_colvec = interpolate_3d_vec(sdf, 'A', 'B', 'C',
                                    x_pixels=50, y_pixels=50,
                                    xlim=(-1, 1), ylim=(-1, 1),
                                    rotation=[rot_z, rot_y, rot_x],
                                    rot_origin=[0, 0, 0],
                                    dens_weight=False,
                                    normalize=False, hmin=False)
    img_xsecvec = interpolate_3d_cross_vec(sdf, 'A', 'B', 'C',
                                           x_pixels=50, y_pixels=50,
                                           xlim=(-1, 1), ylim=(-1, 1),
                                           z_slice=0,
                                           rotation=[rot_z, rot_y, rot_x],
                                           rot_origin=[0, 0, 0],
                                           normalize=False, hmin=False)

    w_col = sdf['m'] / (sdf['rho'] * sdf['h'] ** 2)
    w_xsec = sdf['m'] / (sdf['rho'] * sdf['h'] ** 3)

    pos_x, pos_y, pos_z = rotate((1, 1, 1), rot_z, rot_y, rot_x)
    target_x, target_y, target_z = rotate((4, 5, 6), rot_z, rot_y, rot_x)

    real = -1 + (np.arange(0, 50) + 0.5) * (1 / 25)

    for y in range(50):
        for x in range(50):
            r = np.sqrt((pos_x - real[x]) ** 2
                        + (pos_y - real[y]) ** 2)
            w = column_kernel(r / sdf['h'][0], 3)
            assert img_col[y][x] == approx(w_col[0] * sdf['A'][0] * w)
            assert img_colvec[0][y][x] == approx(w_col[0] * target_x * w)
            assert img_colvec[1][y][x] == approx(w_col[0] * target_y * w)

            r = np.sqrt((pos_x - real[x])**2
                        + (pos_y - real[y])**2
                        + pos_z**2)
            w = kernel.w(r / sdf['h'][0], 3)
            assert img_xsec[y][x] == approx(w_xsec[0] * sdf['A'][0] * w)
            assert img_xsecvec[0][y][x] == approx(w_xsec[0] * target_x * w)
            assert img_xsecvec[1][y][x] == approx(w_xsec[0] * target_y * w)

    img_grid = interpolate_3d_grid(sdf, 'A',
                                   x_pixels=50, y_pixels=50, z_pixels=50,
                                   xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1),
                                   rotation=[rot_z, rot_y, rot_x],
                                   rot_origin=[0, 0, 0],
                                   normalize=False, hmin=False)

    for z in range(50):
        for y in range(50):
            for x in range(50):
                r = np.sqrt((pos_x - real[x])**2
                            + (pos_y - real[y])**2
                            + (pos_z - real[z])**2)
                w = kernel.w(r / sdf['h'][0], 3)
                grid_value = w_xsec[0] * sdf['A'][0] * w

                assert img_grid[z][y][x] == approx(grid_value)


@mark.parametrize("backend", backends)
def test_scipy_rotation_equivalency(backend: str) -> None:
    """
    For interpolation functions, a [z, y, x] rotation defined with degrees
    should be equivalent to the scipy version using from_euler().
    """
    data = {'x': [1], 'y': [1], 'z': [1],
            'A': [4], 'B': [3], 'C': [2],
            'h': [0.9], 'rho': [0.4], 'm': [0.03]}
    sdf = SarracenDataFrame(data, params=dict())
    kernel = CubicSplineKernel()
    sdf.kernel = kernel
    sdf.backend = backend

    rotation = [67, -34, 91]
    scipy_rotation = Rotation.from_euler('zyx', rotation, degrees=True)

    img1 = interpolate_3d_proj(sdf, 'A',
                               x_pixels=50, y_pixels=50,
                               xlim=(-1, 1), ylim=(-1, 1),
                               rotation=rotation,
                               rot_origin=[0, 0, 0],
                               normalize=False, hmin=False)
    img2 = interpolate_3d_proj(sdf, 'A',
                               x_pixels=50, y_pixels=50,
                               xlim=(-1, 1), ylim=(-1, 1),
                               rotation=scipy_rotation,
                               rot_origin=[0, 0, 0],
                               normalize=False, hmin=False)
    assert_allclose(img1, img2)

    img1 = interpolate_3d_cross(sdf, 'A',
                                x_pixels=50, y_pixels=50,
                                xlim=(-1, 1), ylim=(-1, 1),
                                z_slice=0,
                                rotation=rotation,
                                rot_origin=[0, 0, 0],
                                normalize=False, hmin=False)
    img2 = interpolate_3d_cross(sdf, 'A',
                                x_pixels=50, y_pixels=50,
                                xlim=(-1, 1), ylim=(-1, 1),
                                z_slice=0,
                                rotation=scipy_rotation,
                                rot_origin=[0, 0, 0],
                                normalize=False, hmin=False)
    assert_allclose(img1, img2)

    img1_tuple = interpolate_3d_vec(sdf, 'A', 'B', 'C',
                                    x_pixels=50, y_pixels=50,
                                    xlim=(-1, 1), ylim=(-1, 1),
                                    rotation=rotation,
                                    rot_origin=[0, 0, 0],
                                    normalize=False, hmin=False)
    img2_tuple = interpolate_3d_vec(sdf, 'A', 'B', 'C',
                                    x_pixels=50, y_pixels=50,
                                    xlim=(-1, 1), ylim=(-1, 1),
                                    rotation=scipy_rotation,
                                    rot_origin=[0, 0, 0],
                                    normalize=False,
                                    hmin=False)
    assert_allclose(img1_tuple[0], img2_tuple[0])
    assert_allclose(img1_tuple[1], img2_tuple[1])

    img1_tuple = interpolate_3d_cross_vec(sdf, 'A', 'B', 'C',
                                          x_pixels=50, y_pixels=50,
                                          xlim=(-1, 1), ylim=(-1, 1),
                                          z_slice=0,
                                          rotation=rotation,
                                          rot_origin=[0, 0, 0],
                                          normalize=False, hmin=False)
    img2_tuple = interpolate_3d_cross_vec(sdf, 'A', 'B', 'C',
                                          x_pixels=50, y_pixels=50,
                                          xlim=(-1, 1), ylim=(-1, 1),
                                          z_slice=0,
                                          rotation=scipy_rotation,
                                          rot_origin=[0, 0, 0],
                                          normalize=False, hmin=False)
    assert_allclose(img1_tuple[0], img2_tuple[0])
    assert_allclose(img1_tuple[1], img2_tuple[1])

    img1 = interpolate_3d_grid(sdf, 'A',
                               x_pixels=50, y_pixels=50,
                               xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1),
                               rotation=rotation,
                               rot_origin=[0, 0, 0],
                               normalize=False, hmin=False)
    img2 = interpolate_3d_grid(sdf, 'A',
                               x_pixels=50, y_pixels=50,
                               xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1),
                               rotation=scipy_rotation,
                               rot_origin=[0, 0, 0],
                               normalize=False, hmin=False)
    assert_allclose(img1, img2)


@mark.parametrize("backend", backends)
def test_quaternion_rotation(backend: str) -> None:
    """
    Test quaternion rotation defined using scipy.
    """
    data = {'x': [1], 'y': [1], 'z': [1],
            'A': [4], 'B': [3], 'C': [2],
            'h': [1.9], 'rho': [0.4], 'm': [0.03]}
    sdf = SarracenDataFrame(data, params=dict())
    kernel = CubicSplineKernel()
    sdf.kernel = kernel
    sdf.backend = backend

    column_kernel = kernel.get_column_kernel_func(1000)

    quat = Rotation.from_quat([5, 3, 8, 1])
    img_col = interpolate_3d_proj(sdf, 'A',
                                  x_pixels=50, y_pixels=50,
                                  xlim=(-1, 1), ylim=(-1, 1),
                                  rotation=quat,
                                  rot_origin=[0, 0, 0],
                                  dens_weight=False,
                                  normalize=False, hmin=False)
    img_colvec = interpolate_3d_vec(sdf, 'A', 'B', 'C',
                                    x_pixels=50, y_pixels=50,
                                    xlim=(-1, 1), ylim=(-1, 1),
                                    rotation=quat,
                                    rot_origin=[0, 0, 0],
                                    dens_weight=False,
                                    normalize=False, hmin=False)
    img_xsec = interpolate_3d_cross(sdf, 'A',
                                    x_pixels=50, y_pixels=50,
                                    xlim=(-1, 1), ylim=(-1, 1),
                                    z_slice=0,
                                    rotation=quat,
                                    rot_origin=[0, 0, 0],
                                    normalize=False, hmin=False)
    img_xsecvec = interpolate_3d_cross_vec(sdf, 'A', 'B', 'C',
                                           x_pixels=50, y_pixels=50,
                                           xlim=(-1, 1), ylim=(-1, 1),
                                           z_slice=0,
                                           rotation=quat,
                                           rot_origin=[0, 0, 0],
                                           normalize=False, hmin=False)

    w_col = sdf['m'] / (sdf['rho'] * sdf['h'] ** 2)
    w_xsec = sdf['m'] / (sdf['rho'] * sdf['h'] ** 3)

    pos = quat.apply([1, 1, 1])
    val = quat.apply([sdf['A'][0], sdf['B'][0], sdf['C'][0]])
    real = -1 + (np.arange(0, 50) + 0.5) * (1 / 25)

    for y in range(50):
        for x in range(50):
            r = np.sqrt((pos[0] - real[x])**2 + (pos[1] - real[y])**2)
            w = column_kernel(r / sdf['h'][0], 3)

            assert img_col[y][x] == approx(w_col[0] * sdf['A'][0] * w)
            assert img_colvec[0][y][x] == approx(w_col[0] * val[0] * w)
            assert img_colvec[1][y][x] == approx(w_col[0] * val[1] * w)

            r = np.sqrt((pos[0] - real[x])**2
                        + (pos[1] - real[y])**2
                        + pos[2]**2)
            w = kernel.w(r / sdf['h'][0], 3)

            assert img_xsec[y][x] == approx(w_xsec[0] * sdf['A'][0] * w)

            assert img_xsecvec[0][y][x] == approx(w_xsec[0] * val[0] * w)
            assert img_xsecvec[1][y][x] == approx(w_xsec[0] * val[1] * w)

    img_grid = interpolate_3d_grid(sdf, 'A',
                                   x_pixels=50, y_pixels=50, z_pixels=50,
                                   xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1),
                                   rotation=quat,
                                   rot_origin=[0, 0, 0],
                                   normalize=False, hmin=False)

    for z in range(50):
        for y in range(50):
            for x in range(50):
                r = np.sqrt((pos[0] - real[x])**2
                            + (pos[1] - real[y])**2
                            + (pos[2] - real[z])**2)
                w = kernel.w(r / sdf['h'][0], 3)
                grid_value = w_xsec[0] * sdf['A'][0] * w

                assert img_grid[z][y][x] == approx(grid_value)


@mark.parametrize("backend", backends)
def test_rotation_stability(backend: str) -> None:
    """
    Rotation should not change values at the rotation origin.
    """
    data = {'x': [1.0, 3.0], 'y': [1.0, -1.0], 'z': [1.0, -0.5],
            'A': [4.0, 3.0], 'B': [3.0, 2.0], 'C': [1.0, 1.5],
            'h': [0.9, 1.4], 'rho': [0.4, 0.6], 'm': [0.03, 0.06]}
    sdf = SarracenDataFrame(data, params=dict())
    kernel = CubicSplineKernel()
    sdf.kernel = kernel
    sdf.backend = backend

    real = -1 + (np.arange(0, 50) + 0.5) * (1 / 25)
    pixel_x, pixel_y = 12, 30

    functions_list: List[Callable] = [interpolate_3d_proj,
                                      interpolate_3d_cross]
    for func in functions_list:
        img = func(sdf, 'A',
                   x_pixels=50, y_pixels=50,
                   xlim=(-1, 1), ylim=(-1, 1),
                   normalize=False, hmin=False)
        img_rot = func(sdf, 'A',
                       x_pixels=50, y_pixels=50,
                       xlim=(-1, 1), ylim=(-1, 1),
                       rotation=[237, 0, 0],
                       rot_origin=[real[pixel_x], real[pixel_y], 0],
                       normalize=False, hmin=False)

        assert img[pixel_y][pixel_x] == approx(img_rot[pixel_y][pixel_x])


@mark.parametrize("backend", backends)
def test_axes_rotation_separation(backend: str) -> None:
    """
    Rotations should be independent of the defined x & y interpolation axes.
    Similar to test_image_transpose(), but a rotation is applied to all
    interpolations.
    """
    data = {'x': [-1, 1], 'y': [1, -1], 'z': [1, -1],
            'A': [2, 1.5], 'B': [2, 2], 'C': [4, 3],
            'h': [1.1, 1.3], 'rho': [0.55, 0.45], 'm': [0.04, 0.05]}
    sdf = SarracenDataFrame(data, params=dict())
    sdf.backend = backend

    rotation = [234, 90, 48]

    img1 = interpolate_3d_proj(sdf, 'A',
                               x_pixels=50,  y_pixels=50,
                               xlim=(-1, 1), ylim=(-1, 1),
                               rotation=rotation,
                               normalize=False, hmin=False)
    img2 = interpolate_3d_proj(sdf, 'A', x='y', y='x',
                               x_pixels=50,  y_pixels=50,
                               xlim=(-1, 1), ylim=(-1, 1),
                               rotation=rotation,
                               normalize=False, hmin=False)
    assert_allclose(img1, img2.T)

    img1_tuple = interpolate_3d_vec(sdf, 'A', 'B', 'C',
                                    x_pixels=50, y_pixels=50,
                                    xlim=(-1, 1), ylim=(-1, 1),
                                    rotation=rotation,
                                    normalize=False, hmin=False)
    img2_tuple = interpolate_3d_vec(sdf, 'A', 'B', 'C', x='y', y='x',
                                    x_pixels=50, y_pixels=50,
                                    xlim=(-1, 1), ylim=(-1, 1),
                                    rotation=rotation,
                                    normalize=False, hmin=False)
    assert_allclose(img1_tuple[0], img2_tuple[0].T)
    assert_allclose(img1_tuple[1], img2_tuple[1].T)

    img1 = interpolate_3d_cross(sdf, 'A',
                                x_pixels=50, y_pixels=50,
                                xlim=(-1, 1), ylim=(-1, 1),
                                z_slice=0,
                                rotation=rotation,
                                normalize=False, hmin=False)
    img2 = interpolate_3d_cross(sdf, 'A', x='y', y='x',
                                x_pixels=50, y_pixels=50,
                                xlim=(-1, 1), ylim=(-1, 1),
                                z_slice=0,
                                rotation=rotation,
                                normalize=False, hmin=False)
    assert_allclose(img1, img2.T)

    img1_tuple = interpolate_3d_cross_vec(sdf, 'A', 'B', 'C',
                                          x_pixels=50, y_pixels=50,
                                          xlim=(-1, 1), ylim=(-1, 1),
                                          z_slice=0,
                                          rotation=rotation,
                                          normalize=False, hmin=False)
    img2_tuple = interpolate_3d_cross_vec(sdf, 'A', 'B', 'C',
                                          x='y', y='x',
                                          x_pixels=50, y_pixels=50,
                                          xlim=(-1, 1), ylim=(-1, 1),
                                          z_slice=0,
                                          rotation=rotation,
                                          normalize=False, hmin=False)
    assert_allclose(img1_tuple[0], img2_tuple[0].T)
    assert_allclose(img1_tuple[1], img2_tuple[1].T)

    img1 = interpolate_3d_grid(sdf, 'A',
                               x_pixels=50, y_pixels=50, z_pixels=50,
                               xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1),
                               rotation=rotation,
                               normalize=False, hmin=False)
    img2 = interpolate_3d_grid(sdf, 'A', x='y', y='x',
                               x_pixels=50, y_pixels=50, z_pixels=50,
                               xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1),
                               rotation=rotation,
                               normalize=False, hmin=False)
    assert_allclose(img1, img2.transpose(0, 2, 1))


@mark.parametrize("backend", backends)
def test_axes_rotation_equivalency(backend: str) -> None:
    """
    A rotated interpolation (at multiples of 90 degrees) should be equivalent
    to a transformed interpolation with different x & y axes. For example,
    an interpolation rotated by 180 degrees around the z axis should be
    equivalent to the transpose of an unaltered interpolation.
    """
    data = {'x': [-1, 1], 'y': [1, -1], 'z': [1, -1],
            'A': [2, 1.5], 'B': [2, 2], 'C': [4, 3],
            'h': [1.1, 1.3], 'rho': [0.55, 0.45], 'm': [0.04, 0.05]}
    sdf = SarracenDataFrame(data, params=dict())
    sdf.backend = backend

    x, y, z = 'x', 'y', 'z'
    flip_x, flip_y, flip_z = False, False, False
    for i_z in range(4):
        for i_y in range(4):
            for i_x in range(4):
                rot_x, rot_y, rot_z = i_x * 90, i_y * 90, i_z * 90

                functions_list: List[Callable] = [interpolate_3d_proj,
                                                  interpolate_3d_cross]
                for func in functions_list:
                    img1 = func(sdf, 'A',
                                x_pixels=50, y_pixels=50,
                                xlim=(-1, 1), ylim=(-1, 1),
                                rotation=[rot_z, rot_y, rot_x],
                                rot_origin=[0, 0, 0],
                                normalize=False, hmin=False)
                    img2 = func(sdf, 'A',
                                x=x, y=y,
                                x_pixels=50, y_pixels=50,
                                xlim=(-1, 1), ylim=(-1, 1),
                                normalize=False, hmin=False)

                    img2 = img2 if not flip_x else np.flip(img2, 1)
                    img2 = img2 if not flip_y else np.flip(img2, 0)
                    assert_allclose(img1, img2)

                y, z = z, y
                flip_y, flip_z = not flip_z, flip_y
            x, z = z, x
            flip_x, flip_z = flip_z, not flip_x
        x, y = y, x
        flip_x, flip_y = not flip_y, flip_x


def test_com_rotation() -> None:
    """ Rotation around centre of mass should equal rotation around 0
    when positions have been reset around com. """

    x = np.array([0.80500292, 0.80794079, 0.51532556, 0.28580138, 0.0539307,
                  0.38336888, 0.40847321, 0.04527519, 0.04875771, 0.99917612])
    y = np.array([0.55559612, 0.2714516, 0.87965117, 0.06421444, 0.67918153,
                  0.8700885, 0.22731853, 0.89544824, 0.87219547, 0.01851722])
    z = np.array([0.32494264, 0.80621533, 0.31645209, 0.14903858, 0.69851199,
                  0.4485441, 0.79893949, 0.23551646, 0.31978465, 0.79987953])
    h = np.array([0.08582579, 0.08449268, 0.03678807, 0.09510229, 0.03994252,
                  0.09364420, 0.05561597, 0.02401353, 0.07414216, 0.06743897])
    val = np.array([3.9045891, 7.9793389, 3.8047537, 7.1325786, 6.125178,
                    9.4100098, 9.9167672, 7.2367625, 8.0884381, 1.5286502])
    mass = 3.2e-4

    sdf_com = SarracenDataFrame({'x': x, 'y': y, 'z': z, 'h': h, 'val': val},
                                params={'mass': mass, 'hfact': 1.2})

    com = [(x * mass).sum() / (10 * mass),
           (y * mass).sum() / (10 * mass),
           (z * mass).sum() / (10 * mass)]
    x = x - com[0]
    y = y - com[1]
    z = z - com[2]
    sdf_zero = SarracenDataFrame({'x': x, 'y': y, 'z': z, 'h': h, 'val': val},
                                 params={'mass': mass, 'hfact': 1.2})

    functions_list: List[Callable] = [interpolate_3d_proj,
                                      interpolate_3d_cross]
    for func in functions_list:
        img1 = func(sdf_com, 'val',
                    x_pixels=50, y_pixels=50,
                    rotation=[35, 60, 75],
                    rot_origin='com')
        img2 = func(sdf_zero, 'val',
                    x_pixels=50, y_pixels=50,
                    rotation=[35, 60, 75],
                    rot_origin=[0, 0, 0])

        assert_allclose(img1, img2)
