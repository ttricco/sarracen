import numpy as np
from numpy.testing import assert_array_equal
from sarracen import SarracenDataFrame
from sarracen.disc import angular_momentum


def test_mass_equivalency() -> None:
    """ Column mass result should equal global params mass result. """

    # randomly place particles
    rng = np.random.default_rng(seed=5)
    x = rng.random(100)
    y = rng.random(100)
    z = rng.random(100)
    vx = rng.random(100)
    vy = rng.random(100)
    vz = rng.random(100)
    mass = [3.2e-4] * 100

    sdf = SarracenDataFrame(data={'x': x, 'y': y, 'z': z,
                                  'vx': vx, 'vy': vy, 'vz': vz,
                                  'mass': mass})
    Lx1, Ly1, Lz1 = angular_momentum(sdf)

    sdf = SarracenDataFrame(data={'x': x, 'y': y, 'z': z,
                                  'vx': vx, 'vy': vy, 'vz': vz},
                            params={'mass': 3.2e-4})
    Lx2, Ly2, Lz2 = angular_momentum(sdf)

    assert_array_equal(Lx1, Lx2)
    assert_array_equal(Ly1, Ly2)
    assert_array_equal(Lz1, Lz2)


def test_parts_vs_whole() -> None:
    """ Profiles should be the same for matching bins. """

    # randomly place particles
    rng = np.random.default_rng(seed=5)
    x = rng.random(100)
    y = rng.random(100)
    z = rng.random(100)
    vx = rng.random(100)
    vy = rng.random(100)
    vz = rng.random(100)
    mass = [3.2e-4] * 100

    sdf = SarracenDataFrame(data={'x': x, 'y': y, 'z': z,
                                  'vx': vx, 'vy': vy, 'vz': vz,
                                  'mass': mass})
    Lx_in, Ly_in, Lz_in = angular_momentum(sdf, r_in=0.0, r_out=0.5, bins=100)
    Lx_ex, Ly_ex, Lz_ex = angular_momentum(sdf, r_in=0.5, r_out=1.0, bins=100)
    Lx, Ly, Lz = angular_momentum(sdf, r_in=0.0, r_out=1.0, bins=200)

    assert_array_equal(Lx_in, Lx[:100])
    assert_array_equal(Ly_in, Ly[:100])
    assert_array_equal(Lz_in, Lz[:100])
    assert_array_equal(Lx_ex, Lx[100:])
    assert_array_equal(Ly_ex, Ly[100:])
    assert_array_equal(Lz_ex, Lz[100:])
