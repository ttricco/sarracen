import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from sarracen import SarracenDataFrame
from sarracen.disc import surface_density
import pytest


def test_mass_equivalency() -> None:
    """ Column mass result should equal global params mass result. """

    # randomly place particles
    rng = np.random.default_rng(seed=5)
    x = rng.random(100)
    y = rng.random(100)
    z = rng.random(100)
    mass = [3.2e-4] * 100

    sdf = SarracenDataFrame(data={'x': x, 'y': y, 'z': z, 'mass': mass})
    sigma1 = surface_density(sdf)

    sdf = SarracenDataFrame(data={'x': x, 'y': y, 'z': z},
                            params={'mass': 3.2e-4})
    sigma2 = surface_density(sdf)

    assert_array_equal(sigma1, sigma2)


@pytest.mark.parametrize("geometry", ['cylindrical', 'spherical'])
def test_origin(geometry: str) -> None:
    """ Expect same profile regardless of origin. """

    rng = np.random.default_rng(seed=5)
    x = rng.random(100)
    y = rng.random(100)
    z = rng.random(100)

    sdf = SarracenDataFrame(data={'x': x, 'y': y, 'z': z},
                            params={'mass': 3.2e-4})
    sigma_zero = surface_density(sdf, origin=[0.0, 0.0, 0.0],
                                 bins=30, geometry=geometry)

    sdf = SarracenDataFrame(data={'x': x - 2e-8,
                                  'y': y + 2e-8,
                                  'z': z + 3e-9},
                            params={'mass': 3.2e-4})
    sigma_eps = surface_density(sdf, origin=[2e-8, -2e-8, 3e-9],
                                bins=30, geometry=geometry)

    sdf = SarracenDataFrame(data={'x': x + 5.6e4,
                                  'y': y - 8.7e3,
                                  'z': z + 5.4e6},
                            params={'mass': 3.2e-4})
    sigma_large = surface_density(sdf, origin=[5.6e4, -8.7e3, 5.4e6],
                                  bins=30, geometry=geometry)

    assert_allclose(sigma_zero, sigma_eps, atol=2e-8, rtol=0.0)
    assert_allclose(sigma_zero, sigma_large, atol=2e-8, rtol=0.0)


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
    sigma_in = surface_density(sdf, r_in=0.0, r_out=0.5, bins=100)
    sigma_out = surface_density(sdf, r_in=0.5, r_out=1.0, bins=100)
    sigma_all = surface_density(sdf, r_in=0.0, r_out=1.0, bins=200)

    assert_allclose(sigma_in, sigma_all[:100], atol=1e-15, rtol=0.0)
    assert_allclose(sigma_out, sigma_all[100:], atol=1e-15, rtol=0.0)
