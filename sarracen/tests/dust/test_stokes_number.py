import numpy as np
from pytest import approx

from sarracen.kernels import CubicSplineKernel
from sarracen import SarracenDataFrame
from sarracen.dust import stokes_number_2fluid


def test_uniform_gas_density():
    """
    Test a single dust particle in a uniform gas density.

    A few assumptions here.

    The gas density interpolated at the location of the dust particle is not
    calculated on the fly, but hard coded from an external calculation.

    The hfact for the dust is contrived to match the value of the kernel weight
    so that the density sum matches the analytic expression. The test
    intentionally uses the density sum (over 1 particle) because the function
    uses the analytic expression. They are contrived to be equivalent, so not
    an overly strong check on this.
    """

    c_s = 1.0
    params_gas = {'grainsize': 2.0,
                  'graindens': 2.0,
                  'gamma': 1.0,
                  'hfact': 1.2}

    # Make a regular cubic lattice of gas particles
    n = 20
    dx = 1.0 / n
    grid = (np.arange(n) + 0.5) * dx
    x, y, z = (a.ravel() for a in np.meshgrid(grid, grid, grid, indexing='ij'))

    # Gas values
    vel = np.zeros(len(x))
    h_gas = [params_gas['hfact'] * (1.0 / len(x) / 1.0) ** (1 / 3)] * len(x)
    mass_gas = [1.0 / len(x)] * len(x)  # unit density

    sdf_g = SarracenDataFrame({'x': x, 'y': y, 'z': z, 'h': h_gas,
                               'vx': vel, 'vy': vel, 'vz': vel,
                               'mass': mass_gas},
                              params=params_gas)

    # Create 1 dust particle in the centre
    kernel = CubicSplineKernel()
    params_dust = params_gas.copy()
    params_dust['hfact'] = kernel.w(0, 3)**(1/3)
    mass_dust = 0.01  # 1% of total gas mass
    h_dust = [1.0]
    sdf_d = SarracenDataFrame({'x': [0.5], 'y': [0.5], 'z': [0.5], 'h': h_dust,
                               'vx': [0.0], 'vy': [0.0], 'vz': [0.0],
                               'mass': mass_dust},
                              params=params_dust)

    stokes_calculated = stokes_number_2fluid(sdf_g, sdf_d, c_s=c_s)

    # Calculate expected Stokes number solution
    dust_density = mass_dust * kernel.w(0, 3) / h_dust[0]**3

    coef = np.sqrt(np.pi * params_gas['gamma'] / 8)
    interp_gas_dens = 1.0056236562869192
    denom = (interp_gas_dens + dust_density) * c_s
    tstop = coef * params_dust['grainsize'] * params_dust['graindens'] / denom

    interp_h = 1.2 * (mass_gas[0] / interp_gas_dens)**(1/3)
    tdyn = interp_h / c_s

    stokes_solution = tstop / tdyn

    assert len(stokes_calculated) == len(sdf_d)
    assert stokes_calculated[0] == approx(stokes_solution)
