import numpy as np
from pytest import approx, mark

from sarracen.kernels import BaseKernel, CubicSplineKernel, QuinticSplineKernel
from sarracen import SarracenDataFrame
from sarracen.dust import stokes_number_2fluid


def _create_gas_data(nx: int = 20,
                     vel: float = 0.0,
                     kernel: BaseKernel = CubicSplineKernel(),
                     hfact: float = 1.2,
                     grainsize: float = 1.391,
                     graindens: float = 0.0542,
                     gamma: float = 1.0) -> SarracenDataFrame:
    """ Create a gas density SarracenDataFrame. """

    params_gas = {'grainsize': grainsize,
                  'graindens': graindens,
                  'gamma': gamma,
                  'hfact': hfact}

    dx = 1.0 / nx
    grid = (np.arange(nx) + 0.5) * dx
    x, y, z = (a.ravel() for a in np.meshgrid(grid, grid, grid, indexing='ij'))

    # Gas values
    h_gas = [params_gas['hfact'] * (1.0 / len(x) / 1.0) ** (1 / 3)] * len(x)
    rho_gas = [0.6] * len(x)
    mass_gas = [1.0 / len(x)] * len(x)  # unit density
    vel_gas = [vel] * len(x)

    sdf_g = SarracenDataFrame({'x': x, 'y': y, 'z': z, 'h': h_gas,
                               'vx': vel_gas, 'vy': vel_gas, 'vz': vel_gas,
                               'mass': mass_gas, 'rho': rho_gas},
                              params=params_gas)
    sdf_g.kernel = kernel

    return sdf_g


def _create_single_dust_data(sdf_g: SarracenDataFrame) -> SarracenDataFrame:
    """ Create a single dust particle SarracenDataFrame. """

    # Create 1 dust particle in the centre
    params_dust = sdf_g.params.copy()
    params_dust['hfact'] = sdf_g.kernel.w(0, 3)**(1/3)
    mass_dust = 0.01  # 1% of total gas mass
    h_dust = [1.0]
    sdf_d = SarracenDataFrame({'x': [0.5], 'y': [0.5], 'z': [0.5], 'h': h_dust,
                               'vx': [0.0], 'vy': [0.0], 'vz': [0.0],
                               'mass': mass_dust},
                              params=params_dust)
    sdf_d.kernel = sdf_g.kernel

    return sdf_d


def _expected_stokes_number(sdf_g: SarracenDataFrame,
                            sdf_d: SarracenDataFrame,
                            c_s: float = 1.0,
                            kwok: bool = True) -> float:
    """ Calculate the expected Stokes number. """

    # Calculate expected Stokes number solution
    mass_dust = sdf_d.loc[0, 'mass']
    assert isinstance(mass_dust, float)

    h_dust = sdf_d.loc[0, 'h']
    assert isinstance(h_dust, float)

    dust_density = mass_dust * sdf_g.kernel.w(0, 3) / h_dust**3

    coef = np.sqrt(np.pi * sdf_g.params['gamma'] / 8)

    if isinstance(sdf_g.kernel, CubicSplineKernel):
        interp_gas_rho = 1.0056236562869192
    else:
        interp_gas_rho = 1.0001440759936475

    if kwok:
        rho = sdf_g.loc[0, 'rho']
        assert isinstance(rho, float)

        gas_vel = sdf_g.loc[0, 'vx']
        assert isinstance(gas_vel, float)

        interp_gas_vel = interp_gas_rho * gas_vel / rho
        deltavsq = 3 * interp_gas_vel ** 2  # vx**2 + vy**2 + vz**2
        f = np.sqrt(1 + 9 * np.pi / 128 * deltavsq / c_s**2)
    else:
        f = 1.0

    den = (interp_gas_rho + dust_density) * c_s * f
    tstop = coef * sdf_d.params['grainsize'] * sdf_d.params['graindens'] / den

    gas_mass = sdf_g.loc[0, 'mass']
    assert isinstance(gas_mass, float)

    interp_h = 1.2 * (gas_mass / interp_gas_rho)**(1/3)
    tdyn = interp_h / c_s

    return tstop / tdyn


@mark.parametrize("kernel", [CubicSplineKernel(), QuinticSplineKernel()])
def test_single_dust_in_uniform_gas_density(kernel: BaseKernel) -> None:
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

    sdf_g = _create_gas_data(vel=0.0, kernel=kernel)
    sdf_d = _create_single_dust_data(sdf_g)

    # Stokes number from dust module
    stokes_calculated = stokes_number_2fluid(sdf_g, sdf_d, c_s=c_s)

    # Calculate expected Stokes number solution
    stokes_solution = _expected_stokes_number(sdf_g, sdf_d, c_s=c_s)

    assert len(stokes_calculated) == len(sdf_d)
    assert stokes_calculated[0] == approx(stokes_solution)


@mark.parametrize("kernel", [CubicSplineKernel(), QuinticSplineKernel()])
@mark.parametrize("kwok", [True, False])
def test_single_dust_with_deltav(kernel: BaseKernel,
                                 kwok: bool) -> None:
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
    gas_vel = 1.47

    sdf_g = _create_gas_data(vel=gas_vel, kernel=kernel)
    sdf_d = _create_single_dust_data(sdf_g)

    # Stokes number from dust module
    stokes_calculated = stokes_number_2fluid(sdf_g, sdf_d, c_s=c_s, kwok=kwok)

    # # Calculate expected Stokes number solution
    stokes_solution = _expected_stokes_number(sdf_g, sdf_d, c_s=c_s, kwok=kwok)

    assert len(stokes_calculated) == len(sdf_d)
    assert stokes_calculated[0] == approx(stokes_solution)
