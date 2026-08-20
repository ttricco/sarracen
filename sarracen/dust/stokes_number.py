import numpy as np
from sklearn.neighbors import KDTree

from ..kernels.base_kernel import BaseKernel
from ..sarracen_dataframe import SarracenDataFrame


def _verify_columns(data: 'SarracenDataFrame',
                    kwok: bool) -> None:
    """
    Verify that the required columns exist in `data`.

    Parameters
    ----------
    data : SarracenDataFrame
        The particle dataset to check.
    kwok : bool
        If True, then the velocity columns are also checked.

    Raises
    ------
    KeyError
        If a label in columns does not exist in `data`.
    """
    if data.xcol is None:
        raise KeyError("No x-directional column specified.")
    if data.ycol is None:
        raise KeyError("No y-directional column specified.")
    if data.zcol is None:
        raise KeyError("No z-directional column specified.")
    if data.hcol is None:
        raise KeyError("No smoothing length column specified.")
    if kwok:
        if data.vxcol is None:
            raise KeyError("No x-velocity column specified.")
        if data.vycol is None:
            raise KeyError("No y-velocity column specified.")
        if data.vzcol is None:
            raise KeyError("No z-velocity column specified.")


def _get_dust_locations(gas_locations: np.ndarray,
                        dust_locations: np.ndarray,
                        h_gas: np.ndarray,
                        kernel_radius: float) -> np.ndarray:
    """
    Compute the dust locations within each gas particles smoothing volume.

    Parameters
    ----------
    gas_locations : ndarray
        n-dimensioal array of gas particle locations.
    dust_locations : ndarray
        n-dimensional array of dust particle locations.
    h_gas : ndarray
        1-dimensional array of gas particle smoothing lengths.
    kernel_radius : float
        The radial extent of the smoothing kernel.

    Returns
    -------
    ndarray
        An array with size of len(gas_positions). Each element is an ndarray
        with the integer indices of the dust particles that are within the
        corresponding gas particle's smoothing volume.
    """

    if len(gas_locations) != len(h_gas):
        raise ValueError("Length of gas locations and h is inconsistent.")

    # Create a tree to store the dust locations
    dust_tree = KDTree(dust_locations, leaf_size=10)

    # Query the tree using the gas particles.
    # Makes a neighbour list of dust locations for each gas particle.

    # "dust_locations_per_gas_particle" is an array of size len(gas_positions).
    # Each element of "dust_locations_per_gas_particle" is an integer array of
    # indices of "dust_positions" that are neighbours of that gas particle.
    dust_locations_per_gas = dust_tree.query_radius(gas_locations,
                                                    r=kernel_radius * h_gas)

    return dust_locations_per_gas


def _invert_dust_locations(dust_positions: np.ndarray,
                           dust_neighbours: np.ndarray) -> list[np.ndarray]:
    """
    Invert the list of dust neighbours.

    Given a list of gas particles and their dust particle neighbours, invert
    the list such that it is a list of dust particles and the gas particles
    that contribute to that location of the dust particles.

    Note that this is not a list of gas particles within a dust particles
    smoothing volume. Only the smoothing length of the gas is used. The dust
    particle is within the smoothing volume of each of its gas "neighbours",
    but because the dust smoothing lengths are not used, there is no guarantee
    that the gas particle is within the dust smoothing volume.

    Parameters
    ----------
    dust_positions : ndarray
        n-dimensional array of dust particle locations.
    dust_neighbours : ndarray
        An array with length the number of gas particles. Each element is an
        array of dust particle integer indices that are neighbours of the
        corresponding gas particle.

    Returns
    -------
    list of ndarray
        The inverted list of dust neighbours. Each element corresponds to a
        dust particle and contains an ndarray of gas particles that contribute
        to the location of this dust particle
    """

    # Initialize structure to hold the list of gas particles that
    # each dust location gets contribution from
    # Use regular list instead of ndarray for append efficiency
    invert: list[list[int]] = [[] for _ in range(len(dust_positions))]

    # Loop over each gas-dust neighbour pair
    for gas_index, dust_locations in enumerate(dust_neighbours):
        for dust_index in dust_locations:
            invert[dust_index].append(gas_index)

    # Convert the list of lists to a list of ndarrays
    return [np.asarray(indices, dtype=np.int64) for indices in invert]


def _interpolate_gas(gas_particles_per_dust_location: list[np.ndarray],
                     xyz_dust: np.ndarray,
                     xyz_gas: np.ndarray,
                     vxyz_gas: np.ndarray,
                     rho_gas: np.ndarray,
                     h_gas: np.ndarray,
                     kernel: BaseKernel,
                     gas_mass: float,
                     kwok: bool,
                     ndust: int,
                     ndim: int) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Interpolate gas density and velocity to location of dust particles.

    The neighbour list is a list of gas particles that contribute to each dust
    particle location. A reminder that this does not involve the dust
    particles. We are only interpolating the gas to the same position as the
    dust. We need the gas and dust properties at the same location to
    calculate the Stokes number.

    Parameters
    ----------
    gas_particles_per_dust_location : list of ndarray
        The "neighbour" list. It is a list with length the number of dust
        particles. Each element is an ndarray with the indices of gas particles
        that contribute to the location of that dust particle.
    xyz_dust : ndarray
        The x, y and z coordinates of the dust particles.
    xyz_gas : ndarray
        The x, y and z coordinates of the gas particles.
    vxyz_gas : ndarray
        The x, y and z velocities of the gas particles.
    rho_gas : ndarray
        The density of the gas particles.
    h_gas : ndarray
        The smoothing length of the gas particles.
    kernel : BaseKernel
        The smoothing kernel to use.
    gas_mass : float
        The gas particle mass.
    kwok: bool
        Whether to apply the Kwok (1975) correction for supersonic motions to
        the stopping time. If False, then the gas velocity is not interpolated.
    ndust : int
        The number of dust particles.
    ndim : int
        The number of dimensions.

    Returns
    -------
    ndarray
        The interpolated gas density at the location of each dust particle. Has
        length the number of dust particles.
    ndarray or None
        The interpolated x, y and z gas velocity at the location of each dust
        particle.If kwok is False, then the returned velocity is None as it is
        unneeded for the stopping time and Stokes number calculations.
    """

    rho_gas_on_dust = np.zeros(ndust)
    vxyz_gas_on_dust: np.ndarray | None = None

    if kwok:
        vxyz_gas_on_dust = np.zeros((ndust, len(xyz_gas[0])))

    # Interpolate the gas density and velocity at the dust particle locations
    for dust_idx, gas_particles in enumerate(gas_particles_per_dust_location):
        r_dust = xyz_dust[dust_idx]

        for gas_idx in gas_particles:
            q = np.linalg.norm(xyz_gas[gas_idx] - r_dust) / h_gas[gas_idx]
            weight = kernel.w(q, ndim) / h_gas[gas_idx]**ndim

            rho_gas_on_dust[dust_idx] += weight

            if kwok and vxyz_gas_on_dust is not None:
                weight = weight / rho_gas[gas_idx]
                vxyz_gas_on_dust[dust_idx] += vxyz_gas[gas_idx] * weight

    rho_gas_on_dust *= gas_mass

    if kwok and vxyz_gas_on_dust is not None:
        vxyz_gas_on_dust *= gas_mass

    return rho_gas_on_dust, vxyz_gas_on_dust


def _stoppingtime(rho_gas: np.ndarray,
                  rho_dust: np.ndarray,
                  vxyz_gas: np.ndarray | None,
                  vxyz_dust: np.ndarray,
                  c_s: float,
                  rho_grain: float,
                  grain_size: float,
                  gamma: float,
                  kwok: bool) -> np.ndarray:
    """
    Calculate the stopping time per particle.
    """

    coef = np.sqrt(np.pi * gamma * 0.125)

    # Kwok supersonic correction
    if kwok and vxyz_gas is not None:
        deltav_sq = np.linalg.norm(vxyz_gas - vxyz_dust, axis=1)**2
        f = np.sqrt(1 + 0.0703125 * np.pi * deltav_sq / c_s**2)
    else:
        f = 1.0

    return coef * rho_grain * grain_size / ((rho_dust + rho_gas) * c_s * f)


def stokes_number_2fluid(data_gas: 'SarracenDataFrame',
                         data_dust: 'SarracenDataFrame',
                         c_s: float,
                         rho_grain: float | None = None,
                         grain_size: float | None = None,
                         gamma: float | None = None,
                         kernel: BaseKernel | None = None,
                         kwok: bool = True) -> np.ndarray:
    """
    Calculate the Stokes number for each dust particle.

    This interpolates the gas density and velocity at the location of each dust
    particle to calculate the stopping time, and subsequently the Stokes
    number.

    Parameters
    ----------
    data_gas : SarracenDataFrame
        A SarracenDataFrame containing the gas particle data.
    data_dust : SarracenDataFrame
        A SarracenDataFrame containing the dust particle data.
    c_s : float
        The speed of sound. The same units as the particle data are used. This
        is taken to be a constant value assuming an isothermal equation of
        state. Support for other equations of state is to be added in future.
    rho_grain : float, optional
        The intrinsic density of the dust grains. The same units as the
        particle data are used. If not specified, then the value is retrieved
        from the dust SarracenDataFrame's params dict.
    grain_size : float, optional
        The size of the dust grains. The same units as the particle data are
        used. If not specified, then the value is retrieved from the dust
        SarracenDataFrame's params dict.
    gamma : float, optional
        The equation of state gamma. If not specified, then the value is
        retrieved from the gas SarracenDataFrame's params dict.
    kernel : BaseKernel, optional
        The smoothing kernel to use to interpolate the gas density and velocity
        to the location of dust particles. If not specified, then the smoothing
        kernel in the gas SarracenDataFrame is used.
    kwok : bool, optional
        If True, then the stopping time applies the Kwok (1975) correction for
        supersonic motions. This requires interpolation of the gas velocity to
        the location of the dust particles. Defaults to True.

    Returns
    -------
    ndarray
        A NumPy array of the Stokes number for each dust particle.

    Raises
    ------
    ValueError
        If the grain density or size are not specified and not found in params.
    ValueError
        If the thermodynamic gamma is not specified and not found in params.
    ValueError
        If particle mass cannot be found or particles have unequal masses.
    KeyError
        If the SarracenDataFrames are missing position, smoothing length, or
        velocity data.
    """
    if rho_grain is None:
        rho_grain = data_dust.params.get('graindens')
        if rho_grain is None:
            raise ValueError("Grain density not found in dust params.")

    if grain_size is None:
        grain_size = data_dust.params.get('grainsize')
        if grain_size is None:
            raise ValueError("Grain size not found in dust params.")

    if gamma is None:
        gamma = data_gas.params.get('gamma')
        if gamma is None:
            raise ValueError("Gamma not found in gas params.")

    if kernel is None:
        kernel = data_gas.kernel

    # Ensure all required columns exist in their respective dataframes
    _verify_columns(data_dust, kwok)
    _verify_columns(data_gas, kwok)

    # Getting specific dataframe columns as lists
    if data_dust.rhocol is None:
        data_dust.calc_density()
    rho_dust = data_dust[data_dust.rhocol].to_numpy()

    if data_gas.rhocol is None:
        data_gas.calc_density()
    rho_gas = data_gas[data_gas.rhocol].to_numpy()

    if data_gas.mcol is None:
        gas_mass = data_gas.params.get('mass')
        if gas_mass is None:
            raise ValueError("Gas mass not found.")
    else:
        mass = data_gas[data_gas.mcol].unique()
        if len(mass) != 1:
            raise ValueError("Unequal gas particle masses not supported.")
        gas_mass = float(mass[0])

    h_gas = data_gas[data_gas.hcol].to_numpy()

    # Get gas and dust coordinates, velocities
    xyz_gas = data_gas[[data_gas.xcol, data_gas.ycol,
                        data_gas.zcol]].to_numpy()
    xyz_dust = data_dust[[data_dust.xcol, data_dust.ycol,
                          data_dust.zcol]].to_numpy()

    vxyz_gas = data_gas[[data_gas.vxcol, data_gas.vycol,
                         data_gas.vzcol]].to_numpy()
    vxyz_dust = data_dust[[data_dust.vxcol, data_dust.vycol,
                           data_dust.vzcol]].to_numpy()

    # Make a neighbour list of dust particles per gas particle
    dust_neighbours = _get_dust_locations(xyz_gas, xyz_dust,
                                          h_gas, kernel.get_radius())

    # Invert the neighbour list.
    # Each element corresponds to a dust particle. It contains the list of gas
    # particles whose smoothing length touches this dust particle location.
    dust_neighbours_inv = _invert_dust_locations(xyz_dust, dust_neighbours)

    # Interpolate gas quantities to the location of the dust particles
    rho_gas_on_dust, vxyz_gas_on_dust = _interpolate_gas(dust_neighbours_inv,
                                                         xyz_dust,
                                                         xyz_gas,
                                                         vxyz_gas,
                                                         rho_gas,
                                                         h_gas,
                                                         kernel,
                                                         gas_mass,
                                                         kwok,
                                                         len(data_dust),
                                                         data_gas.get_dim())

    # Calculate stopping time for the now co-located gas/dust
    tstop = _stoppingtime(rho_gas_on_dust, rho_dust, vxyz_gas_on_dust,
                          vxyz_dust, c_s, rho_grain, grain_size, gamma, kwok)

    # Get co-located gas smoothing lengths from interpolated gas densities
    h = data_gas.params['hfact'] * (gas_mass / rho_gas_on_dust)**(1/3)
    tdyn = h / c_s

    # Stokes number is ratio of stopping time to dynamical time
    return tstop / tdyn
