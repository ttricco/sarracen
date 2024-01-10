import numpy as np
import pandas as pd
import sys
from ..sarracen_dataframe import SarracenDataFrame


def _get_mass(data: 'SarracenDataFrame'):
    if data.mcol == None:
        if 'mass' not in data.params:
            raise KeyError("'mass' column does not exist in this SarracenDataFrame.")
        return data.params['mass']

    return data[data.mcol]


def _bin_particles_by_radius(data: 'SarracenDataFrame',
                             r_in: float = None,
                             r_out: float = None,
                             bins: int = 300,
                             geometry: str = 'cylindrical',
                             origin: list = None):
    """
    Utility function to bin particles in discrete intervals by radius.

    Parameters
    ----------
    data: SarracenDataFrame
        The particle dataset.
    r_in : float, optional
        Inner radius of the disc. Defaults to the minimum r value.
    r_out : float, optional
        Outer radius of the disc. Defaults to the maximum r value.
    bins : int, optional
        Defines the number of equal-width bins in the range [r_in, r_out].
        Default is 300.
    geometry : str, optional
        Coordinate system to use to calculate the particle radii. Can be
        either *spherical* or *cylindrical*. Defaults to *cylindrical*.
    origin : array-like, optional
        The x, y and z position around which to compute radii. Defaults to
        [0, 0, 0].

    Returns
    -------
    rbins: Series
        The radial bin to which each particle belongs.
    bin_locations: ndarray
        Locations of the bin edges.
    """

    if origin is None:
        origin = [0, 0, 0]

    if geometry == 'spherical':
        r = np.sqrt((data[data.xcol] - origin[0]) ** 2
                    + (data[data.ycol] - origin[1]) ** 2
                    + (data[data.zcol] - origin[2]) ** 2)
    elif geometry == 'cylindrical':
        r = np.sqrt((data[data.xcol] - origin[0]) ** 2
                    + (data[data.ycol] - origin[1]) ** 2)
    else:
        raise ValueError("geometry should be either 'cylindrical' or 'spherical'")

    # should we add epsilon here?
    if r_in is None:
        r_in = r.min() - sys.float_info.epsilon
    if r_out is None:
        r_out = r.max() + sys.float_info.epsilon

    bin_locations = np.linspace(r_in, r_out, bins+1)
    rbins = pd.cut(r, bin_locations)

    return rbins, bin_locations


def surface_density(data: 'SarracenDataFrame',
                    r_in: float = None,
                    r_out: float = None,
                    bins: int = 300,
                    geometry: str = 'cylindrical',
                    origin: list = None,
                    retbins: bool = False):
    """
    Calculates the 1D azimuthally-averaged surface density profile.

    The surface density profile is computed by segmenting the particles into
    radial bins (rings) and dividing the total mass contained within each bin
    by the area of its respective ring.

    Parameters
    ----------
    data : SarracenDataFrame
        Particle data, in a SarracenDataFrame.
    r_in : float, optional
        Inner radius of the disc. Defaults to the minimum r value.
    r_out : float, optional
        Outer radius of the disc. Defaults to the maximum r value.
    bins : int, optional
        Defines the number of equal-width bins in the range [r_in, r_out].
        Default is 300.
    geometry : str, optional
        Coordinate system to use to calculate the particle radii. Can be
        either *spherical* or *cylindrical*. Defaults to *cylindrical*.
    origin : array-like, optional
        The x, y and z position around which to compute radii. Defaults to
        [0, 0, 0].
    retbins : bool, optional
        Whether to return the bin edges or not. Defaults to False.

    Returns
    -------
    array
        A NumPy array of length bins containing the surface density profile.
    array, optional
        The location of the bin edges. Only returned if *retbins=True*.

    Raises
    ------
    ValueError
        If the *geometry* is not *cylindrical* or *spherical*.

    See Also
    --------
    The surface density averaging procedure for SPH is described in section
    3.2.6 of Lodato & Price, MNRAS (2010), `doi:10.1111/j.1365-2966.2010.16526.x
    <https://doi.org/10.1111/j.1365-2966.2010.16526.x>`_.
    """

    rbins, bin_locations = _bin_particles_by_radius(data, r_in, r_out, bins,
                                                    geometry, origin)

    areas = np.pi * (bin_locations[1:] ** 2 - bin_locations[:-1] ** 2)

    mass = _get_mass(data)
    if isinstance(mass, pd.Series):
        sigma = mass.groupby(rbins).sum()
    else:
        sigma = data.groupby(rbins).count().iloc[:, 0] * mass

    if retbins:
        return (sigma / areas).to_numpy(), bin_locations
    else:
        return (sigma / areas).to_numpy()


def _calc_angular_momenta(data: 'SarracenDataFrame',
                          rbins: pd.Series,
                          bin_locations: np.ndarray,
                          unit_vector: bool):
    """
    Utility function to calculate angular momenta of the disc.

    Parameters
    ----------
    data: SarracenDataFrame
        Particle data, in a SarracenDataFrame.
    rbins: Series
        The radial bin to which each particle belongs.
    bin_locations: ndarray
        Locations of the bin edges.
    unit_vector: bool
        Whether to convert the angular momenta to unit vectors.
        Default is True.

    Returns
    -------
    Lx, Ly, Lz: Series
        The x, y and z components of the angular momentum per bin.
    """

    mass = _get_mass(data)

    Lx = data[data.ycol] * data[data.vzcol] - data[data.zcol] * data[data.vycol]
    Ly = data[data.zcol] * data[data.vxcol] - data[data.xcol] * data[data.vzcol]
    Lz = data[data.xcol] * data[data.vycol] - data[data.ycol] * data[data.vxcol]

    if isinstance(mass, float):
        Lx = (mass * Lx).groupby(rbins).sum()
        Ly = (mass * Ly).groupby(rbins).sum()
        Lz = (mass * Lz).groupby(rbins).sum()
    else:
        Lx = (data[data.mcol] * Lx).groupby(rbins).sum()
        Ly = (data[data.mcol] * Ly).groupby(rbins).sum()
        Lz = (data[data.mcol] * Lz).groupby(rbins).sum()

    if unit_vector:
        Lmag = 1.0 / np.sqrt(Lx ** 2 + Ly ** 2 + Lz ** 2)

        Lx = Lx * Lmag
        Ly = Ly * Lmag
        Lz = Lz * Lmag

    return Lx, Ly, Lz


def angular_momenta(data: 'SarracenDataFrame',
                    r_in: float = None,
                    r_out: float = None,
                    bins: int = 300,
                    geometry: str = 'cylindrical',
                    origin: list = None,
                    retbins: bool = False,
                    unit_vector: bool = True):

    rbins, bin_locations = _bin_particles_by_radius(data, r_in, r_out, bins,
                                                    geometry, origin)

    Lx, Ly, Lz = _calc_angular_momenta(data, rbins, bin_locations, unit_vector)
    Lx, Ly, Lz = Lx.to_numpy(), Ly.to_numpy(), Lz.to_numpy()

    if retbins:
        return Lx, Ly, Lz, bin_locations
    else:
        return Lx, Ly, Lz


def scale_height(data: 'SarracenDataFrame',
                 r_in: float = None,
                 r_out: float = None,
                 bins: int = 300,
                 geometry: str = 'cylindrical',
                 origin: list = None,
                 retbins: bool = False):
    rbins, bin_locations = _bin_particles_by_radius(data, r_in, r_out, bins,
                                                    geometry, origin)

    Lx, Ly, Lz = _calc_angular_momenta(data, rbins, bin_locations,
                                       unit_vector=True)

    zdash = rbins.map(Lx).to_numpy() * data[data.xcol] \
            + rbins.map(Ly).to_numpy() * data[data.ycol] \
            + rbins.map(Lz).to_numpy() * data[data.zcol]

    return zdash.groupby(rbins).std().to_numpy()
