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


def _get_origin(origin: list) -> list:
    if origin is None:
        return [0.0, 0.0, 0.0]
    else:
        return origin


def _bin_particles_by_radius(data: 'SarracenDataFrame',
                             r_in: float = None,
                             r_out: float = None,
                             bins: int = 300,
                             log: bool = False,
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
    log : bool, optional
        Whether to bin in log scale or not. Defaults to False.
    geometry : str, optional
        Coordinate system to use to calculate the particle radii. Can be
        either *spherical* or *cylindrical*. Defaults to *cylindrical*.
    origin : array-like, optional
        The x, y and z centre point around which to compute radii. Defaults to
        [0, 0, 0].

    Returns
    -------
    rbins: Series
        The radial bin to which each particle belongs.
    bin_edges: ndarray
        Locations of the bin edges.
    """

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

    if log:
        bin_edges = np.logspace(np.log10(r_in), np.log10(r_out), bins+1)
    else:
        bin_edges = np.linspace(r_in, r_out, bins+1)
    rbins = pd.cut(r, bin_edges)

    return rbins, bin_edges


def _get_bin_midpoints(bin_edges: np.ndarray,
                           log: bool = False) -> np.ndarray:
    """
    Calculate the midpoint of bins given their edges.
    bin_edges: ndarray
        Locations of the bin edges.
    log : bool, optional
        Whether to bin in log scale or not. Defaults to False.
    """

    if log:
        return np.sqrt(bin_edges[:-1] * bin_edges[1:])
    else:
        return 0.5 * (bin_edges[1:] - bin_edges[:-1]) + bin_edges[:-1]


def surface_density(data: 'SarracenDataFrame',
                    r_in: float = None,
                    r_out: float = None,
                    bins: int = 300,
                    log: bool = False,
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
    log : bool, optional
        Whether to bin in log scale or not. Defaults to False.
    geometry : str, optional
        Coordinate system to use to calculate the particle radii. Can be
        either *spherical* or *cylindrical*. Defaults to *cylindrical*.
    origin : array-like, optional
        The x, y and z centre point around which to compute radii. Defaults to
        [0, 0, 0].
    retbins : bool, optional
        Whether to return the midpoints of the bins or not. Defaults to False.

    Returns
    -------
    array
        A NumPy array of length bins containing the surface density profile.
    array, optional
        The midpoint values of each bin. Only returned if *retbins=True*.

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

    origin = _get_origin(origin)
    rbins, bin_edges = _bin_particles_by_radius(data, r_in, r_out, bins, log,
                                                geometry, origin)

    areas = np.pi * (bin_edges[1:] ** 2 - bin_edges[:-1] ** 2)

    mass = _get_mass(data)
    if isinstance(mass, pd.Series):
        sigma = mass.groupby(rbins).sum()
    else:
        sigma = data.groupby(rbins).count().iloc[:, 0] * mass

    if retbins:
        return (sigma / areas).to_numpy(), _get_bin_midpoints(bin_edges, log)
    else:
        return (sigma / areas).to_numpy()


def _calc_angular_momentum(data: 'SarracenDataFrame',
                           rbins: pd.Series,
                           origin: list,
                           unit_vector: bool):
    """
    Utility function to calculate angular momentum of the disc.

    Parameters
    ----------
    data: SarracenDataFrame
        Particle data, in a SarracenDataFrame.
    rbins: Series
        The radial bin to which each particle belongs.
    origin: list
        The x, y and z centre point around which to compute radii.
    unit_vector: bool
        Whether to convert the angular momentum to unit vectors.
        Default is True.

    Returns
    -------
    Lx, Ly, Lz: Series
        The x, y and z components of the angular momentum per bin.
    """

    mass = _get_mass(data)

    x_data = data[data.xcol].to_numpy() - origin[0]
    y_data = data[data.ycol].to_numpy() - origin[1]
    z_data = data[data.zcol].to_numpy() - origin[2]

    Lx = y_data * data[data.vzcol] - z_data * data[data.vycol]
    Ly = z_data * data[data.vxcol] - x_data * data[data.vzcol]
    Lz = x_data * data[data.vycol] - y_data * data[data.vxcol]

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


def angular_momentum(data: 'SarracenDataFrame',
                     r_in: float = None,
                     r_out: float = None,
                     bins: int = 300,
                     log: bool = False,
                     geometry: str = 'cylindrical',
                     origin: list = None,
                     retbins: bool = False,
                     unit_vector: bool = True):
    """
    Calculates the angular momentum profile of the disc.

    The profile is computed by segmenting the particles into radial bins
    (rings) and summing the angular momentum of the particles within each
    bin.

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
    log : bool, optional
        Whether to bin in log scale or not. Defaults to False.
    geometry : str, optional
        Coordinate system to use to calculate the particle radii. Can be
        either *spherical* or *cylindrical*. Defaults to *cylindrical*.
    origin : array-like, optional
        The x, y and z centre point around which to compute radii. Defaults to
        [0, 0, 0].
    retbins : bool, optional
        Whether to return the midpoints of the bins or not. Defaults to False.

    Returns
    -------
    array
        A NumPy array of length bins containing the angular momentum profile.
    array, optional
        The midpoint values of each bin. Only returned if *retbins=True*.

    Raises
    ------
    ValueError
        If the *geometry* is not *cylindrical* or *spherical*.
    """

    origin = _get_origin(origin)
    rbins, bin_edges = _bin_particles_by_radius(data, r_in, r_out, bins, log,
                                                geometry, origin)

    Lx, Ly, Lz = _calc_angular_momentum(data, rbins, origin, unit_vector)
    Lx, Ly, Lz = Lx.to_numpy(), Ly.to_numpy(), Lz.to_numpy()

    if retbins:
        return Lx, Ly, Lz, _get_bin_midpoints(bin_edges, log)
    else:
        return Lx, Ly, Lz


def _calc_scale_height(data: 'SarracenDataFrame',
                       rbins: pd.Series,
                       origin: list = None):
    """
    Utility function to calculate the scale height of the disc.

    Parameters
    ----------
    data: SarracenDataFrame
        Particle data, in a SarracenDataFrame.
    rbins: Series
        The radial bin to which each particle belongs.
    origin : array-like, optional
        The x, y and z centre point around which to compute radii. Defaults to
        [0, 0, 0].

    Returns
    -------
    H: Series
        The scale height of the disc.
    """
    Lx, Ly, Lz = _calc_angular_momentum(data, rbins, origin, unit_vector=True)

    zdash = rbins.map(Lx).to_numpy() * data[data.xcol] \
            + rbins.map(Ly).to_numpy() * data[data.ycol] \
            + rbins.map(Lz).to_numpy() * data[data.zcol]

    return zdash.groupby(rbins).std()


def scale_height(data: 'SarracenDataFrame',
                 r_in: float = None,
                 r_out: float = None,
                 bins: int = 300,
                 log: bool = False,
                 geometry: str = 'cylindrical',
                 origin: list = None,
                 retbins: bool = False):
    """
    Calculates the scale height, H/R, of the disc.

    The scale height, H/R, is computed by segmenting the particles into radial
    bins (rings) and calculating the angular momentum profile of the disc.
    Each particle takes the dot product of its position vector with the
    angular momentum vector of its corresponding bin. The standard deviation
    of this result per bin yields the scale height profile of the disc, which
    is divided by the midpoint radius of each bin.

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
    log : bool, optional
        Whether to bin in log scale or not. Defaults to False.
    geometry : str, optional
        Coordinate system to use to calculate the particle radii. Can be
        either *spherical* or *cylindrical*. Defaults to *cylindrical*.
    origin : array-like, optional
        The x, y and z centre point around which to compute radii. Defaults to
        [0, 0, 0].
    retbins : bool, optional
        Whether to return the midpoints of the bins or not. Defaults to False.

    Returns
    -------
    array
        A NumPy array of length bins scale height, H, profile.
    array, optional
        The midpoint values of each bin. Only returned if *retbins=True*.

    Raises
    ------
    ValueError
        If the *geometry* is not *cylindrical* or *spherical*.

    See Also
    --------
    :func:`angular_momentum` : Calculate the angular momentum profile of a
    disc.
    """

    origin = _get_origin(origin)
    rbins, bin_edges = _bin_particles_by_radius(data, r_in, r_out, bins, log,
                                                geometry, origin)

    midpoints = _get_bin_midpoints(bin_edges, log)
    H = _calc_scale_height(data, rbins, origin).to_numpy() / midpoints

    if retbins:
        return H, midpoints
    else:
        return H


def honH(data: 'SarracenDataFrame',
         r_in: float = None,
         r_out: float = None,
         bins: int = 300,
         log: bool = False,
         geometry: str = 'cylindrical',
         origin: list = None,
         retbins: bool = False):
    """
    Calculates <h>/H, the averaged smoothing length divided by the scale height.

    The profile is computed by segmenting the particles into radial bins
    (rings). The average smoothing length in each bin is divided by the scale
    height as calculated for that bin.

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
    log : bool, optional
        Whether to bin in log scale or not. Defaults to False.
    geometry : str, optional
        Coordinate system to use to calculate the particle radii. Can be
        either *spherical* or *cylindrical*. Defaults to *cylindrical*.
    origin : array-like, optional
        The x, y and z centre point around which to compute radii. Defaults to
        [0, 0, 0].
    retbins : bool, optional
        Whether to return the midpoints of the bins or not. Defaults to False.

    Returns
    -------
    array
        A NumPy array of length bins containing the <h>/H profile.
    array, optional
        The midpoint values of each bin. Only returned if *retbins=True*.

    Raises
    ------
    ValueError
        If the *geometry* is not *cylindrical* or *spherical*.

    See Also
    --------
    :func:`scale_height` : Calculate the scale height of a disc.
    """

    origin = _get_origin(origin)
    rbins, bin_edges = _bin_particles_by_radius(data, r_in, r_out, bins, log,
                                                geometry, origin)

    H = _calc_scale_height(data, rbins, origin).to_numpy()

    mean_h = data.groupby(rbins)[data.hcol].mean().to_numpy()

    if retbins:
        return mean_h / H, _get_bin_midpoints(bin_edges, log)
    else:
        return mean_h / H
