import numpy as np
import pandas as pd
from ..sarracen_dataframe import SarracenDataFrame


def _get_mass(data: 'SarracenDataFrame'):
    if data.mcol == None:
        if 'mass' not in data.params:
            raise KeyError("'mass' column does not exist in this SarracenDataFrame.")
        return data.params['mass']

    return data[data.mcol]


def surface_density(data: 'SarracenDataFrame', r_in: float = None, r_out: float = None,
                    bins: int = 300, geometry: str = 'cylindrical', origin: list = None,
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

    if r_in is None:
        r_in = r.min()
    if r_out is None:
        r_out = r.max()

    bin_locations = np.linspace(r_in, r_out, bins+1)
    areas = np.pi * (bin_locations[1:] ** 2 - bin_locations[:-1] ** 2)
    rbins = pd.cut(r, bin_locations)

    mass = _get_mass(data)
    if isinstance(mass, pd.Series):
        sigma = mass.groupby(rbins).sum()
    else:
        sigma = data.groupby(rbins).count().iloc[:, 0] * mass

    if retbins:
        return (sigma / areas).to_numpy(), bin_locations
    else:
        return (sigma / areas).to_numpy()