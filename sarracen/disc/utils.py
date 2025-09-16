import numpy as np
import pandas as pd
import sys
from ..sarracen_dataframe import SarracenDataFrame
from typing import Tuple, Union


def _get_mass(data: 'SarracenDataFrame') -> Union[float, pd.Series]:
    if data.mcol is None:
        if 'mass' not in data.params:
            raise KeyError("'mass' column does not exist in this "
                           "SarracenDataFrame.")
        return data.params['mass']

    return data[data.mcol]


def _get_origin(origin: Union[list, None]) -> list:
    if origin is None:
        return [0.0, 0.0, 0.0]
    else:
        return origin


def _bin_particles_by_radius(data: 'SarracenDataFrame',
                             r_in: Union[float, None],
                             r_out: Union[float, None],
                             bins: int,
                             log: bool,
                             geometry: str,
                             origin: list) -> Tuple[pd.Series, np.ndarray]:
    """
    Utility function to bin particles in discrete intervals by radius.

    Parameters
    ----------
    data: SarracenDataFrame
        The particle dataset.
    r_in : float
        Inner radius of the disc. Defaults to the minimum r value.
    r_out : float
        Outer radius of the disc. Defaults to the maximum r value.
    bins : int
        Defines the number of equal-width bins in the range [r_in, r_out].
        Default is 300.
    log : bool
        Whether to bin in log scale or not. Defaults to False.
    geometry : str
        Coordinate system to use to calculate the particle radii. Can be
        either *spherical* or *cylindrical*. Defaults to *cylindrical*.
    origin : array-like
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
        raise ValueError("geometry should be either 'cylindrical' or "
                         "'spherical'")

    # should we add epsilon here?
    if r_in is None:
        r_in = r.min() - sys.float_info.epsilon
    if r_out is None:
        r_out = r.max() + sys.float_info.epsilon

    if log:
        bin_edges = np.logspace(np.log10(r_in), np.log10(r_out), bins+1)
    else:
        bin_edges = np.linspace(r_in, r_out, bins+1)
    rbins = pd.cut(r, pd.Series(bin_edges))

    return rbins, bin_edges


def _get_bin_midpoints(bin_edges: np.ndarray,
                       log: bool = False) -> np.ndarray:
    """
    Calculate the midpoint of bins given their edges.

    Parameters
    ----------
    bin_edges: ndarray
        Locations of the bin edges.
    log : bool, optional
        Whether to bin in log scale or not. Defaults to False.
    """

    if log:
        return np.sqrt(bin_edges[:-1] * bin_edges[1:])
    else:
        return 0.5 * (bin_edges[1:] - bin_edges[:-1]) + bin_edges[:-1]
