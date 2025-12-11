import numpy as np
import pandas as pd

from ..sarracen_dataframe import SarracenDataFrame


def classify_bound_particles(sdf: SarracenDataFrame,
                             sdf_sinks: SarracenDataFrame) -> pd.Series:
    """
    Determines to which sink each SPH particle is gravitationally bound.

    Returns a pandas Series that classifies particles by the sink to which they
    are bound, or -1 if they are not bound to any sink.

    Parameters
    ----------
    sdf : SarracenDataFrame
        The particle data.
    sdf_sinks : SarracenDataFrame
        The sink particles.

    Returns
    -------
    Series
        A pandas Series with the sink classification for each particle.

    Raises
    ------
    ValueError
        If the particle SarracenDataFrame does not contain particle mass.

    Notes
    -----
    The index of the Series will match the index of the input
    SarracenDataFrame. The sinks are classified according to their index in the
    sdf_sinks.

    Examples
    --------
    Here is a set of SPH particles and sink particles. There are 4 sink
    particles.

    >>> sdf, sdf_sinks = sarracen.read_phantom('dumpfile')
    >>> len(sdf_sinks)
    4
    >>> sdf.head()
              x         y        z      vx      vy      vz       h
    0   24.0711   34.7009  -9.7994 -0.1229  0.0843  0.0012  1.2793
    1  -16.8095   39.8220  10.3250 -0.1356 -0.0585 -0.0010  1.3448
    2  -31.4485  143.1688  33.4715 -0.0747 -0.0158  0.0002  3.1289
    3   -2.0287 -149.1813 -35.7251  0.0749 -0.0012 -0.0011  3.2898
    4  -93.4897   85.5293   2.2384 -0.0559 -0.0629  0.0006  1.9655

    A pandas Series is returned that classifies which sink particle each SPH
    particle is bound to.

    >>> bound_to_sink = sarracen.ptmass.classify_sink(sdf, sdf_sinks)
    >>> sdf['sink'] = bound_to_sink
    >>> sdf.head()
               x         y       z      vx      vy      vz       h  sink
    0   24.0711   34.7009  -9.7994 -0.1229  0.0843  0.0012  1.2793     0
    1  -16.8095   39.8220  10.3250 -0.1356 -0.0585 -0.0010  1.3448     0
    2  -31.4485  143.1688  33.4715 -0.0747 -0.0158  0.0002  3.1289     3
    3   -2.0287 -149.1813 -35.7251  0.0749 -0.0012 -0.0011  3.2898     2
    4  -93.4897   85.5293   2.2384 -0.0559 -0.0629  0.0006  1.9655     0

    """
    # calculate the energy of particles relative to each sink

    if sdf_sinks.empty:
        raise ValueError("Sink particle DataFrame is empty.")

    if {sdf.mcol}.issubset(sdf.columns):
        mass = sdf[sdf.mcol]
    else:
        if sdf.params is None or 'mass' not in sdf.params:
            raise ValueError("Cannot find particle mass in "
                             "the SarracenDataFrame.")
        mass = sdf.params['mass']

    x = sdf[sdf.xcol].to_numpy(copy=False)
    y = sdf[sdf.ycol].to_numpy(copy=False)
    z = sdf[sdf.zcol].to_numpy(copy=False)
    vx = sdf[sdf.vxcol].to_numpy(copy=False)
    vy = sdf[sdf.vycol].to_numpy(copy=False)
    vz = sdf[sdf.vzcol].to_numpy(copy=False)

    energy = pd.Series(np.inf, index=sdf.index)
    bound_sink = pd.Series(-1, index=sdf.index, dtype=sdf_sinks.index.dtype)

    for sink_idx, sink_row in sdf_sinks.iterrows():

        x_sink = sink_row[sdf_sinks.xcol]
        y_sink = sink_row[sdf_sinks.ycol]
        z_sink = sink_row[sdf_sinks.zcol]
        vx_sink = sink_row[sdf_sinks.vxcol]
        vy_sink = sink_row[sdf_sinks.vycol]
        vz_sink = sink_row[sdf_sinks.vzcol]

        v_relsq = (np.square(vx - vx_sink)
                   + np.square(vy - vy_sink)
                   + np.square(vz - vz_sink))
        r = np.sqrt(np.square(x - x_sink)
                    + np.square(y - y_sink)
                    + np.square(z - z_sink))

        # sum of kinetic and potential energies
        E_total = 0.5 * mass * v_relsq - sink_row[sdf_sinks.mcol] * mass / r

        update = E_total < energy
        energy[update] = E_total[update]
        bound_sink[update] = sink_idx

    bound_sink[energy > 0.0] = -1

    return bound_sink
