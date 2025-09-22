from typing import Union, List

import pandas as pd

from ..sarracen_dataframe import SarracenDataFrame


def read_gradsph(filename: str,
                 separate_types: str = 'sinks') -> Union[List[
                                                         SarracenDataFrame],
                                                         SarracenDataFrame]:
    """
    Read data from a GradSPH dump file.

    Global values stored in the dump file are stored within the data frame in
    the dictionary ``params``.

    Parameters
    ----------
    filename : str
        Name of the file to be loaded.
    separate_types : {None, 'sinks', 'all'}, default='sinks'
        Whether to separate SPH particles and sink particles into separate
        SarracenDataFrames. ``None`` returns all particle types in one
        SarracenDataFrame. '`sinks`' and '`all`' separate sink particles into
        a second SarracenDataFrame.

    Returns
    -------
    SarracenDataFrame or list of SarracenDataFrame

    Examples
    --------
    By default, SPH particles are grouped into one SarracenDataFrame and sink
    particles into a second SarracenDataFrame.

    >>> sdf, sdf_sinks = sarracen.read_gradsph('col3139')
    """
    with open(filename, 'r') as fp:

        n_str, ninactive_str, nsink_str = fp.readline().split()
        n, ninactive, nsink = int(n_str), int(ninactive_str), int(nsink_str)

        t_str, gamma_str = fp.readline().split()
        t, gamma = float(t_str), float(gamma_str)

        params = {'n': n,
                  'ninactive': ninactive,
                  'nsink': nsink,
                  't': t,
                  'gamma': gamma}

        sinks = [fp.readline().split() for _ in range(nsink)]
        parts = [fp.readline().split() for _ in range(n - ninactive)]

        sink_header = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'mass']
        part_header = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'mass', 'h', 'cs',
                       'rho', 'temp']

        df_sinks = pd.DataFrame(sinks, columns=sink_header, dtype=float)
        df_parts = pd.DataFrame(parts, columns=part_header, dtype=float)

        if separate_types == 'sinks' or separate_types == 'all':
            df_list = [SarracenDataFrame(df_parts, params=params),
                       SarracenDataFrame(df_sinks, params=params)]
        else:
            df_list = [SarracenDataFrame(pd.concat([df_parts, df_sinks],
                                                   ignore_index=True),
                                         params=params)]

        return df_list[0] if len(df_list) == 1 else df_list
