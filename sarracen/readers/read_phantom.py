from typing import IO, Tuple, Type, Union, List, overload, Literal

import numpy as np
import pandas as pd

from ..sarracen_dataframe import SarracenDataFrame


def _read_fortran_block(fp: IO, bytesize: int) -> bytes:
    """ Helper function to read Fortran-written data.

    Fortran will add a 4-byte tag before and after any data writes. The value
    of this tag is equal to the number of bytes written. In our case, we do a
    simple sanity check that the start and end tag are consistent, but not
    validate the value of the tag with the size of the data read.
    """
    start_tag = fp.read(4)
    data = fp.read(bytesize)
    end_tag = fp.read(4)

    if (start_tag != end_tag):
        raise AssertionError("Fortran tags mismatch.")

    return data


def _read_capture_pattern(fp: IO) -> Tuple[Type[np.generic],
                                           Type[np.generic],
                                           int, bool]:
    """ Phantom dump validation plus default real and int sizes."""

    start_tag = fp.read(4)  # 4-byte Fortran tag

    def_types: List[Tuple[Type[np.generic],
                          Type[np.generic]]] = [(np.int32, np.float64),
                                                (np.int32, np.float32),
                                                (np.int64, np.float64),
                                                (np.int64, np.float32)]

    i1 = r1 = i2 = 0
    def_int_dtype, def_real_dtype = def_types[0]

    swap_endian = False
    for def_int_dtype, def_real_dtype in def_types:
        i1 = fp.read(def_int_dtype().itemsize)
        r1 = fp.read(def_real_dtype().itemsize)
        i2 = fp.read(def_int_dtype().itemsize)

        i1 = np.frombuffer(i1, count=1, dtype=def_int_dtype)[0]
        r1 = np.frombuffer(r1, count=1, dtype=def_real_dtype)[0]
        i2 = np.frombuffer(i2, count=1, dtype=def_int_dtype)[0]

        if (i1 == def_int_dtype(60769)
                and i2 == def_int_dtype(60878)
                and r1 == def_real_dtype(i2)):
            break
        if (i1.byteswap() == def_int_dtype(60769)
                and i2.byteswap() == def_int_dtype(60878)
                and r1.byteswap() == def_real_dtype(i2.byteswap())):
            swap_endian = True
            i1 = i1.byteswap()
            r1 = r1.byteswap()
            i2 = i2.byteswap()
            break
        else:  # rewind and try again
            fp.seek(-def_int_dtype().itemsize, 1)
            fp.seek(-def_real_dtype().itemsize, 1)
            fp.seek(-def_int_dtype().itemsize, 1)

    if (i1 != def_int_dtype(60769)
            or i2 != def_int_dtype(60878)
            or r1 != def_real_dtype(i2)):
        raise AssertionError("Could not determine default int or float "
                             "precision (i1, r1, i2 mismatch). "
                             "Is this a Phantom data file?")

    # iversion -- we don't actually check this
    iversion = fp.read(def_int_dtype().itemsize)
    iversion = np.frombuffer(iversion, count=1, dtype=def_int_dtype)[0]
    if swap_endian:
        iversion = iversion.byteswap()

    # integer 3 == 690706
    i3 = fp.read(def_int_dtype().itemsize)
    i3 = np.frombuffer(i3, count=1, dtype=def_int_dtype)[0]
    if swap_endian:
        i3 = i3.byteswap()
    if i3 != def_int_dtype(690706):
        raise AssertionError("Capture pattern error. i3 mismatch. "
                             "Is this a Phantom data file?")

    end_tag = fp.read(4)  # 4-byte Fortran tag

    # assert tags equal
    if (start_tag != end_tag):
        raise AssertionError("Capture pattern error. Fortran tags mismatch. "
                             "Is this a Phantom data file?")

    return def_int_dtype, def_real_dtype, iversion, swap_endian


def _read_file_identifier(fp: IO) -> str:
    """ Read the 100 character file identifier.

    The file identifier contains code version and date information.
    """
    return _read_fortran_block(fp, 100).decode('ascii').strip()


def _rename_duplicates(keys: list) -> list:
    seen = dict()

    for i, key in enumerate(keys):
        if key not in seen:
            seen[key] = 1
        else:
            seen[key] += 1
            keys[i] += f'_{seen[key]}'

    return keys


def _read_global_header_block(fp: IO,
                              dtype: Type[np.generic],
                              swap_endian: bool) -> Tuple[list, list]:
    nvars = np.frombuffer(_read_fortran_block(fp, 4), dtype=np.int32)[0]
    if swap_endian:
        nvars = nvars.byteswap()

    keys = []
    data = []

    if (nvars > 0):
        # each tag is 16 characters in length
        keys_str = _read_fortran_block(fp, 16*nvars).decode('ascii')
        keys = [keys_str[i:i+16].strip() for i in range(0, len(keys_str), 16)]

        raw_data = _read_fortran_block(fp, dtype().itemsize*nvars)
        data_np = np.frombuffer(raw_data, count=nvars, dtype=dtype)
        if swap_endian:
            data_np = data_np.byteswap()
        data = list(data_np)

    return keys, data


def _read_global_header(fp: IO,
                        def_int_dtype: Type[np.generic],
                        def_real_dtype: Type[np.generic],
                        swap_endian: bool) -> dict:
    """ Read global variables. """

    dtypes = [def_int_dtype, np.int8, np.int16, np.int32, np.int64,
              def_real_dtype, np.float32, np.float64]

    keys = []
    data = []
    for dtype in dtypes:
        new_keys, new_data = _read_global_header_block(fp, dtype, swap_endian)

        keys += new_keys
        data += new_data

    keys = _rename_duplicates(keys)

    global_vars = dict()
    for i in range(len(keys)):
        global_vars[keys[i]] = data[i]

    return global_vars


def _read_array_block(fp: IO,
                      df: pd.DataFrame,
                      n: int,
                      nums: np.ndarray,
                      def_int_dtype: Type[np.generic],
                      def_real_dtype: Type[np.generic],
                      swap_endian: bool) -> pd.DataFrame:

    dtypes = [def_int_dtype, np.int8, np.int16, np.int32, np.int64,
              def_real_dtype, np.float32, np.float64]

    for i in range(len(nums)):
        dtype = dtypes[i]
        for j in range(nums[i]):

            tag = _read_fortran_block(fp, 16).decode('ascii').strip()

            if tag in df.columns:
                count = 1
                original_tag = tag
                while tag in df.columns:
                    count += 1
                    tag = original_tag + f"_{count}"

            raw_data = _read_fortran_block(fp, dtype().itemsize * n)
            data: np.ndarray = np.frombuffer(raw_data, dtype=dtype)
            if swap_endian:
                data = data.byteswap()
            df[tag] = data

    return df


def _read_array_blocks(fp: IO,
                       def_int_dtype: Type[np.generic],
                       def_real_dtype: Type[np.generic],
                       mpi_blocks: int,
                       swap_endian: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Read particle data.

    Block 2 is always for sink particles. The number of MPI blocks is given by
    'nblocks' in the global header. The first quantity read in this function is
    the total number of blocks."""

    nblocks = np.frombuffer(_read_fortran_block(fp, 4), dtype=np.int32)[0]
    if swap_endian:
        nblocks = nblocks.byteswap()

    df = pd.DataFrame()
    df_sinks = pd.DataFrame()

    nblocks = int(nblocks / mpi_blocks)  # number of blocks per MPI process

    for j in range(0, mpi_blocks):

        n: List[int] = []
        nums: List[np.ndarray] = []

        for i in range(0, nblocks):
            start_tag = fp.read(4)

            n_val = np.frombuffer(fp.read(8), dtype=np.int64)[0]
            nums_val = np.frombuffer(fp.read(32), count=8, dtype=np.int32)
            if swap_endian:
                n_val = n_val.byteswap()
                nums_val = nums_val.byteswap()
            n.append(n_val)
            nums.append(nums_val)

            end_tag = fp.read(4)
            if (start_tag != end_tag):
                raise AssertionError("Fortran tags mismatch in array blocks.")

        df_tmp = pd.DataFrame()
        df_tmp_sinks = pd.DataFrame()

        for i in range(0, nblocks):
            # This assumes the second block is only for sink particles.
            # This is a valid assumption as this is what splash assumes.
            # For now we will just append sinks to the end of the data frame.

            # Can we avoid temporary df?
            if i == 1:
                # Not sure why, but it seems each MPI block repeats sinks
                df_tmp_sinks = _read_array_block(fp, df_tmp_sinks, n[i],
                                                 nums[i], def_int_dtype,
                                                 def_real_dtype, swap_endian)
            else:
                df_tmp = _read_array_block(fp, df_tmp, n[i], nums[i],
                                           def_int_dtype, def_real_dtype,
                                           swap_endian)

        df = pd.concat([df, df_tmp])
        df_sinks = pd.concat([df_sinks, df_tmp_sinks]).drop_duplicates()

    return df, df_sinks


def _create_mass_column(df: pd.DataFrame,
                        header_vars: dict) -> pd.DataFrame:
    """
    Creates a mass column with the mass of each particle when there are
    multiple itypes.
    """
    df['mass'] = header_vars['massoftype']
    for itype in df['itype'].unique():
        if itype > 1:
            mass = header_vars[f'massoftype_{itype}']
            df.loc[df.itype == itype, 'mass'] = mass
    return df


def _create_aprmass_column(df: pd.DataFrame,
                           header_vars: dict) -> pd.DataFrame:
    """
    Creates a mass column with the mass of each particle when there are
    multiple refinement levels.
    """
    df['mass'] = header_vars['massoftype']
    df['mass'] = df['mass']/(2**(df['apr_level'] - 1))

    return df


@overload
def read_phantom(filename: str,
                 separate_types: None,
                 ignore_inactive: bool = True) -> SarracenDataFrame: ...
@overload  # noqa: E302
def read_phantom(filename: str,
                 separate_types: Literal['sinks'] = 'sinks',
                 ignore_inactive: bool = True) -> Union[List[
                                                        SarracenDataFrame],
                                                        SarracenDataFrame]: ...
@overload  # noqa: E302
def read_phantom(filename: str,
                 separate_types: Literal['all'] = 'all',
                 ignore_inactive: bool = True) -> Union[List[
                                                        SarracenDataFrame],
                                                        SarracenDataFrame]: ...
def read_phantom(filename: str,  # noqa: E302
                 separate_types: Union[str, None] = 'sinks',
                 ignore_inactive: bool = True) -> Union[List[
                                                        SarracenDataFrame],
                                                        SarracenDataFrame]:
    """
    Read data from a Phantom dump file.

    This reads the native binary format of Phantom dump files, which in turn
    were derived from the binary file format used by sphNG.

    Global values stored in the dump file (time step, initial momentum, hfact,
    Courant factor, etc) are stored within the data frame in the dictionary
    ``params``.

    Parameters
    ----------
    filename : str
        Name of the file to be loaded.
    separate_types : {None, 'sinks', 'all'}, default='sinks'
        Whether to separate different particle types into several dataframes.
        ``None`` returns all particle types in one data frame. '`sinks`'
        separates sink particles into a second dataframe, and '`all`' returns
        all particle types in different dataframes.
    ignore_inactive : {True, False}, default=True
        If True, particles with negative smoothing length will not be read on
        import. These are typically particles that have been accreted onto a
        sink particle or are otherwise inactive.

    Returns
    -------
    SarracenDataFrame or list of SarracenDataFrame

    See Also
    --------
    :func:`SarracenDataFrame` : A pandas DataFrame with support for SPH data.

    Notes
    -----
    See the `Phantom documentation
    <https://phantomsph.readthedocs.io/en/latest/dumpfile.html>`_ for a full
    description of the Phantom binary file format.

    Examples
    --------
    By default, SPH particles are grouped into one data frame and sink
    particles into a second data frame.

    >>> import sarracen
    >>> sdf, sdf_sinks = sarracen.read_phantom('dumpfile_00000')

    A dump file containing multiple particle types, say gas + dust + sinks,
    can be separated into their own data frames by specifying
    ``separate_types='all'``.

    >>> sdf_g, sdf_d, sdf_sinks = sarracen.read_phantom('dumpfile_00000',
    ...                                                 separate_types='all')

    Global values are stored in the ``params`` dictionary.

    >>> sdf_g.params
    {'nparttot': np.int32(100000),
     'ntypes': np.int32(8),
     'npartoftype': np.int32(100000),
     'massoftype': np.float64(5.05e-7)
     'time': np.float64(0.05),
     'grainsize': np.float64(6.684491978609626e-14),
     'graindens': np.float64(5049628.378663718),
     'udist': np.float64(14960000000000.0)
     'umass': np.float64(1.9891e+33),
     'utime': np.float64(5022728.790082334),
     ...
    }

    """
    with open(filename, 'rb') as fp:
        def_int_dtype, def_real_dtype, iversion, swap_endian = \
            _read_capture_pattern(fp)
        file_identifier = _read_file_identifier(fp)

        header_vars = _read_global_header(fp, def_int_dtype, def_real_dtype,
                                          swap_endian)
        header_vars['file_identifier'] = file_identifier
        header_vars['iversion'] = iversion
        header_vars['def_int_dtype'] = def_int_dtype
        header_vars['def_real_dtype'] = def_real_dtype

        mpi_blocks = header_vars['nblocks'] if 'nblocks' in header_vars else 1

        df, df_sinks = _read_array_blocks(fp, def_int_dtype, def_real_dtype,
                                          mpi_blocks, swap_endian)

        if ignore_inactive and 'h' in df.columns:
            df = df[df['h'] > 0]

        # create mass column if multiple species in single dataframe
        if (separate_types != 'all'
                and 'itype' in df
                and df['itype'].nunique() > 1):
            df = _create_mass_column(df, header_vars)
        # create a column if APR is used and automatically scale masses
        elif 'apr_level' in df:
            df = _create_aprmass_column(df, header_vars)
        else:  # create global mass parameter
            header_vars['mass'] = header_vars['massoftype']

        df_list = []
        if separate_types == 'all':
            if 'itype' in df and df['itype'].nunique() > 1:
                for _, group in df.groupby('itype'):
                    itype = int(group["itype"].iloc[0])
                    mass_key = 'massoftype' if itype == 1 \
                        else f'massoftype_{itype}'
                    params = {**header_vars, **{"mass": header_vars[mass_key]}}
                    df_list.append(SarracenDataFrame(group.dropna(axis=1),
                                                     params=params))
            else:
                df_list = [SarracenDataFrame(df, params=header_vars)]

            if not df_sinks.empty:
                params = {key: value for key, value in header_vars.items()
                          if key != 'mass'}
                df_list.append(SarracenDataFrame(df_sinks,
                                                 params=params))

        elif separate_types == 'sinks':
            df_list = [SarracenDataFrame(df, params=header_vars)]
            if not df_sinks.empty:
                params = {key: value for key, value in header_vars.items()
                          if key != 'mass'}
                df_list.append(SarracenDataFrame(df_sinks,
                                                 params=params))
        else:
            df_list = [SarracenDataFrame(pd.concat([df, df_sinks],
                                                   ignore_index=True),
                                         params=header_vars)]

        return df_list[0] if len(df_list) == 1 else df_list
