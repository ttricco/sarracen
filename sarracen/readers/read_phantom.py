from typing import Union

import numpy as np
import pandas as pd

from ..sarracen_dataframe import SarracenDataFrame

def _read_fortran_block(fp, bytesize):
    """ Helper function to read Fortran data, which is also buffered before and after by 4 bytes."""
    start_tag = fp.read(4)
    data = fp.read(bytesize)
    end_tag = fp.read(4)

    if (start_tag != end_tag):
        raise AssertionError("Fortran tags mismatch.")

    return data


def _read_capture_pattern(fp):
    """ Phantom dump validation plus default real and int sizes."""
    # 4 byte Fortran tag
    start_tag = fp.read(4)

    def_int_dtype = np.int32
    def_real_dtype = np.float32


    # integer 1 == 060769
    i1 = np.frombuffer(fp.read(def_int_dtype().itemsize), count=1, dtype=def_int_dtype)[0]

    # assert i1. Try 8-byte int if fails.
    if (i1 != 60769):
        fp.seek(-8, 1) # rewind based on current file position

        def_int_dtype = np.int64

        i1 = np.frombuffer(fp.read(def_int_dtype), count=1, dtype=def_int_dtype)[0]

        # retry assert
        if (i1 != 60769):
            raise AssertionError("Capture pattern error. i1 mismatch. Is this a Phantom data file?")



    # real 1 == integer 2 == 060878
    r1 = np.frombuffer(fp.read(def_real_dtype().itemsize), count=1, dtype=def_real_dtype)[0]
    i2 = np.frombuffer(fp.read(def_int_dtype().itemsize), count=1, dtype=def_int_dtype)[0]

    # assert r1 and i2. Try 8-byte real if fails.
    if (i2 != 60878 or not np.float32(i2) == r1):
        fp.seek(-8, 1) # rewind

        def_real_dtype = np.float64

        r1 = np.frombuffer(fp.read(def_real_dtype().itemsize), count=1, dtype=def_real_dtype)[0]
        i2 = np.frombuffer(fp.read(def_int_dtype().itemsize), count=1, dtype=def_int_dtype)[0]

        # retry assert
        if (i2 != 60878 or not np.float64(i2) == r1):
            raise AssertionError("Capture pattern error. i2 and r1 mismatch. Is this a Phantom data file?")


    # iversion -- we don't actually check this
    iversion = np.frombuffer(fp.read(def_int_dtype().itemsize), count=1, dtype=def_int_dtype)[0]


    # integer 3 == 690706
    i3 = np.frombuffer(fp.read(def_int_dtype().itemsize), count=1, dtype=def_int_dtype)[0]
    if (i3 != 690706):
        raise AssertionError("Capture pattern error. i3 mismatch. Is this a Phantom data file?")


    # 4 byte Fortran tag
    end_tag = fp.read(4)

    # assert tags equal
    if (start_tag != end_tag):
        raise AssertionError("Capture pattern error. Fortran tags mismatch. Is this a Phantom data file?")

    return def_int_dtype, def_real_dtype



def _read_file_identifier(fp):
    """ Read the 100 character file identifier. Contains code version and date information. """
    return _read_fortran_block(fp, 100).decode('ascii').strip()



def _rename_duplicates(keys):
    seen = dict()

    for i, key in enumerate(keys):
        if key not in seen:
            seen[key] = 1
        else:
            seen[key] += 1
            keys[i] += f'_{seen[key]}'

    return keys


def _read_global_header_block(fp, dtype):
    nvars = np.frombuffer(_read_fortran_block(fp, 4), dtype=np.int32)[0]

    keys = []
    data = []

    if (nvars > 0):
        # each tag is 16 characters in length
        keys = _read_fortran_block(fp, 16*nvars).decode('ascii')
        keys = [keys[i:i+16].strip() for i in range(0, len(keys), 16)]

        data = _read_fortran_block(fp, dtype().itemsize*nvars)
        data = np.frombuffer(data, count=nvars, dtype=dtype)

    return keys, data


def _read_global_header(fp, def_int_dtype, def_real_dtype):
    """ Read global variables. """

    dtypes = [def_int_dtype, np.int8, np.int16, np.int32, np.int64,
                    def_real_dtype, np.float32, np.float64]

    keys = []
    data = []
    for dtype in dtypes:
        new_keys, new_data = _read_global_header_block(fp, dtype)

        keys += new_keys
        data = np.append(data, new_data)

    keys = _rename_duplicates(keys)

    global_vars = dict()
    for i in range(len(keys)):
        global_vars[keys[i]] = data[i]

    return global_vars


def _read_array_block(fp, df, n, nums, def_int_dtype, def_real_dtype):

    dtypes = [def_int_dtype, np.int8, np.int16, np.int32, np.int64,
              def_real_dtype, np.float32, np.float64]

    for i in range(len(nums)):
        dtype = dtypes[i]
        for j in range(nums[i]):

            tag = _read_fortran_block(fp, 16).decode('ascii').strip()
            data = np.frombuffer(_read_fortran_block(fp, dtype().itemsize * n), dtype=dtype)
            df[tag] = data

    return df

def _read_array_blocks(fp, def_int_dtype, def_real_dtype):
    """ Read particle data. Block 2 is always for sink particles?"""
    nblocks = np.frombuffer(_read_fortran_block(fp, 4), dtype=np.int32)[0]

    n = []
    nums = []
    for i in range(0, nblocks):
        start_tag = fp.read(4)

        n.append(np.frombuffer(fp.read(8), dtype=np.int64)[0])
        nums.append(np.frombuffer(fp.read(32), count=8, dtype=np.int32))

        end_tag = fp.read(4)
        if (start_tag != end_tag):
            raise AssertionError("Fortran tags mismatch in array blocks.")

    df = pd.DataFrame()
    df_sinks = pd.DataFrame()
    for i in range(0, nblocks):
        # This assumes the second block is only for sink particles.
        # I believe this is a valid assumption as I think this is what splash assumes.
        # For now we will just append sinks to the end of the data frame.
        if i == 1:
            df_sinks = _read_array_block(fp, df_sinks, n[i], nums[i], def_int_dtype, def_real_dtype)
        else:
            df = _read_array_block(fp, df, n[i], nums[i], def_int_dtype, def_real_dtype)

    return df, df_sinks


def read_phantom(filename: str, separate_types: str = 'sinks', ignore_inactive: bool = True):
    """
    Read data from a Phantom dump file.

    This reads the native binary format of Phantom dump files, which in turn were derived from the binary file format
    used by sphNG.

    Global values stored in the dump file (time step, initial momentum, hfact, Courant factor, etc) are stored within the
    data frame in the dictionary ``params``.

    Parameters
    ----------
    filename : str
        Name of the file to be loaded.
    separate_types : {None, 'sinks', 'all'}, default='sinks'
        Whether to separate different particle types into several dataframes. ``None`` returns all particle types in one
        data frame. '`sinks`' separates sink particles into a second dataframe, and '`all`' returns all particle types in
        different dataframes.
    ignore_inactive : {True, False}, default=True
        If True, particles with negative smoothing length will not be read on import. These are
        typically particles that have been accreted onto a sink particle or are otherwise inactive.

    Returns
    -------
    SarracenDataFrame or list of SarracenDataFrame

    Notes
    -----
    See the `Phantom documentation <https://phantomsph.readthedocs.io/en/latest/dumpfile.html>`_ for a full description
    of the Phantom binary file format.

    Examples
    --------
    By default, SPH particles are grouped into one data frame and sink particles into a second data frame.

    >>> sdf, sdf_sinks = sarracen.read_phantom('dumpfile_00000')

    A dump file containing multiple particle types, say gas + dust + sinks, can separated into their own data frames
    by specifying ``separate_types='all'``.

    >>> sdf_gas, sdf_dust, sdf_sinks = sarracen.read_phantom('multiple_types_00000', separate_types='all')
    """
    with open(filename, 'rb') as fp:
        def_int_dtype, def_real_dtype = _read_capture_pattern(fp)
        file_identifier = _read_file_identifier(fp)

        header_vars = _read_global_header(fp, def_int_dtype, def_real_dtype)
        header_vars['file_identifier'] = file_identifier

        df, df_sinks = _read_array_blocks(fp, def_int_dtype, def_real_dtype)

        if ignore_inactive:
            df = df[df['h'] > 0]

        if separate_types == 'all' and 'itype' in df and df['itype'].nunique() > 1:
            df_list = []
            for _, group in df.groupby('itype'):
                itype = int(group["itype"].iloc[0])
                mass_key = 'massoftype' if itype == 1 else f'massoftype_{itype}'
                df_list.append(SarracenDataFrame(group.dropna(axis=1),
                                                 params={**header_vars, **{"mass": header_vars[mass_key]}}))

            df_list.append(SarracenDataFrame(df_sinks, params=header_vars))

            return df_list

        if (separate_types == 'sinks' or separate_types == 'all') and not df_sinks.empty:
            return [SarracenDataFrame(df, params={**header_vars, **{"mass": header_vars['massoftype']}}),
                    SarracenDataFrame(df_sinks, params=header_vars)]

        return SarracenDataFrame(pd.concat([df, df_sinks], ignore_index=True),
                                 params={**header_vars, **{"mass": header_vars['massoftype']}})
