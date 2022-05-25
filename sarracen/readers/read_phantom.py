import numpy as np
import pandas as pd

from sarracen.sarracen_dataframe import SarracenDataFrame

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
    return _read_fortran_block(fp, 100).decode('ascii')



def _read_global_header_block(fp, dtype):
    nvars = np.frombuffer(_read_fortran_block(fp, 4), dtype=np.int32)[0]

    header_vars = dict()

    if (nvars > 0):
        # each tag is 16 characters in length
        keys = _read_fortran_block(fp, 16*nvars).decode('ascii')
        keys = [keys[i:i+16] for i in range(0, len(keys), 16)]

        data = _read_fortran_block(fp, dtype().itemsize*nvars)
        data = np.frombuffer(data, count=nvars, dtype=dtype)

        for i in range(0, nvars):
            header_vars[keys[i].strip()] = data[i]

    return header_vars


def _read_global_header(fp, def_int_dtype, def_real_dtype):
    """ Read global variables. """

    dtypes = [def_int_dtype, np.int8, np.int16, np.int32, np.int64,
                    def_real_dtype, np.float32, np.float64]

    global_vars = dict()
    for dtype in dtypes:
        global_vars.update(_read_global_header_block(fp, dtype))

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
    """ Read particle data. """
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
    for i in range(0, nblocks):
        df = _read_array_block(fp, df, n[i], nums[i], def_int_dtype, def_real_dtype)

    return df



def read_phantom(filename: str) -> SarracenDataFrame:
    """
    Read data from a Phantom dump file.

    Parameters
    ----------
    filename : str
        Name of the file to be loaded.

    Returns
    -------
    SarracenDataFrame
    """
    with open(filename, 'rb') as fp:
        def_int_dtype, def_real_dtype = _read_capture_pattern(fp)
        file_identifier = _read_file_identifier(fp)

        header_vars = _read_global_header(fp, def_int_dtype, def_real_dtype)
        header_vars['file_identifier'] = file_identifier

        df = _read_array_blocks(fp, def_int_dtype, def_real_dtype)

        return SarracenDataFrame(df, params=header_vars)
