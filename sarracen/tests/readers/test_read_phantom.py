import pandas as pd
import numpy as np
import io
from pandas import testing as tm
from sarracen.readers import _read_capture_pattern
import sarracen
import pytest
import tempfile


def _create_capture_pattern(def_int, def_real):
    """ Construct capture pattern. """

    read_tag = np.array([13], dtype='int32')
    i1 = np.array([60769], dtype=def_int)
    r2 = np.array([60878], dtype=def_real)
    i2 = np.array([60878], dtype=def_int)
    iversion = np.array([0], dtype=def_int)
    i3 = np.array([690706], dtype=def_int)

    capture_pattern = bytearray(read_tag.tobytes())
    capture_pattern += bytearray(i1.tobytes())
    capture_pattern += bytearray(r2.tobytes())
    capture_pattern += bytearray(i2.tobytes())
    capture_pattern += bytearray(iversion.tobytes())
    capture_pattern += bytearray(i3.tobytes())
    capture_pattern += bytearray(read_tag.tobytes())

    return capture_pattern

def _create_file_identifier():
    """ Construct 100-character file identifier. """

    read_tag = np.array([13], dtype='int32')
    file_identifier = "Test of read_phantom".ljust(100)
    bytes_file = bytearray(read_tag.tobytes())
    bytes_file += bytearray(map(ord, file_identifier))
    bytes_file += bytearray(read_tag.tobytes())
    return bytes_file


def _create_global_header(massoftype=1e-6, massoftype_7=None):
    """ Construct global variables. Only massoftype in this example. """

    read_tag = np.array([13], dtype='int32')
    bytes_file = bytearray()
    for i in range(8):
        bytes_file += bytearray(read_tag.tobytes())
        nvars = (i == 5) + (massoftype_7 is not None)
        if i == 5:
            nvars = np.array([nvars], dtype='int32')
        else:
            nvars = np.array([0], dtype='int32')
        bytes_file += bytearray(nvars.tobytes())
        bytes_file += bytearray(read_tag.tobytes())

        if i == 5:
            bytes_file += bytearray(read_tag.tobytes())
            bytes_file += bytearray(map(ord, "massoftype".ljust(16)))
            if massoftype_7 is not None:
                bytes_file += bytearray(map(ord, "massoftype_7".ljust(16)))
            bytes_file += bytearray(read_tag.tobytes())

        if i == 5:
            bytes_file += bytearray(read_tag.tobytes())
            bytes_file += bytearray(np.array([massoftype], dtype=np.float64))
            if massoftype_7 is not None:
                bytes_file += bytearray(np.array([massoftype_7], dtype=np.float64))
            bytes_file += bytearray(read_tag.tobytes())


    return bytes_file

def _create_particle_array(tag, data, dtype=np.float64):
    read_tag = np.array([13], dtype='int32')
    bytes_file = bytearray(read_tag.tobytes())
    bytes_file += bytearray(map(ord,tag.ljust(16)))
    bytes_file += bytearray(read_tag.tobytes())
    bytes_file += bytearray(read_tag.tobytes())
    bytes_file += bytearray(np.array(data, dtype=dtype).tobytes())
    bytes_file += bytearray(read_tag.tobytes())
    return bytes_file


@pytest.mark.parametrize("def_int, def_real",
                         [(np.int32, np.float64), (np.int32, np.float32),
                          (np.int64, np.float64), (np.int64, np.float32)])
def test_determine_default_precision(def_int, def_real):
    """ Test if default int / real precision can be determined. """

    capture_pattern = _create_capture_pattern(def_int, def_real)

    f = io.BytesIO(capture_pattern)
    returned_int, returned_real = _read_capture_pattern(f)
    assert returned_int == def_int
    assert returned_real == def_real


def test_gas_particles_only():

    # capture pattern
    bytes_file = _create_capture_pattern(np.int32, np.float64)

    # file identifier
    bytes_file += _create_file_identifier()

    # global header
    bytes_file += _create_global_header()

    # create 1 block for gas
    read_tag = np.array([13], dtype='int32')
    bytes_file += bytearray(read_tag.tobytes())
    nblocks = np.array([1], dtype='int32')
    bytes_file += bytearray(nblocks.tobytes())
    bytes_file += bytearray(read_tag.tobytes())

    # 8 particles storing 4 real arrays (x, y, z, h)
    bytes_file += bytearray(read_tag.tobytes())
    n = np.array([8], dtype='int64')
    nums = np.array([0, 0, 0, 0, 0, 4, 0, 0], dtype='int32')
    bytes_file += bytearray(n.tobytes())
    bytes_file += bytearray(nums.tobytes())
    bytes_file += bytearray(read_tag.tobytes())

    # write 4 particle arrays
    bytes_file += _create_particle_array("x", [0, 0, 0, 0, 1, 1, 1, 1])
    bytes_file += _create_particle_array("y", [0, 0, 1, 1, 0, 0, 1, 1])
    bytes_file += _create_particle_array("z", [0, 1, 0, 1, 0, 1, 0, 1])
    bytes_file += _create_particle_array("h", [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1])

    with tempfile.NamedTemporaryFile() as fp:
        fp.write(bytes_file)
        fp.seek(0)

        sdf = sarracen.read_phantom(fp.name, separate_types='all')
        assert sdf.params['massoftype'] == 1e-6
        assert sdf.params['mass'] == 1e-6
        assert 'mass' not in sdf.columns
        tm.assert_series_equal(sdf['x'], pd.Series([0, 0, 0, 0, 1, 1, 1, 1]),
                               check_index=False, check_names=False, check_dtype=False)

        sdf = sarracen.read_phantom(fp.name, separate_types='sinks')
        assert sdf.params['massoftype'] == 1e-6
        assert sdf.params['mass'] == 1e-6
        assert 'mass' not in sdf.columns
        tm.assert_series_equal(sdf['x'], pd.Series([0, 0, 0, 0, 1, 1, 1, 1]),
                               check_index=False, check_names=False, check_dtype=False)

        sdf = sarracen.read_phantom(fp.name, separate_types=None)
        assert sdf.params['massoftype'] == 1e-6
        assert sdf.params['mass'] == 1e-6
        assert 'mass' not in sdf.columns
        tm.assert_series_equal(sdf['x'], pd.Series([0, 0, 0, 0, 1, 1, 1, 1]),
                               check_index=False, check_names=False, check_dtype=False)

def test_gas_dust_particles():

    # capture pattern
    bytes_file = _create_capture_pattern(np.int32, np.float64)

    # file identifier
    bytes_file += _create_file_identifier()

    # global header
    bytes_file += _create_global_header(massoftype_7=1e-4)

    # create 1 block for gas
    read_tag = np.array([13], dtype='int32')
    bytes_file += bytearray(read_tag.tobytes())
    nblocks = np.array([1], dtype='int32')
    bytes_file += bytearray(nblocks.tobytes())
    bytes_file += bytearray(read_tag.tobytes())

    # 8 particles storing 4 real arrays (x, y, z, h)
    bytes_file += bytearray(read_tag.tobytes())
    n = np.array([16], dtype='int64')
    nums = np.array([1, 0, 0, 0, 0, 4, 0, 0], dtype='int32')
    bytes_file += bytearray(n.tobytes())
    bytes_file += bytearray(nums.tobytes())
    bytes_file += bytearray(read_tag.tobytes())

    # write 4 particle arrays

    bytes_file += _create_particle_array("itype", [1, 1, 1, 1, 1, 1, 1, 1,
                                         7, 7, 7, 7, 7, 7, 7, 7], np.int32)
    bytes_file += _create_particle_array("x", [0, 0, 0, 0, 1, 1, 1, 1,
                                               0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5])
    bytes_file += _create_particle_array("y", [0, 0, 1, 1, 0, 0, 1, 1,
                                               0.5, 0.5, 1.5, 1.5, 0.5, 0.5, 1.5, 1.5])
    bytes_file += _create_particle_array("z", [0, 1, 0, 1, 0, 1, 0, 1,
                                               0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5])
    bytes_file += _create_particle_array("h", [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1,
                                               1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1])
    with tempfile.NamedTemporaryFile() as fp:
        fp.write(bytes_file)
        fp.seek(0)

        sdf_g, sdf_d = sarracen.read_phantom(fp.name, separate_types='all')
        assert sdf_g.params['massoftype'] == 1e-6
        assert sdf_g.params['massoftype_7'] == 1e-4
        assert sdf_g.params['mass'] == 1e-6
        assert sdf_d.params['massoftype'] == 1e-6
        assert sdf_d.params['massoftype_7'] == 1e-4
        assert sdf_d.params['mass'] == 1e-4
        assert 'mass' not in sdf_g.columns
        assert 'mass' not in sdf_d.columns
        tm.assert_series_equal(sdf_g['x'],
                               pd.Series([0, 0, 0, 0, 1, 1, 1, 1]),
                               check_index=False, check_names=False, check_dtype=False)
        tm.assert_series_equal(sdf_d['x'],
                               pd.Series([0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5]),
                               check_index=False, check_names=False, check_dtype=False)

        sdf = sarracen.read_phantom(fp.name, separate_types='sinks')
        assert sdf.params['massoftype'] == 1e-6
        assert sdf.params['massoftype_7'] == 1e-4
        assert 'mass' not in sdf.params
        assert 'mass' in sdf.columns
        assert sdf[sdf.itype == 1]['mass'].unique() == [1e-6]
        assert sdf[sdf.itype == 7]['mass'].unique() == [1e-4]
        tm.assert_series_equal(sdf[sdf.itype == 1]['x'],
                               pd.Series([0, 0, 0, 0, 1, 1, 1, 1]),
                               check_index=False, check_names=False, check_dtype=False)
        tm.assert_series_equal(sdf[sdf.itype == 7]['x'],
                               pd.Series([0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5]),
                               check_index=False, check_names=False, check_dtype=False)

        sdf = sarracen.read_phantom(fp.name, separate_types=None)
        assert sdf.params['massoftype'] == 1e-6
        assert sdf.params['massoftype_7'] == 1e-4
        assert 'mass' not in sdf.params
        assert 'mass' in sdf.columns
        assert sdf[sdf.itype == 1]['mass'].unique() == [1e-6]
        assert sdf[sdf.itype == 7]['mass'].unique() == [1e-4]
        tm.assert_series_equal(sdf[sdf.itype == 1]['x'],
                               pd.Series([0, 0, 0, 0, 1, 1, 1, 1]),
                               check_index=False, check_names=False, check_dtype=False)
        tm.assert_series_equal(sdf[sdf.itype == 7]['x'],
                               pd.Series([0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5]),
                               check_index=False, check_names=False, check_dtype=False)
