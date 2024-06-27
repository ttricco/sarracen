import pandas as pd
import numpy as np
import io
from pandas import testing as tm
import sarracen
import pytest
import tempfile
import hashlib


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


def _create_global_header(massoftype=1e-6, massoftype_7=None,
                          def_int=np.int32, def_real=np.float64):
    """ Construct global variables. Only massoftype in this example. """

    read_tag = np.array([13], dtype='int32')
    bytes_file = bytearray()
    for i in range(8):  # loop over 8 dtypes
        bytes_file += bytearray(read_tag.tobytes())
        nvars = (i == 5) + (massoftype_7 is not None)
        if i == 5:  # default real
            nvars = np.array([nvars], dtype='int32')
        else:
            nvars = np.array([0], dtype='int32')
        bytes_file += bytearray(nvars.tobytes())
        bytes_file += bytearray(read_tag.tobytes())

        if i == 5:  # default real
            bytes_file += bytearray(read_tag.tobytes())
            bytes_file += bytearray(map(ord, "massoftype".ljust(16)))
            if massoftype_7 is not None:
                bytes_file += bytearray(map(ord, "massoftype_7".ljust(16)))
            bytes_file += bytearray(read_tag.tobytes())

        if i == 5:
            bytes_file += bytearray(read_tag.tobytes())
            bytes_file += bytearray(np.array([massoftype], dtype=def_real))
            if massoftype_7 is not None:
                bytes_file += bytearray(np.array([massoftype_7], dtype=def_real))
            bytes_file += bytearray(read_tag.tobytes())

    return bytes_file


def _create_particle_array(tag, data, dtype=np.float64):
    read_tag = np.array([13], dtype='int32')
    bytes_file = bytearray(read_tag.tobytes())
    bytes_file += bytearray(map(ord, tag.ljust(16)))
    bytes_file += bytearray(read_tag.tobytes())

    bytes_file += bytearray(read_tag.tobytes())
    bytes_file += bytearray(np.array(data, dtype=dtype).tobytes())
    bytes_file += bytearray(read_tag.tobytes())
    return bytes_file


@pytest.mark.parametrize("def_int, def_real",
                         [(np.int32, np.float64), (np.int32, np.float32),
                          (np.int64, np.float64), (np.int64, np.float32)])
def test_determine_default_precision2(def_int, def_real):
    """ Test if default int / real precision can be determined. """

    bytes_file = _create_capture_pattern(def_int, def_real)
    bytes_file += _create_file_identifier()
    bytes_file += _create_global_header(def_int=def_int, def_real=def_real)

    # create 1 block for gas
    read_tag = np.array([13], dtype='int32')
    bytes_file += bytearray(read_tag.tobytes())
    nblocks = np.array([1], dtype='int32')
    bytes_file += bytearray(nblocks.tobytes())
    bytes_file += bytearray(read_tag.tobytes())

    # 2 particles storing 1 default int and real arrays
    bytes_file += bytearray(read_tag.tobytes())
    n = np.array([2], dtype='int64')
    nums = np.array([1, 0, 0, 0, 0, 1, 0, 0], dtype='int32')
    bytes_file += bytearray(n.tobytes())
    bytes_file += bytearray(nums.tobytes())
    bytes_file += bytearray(read_tag.tobytes())

    # write particle arrays
    bytes_file += _create_particle_array("def_int", [1, 2], dtype=def_int)
    bytes_file += _create_particle_array("def_real", [1.0, 2.0], dtype=def_real)

    with tempfile.NamedTemporaryFile() as fp:
        fp.write(bytes_file)
        fp.seek(0)

        sdf = sarracen.read_phantom(fp.name)

        assert list(sdf.dtypes) == [def_int, def_real]


def get_df():
    bytes_file = _create_capture_pattern(np.int32, np.float64)
    bytes_file += _create_file_identifier()
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
    return sdf


class GlobalHeaderElement:
    def __init__(self, d_type, tags, values):
        self.d_type = d_type
        self.tags = tags
        self.values = values


class DtArrayCount:
    def __init__(self, d_type, count, tags):
        self.d_type = d_type
        self.count = count
        self.tags = tags


def _write_global_header(params_dict):
    dtypes = [np.int32, np.int8, np.int16, np.int32, np.int64, np.float64, np.float32, np.float64]

    global_headers = []
    used_keys = []

    for dt in dtypes:
        header_element = GlobalHeaderElement(dt, [], [])
        global_headers.append(header_element)

    for ghe in global_headers:
        for key in params_dict:
            if invalid_key(key, used_keys):
                continue
            if isinstance(params_dict[key], ghe.d_type):
                ghe.tags.append(key)
                used_keys.append(key)
                ghe.values.append(params_dict[key])

    read_tag = np.array([13], dtype='int32')
    bytes_file = bytearray()

    for header in global_headers:

        bytes_file = add_4byte_tag(bytes_file, read_tag)
        nvars = np.array([len(header.tags)], dtype='int32')
        bytes_file += bytearray(nvars.tobytes())
        bytes_file = add_4byte_tag(bytes_file, read_tag)

        if nvars == 0:
            continue

        bytes_file = add_4byte_tag(bytes_file, read_tag)
        for tg in header.tags:
            bytes_file += bytearray(map(ord, tg.ljust(16)))
        bytes_file = add_4byte_tag(bytes_file, read_tag)

        bytes_file = add_4byte_tag(bytes_file, read_tag)
        values = np.array(header.values, dtype=header.d_type)
        bytes_file += bytearray(values.tobytes())
        bytes_file = add_4byte_tag(bytes_file, read_tag)

    return bytes_file


def add_4byte_tag(bytes_file, read_tag):
    bytes_file += bytearray(read_tag.tobytes())
    return bytes_file


def invalid_key(key, used_keys):
    return key == 'file_identifier' or key == 'mass' or key in used_keys


def test_write_phantom():
    # test_sdf = sarracen.read_phantom('hydro32_00020')

    test_sdf = get_df()

    ph_file = _create_capture_pattern(np.int32,
                                      np.float64)  # can create this when can identify def_int_dtype, def_real_dtype

    file_identifier = test_sdf.params['file_identifier'].ljust(100)

    ph_file = write_fortrun_block(ph_file, file_identifier)

    ph_file += _write_global_header(test_sdf.params)

    ph_file += _test_write_arrays(ph_file, test_sdf)

    ph_file = write_test_particle_array(ph_file)

    with tempfile.NamedTemporaryFile() as fp:
        fp.write(ph_file)
        fp.seek(0)

        sdf_written = sarracen.read_phantom(fp.name, separate_types='all')

    sdf_hash = int(hashlib.sha256(pd.util.hash_pandas_object(test_sdf, index=True).values).hexdigest(), 16)
    sdf_written_hash = int(hashlib.sha256(pd.util.hash_pandas_object(sdf_written, index=True).values).hexdigest(), 16)

    assert sdf_hash == sdf_written_hash;


def _test_write_arrays(file, test_sdf):

    dtypes = [np.int32, np.int8, np.int16, np.int32, np.int64, np.float64, np.float32, np.float64]
    read_tag = np.array([13], dtype='int32')
    num_blocks = 1

    file += bytearray(read_tag.tobytes())
    file += bytearray(np.array([num_blocks], dtype='int32').tobytes())
    file += bytearray(read_tag.tobytes())

    nvars = np.array([test_sdf.shape[0]], dtype='int64')
    array_count = []
    dtypes_used = []
    for type in dtypes:
        if type not in dtypes_used:
            dt_array_count = DtArrayCount(type, count_num_dt_arrays(test_sdf, type), get_array_tags(test_sdf, type))
            array_count.append(dt_array_count)
            dtypes_used.append(type)
        else:
            dt_array_count = DtArrayCount(type, 0, [])
            array_count.append(dt_array_count)

    file = bytearray(read_tag.tobytes())
    file += bytearray(nvars.tobytes())
    counts = []
    for ct in array_count:
        counts.append(ct.count)
    file += bytearray(np.array(counts, dtype='int32').tobytes())
    file += bytearray(read_tag.tobytes())

    for ct in array_count:
        if ct.count > 0:
            for tag in ct.tags:
                file += bytearray(read_tag.tobytes())
                file += bytearray(map(ord, tag.ljust(16)))
                file += bytearray(read_tag.tobytes())

                file += bytearray(read_tag.tobytes())
                file += bytearray(np.array(list(test_sdf[tag]), dtype=ct.d_type).tobytes())
                file += bytearray(read_tag.tobytes())
    return file


def get_array_tags(test_sdf, dt):
    return list(test_sdf.select_dtypes(include=[dt]).columns)


def count_num_dt_arrays(test_sdf, dt):
    count_dt = 0
    for element in test_sdf.dtypes.values:
        if element == dt:
            count_dt += 1
    return count_dt

def write_test_particle_array(ph_file):
    read_tag = np.array([13], dtype='int32')
    ph_file += bytearray(read_tag.tobytes())
    nblocks = np.array([1], dtype='int32')
    ph_file += bytearray(nblocks.tobytes())
    ph_file += bytearray(read_tag.tobytes())
    # 8 particles storing 4 real arrays (x, y, z, h)
    ph_file += bytearray(read_tag.tobytes())
    n = np.array([8], dtype='int64')
    nums = np.array([0, 0, 0, 0, 0, 4, 0, 0], dtype='int32')
    ph_file += bytearray(n.tobytes())
    ph_file += bytearray(nums.tobytes())
    ph_file += bytearray(read_tag.tobytes())
    # write 4 particle arrays
    ph_file += _create_particle_array("x", [0, 0, 0, 0, 1, 1, 1, 1])
    ph_file += _create_particle_array("y", [0, 0, 1, 1, 0, 0, 1, 1])
    ph_file += _create_particle_array("z", [0, 1, 0, 1, 0, 1, 0, 1])
    ph_file += _create_particle_array("h", [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1])
    return ph_file


def write_fortrun_block(ph_file, data):
    read_tag = np.array([13], dtype='int32')
    ph_file += bytearray(read_tag.tobytes())
    ph_file += bytearray(map(ord, data))
    ph_file += bytearray(read_tag.tobytes())
    return ph_file


def test_gas_particles_only():
    bytes_file = _create_capture_pattern(np.int32, np.float64)
    bytes_file += _create_file_identifier()
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
    bytes_file = _create_capture_pattern(np.int32, np.float64)
    bytes_file += _create_file_identifier()
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
    nums = np.array([0, 1, 0, 0, 0, 4, 0, 0], dtype='int32')
    bytes_file += bytearray(n.tobytes())
    bytes_file += bytearray(nums.tobytes())
    bytes_file += bytearray(read_tag.tobytes())

    # write 5 gas/dust particle arrays
    bytes_file += _create_particle_array("itype", [1, 1, 1, 1, 1, 1, 1, 1,
                                                   7, 7, 7, 7, 7, 7, 7, 7], np.int8)
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


def test_gas_sink_particles():
    bytes_file = _create_capture_pattern(np.int32, np.float64)
    bytes_file += _create_file_identifier()
    bytes_file += _create_global_header()

    # create 1 block for gas
    read_tag = np.array([13], dtype='int32')
    bytes_file += bytearray(read_tag.tobytes())
    nblocks = np.array([2], dtype='int32')
    bytes_file += bytearray(nblocks.tobytes())
    bytes_file += bytearray(read_tag.tobytes())

    # 8 particles storing 4 real arrays (x, y, z, h)
    bytes_file += bytearray(read_tag.tobytes())
    n = np.array([8], dtype='int64')
    nums = np.array([0, 0, 0, 0, 0, 4, 0, 0], dtype='int32')
    bytes_file += bytearray(n.tobytes())
    bytes_file += bytearray(nums.tobytes())
    bytes_file += bytearray(read_tag.tobytes())

    bytes_file += bytearray(read_tag.tobytes())
    n = np.array([1], dtype='int64')
    nums = np.array([0, 0, 0, 0, 0, 7, 0, 0], dtype='int32')
    bytes_file += bytearray(n.tobytes())
    bytes_file += bytearray(nums.tobytes())
    bytes_file += bytearray(read_tag.tobytes())

    # write 4 gas particle arrays
    bytes_file += _create_particle_array("x", [0, 0, 0, 0, 1, 1, 1, 1])
    bytes_file += _create_particle_array("y", [0, 0, 1, 1, 0, 0, 1, 1])
    bytes_file += _create_particle_array("z", [0, 1, 0, 1, 0, 1, 0, 1])
    bytes_file += _create_particle_array("h", [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1])

    # write 7 sink particle arrays
    bytes_file += _create_particle_array("x", [0.000305])
    bytes_file += _create_particle_array("y", [-0.035809])
    bytes_file += _create_particle_array("z", [-0.000035])
    bytes_file += _create_particle_array("h", [1.0])
    bytes_file += _create_particle_array("spinx", [-3.911744e-8])
    bytes_file += _create_particle_array("spiny", [-1.326062e-8])
    bytes_file += _create_particle_array("spinz", [0.00058])

    with tempfile.NamedTemporaryFile() as fp:
        fp.write(bytes_file)
        fp.seek(0)

        sdf, sdf_sinks = sarracen.read_phantom(fp.name, separate_types='all')
        assert sdf.params['massoftype'] == 1e-6
        assert sdf_sinks.params['massoftype'] == 1e-6
        assert sdf.params['mass'] == 1e-6
        assert 'mass' not in sdf_sinks.params
        assert 'mass' not in sdf.columns
        assert 'mass' not in sdf_sinks.columns
        tm.assert_series_equal(sdf['x'], pd.Series([0, 0, 0, 0, 1, 1, 1, 1]),
                               check_index=False, check_names=False, check_dtype=False)
        tm.assert_series_equal(sdf_sinks['x'], pd.Series([0.000305]),
                               check_index=False, check_names=False, check_dtype=False)
        tm.assert_series_equal(sdf_sinks['spinx'], pd.Series([-3.911744e-8]),
                               check_index=False, check_names=False, check_dtype=False)

        sdf, sdf_sinks = sarracen.read_phantom(fp.name, separate_types='sinks')
        assert sdf.params['massoftype'] == 1e-6
        assert sdf_sinks.params['massoftype'] == 1e-6
        assert sdf.params['mass'] == 1e-6
        assert 'mass' not in sdf_sinks.params
        assert 'mass' not in sdf.columns
        assert 'mass' not in sdf_sinks.columns
        tm.assert_series_equal(sdf['x'], pd.Series([0, 0, 0, 0, 1, 1, 1, 1]),
                               check_index=False, check_names=False, check_dtype=False)
        tm.assert_series_equal(sdf_sinks['x'], pd.Series([0.000305]),
                               check_index=False, check_names=False, check_dtype=False)
        tm.assert_series_equal(sdf_sinks['spinx'], pd.Series([-3.911744e-8]),
                               check_index=False, check_names=False, check_dtype=False)

        sdf = sarracen.read_phantom(fp.name, separate_types=None)
        assert sdf.params['massoftype'] == 1e-6
        assert sdf.params['mass'] == 1e-6
        assert 'mass' not in sdf.columns
        tm.assert_series_equal(sdf['x'], pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 0.000305]),
                               check_index=False, check_names=False, check_dtype=False)
        tm.assert_series_equal(sdf['h'], pd.Series([1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.0]),
                               check_index=False, check_names=False, check_dtype=False)


def test_gas_dust_sink_particles():
    bytes_file = _create_capture_pattern(np.int32, np.float64)
    bytes_file += _create_file_identifier()
    bytes_file += _create_global_header(massoftype_7=1e-4)

    # create 1 block for gas
    read_tag = np.array([13], dtype='int32')
    bytes_file += bytearray(read_tag.tobytes())
    nblocks = np.array([2], dtype='int32')
    bytes_file += bytearray(nblocks.tobytes())
    bytes_file += bytearray(read_tag.tobytes())

    # 8 particles storing 4 real arrays (x, y, z, h)
    bytes_file += bytearray(read_tag.tobytes())
    n = np.array([16], dtype='int64')
    nums = np.array([0, 1, 0, 0, 0, 4, 0, 0], dtype='int32')
    bytes_file += bytearray(n.tobytes())
    bytes_file += bytearray(nums.tobytes())
    bytes_file += bytearray(read_tag.tobytes())

    bytes_file += bytearray(read_tag.tobytes())
    n = np.array([1], dtype='int64')
    nums = np.array([0, 0, 0, 0, 0, 7, 0, 0], dtype='int32')
    bytes_file += bytearray(n.tobytes())
    bytes_file += bytearray(nums.tobytes())
    bytes_file += bytearray(read_tag.tobytes())

    # write 5 gas/dust particle arrays
    bytes_file += _create_particle_array("itype", [1, 1, 1, 1, 1, 1, 1, 1,
                                                   7, 7, 7, 7, 7, 7, 7, 7], np.int8)
    bytes_file += _create_particle_array("x", [0, 0, 0, 0, 1, 1, 1, 1,
                                               0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5])
    bytes_file += _create_particle_array("y", [0, 0, 1, 1, 0, 0, 1, 1,
                                               0.5, 0.5, 1.5, 1.5, 0.5, 0.5, 1.5, 1.5])
    bytes_file += _create_particle_array("z", [0, 1, 0, 1, 0, 1, 0, 1,
                                               0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5])
    bytes_file += _create_particle_array("h", [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1,
                                               1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1])

    # write 7 sink particle arrays
    bytes_file += _create_particle_array("x", [0.000305])
    bytes_file += _create_particle_array("y", [-0.035809])
    bytes_file += _create_particle_array("z", [-0.000035])
    bytes_file += _create_particle_array("h", [1.0])
    bytes_file += _create_particle_array("spinx", [-3.911744e-8])
    bytes_file += _create_particle_array("spiny", [-1.326062e-8])
    bytes_file += _create_particle_array("spinz", [0.00058])

    with tempfile.NamedTemporaryFile() as fp:
        fp.write(bytes_file)
        fp.seek(0)

        sdf_g, sdf_d, sdf_sinks = sarracen.read_phantom(fp.name, separate_types='all')
        assert sdf_g.params['massoftype'] == 1e-6
        assert sdf_g.params['massoftype_7'] == 1e-4
        assert sdf_g.params['mass'] == 1e-6
        assert sdf_d.params['massoftype'] == 1e-6
        assert sdf_d.params['massoftype_7'] == 1e-4
        assert sdf_d.params['mass'] == 1e-4
        assert sdf_sinks.params['massoftype'] == 1e-6
        assert sdf_sinks.params['massoftype_7'] == 1e-4
        assert 'mass' not in sdf_sinks.params
        assert 'mass' not in sdf_g.columns
        assert 'mass' not in sdf_d.columns
        assert 'mass' not in sdf_sinks.columns
        tm.assert_series_equal(sdf_g['x'],
                               pd.Series([0, 0, 0, 0, 1, 1, 1, 1]),
                               check_index=False, check_names=False, check_dtype=False)
        tm.assert_series_equal(sdf_d['x'],
                               pd.Series([0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5]),
                               check_index=False, check_names=False, check_dtype=False)
        tm.assert_series_equal(sdf_sinks['x'], pd.Series([0.000305]),
                               check_index=False, check_names=False, check_dtype=False)
        tm.assert_series_equal(sdf_sinks['spinx'], pd.Series([-3.911744e-8]),
                               check_index=False, check_names=False, check_dtype=False)

        sdf, sdf_sinks = sarracen.read_phantom(fp.name, separate_types='sinks')
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
        assert sdf_sinks.params['massoftype'] == 1e-6
        assert sdf_sinks.params['massoftype_7'] == 1e-4
        assert 'mass' not in sdf_sinks.params
        assert 'mass' not in sdf_sinks.columns
        tm.assert_series_equal(sdf_sinks['x'], pd.Series([0.000305]),
                               check_index=False, check_names=False, check_dtype=False)
        tm.assert_series_equal(sdf_sinks['spinx'], pd.Series([-3.911744e-8]),
                               check_index=False, check_names=False, check_dtype=False)

        sdf = sarracen.read_phantom(fp.name, separate_types=None)
        assert sdf.params['massoftype'] == 1e-6
        assert sdf.params['massoftype_7'] == 1e-4
        assert 'mass' not in sdf.params
        assert 'mass' in sdf.columns
        assert sdf[sdf.itype == 1]['mass'].unique() == [1e-6]
        assert sdf[sdf.itype == 7]['mass'].unique() == [1e-4]
        tm.assert_series_equal(sdf['x'], pd.Series([0, 0, 0, 0, 1, 1, 1, 1,
                                                    0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5,
                                                    0.000305]),
                               check_index=False, check_names=False, check_dtype=False)
        tm.assert_series_equal(sdf['h'], pd.Series([1.1] * 16 + [1.0]),
                               check_index=False, check_names=False, check_dtype=False)
        tm.assert_series_equal(sdf['mass'], pd.Series([1e-6] * 8 + [1e-4] * 8 + [np.nan]),
                               check_index=False, check_names=False, check_dtype=False)
