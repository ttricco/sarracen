from typing import Type, Union

import pandas as pd
import numpy as np
from pandas import testing as tm
import sarracen
import pytest
import tempfile

from sarracen import SarracenDataFrame


def _create_capture_pattern(def_int: Type[np.generic],
                            def_real: Type[np.generic],
                            swap_endian: bool = False) -> bytearray:
    """ Construct capture pattern. """

    read_tag = np.array([13], dtype='int32')
    i1: np.ndarray = np.array([60769], dtype=def_int)
    r2: np.ndarray = np.array([60878], dtype=def_real)
    i2: np.ndarray = np.array([60878], dtype=def_int)
    iversion: np.ndarray = np.array([0], dtype=def_int)
    i3: np.ndarray = np.array([690706], dtype=def_int)

    if swap_endian:
        read_tag = read_tag.byteswap()
        i1 = i1.byteswap()
        r2 = r2.byteswap()
        i2 = i2.byteswap()
        iversion = iversion.byteswap()
        i3 = i3.byteswap()

    capture_pattern = bytearray(read_tag.tobytes())
    capture_pattern += bytearray(i1.tobytes())
    capture_pattern += bytearray(r2.tobytes())
    capture_pattern += bytearray(i2.tobytes())
    capture_pattern += bytearray(iversion.tobytes())
    capture_pattern += bytearray(i3.tobytes())
    capture_pattern += bytearray(read_tag.tobytes())

    return capture_pattern


def _create_file_identifier(swap_endian: bool = False) -> bytearray:
    """ Construct 100-character file identifier. """

    read_tag = np.array([13], dtype='int32')
    if swap_endian:
        read_tag = read_tag.byteswap()
    file_identifier = "Test of read_phantom".ljust(100)
    file = bytearray(read_tag.tobytes())
    file += bytearray(map(ord, file_identifier))
    file += bytearray(read_tag.tobytes())
    return file


def _create_global_header(massoftype: float = 1e-6,
                          massoftype_7: Union[float, None] = None,
                          def_int: Type[np.generic] = np.int32,
                          def_real: Type[np.generic] = np.float64,
                          swap_endian: bool = False) -> bytearray:
    """ Construct global variables. Only massoftype in this example. """

    read_tag = np.array([13], dtype='int32')
    if swap_endian:
        read_tag = read_tag.byteswap()
    file = bytearray()
    for i in range(8):  # loop over 8 dtypes
        file += bytearray(read_tag.tobytes())
        nvars = (i == 5) + (massoftype_7 is not None)
        if i == 5:  # default real
            nvars_arr = np.array([nvars], dtype='int32')
        else:
            nvars_arr = np.array([0], dtype='int32')

        if swap_endian:
            nvars_arr = nvars_arr.byteswap()
        file += bytearray(nvars_arr.tobytes())
        file += bytearray(read_tag.tobytes())

        if i == 5:  # default real
            file += bytearray(read_tag.tobytes())
            file += bytearray(map(ord, "massoftype".ljust(16)))
            if massoftype_7 is not None:
                file += bytearray(map(ord, "massoftype_7".ljust(16)))
            file += bytearray(read_tag.tobytes())

        if i == 5:
            file += bytearray(read_tag.tobytes())
            massoftype_arr = np.array([massoftype], dtype=def_real)
            if swap_endian:
                massoftype_arr = massoftype_arr.byteswap()
            file += bytearray(massoftype_arr)
            if massoftype_7 is not None:
                massoftype_7_arr = np.array([massoftype_7], dtype=def_real)
                if swap_endian:
                    massoftype_7_arr = massoftype_7_arr.byteswap()
                file += bytearray(massoftype_7_arr)
            file += bytearray(read_tag.tobytes())

    return file


def _create_particle_array(tag: str,
                           data: list,
                           dtype: Type[np.generic] = np.float64,
                           swap_endian: bool = False) -> bytearray:
    read_tag = np.array([13], dtype='int32')
    if swap_endian:
        read_tag = read_tag.byteswap()
    file = bytearray(read_tag.tobytes())
    file += bytearray(map(ord, tag.ljust(16)))
    file += bytearray(read_tag.tobytes())
    file += bytearray(read_tag.tobytes())
    data_arr = np.array(data, dtype=dtype)
    if swap_endian:
        data_arr = data_arr.byteswap()
    file += bytearray(data_arr.tobytes())
    file += bytearray(read_tag.tobytes())
    return file


@pytest.mark.parametrize("def_int, def_real",
                         [(np.int32, np.float64), (np.int32, np.float32),
                          (np.int64, np.float64), (np.int64, np.float32)])
def test_determine_default_precision2(def_int: Type[np.generic],
                                      def_real: Type[np.generic]) -> None:
    """ Test if default int / real precision can be determined. """

    file = _create_capture_pattern(def_int, def_real)
    file += _create_file_identifier()
    file += _create_global_header(def_int=def_int, def_real=def_real)

    # create 1 block for gas
    read_tag = np.array([13], dtype='int32')
    file += bytearray(read_tag.tobytes())
    nblocks = np.array([1], dtype='int32')
    file += bytearray(nblocks.tobytes())
    file += bytearray(read_tag.tobytes())

    # 2 particles storing 1 default int and real arrays
    file += bytearray(read_tag.tobytes())
    n = np.array([2], dtype='int64')
    nums = np.array([1, 0, 0, 0, 0, 1, 0, 0], dtype='int32')
    file += bytearray(n.tobytes())
    file += bytearray(nums.tobytes())
    file += bytearray(read_tag.tobytes())

    # write particle arrays
    file += _create_particle_array("def_int", [1, 2], dtype=def_int)
    file += _create_particle_array("def_real", [1.0, 2.0], dtype=def_real)

    with tempfile.NamedTemporaryFile() as fp:
        fp.write(file)
        fp.seek(0)

        sdf = sarracen.read_phantom(fp.name, separate_types=None)

        assert list(sdf.dtypes) == [def_int, def_real]


def test_gas_particles_only() -> None:

    file = _create_capture_pattern(np.int32, np.float64)
    file += _create_file_identifier()
    file += _create_global_header()

    # create 1 block for gas
    read_tag = np.array([13], dtype='int32')
    file += bytearray(read_tag.tobytes())
    nblocks = np.array([1], dtype='int32')
    file += bytearray(nblocks.tobytes())
    file += bytearray(read_tag.tobytes())

    # 8 particles storing 4 real arrays (x, y, z, h)
    file += bytearray(read_tag.tobytes())
    n = np.array([8], dtype='int64')
    nums = np.array([0, 0, 0, 0, 0, 4, 0, 0], dtype='int32')
    file += bytearray(n.tobytes())
    file += bytearray(nums.tobytes())
    file += bytearray(read_tag.tobytes())

    # write 4 particle arrays
    file += _create_particle_array("x", [0, 0, 0, 0, 1, 1, 1, 1])
    file += _create_particle_array("y", [0, 0, 1, 1, 0, 0, 1, 1])
    file += _create_particle_array("z", [0, 1, 0, 1, 0, 1, 0, 1])
    file += _create_particle_array("h", [1.1, 1.1, 1.1, 1.1,
                                         1.1, 1.1, 1.1, 1.1])

    with tempfile.NamedTemporaryFile() as fp:
        fp.write(file)
        fp.seek(0)

        sdf = sarracen.read_phantom(fp.name, separate_types='all')
        assert isinstance(sdf, SarracenDataFrame)
        assert sdf.params is not None
        assert sdf.params['massoftype'] == 1e-6
        assert sdf.params['mass'] == 1e-6
        assert 'mass' not in sdf.columns
        tm.assert_series_equal(sdf['x'], pd.Series([0, 0, 0, 0, 1, 1, 1, 1]),
                               check_index=False, check_names=False,
                               check_dtype=False)

        sdf = sarracen.read_phantom(fp.name, separate_types='sinks')
        assert isinstance(sdf, SarracenDataFrame)
        assert sdf.params is not None
        assert sdf.params['massoftype'] == 1e-6
        assert sdf.params['mass'] == 1e-6
        assert 'mass' not in sdf.columns
        tm.assert_series_equal(sdf['x'], pd.Series([0, 0, 0, 0, 1, 1, 1, 1]),
                               check_index=False, check_names=False,
                               check_dtype=False)

        sdf = sarracen.read_phantom(fp.name, separate_types=None)
        assert isinstance(sdf, SarracenDataFrame)
        assert sdf.params is not None
        assert sdf.params['massoftype'] == 1e-6
        assert sdf.params['mass'] == 1e-6
        assert 'mass' not in sdf.columns
        tm.assert_series_equal(sdf['x'], pd.Series([0, 0, 0, 0, 1, 1, 1, 1]),
                               check_index=False, check_names=False,
                               check_dtype=False)


def test_gas_particles_only_big_endian() -> None:

    file = _create_capture_pattern(np.int32, np.float64, True)
    file += _create_file_identifier(True)
    file += _create_global_header(swap_endian = True)

    # create 1 block for gas
    read_tag = np.array([13], dtype='int32').byteswap()
    file += bytearray(read_tag.tobytes())
    nblocks = np.array([1], dtype='int32').byteswap()
    file += bytearray(nblocks.tobytes())
    file += bytearray(read_tag.tobytes())

    # 8 particles storing 4 real arrays (x, y, z, h)
    file += bytearray(read_tag.tobytes())
    n = np.array([8], dtype='int64').byteswap()
    nums = np.array([0, 0, 0, 0, 0, 4, 0, 0], dtype='int32').byteswap()
    file += bytearray(n.tobytes())
    file += bytearray(nums.tobytes())
    file += bytearray(read_tag.tobytes())

    # write 4 particle arrays
    file += _create_particle_array("x", [0, 0, 0, 0, 1, 1, 1, 1],
                                   swap_endian=True)
    file += _create_particle_array("y", [0, 0, 1, 1, 0, 0, 1, 1],
                                   swap_endian=True)
    file += _create_particle_array("z", [0, 1, 0, 1, 0, 1, 0, 1],
                                   swap_endian=True)
    file += _create_particle_array("h", [1.1, 1.1, 1.1, 1.1,
                                         1.1, 1.1, 1.1, 1.1],
                                   swap_endian=True)

    with tempfile.NamedTemporaryFile() as fp:
        fp.write(file)
        fp.seek(0)

        sdf = sarracen.read_phantom(fp.name, separate_types='all')
        assert isinstance(sdf, SarracenDataFrame)
        assert sdf.params is not None
        assert sdf.params['massoftype'] == 1e-6
        assert sdf.params['mass'] == 1e-6
        assert 'mass' not in sdf.columns
        tm.assert_series_equal(sdf['x'], pd.Series([0, 0, 0, 0, 1, 1, 1, 1]),
                               check_index=False, check_names=False,
                               check_dtype=False)

        sdf = sarracen.read_phantom(fp.name, separate_types='sinks')
        assert isinstance(sdf, SarracenDataFrame)
        assert sdf.params is not None
        assert sdf.params['massoftype'] == 1e-6
        assert sdf.params['mass'] == 1e-6
        assert 'mass' not in sdf.columns
        tm.assert_series_equal(sdf['x'], pd.Series([0, 0, 0, 0, 1, 1, 1, 1]),
                               check_index=False, check_names=False,
                               check_dtype=False)

        sdf = sarracen.read_phantom(fp.name, separate_types=None)
        assert isinstance(sdf, SarracenDataFrame)
        assert sdf.params is not None
        assert sdf.params['massoftype'] == 1e-6
        assert sdf.params['mass'] == 1e-6
        assert 'mass' not in sdf.columns
        tm.assert_series_equal(sdf['x'], pd.Series([0, 0, 0, 0, 1, 1, 1, 1]),
                               check_index=False, check_names=False,
                               check_dtype=False)


def test_gas_dust_particles() -> None:

    file = _create_capture_pattern(np.int32, np.float64)
    file += _create_file_identifier()
    file += _create_global_header(massoftype_7=1e-4)

    # create 1 block for gas
    read_tag = np.array([13], dtype='int32')
    file += bytearray(read_tag.tobytes())
    nblocks = np.array([1], dtype='int32')
    file += bytearray(nblocks.tobytes())
    file += bytearray(read_tag.tobytes())

    # 8 particles storing 4 real arrays (x, y, z, h)
    file += bytearray(read_tag.tobytes())
    n = np.array([16], dtype='int64')
    nums = np.array([0, 1, 0, 0, 0, 4, 0, 0], dtype='int32')
    file += bytearray(n.tobytes())
    file += bytearray(nums.tobytes())
    file += bytearray(read_tag.tobytes())

    # write 5 gas/dust particle arrays
    file += _create_particle_array("itype", [1, 1, 1, 1, 1, 1, 1, 1,
                                             7, 7, 7, 7, 7, 7, 7, 7], np.int8)
    file += _create_particle_array("x", [0, 0, 0, 0,
                                         1, 1, 1, 1,
                                         0.5, 0.5, 0.5, 0.5,
                                         1.5, 1.5, 1.5, 1.5])
    file += _create_particle_array("y", [0, 0, 1, 1,
                                         0, 0, 1, 1,
                                         0.5, 0.5, 1.5, 1.5,
                                         0.5, 0.5, 1.5, 1.5])
    file += _create_particle_array("z", [0, 1, 0, 1,
                                         0, 1, 0, 1,
                                         0.5, 1.5, 0.5, 1.5,
                                         0.5, 1.5, 0.5, 1.5])
    file += _create_particle_array("h", [1.1, 1.1, 1.1, 1.1,
                                         1.1, 1.1, 1.1, 1.1,
                                         1.1, 1.1, 1.1, 1.1,
                                         1.1, 1.1, 1.1, 1.1])
    with tempfile.NamedTemporaryFile() as fp:
        fp.write(file)
        fp.seek(0)

        sdf_g, sdf_d = sarracen.read_phantom(fp.name, separate_types='all')
        assert isinstance(sdf_g, SarracenDataFrame)
        assert isinstance(sdf_d, SarracenDataFrame)
        assert sdf_g.params is not None
        assert sdf_d.params is not None
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
                               check_index=False, check_names=False,
                               check_dtype=False)
        tm.assert_series_equal(sdf_d['x'],
                               pd.Series([0.5, 0.5, 0.5, 0.5,
                                          1.5, 1.5, 1.5, 1.5]),
                               check_index=False, check_names=False,
                               check_dtype=False)

        sdf = sarracen.read_phantom(fp.name, separate_types='sinks')
        assert isinstance(sdf, SarracenDataFrame)
        assert sdf.params is not None
        assert sdf.params['massoftype'] == 1e-6
        assert sdf.params['massoftype_7'] == 1e-4
        assert 'mass' not in sdf.params
        assert 'mass' in sdf.columns
        assert sdf[sdf.itype == 1]['mass'].unique() == [1e-6]
        assert sdf[sdf.itype == 7]['mass'].unique() == [1e-4]
        tm.assert_series_equal(sdf[sdf.itype == 1]['x'],
                               pd.Series([0, 0, 0, 0, 1, 1, 1, 1]),
                               check_index=False, check_names=False,
                               check_dtype=False)
        tm.assert_series_equal(sdf[sdf.itype == 7]['x'],
                               pd.Series([0.5, 0.5, 0.5, 0.5,
                                          1.5, 1.5, 1.5, 1.5]),
                               check_index=False, check_names=False,
                               check_dtype=False)

        sdf = sarracen.read_phantom(fp.name, separate_types=None)
        assert isinstance(sdf, SarracenDataFrame)
        assert sdf.params is not None
        assert sdf.params['massoftype'] == 1e-6
        assert sdf.params['massoftype_7'] == 1e-4
        assert 'mass' not in sdf.params
        assert 'mass' in sdf.columns
        assert sdf[sdf.itype == 1]['mass'].unique() == [1e-6]
        assert sdf[sdf.itype == 7]['mass'].unique() == [1e-4]
        tm.assert_series_equal(sdf[sdf.itype == 1]['x'],
                               pd.Series([0, 0, 0, 0, 1, 1, 1, 1]),
                               check_index=False, check_names=False,
                               check_dtype=False)
        tm.assert_series_equal(sdf[sdf.itype == 7]['x'],
                               pd.Series([0.5, 0.5, 0.5, 0.5, 1.5,
                                          1.5, 1.5, 1.5]),
                               check_index=False, check_names=False,
                               check_dtype=False)


def test_gas_sink_particles() -> None:

    file = _create_capture_pattern(np.int32, np.float64)
    file += _create_file_identifier()
    file += _create_global_header()

    # create 1 block for gas
    read_tag = np.array([13], dtype='int32')
    file += bytearray(read_tag.tobytes())
    nblocks = np.array([2], dtype='int32')
    file += bytearray(nblocks.tobytes())
    file += bytearray(read_tag.tobytes())

    # 8 particles storing 4 real arrays (x, y, z, h)
    file += bytearray(read_tag.tobytes())
    n = np.array([8], dtype='int64')
    nums = np.array([0, 0, 0, 0, 0, 4, 0, 0], dtype='int32')
    file += bytearray(n.tobytes())
    file += bytearray(nums.tobytes())
    file += bytearray(read_tag.tobytes())

    file += bytearray(read_tag.tobytes())
    n = np.array([1], dtype='int64')
    nums = np.array([0, 0, 0, 0, 0, 7, 0, 0], dtype='int32')
    file += bytearray(n.tobytes())
    file += bytearray(nums.tobytes())
    file += bytearray(read_tag.tobytes())

    # write 4 gas particle arrays
    file += _create_particle_array("x", [0, 0, 0, 0, 1, 1, 1, 1])
    file += _create_particle_array("y", [0, 0, 1, 1, 0, 0, 1, 1])
    file += _create_particle_array("z", [0, 1, 0, 1, 0, 1, 0, 1])
    file += _create_particle_array("h", [1.1, 1.1, 1.1, 1.1,
                                         1.1, 1.1, 1.1, 1.1])

    # write 7 sink particle arrays
    file += _create_particle_array("x", [0.000305])
    file += _create_particle_array("y", [-0.035809])
    file += _create_particle_array("z", [-0.000035])
    file += _create_particle_array("h", [1.0])
    file += _create_particle_array("spinx", [-3.911744e-8])
    file += _create_particle_array("spiny", [-1.326062e-8])
    file += _create_particle_array("spinz", [0.00058])

    with tempfile.NamedTemporaryFile() as fp:
        fp.write(file)
        fp.seek(0)

        sdf, sdf_sinks = sarracen.read_phantom(fp.name, separate_types='all')
        assert isinstance(sdf, SarracenDataFrame)
        assert isinstance(sdf_sinks, SarracenDataFrame)
        assert sdf.params is not None
        assert sdf_sinks.params is not None
        assert sdf.params['massoftype'] == 1e-6
        assert sdf_sinks.params['massoftype'] == 1e-6
        assert sdf.params['mass'] == 1e-6
        assert 'mass' not in sdf_sinks.params
        assert 'mass' not in sdf.columns
        assert 'mass' not in sdf_sinks.columns
        tm.assert_series_equal(sdf['x'], pd.Series([0, 0, 0, 0, 1, 1, 1, 1]),
                               check_index=False, check_names=False,
                               check_dtype=False)
        tm.assert_series_equal(sdf_sinks['x'], pd.Series([0.000305]),
                               check_index=False, check_names=False,
                               check_dtype=False)
        tm.assert_series_equal(sdf_sinks['spinx'], pd.Series([-3.911744e-8]),
                               check_index=False, check_names=False,
                               check_dtype=False)

        sdf, sdf_sinks = sarracen.read_phantom(fp.name, separate_types='sinks')
        assert isinstance(sdf, SarracenDataFrame)
        assert isinstance(sdf_sinks, SarracenDataFrame)
        assert sdf.params is not None
        assert sdf_sinks.params is not None
        assert sdf.params['massoftype'] == 1e-6
        assert sdf_sinks.params['massoftype'] == 1e-6
        assert sdf.params['mass'] == 1e-6
        assert 'mass' not in sdf_sinks.params
        assert 'mass' not in sdf.columns
        assert 'mass' not in sdf_sinks.columns
        tm.assert_series_equal(sdf['x'], pd.Series([0, 0, 0, 0, 1, 1, 1, 1]),
                               check_index=False, check_names=False,
                               check_dtype=False)
        tm.assert_series_equal(sdf_sinks['x'], pd.Series([0.000305]),
                               check_index=False, check_names=False,
                               check_dtype=False)
        tm.assert_series_equal(sdf_sinks['spinx'], pd.Series([-3.911744e-8]),
                               check_index=False, check_names=False,
                               check_dtype=False)

        sdf = sarracen.read_phantom(fp.name, separate_types=None)
        assert isinstance(sdf, SarracenDataFrame)
        assert isinstance(sdf_sinks, SarracenDataFrame)
        assert sdf.params is not None
        assert sdf_sinks.params is not None
        assert sdf.params['massoftype'] == 1e-6
        assert sdf.params['mass'] == 1e-6
        assert 'mass' not in sdf.columns
        tm.assert_series_equal(sdf['x'], pd.Series([0, 0, 0, 0,
                                                    1, 1, 1, 1,
                                                    0.000305]),
                               check_index=False, check_names=False,
                               check_dtype=False)
        tm.assert_series_equal(sdf['h'], pd.Series([1.1, 1.1, 1.1, 1.1,
                                                    1.1, 1.1, 1.1, 1.1,
                                                    1.0]),
                               check_index=False, check_names=False,
                               check_dtype=False)


def test_gas_dust_sink_particles() -> None:

    file = _create_capture_pattern(np.int32, np.float64)
    file += _create_file_identifier()
    file += _create_global_header(massoftype_7=1e-4)

    # create 1 block for gas
    read_tag = np.array([13], dtype='int32')
    file += bytearray(read_tag.tobytes())
    nblocks = np.array([2], dtype='int32')
    file += bytearray(nblocks.tobytes())
    file += bytearray(read_tag.tobytes())

    # 8 particles storing 4 real arrays (x, y, z, h)
    file += bytearray(read_tag.tobytes())
    n = np.array([16], dtype='int64')
    nums = np.array([0, 1, 0, 0, 0, 4, 0, 0], dtype='int32')
    file += bytearray(n.tobytes())
    file += bytearray(nums.tobytes())
    file += bytearray(read_tag.tobytes())

    file += bytearray(read_tag.tobytes())
    n = np.array([1], dtype='int64')
    nums = np.array([0, 0, 0, 0, 0, 7, 0, 0], dtype='int32')
    file += bytearray(n.tobytes())
    file += bytearray(nums.tobytes())
    file += bytearray(read_tag.tobytes())

    # write 5 gas/dust particle arrays
    file += _create_particle_array("itype", [1, 1, 1, 1, 1, 1, 1, 1,
                                             7, 7, 7, 7, 7, 7, 7, 7], np.int8)
    file += _create_particle_array("x", [0, 0, 0, 0,
                                         1, 1, 1, 1,
                                         0.5, 0.5, 0.5, 0.5,
                                         1.5, 1.5, 1.5, 1.5])
    file += _create_particle_array("y", [0, 0, 1, 1,
                                         0, 0, 1, 1,
                                         0.5, 0.5, 1.5, 1.5,
                                         0.5, 0.5, 1.5, 1.5])
    file += _create_particle_array("z", [0, 1, 0, 1,
                                         0, 1, 0, 1,
                                         0.5, 1.5, 0.5, 1.5,
                                         0.5, 1.5, 0.5, 1.5])
    file += _create_particle_array("h", [1.1, 1.1, 1.1, 1.1,
                                         1.1, 1.1, 1.1, 1.1,
                                         1.1, 1.1, 1.1, 1.1,
                                         1.1, 1.1, 1.1, 1.1])

    # write 7 sink particle arrays
    file += _create_particle_array("x", [0.000305])
    file += _create_particle_array("y", [-0.035809])
    file += _create_particle_array("z", [-0.000035])
    file += _create_particle_array("h", [1.0])
    file += _create_particle_array("spinx", [-3.911744e-8])
    file += _create_particle_array("spiny", [-1.326062e-8])
    file += _create_particle_array("spinz", [0.00058])

    with tempfile.NamedTemporaryFile() as fp:
        fp.write(file)
        fp.seek(0)

        sdf_g, sdf_d, sdf_sinks = sarracen.read_phantom(fp.name,
                                                        separate_types='all')
        assert isinstance(sdf_g, SarracenDataFrame)
        assert isinstance(sdf_d, SarracenDataFrame)
        assert isinstance(sdf_sinks, SarracenDataFrame)
        assert sdf_g.params is not None
        assert sdf_d.params is not None
        assert sdf_sinks.params is not None
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
                               check_index=False, check_names=False,
                               check_dtype=False)
        tm.assert_series_equal(sdf_d['x'],
                               pd.Series([0.5, 0.5, 0.5, 0.5,
                                          1.5, 1.5, 1.5, 1.5]),
                               check_index=False, check_names=False,
                               check_dtype=False)
        tm.assert_series_equal(sdf_sinks['x'], pd.Series([0.000305]),
                               check_index=False, check_names=False,
                               check_dtype=False)
        tm.assert_series_equal(sdf_sinks['spinx'], pd.Series([-3.911744e-8]),
                               check_index=False, check_names=False,
                               check_dtype=False)

        sdf, sdf_sinks = sarracen.read_phantom(fp.name,
                                               separate_types='sinks')
        assert isinstance(sdf, SarracenDataFrame)
        assert isinstance(sdf_sinks, SarracenDataFrame)
        assert sdf.params is not None
        assert sdf_sinks.params is not None
        assert sdf.params['massoftype'] == 1e-6
        assert sdf.params['massoftype_7'] == 1e-4
        assert 'mass' not in sdf.params
        assert 'mass' in sdf.columns
        assert sdf[sdf.itype == 1]['mass'].unique() == [1e-6]
        assert sdf[sdf.itype == 7]['mass'].unique() == [1e-4]
        tm.assert_series_equal(sdf[sdf.itype == 1]['x'],
                               pd.Series([0, 0, 0, 0, 1, 1, 1, 1]),
                               check_index=False, check_names=False,
                               check_dtype=False)
        tm.assert_series_equal(sdf[sdf.itype == 7]['x'],
                               pd.Series([0.5, 0.5, 0.5, 0.5,
                                          1.5, 1.5, 1.5, 1.5]),
                               check_index=False, check_names=False,
                               check_dtype=False)
        assert sdf_sinks.params['massoftype'] == 1e-6
        assert sdf_sinks.params['massoftype_7'] == 1e-4
        assert 'mass' not in sdf_sinks.params
        assert 'mass' not in sdf_sinks.columns
        tm.assert_series_equal(sdf_sinks['x'], pd.Series([0.000305]),
                               check_index=False, check_names=False,
                               check_dtype=False)
        tm.assert_series_equal(sdf_sinks['spinx'], pd.Series([-3.911744e-8]),
                               check_index=False, check_names=False,
                               check_dtype=False)

        sdf = sarracen.read_phantom(fp.name, separate_types=None)
        assert isinstance(sdf, SarracenDataFrame)
        assert sdf.params is not None
        assert sdf.params['massoftype'] == 1e-6
        assert sdf.params['massoftype_7'] == 1e-4
        assert 'mass' not in sdf.params
        assert 'mass' in sdf.columns
        assert sdf[sdf.itype == 1]['mass'].unique() == [1e-6]
        assert sdf[sdf.itype == 7]['mass'].unique() == [1e-4]
        tm.assert_series_equal(sdf['x'], pd.Series([0, 0, 0, 0,
                                                    1, 1, 1, 1,
                                                    0.5, 0.5, 0.5, 0.5,
                                                    1.5, 1.5, 1.5, 1.5,
                                                    0.000305]),
                               check_index=False, check_names=False,
                               check_dtype=False)
        tm.assert_series_equal(sdf['h'], pd.Series([1.1] * 16 + [1.0]),
                               check_index=False, check_names=False,
                               check_dtype=False)
        tm.assert_series_equal(sdf['mass'], pd.Series([1e-6] * 8
                                                      + [1e-4] * 8
                                                      + [np.nan]),
                               check_index=False, check_names=False,
                               check_dtype=False)
