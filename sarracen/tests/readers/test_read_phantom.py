from typing import Type, Union, List, Tuple, Dict

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
                          mpi_blocks: int = 1,
                          swap_endian: bool = False) -> bytearray:
    """ Construct global variables. Only massoftype in this example. """

    dtypes = [def_int, np.int8, np.int16, np.int32, np.int64,
              def_real, np.float32, np.float64]
    param_dicts: List[Dict] = [dict() for _ in dtypes]

    params_def_int = param_dicts[0]
    params_def_real = param_dicts[5]

    params_def_real['massoftype'] = np.array([massoftype], dtype=def_real)
    if massoftype_7 is not None:
        params_def_real['massoftype_7'] = np.array([massoftype_7],
                                                   dtype=def_real)

    params_def_int['nblocks'] = np.array([mpi_blocks], dtype=def_int)

    dtype_param_pairs: List[Tuple[Type, Dict]] = list(zip(dtypes, param_dicts))

    read_tag = np.array([13], dtype='int32')
    if swap_endian:
        read_tag = read_tag.byteswap()
    file = bytearray()
    for dtype, params in dtype_param_pairs:
        nvars = np.array([len(params)], dtype='int32')
        if swap_endian:
            nvars = nvars.byteswap()
        file += bytearray(read_tag.tobytes())
        file += bytearray(nvars.tobytes())
        file += bytearray(read_tag.tobytes())

        if len(params) > 0:
            file += bytearray(read_tag.tobytes())
            for k in params.keys():
                file += bytearray(map(ord, k.ljust(16)))
            file += bytearray(read_tag.tobytes())

            file += bytearray(read_tag.tobytes())
            for v in params.values():
                v_np = np.array([v], dtype=dtype)
                if swap_endian:
                    v_np = v_np.byteswap()
                file += bytearray(v_np)
            file += bytearray(read_tag.tobytes())

    return file


def _create_particle_array(tag: str,
                           data: list,
                           dtype: Type[np.generic] = np.float64,
                           swap_endian: bool = False) -> bytearray:

    read_tag = np.array([13], dtype='int32')
    data_np = np.array(data, dtype=dtype)
    if swap_endian:
        read_tag = read_tag.byteswap()
        data_np = data_np.byteswap()

    file = bytearray(read_tag.tobytes())
    file += bytearray(map(ord, tag.ljust(16)))
    file += bytearray(read_tag.tobytes())
    file += bytearray(read_tag.tobytes())
    file += bytearray(data_np.tobytes())
    file += bytearray(read_tag.tobytes())
    return file


@pytest.mark.parametrize("def_int, def_real, swap_endian",
                         [(np.int32, np.float64, False),
                          (np.int32, np.float32, False),
                          (np.int64, np.float64, False),
                          (np.int64, np.float32, False),
                          (np.int32, np.float64, True),
                          (np.int32, np.float32, True),
                          (np.int64, np.float64, True),
                          (np.int64, np.float32, True)])
def test_determine_default_precision(def_int: Type[np.generic],
                                     def_real: Type[np.generic],
                                     swap_endian: bool) -> None:
    """ Test if default int / real precision can be determined. """

    file = _create_capture_pattern(def_int, def_real, swap_endian)
    file += _create_file_identifier(swap_endian)
    file += _create_global_header(def_int=def_int, def_real=def_real,
                                  swap_endian=swap_endian)

    # create 1 block for gas
    read_tag = np.array([13], dtype='int32')
    nblocks = np.array([1], dtype='int32')
    if swap_endian:
        read_tag = read_tag.byteswap()
        nblocks = nblocks.byteswap()
    file += bytearray(read_tag.tobytes())
    file += bytearray(nblocks.tobytes())
    file += bytearray(read_tag.tobytes())

    # 2 particles storing 1 default int and real arrays
    n = np.array([2], dtype='int64')
    nums = np.array([1, 0, 0, 0, 0, 1, 0, 0], dtype='int32')
    if swap_endian:
        n = n.byteswap()
        nums = nums.byteswap()
    file += bytearray(read_tag.tobytes())
    file += bytearray(n.tobytes())
    file += bytearray(nums.tobytes())
    file += bytearray(read_tag.tobytes())

    # write particle arrays
    file += _create_particle_array("def_int", [1, 2],
                                   def_int, swap_endian)
    file += _create_particle_array("def_real", [1.0, 2.0],
                                   def_real, swap_endian)

    with tempfile.NamedTemporaryFile() as fp:
        fp.write(file)
        fp.seek(0)

        sdf = sarracen.read_phantom(fp.name, separate_types=None)

        assert list(sdf.dtypes) == [def_int, def_real]


@pytest.mark.parametrize("mpi_blocks, swap_endian",
                         [(1, False), (1, True),
                          (2, False), (2, True),
                          (4, False), (4, True)])
def test_gas_particles_only(mpi_blocks: int, swap_endian: bool) -> None:

    def_int = np.int32
    def_real = np.float64
    file = _create_capture_pattern(def_int, def_real, swap_endian)
    file += _create_file_identifier(swap_endian)
    file += _create_global_header(mpi_blocks=mpi_blocks,
                                  swap_endian=swap_endian)

    # create block for gas (broken into number of mpi_blocks)
    read_tag = np.array([13], dtype='int32')
    nblocks = np.array([mpi_blocks], dtype='int32')
    if swap_endian:
        read_tag = read_tag.byteswap()
        nblocks = nblocks.byteswap()
    file += bytearray(read_tag.tobytes())
    file += bytearray(nblocks.tobytes())
    file += bytearray(read_tag.tobytes())

    x = [0, 0, 0, 0, 1, 1, 1, 1]
    y = [0, 0, 1, 1, 0, 0, 1, 1]
    z = [0, 1, 0, 1, 0, 1, 0, 1]
    h = [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1]

    for i in range(mpi_blocks):
        # block header
        # 8 particles storing 4 real arrays (x, y, z, h)
        n = np.array([8 / mpi_blocks], dtype='int64')
        nums = np.array([0, 0, 0, 0, 0, 4, 0, 0], dtype='int32')
        if swap_endian:
            n = n.byteswap()
            nums = nums.byteswap()
        file += bytearray(read_tag.tobytes())
        file += bytearray(n.tobytes())
        file += bytearray(nums.tobytes())
        file += bytearray(read_tag.tobytes())

        # block particle arrays
        # each mpi_block writes a chunk of the array
        size = len(x) // mpi_blocks
        start = i * size
        end = (i + 1) * size
        file += _create_particle_array("x", x[start:end],
                                       def_real, swap_endian)
        file += _create_particle_array("y", y[start:end],
                                       def_real, swap_endian)
        file += _create_particle_array("z", z[start:end],
                                       def_real, swap_endian)
        file += _create_particle_array("h", h[start:end],
                                       def_real, swap_endian)

    with tempfile.NamedTemporaryFile() as fp:
        fp.write(file)
        fp.seek(0)

        sdf = sarracen.read_phantom(fp.name, separate_types='all')
        assert isinstance(sdf, SarracenDataFrame)
        assert sdf.params is not None
        assert sdf.params['massoftype'] == 1e-6
        assert sdf.params['mass'] == 1e-6
        assert 'mass' not in sdf.columns
        tm.assert_series_equal(sdf['x'], pd.Series(x),
                               check_index=False, check_names=False,
                               check_dtype=False)

        sdf = sarracen.read_phantom(fp.name, separate_types='sinks')
        assert isinstance(sdf, SarracenDataFrame)
        assert sdf.params is not None
        assert sdf.params['massoftype'] == 1e-6
        assert sdf.params['mass'] == 1e-6
        assert 'mass' not in sdf.columns
        tm.assert_series_equal(sdf['x'], pd.Series(x),
                               check_index=False, check_names=False,
                               check_dtype=False)

        sdf = sarracen.read_phantom(fp.name, separate_types=None)
        assert isinstance(sdf, SarracenDataFrame)
        assert sdf.params is not None
        assert sdf.params['massoftype'] == 1e-6
        assert sdf.params['mass'] == 1e-6
        assert 'mass' not in sdf.columns
        tm.assert_series_equal(sdf['x'], pd.Series(x),
                               check_index=False, check_names=False,
                               check_dtype=False)


@pytest.mark.parametrize("mpi_blocks, swap_endian",
                         [(1, False), (1, True),
                          (2, False), (2, True),
                          (4, False), (4, True)])
def test_gas_dust_particles(mpi_blocks: int, swap_endian: bool) -> None:

    def_int = np.int32
    def_real = np.float64
    file = _create_capture_pattern(def_int, def_real, swap_endian)
    file += _create_file_identifier(swap_endian)
    file += _create_global_header(massoftype_7=1e-4, mpi_blocks=mpi_blocks,
                                  swap_endian=swap_endian)

    # create block for gas & dust particles
    read_tag = np.array([13], dtype='int32')
    nblocks = np.array([mpi_blocks], dtype='int32')
    if swap_endian:
        read_tag = read_tag.byteswap()
        nblocks = nblocks.byteswap()
    file += bytearray(read_tag.tobytes())
    file += bytearray(nblocks.tobytes())
    file += bytearray(read_tag.tobytes())

    itype = [1, 1, 1, 1, 1, 1, 1, 1,
             7, 7, 7, 7, 7, 7, 7, 7]
    x = [0, 0, 0, 0, 1, 1, 1, 1,
         0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5]
    y = [0, 0, 1, 1, 0, 0, 1, 1,
         0.5, 0.5, 1.5, 1.5, 0.5, 0.5, 1.5, 1.5]
    z = [0, 1, 0, 1, 0, 1, 0, 1,
         0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5]
    h = [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1,
         1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1]

    for i in range(mpi_blocks):
        # block header
        # 8 particles storing 4 real arrays (x, y, z, h)
        n = np.array([16 / mpi_blocks], dtype='int64')
        nums = np.array([0, 1, 0, 0, 0, 4, 0, 0], dtype='int32')
        if swap_endian:
            n = n.byteswap()
            nums = nums.byteswap()
        file += bytearray(read_tag.tobytes())
        file += bytearray(n.tobytes())
        file += bytearray(nums.tobytes())
        file += bytearray(read_tag.tobytes())

        # block particle arrays
        # each mpi_block writes a chunk of the array
        size = len(x) // mpi_blocks
        start = i * size
        end = (i + 1) * size
        file += _create_particle_array("itype", itype[start:end],
                                       np.int8, swap_endian)
        file += _create_particle_array("x", x[start:end],
                                       def_real, swap_endian)
        file += _create_particle_array("y", y[start:end],
                                       def_real, swap_endian)
        file += _create_particle_array("z", z[start:end],
                                       def_real, swap_endian)
        file += _create_particle_array("h", h[start:end],
                                       def_real, swap_endian)

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


@pytest.mark.parametrize("mpi_blocks, swap_endian",
                         [(1, False), (1, True),
                          (2, False), (2, True),
                          (4, False), (4, True)])
def test_gas_sink_particles(mpi_blocks: int, swap_endian: bool) -> None:

    def_int = np.int32
    def_real = np.float64
    file = _create_capture_pattern(def_int, def_real, swap_endian)
    file += _create_file_identifier(swap_endian)
    file += _create_global_header(mpi_blocks=mpi_blocks,
                                  swap_endian=swap_endian)

    # block 1 = gas, block 2 = sinks
    read_tag = np.array([13], dtype='int32')
    nblocks = np.array([2 * mpi_blocks], dtype='int32')
    if swap_endian:
        read_tag = read_tag.byteswap()
        nblocks = nblocks.byteswap()
    file += bytearray(read_tag.tobytes())
    file += bytearray(nblocks.tobytes())
    file += bytearray(read_tag.tobytes())

    x = [0, 0, 0, 0, 1, 1, 1, 1]
    y = [0, 0, 1, 1, 0, 0, 1, 1]
    z = [0, 1, 0, 1, 0, 1, 0, 1]
    h = [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1]

    for i in range(mpi_blocks):
        # block headers
        n = np.array([8 / mpi_blocks], dtype='int64')
        nums = np.array([0, 0, 0, 0, 0, 4, 0, 0], dtype='int32')
        if swap_endian:
            n = n.byteswap()
            nums = nums.byteswap()
        file += bytearray(read_tag.tobytes())
        file += bytearray(n.tobytes())
        file += bytearray(nums.tobytes())
        file += bytearray(read_tag.tobytes())

        n = np.array([1], dtype='int64')
        nums = np.array([0, 0, 0, 0, 0, 7, 0, 0], dtype='int32')
        if swap_endian:
            n = n.byteswap()
            nums = nums.byteswap()
        file += bytearray(read_tag.tobytes())
        file += bytearray(n.tobytes())
        file += bytearray(nums.tobytes())
        file += bytearray(read_tag.tobytes())

        # write 4 gas particle arrays in block 1
        # each mpi_block writes a chunk of the array
        size = len(x) // mpi_blocks
        start = i * size
        end = (i + 1) * size
        file += _create_particle_array("x", x[start:end],
                                       def_real, swap_endian)
        file += _create_particle_array("y", y[start:end],
                                       def_real, swap_endian)
        file += _create_particle_array("z", z[start:end],
                                       def_real, swap_endian)
        file += _create_particle_array("h", h[start:end],
                                       def_real, swap_endian)

        # write 7 sink particle arrays in block 2
        # each mpi_block writes all sink particles (I believe)
        file += _create_particle_array("x", [0.000305],
                                       def_real, swap_endian)
        file += _create_particle_array("y", [-0.035809],
                                       def_real, swap_endian)
        file += _create_particle_array("z", [-0.000035],
                                       def_real, swap_endian)
        file += _create_particle_array("h", [1.0],
                                       def_real, swap_endian)
        file += _create_particle_array("spinx", [-3.911744e-8],
                                       def_real, swap_endian)
        file += _create_particle_array("spiny", [-1.326062e-8],
                                       def_real, swap_endian)
        file += _create_particle_array("spinz", [0.00058],
                                       def_real, swap_endian)

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
        tm.assert_series_equal(sdf['x'], pd.Series(x),
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
        tm.assert_series_equal(sdf['x'], pd.Series(x),
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
        tm.assert_series_equal(sdf['x'], pd.Series(x + [0.000305]),
                               check_index=False, check_names=False,
                               check_dtype=False)
        tm.assert_series_equal(sdf['h'], pd.Series(h + [1.0]),
                               check_index=False, check_names=False,
                               check_dtype=False)


@pytest.mark.parametrize("mpi_blocks, swap_endian",
                         [(1, False), (1, True),
                          (2, False), (2, True),
                          (4, False), (4, True)])
def test_gas_dust_sink_particles(mpi_blocks: int, swap_endian: bool) -> None:

    def_int = np.int32
    def_real = np.float64
    file = _create_capture_pattern(def_int, def_real, swap_endian)
    file += _create_file_identifier(swap_endian)
    file += _create_global_header(massoftype_7=1e-4, mpi_blocks=mpi_blocks,
                                  swap_endian=swap_endian)

    # create 1 block for gas
    read_tag = np.array([13], dtype='int32')
    nblocks = np.array([2 * mpi_blocks], dtype='int32')
    if swap_endian:
        read_tag = read_tag.byteswap()
        nblocks = nblocks.byteswap()
    file += bytearray(read_tag.tobytes())
    file += bytearray(nblocks.tobytes())
    file += bytearray(read_tag.tobytes())

    itype = [1, 1, 1, 1, 1, 1, 1, 1,
             7, 7, 7, 7, 7, 7, 7, 7]
    x = [0, 0, 0, 0, 1, 1, 1, 1,
         0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5]
    y = [0, 0, 1, 1, 0, 0, 1, 1,
         0.5, 0.5, 1.5, 1.5, 0.5, 0.5, 1.5, 1.5]
    z = [0, 1, 0, 1, 0, 1, 0, 1,
         0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5]
    h = [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1,
         1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1]

    for i in range(mpi_blocks):
        # block headers
        # gas/dust particles in block 1
        n = np.array([16 / mpi_blocks], dtype='int64')
        nums = np.array([0, 1, 0, 0, 0, 4, 0, 0], dtype='int32')
        if swap_endian:
            n = n.byteswap()
            nums = nums.byteswap()
        file += bytearray(read_tag.tobytes())
        file += bytearray(n.tobytes())
        file += bytearray(nums.tobytes())
        file += bytearray(read_tag.tobytes())

        # 1 sink particle in block 2
        n = np.array([1], dtype='int64')
        nums = np.array([0, 0, 0, 0, 0, 7, 0, 0], dtype='int32')
        if swap_endian:
            n = n.byteswap()
            nums = nums.byteswap()
        file += bytearray(read_tag.tobytes())
        file += bytearray(n.tobytes())
        file += bytearray(nums.tobytes())
        file += bytearray(read_tag.tobytes())

        # write gas/dust particle arrays in block 1
        # each mpi_block writes a chunk of the array
        size = len(x) // mpi_blocks
        start = i * size
        end = (i + 1) * size
        file += _create_particle_array("itype", itype[start:end],
                                       np.int8, swap_endian)
        file += _create_particle_array("x", x[start:end],
                                       def_real, swap_endian)
        file += _create_particle_array("y", y[start:end],
                                       def_real, swap_endian)
        file += _create_particle_array("z", z[start:end],
                                       def_real, swap_endian)
        file += _create_particle_array("h", h[start:end],
                                       def_real, swap_endian)

        # write 7 sink particle arrays in block 2
        # each mpi_block writes all sink particles (I believe)
        file += _create_particle_array("x", [0.000305],
                                       def_real, swap_endian)
        file += _create_particle_array("y", [-0.035809],
                                       def_real, swap_endian)
        file += _create_particle_array("z", [-0.000035],
                                       def_real, swap_endian)
        file += _create_particle_array("h", [1.0],
                                       def_real, swap_endian)
        file += _create_particle_array("spinx", [-3.911744e-8],
                                       def_real, swap_endian)
        file += _create_particle_array("spiny", [-1.326062e-8],
                                       def_real, swap_endian)
        file += _create_particle_array("spinz", [0.00058],
                                       def_real, swap_endian)

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
