from typing import Dict, Union, Any

import pandas as pd
import numpy as np
import sarracen
import tempfile
from pytest import fixture, mark, raises
from pandas import testing as tm

from sarracen import SarracenDataFrame

from sarracen.writers.write_phantom import (_check_for_essential_data,
                                            _standardize_dtypes,
                                            _validate_ntypes,
                                            _reorder_params,
                                            _validate_particle_counts,
                                            _validate_particle_masses)


@fixture
def particles_df() -> pd.DataFrame:
    x = [0, 0, 0, 0, 1, 1, 1, 1]
    y = [0, 0, 1, 1, 0, 0, 1, 1]
    z = [0, 1, 0, 1, 0, 1, 0, 1]
    h = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
    vx = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    vy = [-0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9]
    vz = [0.01, -0.01, 0.01, -0.01, 0.01, -0.01, 0.01, -0.01]

    return pd.DataFrame({'x': x, 'y': y, 'z': z, 'h': h,
                         'vx': vx, 'vy': vy, 'vz': vz})


def test_check_for_essential_data(particles_df: pd.DataFrame) -> None:
    """ Test that essential data is present in the dataframe."""

    sdf = SarracenDataFrame(particles_df)
    sdf = sdf.drop(columns=['h'])
    with raises(ValueError, match='columns'):
        _check_for_essential_data(sdf)

    sdf = SarracenDataFrame(particles_df)
    sdf = sdf.drop(columns=['z'])
    with raises(ValueError, match='columns'):
        _check_for_essential_data(sdf)


def test_params_dtype_standardization(particles_df: pd.DataFrame) -> None:
    """ Test that params dtype is correctly standardized."""

    params: Dict[str, Any] = {'massoftype': 1e-4,
                              'ntypes': 8,
                              'nparttot': 8,
                              'massoftype_7': 1e-6,
                              'file_identifier': 'test Phantom write'}

    params = _standardize_dtypes(params)

    assert isinstance(params['massoftype'], np.float64)
    assert isinstance(params['massoftype_7'], np.float64)
    assert isinstance(params['ntypes'], np.int32)
    assert isinstance(params['nparttot'], np.int32)
    assert isinstance(params['file_identifier'], str)


@mark.parametrize("id_method", ('ntypes', 'massoftype',
                                'npartoftype', 'itype'))
def test_validate_ntypes_12(particles_df: pd.DataFrame,
                            id_method: str) -> None:
    """ Test that number of particle types is correctly validated."""

    params: Dict[str, Union[np.generic, str]] = {}

    if id_method == 'ntypes':
        params['ntypes'] = np.int32(12)
    elif id_method == 'massoftype':
        params['massoftype'] = np.float64(1e-4)
        params['massoftype_7'] = np.float64(1e-6)
        params['massoftype_12'] = np.float64(1e-7)
    elif id_method == 'npartoftype':
        params['npartoftype'] = np.int32(10)
        params['npartoftype_7'] = np.int32(2)
        params['npartoftype_12'] = np.int32(6)
        params['npartoftype_13'] = np.int64(10)
        params['npartoftype_20'] = np.int64(2)
        params['npartoftype_24'] = np.int64(6)
    elif id_method == 'itype':
        particles_df['itype'] = [1, 1, 1, 1, 1, 7, 7, 12]

    sdf = sarracen.SarracenDataFrame(particles_df, params)

    params = _validate_ntypes(sdf, params)

    assert params['ntypes'] == 12


@mark.parametrize("id_method", ('base', 'ntypes', 'massoftype',
                                'npartoftype', 'itype'))
def test_validate_ntypes_8(particles_df: pd.DataFrame,
                           id_method: str) -> None:
    """ Test that number of particle types is correctly validated."""

    params: Dict[str, Union[np.generic, str]] = {}

    if id_method == 'ntypes':
        params['ntypes'] = np.int32(8)
    elif id_method == 'massoftype':
        params['massoftype'] = np.float64(1e-4)
        params['massoftype_7'] = np.float64(1e-6)
    elif id_method == 'npartoftype':
        params['npartoftype'] = np.int32(10)
        params['npartoftype_7'] = np.int32(2)
        params['npartoftype_12'] = np.int64(6)
        params['npartoftype_13'] = np.int64(10)
    elif id_method == 'itype':
        particles_df['itype'] = [1, 1, 1, 1, 1, 7, 7, 7]

    sdf = sarracen.SarracenDataFrame(particles_df, params)

    params = _validate_ntypes(sdf, params)

    assert params['ntypes'] == 8


@mark.parametrize("dust, id_method",
                  [(False, 'itype'), (False, 'npartoftype'),
                   (True, 'itype'), (True, 'npartoftype')])
def test_validate_particle_counts(particles_df: pd.DataFrame,
                                  dust: bool,
                                  id_method: str) -> None:
    """ Test that number of each particle type is correctly autofilled."""

    params: Dict[str,
                 Union[np.generic,
                       str]] = {'massoftype': np.float64(1e-4),
                                'ntypes': np.int32(8),
                                'file_identifier': 'test Phantom write'}

    # if npartoftype not present, then it determines based on itype alone
    if id_method == 'npartoftype':
        params['npartoftype'] = np.int32(5) if dust else np.int32(8)
        params['npartoftype_7'] = np.int32(3) if dust else np.int32(0)

    if dust:
        particles_df['itype'] = [1, 1, 1, 1, 1, 7, 7, 7]

    sdf = sarracen.SarracenDataFrame(particles_df, params)

    params = _validate_particle_counts(sdf, params)

    assert params['nparttot'] == 8
    assert params['npartoftype'] == 5 if dust else 8
    assert params['npartoftype_2'] == 0
    assert params['npartoftype_3'] == 0
    assert params['npartoftype_4'] == 0
    assert params['npartoftype_5'] == 0
    assert params['npartoftype_6'] == 0
    assert params['npartoftype_7'] == 3 if dust else 8
    assert params['npartoftype_8'] == 0
    assert params['nparttot_2'] == 8
    assert params['npartoftype_9'] == 5 if dust else 8
    assert params['npartoftype_10'] == 0
    assert params['npartoftype_11'] == 0
    assert params['npartoftype_12'] == 0
    assert params['npartoftype_13'] == 0
    assert params['npartoftype_14'] == 0
    assert params['npartoftype_15'] == 3 if dust else 8
    assert params['npartoftype_16'] == 0


def test_params_reordering(particles_df: pd.DataFrame) -> None:
    """ Test that disordered params keyes are written correctly."""

    params: Dict[str, Union[np.generic,
                            str]] = {'massoftype': np.float64(1e-4),
                                     'massoftype_7': np.float64(1e-6),
                                     'massoftype_3': np.float64(1e-3),
                                     'nparttot': np.int64(8),
                                     'nparttot_2': np.int32(8),
                                     'npartoftype_15': np.int64(3),
                                     'npartoftype_3': np.int32(4),
                                     'npartoftype': np.int32(1),
                                     'npartoftype_7': np.int32(3),
                                     'npartoftype_11': np.int64(4),
                                     'npartoftype_9': np.int64(1),
                                     'file_identifier': 'test Phantom write'}

    params = _reorder_params(params)

    keys = list(params.keys())

    assert keys.index('massoftype') < keys.index('massoftype_3')
    assert keys.index('massoftype_3') < keys.index('massoftype_7')
    assert keys.index('nparttot') < keys.index('nparttot_2')
    assert keys.index('npartoftype') < keys.index('npartoftype_3')
    assert keys.index('npartoftype_3') < keys.index('npartoftype_7')
    assert keys.index('npartoftype_7') < keys.index('npartoftype_9')
    assert keys.index('npartoftype_9') < keys.index('npartoftype_11')
    assert keys.index('npartoftype_11') < keys.index('npartoftype_15')

    assert params['massoftype'] == 1e-4
    assert params['massoftype_3'] == 1e-3
    assert params['massoftype_7'] == 1e-6
    assert params['nparttot'] == 8
    assert params['nparttot_2'] == 8
    assert params['npartoftype'] == 1
    assert params['npartoftype_3'] == 4
    assert params['npartoftype_7'] == 3
    assert params['npartoftype_9'] == 1
    assert params['npartoftype_11'] == 4
    assert params['npartoftype_15'] == 3


@mark.parametrize("id_method", ('params_mass', 'massoftype', 'mcol'))
def test_validate_particle_mass_gas(particles_df: pd.DataFrame,
                                    id_method: bool) -> None:
    """ Test that mass of each particle type is correctly validated."""

    params: Dict[str, Union[np.generic, str]] = {'ntypes': np.int32(8)}

    if id_method == 'params_mass':
        params['mass'] = np.float64(1e-4)
    elif id_method == 'massoftype':
        params['massoftype'] = np.float64(1e-4)
    elif id_method == 'mcol':
        particles_df['m'] = np.float64(1e-4)

    sdf = SarracenDataFrame(particles_df, params)

    params = _validate_particle_masses(sdf, params)

    assert params['massoftype'] == 1e-4
    assert params['massoftype_2'] == 0
    assert params['massoftype_3'] == 0
    assert params['massoftype_4'] == 0
    assert params['massoftype_5'] == 0
    assert params['massoftype_6'] == 0
    assert params['massoftype_7'] == 0
    assert params['massoftype_8'] == 0

    # with tempfile.NamedTemporaryFile() as fp:
    #     sarracen.write_phantom(fp.name, write_sdf)
    #     sdf = sarracen.read_phantom(fp.name)
    #
    #     assert isinstance(sdf, SarracenDataFrame)
    #     assert sdf.params is not None
    #     assert sdf.params['mass'] == 1e-4
    #     assert sdf.params['massoftype'] == 1e-4


@mark.parametrize("id_method", ('massoftype', 'mcol'))
def test_validate_particle_mass_dust(particles_df: pd.DataFrame,
                                     id_method: bool) -> None:
    """ Test that mass of each particle type is correctly validated."""

    params: Dict[str, Union[np.generic, str]] = {'ntypes': np.int32(8)}

    if id_method == 'massoftype':
        params['massoftype'] = np.float64(1e-4)
        params['massoftype_7'] = np.float64(1e-6)
    elif id_method == 'mcol':
        particles_df['m'] = [1e-4] * 5 + [1e-6] * 3

    sdf = SarracenDataFrame(particles_df, params)
    sdf['itype'] = [1, 1, 1, 1, 1, 7, 7, 7]

    params = _validate_particle_masses(sdf, params)

    assert params['massoftype'] == 1e-4
    assert params['massoftype_2'] == 0
    assert params['massoftype_3'] == 0
    assert params['massoftype_4'] == 0
    assert params['massoftype_5'] == 0
    assert params['massoftype_6'] == 0
    assert params['massoftype_7'] == 1e-6
    assert params['massoftype_8'] == 0


def test_write_gas(particles_df: pd.DataFrame) -> None:
    """ Test writing of simple gas-only particle dumpfile."""

    params = {'massoftype': np.float64(1e-4),
              'iexternalforce': np.int32(0),
              'udist': np.float64(2e-3),
              'utime': np.float64(2e-5),
              'umass': np.float64(2e-6),
              'umagfd': np.float64(2e-2),
              'file_identifier': 'test of Phantom writing'}

    write_sdf = SarracenDataFrame(particles_df, params)

    with tempfile.NamedTemporaryFile() as fp:
        sarracen.write_phantom(fp.name, write_sdf)
        sdf = sarracen.read_phantom(fp.name)

        assert isinstance(sdf, SarracenDataFrame)
        assert sdf.params is not None
        assert sdf.params['massoftype'] == 1e-4
        assert sdf.params['mass'] == 1e-4
        assert sdf.params['udist'] == 2e-3
        assert 'mass' in sdf.params
        assert 'mass' not in sdf.columns
        tm.assert_series_equal(sdf['x'], write_sdf['x'],
                               check_index=False, check_names=False,
                               check_dtype=False)
        tm.assert_series_equal(sdf['y'], write_sdf['y'],
                               check_index=False, check_names=False,
                               check_dtype=False)
        tm.assert_series_equal(sdf['z'], write_sdf['z'],
                               check_index=False, check_names=False,
                               check_dtype=False)
        tm.assert_series_equal(sdf['h'], write_sdf['h'],
                               check_index=False, check_names=False,
                               check_dtype=False)
        tm.assert_series_equal(sdf['vx'], write_sdf['vx'],
                               check_index=False, check_names=False,
                               check_dtype=False)
        tm.assert_series_equal(sdf['vy'], write_sdf['vy'],
                               check_index=False, check_names=False,
                               check_dtype=False)
        tm.assert_series_equal(sdf['vz'], write_sdf['vz'],
                               check_index=False, check_names=False,
                               check_dtype=False)


def test_write_gas_and_dust(particles_df: pd.DataFrame) -> None:
    """ Test writing of simple gas and dust-only dumpfile."""

    params = {'massoftype': np.float64(1e-4),
              'massoftype_2': np.float64(0),
              'massoftype_3': np.float64(0),
              'massoftype_4': np.float64(0),
              'massoftype_5': np.float64(0),
              'massoftype_7': np.float64(1e-6),
              'iexternalforce': np.int32(0),
              'udist': np.float64(2e-3),
              'utime': np.float64(2e-5),
              'umass': np.float64(2e-6),
              'umagfd': np.float64(2e-2),
              'file_identifier': 'test of Phantom writing'}

    write_sdf = SarracenDataFrame(particles_df, params)

    write_sdf['itype'] = [1, 1, 1, 1, 1, 7, 7, 7]

    with tempfile.NamedTemporaryFile() as fp:
        sarracen.write_phantom(fp.name, write_sdf)

        sdf = sarracen.read_phantom(fp.name, separate_types=None)
        assert isinstance(sdf, SarracenDataFrame)
        assert sdf.params is not None
        assert sdf.params['massoftype'] == 1e-4
        assert sdf.params['massoftype_7'] == 1e-6
        assert sdf.params['udist'] == 2e-3
        assert 'mass' not in sdf.params
        assert 'mass' in sdf.columns

        mass = [1e-4] * 5 + [1e-6] * 3
        tm.assert_series_equal(sdf['mass'], pd.Series(mass),
                               check_index=False, check_names=False,
                               check_dtype=False)
        tm.assert_series_equal(sdf['itype'], write_sdf['itype'],
                               check_index=False, check_names=False,
                               check_dtype=False)
        tm.assert_series_equal(sdf['x'], write_sdf['x'],
                               check_index=False, check_names=False,
                               check_dtype=False)
        tm.assert_series_equal(sdf['y'], write_sdf['y'],
                               check_index=False, check_names=False,
                               check_dtype=False)
        tm.assert_series_equal(sdf['z'], write_sdf['z'],
                               check_index=False, check_names=False,
                               check_dtype=False)
        tm.assert_series_equal(sdf['h'], write_sdf['h'],
                               check_index=False, check_names=False,
                               check_dtype=False)
        tm.assert_series_equal(sdf['vx'], write_sdf['vx'],
                               check_index=False, check_names=False,
                               check_dtype=False)
        tm.assert_series_equal(sdf['vy'], write_sdf['vy'],
                               check_index=False, check_names=False,
                               check_dtype=False)
        tm.assert_series_equal(sdf['vz'], write_sdf['vz'],
                               check_index=False, check_names=False,
                               check_dtype=False)


def test_write_gas_and_sink(particles_df: pd.DataFrame) -> None:
    """ Test writing of simple gas-only particle dumpfile."""

    params = {'massoftype': np.float64(1e-4),
              'iexternalforce': np.int32(0),
              'udist': np.float64(2e-3),
              'utime': np.float64(2e-5),
              'umass': np.float64(2e-6),
              'umagfd': np.float64(2e-2),
              'file_identifier': 'test of Phantom writing'}

    write_sdf = SarracenDataFrame(particles_df, params)

    sink_x = [0.5, 0.5]
    sink_y = [0.3, 0.3]
    sink_z = [0.1, 0.1]
    sink_m = [0.05, 0.001]
    sink_spinx = [-0.002, -0.002]
    sink_spiny = [1e-7, 1e-6]
    sink_spinz = [0.002, 0.002]

    sink_df = pd.DataFrame({'x': sink_x, 'y': sink_y, 'z': sink_z,
                            'm': sink_m, 'spinx': sink_spinx,
                            'spiny': sink_spiny, 'spinz': sink_spinz})

    write_sdf_sinks = SarracenDataFrame(sink_df, params)

    with tempfile.NamedTemporaryFile() as fp:
        sarracen.write_phantom(fp.name, write_sdf, write_sdf_sinks)
        sdf, sdf_sinks = sarracen.read_phantom(fp.name)

        assert isinstance(sdf, SarracenDataFrame)
        assert sdf.params is not None
        assert sdf.params['massoftype'] == 1e-4
        assert sdf.params['mass'] == 1e-4
        assert sdf.params['udist'] == 2e-3
        assert 'mass' in sdf.params
        assert 'mass' not in sdf.columns
        tm.assert_series_equal(sdf['x'], write_sdf['x'],
                               check_index=False, check_names=False,
                               check_dtype=False)
        tm.assert_series_equal(sdf['y'], write_sdf['y'],
                               check_index=False, check_names=False,
                               check_dtype=False)
        tm.assert_series_equal(sdf['z'], write_sdf['z'],
                               check_index=False, check_names=False,
                               check_dtype=False)
        tm.assert_series_equal(sdf['h'], write_sdf['h'],
                               check_index=False, check_names=False,
                               check_dtype=False)
        tm.assert_series_equal(sdf['vx'], write_sdf['vx'],
                               check_index=False, check_names=False,
                               check_dtype=False)
        tm.assert_series_equal(sdf['vy'], write_sdf['vy'],
                               check_index=False, check_names=False,
                               check_dtype=False)
        tm.assert_series_equal(sdf['vz'], write_sdf['vz'],
                               check_index=False, check_names=False,
                               check_dtype=False)

        assert isinstance(sdf_sinks, SarracenDataFrame)
        tm.assert_series_equal(sdf_sinks['x'], write_sdf_sinks['x'],
                               check_index=False, check_names=False,
                               check_dtype=False)
        tm.assert_series_equal(sdf_sinks['y'], write_sdf_sinks['y'],
                               check_index=False, check_names=False,
                               check_dtype=False)
        tm.assert_series_equal(sdf_sinks['z'], write_sdf_sinks['z'],
                               check_index=False, check_names=False,
                               check_dtype=False)
        tm.assert_series_equal(sdf_sinks['m'], write_sdf_sinks['m'],
                               check_index=False, check_names=False,
                               check_dtype=False)
        tm.assert_series_equal(sdf_sinks['spinx'], write_sdf_sinks['spinx'],
                               check_index=False, check_names=False,
                               check_dtype=False)
        tm.assert_series_equal(sdf_sinks['spiny'], write_sdf_sinks['spiny'],
                               check_index=False, check_names=False,
                               check_dtype=False)
        tm.assert_series_equal(sdf_sinks['spinz'], write_sdf_sinks['spinz'],
                               check_index=False, check_names=False,
                               check_dtype=False)
