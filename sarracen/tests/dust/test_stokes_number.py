import pandas as pd
import numpy as np
from numpy.testing import assert_allclose
from pytest import approx, raises, mark

from sarracen import SarracenDataFrame
from sarracen.dust import getting_dust_locations, inverting_dust_locations

def test_getting_dust_locations():
    datag = {'x': [2, 8, 3], 'y': [8, 5, 2], 'h': [0.5, 0.5, 1]}
    datad = {'x': [2, 4, 9], 'y': [2, 8, 5], 'h': [0.25, 1.5, 1]}
    sdfg = SarracenDataFrame(datag, params=dict())
    sdfd = SarracenDataFrame(datad, params=dict())
    gas_positions = sdfg[["x", "y"]].values
    dust_positions = sdfd[["x", "y"]].values
    h_gas_data = sdfg["h"].values
    h_dust_data = sdfd["h"].values

    dust_locations1 = getting_dust_locations(dust_positions, gas_positions, h_gas_data)
    assert len(dust_locations1) == len(sdfg)
    assert len(dust_locations1[0]) == 0
    assert len(dust_locations1[1]) == 1
    assert len(dust_locations1[2]) == 1
    assert dust_locations1[1][0] == 2
    assert dust_locations1[2][0] == 0

    dust_locations2 = getting_dust_locations(gas_positions, dust_positions, h_dust_data)
    assert len(dust_locations2) == len(sdfd)
    assert len(dust_locations2[0]) == 0
    assert len(dust_locations2[1]) == 1
    assert len(dust_locations2[2]) == 1
    assert dust_locations2[1][0] == 0
    assert dust_locations2[2][0] == 1

def test_getting_dust_locations_diff_size():
    datag = {'x': [2, 8, 3], 'y': [8, 5, 2], 'h': [0.5, 0.5, 1]}
    datad = {'x': [2, 4, 9, 2.1], 'y': [2, 8, 5, 2], 'h': [0.25, 1.5, 1, 0.2]}
    sdfg = SarracenDataFrame(datag, params=dict())
    sdfd = SarracenDataFrame(datad, params=dict())
    gas_positions = sdfg[["x", "y"]].values
    dust_positions = sdfd[["x", "y"]].values
    h_gas_data = sdfg["h"].values
    h_dust_data = sdfd["h"].values

    dust_locations1 = getting_dust_locations(dust_positions, gas_positions, h_gas_data)
    assert len(dust_locations1) == len(sdfg)
    assert len(dust_locations1[0]) == 0
    assert len(dust_locations1[1]) == 1
    assert len(dust_locations1[2]) == 2
    assert dust_locations1[1][0] == 2
    assert dust_locations1[2][0] == 0
    assert dust_locations1[2][1] == 3

    dust_locations2 = getting_dust_locations(gas_positions, dust_positions, h_dust_data)
    assert len(dust_locations2) == len(sdfd)
    assert len(dust_locations2[0]) == 0
    assert len(dust_locations2[1]) == 1
    assert len(dust_locations2[2]) == 1
    assert len(dust_locations1[3]) == 0
    assert dust_locations2[1][0] == 0
    assert dust_locations2[2][0] == 1

def test_inverting_dust_locations():
    datag = {'x': [2, 8, 3], 'y': [8, 5, 2], 'h': [0.5, 0.5, 1]}
    datad = {'x': [2, 4, 9], 'y': [2, 8, 5], 'h': [0.25, 1.5, 1]}
    sdfg = SarracenDataFrame(datag, params=dict())
    sdfd = SarracenDataFrame(datad, params=dict())
    gas_positions = sdfg[["x", "y"]].values
    dust_positions = sdfd[["x", "y"]].values
    h_gas_data = sdfg["h"].values
    h_dust_data = sdfd["h"].values

    # The result from getting_dust_locations(dust_positions, gas_positions, h_gas_data)
    dust_locations1 = [[], [2], [0]]
    dust_locations1_inverse = inverting_dust_locations(dust_positions, dust_locations1)
    assert len(dust_locations1_inverse) == len(sdfd)
    assert len(dust_locations1_inverse[0]) == 1
    assert len(dust_locations1_inverse[1]) == 0
    assert len(dust_locations1_inverse[2]) == 1
    assert dust_locations1_inverse[0][0] == 2
    assert dust_locations1_inverse[2][0] == 1

    # The result from getting_dust_locations(gas_positions, dust_positions, h_dust_data)
    dust_locations2 = [[], [0], [1]]
    dust_locations2_inverse = inverting_dust_locations(gas_positions, dust_locations2)
    assert len(dust_locations2_inverse) == len(sdfg)
    assert len(dust_locations2_inverse[0]) == 1
    assert len(dust_locations2_inverse[1]) == 1
    assert len(dust_locations2_inverse[2]) == 0
    assert dust_locations2_inverse[0][0] == 1
    assert dust_locations2_inverse[1][0] == 2

def test_inverting_dust_locations_diff_size():
    datag = {'x': [2, 8, 3], 'y': [8, 5, 2], 'h': [0.5, 0.5, 1]}
    datad = {'x': [2, 4, 9, 2.1], 'y': [2, 8, 5, 2], 'h': [0.25, 1.5, 1, 0.2]}
    sdfg = SarracenDataFrame(datag, params=dict())
    sdfd = SarracenDataFrame(datad, params=dict())
    gas_positions = sdfg[["x", "y"]].values
    dust_positions = sdfd[["x", "y"]].values
    h_gas_data = sdfg["h"].values
    h_dust_data = sdfd["h"].values

    # The result from getting_dust_locations(dust_positions, gas_positions, h_gas_data)
    dust_locations1 = [[], [2], [0, 3]]
    dust_locations1_inverse = inverting_dust_locations(dust_positions, dust_locations1)
    assert len(dust_locations1_inverse) == len(sdfd)
    assert len(dust_locations1_inverse[0]) == 1
    assert len(dust_locations1_inverse[1]) == 0
    assert len(dust_locations1_inverse[2]) == 1
    assert len(dust_locations1_inverse[3]) == 1
    assert dust_locations1_inverse[0][0] == 2
    assert dust_locations1_inverse[2][0] == 1
    assert dust_locations1_inverse[3][0] == 2

    # The result from getting_dust_locations(gas_positions, dust_positions, h_dust_data)
    dust_locations2 = [[], [0], [1], []]
    dust_locations2_inverse = inverting_dust_locations(gas_positions, dust_locations2)
    assert len(dust_locations2_inverse) == len(sdfg)
    assert len(dust_locations2_inverse[0]) == 1
    assert len(dust_locations2_inverse[1]) == 1
    assert len(dust_locations2_inverse[2]) == 0
    assert dust_locations2_inverse[0][0] == 1
    assert dust_locations2_inverse[1][0] == 2