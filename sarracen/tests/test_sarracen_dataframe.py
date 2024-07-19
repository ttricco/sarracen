"""pytest unit tests for sarracen_dataframe.py functionality."""
import numpy as np
from matplotlib import pyplot as plt

from sarracen import SarracenDataFrame, render


def test_special_columns():
    # The 'x', 'y', 'rho', 'm', and 'h' keywords should be detected.
    # A 'z' column should not be detected.
    data = {'P': [1, 1], 'h': [1, 1], 'rho': [1, 1],
            'x': [5, 6], 'y': [5, 4], 'm': [1, 1]}
    sdf = SarracenDataFrame(data)

    assert sdf.xcol == 'x'
    assert sdf.ycol == 'y'
    assert sdf.zcol is None
    assert sdf.rhocol == 'rho'
    assert sdf.mcol == 'm'
    assert sdf.hcol == 'h'

    # The 'rx', 'ry', 'rz', 'density', and 'mass' keywords should be detected.
    # An 'h' column should not be detected.
    data = {'ry': [-1, 1], 'density': [1, 1], 'rx': [3, 4],
            'P': [1, 1], 'rz': [4, 3], 'mass': [1, 1]}
    sdf = SarracenDataFrame(data)

    assert sdf.xcol == 'rx'
    assert sdf.ycol == 'ry'
    assert sdf.zcol == 'rz'
    assert sdf.rhocol == 'density'
    assert sdf.mcol == 'mass'
    assert sdf.hcol is None

    # No keywords, so fall back to the first two columns for x and y.
    # Even though 'k' exists, this will be assumed to be 2D data.
    # The 'h' column will be detected, but not density or mass columns.
    data = {'i': [3.4, 2.1], 'j': [4.9, 1.6], 'k': [2.3, 2.0],
            'h': [1, 1], 'P': [1, 1]}
    sdf = SarracenDataFrame(data)

    assert sdf.xcol == 'i'
    assert sdf.ycol == 'j'
    assert sdf.zcol is None
    assert sdf.rhocol is None
    assert sdf.mcol is None
    assert sdf.hcol == 'h'


def test_dimensions():
    # This should be detected as 3-dimensional data.
    data = {'P': [1, 1], 'z': [4, 3], 'h': [1, 1], 'rho': [1, 1],
            'x': [5, 6], 'y': [5, 4], 'm': [1, 1]}
    sdf = SarracenDataFrame(data)

    assert sdf.get_dim() == 3

    # This should be detected as 2-dimensional data.
    data = {'P': [1, 1], 'h': [1, 1], 'y': [5, 4], 'rho': [1, 1],
            'm': [1, 1], 'x': [5, 6]}
    sdf = SarracenDataFrame(data)

    assert sdf.get_dim() == 2

    # This should assumed to be 2-dimensional data.
    data = {'P': [1, 1], 'h': [1, 1], 'rho': [1, 1], 'm': [1, 1]}
    sdf = SarracenDataFrame(data)

    assert sdf.get_dim() == 2


def test_column_changing():
    data = {'P': [1], 'z': [2], 'h': [3], 'rho': [4], 'x': [5],
            'y': [6], 'm': [7], 'd': [8], 'smooth': [9], 'ma': [10]}
    sdf = SarracenDataFrame(data)

    assert sdf.xcol == 'x'
    assert sdf.ycol == 'y'
    assert sdf.zcol == 'z'
    assert sdf.rhocol == 'rho'
    assert sdf.mcol == 'm'
    assert sdf.hcol == 'h'

    sdf.xcol = 'z'  # column 'z' exists, assignment will be accepted
    sdf.ycol = 'a'  # column 'a' doesn't exist, assignment will be rejected
    sdf.zcol = 'x'  # accept
    sdf.rhocol = 'e'  # reject
    sdf.mcol = 'ma'  # accept
    sdf.hcol = 'smooth_length'  # reject

    assert sdf.xcol == 'z'
    assert sdf.ycol == 'y'
    assert sdf.zcol == 'x'
    assert sdf.rhocol == 'rho'
    assert sdf.mcol == 'ma'
    assert sdf.hcol == 'h'

    sdf.xcol = 'v'  # reject
    sdf.ycol = 'P'  # accept
    sdf.zcol = 'k'  # reject
    sdf.rhocol = 'd'  # accept
    sdf.mcol = 'mass'  # reject
    sdf.hcol = 'smooth'  # accept

    assert sdf.xcol == 'z'
    assert sdf.ycol == 'P'
    assert sdf.zcol == 'x'
    assert sdf.rhocol == 'd'
    assert sdf.mcol == 'ma'
    assert sdf.hcol == 'smooth'


def test_render_passthrough():
    # Basic tests that both sdf.render() and render(sdf) return the same plots

    # 2D dataset
    data = {'x': [3, 6], 'y': [5, 1], 'P': [1, 1],
            'h': [1, 1], 'rho': [1, 1], 'm': [1, 1]}
    sdf = SarracenDataFrame(data)

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    ax1 = sdf.render('P', ax=ax1)
    ax2 = render(sdf, 'P', ax=ax2)

    assert repr(ax1) == repr(ax2)

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    ax1 = sdf.render('P', xsec=True, ax=ax1)
    ax2 = render(sdf, 'P', xsec=True, ax=ax2)

    assert repr(ax1) == repr(ax2)

    # 3D dataset
    data = {'x': [3, 6], 'y': [5, 1], 'z': [2, 1], 'P': [1, 1],
            'h': [1, 1], 'Ax': [5, 3], 'Ay': [2, 3],
            'Az': [1, -1], 'rho': [1, 1], 'm': [1, 1]}
    sdf = SarracenDataFrame(data)

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    ax1 = sdf.render('P', ax=ax1)
    ax2 = render(sdf, 'P', ax=ax2)

    assert repr(ax1) == repr(ax2)

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    ax1 = sdf.render('P', xsec=True, ax=ax1)
    ax2 = render(sdf, 'P', xsec=True, ax=ax2)

    assert repr(ax1) == repr(ax2)


def test_calc_density():
    # Tests that the density calculation is working as intended.

    # 2D Data
    data = {'x': [3, 6], 'y': [5, 1], 'h': [0.00683, 4.2166]}
    params = {'mass': 89.3452, 'hfact': 1.2}
    sdf = SarracenDataFrame(data, params)

    sdf.calc_density()

    rho_0 = sdf.params['mass'] * (sdf.params['hfact'] / sdf['h'][0])**2
    rho_1 = sdf.params['mass'] * (sdf.params['hfact'] / sdf['h'][1])**2

    assert sdf['rho'][0] == rho_0
    assert sdf['rho'][1] == rho_1

    # 3D Data
    data = {'x': [3, 6], 'y': [5, 1], 'z': [2, 1], 'h': [0.0234, 7.3452]}
    params = {'mass': 63.2353, 'hfact': 1.2}
    sdf = SarracenDataFrame(data, params)

    sdf.calc_density()

    rho_0 = sdf.params['mass'] * (sdf.params['hfact'] / sdf['h'][0])**3
    rho_1 = sdf.params['mass'] * (sdf.params['hfact'] / sdf['h'][1])**3

    assert sdf['rho'][0] == rho_0
    assert sdf['rho'][1] == rho_1


def test_centre_of_mass():
    """ Basic test of centre of mass calculation. """

    # randomly place particles
    rng = np.random.default_rng(seed=5)
    x = rng.random(100)
    y = rng.random(100)
    z = rng.random(100)
    # mirror in 8 quadrants
    x = np.append(x, [x, x, x, -1 * x, -1 * x, -1 * x, -1 * x])
    y = np.append(y, [y, -1 * y, -1 * y, y, y, -1 * y, -1 * y])
    z = np.append(z, [-1 * z, z, -1 * z, z, -1 * z, z, -1 * z])

    sdf = SarracenDataFrame(data={'x': x, 'y': y, 'z': z},
                            params={'mass': 3.2e-4})

    assert sdf.centre_of_mass() == [0.0, 0.0, 0.0]
