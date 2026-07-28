"""pytest unit tests for sarracen_dataframe.py functionality."""
import numpy as np
from numpy.testing import assert_allclose
from matplotlib import pyplot as plt

from sarracen import SarracenDataFrame, render


def test_special_columns() -> None:
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


def test_dimensions() -> None:
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


def test_column_changing() -> None:
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


def test_render_passthrough() -> None:
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


def test_calc_density() -> None:
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


def test_centre_of_mass() -> None:
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


def test_calc_one_fluid_quantities() -> None:
    """ Test calculation of one fluid quantity. """

    x = [0, 0, 0, 0, 1, 1, 1, 1]
    y = [0, 0, 1, 1, 0, 0, 1, 1]
    z = [0, 1, 0, 1, 0, 1, 0, 1]
    h = [1.0, 0.9, 1.2, 1.22, 1.14, 1.05, 1.42, 1.87]
    dustfracs = [0.01, 0.02, 0.035, 0.05, 0.032, 0.001, 0.0001, 0.00001]

    params = {'mass': 89.3452, 'hfact': 1.2, 'ndustsmall': 1}

    sdf = SarracenDataFrame(data={'x': x, 'y': y, 'z': z, 'h': h,
                                  'dustfrac': dustfracs},
                            params=params)

    sdf.calc_one_fluid_quantities()

    rho = sdf.params['mass'] * (sdf.params['hfact'] / sdf['h'])**3
    rho_g = rho * (1.0 - np.array(dustfracs))
    rho_d = rho * dustfracs
    dtg = np.array(dustfracs) / (1.0 - np.array(dustfracs))

    assert 'rho_g' in sdf.columns
    assert 'rho_d' in sdf.columns
    assert 'dtg' in sdf.columns
    assert_allclose(sdf['rho_g'], rho_g)
    assert_allclose(sdf['rho_d'], rho_d)
    assert_allclose(sdf['dtg'], dtg)


def test_calc_one_fluid_quantities_multigrain() -> None:
    """ Test calculation of one fluid quantity. """

    x = [0, 0, 0, 0, 1, 1, 1, 1]
    y = [0, 0, 1, 1, 0, 0, 1, 1]
    z = [0, 1, 0, 1, 0, 1, 0, 1]
    h = [1.0, 0.9, 1.2, 1.22, 1.14, 1.05, 1.42, 1.87]
    dustfrac1 = [0.01, 0.02, 0.035, 0.05, 0.032, 0.001, 0.0001, 0.00001]
    dustfrac2 = [0.001, 0.002, 0.0035, 0.005, 0.0032, 0.005, 0.0002, 0.00002]
    dustfrac3 = [0.003, 0.03, 0.02, 0.067, 0.038, 0.009, 0.1, 0.00003]
    dustfrac4 = [0.02, 0.04, 0.045, 0.005, 0.036, 0.0005, 0.01, 0.00004]

    params = {'mass': 89.3452, 'hfact': 1.2, 'ndustsmall': 4}

    sdf = SarracenDataFrame(data={'x': x, 'y': y, 'z': z, 'h': h,
                                  'dustfrac1': dustfrac1,
                                  'dustfrac2': dustfrac2,
                                  'dustfrac3': dustfrac3,
                                  'dustfrac4': dustfrac4},
                            params=params)

    sdf.calc_one_fluid_quantities()

    dustfrac_total = (np.array(dustfrac1) + np.array(dustfrac2)
                      + np.array(dustfrac3) + np.array(dustfrac4))
    rho = sdf.params['mass'] * (sdf.params['hfact'] / sdf['h'])**3
    rho_g = rho * (1.0 - dustfrac_total)
    rho_d_total = rho * dustfrac_total
    dtg = dustfrac_total / (1.0 - dustfrac_total)

    assert 'rho_g' in sdf.columns
    assert 'rho_d_total' in sdf.columns
    assert 'dtg' in sdf.columns
    assert_allclose(sdf['rho_g'], rho_g)
    assert_allclose(sdf['rho_d_total'], rho_d_total)
    assert_allclose(sdf['dtg'], dtg)

    dustfracs = [dustfrac1, dustfrac2, dustfrac3, dustfrac4]
    labels = ['rho_d', 'rho_d_2', 'rho_d_3', 'rho_d_4']

    for label, dustfrac in zip(labels, dustfracs):
        rho_d = rho * dustfrac
        assert label in sdf.columns
        assert_allclose(sdf[label], rho_d)

