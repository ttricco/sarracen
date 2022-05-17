import pandas as pd
import numpy as np
from pytest import approx

from sarracen import SarracenDataFrame
from sarracen.kernels import CubicSplineKernel
from sarracen.interpolate import interpolate2D, interpolate1DCross, interpolate3DCross


def test_interpolate2d():
    df = pd.DataFrame({'x': [0],
                       'y': [0],
                       'P': [1],
                       'h': [1],
                       'rho': [1],
                       'm': [1]})
    sdf = SarracenDataFrame(df, params=dict())

    image = interpolate2D(sdf, 'x', 'y', 'P', CubicSplineKernel(2), 0.1, 0.1, -2, -2, 40, 40)

    assert image[0][0] == 0
    assert image[20][0] == approx(CubicSplineKernel(2).w(np.sqrt((-1.95) ** 2 + 0.05 ** 2)), rel=1e-8)
    assert image[20][20] == approx(CubicSplineKernel(2).w(np.sqrt(0.05 ** 2 + 0.05 ** 2)), rel=1e-8)
    assert image[12][17] == approx(CubicSplineKernel(2).w(np.sqrt(0.75 ** 2 + 0.25 ** 2)), rel=1e-8)


def test_interpolate1dcross():
    df = df = pd.DataFrame({'x': [0],
                            'y': [0],
                            'P': [1],
                            'h': [1],
                            'rho': [1],
                            'm': [1]})
    sdf = SarracenDataFrame(df, params=dict())

    # first, test a cross-section at y=0
    output = interpolate1DCross(sdf, 'x', 'y', 'P', CubicSplineKernel(2), -2, 0, 2, 0, 40)

    assert output[0] == approx(CubicSplineKernel(2).w(np.sqrt((-1.95) ** 2)), rel=1e-8)
    assert output[20] == approx(CubicSplineKernel(2).w(np.sqrt(0.05 ** 2)), rel=1e-8)
    assert output[17] == approx(CubicSplineKernel(2).w(np.sqrt(0.25 ** 2)), rel=1e-8)

    # next, test a cross-section where x=y
    output = interpolate1DCross(sdf, 'x', 'y', 'P', CubicSplineKernel(2), -2, -2, 2, 2, 40)

    assert output[0] == approx(CubicSplineKernel(2).w(np.sqrt(2*(1.95 ** 2))), rel=1e-8)
    assert output[20] == approx(CubicSplineKernel(2).w(np.sqrt(2*(0.05 ** 2))), rel=1e-8)
    assert output[17] == approx(CubicSplineKernel(2).w(np.sqrt(2*(0.25 ** 2))), rel=1e-8)

def test_interpolate3dcross():
    df = df = pd.DataFrame({'x': [0],
                            'y': [0],
                            'z': [0],
                            'P': [1],
                            'h': [1],
                            'rho': [1],
                            'm': [1]})
    sdf = SarracenDataFrame(df, params=dict())

    # first, test a cross-section at z=0
    image = interpolate3DCross(sdf, 'x', 'y', 'z', 'P', CubicSplineKernel(2), 0, 0.1, 0.1, -2, -2, 40, 40)

    # should be exactly the same as for a 2D rendering
    assert image[0][0] == 0
    assert image[20][0] == approx(CubicSplineKernel(2).w(np.sqrt((-1.95) ** 2 + 0.05 ** 2)), rel=1e-8)
    assert image[20][20] == approx(CubicSplineKernel(2).w(np.sqrt(0.05 ** 2 + 0.05 ** 2)), rel=1e-8)
    assert image[12][17] == approx(CubicSplineKernel(2).w(np.sqrt(0.75 ** 2 + 0.25 ** 2)), rel=1e-8)

    # next, test a cross-section at z=0.5
    image = interpolate3DCross(sdf, 'x', 'y', 'z', 'P', CubicSplineKernel(2), 0.5, 0.1, 0.1, -2, -2, 40, 40)

    assert image[0][0] == 0
    assert image[20][0] == approx(CubicSplineKernel(2).w(np.sqrt((-1.95) ** 2 + 0.05 ** 2 + (0.5 ** 2))), rel=1e-8)
    assert image[20][20] == approx(CubicSplineKernel(2).w(np.sqrt(2 * (0.05 ** 2) + (0.5 ** 2))), rel=1e-8)
    assert image[12][17] == approx(CubicSplineKernel(2).w(np.sqrt(0.75 ** 2 + 0.25 ** 2 + (0.5 ** 2))), rel=1e-8)
