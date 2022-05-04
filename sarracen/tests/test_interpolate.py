import pandas as pd
from pytest import approx

from sarracen import SarracenDataFrame
from sarracen.kernels import CubicSplineKernel
from sarracen.interpolate import interpolate2D


def test_interpolate2d():
    df = pd.DataFrame({'x': [0],
                       'y': [0],
                       'P': [1],
                       'h': [1],
                       'rho': [1],
                       'm': [1]})
    sdf = SarracenDataFrame(df, params=dict())

    image = interpolate2D(sdf, 'x', 'y', 'P', CubicSplineKernel(), -2, -2, 0.1, 0.1, 40, 40)

    assert image[0][0] == 0
    # sqrt((-1.95)^2+(0.05)^2)
    assert image[20][0] == approx(CubicSplineKernel().w(1.95064, 2), rel=1e-4)
    # sqrt((0.05)^2+(0.05)^2)
    assert image[20][20] == approx(CubicSplineKernel().w(0.070711, 2), rel=1e-4)
