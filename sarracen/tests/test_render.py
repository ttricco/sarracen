import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from sarracen import SarracenDataFrame
from sarracen.kernels import CubicSplineKernel


def test_plot_properties():
    df = pd.DataFrame({'x': [3, 6],
                       'y': [5, 1],
                       'P': [1, 1],
                       'h': [1, 1],
                       'rho': [1, 1],
                       'm': [1, 1]})
    sdf = SarracenDataFrame(df)

    fig, ax = sdf.render('P')

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    assert ax.get_xlabel() == 'x'
    assert ax.get_ylabel() == 'y'
    # the colorbar is contained in a second axes object inside the figure
    assert fig.axes[1].get_ylabel() == 'P'

    assert ax.get_xlim() == (3, 6)
    assert ax.get_ylim() == (1, 5)

    # aspect ratio of data max & min is 4/3,
    # pixel count => (256, 341)
    # pixel width => (3/256, 4/341)
    # both particles are in corners
    # therefore closest pixel is => sqrt((3/512)**2, (2/341)**2)
    # use default kernel to determine the max pressure value
    assert fig.axes[1].get_ylim() == (0, CubicSplineKernel(2).w(np.sqrt((3/512)**2 + (2/341)**2)))
