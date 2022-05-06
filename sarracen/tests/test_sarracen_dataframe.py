import pandas as pd

from sarracen import SarracenDataFrame


def test_position():
    # The 'x' and 'y' keywords should be detected.
    df = pd.DataFrame({'P': [1, 1],
                       'h': [1, 1],
                       'rho': [1, 1],
                       'x': [5, 6],
                       'y': [5, 4],
                       'm': [1, 1]})
    sdf = SarracenDataFrame(df)

    assert sdf.xcol == 'x'
    assert sdf.ycol == 'y'

    # The 'rx' and 'ry' keywords should be detected.
    df = pd.DataFrame({'ry': [-1, 1],
                       'h': [1, 1],
                       'rho': [1, 1],
                       'rx': [3, 4],
                       'P': [1, 1],
                       'm': [1, 1]})
    sdf = SarracenDataFrame(df)

    assert sdf.xcol == 'rx'
    assert sdf.ycol == 'ry'

    # No keywords, so fall back to the first two columns.
    df = pd.DataFrame({'i': [3.4, 2.1],
                       'j': [4.9, 1.6],
                       'h': [1, 1],
                       'rho': [1, 1],
                       'P': [1, 1],
                       'm': [1, 1]})
    sdf = SarracenDataFrame(df)

    assert sdf.xcol == 'i'
    assert sdf.ycol == 'j'
