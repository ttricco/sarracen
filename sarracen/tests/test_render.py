import pandas as pd

from sarracen import SarracenDataFrame

def test_snapping():
    df = pd.DataFrame({'x': [0.0001, 5.2],
                       'y': [3.00004, 0.1],
                       'P': [1, 1],
                       'h': [1, 1],
                       'rho': [1, 1],
                       'm': [1, 1]})
    sdf = SarracenDataFrame(df)

    fig, ax = sdf.render('P')

    # [0.0001, 5.2] -> (0.0, 5.2)
    assert ax.get_xlim() == (0.0, 5.2)
    # [3.00004, 0.1] -> (0.1, 3.0)
    assert ax.get_ylim() == (0.1, 3.0)
