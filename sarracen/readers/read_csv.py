import pandas as pd
from ..sarracen_dataframe import SarracenDataFrame


def read_csv(*args, **kwargs) -> SarracenDataFrame:
    """
    Read data from a csv file.

    Parameters
    ----------
    filename : str
        Name of the file to be loaded.

    Returns
    -------
    SarracenDataFrame
    """
    df = SarracenDataFrame(pd.read_csv(*args, **kwargs))

    df.units = _get_units(df.columns)
    df.columns = _get_labels(df.columns)

    return df




def _get_units(columns: pd.Series) -> pd.Series:
     return columns.str.extract(r'((?<=\[).+(?=\]))')[0]


def _get_labels(columns: pd.Series) -> pd.Series:
     return columns.str.extract(r'(^[^\[]*[^\s\[])')[0]
