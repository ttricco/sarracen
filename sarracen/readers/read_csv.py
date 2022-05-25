import pandas as pd
from sarracen.sarracen_dataframe import SarracenDataFrame


def read_csv(filename):
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
    df = SarracenDataFrame(pd.read_csv(filename))

    df.units = _get_units(df.columns)
    df.columns = _get_labels(df.columns)

    return df




def _get_units(columns):
     return columns.str.extract(r'((?<=\[).+(?=\]))')[0]


def _get_labels(columns):
     return columns.str.extract(r'(^[^\[]*[^\s\[])')[0]
