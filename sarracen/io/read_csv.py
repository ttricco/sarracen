import pandas as pd
from sarracen.sarracen_dataframe import SarracenDataFrame


def _get_units(columns):
     return columns.str.extract(r'((?<=\[).+(?=\]))')[0]

def _get_labels(columns):
     return columns.str.extract(r'(^[^\[]*[^\s\[])')[0]

def read_csv(filename):
    df = SarracenDataFrame(pd.read_csv(filename))

    labels = _get_labels(df.columns)
    units = _get_units(df.columns)

    #df.columns = labels
    #df.units = units

    return df
