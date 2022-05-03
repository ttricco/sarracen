import pandas as pd

from sarracen.sarracen_dataframe import SarracenDataFrame

def read_csv(filename):
    df = pd.read_csv(filename)
    return SarracenDataFrame(df)
