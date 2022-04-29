from pandas import DataFrame


class SarracenDataFrame(DataFrame):

    def __init__(self, data=None, *args, **kwargs):

        # call pandas DataFrame contructor
        super().__init__(data, *args, **kwargs)
