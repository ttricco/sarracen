from pandas import DataFrame, Series
import numpy as np


class SarracenDataFrame(DataFrame):

    _metadata = ['_params', '_units']

    def __init__(self, data=None, params=None, *args, **kwargs):

        # call pandas DataFrame contructor
        super().__init__(data, *args, **kwargs)

        self._params = None
        self.params = params

        self._units = None
        self.units = Series(['' for i in range(len(self.columns))])


    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, new_params):
        if new_params is None:
            self._params = None
            return
        if not type(new_params) is dict:
            raise TypeError("Parameters not a dictionary")
        self._params = new_params

    @params.getter
    def params(self):
        return self._params

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, new_units):
        self._units = new_units

    @units.getter
    def units(self):
        return self._units
