from pandas import DataFrame


class SarracenDataFrame(DataFrame):

    _metadata = ['_params', 'params']

    def __init__(self, data=None, *args, **kwargs):

        # call pandas DataFrame contructor
        super().__init__(data, *args, **kwargs)
        self.params = dict()

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, new_params):
        if not type(new_params) is dict:
            raise TypeError("Parameters not a dictionary")
        self._params = new_params

    @params.getter
    def params(self):
        return self._params
