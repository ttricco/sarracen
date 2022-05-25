import pandas as pd
import numpy as np
from pandas import testing as tm
from sarracen.readers import read_csv


def test_get_units():
    test_units = pd.Series(['va_r?s w/ sp3ci@l ch%rs [units /$:. ]',
                 'a [a]',
                 'a',
                 '[]'])
    test_units = read_csv._get_units(test_units)

    answer_units = ['units /$:. ',
                   'a',
                   None,
                   None]
    answer_units = pd.Series(answer_units, name=0)

    tm.assert_series_equal(test_units, answer_units)



def test_get_labels():
    test_labels = pd.Series(['va_r?s w/ sp3ci@l ch%rs [units /$:. ]',
             'a [a]',
             'a',
             '[]'])
    test_labels = read_csv._get_labels(test_labels)

    answer_labels = ['va_r?s w/ sp3ci@l ch%rs',
                'a',
                'a',
                None]
    answer_labels = pd.Series(answer_labels, name=0)

    tm.assert_series_equal(test_labels, answer_labels)
