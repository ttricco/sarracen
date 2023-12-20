import pandas as pd
import numpy as np
import io
from pandas import testing as tm
from sarracen.readers import _read_capture_pattern
import pytest


@pytest.mark.parametrize("def_int, def_real",
                         [(np.int32, np.float64), (np.int32, np.float32),
                          (np.int64, np.float64), (np.int64, np.float32)])
def test_determine_default_precision(def_int, def_real):
    """ Test if default int / real precision can be determined. """
    # construct capture pattern
    read_tag = np.array([13], dtype='int32')
    i1 = np.array([60769], dtype=def_int)
    r2 = np.array([60878], dtype=def_real)
    i2 = np.array([60878], dtype=def_int)
    iversion = np.array([0], dtype=def_int)
    i3 = np.array([690706], dtype=def_int)

    capture_pattern = bytearray(read_tag.tobytes())
    capture_pattern += bytearray(i1.tobytes())
    capture_pattern += bytearray(r2.tobytes())
    capture_pattern += bytearray(i2.tobytes())
    capture_pattern += bytearray(iversion.tobytes())
    capture_pattern += bytearray(i3.tobytes())
    capture_pattern += bytearray(read_tag.tobytes())

    f = io.BytesIO(capture_pattern)
    returned_int, returned_real = _read_capture_pattern(f)
    assert returned_int == def_int
    assert returned_real == def_real

