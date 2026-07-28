from typing import IO, List, Union
import pandas as pd
import numpy as np


def _determine_column_labels(file: IO) -> List[str]:
    """
    Determine the column labels.

    We need to read through the lines starting with '#' to find the specific
    line that contains the column labels. We assume it is the last line.
    """

    last_line = None

    while True:
        pos = file.tell()
        line = file.readline()
        if not line:
            raise ValueError("No data found in file.")

        # only lines starting '#' are valid
        if line.startswith("#"):
            last_line = line.strip()
        else:
            # skipped too far, rewind one line
            file.seek(pos)
            break

    if last_line is None:
        raise ValueError("No header line found in file.")

    # Get header labels
    # Assumes labels are exactly 3 spaces apart.
    # Assumes labels start with 2 digits (which we discard).
    # Seems like brittle code.
    # But this works. If this fails in future, can try switching to regex.
    line = last_line
    line = line[3:-1]
    labels = [x[2:].strip() for x in line.split(']   [')]

    return labels


def _infer_type(s: str) -> Union[np.float64, np.int32, str]:
    """
    Given a value from the ev, determine its type.

    Phantom writes floats using scientific notation always. Consider anything
    with a decimal point and an 'e' as a float. Otherwise it is an int.

    Special case is an integer with leading zeros. These are considered a str,
    as this is most likely the 'dump' column of an APR .ev file.
    """

    if '.' in s and 'e' in s.lower():
        try:
            return np.float64(s)
        except ValueError:
            pass

    if s.startswith('0') and s.isdigit():
        return s

    try:
        return np.int32(s)
    except ValueError:
        return s


def read_phantom_ev(filename: str) -> pd.DataFrame:
    """
    Read a Phantom .ev file and return a pandas DataFrame of the data.

    Each row of the DataFrame corresponds to a single line of the .ev file.
    The reader will skip any header lines that start with '#'. It assumes the
    last header line contains the column names. Values are generally stored
    as np.float64, but the reader will attempt to identify str and int values.

    Parameters
    ----------
    filename : str
        Name of the file to be loaded.

    Returns
    -------
    pandas DataFrame
    """

    data = []

    with open(filename) as file:

        labels = _determine_column_labels(file)

        # Read data
        for line in file:
            parts = line.strip().split()
            row = [_infer_type(x) for x in parts]
            data.append(row)

    return pd.DataFrame(data, columns=labels)
