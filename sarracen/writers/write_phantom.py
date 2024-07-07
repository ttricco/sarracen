import numpy as np

from ..readers.read_phantom import read_capture_pattern
from ..sarracen_dataframe import SarracenDataFrame


def _write_file_identifier(sdf: SarracenDataFrame):

    file_identifier = sdf.params['file_identifier'].ljust(100)

    read_tag = np.array([13], dtype='int32')
    ph_file = bytearray(read_tag.tobytes())
    ph_file += bytearray(map(ord, file_identifier))
    ph_file += bytearray(read_tag.tobytes())
    return ph_file


def _write_capture_pattern(def_int, def_real):
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

    return capture_pattern

class GlobalHeaderElement:
    def __init__(self, d_type, tags, values):
        self.d_type = d_type
        self.tags = tags
        self.values = values


class DtArrayCount:
    def __init__(self, d_type, count, tags):
        self.d_type = d_type
        self.count = count
        self.tags = tags


class SdfDtArrayCounts:
    def __init__(self, dt_array_counts, sdf):
        self.dtArrayCounts = dt_array_counts
        self.sdf = sdf


def invalid_key(key, used_keys):
    return key == 'file_identifier' or key == 'mass' or key in used_keys


def write_global_header_tags_and_values(bytes_file, tags, values, dtype, read_tag):
    bytes_file += bytearray(read_tag.tobytes())
    for tag in tags:
        bytes_file += bytearray(map(ord, tag.ljust(16)))
    bytes_file += bytearray(read_tag.tobytes())

    bytes_file += bytearray(read_tag.tobytes())
    bytes_file += bytearray(np.array(values, dtype=dtype).tobytes())
    bytes_file += bytearray(read_tag.tobytes())
    return bytes_file


def _write_global_header(sdf: SarracenDataFrame, def_int: np.dtype, def_real: np.dtype):

    params_dict = sdf.params
    dtypes = [def_int, np.int8, np.int16, np.int32, np.int64, def_real, np.float32, np.float64]
    global_headers = [GlobalHeaderElement(dt, [], []) for dt in dtypes]
    used_keys = set()

    for ghe in global_headers:
        for key in params_dict:
            if invalid_key(key, used_keys):
                continue
            if isinstance(params_dict[key], ghe.d_type):
                ghe.tags.append(key)
                used_keys.add(key)
                ghe.values.append(params_dict[key])

    read_tag = np.array([13], dtype='int32')
    bytes_file = bytearray()

    for header in global_headers:
        bytes_file += bytearray(read_tag.tobytes())
        nvars = np.array([len(header.tags)], dtype='int32')
        bytes_file += bytearray(nvars.tobytes())
        bytes_file += bytearray(read_tag.tobytes())

        if nvars == 0:
            continue

        bytes_file = write_global_header_tags_and_values(bytes_file, header.tags, header.values, header.d_type, read_tag)

    return bytes_file


def count_num_dt_arrays(test_sdf, dt):
    count_dt = 0
    for element in test_sdf.dtypes.values:
        if element == dt:
            count_dt += 1
    return count_dt


def get_array_tags(test_sdf, dt):
    return list(test_sdf.select_dtypes(include=[dt]).columns)

def get_last_index(sdf):
    return 1 if sdf.index[-1] == 0 else sdf.shape[0]

def _write_value_arrays(ph_file: bytearray, sdfs: [SarracenDataFrame], def_int: np.dtype, def_real: np.dtype):

    dtypes = [def_int, np.int8, np.int16, np.int32, np.int64, def_real, np.float32, np.float64]
    read_tag = np.array([13], dtype='int32')
    num_blocks = len(sdfs)

    file = bytearray(read_tag.tobytes())
    file += bytearray(np.array([num_blocks], dtype='int32').tobytes())
    file += bytearray(read_tag.tobytes())

    array_counts = []
    for sdf in sdfs:
        nvars = np.array([get_last_index(sdf)], dtype='int64')
        array_count = []
        dtypes_used = set()

        for d_type in dtypes:
            if d_type not in dtypes_used:
                dt_array_count = DtArrayCount(d_type, count_num_dt_arrays(sdf, d_type), get_array_tags(sdf, d_type))
                array_count.append(dt_array_count)
                dtypes_used.add(d_type)
            else:
                dt_array_count = DtArrayCount(d_type, 0, [])
                array_count.append(dt_array_count)

        counts = [ct.count for ct in array_count]
        nums = np.array(counts, dtype='int32')

        file += bytearray(read_tag.tobytes())
        file += bytearray(nvars.tobytes())
        file += bytearray(nums.tobytes())
        file += bytearray(read_tag.tobytes())
        array_counts.append(SdfDtArrayCounts(array_count, sdf))

    for sdfac in array_counts:
        for ct in sdfac.dtArrayCounts:
            if ct.count > 0:
                for tag in ct.tags:
                    file += bytearray(read_tag.tobytes())
                    file += bytearray(map(ord, tag.ljust(16)))
                    file += bytearray(read_tag.tobytes())

                    file += bytearray(read_tag.tobytes())
                    file += bytearray(np.array(list(sdfac.sdf[tag]), dtype=ct.d_type).tobytes())
                    file += bytearray(read_tag.tobytes())
    ph_file += file
    return ph_file


def determine_default_types(sdf: SarracenDataFrame, original_ph_file: str = ''):

    dtypes = sdf.dtypes.apply(lambda x: x.type)

    try:
        if original_ph_file:
            with open(original_ph_file, 'rb') as fp:
                def_int_dtype, def_real_dtype = read_capture_pattern(fp)
            return def_int_dtype, def_real_dtype
    except IOError as e:
        print(f"Unable to open file: {e}. Proceeding to determine default types.")

    all_ints = [dtype for dtype in dtypes if np.issubdtype(dtype, np.integer)]
    all_floats = [dtype for dtype in dtypes if np.issubdtype(dtype, np.floating)]

    def_int = max(all_ints, default=np.int32, key=lambda x: np.dtype(x).itemsize)
    def_real = max(all_floats, default=np.float64, key=lambda x: np.dtype(x).itemsize)

    return def_int, def_real


def write_phantom(sdfs: [SarracenDataFrame], original_ph_file: str = '', w_ph_file: str = 'w_ph_file'):

    def_int, def_real = determine_default_types(sdfs[0], original_ph_file)

    ph_file = _write_capture_pattern(def_int, def_real) #No iversion created
    ph_file += _write_file_identifier(sdfs[0])
    ph_file += _write_global_header(sdfs[0], def_int, def_real)
    ph_file += _write_value_arrays(ph_file, sdfs, def_int, def_real)

    with open(w_ph_file, 'wb') as phantom_file:
        phantom_file.write(ph_file)
        phantom_file.close()

    return phantom_file
