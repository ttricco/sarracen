import numpy as np

import sarracen
from ..readers.read_phantom import _read_capture_pattern
from ..sarracen_dataframe import SarracenDataFrame

def _create_capture_pattern(def_int, def_real):
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


def write_file_identifier(ph_file, data):
    read_tag = np.array([13], dtype='int32')
    ph_file += bytearray(read_tag.tobytes())
    ph_file += bytearray(map(ord, data))
    ph_file += bytearray(read_tag.tobytes())
    return ph_file


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

def invalid_key(key, used_keys):
    return key == 'file_identifier' or key == 'mass' or key in used_keys

def add_4byte_tag(bytes_file, read_tag):
    bytes_file += bytearray(read_tag.tobytes())
    return bytes_file


def _write_global_header(sdf: SarracenDataFrame, def_int: np.dtype, def_real: np.dtype):

    dtypes = [def_int, np.int8, np.int16, np.int32, np.int64, def_real, np.float32, np.float64]

    global_headers = []
    used_keys = []

    params_dict = sdf.params

    for dt in dtypes:
        header_element = GlobalHeaderElement(dt, [], [])
        global_headers.append(header_element)

    for ghe in global_headers:
        for key in params_dict:
            if invalid_key(key, used_keys):
                continue
            if isinstance(params_dict[key], ghe.d_type):
                ghe.tags.append(key)
                used_keys.append(key)
                ghe.values.append(params_dict[key])

    read_tag = np.array([13], dtype='int32')
    bytes_file = bytearray()

    for header in global_headers:

        bytes_file = add_4byte_tag(bytes_file, read_tag)
        nvars = np.array([len(header.tags)], dtype='int32')
        bytes_file += bytearray(nvars.tobytes())
        bytes_file = add_4byte_tag(bytes_file, read_tag)

        if nvars == 0:
            continue

        bytes_file = add_4byte_tag(bytes_file, read_tag)
        for tg in header.tags:
            bytes_file += bytearray(map(ord, tg.ljust(16)))
        bytes_file = add_4byte_tag(bytes_file, read_tag)

        bytes_file = add_4byte_tag(bytes_file, read_tag)
        values = np.array(header.values, dtype=header.d_type)
        bytes_file += bytearray(values.tobytes())
        bytes_file = add_4byte_tag(bytes_file, read_tag)

    return bytes_file


def count_num_dt_arrays(test_sdf, dt):
    count_dt = 0
    for element in test_sdf.dtypes.values:
        if element == dt:
            count_dt += 1
    return count_dt


def get_array_tags(test_sdf, dt):
    return list(test_sdf.select_dtypes(include=[dt]).columns)


def _test_write_arrays(file, test_sdf):

    dtypes = [np.int32, np.int8, np.int16, np.int32, np.int64, np.float64, np.float32, np.float64]
    read_tag = np.array([13], dtype='int32')
    num_blocks = 1

    file += bytearray(read_tag.tobytes())
    file += bytearray(np.array([num_blocks], dtype='int32').tobytes())
    file += bytearray(read_tag.tobytes())

    nvars = np.array([test_sdf.shape[0]], dtype='int64')
    array_count = []
    dtypes_used = []
    for type in dtypes:
        if type not in dtypes_used:
            dt_array_count = DtArrayCount(type, count_num_dt_arrays(test_sdf, type), get_array_tags(test_sdf, type))
            array_count.append(dt_array_count)
            dtypes_used.append(type)
        else:
            dt_array_count = DtArrayCount(type, 0, [])
            array_count.append(dt_array_count)

    file = bytearray(read_tag.tobytes())
    file += bytearray(nvars.tobytes())
    counts = []
    for ct in array_count:
        counts.append(ct.count)
    file += bytearray(np.array(counts, dtype='int32').tobytes())
    file += bytearray(read_tag.tobytes())

    for ct in array_count:
        if ct.count > 0:
            for tag in ct.tags:
                file += bytearray(read_tag.tobytes())
                file += bytearray(map(ord, tag.ljust(16)))
                file += bytearray(read_tag.tobytes())

                file += bytearray(read_tag.tobytes())
                file += bytearray(np.array(list(test_sdf[tag]), dtype=ct.d_type).tobytes())
                file += bytearray(read_tag.tobytes())
    return file


def determine_default_types(sdf: SarracenDataFrame, original_phantom_file: str = ''):

    dtypes = sdf.dtypes.apply(lambda x: x.type)

    try:
        if original_phantom_file:
            with open(original_phantom_file, 'rb') as fp:
                def_int_dtype, def_real_dtype = _read_capture_pattern(fp)
            return def_int_dtype, def_real_dtype
    except IOError as e:
        print(f"Unable to open file: {e}. Proceeding to determine default types.")

    all_ints = [dtype for dtype in dtypes if np.issubdtype(dtype, np.integer)]
    all_floats = [dtype for dtype in dtypes if np.issubdtype(dtype, np.floating)]

    def_int = max(all_ints, default=np.int32, key=lambda x: np.dtype(x).itemsize)
    def_real = max(all_floats, default=np.float64, key=lambda x: np.dtype(x).itemsize)

    return def_int, def_real


def write_phantom(sdf: SarracenDataFrame, original_phantom_file: str = ''):

    def_int, def_real = determine_default_types(sdf,  original_phantom_file)
    ph_file = _create_capture_pattern(def_int, def_real) #No iversion created
    file_identifier = sdf.params['file_identifier'].ljust(100)
    ph_file = write_file_identifier(ph_file, file_identifier)
    ph_file += _write_global_header(sdf, def_int,  def_real)
    ph_file += _test_write_arrays(ph_file, sdf)

    with open("ph_file", 'wb') as f:
        f.write(ph_file)
        f.close()

    return f
#    return sarracen.read_phantom("ph_file")
