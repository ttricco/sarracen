import numpy as np

from ..readers.read_phantom import read_capture_pattern
from ..sarracen_dataframe import SarracenDataFrame


class GlobalHeaderElement:
    def __init__(self, d_type, tags, values):
        self.d_type = d_type
        self.tags = tags
        self.values = values


class DataTypeTags:
    def __init__(self, d_type: np.dtype, tags: [str]):
        self.d_type = d_type
        self.tags = tags


class SdfDataTypeTags:
    def __init__(self, data_type_tags: [DataTypeTags], sdf: SarracenDataFrame):
        self.data_type_tags = data_type_tags
        self.sdf = sdf


def _write_fortran_block(value: [],
                         dtype: type):
    write_tag = np.array([len(value) * dtype().itemsize], dtype=np.int32)
    bytes_file = bytearray(write_tag.tobytes())
    bytes_file += bytearray(np.array(value, dtype=dtype).tobytes())
    bytes_file += bytearray(write_tag.tobytes())
    return bytes_file


def _write_file_identifier(sdf: SarracenDataFrame):

    file_identifier = sdf.params['file_identifier'].ljust(100)

    write_tag = np.array([len(file_identifier)], dtype='int32')
    ph_file = bytearray(write_tag.tobytes())
    ph_file += bytearray(map(ord, file_identifier))
    ph_file += bytearray(write_tag.tobytes())
    return ph_file


def _write_capture_pattern(def_int, def_real):

    write_tag = 16 + def_real().itemsize
    write_tag = np.array([write_tag], dtype='int32')
    i1 = np.array([60769], dtype=np.int32)
    r2 = np.array([60878], dtype=def_real)
    i2 = np.array([60878], dtype=np.int32)
    iversion = np.array([1], dtype=np.int32)
    i3 = np.array([690706], dtype=np.int32)

    capture_pattern = bytearray(write_tag.tobytes())
    capture_pattern += bytearray(i1.tobytes())
    capture_pattern += bytearray(r2.tobytes())
    capture_pattern += bytearray(i2.tobytes())
    capture_pattern += bytearray(iversion.tobytes())
    capture_pattern += bytearray(i3.tobytes())
    capture_pattern += bytearray(write_tag.tobytes())

    return capture_pattern


def _invalid_key(key, used_keys):
    exclude = ['file_identifier', 'mass', 'def_int_dtype', 'def_real_dtype']
    return key in exclude or key in used_keys


def _write_global_header_tags_and_values(tags, values, dtype):
    write_tag = np.array([len(tags) * 4], dtype=np.int32)
    bytes_file = bytearray(write_tag.tobytes())
    for tag in tags:
        bytes_file += bytearray(map(ord, tag.ljust(16)))
    bytes_file += bytearray(write_tag.tobytes())

    write_tag = np.array(len(values) * dtype().itemsize, dtype=dtype)
    bytes_file += bytearray(write_tag.tobytes())
    bytes_file += bytearray(np.array(values, dtype=dtype).tobytes())
    bytes_file += bytearray(write_tag.tobytes())
    return bytes_file


def _write_global_header(sdf: SarracenDataFrame,
                         def_int: np.dtype,
                         def_real: np.dtype):

    params_dict = sdf.params
    dtypes = [def_int, np.int8, np.int16, np.int32, np.int64, def_real, np.float32, np.float64]
    global_headers = [GlobalHeaderElement(dt, [], []) for dt in dtypes]
    used_keys = set()

    for ghe in global_headers:
        for key in params_dict:
            if _invalid_key(key, used_keys):
                continue
            if isinstance(params_dict[key], ghe.d_type):
                ghe.tags.append(key)
                used_keys.add(key)
                ghe.values.append(params_dict[key])

    bytes_file = bytearray()

    for header in global_headers:
        nvars = np.array([len(header.tags)], dtype='int32')
        write_tag = np.array([len(nvars)], dtype='int32')
        bytes_file += bytearray(write_tag.tobytes())
        nvars = np.array([len(header.tags)], dtype='int32')
        bytes_file += bytearray(nvars.tobytes())
        bytes_file += bytearray(write_tag.tobytes())

        if nvars == 0:
            continue

        bytes_file += _write_global_header_tags_and_values(header.tags, header.values, header.d_type)

    return bytes_file


def _count_num_dt_arrays(test_sdf, dt):
    count_dt = 0
    for element in test_sdf.dtypes.values:
        if element == dt:
            count_dt += 1
    return count_dt


def _get_array_tags(test_sdf, dt):
    return list(test_sdf.select_dtypes(include=[dt]).columns)


def _get_last_index(sdf):
    return 1 if sdf.index[-1] == 0 else sdf.shape[0]


def _write_value_arrays(sdf: SarracenDataFrame,
                        def_int: np.dtype,
                        def_real: np.dtype):

    dtypes = [def_int, np.int8, np.int16, np.int32, np.int64, def_real, np.float32, np.float64]
    num_blocks = len(sdfs)

    # number of blocks -- 1 for now while testing (block 2 is sink particles)
    file = _write_fortran_block([1], np.int32)

    sdf_data_type_tags = []
    nvars = np.array([_get_last_index(sdf)], dtype='int64')
    data_type_tags = []
    data_types_used = set()


    for d_type in dtypes:
        tags = _get_array_tags(sdf, d_type) if d_type not in data_types_used else []
        data_type_tags.append(DataTypeTags(d_type, tags))
        data_types_used.add(d_type)

    counts = np.array([len(dt_tags.tags) for dt_tags in data_type_tags], dtype='int32')

    write_tag = np.array([len(nvars) * nvars.dtype.itemsize + len(counts) * counts.dtype.itemsize], dtype=np.int32)
    file += write_tag.tobytes() + nvars.tobytes() + counts.tobytes() + write_tag.tobytes()
    sdf_data_type_tags.append(SdfDataTypeTags(data_type_tags, sdf))

    for sdf_dt_tags in sdf_data_type_tags:
        for dt_tags in sdf_dt_tags.data_type_tags:
            if len(dt_tags.tags) > 0:
                for tag in dt_tags.tags:
                    file += bytearray(write_tag.tobytes())
                    file += bytearray(map(ord, tag.ljust(16)))
                    file += bytearray(write_tag.tobytes())

                    file += _write_fortran_block(list(sdf_dt_tags.sdf[tag]), dt_tags.d_type)
    return file


def write_phantom(sdf: SarracenDataFrame,
                  filename: str = 'dumpfile_00000'):

    def_int = sdf.params['def_int_dtype']
    def_real = sdf.params['def_real_dtype']

    ph_file = _write_capture_pattern(def_int, def_real)
    ph_file += _write_file_identifier(sdf)
    ph_file += _write_global_header(sdf, def_int, def_real)
    value_arrays = _write_value_arrays(sdf, def_int, def_real)
    ph_file += value_arrays

    with open(filename, 'wb') as phantom_file:
        phantom_file.write(ph_file)
        phantom_file.close()

    return phantom_file
