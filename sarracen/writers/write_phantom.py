import numpy as np

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


def _write_fortran_block(value: [], dtype: type):
    write_tag = np.array([len(value) * dtype().itemsize], dtype=np.int32)
    file = bytearray(write_tag.tobytes())
    file += bytearray(np.array(value, dtype=dtype).tobytes())
    file += bytearray(write_tag.tobytes())
    return file


def _write_file_identifier(sdf: SarracenDataFrame):
    file_id = sdf.params['file_identifier'].ljust(100)
    file_id = list(map(ord, file_id))
    file = _write_fortran_block(file_id, dtype=np.uint8)
    return file


def _write_capture_pattern(def_int: np.dtype, def_real: np.dtype, iversion: int = 1):
    write_tag = 16 + def_real().itemsize
    write_tag = np.array([write_tag], dtype='int32')
    i1 = np.array([60769], dtype=def_int)
    r2 = np.array([60878], dtype=def_real)
    i2 = np.array([60878], dtype=np.int32)
    iversion = np.array([iversion], dtype=np.int32)
    i3 = np.array([690706], dtype=np.int32)

    capture_pattern = bytearray(write_tag.tobytes())
    capture_pattern += bytearray(i1.tobytes())
    capture_pattern += bytearray(r2.tobytes())
    capture_pattern += bytearray(i2.tobytes())
    capture_pattern += bytearray(iversion.tobytes())
    capture_pattern += bytearray(i3.tobytes())
    capture_pattern += bytearray(write_tag.tobytes())

    return capture_pattern


def _rename_duplicate(tag):
    if len(tag) > 1 and tag[-2] == '_' and tag[-1].isdigit():
        tag = tag[:-2]

    return tag


def _write_global_header_tags_and_values(tags, values, dtype):
    tags = [_rename_duplicate(tag) for tag in tags]
    tags = [list(map(ord, tag.ljust(16))) for tag in tags]
    tags = [c for tag in tags for c in tag]

    file = _write_fortran_block(tags, np.uint8)
    file += _write_fortran_block(values, dtype)

    return file


def _write_global_header(sdf: SarracenDataFrame,
                         def_int: np.dtype,
                         def_real: np.dtype):
    params_dict = _remove_invalid_keys(sdf)
    dtypes = [def_int, np.int8, np.int16, np.int32, np.int64, def_real, np.float32, np.float64]
    global_headers = [GlobalHeaderElement(dt, [], []) for dt in dtypes]
    used_keys = set()

    for gh_element in global_headers:
        for key in params_dict:
            if key in used_keys:
                continue
            if isinstance(params_dict[key], gh_element.d_type):
                gh_element.tags.append(key)
                gh_element.values.append(params_dict[key])
                used_keys.add(key)

    file = bytearray()

    for header in global_headers:
        nvars = len(header.tags)
        file += _write_fortran_block([nvars], dtype=np.int32)

        if nvars == 0:
            continue

        file += _write_global_header_tags_and_values(header.tags, header.values, header.d_type)

    return file


def _remove_invalid_keys(sdf):
    exclude = ['file_identifier', 'mass', 'def_int_dtype', 'def_real_dtype', 'iversion']
    params_dict = {k: v for k, v in sdf.params.items() if k not in exclude}
    return params_dict


def _get_array_tags(test_sdf, dt):
    return list(test_sdf.select_dtypes(include=[dt]).columns)


def _get_last_index(sdf):
    return 1 if sdf.index[-1] == 0 else sdf.shape[0]


def _write_value_arrays(data: SarracenDataFrame,
                        def_int: np.dtype,
                        def_real: np.dtype,
                        sinks: SarracenDataFrame = None):

    dtypes = [def_int, np.int8, np.int16, np.int32, np.int64, def_real, np.float32, np.float64]

    nblocks = 2 if sinks is not None else 1
    file = _write_fortran_block([nblocks], np.int32)

    sdf_and_sinks = [data, sinks] if sinks is not None else [data]

    sdf_data_type_tags = []
    for sdf in sdf_and_sinks:
        nvars = np.array([_get_last_index(sdf)], dtype='int64')
        data_type_tags = []
        data_types_used = set()

        for d_type in dtypes:
            tags = _get_array_tags(sdf, d_type) if d_type not in data_types_used else []
            data_type_tags.append(DataTypeTags(d_type, tags))
            data_types_used.add(d_type)

        counts = np.array([len(dt_tags.tags) for dt_tags in data_type_tags], dtype='int32')

        write_tag = np.array([len(nvars) * nvars.dtype.itemsize
                              + len(counts) * counts.dtype.itemsize], dtype=np.int32)
        file += write_tag.tobytes() + nvars.tobytes() + counts.tobytes() + write_tag.tobytes()

        sdf_data_type_tags.append(SdfDataTypeTags(data_type_tags, sdf))

    for sdf_dt_tags in sdf_data_type_tags:
        for dt_tags in sdf_dt_tags.data_type_tags:
            if len(dt_tags.tags) > 0:
                for tag in dt_tags.tags:
                    file += _write_fortran_block(list(map(ord, tag.ljust(16))), dtype=np.uint8)
                    file += _write_fortran_block(list(sdf_dt_tags.sdf[tag]), dt_tags.d_type)
    return file


def write_phantom(data: SarracenDataFrame, new_file_name: str, sinks: SarracenDataFrame = None):
    if data.isnull().values.any():
        raise ValueError("The data DataFrame contains NaNs or missing values.")

    if sinks is not None and sinks.isnull().values.any():
        raise ValueError("The sinks DataFrame contains NaNs or missing values.")

    def_int = data.params['def_int_dtype']
    def_real = data.params['def_real_dtype']

    file = _write_capture_pattern(def_int, def_real)
    file += _write_file_identifier(data)
    file += _write_global_header(data, def_int, def_real)
    file += _write_value_arrays(data, def_int, def_real, sinks)

    with open(new_file_name, 'wb') as phantom_file:
        phantom_file.write(file)
        phantom_file.close()

    return phantom_file