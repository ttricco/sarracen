import numpy as np

from ..sarracen_dataframe import SarracenDataFrame

from typing import List, Tuple, Union, Type


def _write_fortran_block(value: list,
                         dtype: type) -> bytearray:
    write_tag = np.array([len(value) * dtype().itemsize], dtype=np.int32)
    file = bytearray(write_tag.tobytes())
    file += bytearray(np.array(value, dtype=dtype).tobytes())
    file += bytearray(write_tag.tobytes())
    return file


def _write_file_identifier(sdf: SarracenDataFrame) -> bytearray:
    if sdf.params is None or 'file_identifier' not in sdf.params:
        raise KeyError("'file_identifier' missing from params in this "
                       "SarracenDataFrame.")
    file_id = sdf.params['file_identifier'].ljust(100)
    file_id = list(map(ord, file_id))
    file = _write_fortran_block(file_id, dtype=np.uint8)
    return file


def _write_capture_pattern(def_int: Type[np.generic],
                           def_real: Type[np.generic],
                           iversion: int = 1) -> bytearray:
    write_tag = np.array([16 + def_real().itemsize], dtype=np.int32)
    i1 = np.array([60769], dtype=def_int)
    r2 = np.array([60878], dtype=def_real)
    i2 = np.array([60878], dtype=np.int32)
    iversion_arr = np.array([iversion], dtype=np.int32)
    i3 = np.array([690706], dtype=np.int32)

    capture_pattern = bytearray(write_tag.tobytes())
    capture_pattern += bytearray(i1.tobytes())
    capture_pattern += bytearray(r2.tobytes())
    capture_pattern += bytearray(i2.tobytes())
    capture_pattern += bytearray(iversion_arr.tobytes())
    capture_pattern += bytearray(i3.tobytes())
    capture_pattern += bytearray(write_tag.tobytes())

    return capture_pattern


def _rename_duplicate(tag: str) -> str:
    if len(tag) > 1 and tag[-2] == '_' and tag[-1].isdigit():
        tag = tag[:-2]

    return tag


def _write_global_header_tags_and_values(tags: list,
                                         values: list,
                                         dtype: type) -> bytearray:
    tags = [_rename_duplicate(tag) for tag in tags]
    tags = [list(map(ord, tag.ljust(16))) for tag in tags]
    tags = [c for tag in tags for c in tag]

    file = _write_fortran_block(tags, np.uint8)
    file += _write_fortran_block(values, dtype)

    return file


def _write_global_header(sdf: SarracenDataFrame,
                         def_int: Type[np.number],
                         def_real: Type[np.number]) -> bytearray:
    params_dict = _remove_invalid_keys(sdf)
    dtypes = [def_int, np.int8, np.int16, np.int32, np.int64,
              def_real, np.float32, np.float64]
    header_data: List[Tuple[type, list, list]] = [(dtype, [], [])
                                                  for dtype in dtypes]
    used_keys = set()

    for dtype, tags, values in header_data:
        for key in params_dict:
            if key in used_keys:
                continue
            if isinstance(params_dict[key], dtype):
                tags.append(key)
                values.append(params_dict[key])
                used_keys.add(key)

    file = bytearray()

    for dtype, tags, values in header_data:
        nvars = len(tags)
        file += _write_fortran_block([nvars], dtype=np.int32)

        if nvars > 0:
            file += _write_global_header_tags_and_values(tags, values, dtype)

    return file


def _remove_invalid_keys(sdf: SarracenDataFrame) -> dict:
    if sdf.params is None:
        raise ValueError("Parameters are not set in this SarracenDataFrame.")
    exclude = ['file_identifier', 'mass', 'def_int_dtype',
               'def_real_dtype', 'iversion']
    return {k: v for k, v in sdf.params.items() if k not in exclude}


def _get_array_tags(test_sdf: SarracenDataFrame, dt: Type[np.number]) -> list:
    return list(test_sdf.select_dtypes(include=[dt]).columns)


def _get_last_index(sdf: SarracenDataFrame) -> int:
    return 1 if sdf.index[-1] == 0 else sdf.shape[0]


def _write_value_arrays(data: SarracenDataFrame,
                        def_int: Type[np.number],
                        def_real: Type[np.number],
                        sinks: Union[SarracenDataFrame,
                                     None] = None) -> bytearray:

    dtypes = [def_int, np.int8, np.int16, np.int32, np.int64,
              def_real, np.float32, np.float64]

    nblocks = 2 if sinks is not None else 1
    file = _write_fortran_block([nblocks], np.int32)

    sdf_list = [data, sinks] if sinks is not None else [data]
    sdf_dtype_info = []

    for sdf in sdf_list:
        nvars = np.array([_get_last_index(sdf)], dtype=np.int64)
        dtype_tags = []
        used = set()

        for dtype in dtypes:
            tags = _get_array_tags(sdf, dtype) if dtype not in used else []
            dtype_tags.append((dtype, tags))
            used.add(dtype)

        counts = np.array([len(tags) for _, tags in dtype_tags],
                          dtype=np.int32)
        write_tag = np.array([len(nvars) * nvars.dtype.itemsize
                              + len(counts) * counts.dtype.itemsize],
                             dtype=np.int32)

        file += (write_tag.tobytes() + nvars.tobytes() + counts.tobytes()
                 + write_tag.tobytes())

        sdf_dtype_info.append((sdf, dtype_tags))

    for sdf, dtype_tags in sdf_dtype_info:
        for dtype, tags in dtype_tags:
            if tags:
                for tag in tags:
                    file += _write_fortran_block(list(map(ord, tag.ljust(16))),
                                                 dtype=np.uint8)
                    file += _write_fortran_block(list(sdf[tag]), dtype)
    return file


def write_phantom(filename: str,
                  data: SarracenDataFrame,
                  sinks: Union[SarracenDataFrame, None] = None) -> None:
    if data.isnull().values.any():
        raise ValueError("particle DataFrame contains NaNs or missing values.")

    if sinks is not None and sinks.isnull().values.any():
        raise ValueError("sinks DataFrame contains NaNs or missing values.")

    if data.params is None:
        raise ValueError("Parameters are not set in this SarracenDataFrame.")

    def_int = data.params['def_int_dtype']
    def_real = data.params['def_real_dtype']

    file = _write_capture_pattern(def_int, def_real)
    file += _write_file_identifier(data)
    file += _write_global_header(data, def_int, def_real)
    file += _write_value_arrays(data, def_int, def_real, sinks)

    with open(filename, 'wb') as phantom_file:
        phantom_file.write(file)
        phantom_file.close()
