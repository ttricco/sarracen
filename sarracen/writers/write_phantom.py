from typing import Dict
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


def _remove_invalid_keys(params: dict) -> dict:
    """ Remove keys specific to Sarracen."""

    exclude = ['file_identifier', 'mass', 'def_int_dtype',
               'def_real_dtype', 'iversion']
    return {k: v for k, v in params.items() if k not in exclude}


def _validate_particle_counts(sdf: SarracenDataFrame,
                              params: dict) -> dict:
    """ Update params particle counts to match actual particle counts."""

    n_gas = len(sdf) if 'itype' not in sdf.columns else len(sdf[sdf.itype == 1])
    n_dust = 0 if 'itype' not in sdf.columns else len(sdf[sdf.itype == 7])
    n_total = n_gas + n_dust

    # check total particle counts
    if 'nparttot' not in params:
        params['nparttot'] = n_total
    if params['nparttot'] != n_total:
        params['nparttot'] = params['nparttot'].dtype.type(n_total)

    # check gas particle counts
    if 'npartoftype' not in params:
        params['npartoftype'] = n_gas
    if params['npartoftype'] != n_gas:
        params['npartoftype'] = params['npartoftype'].dtype.type(n_gas)

    # check dust particle counts
    if 'itype' in sdf.columns and len(sdf[sdf.itype == 7]) > 0:
        if 'npartoftype_7' not in params:
            params['npartoftype_7'] = n_dust
        if params['npartoftype_7'] != n_dust:
            params['npartoftype_7'] = params['npartoftype_7'].dtype.type(n_dust)

    # check for second set of particle counts
    if 'nparttot_2' in params:
        if 'ntypes' in params:
            ntypes = params['ntypes']
        else:  # guess
            ntypes = sum(1 for key in params.keys() if key.startswith('npartoftype'))
            if ntypes % 2 == 1:  # odd
                raise ValueError("Guessing number of particle types went wrong.")
            ntypes = ntypes // 2

        if params['nparttot_2'] != n_total:
            params['nparttot_2'] = params['nparttot_2'].dtype.type(n_total)
        if params['npartoftype_' + str(ntypes + 1)] != n_gas:
            params['npartoftype_' + str(ntypes+1)] = params['npartoftype_' + str(ntypes+1)].dtype.type(n_gas)
        if params['npartoftype_' + str(ntypes + 7)] != n_dust:
            params['npartoftype_' + str(ntypes+7)] = params['npartoftype_' + str(ntypes+7)].dtype.type(n_dust)

    return params


def _relocate_special_params(param_dicts: List[Dict]) -> None:
    """Move certain parameters to their required dtype indices."""
    # Integer parameters that must be int32
    for param in ['iexternalforce', 'ieos']:
        if param in param_dicts[0]:
            param_dicts[3][param] = param_dicts[0].pop(param)

    # Real parameters that must be float64
    for param in ['udist', 'umass', 'utime', 'umagfd']:
        if param in param_dicts[5]:
            param_dicts[7][param] = param_dicts[5].pop(param)


def _write_global_header(sdf: SarracenDataFrame,
                         def_int: Type[np.number],
                         def_real: Type[np.number]) -> bytearray:

    params_dict = sdf.params.copy()
    params_dict = _remove_invalid_keys(params_dict)
    params_dict = _validate_particle_counts(sdf, params_dict)

    dtypes = [def_int, np.int8, np.int16, np.int32, np.int64,
              def_real, np.float32, np.float64]

    # create params dict per dtype and populate
    param_dtype_dicts: List[Dict] = [dict() for _ in dtypes]
    for k, v in params_dict.items():
        if isinstance(v, def_int):
            param_dtype_dicts[0][k] = v
        elif isinstance(v, np.int8):
            param_dtype_dicts[1][k] = v
        elif isinstance(v, np.int16):
            param_dtype_dicts[2][k] = v
        elif isinstance(v, np.int32):
            param_dtype_dicts[3][k] = v
        elif isinstance(v, np.int64):
            param_dtype_dicts[4][k] = v

        if isinstance(v, def_real):
            param_dtype_dicts[5][k] = v
        elif isinstance(v, np.float32):
            param_dtype_dicts[6][k] = v
        elif isinstance(v, np.float64):
            param_dtype_dicts[7][k] = v

    # some tags live in specific dtypes
    _relocate_special_params(param_dtype_dicts)

    # create header arrays per dtype
    header_data: List[Tuple[type, list, list]] = []
    for i, dtype in enumerate(dtypes):
        tags = []
        values = []
        for k, v in param_dtype_dicts[i].items():
            tags.append(k)
            values.append(v)
        header_data.append((dtype, tags, values))

    file = bytearray()

    # write header arrays in dtype sequence
    for dtype, tags, values in header_data:
        nvars = len(tags)
        file += _write_fortran_block([nvars], dtype=np.int32)

        if nvars > 0:
            file += _write_global_header_tags_and_values(tags, values, dtype)

    return file


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
