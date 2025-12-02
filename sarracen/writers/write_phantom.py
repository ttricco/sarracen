from typing import Dict, List, Union, Type
import numpy as np
import re

from ..sarracen_dataframe import SarracenDataFrame


def _write_fortran_block(value: List[np.generic],
                         dtype: Type[np.generic]) -> bytearray:
    write_tag = np.array([len(value) * dtype().itemsize], dtype=np.int32)
    file = bytearray(write_tag.tobytes())
    file += bytearray(np.array(value, dtype=dtype).tobytes())
    file += bytearray(write_tag.tobytes())
    return file


def _write_file_identifier(sdf: SarracenDataFrame) -> bytearray:
    """ Write the file identifier.

    The file identifier is a 100 character string that encodes the various
    options that were used.

    First character is either 'F' or 'S', corresponding to Full or Small dump.
    Second character is 'T', for a tagged dump file.

    'Phantom' should appear in the name to declare it as a Phantom dump, not
    an sphNG dump file.
    """

    if 'file_identifier' not in sdf.params:
        fileid = "FT:Phantom:Sarracen"
        sdf.params['file_identifier'] = fileid
    file_id = sdf.params['file_identifier'].ljust(100)
    file_id = list(map(ord, file_id))
    file_id = [np.uint8(c) for c in file_id]
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


def _remove_invalid_keys(params: Dict[str, np.generic]) -> Dict[str,
                                                                np.generic]:
    """ Remove keys specific to Sarracen internal use."""

    exclude = ['file_identifier', 'mass', 'def_int_dtype',
               'def_real_dtype', 'iversion']
    return {k: v for k, v in params.items() if k not in exclude}


def _standardize_dtypes(params: Dict[str, np.generic]) -> Dict[str, np.generic]:
    """ Convert all params to numpy dtypes."""

    for k, v in params.items():
        if isinstance(v, int):
            params[k] = np.int32(v)
        if isinstance(v, float):
            params[k] = np.float64(v)

    return params


def _validate_ntypes(sdf: SarracenDataFrame,
                     params: Dict[str, np.generic]) -> Dict[str, np.generic]:
    """ Update params ntypes to reflect particle data."""

    if 'ntypes' in params:
        ntypes = int(params['ntypes'])

    elif 'itype' in sdf.columns:
        # guess using number of itypes
        ntypes = max(8, sdf['itype'].max())

    elif 'massoftype' in params:
        # guess using number of massoftype
        ntypes = 8
        for key in params:
            if key.startswith('massoftype_'):
                suffix = key[len('massoftype_'):]
                if suffix.isdigit():
                    ntypes = max(ntypes, int(suffix))

    elif 'npartoftype' in params:
        # guess using number of npartoftype
        # more difficult because npartoftype is stored as both int32 and int64
        # find max suffix for both, then divide the larger by 2
        max_int32 = 8
        max_int64 = 8
        for key in params:
            if key.startswith('npartoftype_'):
                suffix = key[len('npartoftype_'):]
                if suffix.isdigit():
                    if isinstance(params[key], np.int64):
                        max_int64 = max(int(suffix), max_int64)
                    else:
                        max_int32 = max(int(suffix), max_int32)
        # hack
        if max_int32 > max_int64:
            max_int32 = max_int32 // 2
        else:
            max_int64 = max_int64 // 2
        ntypes = max(8, max(max_int64, max_int32))

    else:
        ntypes = 8

    params['ntypes'] = np.int32(ntypes)

    return params


def _validate_particle_counts(sdf: SarracenDataFrame,
                              params: Dict[str,
                                           np.generic]) -> Dict[str,
                                                                np.generic]:
    """
    Update params particle counts to match actual particle counts.

    This will count the number of gas and dust particles based on itype. It
    assumes that if itype is not present, then all particles are gas.

    This will populate nparttot and npartoftype accordingly.
    """

    n_gas = len(sdf[sdf.itype == 1]) if 'itype' in sdf.columns else len(sdf)
    n_dust = len(sdf[sdf.itype == 7]) if 'itype' in sdf.columns else 0
    n_total = len(sdf)

    idust = params['idust'] if 'idust' in params else 7
    ntypes = int(params['ntypes'])

    # check total particle count
    if 'nparttot' not in params:
        params['nparttot'] = np.int32(n_total)
    if params['nparttot'] != n_total:
        params['nparttot'] = type(params['nparttot'])(n_total)

    # check int64 total particle count
    if 'nparttot_2' not in params:
        params['nparttot_2'] = np.int64(n_total)
    if params['nparttot_2'] != n_total:
        params['nparttot_2'] = type(params['nparttot_2'])(n_total)

    # set default values for npartoftypes if not set
    default = {'': n_gas}  # insertion order matters
    default.update({f'_{i}': 0 for i in range(2, ntypes + 1)})
    default[f'_{idust}'] = n_dust
    for i, value in default.items():
        key = f'npartoftype{i}'
        if key not in params:
            params[key] = np.int32(value)

    # set default values for int64 npartoftypes if not set
    default = {f'_{i}': 0 for i in range(ntypes + 1, 2 * ntypes + 1)}
    default[f'_{ntypes + 1}'] = n_gas
    default[f'_{ntypes + idust}'] = n_dust
    for i, value in default.items():
        key = f'npartoftype{i}'
        if key not in params:
            params[key] = np.int64(value)

    # update gas and dust npartoftype if values are incorrect
    if params['npartoftype'] != n_gas:
        params['npartoftype'] = type(params['npartoftype'])(n_gas)
    key = f'npartoftype_{idust}'
    if params[key] != n_dust:
        params[key] = type(params[key])(n_dust)

    # update int64 gas and dust npartoftype if values are incorrect
    key = 'npartoftype_' + str(ntypes + 1)
    if params[key] != n_gas:
        params[key] = type(params[key])(n_gas)
    key = 'npartoftype_' + str(ntypes + idust)
    if params[key] != n_dust:
        params[key] = type(params[key])(n_dust)

    return params


def _validate_particle_masses(sdf: SarracenDataFrame,
                              params: Dict[str,
                                           np.generic]) -> Dict[str,
                                                                np.generic]:
    """ Update params particle masses to match actual particle masses."""

    if 'mass' in params:
        m_gas = params['mass']
    elif 'massoftype' in params:
        m_gas = params['massoftype']
    elif sdf.mcol is not None and 'itype' not in sdf.columns:
       m_gas = sdf[sdf.mcol].iloc[0]
    elif sdf.mcol is not None and 'itype' in sdf.columns:
        m_gas = sdf[sdf.itype == 1][sdf.mcol].iloc[0]
    else:
        raise ValueError("Could not find gas mass.")

    # set default values for massoftype if not set
    default = {'': m_gas}  # insertion order matters
    default.update({f'_{i}': 0 for i in range(2, int(params['ntypes']) + 1)})
    for i, value in default.items():
        key = f'massoftype{i}'
        if key not in params:
            params[key] = np.float64(value)

    idust = params['idust'] if 'idust' in params else 7

    # update massoftype for dust, if dust is present
    if 'itype' in sdf.columns and sdf.mcol is not None:
        m_dust_particle = sdf[sdf.itype == idust][sdf.mcol].iloc[0]

        if params[f'massoftype_{idust}'] == 0:
            params[f'massoftype_{idust}'] = m_dust_particle

        if params[f'massoftype_{idust}'] != m_dust_particle:
            raise ValueError("Mismatch in dust params and particle masses")

    return params


def _relocate_special_params(param_dicts: List[Dict[str,
                                                    np.generic]]) -> None:
    """Move certain parameters to their required dtype indices."""
    # Integer parameters that must be int32
    for param in ['iexternalforce', 'ieos']:
        if param in param_dicts[0]:
            param_dicts[3][param] = param_dicts[0].pop(param)

    # Real parameters that must be float64
    for param in ['udist', 'umass', 'utime', 'umagfd']:
        if param in param_dicts[5]:
            param_dicts[7][param] = param_dicts[5].pop(param)


def _rename_duplicate(tag: str) -> str:
    if len(tag) > 1 and tag[-2] == '_' and tag[-1].isdigit():
        tag = tag[:-2]
    if len(tag) > 2 and tag[-3] == '_' and tag[-2:].isdigit():
        tag = tag[:-3]

    return tag


def _write_global_header_array(tags: List[str],
                               values: List[np.generic],
                               dtype: Type[np.generic]) -> bytearray:
    tags = [_rename_duplicate(tag) for tag in tags]
    tags2: List[List[int]] = [list(map(ord, tag.ljust(16))) for tag in tags]
    tags3: List[np.uint8] = [np.uint8(c) for tag in tags2 for c in tag]

    file = _write_fortran_block(tags3, np.uint8)
    file += _write_fortran_block(values, dtype)

    return file


def sort_key(k):
    """
    Returns (basename, number) where basename is the part before any
    numeric suffix, and number is an integer used for sorting.
    Keys without numeric suffix get number = 0.
    """
    m = re.match(r"^(.*?)(?:_(\d+))?$", k)
    base = m.group(1)
    num = int(m.group(2)) if m.group(2) else 0
    return (base, num)


def _reorder_params(params: Dict[str, np.generic]) -> Dict[str, np.generic]:
    return dict(sorted(params.items(), key=lambda item: sort_key(item[0])))


def _write_global_header(sdf: SarracenDataFrame,
                         def_int: Type[np.generic],
                         def_real: Type[np.generic]) -> bytearray:

    params_dict = sdf.params.copy()
    params_dict = _standardize_dtypes(params_dict)
    params_dict = _validate_ntypes(sdf, params_dict)
    params_dict = _validate_particle_counts(sdf, params_dict)
    params_dict = _validate_particle_masses(sdf, params_dict)
    params_dict = _reorder_params(params_dict)
    params_dict = _remove_invalid_keys(params_dict)

    dtypes = [def_int, np.int8, np.int16, np.int32, np.int64,
              def_real, np.float32, np.float64]

    # create params dict per dtype and populate
    param_dtype_dicts: List[Dict[str, np.generic]] = [dict() for _ in dtypes]
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
    header_data = []
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
        nvars = np.int32(len(tags))
        file += _write_fortran_block([nvars], dtype=np.int32)

        if nvars > 0:
            file += _write_global_header_array(tags, values, dtype)

    return file


def _get_array_tags(sdf: SarracenDataFrame,
                    dtype: Type[np.generic]) -> List[str]:
    return list(sdf.select_dtypes(include=[dtype]).columns)


def _get_last_index(sdf: SarracenDataFrame) -> int:
    return 1 if sdf.index[-1] == 0 else sdf.shape[0]


def _write_array_blocks(data: SarracenDataFrame,
                        def_int: Type[np.generic],
                        def_real: Type[np.generic],
                        sinks: Union[SarracenDataFrame,
                                     None] = None) -> bytearray:

    dtypes = [def_int, np.int8, np.int16, np.int32, np.int64,
              def_real, np.float32, np.float64]

    nblocks = 2 if sinks is not None else 1
    file = _write_fortran_block([np.int32(nblocks)], np.int32)

    sdf_list = [data] if sinks is None else [data, sinks]
    sdf_dtype_info = []

    for sdf in sdf_list:
        nvars = np.array([_get_last_index(sdf)], dtype=np.int64)
        dtype_tags = []
        used = set()  # avoid def_int, def_real duplicates

        for dtype in dtypes:
            tags = _get_array_tags(sdf, dtype) if dtype not in used else []
            dtype_tags.append((dtype, tags))
            used.add(dtype)

        counts = np.array([len(tags) for _, tags in dtype_tags],
                          dtype=np.int32)
        write_tag = np.array([len(nvars) * nvars.dtype.itemsize
                              + len(counts) * counts.dtype.itemsize],
                             dtype=np.int32)

        # A mix of int64 and int32, so custom write
        file += write_tag.tobytes()
        file += nvars.tobytes()
        file += counts.tobytes()
        file += write_tag.tobytes()

        sdf_dtype_info.append((sdf, dtype_tags))

    for sdf, dtype_tags in sdf_dtype_info:
        for dtype, tags in dtype_tags:
            for tag in tags:
                base_tag = _rename_duplicate(tag).ljust(16)
                file += _write_fortran_block(list(map(ord, base_tag)),
                                             dtype=np.uint8)
                file +=  _write_fortran_block(list(sdf[tag]), dtype)
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

    def_int = data.params.get('def_int_dtype', np.int32)
    def_real = data.params.get('def_real_dtype', np.float64)

    file = _write_capture_pattern(def_int, def_real)
    file += _write_file_identifier(data)
    file += _write_global_header(data, def_int, def_real)
    file += _write_array_blocks(data, def_int, def_real, sinks)

    with open(filename, 'wb') as phantom_file:
        phantom_file.write(file)
        phantom_file.close()
