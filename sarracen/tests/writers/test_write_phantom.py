import pandas as pd
import numpy as np
import sarracen
import tempfile


def _create_capture_pattern(def_int, def_real):
    """ Construct capture pattern. """

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


def _create_file_identifier():
    """ Construct 100-character file identifier. """

    read_tag = np.array([13], dtype='int32')
    file_identifier = "Test of read_phantom".ljust(100)
    bytes_file = bytearray(read_tag.tobytes())
    bytes_file += bytearray(map(ord, file_identifier))
    bytes_file += bytearray(read_tag.tobytes())
    return bytes_file


def _create_global_header(massoftype=1e-6, massoftype_7=None,
                          def_int=np.int32, def_real=np.float64):
    """ Construct global variables. Only massoftype in this example. """

    read_tag = np.array([13], dtype='int32')
    bytes_file = bytearray()
    for i in range(8):  # loop over 8 dtypes
        bytes_file += bytearray(read_tag.tobytes())
        nvars = (i == 5) + (massoftype_7 is not None)
        if i == 5:  # default real
            nvars = np.array([nvars], dtype='int32')
        else:
            nvars = np.array([0], dtype='int32')
        bytes_file += bytearray(nvars.tobytes())
        bytes_file += bytearray(read_tag.tobytes())

        if i == 5:  # default real
            bytes_file += bytearray(read_tag.tobytes())
            bytes_file += bytearray(map(ord, "massoftype".ljust(16)))
            if massoftype_7 is not None:
                bytes_file += bytearray(map(ord, "massoftype_7".ljust(16)))
            bytes_file += bytearray(read_tag.tobytes())

        if i == 5:
            bytes_file += bytearray(read_tag.tobytes())
            bytes_file += bytearray(np.array([massoftype], dtype=def_real))
            if massoftype_7 is not None:
                bytes_file += bytearray(np.array([massoftype_7], dtype=def_real))
            bytes_file += bytearray(read_tag.tobytes())

    return bytes_file


def _create_particle_array(tag, data, dtype=np.float64):
    read_tag = np.array([13], dtype='int32')
    bytes_file = bytearray(read_tag.tobytes())
    bytes_file += bytearray(map(ord, tag.ljust(16)))
    bytes_file += bytearray(read_tag.tobytes())

    bytes_file += bytearray(read_tag.tobytes())
    bytes_file += bytearray(np.array(data, dtype=dtype).tobytes())
    bytes_file += bytearray(read_tag.tobytes())
    return bytes_file


def get_df():
    bytes_file = _create_capture_pattern(np.int32, np.float64)
    bytes_file += _create_file_identifier()
    bytes_file += _create_global_header()

    # create 1 block for gas
    read_tag = np.array([13], dtype='int32')
    bytes_file += bytearray(read_tag.tobytes())
    nblocks = np.array([1], dtype='int32')
    bytes_file += bytearray(nblocks.tobytes())
    bytes_file += bytearray(read_tag.tobytes())

    # 8 particles storing 4 real arrays (x, y, z, h)
    bytes_file += bytearray(read_tag.tobytes())
    n = np.array([8], dtype='int64')
    nums = np.array([0, 0, 0, 0, 0, 4, 0, 0], dtype='int32')
    bytes_file += bytearray(n.tobytes())
    bytes_file += bytearray(nums.tobytes())
    bytes_file += bytearray(read_tag.tobytes())

    # write 4 particle arrays
    bytes_file += _create_particle_array("x", [0, 0, 0, 0, 1, 1, 1, 1])
    bytes_file += _create_particle_array("y", [0, 0, 1, 1, 0, 0, 1, 1])
    bytes_file += _create_particle_array("z", [0, 1, 0, 1, 0, 1, 0, 1])
    bytes_file += _create_particle_array("h", [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1])

    with tempfile.NamedTemporaryFile() as fp:
        fp.write(bytes_file)
        fp.seek(0)

        sdf = sarracen.read_phantom(fp.name, separate_types='all')
    return sdf


def get_gas_dust_particles():

    bytes_file = _create_capture_pattern(np.int32, np.float64)
    bytes_file += _create_file_identifier()
    bytes_file += _create_global_header(massoftype_7=1e-4)

    # create 1 block for gas
    read_tag = np.array([13], dtype='int32')
    bytes_file += bytearray(read_tag.tobytes())
    nblocks = np.array([1], dtype='int32')
    bytes_file += bytearray(nblocks.tobytes())
    bytes_file += bytearray(read_tag.tobytes())

    # 8 particles storing 4 real arrays (x, y, z, h)
    bytes_file += bytearray(read_tag.tobytes())
    n = np.array([16], dtype='int64')
    nums = np.array([0, 1, 0, 0, 0, 4, 0, 0], dtype='int32')
    bytes_file += bytearray(n.tobytes())
    bytes_file += bytearray(nums.tobytes())
    bytes_file += bytearray(read_tag.tobytes())

    # write 5 gas/dust particle arrays
    bytes_file += _create_particle_array("itype", [1, 1, 1, 1, 1, 1, 1, 1,
                                         7, 7, 7, 7, 7, 7, 7, 7], np.int8)
    bytes_file += _create_particle_array("x", [0, 0, 0, 0, 1, 1, 1, 1,
                                               0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5])
    bytes_file += _create_particle_array("y", [0, 0, 1, 1, 0, 0, 1, 1,
                                               0.5, 0.5, 1.5, 1.5, 0.5, 0.5, 1.5, 1.5])
    bytes_file += _create_particle_array("z", [0, 1, 0, 1, 0, 1, 0, 1,
                                               0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5])
    bytes_file += _create_particle_array("h", [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1,
                                               1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1])
    return bytes_file

def get_gas_sink_particles():

    bytes_file = _create_capture_pattern(np.int32, np.float64)
    bytes_file += _create_file_identifier()
    bytes_file += _create_global_header()

    # create 1 block for gas
    read_tag = np.array([13], dtype='int32')
    bytes_file += bytearray(read_tag.tobytes())
    nblocks = np.array([2], dtype='int32')
    bytes_file += bytearray(nblocks.tobytes())
    bytes_file += bytearray(read_tag.tobytes())

    # 8 particles storing 4 real arrays (x, y, z, h)
    bytes_file += bytearray(read_tag.tobytes())
    n = np.array([8], dtype='int64')
    nums = np.array([0, 0, 0, 0, 0, 4, 0, 0] , dtype='int32')
    bytes_file += bytearray(n.tobytes())
    bytes_file += bytearray(nums.tobytes())
    bytes_file += bytearray(read_tag.tobytes())

    bytes_file += bytearray(read_tag.tobytes())
    n = np.array([1], dtype='int64')
    nums = np.array([0, 0, 0, 0, 0, 7, 0, 0] , dtype='int32')
    bytes_file += bytearray(n.tobytes())
    bytes_file += bytearray(nums.tobytes())
    bytes_file += bytearray(read_tag.tobytes())

    # write 4 gas particle arrays
    bytes_file += _create_particle_array("x", [0, 0, 0, 0, 1, 1, 1, 1])
    bytes_file += _create_particle_array("y", [0, 0, 1, 1, 0, 0, 1, 1])
    bytes_file += _create_particle_array("z", [0, 1, 0, 1, 0, 1, 0, 1])
    bytes_file += _create_particle_array("h", [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1])

    # write 7 sink particle arrays
    bytes_file += _create_particle_array("x", [0.000305])
    bytes_file += _create_particle_array("y", [-0.035809])
    bytes_file += _create_particle_array("z", [-0.000035])
    bytes_file += _create_particle_array("h", [1.0])
    bytes_file += _create_particle_array("spinx", [-3.911744e-8])
    bytes_file += _create_particle_array("spiny", [-1.326062e-8])
    bytes_file += _create_particle_array("spinz", [0.00058])

    return bytes_file


def test_sink_particles(): #PASSES
    with tempfile.NamedTemporaryFile() as fp:
        fp.write(get_gas_sink_particles())
        fp.seek(0)

        test_sdfs = sarracen.read_phantom(fp.name)
        phantom_file = sarracen.write_phantom(test_sdfs, fp.name)
        test_sdfs_from_new_file = sarracen.read_phantom(phantom_file.name)
        pd.testing.assert_frame_equal(test_sdfs_from_new_file[0], test_sdfs[0])
        pd.testing.assert_frame_equal(test_sdfs_from_new_file[1], test_sdfs[1])


def test_write_phantom_one_block(): #PASSES

    test_sdf = sarracen.read_phantom('ot_00003')
    phantom_file = sarracen.write_phantom([test_sdf], 'ot_00003')
    test_sdf_from_new_file = sarracen.read_phantom(phantom_file.name)
    pd.testing.assert_frame_equal(test_sdf, test_sdf_from_new_file)


def test_write_phantom_with_sinks_first_block(): #FAILS -> At positional index 3173, first diff: 3174 != 3173
    test_sdfs = sarracen.read_phantom('jet_00158')
    phantom_file = sarracen.write_phantom([test_sdfs[0]], 'jet_00158')
    test_sdf_from_new_file = sarracen.read_phantom(phantom_file.name)
    pd.testing.assert_frame_equal(test_sdfs[0], test_sdf_from_new_file)

def test_write_phantom_sinks(): #FAILS At positional index 3173, first diff: 3174 != 3173

    test_sdf, sinks_sdf = sarracen.read_phantom('jet_00158')
    phantom_file = sarracen.write_phantom([test_sdf, sinks_sdf], 'jet_00158')
    test_sdf_from_new_file,  test_sdf_from_new_file_sinks = sarracen.read_phantom(phantom_file.name)
    pd.testing.assert_frame_equal(test_sdf, test_sdf_from_new_file)
    pd.testing.assert_frame_equal(sinks_sdf, test_sdf_from_new_file_sinks)


