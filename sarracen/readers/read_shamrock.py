from typing import IO, List, Dict, Any

import struct
import json
import numpy as np
import pandas as pd

from ..sarracen_dataframe import SarracenDataFrame


class FileReader:
    def __init__(self, file: IO) -> None:
        """
        Initialize the FileReader with a file object.

        Parameters
        ----------
        file : file object
            The file object to read from.

        Notes
        -----
        The file object is stored and the last read position is set to the
        current file position with `file.tell()`.
        """
        self.file = file
        self.last_position = file.tell()  # Keeps track of
        # the last read position

    def read_int64_and_string(self) -> str:
        # Read the 64-bit integer (8 bytes)
        """
        Reads a 64-bit integer and the following string from the file.
        The last position is updated to the current file position after
        the string read.

        Returns
        -------
        str
            The read string.
        """
        int_bytes = self.file.read(8)
        if len(int_bytes) != 8:
            raise ValueError("Failed to read 64-bit integer")

        # Unpack the 64-bit integer from the bytes
        length = struct.unpack("q", int_bytes)[0]  # 'q' is for
        # signed 64-bit integer

        # Read the string of 'length' bytes
        string_bytes = self.file.read(length)
        if len(string_bytes) != length:
            raise ValueError("Failed to read the full string")

        # Update the last position to the current
        # file pointer after the string read
        self.last_position = self.file.tell()

        # Decode the string from bytes to UTF-8 and return
        return string_bytes.decode("utf-8")

    def read_from_position(self, offset: int, num_bytes: int) -> bytes:
        """
        Read data from a specific position for a specified number of bytes.

        Parameters
        ----------
        offset : int
            The offset from the last read position to start reading.
        num_bytes : int
            The number of bytes to read from the file.

        Returns
        -------
        bytes
            The data read from the file.

        Raises
        ------
        ValueError
            If length of the data read does not correspond to
            the expected number of bytes.
        """
        # Calculate the absolute position to read from.
        position_to_seek = self.last_position + offset

        # Move to the specified position in the file
        self.file.seek(position_to_seek)

        # Now read the specified number of bytes
        data = self.file.read(num_bytes)
        if len(data) != num_bytes:
            raise ValueError("Failed to read the expected number of bytes")

        return data


def decode_bytes_to_doubles(byte_data: bytes) -> List[float]:
    """
    Decodes a byte array into a list of double precision floats.

    Parameters
    ----------
    byte_data : bytes
        The byte array to decode.

    Returns
    -------
    List[float]
        A list of double-precision floating-point numbers decoded from
        the byte array.

    Raises
    ------
    ValueError
        If the length of byte_data is not a multiple of 8.
    """
    # Ensure that the byte_data length is a multiple of 8 (since each
    # double is 8 bytes)
    if len(byte_data) % 8 != 0:
        raise ValueError("The length of the byte data must be a multiple of 8")

    num_doubles = len(byte_data) // 8  # Number of doubles in the byte array

    # Unpack the byte data into a tuple of doubles ('d' format
    # for double-precision floats)
    doubles = struct.unpack(f"{num_doubles}d", byte_data)

    # Return the list of double-precision floats
    return list(doubles)


def get_head_inc(off: int) -> int:
    if off % 8 > 0:
        off += 8 - (off % 8)
    return off


def decode_patchdata(pdat: bytes, pdat_layout: List[Dict[str, Any]]) -> dict:
    """
    Decode a patchdata bytearray into a dictionary of numpy arrays.

    Returns
    -------
    dic_out
        A dictionary with the decoded patchdata.
    """
    print(f"  Decoding patchdata with layout = {pdat_layout}")

    dic_out = {}

    pdl = pdat_layout

    lay_data = pdat[8:8 + len(pdat_layout) * 8]

    # the format start with the prehead lenght, so we skip it
    head = 0
    for i in range(len(pdat_layout)):
        field = pdat_layout[i]
        print(f"  Decoding {field} current head {head}")

        # because of direct GPU with use alignment 64
        nobj = struct.unpack("q", lay_data[head: head + 8])[0]
        print(f"    -> nobj = {nobj}")
        head += 8

        pdl[i]["nobj"] = nobj

    sz = 0
    for field in pdl:
        print(f"  Computing_size {field} current head {head}")
        nobj = field["nobj"]

        if field["type"] == "f64_3":
            sz += get_head_inc(8 * nobj * 3)
        elif field["type"] == "f64":
            sz += get_head_inc(8 * nobj)
        else:
            raise TypeError("No decoder for this type")

    head = 0
    pdat_dat = pdat[8 + len(pdat_layout) * 8:]

    print(sz, len(pdat_dat))

    for field in pdl:
        print(f"  Decoding {field} current head {head}")

        nobj = field["nobj"]

        if field["type"] == "f64_3":
            tmp = pdat_dat[head: head + 8 * nobj * 3]
            print(
                len(tmp),
                len(tmp) % 8,
                len(tmp) % 3,
                len(tmp) % (3 * 8),
                len(pdat_dat),
                head + 8 * nobj * 3,
            )
            data = decode_bytes_to_doubles(pdat_dat[head: head + 8 * nobj * 3])
            array_size = len(data)
            if array_size % (3 * nobj) != 0:
                raise ValueError(
                    f"Array size {array_size} is not equal to {3*nobj}"
                )
            else:
                dic_out[
                    field["field_name"]] = np.array(data).reshape((nobj, 3))

            head += get_head_inc(8 * nobj * 3)
        elif field["type"] == "f64":
            data = decode_bytes_to_doubles(pdat_dat[head: head + 8 * nobj])

            array_size = len(data)
            if array_size % nobj != 0:
                raise ValueError(
                    f"Array size {array_size} is not equal to {nobj}"
                    )

            dic_out[field["field_name"]] = np.array(data)

            head += get_head_inc(8 * nobj)
        else:
            raise TypeError("No decoder for this type")

    return dic_out


class ShamrockDumpReader:

    def __init__(self, file: IO) -> None:
        self.reader = FileReader(file)

        user_header = self.reader.read_int64_and_string()
        scheduler_header = self.reader.read_int64_and_string()
        filecontent_header = self.reader.read_int64_and_string()

        self.user_meta = json.loads(user_header)
        self.sched_meta = json.loads(scheduler_header)
        file_header = json.loads(filecontent_header)

        self.file_map = {}

        for bcount, off, pid in zip(
            file_header["bytecounts"],
            file_header["offsets"],
            file_header["pids"]
        ):
            self.file_map[pid] = {"bytecount": bcount, "offset": off}

    def read_patch(self, pid: np.uint64) -> dict:
        bcount = self.file_map[pid]["bytecount"]
        off = self.file_map[pid]["offset"]

        print(f"Reading patch pid={pid} (offset={off}, bytecount={bcount})")
        patchdata = self.reader.read_from_position(off, bcount)

        print(
            f"Decoding patchdata pid={pid} (offset={off}, "
            f"bytecount={bcount}), "
            f"len={len(patchdata)}"
        )

        return decode_patchdata(patchdata, self.sched_meta["patchdata_layout"])


def read_shamrock(filename: str) -> SarracenDataFrame:
    """
    Read data from a Shamrock native binary format dump file.

    Parameters
    ----------
    filename : str
        Name of the file to be loaded.

    Returns
    -------
    SarracenDataFrame

    Notes
    -----
    For now in the SPH solver there is just gas (no dust). Sinks information
    is not included in the dumps (see the Shamrock documentation
    https://shamrock-code.github.io/Shamrock/mkdocs/features/sph/sinks/)
    """
    full_df = []  # initialize the dataframe

    with open(filename, "rb") as f:
        reader = ShamrockDumpReader(f)

        # read metadata
        metadata = reader.user_meta
        mass = metadata["solver_config"]["gpart_mass"]
        metadata["mass"] = mass

        kernel = metadata["solver_config"]["kernel_id"]
        if kernel[:2] == "M4":
            hfact = 1.2
        elif kernel[:2] == "M5":
            hfact = 1.2
        elif kernel[:2] == "M6":
            hfact = 1.0
        else:
            raise KeyError("Unrecognised kernel.")
        metadata["hfact"] = hfact
        # end read metadata

        # read patches
        for pid in reader.file_map.keys():
            print(f"Reading patch {pid}")
            try:
                data = reader.read_patch(pid)
                patch_df = pd.DataFrame()

                for col in data.keys():
                    if data[col].ndim == 1:
                        if col == "hpart":
                            patch_df["h"] = data[col]

                        patch_df[col] = data[col]
                    else:
                        if col == "xyz":
                            patch_df["x"] = data[col][:, 0]
                            patch_df["y"] = data[col][:, 1]
                            patch_df["z"] = data[col][:, 2]
                        elif col == "vxyz":
                            patch_df["vx"] = data[col][:, 0]
                            patch_df["vy"] = data[col][:, 1]
                            patch_df["vz"] = data[col][:, 2]
                        else:
                            patch_df[col + "x"] = data[col][:, 0]
                            patch_df[col + "y"] = data[col][:, 1]
                            patch_df[col + "z"] = data[col][:, 2]

                full_df.append(patch_df)

            except Exception as e:
                print(f"Error reading patch {pid}: {e}")
                continue

    if not full_df:
        raise ValueError("No patch in this file!")

    final_df = pd.concat(
        full_df, ignore_index=True
    )  # combine in one df with coherent indexing

    return SarracenDataFrame(final_df, metadata)
