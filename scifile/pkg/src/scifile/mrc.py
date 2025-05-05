"""Read and write [MRC/CCP4](https://www.ccpem.ac.uk/mrc-format/mrc2014) map files.


"""

from pathlib import Path

import numpy as np

import scids

HEADER_DTYPE = np.dtype(
    [
        ("N", ("i4", 3)),
        ("MODE", "i4"),  # Mode; indicates type of values stored in data block
        ("NSTART", ("i4", 3)),
        ("M", ("i4", 3)),
        ("CELLA", ("f4", 3)),
        ("CELLB", ("f4", 3)),
        ("MAPCRS", ("i4", 3)),  # map section 1=x,2=y,3=z.
        ("dmin", "f4"),  # Minimum pixel value
        ("dmax", "f4"),  # Maximum pixel value
        ("dmean", "f4"),  # Mean pixel value
        ("ispg", "i4"),  # space group number
        ("nsymbt", "i4"),  # number of bytes in extended header
        ("extra1", "V8"),  # extra space, usage varies by application
        ("exttyp", "S4"),  # code for the type of extended header
        ("nversion", "i4"),  # version of the MRC format
        ("extra2", "V84"),  # extra space, usage varies by application
        (
            "origin",
            [  # Origin of image
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
            ],
        ),
        ("map", "S4"),  # Contains 'MAP ' to identify file type
        ("machst", "u1", 4),  # Machine stamp; identifies byte order
        ("rms", "f4"),  # RMS deviation of densities from mean density
        ("nlabl", "i4"),  # Number of labels with useful data
        ("label", "S80", 10),  # 10 labels of 80 characters
    ]
)


MODE = {
    0: np.int8,
    1: np.int16,
    2: np.float32,
    6: np.uint16,
    12: np.float16,
}

HEADER_LEN = 1024  # Bytes.


def parse(file: bytes | Path):
    """Read an MRC/CCP4 file.

    Parameters
    ----------
    file
        MRC/CCP4 file content (in bytes) or path.
    """
    if isinstance(file, Path):
        content = file.read_bytes()
    elif isinstance(file, bytes):
        content = file
    else:
        raise TypeError(f"Expected bytes or Path, but got {type(file).__name__}")

    # Read MACHST bytes directly to determine byte order of the file.
    machst = np.frombuffer(content[212:216], dtype=np.uint8)
    if (machst == [0x44, 0x41, 0x00, 0x00]).all():
        endian = "<"  # Little-endian (Intel)
    elif (machst == [0x11, 0x11, 0x00, 0x00]).all():
        endian = ">"  # Big-endian (old SGI format)
    else:
        raise ValueError(f"Unrecognized MACHST field: {machst.tolist()}")

    # Parse header using correct endianness
    content_int32 = np.frombuffer(content, dtype=endian + "i4", count=256, offset=0)
    content_float32 = np.frombuffer(content, dtype=endian + "f4", count=256, offset=0)
    content_uint8 = np.frombuffer(content, dtype="u1", count=1024, offset=0)

    word_map = content_uint8[208:212].view("S4")[0].decode()
    if word_map != "MAP ":
        raise ValueError(f"Invalid MRC file: missing MAP marker, got {word_map}")

    mode = content_int32[3]
    if mode not in MODE:
        raise ValueError(f"Unsupported MODE: {mode}. Supported modes: {sorted(MODE)}")

    nx_ny_nz = content_int32[0:3]  # grid shape
    nxstart_nystart_nzstart = content_int32[4:7]  # grid origin in unit cell
    mx_my_mz = content_int32[7:10]  # number of spacings
    cella = content_float32[10:13]  # cell dimensions in Angstroms
    cellb = content_float32[13:16]  # cell angles in degrees
    mapc_mapr_maps = content_int32[16:19]  # axes
    dmin_dmax_dmean = content_float32[19:22]
    ispg = content_int32[22]
    nsymbt = content_int32[23]
    exttyp = content_uint8[104:108].view("S4")[0].decode()
    nversion = content_int32[27]
    origin = content_float32[49:52]
    rms = content_float32[54]
    nlabl = content_int32[55]
    labels = [content_uint8[224 + 80 * i : 224 + 80 * (i + 1)].tobytes().rstrip(b"\x00").decode(errors="replace") for i in range(nlabl)]

    word_machst = content_uint8[212:216]
    extended_header = content[HEADER_BYTES_COUNT : HEADER_BYTES_COUNT + nsymbt] if nsymbt > 0 else b""

    real_origin = origin if origin and nversion > 0 else nxstart_nystart_nzstart * cella / mx_my_mz

    map_values = np.frombuffer(
        buffer=content,
        dtype=endian + MODE[mode].dtype.str,
        count=np.prod(nx_ny_nz),
        offset=HEADER_BYTES_COUNT + nsymbt
    ).reshape(nx_ny_nz, order="F")

    grid = scids.grid.from_shape_size_anchor(
        shape=nx_ny_nz,
        size=cella,
        anchor_coord=nxstart_nystart_nzstart * cella / mx_my_mz,
        anchor="lower",
    )

    if np.all(np.isin(map_values, (0, 1))):
        map_values = map_values.astype(np.bool_)
        return scids.volume.ToxelVolume(grid=grid, toxels=map_values)
    return scids.field.from_tensor_grid(tensor=map_values, grid=grid)

