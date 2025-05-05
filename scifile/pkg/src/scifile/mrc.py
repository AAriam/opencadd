"""Read and write [MRC/CCP4](https://www.ccpem.ac.uk/mrc-format) map files.

References
----------
- [MRC/CCP4 2014 file format specification](https://www.ccpem.ac.uk/mrc-format/mrc2014)
- [MRC2014: Extensions to the MRC format header for electron cryo-microscopy and tomography](https://www.sciencedirect.com/science/article/pii/S104784771500074X)
- [MRC2020: improvements to Ximdisp and the MRC image-processing programs](https://journals.iucr.org/m/issues/2023/05/00/eh5017/index.html)
- [mrcfile Python package](https://github.com/ccpem/mrcfile)
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

if TYPE_CHECKING:
    from typing import Literal


HEADER_BYTES_COUNT = 1024
HEADER_DTYPE = np.dtype(
    [
        ("n_xyz", ("i4", 3)),
        ("mode", "i4"),
        ("nstart_xyz", ("i4", 3)),
        ("m_xyz", ("i4", 3)),
        ("cell_a", ("f4", 3)),
        ("cell_b", ("f4", 3)),
        ("map_xyz", ("i4", 3)),
        ("dmin", "f4"),
        ("dmax", "f4"),
        ("dmean", "f4"),
        ("ispg", "i4"),
        ("nsymbt", "i4"),
        ("extra", "V8"),
        ("exttyp", "S4"),
        ("nversion", "i4"),
        ("extra2", "V84"),
        ("origin", ("f4", 3)),
        ("map", "S4"),
        ("machst", "u1", 4),
        ("rms", "f4"),
        ("nlabl", "i4"),
        ("label", "S80", 10),
    ]
)
MODE = {
    0: np.int8,
    1: np.int16,
    2: np.float32,
    6: np.uint16,
    12: np.float16,
}



@dataclass
class MrcFile:
    """MRC/CCP4 map file (MRC2014 format).

    Attributes
    ----------
    data
        Voxel data with shape (NC, NR, NS) and `mode` data type.
        Note that the axes must always be ordered as XYZ,
        i.e., columns, rows, and sections (fastest to slowest axis).
        This means that MAPC, MAPR, and MAPS header variables
        are always set to 1, 2, and 3, respectively.
    mode
        Data type of voxel values:
        - 0:   8-bit signed integer (range -128 to 127)
        - 1:   16-bit signed integer
        - 2:   32-bit signed real
        - 3:   transform : complex 16-bit integers
        - 4:   transform : complex 32-bit reals
        - 6:   16-bit unsigned integer
        - 12:  16-bit float (IEEE754)
        - 101: 4-bit data packed two per byte
    nstart_xyz
        Index of the first voxel in each axis of the full unit cell.
        This corresponds to [NXSTART, NYSTART, NZSTART],
        i.e., columns, rows, and sections (fastest to slowest axis).
    m_xyz
        Number of intervals (samples - 1)
        along each axis in the full unit cell.
        This corresponds to [MX, MY, MZ],
        i.e., columns, rows, and sections (fastest to slowest axis).
    cell_a
        Physical dimensions of the unit cell in Ångströms (X, Y, Z).
    cell_b
        Angles between unit cell axes in degrees (alpha, beta, gamma).
    ispg
        Space group number (usually 0 or 1; not commonly used).
    extra
        Extra space for application-specific data.
    exttyp
        4-character code for the extended header type:
    nversion
        MRC format version. Use `origin` field only if `nversion > 0`.
    extra2
        Extra space for application-specific data.
    origin
        Real-space coordinates (in Ångströms) of the origin voxel (0, 0, 0).
    labels
        List of up to 10 textual labels, each up to 80 characters.
    extended_header
        Optional binary block immediately following the 1024-byte header.
        Length must equal `nsymbt`.
    endian
        Byte order of the file.
    """
    data: np.ndarray
    mode: Literal[0, 1, 2, 3, 4, 6, 12, 101]
    nstart_xyz: tuple[int, int, int] | np.ndarray
    m_xyz: tuple[int, int, int] | np.ndarray
    cell_a: tuple[float, float, float] | np.ndarray
    cell_b: tuple[float | float | float] | np.ndarray
    map_xyz: tuple[Literal[1, 2, 3], Literal[1, 2, 3], Literal[1, 2, 3]]
    ispg: int = 0
    extra: bytes = b"\x00" * 8
    exttyp: str = ""
    nversion: int = 0
    extra2: bytes = b"\x00" * 84
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    labels: list[str] = field(default_factory=list)
    extended_header: bytes = b""
    endian: Literal["little", "big"] = "little"

    def __post_init__(self):
        if not isinstance(self.data, np.ndarray):
            raise TypeError("data must be a NumPy array")
        if self.mode not in MODE:
            raise ValueError(f"Unsupported mode: {self.mode}")
        if self.data.ndim != 3:
            raise ValueError("data must be 3D")
        if len(self.labels) > 10:
            raise ValueError("Maximum of 10 labels allowed")
        # ensure origin is ndarray
        self.origin = np.array(self.origin, dtype=np.float32)

    @property
    def n_xyz(self) -> np.ndarray:
        """Number of voxels in each dimension.

        This corresponds to the header variables [NX, NY, NZ],
        i.e., for columns, rows, and sections,
        respectively (fastest to slowest axis).
        """
        return np.array(self.data.shape, dtype=np.int32)

    @property
    def map_xyz(self) -> tuple[int, int, int]:
        """Axis corresponding to each dimension of the voxel grid.

        This corresponds to the header variables [MAPC, MAPR, MAPS],
        i.e., for columns, rows, and sections (fastest to slowest axis).

        Each value can be:
        - 1: X axis
        - 2: Y axis
        - 3: Z axis

        For example, [3, 2, 1] means the voxel grid is ordered as ZYX.
        This class assumes the voxel grid is always ordered as XYZ.
        """
        return (1, 2, 3)

    @property
    def dmin(self) -> float:
        """Minimum voxel value."""
        return float(self.data.min())

    @property
    def dmax(self) -> float:
        """Maximum voxel value."""
        return float(self.data.max())

    @property
    def dmean(self) -> float:
        """Mean voxel value."""
        return float(self.data.mean())

    @property
    def rms(self) -> float:
        """Root mean square deviation of voxel values from the mean."""
        return float(np.sqrt(np.mean((self.data - self.dmean) ** 2)))

    @property
    def nsymbt(self) -> int:
        """Number of bytes in the extended header."""
        return len(self.extended_header)

    @property
    def nlabl(self) -> int:
        """Number of labels in the header."""
        return len(self.labels)

    def __bytes__(self) -> bytes:
        """Serialize the MRC/CCP4 file to bytes."""
        header = np.zeros((), dtype=self.endian + HEADER_DTYPE.str)
        header["n_xyz"] = self.n_xyz
        header["mode"] = self.mode
        header["nstart_xyz"] = self.nstart_xyz
        header["m_xyz"] = self.m_xyz
        header["cell_a"] = self.cell_a
        header["cell_b"] = self.cell_b
        header["map_xyz"] = self.map_xyz
        header["dmin"] = self.dmin
        header["dmax"] = self.dmax
        header["dmean"] = self.dmean
        header["ispg"] = self.ispg
        header["extra"] = self.extra
        header["nsymbt"] = self.nsymbt
        header["exttyp"] = self.exttyp.encode("ascii")
        header["nversion"] = self.nversion
        header["origin"] = self.origin
        header["map"] = b"MAP "
        header["machst"] = np.array([0x44, 0x41, 0x00, 0x00], dtype=np.uint8)
        header["rms"] = self.rms
        header["nlabl"] = self.nlabl

        # Encode up to 10 labels, each 80 bytes
        padded_labels = np.zeros((10,), dtype="S80")
        for i, label in enumerate(self.labels[:10]):
            padded_labels[i] = label.encode("ascii", errors="replace")[:80]
        header["label"] = padded_labels
        # Pack header
        header_bytes = header.tobytes()
        # Pad or truncate extended header to match nsymbt
        extended = self.extended_header
        if len(extended) < self.nsymbt:
            extended += b"\x00" * (self.nsymbt - len(extended))
        elif len(extended) > self.nsymbt:
            extended = extended[:self.nsymbt]
        # Serialize volume data in Fortran order with correct dtype and endian
        data_bytes = self.data.astype(self.endian + MODE[self.mode].dtype.str).tobytes(order="F")
        return header_bytes + extended + data_bytes


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

