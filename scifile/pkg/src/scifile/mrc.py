"""Read and write [MRC/CCP4](https://www.ccpem.ac.uk/mrc-format) map files.

MRC/CCP4 is a file format used for storing 3D volumetric data,
such as electron density maps and tomograms.
The format is widely used in structural biology and electron microscopy.

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
    3: "complex-int16",
    4: np.complex64,
    6: np.uint16,
    12: np.float16,
    101: "packed-4bit",
}

MACHINE_STAMP_LITTLE_ENDIAN = [0x44, 0x41, 0x00, 0x00]
MACHINE_STAMP_BIG_ENDIAN = [0x11, 0x11, 0x00, 0x00]


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
    cell_b: tuple[float, float, float] | np.ndarray
    ispg: int = 0
    extra: bytes = b"\x00" * 8
    exttyp: str = ""
    nversion: int = 0
    extra2: bytes = b"\x00" * 84
    origin: tuple[float, float, float] | np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
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
        if self.origin.shape != (3,):
            raise ValueError("origin must be a 3-element float32 array")
        self.cell_a = np.array(self.cell_a, dtype=np.float32)
        self.cell_b = np.array(self.cell_b, dtype=np.float32)
        if self.cell_a.shape != (3,) or self.cell_b.shape != (3,):
            raise ValueError("cell_a and cell_b must be 3-element float32 vectors")
        if len(self.extra) != 8:
            raise ValueError("extra must be exactly 8 bytes")
        if len(self.extra2) != 84:
            raise ValueError("extra2 must be exactly 84 bytes")
        if not self.exttyp.isascii() or not self.exttyp.isprintable():
            raise ValueError("exttyp must be ASCII printable")
        if self.mode == 3 and not np.issubdtype(self.data.dtype, np.complexfloating):
            raise TypeError("MODE 3 requires complex-valued data")
        if self.mode == 4 and self.data.dtype != np.complex64:
            raise TypeError("MODE 4 requires complex64 dtype")
        if self.mode == 101 and not np.all((self.data >= 0) & (self.data <= 15)):
            raise ValueError("MODE 101 data must be integers in [0, 15]")
        return

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
        # Write header
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
        header["nsymbt"] = self.nsymbt
        header["extra"] = self.extra
        header["exttyp"] = self.exttyp.encode("ascii").ljust(4, b"\x00")[:4]
        header["nversion"] = self.nversion
        header["origin"] = self.origin
        header["map"] = b"MAP "
        header["machst"] = np.array(
            MACHINE_STAMP_LITTLE_ENDIAN if self.endian == "little" else MACHINE_STAMP_BIG_ENDIAN,
            dtype=np.uint8
        )
        header["rms"] = self.rms
        header["nlabl"] = self.nlabl
        header["extra2"] = self.extra2
        # Encode up to 10 labels, each 80 bytes
        padded_labels = np.zeros((10,), dtype="S80")
        for i, label in enumerate(self.labels[:10]):
            padded_labels[i] = label.encode("ascii", errors="replace")[:80].ljust(80, b"\x00")
        header["label"] = padded_labels

        # Write data to bytes
        endian = "<" if self.endian == "little" else ">"
        if self.mode in (0, 1, 2, 6, 12):
            dtype = MODE[self.mode]
            data_bytes = self.data.astype(np.dtype(dtype).newbyteorder(endian)).tobytes(order="F")
        elif self.mode == 3:
            # Complex int16: pack as interleaved int16 (real, imag)
            arr = np.empty(self.data.size * 2, dtype=endian + "i2")
            arr[0::2] = np.real(self.data).astype(endian + "i2").ravel(order="F")
            arr[1::2] = np.imag(self.data).astype(endian + "i2").ravel(order="F")
            data_bytes = arr.tobytes()
        elif self.mode == 4:
            # Complex float32: write as np.complex64
            data_bytes = self.data.astype(endian + "c8").tobytes(order="F")
        elif self.mode == 101:
            # 4-bit packed: two voxels per byte
            flat = self.data.astype(np.uint8).ravel(order="F")
            if flat.size % 2 != 0:
                flat = np.append(flat, 0)  # pad
            packed = ((flat[::2] & 0x0F) << 4) | (flat[1::2] & 0x0F)
            data_bytes = packed.astype(np.uint8).tobytes()
        else:
            raise ValueError(f"Unsupported MODE {self.mode} during serialization")
        return header.tobytes() + self.extended_header + data_bytes


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

    if len(content) <= HEADER_BYTES_COUNT:
        raise ValueError("File too small to contain valid MRC data.")

    # Read MACHST bytes directly to determine byte order of the file.
    machst = np.frombuffer(content[212:216], dtype=np.uint8)
    if (machst == np.array(MACHINE_STAMP_LITTLE_ENDIAN, dtype=np.uint8)).all():
        endian = "little"
        np_endian = "<"
    elif (machst == np.array(MACHINE_STAMP_BIG_ENDIAN, dtype=np.uint8)).all():
        endian = "big"
        np_endian = ">"
    else:
        raise ValueError(f"Unrecognized MACHST field: {machst.tolist()}")

    header = np.frombuffer(
        content[:HEADER_BYTES_COUNT],
        dtype=HEADER_DTYPE.newbyteorder(np_endian),
    )[()]  # Unwrap the 0-dimensional structured array into a plain structured scalar.

    if header["map"].tobytes() != b"MAP ":
        raise ValueError(f"Invalid MRC file: missing MAP marker, got {header['map'].tobytes()}")
    mode = int(header["mode"])
    if mode not in MODE:
        raise ValueError(f"Unsupported MODE {mode}")

    shape = header["n_xyz"]
    offset = HEADER_BYTES_COUNT + int(header["nsymbt"])
    count = np.prod(shape)

    # Extract voxel data
    if mode in (0, 1, 2, 6, 12):
        dtype = np.dtype(MODE[mode]).newbyteorder(np_endian)
        data = np.frombuffer(
            content,
            dtype=dtype,
            count=count,
            offset=offset
        ).reshape(shape, order="F")
    elif mode == 3:
        arr = np.frombuffer(
            content,
            dtype=np_endian + "i2",
            count=count * 2,
            offset=offset
        )
        data = (arr[0::2] + 1j * arr[1::2]).reshape(shape, order="F")
    elif mode == 4:
        data = np.frombuffer(
            content,
            dtype=np_endian + "c8",
            count=count,
            offset=offset
        ).reshape(shape, order="F")
    elif mode == 101:
        packed = np.frombuffer(
            content,
            dtype=np.uint8,
            count=(count + 1) // 2,
            offset=offset
        )
        unpacked = np.empty(count, dtype=np.uint8)
        unpacked[0::2] = (packed >> 4) & 0x0F
        unpacked[1::2] = packed & 0x0F
        data = unpacked[:count].reshape(shape, order="F")
    else:
        raise ValueError(f"MODE {mode} is recognized but not implemented")

    return MrcFile(
        data=data,
        mode=mode,
        nstart_xyz=header["nstart_xyz"],
        m_xyz=header["m_xyz"],
        cell_a=header["cell_a"],
        cell_b=header["cell_b"],
        ispg=header["ispg"],
        extra=bytes(header["extra"]),
        exttyp=header["exttyp"].tobytes().decode("ascii", errors="replace").strip(),
        nversion=header["nversion"],
        extra2=bytes(header["extra2"]),
        origin=(
            header["origin"] if header["nversion"] > 0
            else header["nstart_xyz"] * header["cell_a"] / header["m_xyz"]
        ),
        labels=[
            label.rstrip(b"\x00").decode("ascii", errors="replace")
            for label in header["label"][:header["nlabl"]]
        ],
        extended_header=(
            content[HEADER_BYTES_COUNT:HEADER_BYTES_COUNT + int(header["nsymbt"])]
            if header["nsymbt"] > 0 else b""
        ),
        endian=endian,
    )
