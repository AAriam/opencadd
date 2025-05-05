"""Read and write [AutoDock](https://autodock.scripps.edu/) MAP files.

Notes
-----
The first 6 lines of a MAP file are headers
composed of key-value pairs separated by whitespace.
The field values (i.e., calculated energy values for each grid point)
start at line 7, and are written one per line, until the end of file.
They are given as a 1-dimensional array of floats in Fortran order (column-major),
i.e., according to the nested loops z(y(x)),
so the x-coordinate is changing fastest.
The coordinate system is right-handed.

Example MAP file:
```
GRID_PARAMETER_FILE vac1.nbc.gpf
GRID_DATA_FILE 4phv.nbc_maps.fld
MACROMOLECULE 4phv.new.pdbq
SPACING 0.375
NELEMENTS 50 50 80
CENTER -0.026 4.353 -0.038
125.095596
123.634560
116.724602
108.233879
```
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from scifile import exception

if TYPE_CHECKING:
    from scifile.typing import PathLike


__all__ = [
    "AutodockMapFile",
    "AutodockMapFileOptionalHeader",
    "parse",
]

FILETYPE = "autodock_map"


@dataclass(kw_only=True)
class AutodockMapFileOptionalHeader:
    """AutoDock MAP file header attributes.

    These are only the optional attributes.
    They can be unique to each MAP file,
    even when the same grid is used.

    Attributes
    ----------
    grid_parameter_file
        Path to the grid parameter file (GPF)
        used to generate this MAP file.
    grid_data_file
        Path to the grid data file (FLD).
    macromolecule
        Path to the macromolecule file (PDBQT)
        used to generate this MAP file.
    """
    grid_parameter_file: Path | None = None
    grid_data_file: Path | None = None
    macromolecule: Path | None = None


@dataclass(kw_only=True)
class AutodockMapFile(AutodockMapFileOptionalHeader):
    """AutoDock MAP file.

    Attributes
    ----------
    spacing
        The grid-point spacing, i.e.,
        distance between two grid points, in angstroms (Å).
        Grid points are orthogonal and uniformly spaced in AutoDock,
        i.e. this value is for all three dimensions.
    nelements
        Number of grid points (minus 1; for the center point)
        in each dimension (x, y, z).
        This is the same as the `npts` input parameter used in the GPF file.
    center
        Coordinates of the center of the grid in (x, y, z) format.
    field
        Array of shape (nelements_x + 1, nelements_y + 1, nelements_z + 1)
        containing the grid point values.
    """
    spacing: float
    nelements: np.ndarray
    center: np.ndarray
    field: np.ndarray

    def __str__(self) -> str:
        lines = []
        for key in ("grid_parameter_file", "grid_data_file", "macromolecule"):
            value = getattr(self, key)
            if value:
                lines.append(f"{key.upper()} {value}")
        lines.append(f"SPACING {self.spacing}")
        lines.append(f"NELEMENTS {' '.join(map(str, self.nelements))}")
        lines.append(f"CENTER {' '.join(map(str, self.center))}")
        lines.extend(self.field.flatten(order='F').astype(str))
        return f"{"\n".join(lines)}\n"


def parse(
    file: str | bytes | Path,
    field_dtype: np.dtype = np.single,
    strict: bool = True,
    file_label: str | None = None,
) -> AutodockMapFile:
    """Read an AutoDock MAP file.

    Parameters
    ----------
    file
        MAP file content or path.
        If a string, it is treated as the content of the file.
        If bytes, it is decoded to UTF-8.
        If a Path, the file is read as text.
    field_dtype
        Numpy datatype of the output array.
        Default is 32-bit float (numpy.single).
    strict
        Treat any parsing problems as errors.
        If False, only critical problems are raised as errors,
        and all other problems are reported as warnings.
    file_label
        Path or similar identifier for the MAP file.
        This is used for error reporting only.
    """
    # Mapping of metadata keys to their value parsing functions
    token_parser = {
        "grid_parameter_file": lambda x: Path(" ".join(x[1:])),
        "grid_data_file": lambda x: Path(" ".join(x[1:])),
        "macromolecule": lambda x: Path(" ".join(x[1:])),
        "spacing": lambda x: float(x[1]),
        "nelements": lambda x: np.array(x[1:4], dtype=np.short),
        "center": lambda x: np.array(x[1:4], dtype=field_dtype),
    }
    if isinstance(file, Path):
        content = file.read_text()
    elif isinstance(file, bytes):
        content = content.decode("utf-8")
    elif isinstance(file, str):
        content = file
    else:
        raise TypeError(
            f"Expected a string, bytes, or Path object, "
            f"but got {type(file).__name__}."
        )
    lines = content.splitlines()
    metadata = {}
    for line_idx, line in enumerate(lines):
        # First few lines are headers composed of key-value pairs separated by whitespace
        try:
            float(line)
        except ValueError:
            tokens = line.split()
            if len(tokens) in [0, 1]:
                _raise_or_warn(
                    f"Line is empty or contains only one token.",
                    strict=strict,
                    filepath=file_label,
                    content=content,
                    line_idx=line_idx,
                    token=line,
                )
                continue
            metadata_id = tokens[0].lower()
            parser = token_parser.get(metadata_id)
            if parser:
                metadata[metadata_id] = parser(tokens)
            else:
                _raise_or_warn(
                    f"Header contains unknown metadata key '{metadata_id}'.",
                    strict=strict,
                    filepath=file_label,
                    content=content,
                    line_idx=line_idx,
                    token=line,
                )
        else:
            # The first line with a number is the start of the grid point values
            grid_point_values = np.array(lines[line_idx:], dtype=field_dtype)
            break
    else:
        _raise_or_warn(
            "No grid point values found in the file.",
            critical=True,
            filepath=file_label,
            content=content,
        )
    for metadata_id in ("center", "nelements", "spacing"):
        if metadata_id not in metadata:
            _raise_or_warn(
                f"Missing required header metadata key '{metadata_id.upper()}'.",
                critical=True,
                filepath=file_label,
                content=content,
            )
    grid_shape = metadata["nelements"] + 1
    grid_point_count = np.prod(grid_shape)
    if grid_point_values.size != grid_point_count:
        _raise_or_warn(
            f"Grid point values array has size {grid_point_values.size}, "
            f"but expected {grid_point_count}.",
            critical=True,
            filepath=file_label,
            content=content,
        )
    return AutodockMapFile(field=grid_point_values.reshape(grid_shape, order="F"), **metadata)


def _raise_or_warn(
    message: str,
    *,
    strict: bool = True,
    critical: bool = False,
    filepath: PathLike | None = None,
    content: str | bytes | None = None,
) -> None:
    error = exception.SciFileReadError(
        file_type=FILETYPE,
        message=message,
        filepath=filepath,
        content=content,
    )
    exception.raise_or_warn(error, strict=strict, critical=critical)
    return
