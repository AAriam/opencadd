"""Read and write AutoDock MAP files.

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


import scids
from scids import exception

if TYPE_CHECKING:
    from typing import Any, Sequence, Literal
    from scids.typing import PathLike


__all__ = [
    "AutodockMapFile",
    "AutodockMapFileHeader",
    "from_filepath",
    "from_content",
    "parse",
]


@dataclass
class AutodockMapFile:
    """AutoDock MAP file.

    Attributes
    ----------
    field
        Numpy array of shape (nelements_x + 1, nelements_y + 1, nelements_z + 1)
        containing the grid point values.
    grid_parameter_files
        Path to the grid parameter file (GPF)
        used to generate this MAP file.
    grid_data_files
        Path to the grid data file (FLD).
    macromolecules
        Path to the macromolecule file (PDBQT)
        used to generate this MAP file.
    """
    field: scids.field.ToxelField
    grid_parameter_files: list[list[Path]] | str = None
    grid_data_files: list[list[Path]] | str = None
    macromolecules: list[list[Path]] | str = None

    def __str__(self) -> str:
        lines = []
        return


@dataclass
class AutodockMapFileHeader:
    """AutoDock MAP file header.

    Attributes
    ----------
    center
        Coordinates of the center of the grid in (x, y, z) format.
    nelements
        The `npts` input parameter used in the GPF,
        i.e., the number of grid points (minus 1)
        in each dimension (x, y, z) in the MAP file.
    spacing
        Spacing between grid points in the MAP file.
    grid_data_file
        Path to the grid data file (FLD).
    grid_parameter_file
        Path to the grid parameter file (GPF)
        used to generate this MAP file.
    macromolecule
        Path to the macromolecule file (PDBQT)
        used to generate this MAP file.
    """
    center: np.ndarray
    nelements: np.ndarray
    spacing: float
    grid_data_file: Path | None = None
    grid_parameter_file: Path | None = None
    macromolecule: Path | None = None

    def __str__(self) -> str:
        lines = []
        for key in ("grid_parameter_file", "grid_data_file", "macromolecule"):
            value = getattr(self, key)
            if value:
                lines.append(f"{key.upper()} {value}")
        lines.append(f"SPACING {self.spacing}")
        lines.append(f"NELEMENTS {' '.join(map(str, self.nelements))}")
        lines.append(f"CENTER {' '.join(map(str, self.center))}")
        return "\n".join(lines)


def from_filepath(
    filepaths: list[list[PathLike]],
    field_dtype: np.dtype = np.single,
    field_names: Sequence[Any] | None = None,
    strict: bool = True,
) -> AutodockMapFile:
    """Read AutoDock MAP files from their filepaths.

    Parameters
    ----------
    filepaths
        Paths to the MAP file.
    field_dtype
        Numpy datatype of the output array.
        Default is 32-bit float (numpy.single).
    strict
        Treat any parsing problems as errors.
        If False, only critical problems are raised as errors,
        and all other problems are reported as warnings.
    """
    return _from_file_or_content(
        mode="file",
        entries=filepaths,
        field_dtype=field_dtype,
        field_names=field_names,
        strict=strict,
    )


def from_content(
    contents: list[list[str]],
    field_dtype: np.dtype = np.single,
    field_names: Sequence[Any] | None = None,
    strict: bool = True,
    filepath: list[list[PathLike]] | None = None,
) -> AutodockMapFile:
    """Read an AutoDock MAP file from its content.

    Parameters
    ----------
    contents
        String content of the MAP file.
    field_dtype
        Numpy datatype of the output array.
        Default is 32-bit float (numpy.single).
    field_names
        Labels for the fields.
    strict
        Treat any parsing problems as errors.
        If False, only critical problems are raised as errors,
        and all other problems are reported as warnings.
    filepath
        Path to the MAP file.
        This is used for error reporting only.
    """
    return _from_file_or_content(
        mode="content",
        entries=contents,
        field_dtype=field_dtype,
        field_names=field_names,
        strict=strict,
        filepath=filepath,
    )


def parse(
    content: str,
    field_dtype: np.dtype = np.single,
    strict: bool = True,
    filepath: PathLike | None = None,
) -> tuple[np.ndarray, AutodockMapFileHeader]:
    """Read an AutoDock MAP file from its content.

    Parameters
    ----------
    content
        String content of the MAP file.
    field_dtype
        Numpy datatype of the output array.
        Default is 32-bit float (numpy.single).
    strict
        Treat any parsing problems as errors.
        If False, only critical problems are raised as errors,
        and all other problems are reported as warnings.
    filepath
        Path to the MAP file.
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
                    filepath=filepath,
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
                    filepath=filepath,
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
            filepath=filepath,
            content=content,
        )
    for metadata_id in ("center", "nelements", "spacing"):
        if metadata_id not in metadata:
            _raise_or_warn(
                f"Missing required header metadata key '{metadata_id.upper()}'.",
                critical=True,
                filepath=filepath,
                content=content,
            )
    grid_shape = metadata["nelements"] + 1
    grid_point_count = np.prod(grid_shape)
    if grid_point_values.size != grid_point_count:
        _raise_or_warn(
            f"Grid point values array has size {grid_point_values.size}, "
            f"but expected {grid_point_count}.",
            critical=True,
            filepath=filepath,
            content=content,
        )
    return grid_point_values.reshape(grid_shape, order="F"), AutodockMapFileHeader(**metadata)


def _from_file_or_content(
    mode: Literal["file", "content"],
    entries: list[list[str | PathLike]],
    field_dtype: np.dtype = np.single,
    field_names: Sequence[Any] | None = None,
    strict: bool = True,
    filepath: list[list[PathLike]] | None = None,
) -> AutodockMapFile:
    """Read a series of AutoDock MAP files.

    Parameters
    ----------
    mode
        Whether `entries` are filepaths or file contents.
    entries
        List of lists of filepaths or file contents.
    field_dtype
        Numpy datatype of the output array.
        Default is 32-bit float (numpy.single).
    field_names
        Labels for the fields.
    strict
        Treat any parsing problems as errors.
        If False, only critical problems are raised as errors,
        and all other problems are reported as warnings.
    filepath
        Path to the MAP file.
        This is used for error reporting only.
    """
    # Parse the first file to get the grid shape first
    first_entry = entries[0][0]
    first_map_content = first_entry if mode == "content" else Path(first_entry).read_text()
    first_map, first_header = parse(
        first_map_content,
        field_dtype=field_dtype,
        strict=strict,
        filepath=filepath[0][0] if filepath else None,
    )

    # Create the grid
    grid_shape = first_header.nelements + 1
    grid = scids.grid.from_center_spacing_shape(
        center=first_header.center,
        spacings=first_header.spacing,
        shape=grid_shape
    )

    # Create the field tensor
    time_point_count = len(entries)
    field_count = len(entries[0])
    toxel_field_shape = (time_point_count, *grid_shape, field_count)
    fields = np.empty(shape=toxel_field_shape, dtype=field_dtype)
    fields[0, ..., 0] = first_map

    # Create filepath arrays
    filepaths_shape = (time_point_count, field_count)
    paths_gpf = np.empty(shape=filepaths_shape, dtype=object)
    paths_fld = np.empty(shape=filepaths_shape, dtype=object)
    paths_pdbqt = np.empty(shape=filepaths_shape, dtype=object)

    # Parse the rest of the files
    for idx_instance, instance in enumerate(entries):
        for idx_map, map_entry in enumerate(instance[1:], start=1):
            map_filepath = (
                Path(filepath[idx_instance][idx_map]) if filepath else None
            ) if mode == "content" else Path(map_entry)
            map_content = map_entry if mode == "content" else map_filepath.read_text()
            field, header = parse(
                content=map_content,
                field_dtype=field_dtype,
                strict=strict,
                filepath=filepath[idx_instance][idx_map] if filepath else None,
            )
            fields[idx_instance, ..., idx_map] = field
            paths_gpf[idx_instance, idx_map] = header.grid_parameter_file
            paths_fld[idx_instance, idx_map] = header.grid_data_file
            paths_pdbqt[idx_instance, idx_map] = header.macromolecule
            # Check for consistency in the header values
            for key in ("center", "nelements", "spacing"):
                if getattr(header, key) != getattr(first_header, key):
                    _raise_or_warn(
                        f"Header '{key.upper()}' values do not match across MAP files. "
                        f"Expected {getattr(first_header, key)}, but got {getattr(header, key)} "
                        f"for MAP file {idx_instance + 1} map {idx_map + 1}.",
                        critical=True,
                        filepath=map_filepath,
                        content=map_content,
                    )
    toxel_field = scids.field.from_tensor_grid(tensor=fields, grid=grid, names=field_names)
    return AutodockMapFile(
        field=toxel_field,
        grid_parameter_file=paths_gpf,
        grid_data_file=paths_fld,
        macromolecule=paths_pdbqt,
    )


def _raise_or_warn(
    message: str,
    *,
    strict: bool = True,
    critical: bool = False,
    filepath: PathLike | None = None,
    content: str | bytes | None = None,
) -> None:
    error = exception.ScidsReadError(
        file_type="autodock_map",
        message=message,
        filepath=filepath,
        content=content,
    )
    exception.raise_or_warn(error, strict=strict, critical=critical)
    return
