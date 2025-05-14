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

import arrayer
import numpy as np

from scifile import exception, util

if TYPE_CHECKING:
    from typing import Any, Generator
    from scifile.typing import PathLike, ArrayLike


__all__ = [
    "AutodockMapFile",
    "read",
    "parse",
]

FILETYPE = "autodock_map"


@dataclass
class AutodockMapFile:
    """AutoDock MAP file(s).

    This can represent one map, or a collection of maps
    with the same `nelements`, `spacing`, and `center` values.

    Attributes
    ----------
    field
        An (n+3)-dimensional array (n >= 0)
        containing the grid point values of the map(s).
        The last three dimensions are the grid dimensions (x, y, z),
        which have the shape (nelements[0] + 1, nelements[1] + 1, nelements[2] + 1).
        The first n dimensions are extra dimensions,
        which can be used to represent multiple maps.
    center
        Coordinates of the center point of the grid
        as an array of (x, y, z) values.
    spacing
        The grid-point spacing, i.e.,
        distance between two grid points, in angstroms (Å).
        Grid points are orthogonal and uniformly spaced in AutoDock,
        i.e. this value is for all three dimensions.
    grid_parameter_file
        Path(s) to the grid parameter file(s) (GPF)
        used to generate the map(s).
        If provided, this is an n-dimensional array (n >= 0)
        with the same shape as the first n dimensions of `field`.
    grid_data_file
        Path(s) to the grid data file(s) (FLD).
        If provided, this is an n-dimensional array (n >= 0)
        with the same shape as the first n dimensions of `field`.
    macromolecule
        Path(s) to the macromolecule file(s) (PDBQT)
        used to generate the map(s).
        If provided, this is an n-dimensional array (n >= 0)
        with the same shape as the first n dimensions of `field`.
    """
    field: np.ndarray
    center: np.ndarray
    spacing: float
    grid_parameter_file: Path | np.ndarray | None = None
    grid_data_file: Path | np.ndarray | None = None
    macromolecule: Path | np.ndarray | None = None

    def __post_init__(self):
        self._array_attr_names = ("grid_parameter_file", "grid_data_file", "macromolecule")
        for attr_name in self._array_attr_names:
            attr = getattr(self, attr_name)
            if attr is None:
                continue
            if self.prefix_ndim == 0:
                if not isinstance(attr, Path):
                    raise TypeError(
                        f"Expected a Path object for '{attr_name}', "
                        f"but got {type(attr).__name__}."
                    )
            elif not isinstance(attr, np.ndarray):
                raise TypeError(
                    f"Expected a numpy array for '{attr_name}', "
                    f"but got {type(attr).__name__}."
                )
            elif attr.shape != self.prefix_shape:
                raise ValueError(
                    f"Expected a numpy array with shape {self._extra_shape} for '{attr_name}', "
                    f"but got {attr.shape}."
                )
        return

    @property
    def grid_shape(self) -> np.ndarray:
        """Shape of the grid dimensions (x, y, z) in the map."""
        return np.array(self.field.shape[-3:])

    @property
    def nelements(self) -> np.ndarray:
        """Number of grid points (minus 1; for the center point) in each dimension (x, y, z).

        This is the same as the `npts` input parameter used in the GPF file.
        """
        return self.grid_shape - 1

    @property
    def prefix_ndim(self) -> int:
        """Number of extra dimensions before the grid dimensions."""
        return self.field.ndim - 3

    @property
    def prefix_shape(self) -> np.ndarray:
        """Shape of the extra dimensions before the grid dimensions."""
        return np.array(self.field.shape[:-3], dtype=np.int32)

    def write(self, *index) -> str | Generator[str, None, None]:
        """Write the MAP file(s) to a string.

        Parameters
        ----------
        index
            Index to select a specific (set of) map file(s).
            This can be anything Numpy accepts as an index,
            e.g., a single integer, a slice, a tuple of integers,
            or a boolean array.
            If not provided, all map files are selected.
            This only has an effect if multiple maps are present.

        Returns
        -------
        If a single map file is selected,
        the MAP file content is returned as a string.
        Otherwise, a generator is returned,
        yielding the string content for each selected map file.
        """
        if self.prefix_ndim == 0:
            return self._write(
                field=self.field,
                grid_parameter_file=self.grid_parameter_file,
                grid_data_file=self.grid_data_file,
                macromolecule=self.macromolecule,
            )
        selected_fields = self.field[index]
        # Validate that result contains 3D grids
        if selected_fields.ndim < 3:
            raise ValueError("Selection does not include any 3D grid.")
        if selected_fields.shape[-3:] != self.field.shape[-3:]:
            raise ValueError(
                f"Expected 3D grid shape {self.field.shape[-3:]}, but got {selected_fields.shape[-3:]}. "
                "You may have indexed into the grid dimensions."
            )

        iter_shape = selected_fields.shape[:-3]
        selected_attrs = {}
        for attr_name in self._array_attr_names:
            attr = getattr(self, attr_name)
            if not attr:
                selected_attrs[attr_name] = None
                continue
            selected_attr = attr[index]
            if not isinstance(selected_attr, np.ndarray):
                if selected_fields.ndim != 3:
                    raise ValueError(
                        "The provided index is not valid; "
                        f"it selects a single '{attr_name}' value, "
                        f"but the field has {selected_fields.ndim} dimensions."
                    )
            elif selected_attr.shape != iter_shape:
                raise ValueError(
                    f"Expected array with shape {iter_shape} for '{attr_name}', "
                    f"but got {selected_attr.shape}."
                )
            selected_attrs[attr_name] = selected_attr

        if not iter_shape:
            # No leading dimensions, return a single string
            return self._write(
                field=selected_fields,
                grid_parameter_file=selected_attrs["grid_parameter_file"],
                grid_data_file=selected_attrs["grid_data_file"],
                macromolecule=selected_attrs["macromolecule"],
            )
        # Leading dimensions present, return a generator
        for iter_index in np.ndindex(iter_shape):
            yield self._write(
                field=selected_fields[iter_index],
                grid_parameter_file=selected_attrs["grid_parameter_file"][iter_index] if selected_attrs["grid_parameter_file"] is not None else None,
                grid_data_file=selected_attrs["grid_data_file"][iter_index] if selected_attrs["grid_data_file"] is not None else None,
                macromolecule=selected_attrs["macromolecule"][iter_index] if selected_attrs["macromolecule"] is not None else None,
            )
        return

    def __str__(self) -> str:
        """Return the string representation of the MAP file(s)."""
        files  = self.write()
        return files if self.prefix_ndim == 0 else "\n".join(files)

    def _write(
        self,
        field: np.ndarray,
        grid_parameter_file: Path | None,
        grid_data_file: Path | None,
        macromolecule: Path | None,
    ) -> str:
        """Write a single MAP file.

        Parameters
        ----------
        index
            Index to select a specific map file,
            or None when only one map file is present.

        Returns
        -------
        MAP file content as a string.
        """
        args = locals()
        lines = []
        for key in self._array_attr_names:
            value = args[key]
            if value:
                lines.append(f"{key.upper()} {value}")
        lines.append(f"SPACING {self.spacing}")
        lines.append(f"NELEMENTS {' '.join(map(str, self.nelements))}")
        lines.append(f"CENTER {' '.join(map(str, self.center))}")
        lines.extend(field.flatten(order='F').astype(str))
        return f"{"\n".join(lines)}\n"


def read(
    files: str | bytes | Path | ArrayLike,
    field_dtype: np.dtype = np.float32,
    strict: bool = True,
    file_labels: Any = None,
    nelements: tuple[int, int, int] | np.ndarray | None = None,
    spacing: float | None = None,
    center: tuple[float, float, float] | np.ndarray | None = None,
) -> AutodockMapFile:
    """Read AutoDock MAP files.

    This can read any number of map files
    as long as they all have the same
    `nelements`, `spacing`, and `center` values.

    Parameters
    ----------
    files
        MAP file contents or paths.
        If a string, it is treated as the content of the file.
        If bytes, it is decoded to UTF-8.
        If a Path, the file is read as text.
        You can also pass a regular (i.e. not ragged) array of any shape,
        where each element is any of the above types,
        as long as they all have the same number of grid points.
    field_dtype
        Numpy datatype of the output array.
        Default is 32-bit float (numpy.single).
    strict
        Treat any parsing problems as errors.
        If False, only critical problems are raised as errors,
        and all other problems are reported as warnings.
    file_labels
        Identifiers for the MAP files.
        This is used for error reporting only.
        If provided, this must be the same format as `files`,
        i.e., either a single value or a regular array of the same shape as `files`.
        The values can be any type that can be converted to a string.
    nelements
        Number of grid points (minus 1; for the center point)
        in each dimension (x, y, z).
        This is the same as the `npts` input parameter used in the GPF file.
        If provided, the `nelements` values in the MAP files are ignored,
        so you can use this to override `nelements` in MAP files
        or to read MAP files without an `nelements` header.
    spacing
        The grid-point spacing, i.e.,
        distance between two grid points, in angstroms (Å).
        Grid points are orthogonal and uniformly spaced in AutoDock,
        i.e. this value is for all three dimensions.
        If provided, the `spacing` values in the MAP files are ignored,
        so you can use this to override `spacing` in MAP files
        or to read MAP files without a `spacing` header.
    center
        Coordinates of the center of the grid in (x, y, z) format.
        If provided, the `center` values in the MAP files are ignored,
        so you can use this to override `center` in MAP files
        or to read MAP files without a `center` header.
    """
    if isinstance(files, Path | bytes | str):
        single_file = True
        first_label = file_labels
        first_metadata, first_grid_point_values, first_content = parse(
            file=files,
            strict=strict,
            file_label=first_label,
        )
    else:
        single_file = False
        files = np.asarray(files)
        file_labels = np.asarray(file_labels)
        if files.ndim == 0:
            raise TypeError(
                f"Expected a string, bytes, Path object, or an array of these types, "
                f"but got {type(files).__name__}."
            )
        if file_labels and file_labels.shape != files.shape:
            raise ValueError(
                f"Expected file_labels to have the same shape as files, "
                f"but got {file_labels.shape} and {files.shape}."
            )
        first_label = file_labels.flat[0]
        first_metadata, first_grid_point_values, first_content = parse(
            file=files.flat[0],
            strict=strict,
            file_label=first_label,
        )
    ref_values = {}
    for metadata_id, input_value, caster in (
        ("nelements", nelements, lambda x: np.array(x, dtype=np.int64)),
        ("spacing", spacing, float),
        ("center", center, lambda x: np.array(x, dtype=np.float64)),
    ):
        if input_value is not None:
            ref_values[metadata_id] = caster(input_value)
        elif metadata_id in first_metadata:
            ref_values[metadata_id] = first_metadata[metadata_id]
        else:
            _raise_or_warn(
                f"Missing required header metadata key '{metadata_id.upper()}'.",
                critical=True,
                filepath=first_label,
                content=first_content,
            )
    grid_shape = ref_values["nelements"] + 1
    grid_point_count = np.prod(grid_shape)
    if len(first_grid_point_values) != grid_point_count:
        _raise_or_warn(
            f"Grid point values array has size {first_field.size}, "
            f"but expected {grid_point_count}.",
            critical=True,
            filepath=first_label,
            content=first_content,
        )
    first_field = np.array(first_grid_point_values, dtype=field_dtype).reshape(grid_shape, order="F")
    if single_file:
        return AutodockMapFile(
            field=first_field,
            center=ref_values["center"],
            spacing=ref_values["spacing"],
            grid_parameter_file=first_metadata.get("grid_parameter_file"),
            grid_data_file=first_metadata.get("grid_data_file"),
            macromolecule=first_metadata.get("macromolecule"),
        )
    field = np.empty(shape=(*files.shape, *first_field.shape), dtype=field_dtype)
    grid_parameter_file = np.empty(shape=files.shape, dtype=object)
    grid_data_file = np.empty(shape=files.shape, dtype=object)
    macromolecule = np.empty(shape=files.shape, dtype=object)

    for index in np.ndindex(files.shape):
        file = files[index]
        label = file_labels[index] if file_labels else None
        metadata, grid_point_values, content = parse(
            file=file,
            strict=strict,
            file_label=label,
        )
        if len(grid_point_values) != grid_point_count:
            _raise_or_warn(
                f"Grid point values array has size {len(grid_point_values)}, "
                f"but expected {grid_point_count}.",
                critical=True,
                filepath=label,
                content=content,
            )
        for metadata_id, input_value in (
            ("nelements", nelements),
            ("spacing", spacing),
            ("center", center),
        ):
            if input_value is not None:
                continue
            if metadata_id not in metadata:
                _raise_or_warn(
                    f"Missing required header metadata key '{metadata_id.upper()}'.",
                    stric=strict,
                    filepath=label,
                    content=content,
                )
            elif not arrayer.tensor.is_equal(metadata[metadata_id], ref_values[metadata_id]):
                _raise_or_warn(
                    f"Header metadata key '{metadata_id.upper()}' has value {metadata[metadata_id]}, "
                    f"but expected {ref_values[metadata_id]}.",
                    strict=strict,
                    filepath=label,
                    content=content,
                )
        field[index] = np.array(grid_point_values, dtype=field_dtype).reshape(grid_shape, order="F")
        grid_parameter_file[index] = metadata.get("grid_parameter_file")
        grid_data_file[index] = metadata.get("grid_data_file")
        macromolecule[index] = metadata.get("macromolecule")
    return AutodockMapFile(
        field=field,
        center=ref_values["center"],
        spacing=ref_values["spacing"],
        grid_parameter_file=grid_parameter_file,
        grid_data_file=grid_data_file,
        macromolecule=macromolecule,
    )


def parse(
    file: str | bytes | Path,
    strict: bool = True,
    file_label: str | None = None
) -> tuple[dict[str, Any], list[str], str]:
    """Parse a signle AutoDock MAP file.

    This separates the map metadata (header)
    from the grid point values,
    parses the metadata into a dictionary,
    and the grid point values into a list of strings.

    Parameters
    ----------
    file
        MAP file content or path.
        If a string, it is treated as the content of the file.
        If bytes, it is decoded to UTF-8.
        If a Path, the file is read as text.
    strict
        Treat any parsing problems as errors.
        If False, only critical problems are raised as errors,
        and all other problems are reported as warnings.
    file_label
        Path or similar identifier for the MAP file.
        This is used for error reporting only.

    Returns
    -------
    A 3-tuple containing:
    - Map metadata (header) as a dictionary of following keys (if available):
      - grid_parameter_file: `pathlib.Path`
      - grid_data_file: `pathlib.Path`
      - macromolecule: `pathlib.Path`
      - spacing: `float`
      - nelements: `numpy.ndarray` (shape: `(3,)`, dtype: `numpy.int32`)
      - center: `numpy.ndarray` (shape: `(3,)`, dtype: `numpy.float64`)
    - Map values as a list of strings.
    - The original file content as a string.
    """
    token_parser = {
        "grid_parameter_file": lambda x: Path(" ".join(x[1:])),
        "grid_data_file": lambda x: Path(" ".join(x[1:])),
        "macromolecule": lambda x: Path(" ".join(x[1:])),
        "spacing": lambda x: float(x[1]),
        "nelements": lambda x: np.array(x[1:4], dtype=np.int64),
        "center": lambda x: np.array(x[1:4], dtype=np.float64),
    }
    if isinstance(file, Path):
        content = file.read_text()
        if not file_label:
            file_label = str(file)
    elif isinstance(file, bytes):
        content = content.decode("utf-8")
    elif isinstance(file, str):
        content = file
    else:
        raise TypeError(
            f"Expected a string, bytes, or Path object, "
            f"but got {type(file).__name__}{f" for file {file_label}" if file_label else ""}."
        )
    lines = content.strip().splitlines()
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
                    content=file,
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
                    content=file,
                    line_idx=line_idx,
                    token=line,
                )
        else:
            # The first line with a number is the start of the grid point values
            grid_point_values = lines[line_idx:]
            break
    else:
        _raise_or_warn(
            "No grid point values found in the file.",
            critical=True,
            filepath=file_label,
            content=file,
        )
    return metadata, grid_point_values, content


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
