"""Field generation for pharmacophore modeling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence
import scids
import caddpy
import jax.numpy as jnp
from scids.field import Field, from_npz

if TYPE_CHECKING:
    from collections.abc import Sequence
    from jax.typing import ArrayLike, DTypeLike
    from scids.grid import Grid
    from pathlib import Path


__all__ = [
    "Field",
    "from_tensor",
    "from_autogrid",
    "from_npz",
]


def from_tensor(
    *,
    tensor: ArrayLike,
    grid: Grid,
    feature_ids: Sequence[str],
    dtype: DTypeLike | None = None,
    batch: Sequence[tuple[str, Sequence[str]]] | None = None,
) -> Field:
    """Create a field from pre-computed data.

    Parameters
    ----------
    tensor
        N-dimensional `(N >= 4)` array-like object
        (e.g., list, NumPy array, JAX array)
        containing the field data.
        The first dimension of the array
        must correspond to different features
        whose IDs are provided in `feature_ids`.
        The last three dimensions of the array
        must match the shape of the grid.
        All other dimensions are considered batch dimensions
        corresponding to different instances of the field
        (e.g., for different receptor conformations).
    grid
        Grid defining the spatial structure of the field.
    feature_ids
        Labels for the pharmacophore features
        the tensor values correspond to.
        This should be a sequence of strings,
        where the i-th string is a unique identifier
        for the i-th feature in the tensor
        (i.e., the i-th element along the first axis of `tensor`).
    dtype
        JAX/NumPy data type of the tensor (e.g., `numpy.float32`, `numpy.float64`).
        If `None`, the data type will be inferred
        from the tensor and your default JAX dtype (usually `float32`).
    batch
        Information about the batch dimensions of the tensor.
        If provided, it should be a sequence of 2-tuples,
        where each tuple corresponds to a batch dimension in `tensor`.
        The first element of the tuple is a string label for the batch dimension,
        and the second element is a sequence of strings representing the element labels
        for that batch dimension.
    """
    # Check grid
    if not isinstance(grid, scids.grid.Grid):
        raise TypeError(
            f"Expected Grid object, got {type(grid).__name__}."
        )
    if grid.dimension != 3:
        raise ValueError(
            f"Expected 3D grid, got {grid.dimension}D."
        )
    # Check feature IDs
    if isinstance(feature_ids, str | bytes) or not isinstance(feature_ids, Sequence):
        raise TypeError(
            f"Expected sequence of feature ID strings, got {type(feature_ids).__name__}."
        )
    # Check tensor
    tensor = jnp.asarray(tensor, dtype=dtype)
    if not jnp.issubdtype(tensor.dtype, jnp.floating):
        raise TypeError(
            f"Expected floating-point tensor, got {tensor.dtype}."
        )
    if tensor.ndim < 4:
        raise ValueError(
            f"Expected at least 4D tensor, got {tensor.ndim}D."
        )
    if tensor.shape[0] != len(feature_ids):
        raise ValueError(
            f"Tensor first dimension ({tensor.shape[0]}) does not match "
            f"number of feature IDs ({len(feature_ids)})."
        )
    if jnp.any(grid.shape != tensor.shape[-3:]):
        raise ValueError(
            f"Grid shape {grid.shape} does not match tensor shape along last axes {tensor.shape[-3:]}"
        )
    batch_data = [("feature", feature_ids)]
    if batch is not None:
        if isinstance(batch, str | bytes) or not isinstance(batch, Sequence):
            raise TypeError(
                f"Expected sequence of batch information, got {type(batch).__name__}."
            )
        if len(batch) != tensor.ndim - 4:
            raise ValueError(
                f"Batch length ({len(batch)}) does not match tensor batch dimensions ({tensor.ndim - 4})."
            )
        batch_data.extend(batch)
    elif tensor.ndim > 4:
        # If no batch is provided, assume all dimensions after the first are batch dimensions
        batch_data.extend(
            [f"receptor_{i}" for i in range(1, tensor.ndim - 3)]
        )

    return scids.field.from_tensor(
        tensor=tensor,
        grid=grid,
        batch=batch_data,
        dtype=dtype,
    )


def from_autogrid(
    grid: Grid,
    # grid_shape: Sequence[int],
    # grid_size: Sequence[float],
    # grid_spacing: Sequence[float],
    # grid_lower: Sequence[float],
    # grid_upper: Sequence[float],
    receptor_files: str | bytes | Path | ArrayLike,
    ligand_types: Sequence[str] = ("HD", "C", "OA", "e-", "e+"),
    receptor_types: Sequence[str] | None = None,
    identical_receptor_types: bool = True,
    smooth: float = 0.5,
    dielectric: float = -0.1465,
    parameter_files: str | bytes | Path | ArrayLike | None = None,
    receptor_file_ids: str | Sequence[tuple[str, Sequence[str]]] | None = None,
    parameter_file_ids: str | Sequence[tuple[str, Sequence[str]]] | None = None,
    field_dtype: DTypeLike = jnp.single,
    output_dir: str | Path = None,
    allow_copy: bool = True,
) -> Field:
    """Create a field using AutoDock AutoGrid4.

    Parameters
    ----------
    grid
        A `Grid` object containing the grid information.
        The grid must be a 3D orthogonal grid
        with equal spacing in all dimensions.
        However, in contrast to working directly with AutoGrid,
        here the grid does not need to have an odd number of grid points
        in each dimension.
    receptor_files
        PDBQT file contents (as string or bytes)
        or paths (as string or pathlib.Path).
        This can be a single file or an array of files with any shape.
    ligand_types
        AutoDock atom types for which interaction energies must be calculated.
        In addition to the standard AutoDock atom types,
        the following types are also supported:
        - `e+`: Electrostatic potential (for a positive charge)
        - `e-`: Electrostatic potential (for a negative charge)
        - `dsolv`: Solvation energy
    receptor_types
        AutoDock atom types present in the receptor.
        If provided, all input PDBQT files are assumed to have identical receptor types.
        If not provided, they will be extracted from the input PDBQT files.
    identical_receptor_types
        This only applies if `receptor_types` is not provided.
        If `True`, all input PDBQT files are assumed to have identical receptor types.
        This means that the receptor types will be extracted only once from the first file,
        and the same types will be used for all other files.
        If `False`, the receptor types will be extracted from each file separately.
    smooth
        Smoothing parameter for the pairwise atomic affinity potentials
        (both van der Waals and hydrogen bonds), in angstroms (Å).
        For AutoDock4, the force field has been optimized for a value of 0.5 Å.
    dielectric
        Dielectric function flag.
        If negative, AutoGrid will use distance-dependent dielectric of Mehler and Solmajer;
        if positive, AutoGrid will use this value as the dielectric constant.
        AutoDock4 has been calibrated to use a value of -0.1465.
    parameter_files
        User-defined atomic parameter file(s).
        If not provided, AutoGrid uses internal parameters.
        Similar to the `files` parameter,
        this can be a single file or an array of files.
        If an array is provided, all parameter files will be used for each receptor file,
        i.e., generating a matrix of jobs.
    receptor_file_ids
        File ID(s) for the input receptor file(s).
        If `receptor_files` is a single file,
        this must be a single string.
        Otherwise, this must be a sequence of 2-tuples,
        one for each dimension of the `receptor_files` array.
        Each tuple must contain a string defining the axis name,
        followed by a sequence of file IDs for that axis.
        The file IDs must be unique and must not contain spaces.
        If not provided, the file IDs will be generated automatically.
    parameter_file_ids
        File ID(s) for the parameter file(s).
        If `parameter_files` is a single file,
        this must be a single string.
        Otherwise, this must be a sequence of 2-tuples,
        one for each dimension of the `parameter_files` array.
        Each tuple must contain a string defining the axis name,
        followed by a sequence of file IDs for that axis.
        The file IDs must be unique and must not contain spaces.
        If not provided, the file IDs will be generated automatically.
    field_dtype
        Numpy data type of the output field.
    field_batch_order
        The order of the batch dimensions in the output field.
        This must be a tuple of three unique strings,
        each being one of "receptor", "parameter", or "ligand".
    output_dir
        Path to a directory to write the output files in.
        If not provided, a temporary directory will be used.
        If a non-existing path is given,
        a new directory will be created with all necessary parent directories.
    allow_copy
        Allow copying files with spaces in their names to the output directory.
        AutoGrid4 does not support spaces in file names,
        so if this is set to `False`, an error will be raised
        if any of the input files contains spaces in its name.
    ligand_axis_id
        The batch axis ID for the ligand dimension in the output field.
    """
    # grid = scids.grid.from_data(
    #     shape=grid_shape,
    #     size=grid_size,
    #     spacing=grid_spacing,
    #     lower=grid_lower,
    #     upper=grid_upper,
    # )
    return caddpy.mif.autogrid.from_pdbqt(
        receptor_files=receptor_files,
        grid=grid,
        ligand_types=ligand_types,
        receptor_types=receptor_types,
        identical_receptor_types=identical_receptor_types,
        smooth=smooth,
        dielectric=dielectric,
        parameter_files=parameter_files,
        receptor_file_ids=receptor_file_ids,
        parameter_file_ids=parameter_file_ids,
        field_dtype=field_dtype,
        field_batch_order=("ligand", "receptor", "parameter"),
        output_dir=output_dir,
        allow_copy=allow_copy,
        ligand_axis_id="feature",
    )
