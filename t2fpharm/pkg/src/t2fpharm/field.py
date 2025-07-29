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
    "from_data",
    "from_autogrid",
    "from_npz",
]


def from_data(
    *,
    grid_shape: Sequence[int],
    grid_size: Sequence[float],
    grid_spacing: Sequence[float],
    grid_lower: Sequence[float],
    grid_upper: Sequence[float],
    dtype: DTypeLike,
    batch: Sequence[tuple[str, Sequence[str]]],
    tensor: ArrayLike,
) -> Field:
    """Create a field from pre-computed data.

    Parameters
    ----------
    grid
        Grid defining the spatial structure of the field.
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
    feature_ids
        Labels for the pharmacophore features
        the tensor values correspond to.
        This should be a sequence of strings,
        where the i-th string is a unique identifier
        for the i-th feature in the tensor
        (i.e., the i-th element along the first axis of `tensor`).
    dtype
        JAX-NumPy data type of the tensor (e.g., `numpy.float32`, `numpy.float64`).
        If `None`, the data type will be inferred
        from the tensor and your default JAX dtype (usually `float32`).
    """
    grid = scids.grid.from_data(
        shape=grid_shape,
        size=grid_size,
        spacing=grid_spacing,
        lower=grid_lower,
        upper=grid_upper,
    )
    batch = list(batch)
    batch[0][0] = "feature"  # Ensure the first batch element is labeled 'feature'
    feature_ids = batch[0][1]

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
    return scids.field.from_tensor(
        tensor=tensor,
        grid=grid,
        batch=batch,
        dtype=dtype,
    )


def from_autogrid(
    grid_shape: Sequence[int],
    grid_size: Sequence[float],
    grid_spacing: Sequence[float],
    grid_lower: Sequence[float],
    grid_upper: Sequence[float],
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
    grid = scids.grid.from_data(
        shape=grid_shape,
        size=grid_size,
        spacing=grid_spacing,
        lower=grid_lower,
        upper=grid_upper,
    )
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
