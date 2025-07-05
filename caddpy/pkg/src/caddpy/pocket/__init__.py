from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import jax.numpy as jnp
import pandas as pd
import scipy as sp

import sciapi
import scifile
import scids

from caddpy.pocket.detector import Detector
from caddpy.pocket.detector_gui import DetectorGUI
from caddpy.pocket.pocket import Pocket
from caddpy.pocket.pockets import Pockets

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal
    from caddpy.chemsys import ChemicalSystem
    from scids.field import Field
    from scids.grid import Grid
    from caddpy.typing import ArrayLike


__all__ = [
    "detector",
    "from_data",
]


def detector(
    system: ChemicalSystem,
    *,
    field: Field | None = None,
    grid: int | float | Sequence[int | float] | Grid = 0.3,
    minimize_aabb: bool = True,
    gui: bool = False,
    display: bool = True
) -> Detector | DetectorGUI:
    """Create a grid-based pocket detector from a chemical system.

    Parameters
    ----------
    system
        A `ChemicalSystem` object containing the receptor structure.
    field
        An optional `Field` representing the receptor's voxel grid.
        If provided, it will be used directly
        and all other parameters below will be ignored.
        If not provided, the field will be generated from the receptor.
    grid
        The grid spacing for the voxel grid.
        This can be a single value (e.g. `0.5` for 0.5 Ångstrom spacing),
        or a Grid object specifying the grid.
    minimize_aabb
        Whether to minimize the axis-aligned bounding box (AABB) of the receptor
        before creating the voxel grid, in order to reduce the size of the grid.
    gui
        Whether to create a GUI for the grid detector.
    """
    if not field:
        if minimize_aabb:
            system = system.new(trajectory=system.trajectory.minimize_aabb())
        field = system.toxelate(grid=grid)
    detector = Detector(receptor=system, field=field)
    if not gui:
        return detector
    detector_gui = DetectorGUI(detector)
    if display:
        detector_gui.display()
    return detector_gui


def from_data(
    voxels: ArrayLike,
    grid: Grid,
    batch: Sequence[str | tuple[str, Sequence[str]]] | None = None,
) -> Pocket:
    """Create a pocket from voxel field data.

    Parameters
    ----------
    voxels
        An `(n_batches + 3)`-dimensional array-like object
        containing the pocket voxels.
        The first `n_batches >= 0` dimensions represent batch dimensions,
        along which different instances of the pocket can be stored.
        The last three dimensions represent the spatial dimensions
        of the pocket along x, y, and z axes, respectively,
        and must match the shape of the provided grid.
        The values in the array are interpreted as boolean values
        each representing a voxel,
        where voxels that make up the pocket are `True`.
    grid
        The grid on which the pocket voxels are defined.
    """
    return Pocket(
        tensor=voxels,
        grid=grid,
        batch=batch,
    )


def from_ligand(
    system: ChemicalSystem,
    ligand_mask: ArrayLike,
    ligand_radii: ArrayLike | None = None,
    ligand_radii_offset: float | Sequence[float] = 2.5,
    grid: float | Sequence[float] | Grid = 0.3,
) -> Pocket:
    """Create a pocket from a ligand."""
    ligand = system.select(selection=ligand_mask)
    ligand_radii = ligand.composition.vdw_radius if ligand_radii is None else jnp.asarray(ligand_radii)
    ligand_volume = ligand.toxelate(
        grid=grid,
        radii=ligand_radii + jnp.asarray(ligand_radii_offset),
    )
    receptor = system.select(selection=system.composition.atoms["res_poly"])
    receptor_volume = receptor.toxelate(grid=ligand_volume.grid)
    empty_voxels = jnp.logical_not(receptor_volume.tensor)
    ligand_voxels = ligand_volume.tensor
    pocket_voxels = jnp.logical_and(ligand_voxels, empty_voxels)
    labels, _ = sp.ndimage.label(pocket_voxels)
    label_set = jnp.unique(labels)
    num_points_per_label = jnp.bincount(labels.ravel())
    if label_set[0] == 0:
        label_set = label_set[1:]
        num_points_per_label = num_points_per_label[1:]
    main_label = label_set[num_points_per_label.argmax()]
    return Pocket(
        tensor=labels == main_label,
        grid=ligand_volume.grid,
        receptor=system,
    )


def from_dogsite(
    system: ChemicalSystem,
    chain_id: str | None = None,
    ligand_id: str | tuple[str, str, int] | None = None,
    include_subpockets: bool = True,
    calculate_druggability: bool = True,
    algorithm: Literal["scorer", "3"] = "3",
    ligand_bias: bool = False,
) -> Pockets:
    api = sciapi.proteinsplus()
    pdb_content = str(system.to_pdb(frames=system.trajectory.instance_index(0)))
    dummy_pdb_id = api.upload_pdb(pdb_content).dummy_pdb_id
    pockets = api.dogsite(
        pdb_id=dummy_pdb_id,
        chain_id=chain_id,
        ligand_id=ligand_id,
        include_subpockets=include_subpockets,
        calculate_druggability=calculate_druggability,
        algorithm=algorithm,
        ligand_bias=ligand_bias,
    ).full_data

    main_pockets = []
    sub_pockets = []
    starts = []
    shapes = []
    grid_vectors = []
    grid_origins = []
    for pocket in pockets:
        pocket["mrc"] = scifile.mrc.read(pocket["mrc"])
        starts.append(pocket["mrc"].nstart_xyz)
        shapes.append(pocket["mrc"].n_xyz)
        grid_vectors.append(pocket["mrc"].grid_vectors)
        grid_origins.append(pocket["mrc"].grid_origin)
        name = pocket["name"].split("_")
        if len(name) == 2:
            main_pockets.append(pocket)
        elif len(name) == 3:
            sub_pockets.append(pocket)
        else:
            raise ValueError(f"Unexpected pocket name format: {pocket['name']}.")
    assert np.all(
        [
            np.allclose(grid_vectors[0], grid_vectors_n)
            for grid_vectors_n in grid_vectors[1:]
        ]
    ), "Pockets do not share the same grid vectors. Please open an issue ticket."
    starts = np.array(starts)
    min_start = starts.min(axis=0)
    starts = starts - min_start  # Normalize starts to (0, 0, 0)
    ends = starts + np.array(shapes)
    full_shape = np.max(ends, axis=0)
    labels_main = np.zeros(full_shape, dtype=np.uint8)
    labels_sub = np.zeros(full_shape, dtype=np.uint8)
    main_pockets = sorted(main_pockets, key=lambda x: x["volume"], reverse=True)
    sub_pockets = sorted(sub_pockets, key=lambda x: x["name"])
    name_to_index = {}
    for idx, pocket in enumerate(main_pockets, start=1):
        name_to_index[pocket["name"]] = pocket["label"] = idx
        pocket["parent_label"] = pocket["label"]
        start = pocket["mrc"].nstart_xyz - min_start
        end = start + pocket["mrc"].n_xyz
        slices = tuple(slice(start[i], end[i]) for i in range(3))
        pocket_mask = pocket["mrc"].data.astype(bool)
        labels_main[slices][pocket_mask] = idx
    subpocket_parent_labels = {}
    for idx, pocket in enumerate(sub_pockets, start=len(main_pockets) + 1):
        pocket["label"] = idx
        parent_name = "_".join(pocket["name"].split("_")[:2])
        pocket["parent_label"] = name_to_index[parent_name]
        subpocket_parent_labels[idx] = name_to_index[parent_name]
        start = pocket["mrc"].nstart_xyz - min_start
        end = start + pocket["mrc"].n_xyz
        slices = tuple(slice(start[i], end[i]) for i in range(3))
        pocket_mask = pocket["mrc"].data.astype(bool)
        labels_sub[slices][pocket_mask] = idx
    origin = np.array(grid_origins).min(axis=0)
    spacings = np.diag(grid_vectors[0])
    grid = scids.grid.from_anchor_shape_spacing(
        shape=full_shape,
        spacing=spacings,
        anchor_type="lower",
        anchor_coord=origin,
    )
    return Pockets(
        grid=grid,
        pocket_labels=labels_main,
        subpocket_labels=labels_sub if len(sub_pockets) > 0 else None,
        subpocket_parent_labels=subpocket_parent_labels,
        receptor=system,
        external_data=pd.DataFrame(main_pockets + sub_pockets).convert_dtypes().set_index("label", drop=False)
    )


