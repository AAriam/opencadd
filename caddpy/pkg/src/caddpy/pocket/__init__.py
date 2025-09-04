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
from caddpy.pocket.ligsite import LigSite

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal
    from caddpy.chemsys import ChemicalSystem
    from scids.field import Field
    from scids.grid import Grid
    from caddpy.typing import ArrayLike, PathLike
    from jax.typing import DTypeLike


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


def from_tensor(
    *,
    tensor: ArrayLike,
    grid: Grid,
    batch: Sequence[str | tuple[str, Sequence[str]]] | None = None,
    receptor: ChemicalSystem | None = None,
    pocket_atom_serials: ArrayLike | None = None,
    trim: bool = True,
) -> Pocket:
    """Create a pocket from a Grid and voxel field tensor.

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
    batch
        Information about the batch dimensions of the tensor.
        This can be a sequence of strings or tuples,
        where each string represents a batch dimension name,
        and each tuple contains a batch dimension name and a sequence of batch element names.
    receptor
        A `ChemicalSystem` object representing the receptor structure.
        If provided, it will be used to associate the pocket with the receptor,
        e.g., for visualization or further analysis.
    pocket_atom_serials
        A 1D array-like object containing the serial numbers of the atoms
        that make up the pocket.
    trim
        Whether to trim the pocket tensor to remove all-zero borders.
    """
    return Pocket(
        tensor=tensor,
        grid=grid,
        batch=batch,
        receptor=receptor,
        pocket_atom_serials=pocket_atom_serials,
        trim=trim
    )


def from_data(
    *,
    grid_shape: Sequence[int],
    grid_size: Sequence[float],
    grid_spacing: Sequence[float],
    grid_lower: Sequence[float],
    grid_upper: Sequence[float],
    batch: int | Sequence[str | tuple[str, Sequence[str]]],
    tensor: ArrayLike,
    receptor: ChemicalSystem | None = None,
    pocket_atom_serials: ArrayLike | None = None,
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
    grid = scids.grid.from_data(
        shape=grid_shape,
        size=grid_size,
        spacing=grid_spacing,
        lower=grid_lower,
        upper=grid_upper,
    )
    return Pocket(
        tensor=tensor,
        grid=grid,
        batch=batch,
        receptor=receptor,
        pocket_atom_serials=pocket_atom_serials
    )


def from_ligand(
    system: ChemicalSystem,
    ligand_mask: ArrayLike,
    ligand_radii: ArrayLike | None = None,
    ligand_radii_offset: float | Sequence[float] = 2.5,
    psp_count_lower: int | None = None,
    psp_count_upper: int | None = None,
    psp_dist_lower: float | None = None,
    psp_dist_upper: float | None = None,
    psp_dist_lower_mode: Literal["any", "all", "max", "min", "mean"] = "all",
    psp_dist_upper_mode: Literal["any", "all", "max", "min", "mean"] = "any",
    erosion_radius: float = 0,
    opening_radius: float = 0,
    morphology_order: tuple[Literal["opening", "erosion"], Literal["opening", "erosion"]] = ("opening", "erosion"),
    grid: float | Sequence[float] | Grid = 0.3,
    trim: bool = True,
) -> Pocket:
    """Create a pocket from a ligand.

    This function works as follows:
    1. Ligand atoms are selected from the `system` using `ligand_mask`.
    2. The ligand volume is converted to a voxel grid using the provided `grid`,
       where each ligand atom is represented by a sphere of radius `ligand_radii + ligand_radii_offset`.
    3. The receptor volume is converted to a voxel grid using the same `grid`.
    4. The pocket voxels are determined as the voxels that are occupied by the ligand
       but not occupied by the receptor.
    5. Morphological operations (erosion and opening) are applied to the pocket voxels
       in the specified order (`morphology_order`) to remove small artifacts and refine the pocket shape.
    6. The pocket voxels are labeled based on their connectivity,
       and the largest connected component is selected as the pocket.

    Parameters
    ----------
    system
        A `ChemicalSystem` object containing the receptor and ligand structures.
    ligand_mask
        A 1D boolean array to select the ligand atoms from `system.composition.atoms`.
    ligand_radii
        A sequence of non-negative real numbers representing the radius of each ligand atom.
        If `None`, the default van der Waals radii of the ligand atoms will be used.
    ligand_radii_offset
        A non-negative real number or a sequence of non-negative real numbers
        to be added to the ligand radii.
        This is useful when `ligand_radii` is not provided.
    erosion_radius
        A non-negative real number representing the radius of the morphological erosion operation
        applied to the pocket voxels.
        If `0`, no erosion is applied.
    opening_radius
        A non-negative real number representing the radius of the morphological opening operation
        applied to the pocket voxels.
        If `0`, no opening is applied.
    morphology_order
        A tuple of two distinct strings, either `("opening", "erosion")` or `("erosion", "opening")`,
        specifying the order of morphological operations applied to the pocket voxels.
        The first operation is applied first, followed by the second operation.
    grid
        Grid specification for the pocket voxels.
        - If a single float is provided, it will be used as the grid spacing in all three dimensions.
        - If a sequence of three floats is provided, they will be used as the grid spacing in the x, y, and z dimensions, respectively.
        - If a `Grid` object is provided, it will be used directly.
    trim
        Whether to trim the pocket tensor to remove all-zero borders.
    """
    def get_pocket_atom_serials(
        ligand_voxels: jnp.ndarray,
        receptor_voxels: jnp.ndarray,
        receptor: ChemicalSystem,
    ):
        overlap = jnp.logical_and(ligand_voxels, receptor_voxels)
        receptor_atom_indices = jnp.unique(receptor_voxels[overlap]) - 1  # Convert to 0-based indices
        receptor_atoms = receptor.composition.atoms.iloc[receptor_atom_indices]
        return receptor_atoms["serial"].to_numpy(dtype=jnp.int32)

    if (
        len(morphology_order) != 2
        or not all(op in ("opening", "erosion") for op in morphology_order)
        or morphology_order[0] == morphology_order[1]
    ):
        raise ValueError(
            "morphology_order must be a tuple of two distinct operations, "
            "either ('opening', 'erosion') or ('erosion', 'opening'), "
            f"but got {morphology_order}."
        )

    ligand = system.select(selection=ligand_mask)
    ligand_radii = ligand.composition.vdw_radius if ligand_radii is None else jnp.asarray(ligand_radii)
    ligand_volume = ligand.toxelate(
        grid=grid,
        radii=ligand_radii + jnp.asarray(ligand_radii_offset),
    )
    receptor = system.select(selection=system.composition.atoms["res_poly"])

    if any(arg is not None for arg in (psp_count_lower, psp_count_upper, psp_dist_lower, psp_dist_upper)):
        receptor_lb = jnp.minimum(receptor.trajectory.points.min(axis=0), ligand_volume.grid.lower_bounds)
        receptor_ub = jnp.maximum(receptor.trajectory.points.max(axis=0), ligand_volume.grid.upper_bounds)
        receptor_grid = ligand_volume.grid.new_aligned_grid(lower=receptor_lb, upper=receptor_ub, rounding="expand")
        receptor_full_volume = receptor.toxelate(grid=receptor_grid)
        slice_receptor, _ = receptor_grid.overlap_slice(ligand_volume.grid)
        receptor_volume_tensor = receptor_full_volume.tensor[slice_receptor]
        ligsite = LigSite(field=receptor_full_volume)
        ligsite_mask = ligsite.psp_mask(
            count_lower=psp_count_lower,
            count_upper=psp_count_upper,
            dist_lower=psp_dist_lower,
            dist_upper=psp_dist_upper,
            dist_lower_mode=psp_dist_lower_mode,
            dist_upper_mode=psp_dist_upper_mode,
        )[slice_receptor]
        psp_mask = jnp.logical_and(jnp.logical_not(receptor_volume_tensor), ligsite_mask)
    else:
        receptor_volume_tensor = receptor.toxelate(grid=ligand_volume.grid).tensor
        psp_mask = None

    empty_voxels = jnp.logical_not(receptor_volume_tensor)
    ligand_voxels = ligand_volume.tensor
    pocket_voxels = jnp.logical_and(ligand_voxels, empty_voxels)
    if psp_mask is not None:
        pocket_voxels = jnp.logical_and(pocket_voxels, psp_mask)

    for morphology_operation in morphology_order:
        if morphology_operation == "opening":
            morphology_radius = opening_radius
            morphology_func = sp.ndimage.binary_opening
        else:
            morphology_radius = erosion_radius
            morphology_func = sp.ndimage.binary_erosion
        if morphology_radius > 0:
            pocket_voxels = morphology_func(
                pocket_voxels,
                structure=ligand_volume.grid.footprint_spherical(radius=morphology_radius),
                border_value=0,
            )
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
        pocket_atom_serials=get_pocket_atom_serials(
            ligand_voxels=ligand_voxels,
            receptor_voxels=receptor_volume_tensor,
            receptor=receptor,
        ),
        trim=trim
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
    """Detect pockets using the DoGSite algorithms using the ProteinsPlus web API.

    References
    ----------
    1.  Volkamer, A.; Griewel, A.; Grombacher, T.; Rarey, M.,
        Analyzing the topology of active sites: on the prediction of pockets and subpockets.
        J. Chem. Inf. Model. 2010, 50 (11), 2041-52. DOI: https://doi.org/10.1021/ci100241y.
    2.  Volkamer, A.; Kuhn, D.; Grombacher, T.; Rippmann, F.; Rarey, M.,
        Combining global and local measures for structure-based druggability predictions.
        J. Chem. Inf. Model. 2012, 52 (2), 360-72. DOI: https://doi.org/10.1021/ci200454v.
    """
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
        anchor=origin,
    )
    return Pockets(
        grid=grid,
        pocket_labels=labels_main,
        subpocket_labels=labels_sub if len(sub_pockets) > 0 else None,
        subpocket_parent_labels=subpocket_parent_labels,
        receptor=system,
        external_data=pd.DataFrame(main_pockets + sub_pockets).convert_dtypes().set_index("label", drop=False)
    )


def from_npz(
    filepath: str | PathLike,
    receptor: ChemicalSystem | None = None,
    trim: bool = True,
) -> Pocket:
    """Create a Pocket from a .npz file."""
    data = scids.dataset.from_npz(filepath=filepath, data_key="tensor", return_dict=True)
    grid = scids.grid.from_data(
        shape=data["grid_shape"],
        size=data["grid_size"],
        spacing=data["grid_spacing"],
        lower=data["grid_lower"],
        upper=data["grid_upper"],
    )
    return Pocket(
        tensor=data["tensor"],
        grid=grid,
        batch=data["batch"],
        receptor=receptor,
        pocket_atom_serials=data.get("pocket_atom_serials", None),
        trim=trim,
    )
