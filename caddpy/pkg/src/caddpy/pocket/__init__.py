from __future__ import annotations

from typing import TYPE_CHECKING

from scids.grid import Grid

from caddpy.pocket.detector import Detector
from caddpy.pocket.detector_gui import DetectorGUI
from caddpy.pocket.pocket import Pocket

if TYPE_CHECKING:
    from collections.abc import Sequence
    from caddpy.chemsys import ChemicalSystem
    from scids.field import Field
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
