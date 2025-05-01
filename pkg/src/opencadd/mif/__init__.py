from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import opencadd as oc
from opencadd import pocket, spacetime
from opencadd.const.autodock import AtomType

from . import _autogrid
from .mif import MolecularInteractionField

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    import numpy.typing as npt


def autogrid(
    ensemble,
    ligand_types: Sequence[AtomType] = (AtomType.C, AtomType.A, AtomType.HD, AtomType.OA),
    smooth: float = 0.5,
    dielectric: float = -0.1465,
    confine: pocket.BindingPocket
    | spacetime.grid.Grid
    | spacetime.volume.ToxelVolume
    | None = None,
    param_filepath: Path | None = None,
    field_datatype: npt.DTypeLike = np.single,
):
    pdbqt_files = oc.io.autodock.pdbqt.write.from_ensemble(ensemble)
    # Receptor types are stored in the last column of ATOM records
    atom_types = [
        line.split()[-1] for line in pdbqt_files[0].splitlines() if line[:6] in ("ATOM  ", "HETATM")
    ]
    unique_atom_types = set(atom_types)
    receptor_types = tuple(AtomType[atom_type] for atom_type in unique_atom_types)

    if isinstance(confine, oc.spacetime.grid.Grid):
        grid = confine
    elif isinstance(confine, oc.spacetime.volume.ToxelVolume):
        grid = confine.grid
    elif isinstance(confine, oc.pocket.BindingPocket):
        grid = confine.volume.grid
    else:
        raise TypeError

    toxel_field = _autogrid._from_pdbqt_content(
        content=pdbqt_files,
        receptor_types=receptor_types,
        ligand_types=ligand_types,
        grid=grid,
        smooth=smooth,
        dielectric=dielectric,
        param_filepath=param_filepath,
        field_datatype=field_datatype,
    )

    return toxel_field
