from __future__ import annotations

from typing import TYPE_CHECKING

from collections.abc import Sequence
from typing import Literal

import jax.numpy as jnp
import numpy as np
import pandas as pd
import nglview as ngl
import scicoda

import scids
import scids.file.pdb._writer

if TYPE_CHECKING:
    from pathlib import Path


class ChemicalSystem:
    def __init__(self, composition: pd.DataFrame, conformation: jnp.ndarray):
        self._composition = composition
        self._conformation = scids.pointcloud.from_array(conformation)
        self._pdb_writer = scids.file.pdb._writer.EnsemblePDBWriter(ensemble=self)

        self._ngl_widget: ngl.NGLWidget | None = None
        return

    @property
    def composition(self):
        return self._composition

    @property
    def conformation(self):
        return self._conformation

    def remove(self, *args: Literal["nonpoly"]):
        composition = self._composition[self._composition.res_poly]
        conformation = self._conformation.points[:, self._composition.res_poly.to_numpy()]
        return ChemicalSystem(composition=composition, conformation=conformation)

    def display_nglview(self, widget: ngl.NGLWidget | Literal["self"] | None = "self"):
        if not widget:
            output_widget = self._ngl_widget = ngl.NGLWidget()
        elif widget == "self":
            if self._ngl_widget:
                output_widget = self._ngl_widget
            else:
                output_widget = self._ngl_widget = ngl.NGLWidget()
        else:
            output_widget = widget
        output_widget.add_trajectory(_ChemicalSystemNGLViewAdaptor(self))
        return output_widget

    def to_pdb(
        self, model: int | Sequence[int],
        separate_models: bool = True,
    ) -> str | tuple[str, ...]:
        return self._pdb_writer.write(models=model, separate_models=separate_models)
class ChemicalComposition:
    def __init__(self, atoms: pd.DataFrame):
        self._atoms = atoms

        self._data_autodock_atom_types: pd.DataFrame = None
        self._autodock_atom_type_indices: np.ndarray = None
        return

    @property
    def atoms(self) -> pd.DataFrame:
        return self._atoms

    def autodock_atom_type(self) -> pd.DataFrame:
        return self._atoms["autodock_atom_type"].values

    def hbond_acceptor(self) -> np.ndarray:
        indices = self.autodock_atom_type_indices
        return self._data_autodock_atom_types["hbond_acceptor"][indices]

    def hbond_donor(self) -> np.ndarray:
        indices = self.autodock_atom_type_indices
        return self._data_autodock_atom_types["hbond_donor"][indices]

    def hbond_count(self) -> np.ndarray:
        indices = self.autodock_atom_type_indices
        return self._data_autodock_atom_types["hbond_count"][indices]

    @property
    def autodock_atom_type_indices(self) -> np.ndarray:
        if self._autodock_atom_type_indices:
            return self._autodock_atom_type_indices
        self._data_autodock_atom_types = scicoda.data.autodock_atom_types
        self._autodock_atom_type_indices = np.where(
            self.autodock_atom_type[..., np.newaxis] == self._data_autodock_atom_types["type"].values
        )[1]
        return self._autodock_atom_type_indices




class _ChemicalSystemNGLViewAdaptor(ngl.Structure, ngl.Trajectory):
    def __init__(self, chemsys: ChemicalSystem):
        self._chemsys = chemsys
        self.ext = "pdb"
        self.params = {}
        self.id = 0
        return

    def get_structure_string(self):
        return self._chemsys.to_pdb(0)

    def get_coordinates(self, index):
        return self._chemsys.conformation.points[index]

    @property
    def n_frames(self):
        return self._chemsys.conformation.points.shape[0]


def from_pdb(
    files: list[str | bytes | Path],
    parse_only: Sequence[parts.Records | parts.Sections | str] | None = None,
    strictness: Literal[0, 1, 2, 3] = 0,
):
    """Create a ChemicalSystem from a PDB file."""
    # Parse the first file to get the composition first
    first_file = scids.file.pdb.parse(
        file=files[0],
        parse_only=parse_only,
        strictness=strictness,
    )

    # Create the conformation tensor
    time_steps = len(files)
    conformation = np.zeros(
        shape=(time_steps, len(first_file.atom), 3),
        dtype=np.float32,
    )
    conformation[0] = first_file.atom[["x", "y", "z"]]

    for idx_instance, file in enumerate(files[1:], start=1):
        pdbfile = scids.file.pdb.parse(
            file=file,
            parse_only=parse_only,
            strictness=strictness,
        )
        conformation[idx_instance] = pdbfile.atom[["x", "y", "z"]]
    # atoms = structure.atom
    # composition = (
    #     atoms  # .drop(["model_num", "alt_loc", "occupancy", "x", "y", "z", "temp_factor"], axis=1)
    # )
    # conformation = jnp.expand_dims(jnp.array(atoms[["x", "y", "z"]]), axis=0)
    return ChemicalSystem(composition=first_file.atom, conformation=jnp.asarray(conformation))
