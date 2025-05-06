from __future__ import annotations

from typing import TYPE_CHECKING

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
import nglview as ngl
import scicoda

import scids

if TYPE_CHECKING:
    from pathlib import Path
    from scids.pointcloud import DynamicPointCloud
    from scids.file.pdb import PDBFileRecords, PDBFileSections


class ChemicalSystem:
    """A chemical system with defined composition and trajectory.

    Parameters
    ----------
    composition
        Atomic composition of the system, their connectivity, and other properties.
        Depending on the context, this can represent a single molecule
        or an ensemble of molecules.
    trajectory
        A collection of points in 3D space, which can be used to represent
        the conformation of the system over time or in different states.
    """
    def __init__(self, composition: ChemicalComposition, trajectory: DynamicPointCloud):
        self._composition = composition
        self._trajectory = trajectory

        self._ngl_widget: ngl.NGLWidget | None = None
        return

    @property
    def composition(self):
        return self._composition

    @property
    def trajectory(self):
        """Trajectory of the system."""
        return self._trajectory

    def remove(self, *args: Literal["nonpoly"]):
        composition = self._composition[self._composition.res_poly]
        conformation = self._trajectory.points[:, self._composition.res_poly.to_numpy()]
        return ChemicalSystem(composition=composition, trajectory=conformation)

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
        self,
        frames: int | Sequence[int] | None = None,
        multimodel: bool = False,
    ) -> str | tuple[str, ...]:
        """Write the system as PDB files.

        Parameters
        ----------
        frames
            Indices of trajectory frames to write.
            Can be a single index or a list of indices.
            If None, all frames are written.
        multimodel
            If True and multiple frames requested,
            write frames as separate models in a single PDB file.
            Otherwise, write each frame as a separate PDB file.

        Returns
        -------
        If multimodel is True, a single PDB file string is returned.
        Otherwise, a tuple of srings each representing a single-model PDB file.
        """
        if frames is None:
            frames = range(self.trajectory.count_instances)
        elif isinstance(frames, int):
            frames = [frames]
        if not isinstance(frames, Sequence):
            raise TypeError("frames must be an int or a sequence of ints")
        if multimodel:
            atoms_full = pd.concat([self.composition.atoms.assign(model_num=i+1) for i in range(len(frames))], ignore_index=True)
            atoms_full[["x", "y", "z"]] = self.trajectory.points[frames, :, :].reshape(-1, 3)
            pdbfile = scids.file.pdb.PDBFile(atom=atoms_full)
            return pdbfile.to_file(multimodel=True)
        pdbs = []
        for frame in frames:
            atoms = self.composition.atoms.assign(model_num=0)
            atoms[["x", "y", "z"]] = self.trajectory.points[frame, :, :]
            pdbfile = scids.file.pdb.PDBFile(atom=atoms)
            pdbs.append(pdbfile.to_file(multimodel=False))
        return tuple(pdbs)

    def to_pdbqt(
        self,
        frames: int | Sequence[int] | None = None,
    ):
        """Write the system as PDBQT files."""
        if frames is None:
            frames = range(self.trajectory.count_instances)
        elif isinstance(frames, int):
            frames = [frames]
        if not isinstance(frames, Sequence):
            raise TypeError("frames must be an int or a sequence of ints")
        pdbqts = []
        for frame in frames:
            atoms = self.composition.atoms.assign(model_num=0)
            atoms[["x", "y", "z"]] = self.trajectory.points[frame, :, :]
            if "autodock_atom_type" not in atoms:
                atoms["autodock_atom_type"] = self.composition.autodock_atom_type()
            if "partial_charge" not in atoms:
                atoms["partial_charge"] = self.composition.partial_charge()
            pdbfile = scids.file.pdb.PDBFile(atom=atoms)
            pdbqts.append(pdbfile.to_file(variant="pdbqt", multimodel=False))
        return tuple(pdbqts)


class ChemicalComposition:
    def __init__(self, atoms: pd.DataFrame):
        self._atoms = atoms

        self._data_autodock_atom_types: pd.DataFrame = None
        self._autodock_atom_type_indices: np.ndarray = None
        return

    @property
    def atoms(self) -> pd.DataFrame:
        return self._atoms

    def partial_charge(self) -> np.ndarray:
        return self._atoms["partial_charge"].values

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

    def __getitem__(self, item):
        return self._atoms[item]


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
        return self._chemsys.trajectory.points[index]

    @property
    def n_frames(self):
        return self._chemsys.trajectory.points.shape[0]


def from_pdb(
    files: list[str | bytes | Path],
    parse_only: Sequence[PDBFileSections | PDBFileRecords | str] | None = None,
    strictness: Literal[0, 1, 2, 3] = 0,
):
    """Create a ChemicalSystem from a PDB file."""
    # Parse the first file to get the composition first
    first_file = scids.file.pdb.parse(
        file=files[0],
        parse_only=parse_only,
        strictness=strictness,
    )
    num_models = first_file.atom["model_num"].nunique()
    if len(files) > 1 and num_models > 1:
        raise ValueError(
            "Either provide a single multimodel PDB file or a list of PDB files with a single model each."
        )

    # Create the conformation tensor
    num_instances = num_models if num_models > 1 else len(files)
    conformation = np.zeros(
        shape=(num_instances, len(first_file.atom), 3),
        dtype=np.float32,
    )
    if num_models > 1:
        for model_idx in range(num_models):
            conformation[model_idx] = first_file.atom[first_file.atom["model_num"] == model_idx][["x", "y", "z"]]
        composition = first_file.atom[first_file.atom["model_num"] == 1]
    else:
        conformation[0] = first_file.atom[["x", "y", "z"]]
        for idx_instance, file in enumerate(files[1:], start=1):
            pdbfile = scids.file.pdb.parse(
                file=file,
                parse_only=parse_only,
                strictness=strictness,
            )
            conformation[idx_instance] = pdbfile.atom[["x", "y", "z"]]
    composition = composition.drop(["model_num", "x", "y", "z"], axis=1)
    return ChemicalSystem(
        composition=ChemicalComposition(composition),
        trajectory=scids.pointcloud.from_array(conformation)
    )


def from_pdbqt():
    trajectory = records["ATOM"][["x", "y", "z"]].to_numpy()[np.newaxis]


from openbabel import pybel
def write_pdbqt_from_pdb_filepath(
    filepath: _typing.PathLike,
    output_filename: str | None = None,
    output_path: _typing.PathLike | None = None,
):
    """
    Convert a PDB file to a PDBQT file, and save it in the given filepath.

    Parameters
    ----------
    filepath: str or pathlib.Path
        Path to input PDB file.
    output_path: str or pathlib.Path
        Path to output PDBQT file.
    add_hydrogens : bool, Optional, default: True
        Whether to add hydrogen atoms to the structure.
    protonate_for_pH : float | None, Optional, default: 7.4
        pH value to optimize protonation state of the structure. Disabled if `None`.
    calculate_partial_charges : bool, Optional, default: True
        Whether to calculate partial charges for each atom.

    Returns
    -------
    openbabel.pybel.Molecule
        Molecule object of PDB file, modified according to the input.
        The PDBQT file will be stored in the provided path.

    References
    ----------
    https://open-babel.readthedocs.io/en/latest/FileFormats/AutoDock_PDBQT_format.html
    """
    # pybel.readfile() provides an iterator over the Molecules in a file.
    # To access the first (and possibly only) molecule in a file, we use next()
    input_path = Path(filepath)
    molecule = next(pybel.readfile("pdb", str(input_path)))
    # if protonate_for_pH:
    molecule.OBMol.CorrectForPH(7.4)
    molecule.addh()
    # if add_hydrogens:

    # if calculate_partial_charges:
    for atom in molecule.atoms:
        atom.OBAtom.GetPartialCharge()
    # TODO: expose write options to function sig (see ref.)
    if output_path is None:
        output_filepath = (input_path.parent / input_path.stem).with_suffix(".pdbqt")
    else:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        output_filepath = (output_path / output_filename).with_suffix(".pdbqt")
    molecule.write(
        format="pdbqt",
        filename=str(output_filepath),
        overwrite=True,
        opt={"r": None, "n": None, "p": None, "h": None},
    )
    if output_filename is not None:
        return output_filepath.resolve()
    with open(output_filepath) as f:
        pdbqt_str = f.read()
    output_filepath.unlink()
    return pdbqt_str
