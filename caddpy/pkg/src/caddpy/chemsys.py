from __future__ import annotations

from typing import TYPE_CHECKING
import itertools

from collections.abc import Sequence
from typing import Literal
from pathlib import Path
import uuid

import jax
import numpy as np
import pandas as pd
import nglview as ngl
from openbabel import pybel
import scicoda
import scifile
import scids
from scids.typing import NonNegativeFloat

from caddpy import exception

if TYPE_CHECKING:
    from typing import Any
    from pathlib import Path
    from scids.pointcloud import PointCloud
    from scids.grid import Grid
    from scifile.pdb import PDBFileRecords, PDBFileSections
    from numpy.typing import ArrayLike


class ChemicalSystem:
    """A chemical system with defined composition and trajectory.

    Parameters
    ----------
    composition
        Atomic composition of the system and their properties.
        Depending on the context, this can represent a single molecule
        or an ensemble of molecules.
    trajectory
        A 3D point cloud representing
        the conformation of the system over time or in different states.
    """
    def __init__(self, composition: ChemicalComposition, trajectory: PointCloud):
        if composition.atom_count != trajectory.point_count:
            raise exception.InputError(
                name="trajectory",
                message="Composition and trajectory must have the same number of atoms, "
                        f"but composition has {composition.atom_count} atoms and trajectory has {trajectory.point_count}.",
            )
        if trajectory.point_dim != 3:
            raise exception.InputError(
                name="trajectory",
                message=f"Trajectory must be a 3D point cloud, "
                        f"but is {trajectory.point_dim}D with shape {trajectory.points.shape}.",
            )
        self._composition = composition
        self._trajectory = trajectory
        return

    @property
    def composition(self):
        """Atomic composition of the system."""
        return self._composition

    @property
    def trajectory(self):
        """Trajectory of the system."""
        return self._trajectory

    def toxelate(
        self,
        grid: float | Sequence[float] | Grid = 0.3,
        radii: ArrayLike | None = None,
        padding: float | None = None,
        instance_selection: Any = None,
    ):
        return self.trajectory.toxelate(
            grid=grid,
            point_radii=radii if radii is not None else self.composition.vdw_radius,
            padding=padding,
            instance_selection=instance_selection,
        )

    def minimize_aabb(
        self,
        instance_selection: Any = None,
        mode: Literal["per_instance", "one_for_all", "one_for_slice"] = "per_instance",
        algorithm: Literal["pca", "hull", "best"] = "best",
    ):
        return self.new(
            trajectory=self.trajectory.minimize_aabb(
                instance_selection=instance_selection,
                mode=mode,
                algorithm=algorithm,
            )
        )

    def select(self, selection: ArrayLike | pd.Series):
        selection = np.asarray(selection, dtype=bool)
        atoms = self.composition.atoms[selection].copy()
        positions = self.trajectory.points[..., selection, :]
        return self.new(composition=atoms, trajectory=positions)

    def to_pdb(
        self,
        frames: Any = (),
        multimodel: bool = False,
    ) -> scifile.pdb.PDBFile | np.ndarray:
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
        if self.trajectory.batch_size == 1:
            atoms = self.composition.atoms.assign(model_num=0)
            atoms[["x", "y", "z"]] = self.trajectory.points[(0, ) * self.trajectory.batch_ndim]
            return scifile.pdb.PDBFile(atom=atoms)
        coordinates = self.trajectory.points[frames]
        if coordinates.shape[-2:] != (self.trajectory.point_count, 3):
            raise exception.InputError(
                name="frames",
                message=f"Invalid frame selection: {frames}. "
                        f"Expected shape (..., {self.trajectory.point_count}, 3), "
                        f"but got {coordinates.shape}.",
            )
        if coordinates.ndim == 2:
            atoms = self.composition.atoms.assign(model_num=0)
            atoms[["x", "y", "z"]] = coordinates
            return scifile.pdb.PDBFile(atom=atoms)
        if multimodel:
            atoms_full = pd.concat(
                [
                    self.composition.atoms.assign(model_num=i+1)
                    for i in range(np.prod(coordinates.shape[:-2]))
                ],
                ignore_index=True
            )
            atoms_full[["x", "y", "z"]] = coordinates.reshape(-1, 3)
            return scids.file.pdb.PDBFile(atom=atoms_full)
        pdbs = np.empty(shape=coordinates.shape[:-2], dtype=object)
        for index in np.ndindex(coordinates.shape[:-2]):
            atoms = self.composition.atoms.assign(model_num=0)
            atoms[["x", "y", "z"]] = coordinates[index]
            pdbs[index] = scifile.pdb.PDBFile(atom=atoms)
        return pdbs

    def to_pdbqt(
        self,
        frames: Any = (),
        autobond: bool = False,
        rigid: bool = True,
        combine: bool = False,
        flexible: bool = False,
        preserve_serials: bool = True,
        preserve_hydrogens: bool = False,
        preserve_names: bool = True,
        charge_model: Literal[
            'eem',
            'eem2015ba',
            'eem2015bm',
            'eem2015bn',
            'eem2015ha',
            'eem2015hm',
            'eem2015hn',
            'eqeq',
            'fromfile',
            'gasteiger',
            'mmff94',
            'none',
            'qeq',
            'qtpie',
        ] = 'gasteiger',
        add_hydrogens: bool = False,
        protonation_ph: float | None = None,
    ) -> str | np.ndarray:
        """Write the system as PDBQT files.

        Parameters
        ----------
        frames
            Any array indexing object
            to select a subset of instances to convert to PDBQT.
            By default, all instances are converted.
        autobond
            Enable automatic bonding.
        rigid
            Output a rigid molecule, i.e., no branches or torsion trees.
        combine
            Combine separate molecular pieces of input
            into a single rigid molecule.
            This only has an effect when `rigid` is `True`.
        flexible
            Output as a flexible residue.
        preserve_serials
            Preserve atom serial numbers.
            If `False`, atoms are renumbered sequentially.
        preserve_hydrogens
            Preserve non-polar hydrogen atoms in the output.
            If `False`, only polar hydrogens are preserved.
        preserve_names
            Preserve atom names in the output.
        charge_model
            Charge model to use for calculating partial charges.
        add_hydrogens
            Add missing hydrogens to the structure before conversion.
        protonation_ph
            pH value to optimize protonation state of the structure.
            If `None`, no protonation correction is performed.

        Returns
        -------
        PDBQT file content as string.
        If multiple frames are available and `frames` selects more than one,
        a numpy array of strings is returned with the same shape as the selected frames,
        where each string represents a PDBQT file for a single frame.
        Otherwise, a single string is returned.

        References
        ----------
        - [Open Babel documentation: PDBQT format](https://open-babel.readthedocs.io/en/latest/FileFormats/AutoDock_PDBQT_format.html)
        """
        # Create PDB files for pybel input.
        pdbs = self.to_pdb(frames=frames, multimodel=False)
        if isinstance(pdbs, scifile.pdb.PDBFile):
            single_file = True
            pdbs = np.array([pdbs])
        else:
            single_file = False
        # Set pybel write options based on the provided arguments.
        # https://open-babel.readthedocs.io/en/latest/FileFormats/AutoDock_PDBQT_format.html
        pybel_write_options = {
            flag: None
            for arg, flag in (
                (autobond, "b"),
                (rigid, "r"),
                (combine, "c"),
                (flexible, "s"),
                (preserve_serials, "p"),
                (preserve_hydrogens, "h"),
                (preserve_names, "n"),
            ) if arg
        }
        pdbqts = np.empty(shape=pdbs.shape, dtype=object)
        for index, pdb in np.ndenumerate(pdbs):
            pdb_str = str(pdb)
            # https://open-babel.readthedocs.io/en/latest/UseTheLibrary/Python_PybelAPI.html#pybel.readstring
            pybel_molecule = pybel.readstring(format="pdb", string=pdb_str)
            if add_hydrogens:
                # https://open-babel.readthedocs.io/en/latest/UseTheLibrary/Python_PybelAPI.html#pybel.Molecule.addh
                pybel_molecule.addh()
            if protonation_ph is not None:
                pybel_molecule.OBMol.CorrectForPH(protonation_ph)
            pybel_molecule.calccharges(model=charge_model)
            # https://open-babel.readthedocs.io/en/latest/UseTheLibrary/Python_PybelAPI.html#pybel.Molecule.write
            pdbqt_str = pybel_molecule.write(format="pdbqt", opt=pybel_write_options)
            pdbqts[index] = pdbqt_str
        if single_file:
            return pdbqts[0]
        return pdbqts

    def new(
        self,
        composition: pd.DataFrame | ChemicalComposition | None = None,
        trajectory: ArrayLike | PointCloud | None = None
    ) -> ChemicalSystem:
        """Create a new ChemicalSystem from this one.

        Parameters
        ----------
        composition
            New composition for the system.
            If None, the current composition is used.
        trajectory
            New trajectory for the system.
            If None, the current trajectory is used.
        """
        if composition is None:
            composition = self._composition
        elif isinstance(composition, pd.DataFrame):
            composition = ChemicalComposition(composition)
        if trajectory is None:
            trajectory = self._trajectory
        elif isinstance(trajectory, np.ndarray | jax.Array):
            trajectory = scids.pointcloud.from_array(trajectory)
        return ChemicalSystem(composition=composition, trajectory=trajectory)


class ChemicalComposition:
    def __init__(self, atoms: pd.DataFrame):
        # Verify element symbols and assign element indices.
        ref_element_symbols = np.strings.lower(scicoda.atom.symbols())
        element_symbols = atoms["element"].str.lower()
        ref_map = pd.Series(data=np.arange(len(ref_element_symbols)), index=ref_element_symbols)
        element_indices = element_symbols.map(ref_map)
        if element_indices.isnull().any():
            missing = element_indices[element_indices.isnull()]
            bad_indices = missing.index.to_numpy()
            bad_values = atoms["element"].loc[bad_indices].to_numpy()
            details = ", ".join(f"{i}: {v}" for i, v in zip(bad_indices, bad_values))
            raise exception.InputError(
                name="atoms",
                message=f"Invalid element symbols (index: value): {details}."
            )
        atoms["element_index"] = element_indices

        self._atoms = atoms
        self._data_autodock_atom_types: pd.DataFrame = None
        self._autodock_atom_type_indices: np.ndarray = None
        return

    @property
    def atoms(self) -> pd.DataFrame:
        return self._atoms

    @property
    def element_index(self) -> np.ndarray:
        """Atomic index (i.e. atomic number minus 1) of the atoms."""
        return self._atoms["element_index"].values

    @property
    def vdw_radius(self) -> np.ndarray:
        """Van der Waals radii of the atoms."""
        if "r_vdw" in self._atoms:
            return self._atoms["r_vdw"].values
        ref_vdw_radii = scicoda.atom.van_der_waals_radii()
        self.atoms["r_vdw"] = ref_vdw_radii[self.element_index]
        return self._atoms["r_vdw"].values

    @property
    def atom_count(self) -> int:
        return len(self.atoms)

    def autodock_atom_type(self) -> pd.DataFrame:
        """Autodock types of the atoms."""
        if "autodock_atom_type" not in self._atoms:
            first_frame_index = 0 if self._chemsys.trajectory.batch_ndim == 0 else np.unravel_index(
                0, self._chemsys.trajectory.batch_shape
            )
            pdb_string = str(self._chemsys.to_pdb(frames=first_frame_index))
            pybel_molecule = pybel.readstring(format="pdb", string=pdb_string)
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
        self._data_autodock_atom_types = scicoda.atom.autodock_atom_types()
        self._autodock_atom_type_indices = np.where(
            self.autodock_atom_type[..., np.newaxis] == self._data_autodock_atom_types["type"].values
        )[1]
        return self._autodock_atom_type_indices

    def __getitem__(self, item):
        return self._atoms[item]


@ngl.register_backend("caddpy")
class _ChemicalSystemNGLViewAdaptor(ngl.Structure, ngl.Trajectory):
    """NGLView adaptor for ChemicalSystem.

    References
    ----------
    - [NGLView documentation:Extend NGLView classes](https://github.com/nglviewer/nglview/blob/master/docs/interface_classes.md)
    """
    def __init__(self, chemsys: ChemicalSystem):
        self._chemsys = chemsys
        self.ext = "pdb"
        self.params = {}
        self.id = str(uuid.uuid4())
        return

    def get_structure_string(self):
        first_frame_index = 0 if self._chemsys.trajectory.batch_ndim == 0 else np.unravel_index(
            0, self._chemsys.trajectory.batch_shape
        )
        return str(self._chemsys.to_pdb(frames=first_frame_index))

    def get_coordinates(self, index):
        index_unraveled = index if self._chemsys.trajectory.batch_ndim == 0 else np.unravel_index(
            index, self._chemsys.trajectory.batch_shape
        )
        return self._chemsys.trajectory.points[index_unraveled]

    @property
    def n_frames(self):
        return self._chemsys.trajectory.batch_size


def from_pdb(files: scifile.pdb.PDBFile | Path | bytes | str | ArrayLike):
    """Create a ChemicalSystem from PDB file(s)."""
    if isinstance(files, scifile.pdb.PDBFile | Path | bytes | str):
        atom, trajectory = _read_single_pdb(files)
        return ChemicalSystem(
            composition=ChemicalComposition(atoms=atom),
            trajectory=scids.pointcloud.from_array(trajectory)
        )
    files = np.asarray(files, dtype=object)
    # Parse the first file to get the composition first
    first_atom, first_trajectory = _read_single_pdb(file=files.flat[0])
    model_count = first_trajectory.shape[0] if first_trajectory.ndim == 3 else 1
    atom_count = first_trajectory.shape[-2]
    batch_shape = files.shape if model_count == 1 else (*files.shape, model_count)
    trajectory = np.zeros(shape=(*batch_shape, atom_count, 3), dtype=np.float32)
    trajectory[(0, ) * len(batch_shape)] = first_trajectory
    # Iterate over file indices, skipping the first file since we already parsed it
    for index in itertools.islice(np.ndindex(batch_shape), 1, None):
        atom, traj = _read_single_pdb(file=files[index])
        nmodel = traj.shape[0] if traj.ndim == 3 else 1
        if nmodel != model_count:
            raise exception.InputError(
                name="files",
                message=f"All files must have the same number of models, "
                        f"but file at index {index} has {nmodel} models while the first file has {model_count} models."
            )
        trajectory[index] = traj
    return ChemicalSystem(
        composition=ChemicalComposition(first_atom),
        trajectory=scids.pointcloud.from_array(trajectory)
    )


def _read_single_pdb(file: scifile.pdb.PDBFile | Path | bytes | str) -> tuple[pd.DataFrame, np.ndarray]:
    pdbfile = file if isinstance(file, scifile.pdb.PDBFile) else scifile.pdb.read(file=file, parse_only=["atom"])
    atom = pdbfile.atom
    # Create the trajectory array
    trajectory = atom[["x", "y", "z"]].to_numpy(dtype=np.float32)
    if pdbfile.nummdl:
        trajectory = trajectory.reshape(pdbfile.nummdl, -1, 3)
        atom = atom[atom["model_num"] == 1]
    atom = atom.drop(["model_num", "x", "y", "z"], axis=1)
    return atom, trajectory
