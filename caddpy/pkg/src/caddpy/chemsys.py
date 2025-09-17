from __future__ import annotations

from typing import TYPE_CHECKING
import itertools

from collections.abc import Sequence
import io
from typing import Literal
from pathlib import Path
import uuid
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    import nglview as ngl

import fileex
from pdbfixer import PDBFixer
from openmm.app import PDBFile
import jax
import numpy as np
import pandas as pd
from openbabel import pybel
import scicoda
import scids.dataset
import scifile
import scids
from scids.typing import NonNegativeFloat
import scishow

from caddpy import exception
from caddpy import pdb_check
from caddpy.pdb_atom_matcher import PDBAtomMatcher

if TYPE_CHECKING:
    from typing import Any, Self, Iterable, Literal
    from pathlib import Path
    from scids.pointcloud import PointCloud
    from scids.grid import Grid
    from scifile.pdb import PDBFileRecords, PDBFileSections
    from numpy.typing import ArrayLike
    from scids.typing import PathLike


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
    def __init__(
        self,
        composition: ChemicalComposition,
        trajectory: PointCloud,
        name: str = "System",
    ):
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
        self._name = name

        self._composition._chemsys = self  # back-reference
        return

    @property
    def composition(self):
        """Atomic composition of the system."""
        return self._composition

    @property
    def trajectory(self):
        """Trajectory of the system."""
        return self._trajectory

    @property
    def name(self) -> str:
        """Name of the system."""
        return self._name

    def toxelate(
        self,
        grid: float | Sequence[float] | Grid = 0.3,
        radii: ArrayLike | None = None,
        padding: float | None = None,
        instance_selection: Any = None,
    ) -> scids.field.Field:
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

    def select(self, selection: ArrayLike | pd.Series) -> Self:
        selection = np.asarray(selection, dtype=bool)
        atoms = self.composition.atoms[selection].copy()
        positions = self.trajectory.points[..., selection, :]
        return self.new(composition=atoms, trajectory=positions)

    def display(self, nglwidget: ngl.NGLWidget | None = None) -> ngl.NGLWidget:
        if nglwidget is None:
            nglwidget = scishow.nglview.NGLWidget()
        nglwidget.add_trajectory(self, name=self._name)
        nglwidget.display(gui=True)
        return nglwidget

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
            atoms[["x", "y", "z"]] = self.trajectory.points[self.trajectory.instance_index(0)]
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

    def to_npz(self, filepath: PathLike | None = None, compress: bool = False) -> dict[str, Any]:
        """Save the system to a .npz file.

        Parameters
        ----------
        filepath
            Path to the .npz file to save the system data.
        """
        kwds = self.trajectory.to_npz()
        kwds["atoms"] = self.composition.atoms.to_records(index=False)
        if filepath is not None:
            if compress:
                np.savez_compressed(filepath, **kwds, allow_pickle=True)
            else:
                np.savez(filepath, **kwds, allow_pickle=True)
        return kwds

    def new(
        self,
        composition: pd.DataFrame | ChemicalComposition | None = None,
        trajectory: ArrayLike | PointCloud | None = None,
        name: str | None = None
    ) -> Self:
        """Create a new ChemicalSystem from this one.

        Parameters
        ----------
        composition
            New composition for the system.
            If None, the current composition is used.
        trajectory
            New trajectory for the system.
            If None, the current trajectory is used.
        name
            New name for the system.
            If None, the current name is used.
        """
        if composition is None:
            composition = self._composition
        elif isinstance(composition, pd.DataFrame):
            composition = ChemicalComposition(composition)
        if trajectory is None:
            trajectory = self._trajectory
        elif isinstance(trajectory, np.ndarray | jax.Array):
            trajectory = scids.pointcloud.from_array(trajectory)
        if name is None:
            name = self._name
        return ChemicalSystem(composition=composition, trajectory=trajectory, name=name)


class ChemicalComposition:
    def __init__(self, atoms: pd.DataFrame):
        # Verify element symbols and assign element indices.
        atoms = atoms.convert_dtypes()
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
        self._atoms_poly: pd.DataFrame = None
        self._atoms_mono: pd.DataFrame = None
        self._residues: pd.DataFrame | None = None
        return

    @property
    def atoms(self) -> pd.DataFrame:
        """Atomic composition of the system."""
        return self._atoms

    @property
    def atoms_poly(self) -> pd.DataFrame:
        """Polymeric atoms, i.e. those that are part of a polymeric chain."""
        if self._atoms_poly is None:
            self._atoms_poly = self._atoms[self._atoms["res_poly"]]
        return self._atoms_poly

    @property
    def atoms_mono(self) -> pd.DataFrame:
        """Monomeric atoms, i.e. those that are not part of a polymeric chain."""
        if self._atoms_mono is None:
            self._atoms_mono = self._atoms[~self._atoms["res_poly"]]
        return self._atoms_mono

    @property
    def residues(self) -> pd.DataFrame:
        """Residues in the system."""
        if self._residues is None:
            atoms = self.atoms
            group_cols = ["chain_id", "res_name", "res_seq", "res_poly", "res_std"]
            # identify block boundaries: start a new block when any group_col differs from the previous row
            boundaries = (atoms[group_cols] != atoms[group_cols].shift()).any(axis=1)
            block_ids = boundaries.cumsum()
            # Determine which columns to aggregate into arrays
            payload_cols = [col for col in atoms.columns if col not in group_cols]
            # # aggregate payload columns into numpy arrays per block
            # collapsed = []
            # for _, group in atoms.groupby(block_ids, sort=False):
            #     row = {col: group.iloc[0][col] for col in group_cols}
            #     for col in payload_cols:
            #         row[col] = group[col].to_numpy()
            #     collapsed.append(row)
            # # assemble result
            # Identify run boundaries

            # Build aggregation mapping: first() for group_cols, list for payloads
            agg_map = {c: "first" for c in group_cols}
            agg_map.update({c: list for c in payload_cols})
            # Single groupby + aggregation
            grouped = atoms.groupby(block_ids, sort=False).agg(agg_map)
            # Convert payload lists to numpy arrays
            for c in payload_cols:
                grouped[c] = grouped[c].apply(np.array)
            self._residues = grouped.reset_index(drop=True)
            # self._residues = pd.DataFrame(collapsed)
        return self._residues

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

    def sequence(self, chain_id: str | None = None) -> np.ndarray:
        """Get the sequence of residue names in a polymeric chain.

        Parameters
        ----------
        chain_id
            Chain ID of the polymeric chain to get the sequence for.
            If `None`, the sequence of the first polymeric chain is returned.

        Returns
        -------
        1D array of residue names in the specified chain.
        """
        if chain_id is None:
            poly_chain_ids = self.chain_ids(poly=True)
            if poly_chain_ids.size == 0:
                raise exception.InputError(
                    name="chain_id",
                    message="No polymeric chains found in the system."
                )
            chain_id = poly_chain_ids[0]
        atoms = self.atoms_chain(chain_id=chain_id, poly=True)
        return atoms.drop_duplicates(subset=['res_seq', 'res_name'], keep='first')['res_name'].to_numpy(dtype=str)

    def atoms_chain(self, chain_id: str | None = None, poly: bool | None = None) -> pd.DataFrame:
        """Get atoms of a specific chain.

        Parameters
        ----------
        chain_id
            Chain ID of the chain to get atoms for.
            If `None`, the first chain is used.
        poly
            - `True`: return only atoms of polymeric chains.
            - `False`: return only atoms of non-polymeric chains.
            - `None`: return atoms from all chains.

        Returns
        -------
        DataFrame with atoms of the specified chain.
        """
        if poly is None:
            atoms = self.atoms
        elif poly:
            atoms = self.atoms_poly
        else:
            atoms = self.atoms_mono

        if chain_id is None:
            chain_ids = self.chain_ids(poly=poly)
            if not chain_ids.size:
                raise exception.InputError(
                    name="chain_id",
                    message=f"No {('polymeric' if poly else 'monomeric')} chains found in the system."
                )
            chain_id = chain_ids[0]
        return atoms[atoms["chain_id"] == chain_id]

    def chain_ids(self, poly: bool | None = None) -> np.ndarray:
        """Get the unique chain IDs in the system.

        Parameters
        ----------
        poly
            - `True`: return only chain IDs of polymeric chains.
            - `False`: return only chain IDs of non-polymeric chains.
            - `None`: return all chain IDs.

        Returns
        -------
        1D array of strings containing unique chain IDs in the system.
        """
        if poly is None:
            return self.atoms["chain_id"].unique().to_numpy(dtype=str)
        if poly:
            return self.atoms_poly["chain_id"].unique().to_numpy(dtype=str)
        return self.atoms_mono["chain_id"].unique().to_numpy(dtype=str)

    def autodock_atom_type(self) -> pd.DataFrame:
        """Autodock types of the atoms."""
        if "autodock_atom_type" not in self._atoms:
            first_frame_index = 0 if self._chemsys.trajectory.batch_ndim == 0 else np.unravel_index(
                0, self._chemsys.trajectory.batch_shape
            )
            pdbqt_string = self._chemsys.to_pdbqt(frames=first_frame_index)
            pdbqt = scifile.pdb.read(pdbqt_string, variant="pdbqt", parse_only=["ATOM"])
            atom = pdbqt.atom.reset_index(drop=True)  # drop the "serial" index to avoid merge conflicts
            # Can't merge on "serial" since it is not preserved during PDBQT conversion by pybel.
            merge_cols = ["chain_id", "res_name", "res_seq", "i_code", "name"]
            atom = pdbqt.atom[merge_cols + ["autodock_atom_type", "partial_charge"]]
            self._atoms = self._atoms.merge(atom, on=merge_cols, how="left", validate="1:1")
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
        return str(self._chemsys.to_pdb(frames=self._chemsys.trajectory.instance_index(0)))

    def get_coordinates(self, index):
        return self._chemsys.trajectory.points[self._chemsys.trajectory.instance_index(index)]

    @property
    def n_frames(self):
        return self._chemsys.trajectory.batch_size


def from_pdb(
    files: scifile.pdb.PDBFile | Path | bytes | str | ArrayLike,
    name: str | None = None
) -> ChemicalSystem:
    """Create a ChemicalSystem from PDB file(s)."""
    if isinstance(files, scifile.pdb.PDBFile | Path | bytes | str):
        atom, trajectory, pdb_id = _read_single_pdb(files)
        return ChemicalSystem(
            composition=ChemicalComposition(atoms=atom),
            trajectory=scids.pointcloud.from_array(trajectory),
            name=name or pdb_id or "System"
        )
    files = np.asarray(files, dtype=object)
    # Parse the first file to get the composition first
    first_atom, first_trajectory, first_pdb_id = _read_single_pdb(file=files.flat[0])
    model_count = first_trajectory.shape[0] if first_trajectory.ndim == 3 else 1
    atom_count = first_trajectory.shape[-2]
    batch_shape = files.shape if model_count == 1 else (*files.shape, model_count)
    trajectory = np.zeros(shape=(*batch_shape, atom_count, 3), dtype=np.float32)
    trajectory[(0, ) * len(batch_shape)] = first_trajectory
    # Iterate over file indices, skipping the first file since we already parsed it
    for index in itertools.islice(np.ndindex(batch_shape), 1, None):
        atom, traj, pdb_id = _read_single_pdb(file=files[index])
        first_pdb_id = first_pdb_id or pdb_id
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
        trajectory=scids.pointcloud.from_array(trajectory),
        name=name or first_pdb_id or "System"
    )


def from_npz(filepath: PathLike | str) -> ChemicalSystem:
    """Create a ChemicalSystem from a .npz file."""
    data = scids.dataset.from_npz(filepath=filepath, data_key="points", return_dict=True)
    return ChemicalSystem(
        composition=ChemicalComposition(pd.DataFrame.from_records(data["atoms"])),
        trajectory=scids.pointcloud.from_array(points=data["points"], batch=data["batch"])
    )


def fix_pdb(
    file: Path | bytes | str,
    remove_chain_ids: str | Sequence[str] | None = None,
    keep_chain_ids: str | Sequence[str] | None = None,
    add_missing_residues: bool = True,
    replace_nonstandard_residues: bool = False,
    add_missing_heavy_atoms: bool = True,
    add_missing_atoms_seed: int = 42,
    add_missing_hydrogens: NonNegativeFloat | None = 7.0,
    add_missing_hydrogens_forcefield: Any = None,
    keep_ids: bool = True,
) -> tuple[str, dict[tuple[int, int], list[str]] | None, list[tuple] | None, dict | None, dict | None]:
    """Fix a PDB file"""
    def remove_nonpolymeric_ter_records(pdb_str: str) -> str:
        """Temporary fix for https://github.com/openmm/pdbfixer/issues/336"""
        lines = pdb_str.splitlines(keepends=True)
        # Gather indices of all 'TER ' lines
        ter_indices = [i for i, line in enumerate(lines) if line.startswith("TER ")]
        if not ter_indices:
            return pdb_str
        # Determine which TER lines to remove
        remove_indices: list[int] = []
        for prev, curr in zip(ter_indices, ter_indices[1:]):
            segment = lines[prev + 1 : curr]
            # Only remove if all lines start with 'HETATM'
            if not segment or all(line.startswith("HETATM") for line in segment):
                remove_indices.append(curr)
        # Remove in reverse order to keep indices valid
        for idx in sorted(remove_indices, reverse=True):
            lines.pop(idx)
        return ''.join(lines)

    if remove_chain_ids is not None and keep_chain_ids is not None:
        raise exception.InputError(
            name="keep_chain_ids",
            message="Cannot specify both remove_chain_ids and keep_chain_ids. "
                    "Please choose one of them."
        )

    open_file = fileex.file.open_file(file)
    fixer = PDBFixer(pdbfile=open_file)
    if remove_chain_ids is not None:
        remove_chain_ids = [remove_chain_ids] if isinstance(remove_chain_ids, str) else list(remove_chain_ids)
        fixer.removeChains(chainIds=remove_chain_ids)
    if keep_chain_ids is not None:
        # Remove all chains not in keep_chain_ids
        all_chain_ids = set([chain.id for chain in fixer.topology.chains()])
        keep_chain_ids = [keep_chain_ids] if isinstance(keep_chain_ids, str) else list(keep_chain_ids)
        keep_chain_ids_set = set(keep_chain_ids)
        remove_chain_ids = all_chain_ids - keep_chain_ids_set
        fixer.removeChains(chainIds=list(remove_chain_ids))
    if add_missing_residues:
        fixer.findMissingResidues()
        missing_residues = fixer.missingResidues
    else:
        missing_residues = None
    if replace_nonstandard_residues:
        fixer.findNonstandardResidues()
        nonstandard_residues = fixer.nonstandardResidues
        fixer.replaceNonstandardResidues()
    else:
        nonstandard_residues = None
    if add_missing_heavy_atoms:
        fixer.findMissingAtoms()
        missing_atoms = fixer.missingAtoms
        missing_terminals = fixer.missingTerminals
    else:
        missing_atoms = None
        missing_terminals = None
    if add_missing_residues or replace_nonstandard_residues or add_missing_heavy_atoms:
        fixer.addMissingAtoms(seed=add_missing_atoms_seed)
    if add_missing_hydrogens is not None:
        fixer.addMissingHydrogens(pH=add_missing_hydrogens, forcefield=add_missing_hydrogens_forcefield)

    pdb_fixed_buffer = io.StringIO()
    PDBFile.writeFile(
        topology=fixer.topology,
        positions=fixer.positions,
        file=pdb_fixed_buffer,
        keepIds=keep_ids,
    )
    pdb_fixed_buffer.seek(0)
    pdb_fixed_str = pdb_fixed_buffer.getvalue()
    pdb_fixed_valid_str = remove_nonpolymeric_ter_records(pdb_fixed_str)
    pdb_fixed_final_str = "\n".join(
        line for line in pdb_fixed_valid_str.splitlines()
        if not line.startswith("REMARK")
    )
    return pdb_fixed_final_str, missing_residues, nonstandard_residues, missing_atoms, missing_terminals


def _read_single_pdb(
    file: scifile.pdb.PDBFile | Path | bytes | str
) -> tuple[pd.DataFrame, np.ndarray, str | None]:
    pdbfile = file if isinstance(file, scifile.pdb.PDBFile) else scifile.pdb.read(file=file, parse_only=["atom"])
    _pdb_pre_check(pdb=pdbfile)
    atom = pdbfile.atom
    # Create the trajectory array
    trajectory = atom[["x", "y", "z"]].to_numpy(dtype=np.float32)
    if pdbfile.nummdl:
        trajectory = trajectory.reshape(pdbfile.nummdl, -1, 3)
        atom = atom[atom["model_num"] == 1]
    atom_matcher = _augment_atom_df(atom)
    atom_merged = atom_matcher.atom_merged.drop(["model_num", "x", "y", "z"], axis=1)
    return atom_merged, trajectory, pdbfile.header.id_code if pdbfile.header else None


def _pdb_pre_check(pdb: scifile.pdb.PDBFile) -> None:
    """Perform preliminary checks on a PDB file."""

    # Within each residue, atom names must be unique.
    pdb_check.assert_group_uniques(
        df=pdb.atom,
        group_by=["chain_id", "res_seq", "i_code"],
        unique_cols=["name"],
    )
    return


def _augment_atom_df(df: pd.DataFrame) -> PDBAtomMatcher:
    """Augment an atom DataFrame with additional columns."""
    residues = df.groupby(["chain_id", "res_seq", "i_code"], sort=False)
    atom_res_key_col_name = "res_idx"
    df[atom_res_key_col_name] = residues.ngroup()
    df["atom_idx"] = np.arange(len(df), dtype=np.int32)

    # Assign unique molecule index to each molecule
    mol_keys = np.where(
        df["res_poly"].to_numpy(),
        pd.Series(
            list(zip(np.repeat("poly", len(df)), df["chain_id"])),
            index=df.index,
        ).to_numpy(),
        pd.Series(
            list(zip(np.repeat("nonpoly", len(df)), df["chain_id"], df["res_seq"], df["i_code"])),
            index=df.index,
        ).to_numpy()
    )
    mol_idx, _ = pd.Series(pd.factorize(mol_keys, sort=False), index=df.index)
    df["mol_idx"] = mol_idx

    matcher = PDBAtomMatcher(
        atom=df,
        ccd_atom=_ccd("chem_comp_atom"),
        ccd_bond=_ccd("chem_comp_bond"),
        atom_res_key_col=atom_res_key_col_name,
    )
    return matcher


def _ccd(category: str) -> pd.DataFrame:
    """Get a DataFrame from the Chemical Component Dictionary (CCD)."""
    def get_parent_id(comp_id):
        data = chem_comp[chem_comp["id"] == comp_id].iloc[0]
        parent_id = data["mon_nstd_parent_comp_id"]
        return parent_id if pd.notna(parent_id) else data["id"]

    def comp_id_suffix(comp_id):
        parts = comp_id.split("_")
        return "" if len(parts) == 1 else "_".join(parts[1:])

    global _ccd_df
    df = _ccd_df.get(category)
    if df is not None:
        return df

    chem_comp = _ccd_df.get("chem_comp")
    if chem_comp is None:
        chem_comp = _ccd_df["chem_comp"] = scicoda.pdb.ccd("chem_comp")

    df = scicoda.pdb.ccd(category) if category != "chem_comp" else chem_comp
    id_col = "id" if category == "chem_comp" else "comp_id"

    df["comp_id_suffix"] = df[id_col].apply(comp_id_suffix)

    # Create a lookup mapping comp_chem.id -> comp_chem.mon_nstd_parent_comp_id
    lookup = chem_comp.set_index("id")["mon_nstd_parent_comp_id"]
    # Assign main_comp_id depending on is_aa_variant
    df["main_comp_id"] = np.where(
        df["is_aa_variant"],
        df[id_col].map(lookup),
        df[id_col]
    )

    df = df.convert_dtypes()
    _ccd_df[category] = df
    return df


_ccd_df: dict[str, pd.DataFrame] = {}
