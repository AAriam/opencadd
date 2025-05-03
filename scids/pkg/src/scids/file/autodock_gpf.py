"""Read and write AutoDock Grid Parameter Files (GPF).

The grid parameter file specifies an AutoGrid calculation,
including the size and location of the grid,
the atom types that will be used,
the coordinate file for the rigid receptor,
and other parameters for calculation of the grids

Notes
-----
- All delimiters where needed are white spaces.
- A comment must be prefixed by the “#” symbol,
  and can be placed after a space at the end of a parameter line,
  or on a line of its own.
- Upper/lower case is ignored in keywords
  but is significant in atom names and file names.
- File names cannot contain whitespace or non-ASCII characters.
- Although ideally it should be possible to give AutoGrid keywords in any order,
  not every possible combination has been tested,
  so it would be wise to stick to the following order.

Example GPF file:

```
npts 60 60 60                   # num.grid points in xyz
gridfld 1hsg.maps.fld           # grid_data_file
spacing 0.375                   # spacing(A)
receptor_types A C HD N OA SA   # receptor atom types
ligand_types A C NA OA N HD     # ligand atom types
receptor 1hsg.pdbqt             # macromolecule
gridcenter 2.5 6.5 -7.5         # xyz-coordinates or auto
smooth 0.5                      # store minimum energy w/in rad(A)
map 1hsg.A.map                  # atom-specific affinity map
map 1hsg.C.map                  # atom-specific affinity map
map 1hsg.NA.map                 # atom-specific affinity map
map 1hsg.OA.map                 # atom-specific affinity map
map 1hsg.N.map                  # atom-specific affinity map
map 1hsg.HD.map                 # atom-specific affinity map
elecmap 1hsg.e.map              # electrostatic potential map
dsolvmap 1hsg.d.map             # desolvation potential map
dielectric -0.1465              # <0, AD4 distance-dep.diel;>0,constant
```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from pathlib import Path

import numpy as np

from scids import exception
from scids.typing import PathLike
from scids.data import Autodock4AtomType

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal


@dataclass
class AutodockGpfFile:
    """AutoDock Grid Parameter File (GPF).

    Parameters
    ----------
    gridfld
        Path to the grid field file (.fld) to be generated.
    receptor
        Path to the PDBQT structure file of the macromolecule.
    maps
        List of paths to the ligand-specific affinity map files (.map) to be generated.
        This list must have the same order and size as `ligand_types`.
    elecmap
        Path to the electrostatic potential map file (.e.map) to be generated.
    dsolvmap
        Path to the desolvation potential energy map file (.d.map) to be generated.
    parameter_file
        User-defined atomic parameter file.
        If not provided, AutoGrid uses internal parameters.
    npts
        Number of grid points to add to the central grid point,
        along x-, y- and z-axes, respectively.
        Each value must be an even integer;
        when added to the central grid point,
        there will be an odd number of points in each dimension.
        The number of x-, y and z-grid points need not be equal.
    spacing
        The grid-point spacing, i.e.,
        distance between two grid points, in angstroms (Å).
        Grid points are orthogonal and uniformly spaced in AutoDock,
        i.e. this value is used for all three dimensions.
    receptor_types
        AutoDock atom types present in the receptor.
    ligand_types
        Atom types present in the ligand, i.e.,
        types of atoms for which interaction energies must be calculated.
    gridcenter
        Coordinates (x, y, z) of the center of grid map
        in the reference frame of the target structure, in angstroms (Å).
        If set to "auto", AutoGrid automatically centers the grid
        on the center of macromolecule.
    smooth
        Smoothing parameter for the pairwise atomic affinity potentials
        (both van der Waals and hydrogen bonds), in angstroms (Å).
        For AutoDock4, the force field has been optimized for a value of 0.5 Å.
    dielectric
        Dielectric function flag.
        If negative, AutoGrid will use distance-dependent dielectric of Mehler and Solmajer;
        if positive, AutoGrid will use this value as the dielectric constant.
        AutoDock4 has been calibrated to use a value of -0.1465.
    """
    def __init__(
        self,
        receptor: PathLike,
        gridfld: PathLike,
        maps: Sequence[PathLike],
        elecmap: PathLike,
        dsolvmap: PathLike,
        parameter_file: PathLike | None = None,
        npts: tuple[int, int, int] = (40, 40, 40),
        spacing: float = 0.375,
        receptor_types: Sequence[Autodock4AtomType | str] = (
            Autodock4AtomType.A,
            Autodock4AtomType.C,
            Autodock4AtomType.HD,
            Autodock4AtomType.N,
            Autodock4AtomType.OA,
            Autodock4AtomType.SA,
        ),
        ligand_types: Sequence[Autodock4AtomType | str] = (
            Autodock4AtomType.A,
            Autodock4AtomType.C,
            Autodock4AtomType.HD,
            Autodock4AtomType.N,
            Autodock4AtomType.NA,
            Autodock4AtomType.OA,
            Autodock4AtomType.SA,
        ),
        gridcenter: tuple[float, float, float] | Literal["auto"] = "auto",
        smooth: float = 0.5,
        dielectric: float = -0.1465,
    ):
        self.receptor = receptor
        self.gridfld = gridfld
        self.maps = maps
        self.elecmap = elecmap
        self.dsolvmap = dsolvmap
        self.parameter_file = parameter_file
        self.npts = npts
        self.spacing = spacing
        self.receptor_types = receptor_types
        self.ligand_types = ligand_types
        self.gridcenter = gridcenter
        self.smooth = smooth
        self.dielectric = dielectric
        return

    @property
    def receptor(self) -> Path:
        return self._receptor

    @receptor.setter
    def receptor(self, filepath: PathLike):
        self._receptor = self._verify_filepath(filepath)
        return

    @property
    def gridfld(self):
        return self._gridfld

    @gridfld.setter
    def gridfld(self, filepath: PathLike):
        self._gridfld = self._verify_filepath(filepath)
        return

    @property
    def maps(self) -> tuple[Path, ...]:
        return self._maps

    @maps.setter
    def maps(self, filepaths: Sequence[PathLike]):
        if not filepaths:
            raise ValueError("At least one map file must be provided.")
        self._maps = tuple(self._verify_filepath(filepath) for filepath in filepaths)
        return

    @property
    def elecmap(self):
        return self._elecmap

    @elecmap.setter
    def elecmap(self, filepath: PathLike):
        self._elecmap = self._verify_filepath(filepath)
        return

    @property
    def dsolvmap(self):
        return self._dsolvmap

    @dsolvmap.setter
    def dsolvmap(self, filepath: PathLike):
        self._dsolvmap = self._verify_filepath(filepath)
        return

    @property
    def parameter_file(self):
        return self._parameter_file

    @parameter_file.setter
    def parameter_file(self, filepath: PathLike | None):
        self._parameter_file = self._verify_filepath(filepath) if filepath else None
        return

    @property
    def npts(self) -> np.ndarray:
        return self._npts

    @npts.setter
    def npts(self, value):
        npts = np.asarray(value)
        # _exceptions.raise_array(
        #     parent_name=self.__class__.__name__,
        #     param_name="npts",
        #     array=npts,
        #     ndim_eq=1,
        #     size_eq=3,
        #     dtype=np.integer,
        # )
        if np.any(npts % 2 != 0):
            raise ValueError("Number of grid points must be even.")
        self._npts = npts
        return

    @property
    def spacing(self):
        return self._spacing

    @spacing.setter
    def spacing(self, value):
        # _exceptions.check_number(value, dtypes="real", gt=0)
        self._spacing = value
        return

    @property
    def receptor_types(self) -> tuple[Autodock4AtomType, ...]:
        return self._receptor_types

    @receptor_types.setter
    def receptor_types(self, value: Sequence[Autodock4AtomType]):
        self._receptor_types = self._verify_atom_types(value)
        return

    @property
    def ligand_types(self):
        return self._ligand_types

    @ligand_types.setter
    def ligand_types(self, value: Sequence[Autodock4AtomType]):
        self._ligand_types = self._verify_atom_types(value)
        return

    @property
    def gridcenter(self):
        return self._gridcenter

    @gridcenter.setter
    def gridcenter(self, value):
        if isinstance(value, str):
            if value != "auto":
                raise ValueError
            self._gridcenter = value
            return
        center = np.asarray(value)
        # _exceptions.raise_array(
        #     parent_name=self.__class__.__name__,
        #     param_name="gridcenter",
        #     array=center,
        #     ndim_eq=1,
        #     size_eq=3,
        #     dtype=(np.integer, np.floating),
        # )
        self._gridcenter = center
        return

    @property
    def smooth(self):
        return self._smooth

    @smooth.setter
    def smooth(self, value):
        # _exceptions.check_number(value, dtypes="real", ge=0)
        self._smooth = value
        return

    @property
    def dielectric(self):
        return self._dielectric

    @dielectric.setter
    def dielectric(self, value):
        # _exceptions.check_number(value, dtypes="real")
        self._dielectric = value
        return

    def __str__(self):
        """Return a string representation of the GPF file."""
        # It is recommended by AutoDock to generate the gpf file in this exact order.
        lines = []
        if self.parameter_file:
            lines.append(f"parameter_file {self.parameter_file}")
        lines.extend(
            [
                f"npts {' '.join(self.npts.astype(str))}"
                f"gridfld {self.gridfld}"
                f"spacing {self.spacing}"
                f"receptor_types {' '.join(receptor_type.name for receptor_type in self.receptor_types)}"
                f"ligand_types {' '.join(ligand_type.name for ligand_type in self.ligand_types)}"
                f"receptor {self.receptor}"
                f"gridcenter {' '.join(self.gridcenter.astype(str))}"
                f"smooth {self.smooth}"
            ]
        )
        for ligand_map in self.maps:
            lines.append(f"map {ligand_map}")
        lines.extend(
            [
                f"elecmap {self.elecmap}"
                f"dsolvmap {self.dsolvmap}"
                f"dielectric {self.dielectric}"
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _verify_filepath(filepath: PathLike) -> Path:
        """Verify the filepath and return a Path object.

        AutoGrid does not accept filepaths with whitespace or non-ASCII characters.
        """
        if not filepath:
            raise ValueError("Filepath cannot be empty.")
        path_str = str(filepath)
        if " " in path_str:
            raise ValueError("Path cannot contain spaces.")
        if not path_str.isascii():
            raise ValueError("Path contains non-ASCII characters.")
        return Path(filepath)

    @staticmethod
    def _verify_atom_types(atom_types: Sequence[Autodock4AtomType | str]) -> tuple[Autodock4AtomType, ...]:
        types = []
        for atom_type in atom_types:
            if isinstance(atom_type, Autodock4AtomType):
                types.append(atom_type)
                continue
            if not isinstance(atom_type, str):
                raise exceptions.InvalidAtomType(f"Invalid atom type: {atom_type}")
            try:
                types.append(Autodock4AtomType[atom_type])
            except KeyError:
                raise exceptions.InvalidAtomType(f"Invalid atom type: {atom_type}")
        return tuple(types)


def from_spec(
    receptor: PathLike,
    gridfld: PathLike | None = None,
    maps: Sequence[PathLike] | None = None,
    parameter_file: PathLike | None = None,
    elecmap: PathLike | None = None,
    dsolvmap: PathLike | None = None,
    npts: tuple[int, int, int] = (40, 40, 40),
    spacing: float = 0.375,
    receptor_types: Sequence[Autodock4AtomType | str] = (
        Autodock4AtomType.A,
        Autodock4AtomType.C,
        Autodock4AtomType.HD,
        Autodock4AtomType.N,
        Autodock4AtomType.OA,
        Autodock4AtomType.SA,
    ),
    ligand_types: Sequence[Autodock4AtomType | str] = (
        Autodock4AtomType.A,
        Autodock4AtomType.C,
        Autodock4AtomType.HD,
        Autodock4AtomType.N,
        Autodock4AtomType.NA,
        Autodock4AtomType.OA,
        Autodock4AtomType.SA,
    ),
    gridcenter: tuple[float, float, float] | Literal["auto"] = "auto",
    smooth: float = 0.5,
    dielectric: float = -0.1465,
    output_path: PathLike = None,
) -> tuple[dict | None, str | None] | None:
    """Create a Grid Parameter File (GPF) from the provided parameters.

    All parameters are identical to those in the `AutodockGpfFile` class,
    with the addition of `output_path`, which must be a filepath without a suffix.
    It makes the arguments `gridfld`, `maps`, `elecmap`, and `dsolvmap` optional,
    by generating them if not provided. These are generated by adding the following suffixes
    to the `output_path`:
    - `.maps.fld` for `gridfld`
    - `.X.map` for each ligand-type in `ligand_types`, where `X` is the name of the ligand-type.
    - `.e.map` for `elecmap`
    - `.d.map` for `dsolvmap`
    If `output_path` itself is not provided, the suffixes are added to the `receptor` filepath.
    """
    path_common = Path(output_path) if output_path else Path(receptor)
    return AutodockGpfFile(
        receptor=receptor,
        gridfld=gridfld or path_common.with_suffix(".maps.fld"),
        maps=maps or tuple(path_common.with_suffix(f".{ligand_type.name}.map") for ligand_type in ligand_types),
        elecmap=elecmap or path_common.with_suffix(".e.map"),
        dsolvmap=dsolvmap or path_common.with_suffix(".d.map"),
        parameter_file=parameter_file,
        npts=npts,
        spacing=spacing,
        receptor_types=receptor_types,
        ligand_types=ligand_types,
        gridcenter=gridcenter,
        smooth=smooth,
        dielectric=dielectric,
    )


def get_npts_from_size(
    size: tuple[float, float, float],
    spacing: float,
) -> tuple[int, int, int]:
    """Calculate the AutoGrid input argument `npts` from grid size and spacing.

    The calculated values are the smallest valid values (i.e. even integers)
    that are needed to cover the whole cuboid pocket.
    Therefore, in cases where a dimension is not divisible by the spacing value,
    or the resulting value is an odd number,
    the value will be rounded up to the next even integer.

    Parameters
    ----------
    size
        Length of the grid along x-, y-, and z-axes, respectively.
    spacing
        The same parameter as in AutoGrid, i.e. the grid-point spacing.

    Notes
    -----
    The units of values in do not matter in this function,
    as long as they are both in the same units.
    However, notice that in AutoGrid functions,
    the `spacing` argument must be in Ångstrom.
    """
    npts_min = np.ceil(np.array(size) / spacing)
    return tuple(np.where(npts_min % 2 == 0, npts_min, npts_min + 1).astype(int))
