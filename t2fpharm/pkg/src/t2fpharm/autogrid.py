"""AutoGrid4 API.

This module contains functions that communicate
with the AutoGrid4 program via shell command executions,
and get the results as numpy arrays.

References
----------
- https://autodock.scripps.edu/wp-content/uploads/sites/56/2022/04/AutoDock4.2.6_UserGuide.pdf
- https://autodock.scripps.edu/wp-content/uploads/sites/56/2021/10/AutoDock4.2.6_UserGuide.pdf
- https://www.csb.yale.edu/userguides/datamanip/autodock/html/Using_AutoDock_305.21.html
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
from typing import TYPE_CHECKING

import numpy as np
import scids


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal
    import numpy.typing as npt
    from scids.data import Autodock4AtomType


def from_pdbqt_contents(
    contents: Sequence[str],
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
    field_datatype: npt.DTypeLike = np.single,
    output_path: PathLike | None = None,
    receptor_names: list[str] | None = None,
):
    """Run AutoGrid and get the results.

    Parameters
    ----------
    contents
        Paths to the PDBQT files of the macromolecule receptor.
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
        The grid-point spacing, i.e., distance between two grid points, in angstroms (Å).
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
    output_path
        Path to a directory to write the output files in.
        If a non-existing path is given,
        a new directory will be created with all necessary parent directories.
        If not provided, the output files will be stored in a temporary directory.

    Returns
    -------
    scids.file.autodock_map.AutodockMapFile

    A 5-dimensional array of shape (n_t, n_x, n_y, n_z, n_l + 2), with
        n_t: number of input protein structures.
        n_x, n_y, n_z: number of grid points along x, y, and z directions.
        n_l: number of input ligand types, plus two additional fields for
        electrostatic potential and desolvation energy.

    Calculated grid-point energies, as a tuple of 1-dimensional arrays containing the
    energy values for each grid point for a specific type of energy. The grid points are
    ordered according to the nested loops z(y(x)), so the x-coordinate is changing fastest.
    The tuple of energy arrays is ordered in the same way as the input `ligand_types`,
    with two additional grids, namely electrostatic potential, and desolvation energy,
    added to the end of the tuple, respectively. The second tuple contains the paths to each
    of the energy map files in the same order.

    The electrostatic potential field values are in kcal.mol^-1.e^-1.
    """
    return _from_pdbq_files_or_contents(
        mode="content",
        entries=contents,
        parameter_file=parameter_file,
        npts=npts,
        spacing=spacing,
        receptor_types=receptor_types,
        ligand_types=ligand_types,
        gridcenter=gridcenter,
        smooth=smooth,
        dielectric=dielectric,
        field_datatype=field_datatype,
        output_path=output_path,
        entry_names=receptor_names,
    )


def run(
    gpf_filepath: PathLike,
    glg_filepath: PathLike | None = None,
) -> None:
    """Run AutoGrid4.

    Parameters
    ----------
    gpf_filepath
        Path to the input Grid Parameter File (GPF).
        This is the input specification file used by AutoGrid.
    glg_filepath
        Path to a new file to store the AutoGrid output log.
        If `None`, the log is written to the standard output.

    Notes
    -----
    Calculated grid-point energies are written to respective MAP files,
    as specified in the input GPF file.

    Raises
    ------
    subprocess.CalledProcessError
        If the process exits with a non-zero exit code,
        an exception will be raised, with attributes
        `returncode`, `stdout` and `stderr`,
        which hold the exit code, console output
        and error message of the process.
    """
    # _PATH_EXECUTABLE = Path(oc.__file__).parent.resolve() / "_exec" / "autogrid4"
    cmd = ["autogrid4", "-p", str(Path(gpf_filepath).resolve())]
    if glg_filepath:
        cmd.extend(["-l", str(Path(glg_filepath).resolve())])
    process = subprocess.run(
        args=cmd,
        check=True,
    )
    return process


def _from_pdbq_files_or_contents(
    mode: Literal["filepath", "content"],
    entries: Sequence[PathLike],
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
    field_datatype: npt.DTypeLike = np.single,
    output_path: PathLike | None = None,
    entry_names: list[str] | None = None,
):
    if output_path:
        output_path = Path(output_path)
    else:
        tempdir = tempfile.TemporaryDirectory()
        output_path = Path(tempdir.name)

    map_filepaths = []
    for receptor_idx, receptor_entry in enumerate(entries):
        receptor_name = entry_names[receptor_idx] if entry_names else f"receptor_{receptor_idx + 1}"
        receptor_dir = output_path / receptor_name
        if mode == "filepath":
            receptor_filepath = Path(receptor_entry).resolve()
            if not receptor_filepath.exists():
                raise FileNotFoundError(f"File {receptor_filepath} does not exist.")
        elif mode == "content":
            receptor_filepath = receptor_dir / f"{receptor_name}.pdbqt"
            receptor_filepath.write_text(receptor_entry)
        else:
            raise ValueError("mode must be either 'filepath' or 'content'.")
        gpf = scids.file.autodock_gpf.from_spec(
            receptor=receptor_filepath,
            parameter_file=parameter_file,
            npts=npts,
            spacing=spacing,
            receptor_types=receptor_types,
            ligand_types=ligand_types,
            gridcenter=gridcenter,
            smooth=smooth,
            dielectric=dielectric,
            output_path=receptor_dir / receptor_name,
        )
        gpf_filepath = receptor_dir / f"{receptor_name}.gpf"
        gpf_filepath.write_text(str(gpf))
        run(gpf_filepath=gpf_filepath, glg_filepath=receptor_dir / f"{receptor_name}.glg")
        map_filepaths.append([*gpf.maps, gpf.elecmap, gpf.dsolvmap])
    return scids.files.autodock_map.from_filepath(
        filepaths=map_filepaths,
        field_dtype=field_datatype,
        field_names=(*(ligand_type.name for ligand_type in ligand_types), "e", "d"),
        strict=True,
    )






def _extract_grid_values(grid: spacetime.grid.Grid):
    if grid.dimension != 3:
        raise ValueError(
            f"AutoGrid only accepts 3D grids, but the input grid had {grid.dimension} dimensions."
        )
    if not np.allclose(grid.spacings, grid.spacings[0]):
        raise ValueError("AutoGrid only accepts grids with equal spacing in all dimensions.")
    is_odd = grid.shape % 2
    if np.all(is_odd):
        return grid.center, grid.shape - 1, grid.spacings[0], slice(None)
    npts = np.where(is_odd, grid.shape - 1, grid.shape)
    center = grid.coordinates[tuple(grid.shape // 2)]
    slices = tuple(np.where(is_odd, slice(None), slice(-1)))
    return center, npts, grid.spacings[0], slices


def calculate_npts(
    grid_size: tuple[float, float, float],
    grid_spacing: float,
) -> tuple[int, int, int]:
    """Calculate the AutoGrid input argument `npts`.

    This is the number of grid points in each Cartesian direction (x, y, z).

    Parameters
    ----------
    grid_size
        Length of the grid along x-, y-, and z-axis, respectively.
    grid_spacing
        The same parameter as in AutoGrid, i.e. the grid-point spacing.

    Returns
    -------
    npts
        Can be used directly as input `npts` for AutoGrid functions. The values are the smallest
        valid values (i.e. even integers) that are needed to cover the whole cuboid pocket.
        Therefore, in cases where a dimension is not divisible by the spacing value, or the
        resulting value is an odd number, the value will be rounded up to the next even integer.

    Notes
    -----
    The units of values in `grid_size` and `grid_spacing` don't matter in this function,
    as long as they are both in the same units. Notice that in AutoGrid functions, the `spacing`
    argument must be in Ångstrom.

    See Also
    --------
    For more information on AutoGrid parameters `spacing` and `npts`, see the function
    `routine_run` in this module.
    """
    npts_min = np.ceil(np.array(grid_size) / grid_spacing)
    return tuple(np.where(npts_min % 2 == 0, npts_min, npts_min + 1).astype(int))

