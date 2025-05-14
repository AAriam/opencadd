"""Calculate molecular interaction energy fields using [AutoDock](https://autodock.scripps.edu/)'s AutoGrid4.

References
----------
- [AutoDock User Guide](https://autodock.scripps.edu/wp-content/uploads/sites/56/2022/04/AutoDock4.2.6_UserGuide.pdf)
"""

from __future__ import annotations

from pathlib import Path
import shutil
import os
import tempfile
from typing import TYPE_CHECKING

import numpy as np
from loggerman import logger
import pyshellman
import scifile
import scids

from caddpy import exception

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal
    from caddpy.typing import PathLike
    from scids.grid import Grid
    from scids.field import Field
    import numpy.typing as npt
    from caddpy.chemsys import ChemicalSystem


def from_chemsys(
    system: ChemicalSystem,
    grid: Grid,
    ligand_types: Sequence[str] = ("A", "C", "HD", "N", "NA", "OA", "SA"),
    smooth: float = 0.5,
    dielectric: float = -0.1465,
    parameter_files: str | bytes | Path | Sequence[str | bytes | Path] | None = None,
    field_dtype: npt.DTypeLike = np.single,
    output_dir: PathLike = None,
    allow_copy: bool = True,
):



    pdbqt_files = oc.io.autodock.pdbqt.write.from_ensemble(system)
    # Receptor types are stored in the last column of ATOM records
    atom_types = [
        line.split()[-1] for line in pdbqt_files[0].splitlines() if line[:6] in ("ATOM  ", "HETATM")
    ]
    unique_atom_types = set(atom_types)
    receptor_types = tuple(Autodock4AtomType[atom_type] for atom_type in unique_atom_types)

    return from_pdbqt(
        files=pdbqt_files,
        grid=grid,
        ligand_types=ligand_types,
        receptor_types=receptor_types,
        smooth=smooth,
        dielectric=dielectric,
        parameter_files=parameter_files,
        field_dtype=field_dtype,
        output_dir=output_dir,
        allow_copy=allow_copy,
    )


def from_pdbqt(
    files: str | bytes | Path | Sequence[str | bytes | Path],
    grid: Grid,
    ligand_types: Sequence[str] = ("A", "C", "HD", "N", "NA", "OA", "SA"),
    receptor_types: Sequence[str] | None = None,
    identical_receptor_types: bool = False,
    smooth: float = 0.5,
    dielectric: float = -0.1465,
    parameter_files: str | bytes | Path | Sequence[str | bytes | Path] | None = None,
    file_ids: str | Sequence[str] | None = None,
    parameter_file_ids: str | Sequence[str] | None = None,
    field_dtype: npt.DTypeLike = np.single,
    output_dir: PathLike = None,
    allow_copy: bool = True,
) -> Field:
    """Run AutoGrid4 on a set of PDBQT files.

    This function can run AutoGrid4 on one or multiple
    macromolecule receptors and parameter files.

    Parameters
    ----------
    files
        PDBQT file contents (as string or bytes)
        or paths (as Path). This can be a single
        file or a sequence of files.
    grid
        A `Grid` object containing the grid information.
        The grid must be a 3D orthogonal grid
        with equal spacing in all dimensions.
        However, in contrast to working directly with AutoGrid,
        here the grid does not need to have an odd number of grid points
        in each dimension.
    ligand_types
        AutoDock atom types for which interaction energies must be calculated.
    receptor_types
        AutoDock atom types present in the receptor.
        If provided, all input PDBQT files are assumed to have identical receptor types.
        If not provided, they will be extracted from the input PDBQT files.
    identical_receptor_types
        This only applies if `receptor_types` is not provided:
        If `True`, all input PDBQT files are assumed to have identical receptor types.
        This means that the receptor types will be extracted only once from the first file,
        and the same types will be used for all other files.
        If `False`, the receptor types will be extracted from each file separately.
    smooth
        Smoothing parameter for the pairwise atomic affinity potentials
        (both van der Waals and hydrogen bonds), in angstroms (Å).
        For AutoDock4, the force field has been optimized for a value of 0.5 Å.
    dielectric
        Dielectric function flag.
        If negative, AutoGrid will use distance-dependent dielectric of Mehler and Solmajer;
        if positive, AutoGrid will use this value as the dielectric constant.
        AutoDock4 has been calibrated to use a value of -0.1465.
    parameter_files
        User-defined atomic parameter file(s).
        If not provided, AutoGrid uses internal parameters.
        Similar to the `files` parameter,
        this can be a single file (string, bytes, or Path) or a sequence of files.
        If a sequence is provided, all parameter files will be used for each receptor file,
        i.e., generating a matrix of jobs.
    file_ids
        A list of file IDs for the input files.
        If not provided, the file IDs will be generated automatically.
        The file IDs must be unique and must not contain spaces.
    parameter_file_ids
        A list of file IDs for the parameter files.
        If not provided, the file IDs will be generated automatically.
        The file IDs must be unique and must not contain spaces.
    field_dtype
        Data type of the output field.
    output_dir
        Path to a directory to write the output files in.
        If not provided, a temporary directory will be used.
        If a non-existing path is given,
        a new directory will be created with all necessary parent directories.
    allow_copy
        Allow copying files with spaces in their names to the output directory.
        AutoGrid4 does not support spaces in file names,
        so if this is set to `False`, an error will be raised
        if any of the input files contain spaces in their names.

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

    def is_path(file: str | bytes | Path) -> bool:
        """Check if the input is a filepath."""
        return isinstance(file, Path) or (isinstance(file, str) and "\n" not in file and Path(file).exists())

    def get_receptor_types(file_idx: int) -> Sequence[str]:
        nonlocal default_receptor_types
        if receptor_types:
            return receptor_types
        if default_receptor_types:
            return default_receptor_types
        pdbqt = scifile.autodock_pdbqt.read(files[file_idx], parse_only=["ATOM"])
        extracted_receptor_types = pdbqt.atom["autodock_atom_type"].unique()
        if identical_receptor_types:
            default_receptor_types = extracted_receptor_types
        return extracted_receptor_types

    def process_file_inputs(
        input_files: str | bytes | Path | Sequence[str | bytes | Path] | None,
        input_file_ids: str | Sequence[str] | None,
        receptor: bool,
    ) -> tuple[Sequence[Path | None], Sequence[str], bool]:
        arg_name = "files" if receptor else "parameter_files"
        if not input_files:
            if receptor:
                raise exception.InputError(
                    name=arg_name,
                    message="No files provided."
                )
            if input_file_ids:
                raise exception.InputError(
                    name=f"{arg_name}_ids",
                    message="File IDs were provided, but no files were given."
                )
            return [None], [None], [None], True
        if isinstance(input_files, (str, bytes, Path)):
            input_files = [input_files]
            single_file = True
            count_files = 1
        else:
            single_file = False
            count_files = len(input_files)
        if not input_file_ids:
            file_id_prefix = "receptor" if receptor else "parameter_file"
            file_ids = [f"{file_id_prefix}_{i}" for i in range(count_files)]
        elif isinstance(input_file_ids, str):
            file_ids = [input_file_ids]
        if (count_labels := len(file_ids)) != count_files:
            raise exception.InputError(
                name=f"{arg_name}_ids",
                message="The number of file IDs must match the number of input files, "
                f"but {count_labels} labels were provided for {count_files} files."
            )
        if count_labels != (count_unique_labels := len(set(file_ids))):
            raise exception.InputError(
                name=f"{arg_name}_ids",
                message="The file labels must be unique, "
                f"but the provided labels contain {count_labels - count_unique_labels} duplicates."
            )
        if any(" " in file_id for file_id in file_ids):
            raise exception.InputError(
                name=f"{arg_name}_ids",
                message="The file IDs must not contain spaces."
            )
        final_filepaths = []
        file_suffix = ".pdbqt" if receptor else ".dat"
        for file_idx, (file_id, file) in enumerate(zip(file_ids, input_files)):
            if is_path(file):
                filepath = Path(file).resolve()
                if not filepath.is_file():
                    raise exception.InputError(
                        name=arg_name,
                        message=f"The file '{file}' for file ID '{file_id}' at index '{file_idx}' does not exist."
                    )
                if " " not in str(filepath):
                    final_filepaths.append(filepath)
                else:
                    if not allow_copy:
                        raise exception.InputError(
                            name=arg_name,
                            message=f"The file '{file}' for file ID '{file_id}' at index '{file_idx}' contains spaces. "
                            "Please provide a path without spaces."
                        )
                    final_filepath = (output_dir / file_id).with_suffix(file_suffix)
                    if final_filepath.exists():
                        raise exception.InputError(
                            name=arg_name,
                            message=f"The file '{file}' for file ID '{file_id}' at index '{file_idx}' already exists in the output directory."
                        )
                    shutil.copy2(filepath, final_filepath)
                    final_filepaths.append(final_filepath)
            else:
                final_filepath = (output_dir / file_id).with_suffix(file_suffix)
                if final_filepath.exists():
                    raise exception.InputError(
                        name=arg_name,
                        message=f"The file '{file}' for file ID '{file_id}' at index '{file_idx}' already exists in the output directory."
                    )
                final_filepath.write_text(file) if isinstance(file, str) else final_filepath.write_bytes(file)
                final_filepaths.append(final_filepath)
        return final_filepaths, input_files, file_ids, single_file

    if not output_dir:
        output_dir = tempfile.TemporaryDirectory().name
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        if not output_dir.is_dir():
            raise exception.InputError(
                name="output_dir",
                message=f"The specified output path '{output_dir}' is not a directory."
            )
        if any(output_dir.iterdir()):
            raise exception.InputError(
                name="output_dir",
                message=f"The specified output path '{output_dir}' is not empty."
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    filepaths, files, file_ids, single_file = process_file_inputs(files, file_ids, receptor=True)
    parameter_filepaths, parameter_files, parameter_file_ids, single_parameter_file = process_file_inputs(
        parameter_files, parameter_file_ids, receptor=False
    )

    gridcenter, npts, slices = calculate_grid_parameters(grid)
    default_receptor_types = None
    map_filepaths = []

    for file_idx, (file_id, filepath) in enumerate(zip(file_ids, filepaths)):
        if single_file:
            maps_for_receptor_file = map_filepaths
        else:
            maps_for_receptor_file = []
            map_filepaths.append(maps_for_receptor_file)
        for parameter_file_idx, (parameter_file_id, parameter_filepath) in enumerate(zip(parameter_file_ids, parameter_filepaths)):
            if single_parameter_file:
                output_prefix = file_id
                maps_for_parameter_file = maps_for_receptor_file
            else:
                output_prefix = f"{file_id}_{parameter_file_id}"
                maps_for_parameter_file = []
                map_filepaths.append(maps_for_parameter_file)
            gpf = scifile.autodock_gpf.from_spec(
                receptor=filepath,
                parameter_file=parameter_filepath,
                npts=npts,
                spacing=grid.spacings[0],
                receptor_types=get_receptor_types(file_idx),
                ligand_types=ligand_types,
                gridcenter=gridcenter,
                smooth=smooth,
                dielectric=dielectric,
                output_path=output_dir / output_prefix,
            )
            gpf_filepath = output_dir / f"{output_prefix}.gpf"
            gpf_filepath.write_text(str(gpf))
            run(
                gpf_filepath=gpf_filepath,
                glg_filepath=output_dir / f"{output_prefix}.glg",
                cwd=output_dir,
            )
            maps_for_parameter_file.extend([*gpf.maps, gpf.elecmap, gpf.dsolvmap])

    maps = scifile.autodock_map.read(
        files=map_filepaths,
        field_dtype=field_dtype,
        nelements=npts,
        spacing=grid.spacings[0],
        center=gridcenter,
    )
    prefix = []
    if not single_file:
        prefix.append(("receptor", file_ids))
    if not single_parameter_file:
        prefix.append(("parameter_file", parameter_file_ids))
    prefix.append(("ligand_type", (*ligand_types, "e", "d")))
    return scids.field.from_tensor(
        tensor=maps.field[..., *slices],
        grid=grid,
        dtype=field_dtype,
        prefix=prefix,
    )


def run(
    gpf_filepath: PathLike,
    glg_filepath: PathLike | None = None,
    cwd: PathLike | None = None,
) -> pyshellman.ShellOutput:
    """Run AutoGrid4.

    This is a wrapper around the AutoGrid4 command-line tool.
    It runs AutoGrid4 with the specified grid parameter file (GPF).

    Parameters
    ----------
    gpf_filepath
        Path to the input Grid Parameter File (GPF).
        This is the input specification file used by AutoGrid.
    glg_filepath
        Path to a new file to store the AutoGrid output log.
        If `None`, AutoGrid writes the log to the standard error output,
        which is then captured and returned by this function.

    Returns
    -------
    An object with the following attributes:
    - `err`: Standard error output from the AutoGrid process.
       If `glg_filepath` is `None`, this will contain the log output.
    - `out`: Standard output from the AutoGrid process.
       This is normally empty.
    - `command`: The command that was executed as a list of strings.

    Notes
    -----
    Calculated grid-point energies are written to respective MAP files,
    as specified in the input GPF file.

    Raises
    ------
    MissingDependencyError
        If the AutoGrid4 executable is not found in the system PATH.
    SubprocessError
        If the process exits with a non-zero exit code.
    """
    exec_name = "AutoGrid4"
    gpf_filepath = Path(gpf_filepath)
    cwd = Path(cwd).resolve() if cwd else Path.cwd()
    gpf_abspath = gpf_filepath if gpf_filepath.is_absolute() else cwd / gpf_filepath
    cmd = [exec_name.lower(), "-p", str(gpf_abspath)]
    if glg_filepath:
        glg_filepath = Path(glg_filepath)
        glg_abspath = glg_filepath if glg_filepath.is_absolute() else cwd / glg_filepath
        cmd.extend(["-l", str(glg_abspath)])
    try:
        process = pyshellman.run(
            command=cmd,
            cwd=cwd,
            logger=logger,
            log_title=exec_name,
        )
    except pyshellman.exception.PyShellManExecutionError as e:
        raise exception.MissingDependencyError(exec_name) from e
    except pyshellman.exception.PyShellManNonZeroExitCodeError as e:
        raise exception.SubprocessError(
            name=exec_name,
            command=cmd,
            cwd=cwd,
            code=e.output.code,
            stdout=e.output.out,
            stderr=e.output.err,
        ) from e
    return process


def calculate_grid_parameters(grid: Grid) -> tuple[np.ndarray, np.ndarray, tuple[slice, slice, slice]]:
    """Calculate input grid parameters for AutoGrid from a `Grid`.

    Parameters
    ----------
    grid
        A `Grid` object containing the grid information.
        The grid must be a 3D orthogonal grid with equal spacing in all dimensions.

    Returns
    -------
    A 3-tuple containing:

    1. The `gridcenter` value for the GPF file.
       This is the coordinates of the grid center as a 1D array of shape (3,).
    2. The `npts` value for the GPF file.
       This is the number of grid points (minus 1 for the center point) in each dimension,
       as a 1D array of shape (3,).
       Each value is an even integer;
       when added to the central grid point,
       there will be an odd number of points in each dimension.
    3. A 3-tuple of slices; one for each dimension.
       It extracts the input grid from the calculated AutoGrid grid.

    Notes
    -----
    AutoGrid's input Grid Parameter File (GPF) requires three grid-related parameters:
    - `gridcenter`: Coordinates of the center of the grid.
    - `npts`: Number of grid points (minus 1 for the center point) in each dimension.
    - `spacing`: Grid spacing in angstroms (Å).

    The grid must be a 3D orthogonal grid with equal spacing in all dimensions.
    Moreover, all three `npts` values (i.e. in x, y, and z dimensions) must be even integers,
    as they correspond to the number of grid points added to the center point.
    This means the final grid in AutoGrid will always have an odd number of grid points in each dimension.
    To make any 3D orthogonal grid compatible with AutoGrid,
    this function pads the grid in dimensions that have an even number of grid points with an additional grid point.
    It then calculates the `gridcenter` and `npts` values for this padded grid,
    along with the slices to extract the original grid from the padded grid.
    """
    if grid.dimension != 3:
        raise exception.InputError(
            name="grid",
            message=f"AutoGrid only accepts 3D grids, but the input grid has {grid.dimension} dimensions."
        )
    if not np.allclose(grid.spacings, grid.spacings[0]):
        raise exception.InputError(
            "grid",
            "AutoGrid only accepts grids with equal spacing in all dimensions, "
            f"but the input grid has spacings {grid.spacings}."
        )
    is_odd = grid.shape % 2
    if np.all(is_odd):
        return grid.center, grid.shape - 1, (slice(None),)
    npts = np.where(is_odd, grid.shape - 1, grid.shape)
    # In dimensions that are padded, the grid center is shifted by half a grid spacing.
    # We don't need to worry about basis vectors, since the grid is orthogonal
    # (i.e. the grid spacing matrix is diagonal).
    gridcenter = np.where(is_odd, grid.center, grid.center + grid.spacings / 2)
    slices = tuple(np.where(is_odd, slice(None), slice(-1)))
    return gridcenter, npts, slices
