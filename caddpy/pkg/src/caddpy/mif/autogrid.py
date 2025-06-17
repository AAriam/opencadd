"""Calculate molecular interaction energy fields using [AutoDock](https://autodock.scripps.edu/)'s AutoGrid4.

References
----------
- [AutoDock User Guide](https://autodock.scripps.edu/wp-content/uploads/sites/56/2022/04/AutoDock4.2.6_UserGuide.pdf)
"""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
from typing import TYPE_CHECKING

import numpy as np
from loggerman import logger
import pyshellman
import scifile
import scids

from caddpy import exception

if TYPE_CHECKING:
    from collections.abc import Sequence, Any
    from typing import Literal
    from caddpy.typing import PathLike, ArrayLike
    from scids.grid import Grid
    from scids.field import Field
    import numpy.typing as npt
    from caddpy.chemsys import ChemicalSystem

    _BatchSection = Literal["receptor", "parameter", "ligand"]


def from_chemsys(
    system: ChemicalSystem,
    grid: Grid,
    frames: Any = (),
    ligand_types: Sequence[str] = ("A", "C", "HD", "N", "NA", "OA", "SA"),
    smooth: float = 0.5,
    dielectric: float = -0.1465,
    parameter_files: str | bytes | Path | ArrayLike | None = None,
    parameter_file_ids: str | ArrayLike | None = None,
    field_dtype: npt.DTypeLike = np.single,
    field_batch_order: tuple[_BatchSection, _BatchSection, _BatchSection] = ("receptor", "parameter", "ligand"),
    output_dir: PathLike = None,
    allow_copy: bool = True,
    include_elecmap: bool = True,
    include_dsolvmap: bool = True,
    pdbqt_autobond: bool = False,
    pdbqt_rigid: bool = True,
    pdbqt_combine: bool = False,
    pdbqt_flexible: bool = False,
    pdbqt_preserve_serials: bool = True,
    pdbqt_preserve_hydrogens: bool = False,
    pdbqt_preserve_names: bool = True,
    pdbqt_charge_model: Literal[
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
    pdbqt_add_hydrogens: bool = False,
    pdbqt_protonation_ph: float | None = None,
) -> Field:
    pdbqts = system.to_pdbqt(
        frames=frames,
        autobond=pdbqt_autobond,
        rigid=pdbqt_rigid,
        combine=pdbqt_combine,
        flexible=pdbqt_flexible,
        preserve_serials=pdbqt_preserve_serials,
        preserve_hydrogens=pdbqt_preserve_hydrogens,
        preserve_names=pdbqt_preserve_names,
        charge_model=pdbqt_charge_model,
        add_hydrogens=pdbqt_add_hydrogens,
        protonation_ph=pdbqt_protonation_ph,
    )
    return from_pdbqt(
        files=pdbqts,
        grid=grid,
        ligand_types=ligand_types,
        identical_receptor_types=True,
        smooth=smooth,
        dielectric=dielectric,
        parameter_files=parameter_files,
        parameter_file_ids=parameter_file_ids,
        field_dtype=field_dtype,
        field_batch_order=field_batch_order,
        output_dir=output_dir,
        allow_copy=allow_copy,
        include_elecmap=include_elecmap,
        include_dsolvmap=include_dsolvmap,
    )


def from_pdbqt(
    files: str | bytes | Path | ArrayLike,
    grid: Grid,
    ligand_types: str | Sequence[str] = ("A", "C", "HD", "N", "NA", "OA", "SA"),
    receptor_types: Sequence[str] | None = None,
    identical_receptor_types: bool = False,
    smooth: float = 0.5,
    dielectric: float = -0.1465,
    parameter_files: str | bytes | Path | ArrayLike | None = None,
    file_ids: str | ArrayLike | None = None,
    parameter_file_ids: str | ArrayLike | None = None,
    field_dtype: npt.DTypeLike = np.single,
    field_batch_order: tuple[_BatchSection, _BatchSection, _BatchSection] = ("receptor", "parameter", "ligand"),
    output_dir: PathLike = None,
    allow_copy: bool = True,
    include_elecmap: bool = True,
    include_dsolvmap: bool = True,
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

    def get_receptor_types(file: str | bytes | Path) -> Sequence[str]:
        nonlocal default_receptor_types
        if receptor_types:
            return receptor_types
        if default_receptor_types:
            return default_receptor_types
        pdbqt = scifile.autodock_pdbqt.read(file, parse_only=["ATOM"])
        extracted_receptor_types = pdbqt.atom["autodock_atom_type"].unique()
        if identical_receptor_types:
            default_receptor_types = extracted_receptor_types
        return extracted_receptor_types

    def process_file_inputs(
        input_files: str | bytes | Path | ArrayLike | None,
        input_file_ids: str | ArrayLike | None,
        receptor: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        arg_name = "files" if receptor else "parameter_files"
        if not input_files:
            if receptor:
                raise exception.InputError(
                    name=arg_name,
                    message="No receptor files provided."
                )
            if input_file_ids:
                raise exception.InputError(
                    name=f"{arg_name}_ids",
                    message="File IDs were provided, but no files were given."
                )
            return np.array([None]), np.array([None]), np.array([None]), True
        if isinstance(input_files, (str, bytes, Path)):
            input_files = np.array([input_files], dtype=object)
            single_file = True
        else:
            input_files = np.asarray(input_files, dtype=object)
            single_file = False
        if input_file_ids is None:
            file_id_prefix = "receptor" if receptor else "parameter_file"
            file_ids = np.array([f"{file_id_prefix}_{"_".join(map(str, i))}" for i in np.ndindex(input_files.shape)])
        elif isinstance(input_file_ids, str):
            file_ids = np.array([input_file_ids])
        if (shape_ids := file_ids.shape) != (shape_files := input_files.shape):
            raise exception.InputError(
                name=f"{arg_name}_ids",
                message="The shape of file IDs must match the shape of input files, "
                f"but got file IDs with shape {shape_ids} for files with shape {shape_files}."
            )
        if (count_labels := file_ids.size) != (count_unique_labels := np.unique(file_ids).size):
            raise exception.InputError(
                name=f"{arg_name}_ids",
                message="The file IDs must be unique, "
                f"but the provided IDs contain {count_labels - count_unique_labels} duplicates."
            )
        if any(" " in file_id for file_id in file_ids):
            raise exception.InputError(
                name=f"{arg_name}_ids",
                message="The file IDs must not contain spaces."
            )
        final_filepaths = np.empty(shape=input_files.shape, dtype=object)
        file_suffix = ".pdbqt" if receptor else ".dat"
        for file_idx in np.ndindex(input_files.shape):
            file_id = file_ids[file_idx]
            file = input_files[file_idx]
            if is_path(file):
                filepath = Path(file).resolve()
                if not filepath.is_file():
                    raise exception.InputError(
                        name=arg_name,
                        message=f"The file '{file}' for file ID '{file_id}' at index '{file_idx}' does not exist."
                    )
                if " " not in str(filepath):
                    final_filepaths[file_idx] = filepath
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
                    final_filepaths[file_idx] = final_filepath
            else:
                final_filepath = (output_dir / file_id).with_suffix(file_suffix)
                if final_filepath.exists():
                    raise exception.InputError(
                        name=arg_name,
                        message=f"The file '{file}' for file ID '{file_id}' at index '{file_idx}' already exists in the output directory."
                    )
                final_filepath.write_text(file) if isinstance(file, str) else final_filepath.write_bytes(file)
                final_filepaths[file_idx] = final_filepath
        return final_filepaths, file_ids, input_files, single_file

    def get_map_filepaths_shape(
        shape_receptor_files: tuple[int, ...] | None,
        shape_parameter_files: tuple[int, ...] | None,
    ):
        if any(batch_name not in ("receptor", "parameter", "ligand") for batch_name in field_batch_order):
            raise exception.InputError(
                name="field_batch_order",
                message=f"The field batch order must contain only 'receptor', 'parameter', and 'ligand', "
                        f"but got {field_batch_order}."
            )
        if len(field_batch_order) != 3:
            raise exception.InputError(
                name="field_batch_order",
                message=f"The field batch order must contain exactly three elements, "
                        f"but got {len(field_batch_order)} elements: {field_batch_order}."
            )
        if len(field_batch_order) != len(set(field_batch_order)):
            raise exception.InputError(
                name="field_batch_order",
                message=f"The field batch order must not contain duplicate elements, "
                        f"but got {field_batch_order}."
            )
        batch_order = list(field_batch_order)
        if shape_receptor_files is None:
            batch_order.remove("receptor")
        if shape_parameter_files is None:
            batch_order.remove("parameter")
        single_ligand_type = isinstance(ligand_types, str)
        count_ligand_types = 1 if single_ligand_type else len(ligand_types)
        count_fields = count_ligand_types + int(include_elecmap) + int(include_dsolvmap)
        if single_ligand_type and count_fields == 1:
            batch_order.remove("ligand")
        if not batch_order:
            return (1,), batch_order
        batch_shape = []
        for batch_name in batch_order:
            if batch_name == "receptor":
                batch_shape.extend(shape_receptor_files)
            elif batch_name == "parameter":
                batch_shape.extend(shape_parameter_files)
            elif batch_name == "ligand":
                batch_shape.append(count_fields)
        return tuple(batch_shape), batch_order

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

    receptor_filepaths, file_ids, input_receptor_files, single_file = process_file_inputs(files, file_ids, receptor=True)
    parameter_filepaths, parameter_file_ids, input_param_files, single_parameter_file = process_file_inputs(
        parameter_files, parameter_file_ids, receptor=False
    )
    batch_shape, batch_order = get_map_filepaths_shape(
        shape_receptor_files=None if single_file else receptor_filepaths.shape,
        shape_parameter_files=None if single_parameter_file else parameter_filepaths.shape,
    )
    all_map_filepaths = np.empty(shape=batch_shape, dtype=object)

    gridcenter, npts, slices = calculate_grid_parameters(grid)
    default_receptor_types = None

    for receptor_file_idx in np.ndindex(receptor_filepaths.shape):
        file_id = file_ids[receptor_file_idx]
        filepath = receptor_filepaths[receptor_file_idx]
        for parameter_file_idx in np.ndindex(parameter_filepaths.shape):
            parameter_file_id = parameter_file_ids[parameter_file_idx]
            parameter_filepath = parameter_filepaths[parameter_file_idx]
            if single_parameter_file:
                output_prefix = file_id
            else:
                output_prefix = f"{file_id}_{parameter_file_id}"
            gpf = scifile.autodock_gpf.from_spec(
                receptor=filepath,
                parameter_file=parameter_filepath,
                npts=npts,
                spacing=grid.spacings[0],
                receptor_types=get_receptor_types(input_receptor_files[receptor_file_idx]),
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
            if not batch_order:
                all_map_filepaths[0] = gpf.maps[0]
            else:
                map_filepaths = list(gpf.maps)
                if include_elecmap:
                    map_filepaths.append(gpf.elecmap)
                if include_dsolvmap:
                    map_filepaths.append(gpf.dsolvmap)
                for map_idx, map_filepath in enumerate(map_filepaths):
                    target_index = []
                    for axis_name in batch_order:
                        if axis_name == "receptor":
                            target_index.extend(receptor_file_idx)
                        elif axis_name == "parameter":
                            target_index.extend(parameter_file_idx)
                        elif axis_name == "ligand":
                            target_index.append(map_idx)
                    all_map_filepaths[tuple(target_index)] = map_filepath
    maps = scifile.autodock_map.read(
        files=all_map_filepaths if batch_order else all_map_filepaths[0],
        field_dtype=field_dtype,
        nelements=npts,
        spacing=grid.spacings[0],
        center=gridcenter,
    )
    # batch = []
    # if not single_file:
    #     batch.append(("receptor", file_ids))
    # if not single_parameter_file:
    #     batch.append(("parameter_file", parameter_file_ids))
    # batch.append(("ligand_type", (*ligand_types, "e", "d")))
    return scids.field.from_tensor(
        tensor=maps.field[..., *slices],
        grid=grid,
        dtype=field_dtype,
        batch=len(batch_shape),
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
