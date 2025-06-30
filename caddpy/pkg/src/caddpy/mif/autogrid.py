"""Calculate molecular interaction energy fields using [AutoDock](https://autodock.scripps.edu/)'s AutoGrid4.

References
----------
- [AutoDock User Guide](https://autodock.scripps.edu/wp-content/uploads/sites/56/2022/04/AutoDock4.2.6_UserGuide.pdf)
"""

from __future__ import annotations

from pathlib import Path
import shutil
import fileex
from typing import TYPE_CHECKING, Sequence

import numpy as np
from loggerman import logger
import pyshellman
import scifile
import scids
import scicoda

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

_AUTODOCK_LIGAND_TYPES = tuple(scicoda.atom.autodock_atom_types()["type"].tolist())


def from_chemsys(
    system: ChemicalSystem,
    grid: Grid,
    frames: Any = (),
    ligand_types: str | Sequence[str] = _AUTODOCK_LIGAND_TYPES + ("e+", "e-", "dsolv"),
    smooth: float = 0.5,
    dielectric: float = -0.1465,
    parameter_files: str | bytes | Path | ArrayLike | None = None,
    parameter_file_ids: str | ArrayLike | None = None,
    field_dtype: npt.DTypeLike = np.single,
    field_batch_order: tuple[_BatchSection, _BatchSection, _BatchSection] = ("receptor", "parameter", "ligand"),
    output_dir: PathLike = None,
    allow_copy: bool = True,
    ligand_axis_id: str = "ligand",
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
        receptor_files=pdbqts,
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
        ligand_axis_id=ligand_axis_id,
    )


def from_pdbqt(
    receptor_files: str | bytes | Path | ArrayLike,
    grid: Grid,
    ligand_types: str | Sequence[str] = _AUTODOCK_LIGAND_TYPES + ("e+", "e-", "dsolv"),
    receptor_types: Sequence[str] | None = None,
    identical_receptor_types: bool = False,
    smooth: float = 0.5,
    dielectric: float = -0.1465,
    parameter_files: str | bytes | Path | ArrayLike | None = None,
    receptor_file_ids: str | Sequence[tuple[str, Sequence[str]]] | None = None,
    parameter_file_ids: str | Sequence[tuple[str, Sequence[str]]] | None = None,
    field_dtype: npt.DTypeLike = np.single,
    field_batch_order: tuple[_BatchSection, _BatchSection, _BatchSection] = ("parameter", "ligand", "receptor"),
    output_dir: PathLike = None,
    allow_copy: bool = True,
    ligand_axis_id: str = "ligand",
) -> Field:
    """Run AutoGrid4 on a set of PDBQT files.

    This function can run AutoGrid4 on one or multiple
    macromolecule receptors and parameter files.

    Parameters
    ----------
    files
        PDBQT file contents (as string or bytes)
        or paths (as string or pathlib.Path).
        This can be a single file or an array of files with any shape.
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
        This only applies if `receptor_types` is not provided.
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
        this can be a single file or an array of files.
        If an array is provided, all parameter files will be used for each receptor file,
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

    with fileex.directory.get_or_make(path=output_dir, ensure_empty=True) as output_dir:
        receptor_files, receptor_file_ids, receptor_filepaths, receptor_file_ids_array, single_receptor_file = _process_file_inputs(
            files=receptor_files,
            ids=receptor_file_ids,
            is_receptor=True,
            allow_copy=allow_copy,
            output_dir=output_dir,
        )
        parameter_files, parameter_file_ids, parameter_filepaths, parameter_file_ids_array, single_parameter_file = _process_file_inputs(
            files=parameter_files,
            ids=parameter_file_ids,
            is_receptor=False,
            allow_copy=allow_copy,
            output_dir=output_dir,
        )
        all_ligand_types, autodock_ligand_types, extra_ligand_types, anion_type_idx = _process_ligand_types(ligand_types)
        batch_shape, batch_order, batch_labels, ligand_axis_idx = _get_map_filepaths_shape(
            receptor_file_ids=None if single_receptor_file else receptor_file_ids,
            parameter_file_ids=None if single_parameter_file else parameter_file_ids,
            receptor_files_shape=receptor_files.shape,
            parameter_files_shape=parameter_files.shape,
            ligand_types=ligand_types,
            ligand_batch_label=ligand_axis_id,
            field_batch_order=field_batch_order,
        )
        all_map_filepaths = np.empty(shape=batch_shape, dtype=object)

        gridcenter, npts, slices = calculate_grid_parameters(grid)
        default_receptor_types = None

        for receptor_file_idx in np.ndindex(receptor_filepaths.shape):
            file_id = receptor_file_ids_array[receptor_file_idx]
            filepath = receptor_filepaths[receptor_file_idx]
            for parameter_file_idx in np.ndindex(parameter_filepaths.shape):
                parameter_file_id = parameter_file_ids_array[parameter_file_idx]
                parameter_filepath = parameter_filepaths[parameter_file_idx]
                if single_parameter_file:
                    output_prefix = file_id
                else:
                    output_prefix = f"{file_id}--{parameter_file_id}"
                gpf = scifile.autodock_gpf.from_spec(
                    receptor=filepath,
                    parameter_file=parameter_filepath,
                    npts=npts,
                    spacing=grid.spacings[0],
                    receptor_types=get_receptor_types(receptor_files[receptor_file_idx]),
                    ligand_types=autodock_ligand_types,
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
                    extra_filepaths = []
                    for extra_type in extra_ligand_types:
                        if extra_type.startswith("e"):
                            extra_filepaths.append(gpf.elecmap)
                        else:
                            extra_filepaths.append(gpf.dsolvmap)
                    map_filepaths = _process_map_filepaths(
                        all_types=all_ligand_types,
                        extra_types=extra_ligand_types,
                        autodock_types_filepaths=list(gpf.maps),
                        extra_types_filepaths=extra_filepaths,
                    )
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
    maps = maps.field[..., *slices]
    if anion_type_idx is not None:
        slicer = [slice(None)] * maps.ndim
        slicer[ligand_axis_idx] = anion_type_idx
        maps[tuple(slicer)] *= -1
    return scids.field.from_tensor(
        tensor=maps,
        grid=grid,
        dtype=field_dtype,
        batch=batch_labels,
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


def _process_file_inputs(
    files: str | bytes | Path | ArrayLike | None,
    ids: str | Sequence[tuple[str, Sequence[str]]] | None,
    is_receptor: bool,
    allow_copy: bool,
    output_dir: Path,
) -> tuple[np.ndarray, list[tuple[str, list[str]]], np.ndarray, np.ndarray, bool]:
    argname_files = "files" if is_receptor else "parameter_files"
    argname_ids = "file_ids" if is_receptor else "parameter_file_ids"
    if not files:
        if is_receptor:
            raise exception.InputError(
                name=argname_files,
                message="No receptor files provided."
            )
        if ids:
            raise exception.InputError(
                name=argname_ids,
                message="File IDs were provided, but no files were given."
            )
        return np.array([None]), [], np.array([None]), np.array([None]), True
    if isinstance(files, (str, bytes, Path)):
        files = np.array([files], dtype=object)
        single_file = True
    else:
        files = np.asarray(files, dtype=object)
        single_file = False

    ids, ids_array = _process_file_ids(
        ids=ids,
        shape=files.shape,
        is_receptor=is_receptor,
        single_file=single_file,
    )

    filepaths = np.empty(shape=files.shape, dtype=object)
    suffix = ".pdbqt" if is_receptor else ".dat"
    for file_idx in np.ndindex(files.shape):
        file_id = ids_array[file_idx]
        file = files[file_idx]
        if fileex.path.is_path(file):
            filepath = Path(file).resolve()
            if not filepath.is_file():
                raise exception.InputError(
                    name=argname_files,
                    message=f"The file '{file}' for file ID '{file_id}' "
                            f"at index '{file_idx}' does not exist."
                )
            if " " not in str(filepath):
                filepaths[file_idx] = filepath
            else:
                if not allow_copy:
                    raise exception.InputError(
                        name=argname_files,
                        message=f"The file '{file}' for file ID '{file_id}' "
                                f"at index '{file_idx}' contains spaces. "
                                "Please provide a path without spaces."
                    )
                final_filepath = (output_dir / file_id).with_suffix(suffix)
                if final_filepath.exists():
                    raise exception.InputError(
                        name=argname_files,
                        message=f"The file '{file}' for file ID '{file_id}' "
                                f"at index '{file_idx}' already exists in the output directory."
                    )
                shutil.copy2(filepath, final_filepath)
                filepaths[file_idx] = final_filepath
        else:
            final_filepath = (output_dir / file_id).with_suffix(suffix)
            if final_filepath.exists():
                raise exception.InputError(
                    name=argname_files,
                    message=f"The file '{file}' for file ID '{file_id}' "
                            f"at index '{file_idx}' already exists in the output directory."
                )
            final_filepath.write_text(file) if isinstance(file, str) else final_filepath.write_bytes(file)
            filepaths[file_idx] = final_filepath
    return files, ids, filepaths, ids_array, single_file


def _process_file_ids(
    ids: str | Sequence[tuple[str, Sequence[str]]] | None,
    shape: tuple[int, ...],
    is_receptor: bool,
    single_file: bool,
) -> tuple[list[tuple[str, list[str]]], np.ndarray]:
    arg_name = "file_ids" if is_receptor else "parameter_file_ids"
    prefix = "receptor" if is_receptor else "parameter"
    if ids is None:
        ids = []
        for axis_idx, axis_size in enumerate(shape):
            axis_label = f"{prefix}{axis_idx}"
            axis_element_labels = list(map(str, range(axis_size)))
            ids.append((axis_label, axis_element_labels))
    elif isinstance(ids, str):
        if not single_file:
            raise exception.InputError(
                name=arg_name,
                message="A single file ID was provided for multiple files."
            )
        if not ids.isalnum():
            raise exception.InputError(
                name=arg_name,
                message="File IDs must be alphanumeric."
            )
        ids = [(prefix, [ids])]
    elif isinstance(ids, Sequence):
        ids = list(ids)
        if single_file:
            raise exception.InputError(
                name=arg_name,
                message="File IDs were provided as a sequence, but only a single file was given."
            )
        if (n_axis_ids := len(ids)) != (n_axes := len(shape)):
            raise exception.InputError(
                name=arg_name,
                message=f"The number of 2-tuple elements must match the number of input file batch dimensions, "
                        f"but got {n_axis_ids} 2-tuples for {n_axes}D batch dimensions."
            )
        for axis_idx, axis_data, axis_size in enumerate(zip(ids, shape)):
            if not isinstance(axis_data, tuple):
                raise exception.InputError(
                    name=arg_name,
                    message="IDs for each axis must be a sequence of 2-tuples, "
                    f"but got {type(axis_data).__name__} at index {axis_idx}."
                )
            if len(axis_data) != 2:
                raise exception.InputError(
                    name=arg_name,
                    message="IDs for each axis must be a sequence of 2-tuples, "
                    f"but got a tuple with {len(axis_data)} elements at index {axis_idx}."
                )
            axis_id, element_ids = axis_data
            if not isinstance(axis_id, str):
                raise exception.InputError(
                    name=arg_name,
                    message="Axis IDs must be strings, "
                    f"but got {type(axis_id).__name__} for axis at index {axis_idx}."
                )
            if not axis_id.isalnum() or not axis_id[0].isalpha():
                raise exception.InputError(
                    name=arg_name,
                    message="Axis IDs must be alphanumeric and start with a letter, "
                    f"but got '{axis_id}' for axis at index {axis_idx}."
                )
            if not isinstance(element_ids, Sequence):
                raise exception.InputError(
                    name=arg_name,
                    message="Axis element IDs must be a sequence of strings, "
                    f"but got {type(element_ids).__name__} for axis '{axis_id}' at index {axis_idx}."
                )
            if len(element_ids) != axis_size:
                raise exception.InputError(
                    name=arg_name,
                    message=f"The number of element IDs for axis '{axis_id}' "
                            "must match the size of the input files along that axis, "
                            f"but got {len(element_ids)} IDs for {axis_size} files."
                )
            for elem_idx, elem_id in enumerate(element_ids):
                if not isinstance(elem_id, str):
                    raise exception.InputError(
                        name=arg_name,
                        message=f"File IDs must be a sequence of 2-tuples of (string, sequence), "
                        f"but the element at index {elem_idx} of the second element of the tuple at index {axis_idx} is not a string."
                    )
                if not elem_id.isalnum():
                    raise exception.InputError(
                        name=arg_name,
                        message=f"File IDs must be alphanumeric, but the element at index {elem_idx} of the second element of the tuple at index {axis_idx} is '{elem_id}'."
                    )
            if len(element_ids) != len(set(element_ids)):
                raise exception.InputError(
                    name=arg_name,
                    message=f"The file IDs for axis '{axis_id}' must be unique, "
                    f"but the provided IDs contain duplicates."
                )
        axis_ids = [axis_data[0] for axis_data in ids]
        if len(set(axis_ids)) != len(axis_ids):
            raise exception.InputError(
                name=arg_name,
                message="The file IDs must be unique across all axes, "
                f"but the provided IDs contain duplicates in the axis names: {axis_ids}."
            )
    else:
        raise exception.InputError(
            name=arg_name,
            message="File IDs must be a string, a sequence of 2-tuples, but got "
            f"{type(ids).__name__}."
        )
    ids_array = np.empty(shape=shape, dtype=object)
    for file_idx in np.ndindex(shape):
        file_id = []
        for axis_idx, element_idx in enumerate(file_idx):
            input_data = ids[axis_idx]
            file_id.append(f"{input_data[0]}_{input_data[1][element_idx]}")
        ids_array[file_idx] = "-".join(file_id)
    return ids, ids_array


def _process_ligand_types(ligand_types: str | Sequence[str]):
    if isinstance(ligand_types, str):
        ligand_types = [ligand_types]
    if not isinstance(ligand_types, Sequence):
        raise exception.InputError(
            name="ligand_types",
            message=f"Expected a string or a sequence of strings for ligand types, "
                    f"but got {type(ligand_types).__name__}."
        )
    if len(ligand_types) != len(set(ligand_types)):
        raise exception.InputError(
            name="ligand_types",
            message="Ligand types must be unique, "
                    f"but got duplicates in {ligand_types}."
        )
    all_ligand_types = []
    autodock_types = []
    extra_types = []
    anion_idx = None
    for idx, ligand_type in enumerate(ligand_types):
        if not isinstance(ligand_type, str):
            raise exception.InputError(
                name="ligand_types",
                message=f"Expected a string for ligand type at index {idx}, "
                        f"but got {type(ligand_type).__name__}."
            )
        ligand_type_normalized = ligand_type.casefold()
        for autodock_type in _AUTODOCK_LIGAND_TYPES:
            if ligand_type_normalized == autodock_type.casefold():
                all_ligand_types.append(autodock_type)
                autodock_types.append(autodock_type)
                break
        else:
            for extra_type in ("e+", "e-", "dsolv"):
                if ligand_type_normalized == extra_type:
                    all_ligand_types.append(extra_type)
                    extra_types.append(extra_type)
                    if extra_type == "e-":
                        anion_idx = idx
                    break
            else:
                raise exception.InputError(
                    name="ligand_types",
                    message=f"Unknown ligand type '{ligand_type}' at index {idx}. "
                            f"Supported types are (case-insensitive): {_AUTODOCK_LIGAND_TYPES + ['e+', 'e-', 'dsolv']}."
                )
    if not autodock_types:
        raise exception.InputError(
            name="ligand_types",
            message="At least one AutoDock ligand type must be provided, "
                    f"but got {ligand_types}."
        )
    return all_ligand_types, autodock_types, extra_types, anion_idx


def _get_map_filepaths_shape(
    receptor_file_ids: Sequence[tuple[str, Sequence[str]]] | None,
    parameter_file_ids: Sequence[tuple[str, Sequence[str]]] | None,
    receptor_files_shape: tuple[int, ...] | None,
    parameter_files_shape: tuple[int, ...] | None,
    ligand_types: Sequence[str],
    ligand_batch_label: str,
    field_batch_order: tuple[_BatchSection, _BatchSection, _BatchSection],
):
    if any(batch_name not in ("ligand", "parameter", "receptor") for batch_name in field_batch_order):
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
    if receptor_file_ids is None:
        batch_order.remove("receptor")
    if parameter_file_ids is None:
        batch_order.remove("parameter")
    single_ligand_type = isinstance(ligand_types, str)
    count_fields = 1 if single_ligand_type else len(ligand_types)
    if single_ligand_type:
        batch_order.remove("ligand")
    if not batch_order:
        return (1,), batch_order, (), 0
    batch_shape = []
    batch_labels = []
    ligand_axis_idx = 0
    for batch_name in batch_order:
        if batch_name == "receptor":
            batch_shape.extend(receptor_files_shape)
            batch_labels.extend(receptor_file_ids)
        elif batch_name == "parameter":
            batch_shape.extend(parameter_files_shape)
            batch_labels.extend(parameter_file_ids)
        elif batch_name == "ligand":
            ligand_axis_idx = len(batch_shape)
            batch_shape.append(count_fields)
            batch_labels.append((ligand_batch_label, ligand_types))
    return tuple(batch_shape), batch_order, tuple(batch_labels), ligand_axis_idx


def _process_map_filepaths(
    all_types: list[str],
    extra_types: list[str],
    autodock_types_filepaths: list[Path],
    extra_types_filepaths: list[Path],
) -> list[Path]:
    """Merge map filepaths back into the original order.

    Given an original list of strings `all_types`,
    a list of strings `extra_types` indicating
    which elements were removed,
    plus `autodock_types_filepaths` for the kept elements
    and `extra_types_filepaths` for the removed elements
    (each in the order they appeared in `all_types`),
    reconstruct a single list of values
    from `autodock_types_filepaths` and `extra_types_filepaths`
    aligned with `all_types`.
    """
    removed_set = set(extra_types)
    expected_removed = sum(1 for lt in all_types if lt in removed_set)
    expected_kept = len(all_types) - expected_removed
    if expected_removed != len(extra_types_filepaths):
        raise ValueError(
            f'Expected {expected_removed} removed values, got {len(extra_types_filepaths)}'
        )
    if expected_kept != len(autodock_types_filepaths):
        raise ValueError(
            f'Expected {expected_kept} cleaned values, got {len(autodock_types_filepaths)}'
        )
    cleaned_iter = iter(autodock_types_filepaths)
    removed_iter = iter(extra_types_filepaths)
    merged = []
    for lt in all_types:
        if lt in removed_set:
            merged.append(next(removed_iter))
        else:
            merged.append(next(cleaned_iter))
    return merged
