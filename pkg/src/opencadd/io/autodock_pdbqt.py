"""Read/Write AutoDock PDBQT files."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from scids.typing import PathLike

from collections.abc import Sequence
from pathlib import Path

from openbabel import pybel

import opencadd as oc
from opencadd import _typing




def read_file(filepath: PathLike):
    """
    Parse a PDBQT file.

    Parameters
    ----------
    filepath : PathLike
        Path to PDB file.

    Returns
    -------
    dict(str, pandas.DataFrame)
        A dictionary of record names (e.g. "ATOM") and their corresponding dataframes.

    References
    ----------
    PDB file format documentation:
        https://ftp.wwpdb.org/pub/pdb/doc/format_descriptions/Format_v33_A4.pdf
    """

    def parse_atom_records(record_lines_atom: np.ndarray):
        """
        Parse ATOM records
        """
        columns = {
            "serial": ((7, 11), int),
            "name": ((13, 16), (str, 4)),
            "altLoc": ((17, 17), (str, 1)),
            "resName": ((18, 20), (str, 3)),
            "chainID": ((22, 22), (str, 1)),
            "resSeq": ((23, 26), int),
            "iCode": ((27, 27), (str, 1)),
            "x": ((31, 38), float),
            "y": ((39, 46), float),
            "z": ((47, 54), float),
            "occupancy": ((55, 60), float),
            "tempFactor": ((61, 66), float),
            "partial_charge": ((67, 76), float),
            "autodock_atom_type": ((78, 79), (str, 2)),
        }

        df = pd.DataFrame()
        for col_name, (col_range, col_dtype) in columns.items():
            df[col_name] = np.char.strip(
                extract_column_from_string_array(
                    array=record_lines_atom, char_range=(col_range[0] - 1, col_range[1])
                )
            ).astype(col_dtype)

        autodock_atom_types_ids = np.array([atom_type.name for atom_type in autodock.Autodock4AtomType])
        autodock_atom_types_data = [
            np.array([getattr(atom_type, attr) for atom_type in autodock.Autodock4AtomType])
            for attr in ["hbond_status", "hbond_count"]
        ]
        indices_target_atom_types = np.where(
            df.autodock_atom_type.values[..., np.newaxis] == autodock_atom_types_ids
        )[1]
        df["hbond_acc"] = autodock_atom_types_data[0][indices_target_atom_types] == 1
        df["hbond_don"] = autodock_atom_types_data[0][indices_target_atom_types] == -1
        df["hbond_count"] = autodock_atom_types_data[1][indices_target_atom_types]
        return df

    record_parsers = {"ATOM": parse_atom_records}

    with open(filepath[0]) as f:
        lines = np.array(f.readlines())

    records = dict()
    for record, parser in record_parsers.items():
        record_mask = np.char.startswith(lines, prefix=record)
        records[record] = parser(lines[record_mask])

    trajectory = records["ATOM"][["x", "y", "z"]].to_numpy()[np.newaxis]

    return cls(
        atom_data=records["ATOM"],
        trajectory=trajectory,
    )



def write_from_ensemble(
    ensemble,
    output_filename: str | None = None,
    output_path: _typing.PathLike = None,
    models: int | Sequence[int] | None = None,
):
    pdb_strings = oc.io.pdb.write.from_chemsys(system=ensemble, models=models, separate_models=True)
    return_vals = []
    temp_filepath = Path.cwd() / "_temp_opencadd.pdb"
    for pdb_string in pdb_strings:
        with open(temp_filepath, "w") as f:
            f.write(pdb_string)
        return_vals.append(from_pdb_filepath(filepath=temp_filepath))
    return return_vals


def write_from_pdb_filepath(
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
