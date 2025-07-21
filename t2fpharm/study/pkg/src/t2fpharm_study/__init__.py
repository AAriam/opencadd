from pathlib import Path

import pandas as pd
import pkgdata
import pyserials

from t2fpharm_study.manager import Manager


__all__ = ["manager"]


def manager(
    dirpath_data: Path | str | None = None,
    *,
    filepath_inputs: Path | str  = "inputs.yaml",
    dirpath_pdb_raw: Path | str = "structure/1-pdb-raw",
    dirpath_pdb_fixed: Path | str = "structure/2-pdb-fixed",
    dirpath_pdb_aligned: Path | str = "structure/3-pdb-aligned",
    dirpath_pdb_apo: Path | str = "structure/4-pdb-apo",
    dirpath_pdbqt: Path | str = "structure/5-pdbqt",
    dirpath_affinity: Path | str = "affinity",
    dirpath_pocket: Path | str = "pocket",
    dirpath_autogrid: Path | str = "autogrid",
    dirpath_field: Path | str = "field",
    dirpath_ligand_plip: Path | str = "ligand/plip",
    dirpath_ligand_features: Path | str = "ligand/features",
    dirpath_results: Path | str = "results",
) -> Manager:
    """Load the manager.

    Parameters
    ----------
    dirpath_data
        Path to the data directory.
        If not provided, the default data directory is used.
    filepath_inputs
        Path to the inputs file (JSON, YAML, or TOML)
        relative to `dirpath`.
    """
    dirpath_data = (
        Path(dirpath_data) if dirpath_data else
        pkgdata.get_package_path_from_caller(top_level=True) / "data"
    )
    input_data = pyserials.read.from_file(
        path=dirpath_data / filepath_inputs,
        json_strict=True,
        yaml_safe=True,
        toml_as_dict=True,
    )
    inputs = input_data["data"]
    rows = []
    group_color = {}
    for group_data in inputs["receptor_groups"]:
        group_color[group_data["id"]] = {
            "bg": group_data.get("color_bg"),
            "text": group_data.get("color_text"),
        }
        group = {
            "group_id": group_data["id"],
            "group_name": group_data["name"],
            "uniprot_id": group_data.get("uniprot_id"),
        }
        row = _make_structure(
            group=group,
            structure=group_data["ref_structure"],
            is_ref=True
        )
        rows.append(row)
        for structure_data in group_data.get("structures", []):
            row = _make_structure(
                group=group,
                structure=structure_data,
                is_ref=False
            )
            rows.append(row)
    df = pd.DataFrame(rows).convert_dtypes()
    df.set_index("pdb_id", inplace=True, drop=False)
    return Manager(
        dataset=df,
        pocket_inputs=inputs["pocket"],
        field_inputs=inputs["field"],
        job_inputs=inputs["job"],
        group_color=group_color,
        dirpath_data=dirpath_data,
        dirpath_pdb_raw=dirpath_pdb_raw,
        dirpath_pdb_fixed=dirpath_pdb_fixed,
        dirpath_pdb_aligned=dirpath_pdb_aligned,
        dirpath_pdb_apo=dirpath_pdb_apo,
        dirpath_pdbqt=dirpath_pdbqt,
        dirpath_affinity=dirpath_affinity,
        dirpath_pocket=dirpath_pocket,
        dirpath_autogrid=dirpath_autogrid,
        dirpath_field=dirpath_field,
        dirpath_ligand_plip=dirpath_ligand_plip,
        dirpath_ligand_features=dirpath_ligand_features,
        dirpath_results=dirpath_results,
    )


def _make_structure(
    group: dict,
    structure: dict,
    is_ref: bool = False
):
    structure_full = group | {
        "pdb_id": structure["pdb_id"].upper(),
        "is_ref": is_ref,
        "chain_id": structure.get("chain_id"),
        "ligand_res_name": structure.get("ref_ligand", {}).get("res_name"),
        "ligand_chain_id": structure.get("ref_ligand", {}).get("chain_id"),
        "ligand_res_seq": structure.get("ref_ligand", {}).get("res_seq"),
    }
    return structure_full
