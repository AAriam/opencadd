from pathlib import Path
import shutil

import pandas as pd

import pyserials
import pkgdata

import sciapi
import scifile
import caddpy

import t2fpharm


class Manager:
    def __init__(
        self,
        data: pd.DataFrame,
        group_color: dict[str, dict[str, str]],
    ):
        self._data = data
        self._group_color = group_color
        self._pdb = None
        return

    @property
    def data(self) -> pd.DataFrame:
        """Dataset as a pandas DataFrame."""
        return self._data

    @property
    def pdb(self) -> scifile.pdb.PDBDataset:
        class StyledDF(pd.DataFrame):
            @property
            def _constructor(self):
                return self.__class__

            def _repr_html_(self2):
                def color_groups(group_ids: pd.Series) -> list[str]:
                    return [
                        f"background-color: {self._group_color[group_id]["bg"] or '#ffffff'}; color: {self._group_color[group_id]["text"] or '#000000'}"
                        for group_id in group_ids
                    ]
                def color_is_ref(is_ref: pd.Series) -> list[str]:
                    return [
                        "background-color: rgb(0 200 0); color: #000000" if ref else ""
                        for ref in is_ref
                    ]
                out = self2.style
                if "group_id" in out.columns:
                    out = out.apply(color_groups, subset='group_id')
                if "is_ref" in out.columns:
                    out = out.apply(color_is_ref, subset='is_ref')
                return out._repr_html_()

        if self._pdb:
            return self._pdb
        pdb = scifile.pdb.merge(self._data["pdb_raw"].values)
        get_group_id = lambda pdb_id: self.data.loc[pdb_id]["group_id"]
        is_ref_structure = lambda pdb_id: self.data.loc[pdb_id]["is_ref"]
        styled_params = {}
        for record in (
            "header",
            "obslte",
            "title",
            "split",
            "caveat",
            "compnd",
            "source",
            "keywds",
            "expdta",
            "nummdl",
            "mdltyp",
            "author",
            "revdat",
            "sprsde",
            "jrnl",
            "dbref",
            "seqadv",
            "seqres",
            "modres",
            "het",
            "hetnam",
            "helix",
            "sheet",
            "ssbond",
            "link",
            "cispep",
            "site",
            "cryst1",
            "origx",
            "scale",
            "mtrix",
            "atom",
            "anisou",
            "ter",
            "conect",
        ):
            df = getattr(pdb, record)
            if df is None:
                continue
            group_ids = df["id_code"].map(get_group_id)
            is_ref = df["id_code"].map(is_ref_structure)
            df.insert(0, "group_id", group_ids)
            df.insert(1, "is_ref", is_ref)
            styled_params[record] = StyledDF(df)
        if hasattr(pdb, "remark"):
            remark_params = {}
            for attr_name in ("full_text", "related_publications", "resolution", "format"):
                df = getattr(pdb.remark, attr_name)
                if df is None:
                    continue
                group_ids = df["id_code"].map(get_group_id)
                is_ref = df["id_code"].map(is_ref_structure)
                df.insert(0, "group_id", group_ids)
                df.insert(1, "is_ref", is_ref)
                init_param_name = attr_name if attr_name != "full_text" else "full"
                remark_params[init_param_name] = StyledDF(df)
            styled_params["remark"] = scifile.pdb.records.RemarkDataset(**remark_params)
        self._pdb = scifile.pdb.PDBDataset(**styled_params)
        return self._pdb


def load(
    dirpath_data: Path | str | None = None,
    *,
    filepath_inputs: Path | str  = "inputs.yaml",
    dirpath_pdb_raw: Path | str = "structure/1-pdb-raw",
    dirpath_pdb_fixed: Path | str = "structure/2-pdb-fixed",
    dirpath_pdb_apo: Path | str = "structure/3-pdb-apo",
    dirpath_pdbqt: Path | str = "structure/4-pdbqt",
    dirpath_pocket: Path | str = "pocket",
    dirpath_autogrid: Path | str = "autogrid",
    dirpath_field: Path | str = "field",
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
    dirpath_data = Path(dirpath_data) if dirpath_data else pkgdata.get_package_path_from_caller(top_level=True) / "data"
    filepath_inputs = dirpath_data / filepath_inputs
    dirpath_pdb_raw = dirpath_data / dirpath_pdb_raw
    dirpath_pdb_fixed = dirpath_data / dirpath_pdb_fixed
    dirpath_pdb_apo = dirpath_data / dirpath_pdb_apo
    dirpath_pdbqt = dirpath_data / dirpath_pdbqt
    dirpath_pocket = dirpath_data / dirpath_pocket
    dirpath_autogrid = dirpath_data / dirpath_autogrid
    dirpath_field = dirpath_data / dirpath_field

    for dirpath in [
        dirpath_pdb_raw,
        dirpath_pdb_fixed,
        dirpath_pdb_apo,
        dirpath_pdbqt,
        dirpath_pocket,
        dirpath_autogrid,
        dirpath_field,
    ]:
        dirpath.mkdir(parents=True, exist_ok=True)

    input_data = pyserials.read.from_file(
        path=filepath_inputs,
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
        group = _make_group(group_data)
        row = _make_structure(
            group=group,
            structure=group_data["ref_structure"],
            dirpath_pdb_raw=dirpath_pdb_raw,
            dirpath_pdb_fixed=dirpath_pdb_fixed,
            dirpath_pdb_apo=dirpath_pdb_apo,
            dirpath_pdbqt=dirpath_pdbqt,
            dirpath_pocket=dirpath_pocket,
            dirpath_autogrid=dirpath_autogrid,
            dirpath_field=dirpath_field,
            pocket_data=inputs["pocket"],
            field_data=inputs["field"],
            is_ref=True
        )
        rows.append(row)
        for structure_data in group_data.get("structures", []):
            row = _make_structure(
                group=group,
                structure=structure_data,
                dirpath_pdb_raw=dirpath_pdb_raw,
                dirpath_pdb_fixed=dirpath_pdb_fixed,
                dirpath_pdb_apo=dirpath_pdb_apo,
                dirpath_pdbqt=dirpath_pdbqt,
                dirpath_pocket=dirpath_pocket,
                dirpath_autogrid=dirpath_autogrid,
                dirpath_field=dirpath_field,
                pocket_data=inputs["pocket"],
                field_data=inputs["field"],
                is_ref=False
            )
            rows.append(row)
    df = pd.DataFrame(rows).convert_dtypes()
    df.set_index("pdb_id", inplace=True, drop=False)
    return Manager(data=df, group_color=group_color)


def _make_group(group: dict) -> dict:
    """Create a group dictionary with relevant fields."""
    return {
        "group_id": group["id"],
        "group_name": group["name"],
        "uniprot_id": group.get("uniprot_id"),
    }


def _make_structure(
    group: dict,
    structure: dict,
    dirpath_pdb_raw: Path,
    dirpath_pdb_fixed: Path,
    dirpath_pdb_apo: Path,
    dirpath_pdbqt: Path,
    dirpath_pocket: Path,
    dirpath_autogrid: Path,
    dirpath_field: Path,
    pocket_data: dict,
    field_data: dict,
    is_ref: bool = False
):
    heterogens = []
    for heterogen in structure.get("heterogens", []):
        heterogens.append(
            {
                "res_name": heterogen.get("res_name"),
                "chain_id": heterogen.get("chain_id"),
                "res_seq": heterogen.get("res_seq"),
            }
        )
    heterogens = pd.DataFrame(heterogens, columns=["res_name", "chain_id", "res_seq"])
    pdb_id = structure["pdb_id"].upper()
    structure_full = group | {
        "pdb_id": pdb_id,
        "is_ref": is_ref,
        "chain_id": structure.get("chain_id"),
        "ligand_res_name": structure.get("ref_ligand", {}).get("res_name"),
        "ligand_chain_id": structure.get("ref_ligand", {}).get("chain_id"),
        "ligand_res_seq": structure.get("ref_ligand", {}).get("res_seq"),
        "heterogens": heterogens,
        "filepath_pdb_raw": dirpath_pdb_raw / f"{pdb_id}.pdb",
        "filepath_pdb_fixed": dirpath_pdb_fixed / f"{pdb_id}.pdb",
        "filepath_pdb_apo": dirpath_pdb_apo / f"{pdb_id}.pdb",
        "filepath_pdbqt": dirpath_pdbqt / f"{pdb_id}.pdbqt",
        "filepath_pocket": dirpath_pocket / f"{pdb_id}.yaml",
    }
    _prepare_structure(structure_full, is_ref=is_ref)
    _prepare_pocket(
        structure=structure_full,
        dirpath_pocket=dirpath_pocket,
        grid_spacing=pocket_data["grid_spacing"],
        ligand_radii_offset=pocket_data["ligand_radii_offset"],
    )
    _prepare_field(
        structure=structure_full,
        dirpath_autogrid=dirpath_autogrid,
        dirpath_field=dirpath_field,
        ligand_types=field_data["ligand_types"],
        smooth=field_data["smooth"],
        dielectric=field_data["dielectric"],
    )
    return structure_full


def _prepare_structure(structure: dict, is_ref: bool = False):
    try:
        # Raw structure
        filepath_pdb_raw = structure["filepath_pdb_raw"]
        if not filepath_pdb_raw.is_file():
            pdb_raw_bytes = sciapi.pdb.file.entry(pdb_id=structure["pdb_id"], file_format="pdb")
            filepath_pdb_raw.write_bytes(pdb_raw_bytes)
        else:
            pdb_raw_bytes = filepath_pdb_raw.read_bytes()
        structure["pdb_raw"] = pdb = scifile.pdb.read(pdb_raw_bytes)

        # Fixed structure
        filepath_pdb_fixed = structure["filepath_pdb_fixed"]
        if not filepath_pdb_fixed.is_file():
            (
                pdb_fixed_str,
                missing_residues,
                nonstandard_residues,
                missing_atoms,
                missing_terminals
            ) = caddpy.chemsys.fix_pdb(
                file=pdb_raw_bytes,
                add_missing_residues=True,
                replace_nonstandard_residues=False,
                add_missing_heavy_atoms=True,
                add_missing_atoms_seed=42,
                add_missing_hydrogens=7.0,
                keep_ids=True,
            )
            filepath_pdb_fixed.write_text(pdb_fixed_str)
        else:
            pdb_fixed_str = filepath_pdb_fixed.read_text()
        structure["complex"] = comp = t2fpharm.receptor.from_pdb(pdb_fixed_str)

        # Apo structure
        filepath_pdb_apo = structure["filepath_pdb_apo"]
        if not filepath_pdb_apo.is_file():
            structure["receptor"] = receptor = comp.select(comp.composition.atoms["res_poly"])
            filepath_pdb_apo.write_text(str(receptor.to_pdb()))
        else:
            structure["receptor"] = receptor = t2fpharm.receptor.from_pdb(filepath_pdb_apo)

        filepath_pdbqt = structure["filepath_pdbqt"]
        if not filepath_pdbqt.is_file():
            pdbqt_str = receptor.to_pdbqt(
                autobond=False,
                rigid=True,
                combine=False,
                flexible=False,
                preserve_serials=True,
                preserve_hydrogens=False,
                preserve_names=True,
                charge_model="gasteiger",
                add_hydrogens=False,
            )
            filepath_pdbqt.write_text(pdbqt_str)
    except Exception as e:
        raise RuntimeError(
            f"Failed to prepare structure {structure['pdb_id']}: {e}"
        ) from e
    return


def _prepare_pocket(
    structure: dict,
    dirpath_pocket: Path,
    grid_spacing: float,
    ligand_radii_offset: float,
):
    """Prepare the pocket for the structure."""
    filepath_pocket = dirpath_pocket / f"{structure['pdb_id']}.yaml"
    if not filepath_pocket.is_file():
        atoms = structure["complex"].composition.atoms
        ligand_mask = (
            (atoms["res_name"] == structure["ligand_res_name"]) &
            (atoms["chain_id"] == structure["ligand_chain_id"]) &
            (atoms["res_seq"] == structure["ligand_res_seq"])
        )
        if ligand_mask.sum() == 0:
            raise ValueError(
                f"Ligand {structure['ligand_res_name']} "
                f"not found in structure {structure['pdb_id']}"
            )
        structure["pocket"] = pocket = t2fpharm.pocket.from_ligand(
            system=structure["complex"],
            ligand_mask=ligand_mask,
            ligand_radii=None,
            grid=grid_spacing,
            ligand_radii_offset=ligand_radii_offset,
        )
        pocket_data = pocket.to_dict()
        pyserials.write.to_yaml_file(
            data=pocket_data,
            path=filepath_pocket,
        )
    else:
        pocket_data = pyserials.read.yaml_from_file(filepath_pocket)
        structure["pocket"] = t2fpharm.pocket.from_data(**pocket_data)
    return


def _prepare_field(
    structure: dict,
    dirpath_autogrid: Path,
    dirpath_field: Path,
    ligand_types: list[str],
    smooth: float,
    dielectric: float,
):
    filepath_field = dirpath_field / f"{structure['pdb_id']}.json"
    if not filepath_field.is_file():
        dirpath_autogrid = dirpath_autogrid / structure["pdb_id"]
        if dirpath_autogrid.exists():
            shutil.rmtree(dirpath_autogrid)
        dirpath_autogrid.mkdir(parents=True, exist_ok=True)
        pocket_data = structure["pocket"].to_dict()
        grid_data = {k: v for k, v in pocket_data.items() if k.startswith("grid_")}
        structure["field"] = field = t2fpharm.field.from_autogrid(
            receptor_files=structure["filepath_pdbqt"],
            ligand_types=ligand_types,
            smooth=smooth,
            dielectric=dielectric,
            output_dir=dirpath_autogrid,
        )
        field_data = field.to_dict()
        pyserials.write.to_json_file(
            data=field_data,
            path=filepath_field,
        )
    else:
        field_data = pyserials.read.json_from_file(filepath_field)
        structure["field"] = t2fpharm.field.from_data(**field_data)
    return
