from pathlib import Path

import pandas as pd
from pdbfixer import PDBFixer
from openmm.app import PDBFile

import pyserials

import sciapi
import scifile


class Dataset:
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
    filepath_inputs: Path | str  = "data/inputs.yaml",
    dirpath_pdb_raw: Path | str = "data/pdb_raw",
    dirpath_pdb_fixed: Path | str = "data/pdb_fixed",
    dirpath_pdb_apo: Path | str = "data/pdb_apo",
    dirpath_pdbqt: Path | str = "data/pdbqt",
) -> Dataset:
    """Load the data defining the evaluation dataset.

    Parameters
    ----------
    filepath_inputs
        Path to the inputs file (JSON, YAML, or TOML).
    """
    def make_group(group):
        return {
            "group_id": group["id"],
            "group_name": group["name"],
            "uniprot_id": group.get("uniprot_id"),
        }

    def make_structure(group, structure, is_ref: bool = False):
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
            "filepath_pdb_apo": dirpath_pdb_apo / f"{pdb_id}.pdb" if is_ref else None,
            "filepath_pdbqt": dirpath_pdbqt / f"{pdb_id}.pdbqt" if is_ref else None,
        }
        prepare_structure(structure_full, is_ref=is_ref)
        return structure_full

    def prepare_structure(structure, is_ref: bool = False):
        filepath_pdb_raw = structure["filepath_pdb_raw"]
        if not filepath_pdb_raw.is_file():
            pdb_raw_bytes = sciapi.pdb.file.entry(pdb_id=structure["pdb_id"], file_format="pdb")
            filepath_pdb_raw.write_bytes(pdb_raw_bytes)
        else:
            pdb_raw_bytes = filepath_pdb_raw.read_bytes()
        pdb = scifile.pdb.read(pdb_raw_bytes)
        structure["pdb_raw"] = pdb
        filepath_pdb_fixed = structure["filepath_pdb_fixed"]
        if not filepath_pdb_fixed.is_file():
            fixer = PDBFixer(filename=str(filepath_pdb_raw))
            fixer.findMissingResidues()
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()
            fixer.addMissingHydrogens(7.0)
            PDBFile.writeFile(fixer.topology, fixer.positions, str(filepath_pdb_fixed), keepIds=True)
        # if is_ref:
        #     filepath_pdb_apo = structure["filepath_pdb_apo"]
        #     if not filepath_pdb_apo.is_file():

        return


    filepath_inputs = Path(filepath_inputs)
    dirpath_pdb_raw = Path(dirpath_pdb_raw)
    dirpath_pdb_fixed = Path(dirpath_pdb_fixed)
    dirpath_pdb_apo = Path(dirpath_pdb_apo)
    dirpath_pdbqt = Path(dirpath_pdbqt)

    for dirpath in [dirpath_pdb_raw, dirpath_pdb_fixed, dirpath_pdb_apo, dirpath_pdbqt]:
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
        group = make_group(group_data)
        row = make_structure(group, group_data["ref_structure"], is_ref=True)
        rows.append(row)
        for structure_data in group_data.get("structures", []):
            row = make_structure(group, structure_data)
            rows.append(row)
    df = pd.DataFrame(rows).convert_dtypes()
    df.set_index("pdb_id", inplace=True, drop=False)
    return Dataset(data=df, group_color=group_color)
