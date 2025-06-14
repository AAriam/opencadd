
from pathlib import Path

import pandas as pd
from pydantic import BaseModel
import pyserials
import sciapi
import scifile

from .path import INPUT, INTERMEDIATE


class Ligand(BaseModel):
    """Definition of a non-polymer molecule in a PDB structure.

    Attributes
    ----------
    res_name
        Residue name.
    res_seq
        Residue sequence number.
    chain_id
        Residue chain identifier.
    """
    res_name: str
    res_seq: int | None = None
    chain_id: str | None = None


class Protein(BaseModel):
    """Protein structure defining a single Protein structure."""
    pdb_id: str
    ref_ligand: Ligand | None = None
    chain_id: str | None = None
    ligands: list[Ligand] | None = None
    heterogens: list[Ligand] | None = None


class ProteinGroup(BaseModel):
    """Protein group defining a set of Protein structures."""

    id: str
    name: str
    uniprot_id: str | None = None
    ref_structure: Protein
    structures: list[Protein] | None = None
    color_bg: str | None = None
    color_text: str | None = None


class Dataset(BaseModel):
    """Evaluation dataset defining Protein structures."""
    groups: list[ProteinGroup]



class DatasetManager:
    """Manager for the evaluation dataset."""
    def __init__(
        self,
        dataset: Dataset,
    ):
        self._data = dataset
        self._pdbfile: dict[str, scifile.pdb.PDBFile] = {}
        self._pdb: scifile.pdb.PDBDataset = None
        self._pdbid_to_group: dict[str, ProteinGroup] = {}
        self._group_id_to_group: dict[str, ProteinGroup] = {
            group.id: group for group in self._data.groups
        }
        return

    @property
    def pdb(self) -> scifile.pdb.PDBDataset:

        class StyledDF(pd.DataFrame):
            @property
            def _constructor(self):
                return self.__class__

            def _repr_html_(self2):
                def color_groups(group_ids: pd.Series) -> list[str]:
                    return [
                        f"background-color: {self._group_id_to_group[group_id].color_bg or '#ffffff'}; color: {self._group_id_to_group[group_id].color_text or '#000000'}"
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

        def is_ref_structure(pdb_id: str) -> bool:
            for group in self._data.groups:
                if group.ref_structure.pdb_id.upper() == pdb_id.upper():
                    return True
            return False

        if self._pdb:
            return self._pdb
        if not self._pdbfile:
            self.load_pdb_files()
        pdb = scifile.pdb.merge(self._pdbfile.values())
        get_group_id = lambda pdb_id: self.group_from_pdb_id(pdb_id).id
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


    def iter_structures(self):
        """Iterate over the structures in the dataset."""
        for group in self._data.groups:
            yield group.ref_structure
            if group.structures:
                yield from group.structures
        return

    def load_pdb_files(self, dirpath: Path | str | None = None) -> None:
        """Load PDB files for all structures in the dataset.

        Parameters
        ----------
        dirpath
            Path to the directory where PDB files are stored or will be downloaded.
            If None, defaults to the `pdb_input` directory in the intermediate data directory.
        """
        dirpath = Path(dirpath) if dirpath else INTERMEDIATE / "pdb_input"
        dirpath.mkdir(parents=True, exist_ok=True)
        for structure in self.iter_structures():
            pdb_id = structure.pdb_id.upper()
            pdb_filepath = dirpath / f"{pdb_id}.pdb"
            if not pdb_filepath.exists():
                pdb_bytes = sciapi.pdb.file.entry(
                    pdb_id=pdb_id,
                    file_format="pdb",
                )
                pdb_filepath.write_bytes(pdb_bytes)
            else:
                pdb_bytes = pdb_filepath.read_bytes()
            pdbfile = scifile.pdb.read(pdb_bytes)
            self._validate_pdb_file(
                protein_group=structure,
                protein=structure,
                pdbfile=pdbfile,
                filepath=pdb_filepath,
            )
            self._pdbfile[pdb_id] = pdbfile
        return

    def group_from_pdb_id(self, pdb_id: str) -> ProteinGroup:
        """Get the ProteinGroup for a given PDB ID."""
        if self._pdbid_to_group:
            return self._pdbid_to_group[pdb_id.upper()]
        for group in self._data.groups:
            for structure in [group.ref_structure] + (group.structures or []):
                self._pdbid_to_group[structure.pdb_id.upper()] = group
        return self._pdbid_to_group[pdb_id.upper()]

    def _validate_pdb_file(
        self,
        protein_group: ProteinGroup,
        protein: Protein,
        pdbfile: scifile.pdb.PDBFile,
        filepath: Path,
    ) -> None:
        """Validate the PDB file for a given protein structure."""
        # Check if the PDB ID matches
        if pdbfile.header.id_code != protein.pdb_id:
            raise ValueError(
                f"PDB ID mismatch: {pdbfile.pdb_id} != {protein.pdb_id}"
            )
        return


def load_from_file(
    filepath: Path | str | None = None,
) -> Dataset:
    """Load the data defining the evaluation dataset.

    Parameters
    ----------
    filepath
        Path to the dataset file (JSON, YAML, or TOML).
    """
    filepath = Path(filepath) if filepath else INPUT / "evaluation_dataset.yaml"
    file = pyserials.read.from_file(
        path=filepath,
        json_strict=True,
        yaml_safe=True,
        toml_as_dict=True,
    )
    data = file["data"]
    schema = file["schema"]
    # pyserials.validate.jsonschema(
    #     data=data,
    #     schema=schema,
    #     fill_defaults=True,
    # )
    return DatasetManager(dataset=Dataset(groups=data))


