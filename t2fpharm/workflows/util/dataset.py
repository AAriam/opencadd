
from pathlib import Path

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
        return

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


