"""[PDBe](https://www.ebi.ac.uk/pdbe/) web API.

Protein Data Banl in Europe (PDBe)
is a founding member of the Worldwide Protein Data Bank (wwPDB),
which collects, organises and disseminates data
on biological macromolecular structures.

References
----------
- [PDBe API](https://www.ebi.ac.uk/pdbe/pdbe-rest-api)
- [PDBe REST API documentation](https://www.ebi.ac.uk/pdbe/api/doc/pdb.html)
- [PDBe Graph API documentation](https://www.ebi.ac.uk/pdbe/graph-api/pdbe_doc/)
- [PDBe Graph API paper](https://academic.oup.com/bioinformatics/article/37/21/3950/6291664)
- [PDBe API webinar series](https://pdbeurope.github.io/api-webinars)
- [PDBe API Training Notebooks](https://github.com/PDBeurope/pdbe-api-training)
- [PDBe GitHub organization](https://github.com/pdbeurope)
- [PDBe Graph API codebase](https://gitlab.ebi.ac.uk/pdbe-kb/services/pdbe-graph-api)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pylinks as pl
from pylinks.exception.api import WebAPIPersistentStatusCodeError


if TYPE_CHECKING:
    from typing import Sequence, Any, Literal


class PDBeAPI:
    """PDBe web API."""
    def __init__(
        self,
        base_url: str = "https://www.ebi.ac.uk/pdbe",
        status_num_tries: int = 5,
        status_sleep_init: int = 1,
        status_sleep_scale: int = 2,
        response_num_tries: int = 500,
        response_sleep_init: int = 10,
        response_sleep_scale: int = 1,
        retry_status_codes: Sequence[int] = (408, 429, 500, 502, 503, 504),
    ):
        self._base_url = pl.url.create(base_url)
        self._retry_config = pl.http.HTTPRequestRetryConfig(
            status_codes_to_retry=retry_status_codes,
            config_status=pl.http.RetryConfig(
                num_tries=status_num_tries,
                sleep_time_init=status_sleep_init,
                sleep_time_scale=status_sleep_scale,
            ),
            config_response=pl.http.RetryConfig(
                num_tries=response_num_tries,
                sleep_time_init=response_sleep_init,
                sleep_time_scale=response_sleep_scale,
            ),
        )
        return

    def ligand_sites(self, uniprot: str, explode: bool = False) -> dict[str, Any]:
        """Get ligand binding site residues for a UniProt entry.

        Parameters
        ----------
        uniprot
            UniProt accession code, e.g., "P12345".
        explode
            Explode the data into a flat dictionary.

        Returns
        -------
        Dictionary with ligand binding site residues.
        If `explode` is `False`, the data will be returned as a nested dictionary
        with the following key-value pairs:
        - "sequence" (str): Sequence of the entity (available for polymeric entities only).
          Usually there is one character per sequence position,
          but not if single-letter-code is actually multiple characters.
          Therefore, this string might be longer than the length field suggests.
        - "length" (int): Length of the sequence.
        - "dataType" (str): Type of data provided in the section, i.e., "LIGAND BINDING SITES".
        - "data" (list of dicts): List of ligand binding site residue information.
          Each dictionary corresponds to a unique ligand, and contains the following key-value pairs:
          - "name" (str): IUPAC name of the ligand.
          - "accession" (str): Accession code of the ligand.
          - "additionalData" (dict): Additional data about the ligand, including:
            - "numAtoms" (int): Number of atoms in the ligand.
            - "scaffoldId" (str): Scaffold ID of the ligand.
            - "coFactorId" (str): Co-factor ID of the ligand.
            - "reactionId" (str): Reaction ID of the ligand.
            - "chemblId" (str): ChEMBL ID of the ligand.
            - "drugBankId" (str): DrugBank ID of the ligand.
            - "targetUniProts" (list of str): List of UniProt IDs for the targets of the ligand.
            - "pdbEntries" (list of str): List of PDB IDs for the entries containing the ligand.
        - "residues" (list of dicts): List of residues interacting with the ligand.
          Each dictionary contains the following key-value pairs:
          - "startIndex" (int): Starting residue number as an mmcif-style residue index (within entity or 'struct_asym_id') in case of PDB.
          - "startCode" (str): Amino acid three-letter code for the residue in startIndex.
          - "endIndex" (int): Ending residue number as an mmcif-style residue index (within entity or 'struct_asym_id') in case of PDB.
          - "endCode" (str): Amino acid three-letter code for the residue in endIndex.
          - "indexType" (str): Name of the database for the "startIndex" and "endIndex" (e.g., "PDB", "UniProt").
          - "allPDBEntries" (list of str): List of all PDB entries containing the residue.
          - "interactingPDBEntries" (list of dicts): List of PDB entries where the ligand interacts with the specified residues.
            Each dictionary contains the following key-value pairs:
            - "pdbId" (str): PDB ID of the entry.
            - "entityId" (int): Entity ID within the PDB entry as an mmcif-style molecule number.
            - "chainIds" (str): Chain ID of the best chain within the PDB entry as an mmcif-style 'auth_asym_id'.

        If `explode` is `True`, the same dictionary is returned,
        but the "data" field is flattened/exploded into a list of dictionaries,
        where each dictionary contains all the terminal keys mentioned above under "data",
        i.e., "name", "accession", "numAtoms", "scaffoldId", "coFactorId",
        "reactionId", "chemblId", "drugBankId", "targetUniProts", "pdbEntries",
        "startIndex", "startCode", "endIndex", "endCode", "indexType", "allPDBEntries",
        "pdbId", "entityId", and "chainIds".
        The "data" field can thus be used directly to create a table such as a `pandas.DataFrame`.

        References
        ----------
        - [PDBe Graph API documentation: UniProt - Get ligand binding residues for a UniProt accession](https://www.ebi.ac.uk/pdbe/graph-api/pdbe_doc/#api-UniProt-GetUNPLigandSites)
        """
        response = self.request(endpoint=f"graph-api/uniprot/ligand_sites/{uniprot}")
        data = response.get(uniprot.upper(), {})
        if not explode or not data:
            return data
        data_flat = []
        for ligand in data["data"]:
            accession = ligand["accession"]
            name = ligand["name"]
            num_atoms = ligand["additionalData"]["numAtoms"]
            scaffold_id = ligand["additionalData"]["scaffoldId"]
            cofactor_id = ligand["additionalData"]["coFactorId"]
            reaction_id = ligand["additionalData"]["reactionId"]
            chembl_id = ligand["additionalData"]["chemblId"]
            drugbank_id = ligand["additionalData"]["drugBankId"]
            target_uniprots = ligand["additionalData"]["targetUniProts"]
            pdb_entries = ligand["additionalData"]["pdbEntries"]
            for target_residue in ligand["residues"]:
                start_code = target_residue["startCode"]
                start_index = target_residue["startIndex"]
                end_code = target_residue["endCode"]
                end_index = target_residue["endIndex"]
                index_type = target_residue["indexType"]
                all_pdb_entries = target_residue["allPDBEntries"]
                for pdb in target_residue["interactingPDBEntries"]:
                    row = {
                        "name": name,
                        "accession": accession,
                        "numAtoms": num_atoms,
                        "scaffoldId": scaffold_id,
                        "coFactorId": cofactor_id,
                        "reactionId": reaction_id,
                        "chemblId": chembl_id,
                        "drugBankId": drugbank_id,
                        "targetUniProts": target_uniprots,
                        "pdbEntries": pdb_entries,
                        "startIndex": start_index,
                        "startCode": start_code,
                        "endIndex": end_index,
                        "endCode": end_code,
                        "indexType": index_type,
                        "allPDBEntries": all_pdb_entries,
                        "pdbId": pdb["pdbId"],
                        "entityId": pdb["entityId"],
                        "chainIds": pdb["chainIds"],
                    }
                    data_flat.append(row)
        return {
            "sequence": data["sequence"],
            "length": data["length"],
            "dataType": data["dataType"],
            "data": data_flat,
        }

    def pdb_residue_listing(self, pdb_id: str, explode: bool = False) -> list[dict[str, Any]]:
        response = self.request(endpoint=f"api/pdb/entry/residue_listing/{pdb_id}")
        data = response.get(pdb_id.lower(), {}).get("molecules", [])
        if not explode or not data:
            return data
        data_flat = []
        for molecule in data:
            entity_id = molecule["entity_id"]
            for chain in molecule["chains"]:
                chain_id = chain["chain_id"]
                struct_asym_id = chain["struct_asym_id"]
                for residue in chain["residues"]:
                    entry = {
                        "entity_id": entity_id,
                        "chain_id": chain_id,
                        "struct_asym_id": struct_asym_id,
                        "residue_name": residue["residue_name"],
                        "residue_number": residue["residue_number"],
                        "author_residue_number": residue["author_residue_number"],
                        "author_insertion_code": residue["author_insertion_code"],
                        "observed_ratio": residue["observed_ratio"],
                    }
                    data_flat.append(entry)
        return data_flat

    def pdb_modified_residues(
        self,
        pdb_id: str | Sequence[str],
        explode: bool = False
    ) -> dict | list[dict[str, Any]]:
        single_pdb_id = False
        if isinstance(pdb_id, str):
            single_pdb_id = True
            pdb_id = [pdb_id]
        try:
            response = self.request(
                endpoint="api/pdb/entry/modified_AA_or_NA",
                verb="post",
                data=",".join(pdb_id),
            )
        except WebAPIPersistentStatusCodeError as e:
            if e.response.status_code == 404 and e.response.reason == "Not Found":
                # PDBe raises 404 Not Found for entries that do not have modified residues.
                if single_pdb_id or explode:
                    return []
                return {}
            else:
                raise e
        if single_pdb_id:
            return response.get(pdb_id[0].lower(), [])
        if not explode or not response:
            return response
        data_flat = []
        for pdb_id, residues in response.items():
            pdb_id = pdb_id.upper()
            for residue in residues:
                data_flat.append({"pdb_id": pdb_id} | residue)
        return data_flat

    def pdb_mutated_residues(
        self,
        pdb_id: str | Sequence[str],
        explode: bool = False
    ) -> dict | list[dict[str, Any]]:
        single_pdb_id = False
        if isinstance(pdb_id, str):
            single_pdb_id = True
            pdb_id = [pdb_id]
        try:
            response = self.request(
                endpoint="api/pdb/entry/mutated_AA_or_NA",
                verb="post",
                data=",".join(pdb_id),
            )
        except WebAPIPersistentStatusCodeError as e:
            if e.response.status_code == 404 and e.response.reason == "Not Found":
                # PDBe raises 404 Not Found for entries that do not have mutated residues.
                if single_pdb_id or explode:
                    return []
                return {}
            else:
                raise e
        if not explode or not response:
            if single_pdb_id:
                return response.get(pdb_id[0].lower(), [])
            return response
        data_flat = []
        for pdb_id, residues in response.items():
            pdb_id = pdb_id.upper()
            for residue in residues:
                details = {f"mutation_{k}": v for k, v in residue.pop("mutation_details", {}).items()}
                data_flat.append({"pdb_id": pdb_id} | residue | details)
        return data_flat

    def sifts_pdb_uniprot(self, pdb_id: str, explode: bool = False, expand: bool = False) -> dict | list[dict]:
        """Get [SIFTS](https://www.ebi.ac.uk/pdbe/docs/sifts/index.html) mappings from a PDB structure to UniProt.

        Parameters
        ----------
        pdb_id
            PDB ID of the structure.
        explode
            Explode the mappings into a flat dictionary.
        expand
            Expand the mappings to include all residues in each range.
            This only applies if `explode` is also `True`.

        Returns
        -------
        Dictionary with ligand binding site residues.
        If `explode` is `False`, the data will be returned as a nested dictionary
        where each key is a UniProt accession code and the value is a dictionary
        with the following key-value pairs:
        - "name" (str): Name of the UniProt entry, e.g., "CDK2_HUMAN".
        - "identifier" (str): UniProt identifier (usually same as name), e.g., "CDK2_HUMAN".
        - "mappings" (list of dicts): List of mappings from UniProt to PDB.
          Each dictionary contains the following key-value pairs:
          - "entity_id" (int): Entity ID in the PDB entry.
          - "chain_id" (str): Chain ID in the PDB entry.
          - "struct_asym_id" (str): Structural asymmetry ID in the PDB entry.
          - "unp_start" (int): Starting residue number in the UniProt entry.
          - "unp_end" (int): Ending residue number in the UniProt entry.
          - "start" (dict): Starting residue information in the PDB entry.
            This is a dictionary with the following key-value pairs:
            - "residue_number" (int): Residue number in the PDB entry.
            - "author_residue_number" (int): Author residue number in the PDB entry.
            - "author_insertion_code" (str): Author insertion code in the PDB entry.
          - "end" (dict): Ending residue information in the PDB entry.
            This is a dictionary with the same structure as "start".

        If `explode` is `True` and `expand` is `False`, a list of dictionaries is returned,
        where each dictionary contains all the terminal keys mentioned above,
        i.e., "accession", "name", "identifier", "entity_id", "chain_id",
        "struct_asym_id", "unp_start", "unp_end",
        "start_residue_number", "start_author_residue_number", "start_author_insertion_code",
        "end_residue_number", "end_author_residue_number", "end_author_insertion_code".

        If both `explode` and `expand` are `True`, a list of dictionaries is returned
        where each dictionary contains the keys
        "accession", "name", "identifier", "entity_id", "chain_id", "struct_asym_id",
        "unp_residue_number", "pdb_residue_number".

        References
        ----------
        - [PDBe SIFTS API documentation: SIFTS Mappings (PDB -> UniProt)](https://www.ebi.ac.uk/pdbe/api/sifts.html#sifts_apidiv_call_1_calltitle)
        """
        response = self.request(endpoint=f"api/mappings/uniprot/{pdb_id}")
        data = response.get(pdb_id.lower(), {}).get("UniProt", {})
        if not explode or not data:
            return data
        data_flat = []
        if not expand:
            for uniprot_accession, uniprot_data in data.items():
                name = uniprot_data.get("name")
                identifier = uniprot_data.get("identifier")
                for mapping in uniprot_data["mappings"]:
                    entry = {
                        "accession": uniprot_accession,
                        "name": name,
                        "identifier": identifier,
                        "entity_id": mapping["entity_id"],
                        "chain_id": mapping["chain_id"],
                        "struct_asym_id": mapping["struct_asym_id"],
                        "unp_start": mapping["unp_start"],
                        "unp_end": mapping["unp_end"],
                        "start_residue_number": mapping["start"]["residue_number"],
                        "start_author_residue_number": mapping["start"]["author_residue_number"],
                        "start_author_insertion_code": mapping["start"]["author_insertion_code"],
                        "end_residue_number": mapping["end"]["residue_number"],
                        "end_author_residue_number": mapping["end"]["author_residue_number"],
                        "end_author_insertion_code": mapping["end"]["author_insertion_code"],
                    }
                    data_flat.append(entry)
            return data_flat
        for uniprot_accession, uniprot_data in data.items():
            name = uniprot_data.get("name")
            identifier = uniprot_data.get("identifier")
            for mapping in uniprot_data["mappings"]:
                entity_id = mapping["entity_id"]
                chain_id = mapping["chain_id"]
                struct_asym_id = mapping["struct_asym_id"]
                pdb_start_res_num = mapping["start"].get("residue_number")
                pdb_start_author_res_num = mapping["start"].get("author_residue_number")
                for pdb_res_num_offset, uniprot_residue_num in enumerate(range(mapping["unp_start"], mapping["unp_end"] + 1)):
                    entry = {
                        "accession": uniprot_accession,
                        "name": name,
                        "identifier": identifier,
                        "entity_id": entity_id,
                        "chain_id": chain_id,
                        "struct_asym_id": struct_asym_id,
                        "unp_residue_number": uniprot_residue_num,
                        "pdb_residue_number": pdb_start_res_num + pdb_res_num_offset,
                        "pdb_author_residue_number": pdb_start_author_res_num + pdb_res_num_offset if pdb_start_author_res_num is not None else None,
                    }
                    data_flat.append(entry)
        return data_flat

    def validation_global_percentiles(
        self,
        pdb_id: str | Sequence[str],
        explode: bool = False
    ) -> dict | list[dict[str, Any]]:
        single_pdb_id = False
        if isinstance(pdb_id, str):
            single_pdb_id = True
            pdb_id = [pdb_id]
        response = self.request(
            endpoint="api/validation/global-percentiles/entry",
            verb="post",
            data=",".join(pdb_id),
        )
        if not explode or not response:
            if single_pdb_id:
                return response.get(pdb_id[0].lower(), {})
            return response
        data_flat = []
        for pdb_id, data in response.items():
            row = {"pdb_id": pdb_id.upper()}
            for metric_name, value_dict in data.items():
                for value_type, value in value_dict.items():
                    row[f"{metric_name}_{value_type}"] = value
            data_flat.append(row)
        return data_flat

    def validation_residuewise_outlier_summary(
        self,
        pdb_id: str | Sequence[str],
        explode: bool = False
    ) -> dict | list[dict[str, Any]]:
        single_pdb_id = False
        if isinstance(pdb_id, str):
            single_pdb_id = True
            pdb_id = [pdb_id]
        response = self.request(
            endpoint="api/validation/residuewise_outlier_summary/entry",
            verb="post",
            data=",".join(pdb_id),
        )
        if not explode or not response:
            if single_pdb_id:
                return response.get(pdb_id[0].lower(), {}).get("molecules", [])
            return response
        data_flat = []
        for pdb_id, data in response.items():
            pdb_id = pdb_id.upper()
            for molecule in data.get("molecules", []):
                entity_id = molecule["entity_id"]
                for chain in molecule["chains"]:
                    chain_id = chain["chain_id"]
                    struct_asym_id = chain["struct_asym_id"]
                    for model in chain["models"]:
                        model_id = model["model_id"]
                        for residue in model["residues"]:
                            residue_number = residue["residue_number"]
                            author_residue_number = residue["author_residue_number"]
                            author_insertion_code = residue["author_insertion_code"]
                            alt_code = residue["alt_code"]
                            for outlier_type in residue["outlier_types"]:
                                entry = {
                                    "pdb_id": pdb_id,
                                    "entity_id": entity_id,
                                    "chain_id": chain_id,
                                    "struct_asym_id": struct_asym_id,
                                    "model_id": model_id,
                                    "residue_number": residue_number,
                                    "author_residue_number": author_residue_number,
                                    "author_insertion_code": author_insertion_code,
                                    "alt_code": alt_code,
                                    "outlier_type": outlier_type,
                                }
                                data_flat.append(entry)
        return data_flat

    def request(
        self,
        endpoint: str,
        verb: Literal["post", "get"] | None = None,
        headers: dict[str, str] | None = None,
        data: Any | None = None,
        json: dict[str, Any] | None = None,
    ) -> dict | list:
        """Submit a request to the PDBe web API.

        This method is not meant to be called directly,
        unless you want to submit a custom request to the server,
        or if you want to use an endpoint that is not yet implemented in this class.

        Parameters
        ----------
        endpoint
            API endpoint to submit the request to,
            e.g., "pdb/entry/summary/{pdb_id}".
        verb
            HTTP verb to use for the request, either "post" or "get".
            If not provided, it will be determined based on the presence of `json`.
        headers
            Headers to include in the request.
        json
            JSON payload to include in the request body.

        Returns
        -------
        API response as a JSON decoded dictionary/list.
        """
        if verb is None:
            verb = "post" if json is not None or data is not None else "get"
        return pl.http.request(
            url=self._base_url / endpoint,
            verb=verb.upper(),
            data=data,
            json=json,
            headers=headers,
            response_type="json",
            retry_config=self._retry_config,
        )
