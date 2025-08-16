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

    def ligand_sites(self, uniprot: str, explode: bool = False) -> dict:
        """Get ligand binding site residues for a UniProt accession.

        Parameters
        ----------
        uniprot
            UniProt accession, e.g., "P12345".
        """
        response = self.request(endpoint=f"graph-api/uniprot/ligand_sites/{uniprot}")
        data = response[uniprot.upper()]
        if not explode:
            return data
        rows = []
        for ligand in data["data"]:
            lig_id = ligand["accession"]
            lig_name = ligand["name"]
            lig_atom_count = ligand["additionalData"]["numAtoms"]
            lig_scaffold_id = ligand["additionalData"]["scaffoldId"]
            lig_cofactor_id = ligand["additionalData"]["coFactorId"]
            lig_reaction_id = ligand["additionalData"]["reactionId"]
            lig_chembl_id = ligand["additionalData"]["chemblId"]
            lig_drugbank_id = ligand["additionalData"]["drugBankId"]
            lig_target_uniprot_codes = ligand["additionalData"]["targetUniProts"]
            lig_pdb_ids = ligand["additionalData"]["pdbEntries"]
            for residue in ligand["residues"]:
                res_start_name = residue["startCode"]
                res_start_num = residue["startIndex"]
                res_end_name = residue["endCode"]
                res_end_num = residue["endIndex"]
                res_num_db = residue["indexType"]
                for pdb in residue["interactingPDBEntries"]:
                    row = {
                        "res_start_name": res_start_name,
                        "res_start_num": res_start_num,
                        "res_end_name": res_end_name,
                        "res_end_num": res_end_num,
                        "res_num_db": res_num_db,
                        "pdb_id": pdb["pdbId"],
                        "entity_id": pdb["entityId"],
                        "chain_id": pdb["chainIds"],
                        "lig_id": lig_id,
                        "lig_name": lig_name,
                        "lig_atom_count": lig_atom_count,
                        "lig_scaffold_id": lig_scaffold_id,
                        "lig_cofactor_id": lig_cofactor_id,
                        "lig_reaction_id": lig_reaction_id,
                        "lig_chembl_id": lig_chembl_id,
                        "lig_drugbank_id": lig_drugbank_id,
                        "lig_target_uniprot_codes": lig_target_uniprot_codes,
                        "lig_pdb_ids": lig_pdb_ids,
                    }
                    rows.append(row)
        return {
            "residues": rows,
            "unp_seq": data["sequence"],
            "unp_seq_len": data["length"],
        }

    def sifts_pdb_uniprot(self, pdb_id: str, explode: bool = False) -> dict:
        """Get [SIFTS](https://www.ebi.ac.uk/pdbe/docs/sifts/index.html) mappings from a PDB structure to UniProt.

        Parameters
        ----------
        pdb_id
            PDB ID of the structure.
        explode
            Explode the mappings into a flat dictionary.
        """
        response = self.request(endpoint=f"api/mappings/uniprot/{pdb_id}")
        data = response[pdb_id.lower()]["UniProt"]
        if not explode:
            return data
        exploded = {
            "unp_code": [],
            "unp_id": [],
            "unp_name": [],
            "unp_res_num": [],
            "pdb_entity_id": [],
            "pdb_chain_id": [],
            "pdb_struct_asym_id": [],
            "pdb_res_num": [],
            "pdb_res_num_author": [],
        }
        for uniprot_accession, uniprot_data in data.items():
            uniprot_name = uniprot_data.get("name")
            uniprot_id = uniprot_data.get("identifier")
            for mapping in uniprot_data["mappings"]:
                pdb_entity_id = mapping["entity_id"]
                pdb_chain_id = mapping["chain_id"]
                pdb_struct_asym_id = mapping["struct_asym_id"]
                pdb_start_res_num = mapping["start"].get("residue_number")
                pdb_start_author_res_num = mapping["start"].get("author_residue_number")
                for pdb_res_num_offset, uniprot_residue_num in enumerate(range(mapping["unp_start"], mapping["unp_end"] + 1)):
                    exploded["unp_code"].append(uniprot_accession)
                    exploded["unp_id"].append(uniprot_id)
                    exploded["unp_name"].append(uniprot_name)
                    exploded["unp_res_num"].append(uniprot_residue_num)

                    exploded["pdb_entity_id"].append(pdb_entity_id)
                    exploded["pdb_chain_id"].append(pdb_chain_id)
                    exploded["pdb_struct_asym_id"].append(pdb_struct_asym_id)
                    exploded["pdb_res_num"].append(
                        pdb_start_res_num + pdb_res_num_offset if pdb_start_res_num is not None else None
                    )
                    exploded["pdb_res_num_author"].append(
                        pdb_start_author_res_num + pdb_res_num_offset if pdb_start_author_res_num is not None else None
                    )
        return exploded

    def request(
        self,
        endpoint: str,
        verb: Literal["post", "get"] | None = None,
        headers: dict[str, str] | None = None,
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
            verb = "post" if json is not None else "get"
        return pl.http.request(
            url=self._base_url / endpoint,
            verb=verb.upper(),
            json=json,
            headers=headers,
            response_type="json",
            retry_config=self._retry_config,
        )
