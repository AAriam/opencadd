from __future__ import annotations

from typing import TYPE_CHECKING
import sciapi
import numpy as np
import pandas as pd
import rcsbapi.data

if TYPE_CHECKING:
    from typing import Sequence


ARTIFACT_LIGAND_IDS = (
    "ACE",
    "ACT",
    "BME",
    "BR",
    "CA",
    "CAC",
    "CL",
    "DMF",
    "DMS",
    "DOD",
    "EDO",
    "FMT",
    "GOL",
    "HEP",
    "HOH",
    "IOD",
    "IOD",
    "IPA",
    "K",
    "MES",
    "MG",
    "MPD",
    "NA",
    "NH4",
    "NO3",
    "PEG",
    "PGE",
    "PO4",
    "SO4",
    "TLA",
    "TRS",
    "WAT",
    "ZN",
)


class StructureFinder:
    def __init__(
        self,
        uniprot: str,
        *,
        artifact_ligand_ids: Sequence[str] = ARTIFACT_LIGAND_IDS,
        minimum_ligand_atoms: int = 10,
        score_weight_coverage: float = 1,
        score_weight_modification: float = 1,
        score_weight_mutation: float = 1,
    ):
        self._uniprot = uniprot
        self._artifact_ligand_ids = set(artifact_ligand_ids)
        self._minimum_ligand_atoms = minimum_ligand_atoms

        self._pdbe = sciapi.pdbe()

        self._scores = None
        self._sites = None
        self._sequence_length = None
        self._site_residues = None
        self._residue_weights = None
        self._modified_residues = None
        self._mutated_residues = None
        self._all_pdb_ids = None
        self._compatible_pdb_ids = None
        self._default_residue_weight = None
        self._residue_map: dict[str, pd.DataFrame] = {}
        self._score_weight = {
            "score_coverage": score_weight_coverage,
            "score_modification": score_weight_modification,
            "score_mutation": score_weight_mutation,
        }
        return

    @property
    def scores(self) -> pd.DataFrame:
        """Quality scores for PDB entries associated with the UniProt ID."""
        if self._scores is not None:
            return self._scores
        scores = pd.concat(
            [
                df.set_index(["pdb_id", "chain_id"])
                for df in (self._score_coverage(), self._score_modification(), self._score_mutation())
            ],
            axis=1
        ).reset_index()
        scores["score_total"] = (
            sum(scores[col] * self._score_weight[col] for col in self._score_weight) / sum(self._score_weight.values())
        )
        scores = scores.sort_values(
            ["score_total", "pdb_id", "chain_id"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
        self._scores = scores
        return self._scores

    @property
    def sites(self) -> pd.DataFrame:
        """Binding site information."""
        if self._sites is not None:
            return self._sites
        column_name_map = {
            "startCode": "res_start_name",
            "startIndex": "res_start_num",
            "endCode": "res_end_name",
            "endIndex": "res_end_num",
            "accession": "lig_id",
            "numAtoms": "lig_atom_count",
            "pdbId": "pdb_id",
            "entityId": "entity_id",
            "chainIds": "chain_id",
            "indexType": "res_num_db",
        }
        sites = self._pdbe.ligand_sites(self._uniprot, explode=True)
        self._sequence_length = sites["length"]
        df = pd.DataFrame(sites["data"])
        df.rename(columns=column_name_map, inplace=True)
        df = df[
            (df["res_num_db"] == "UNIPROT") &
            (~df["lig_id"].isin(self._artifact_ligand_ids)) &
            (df["lig_atom_count"] >= self._minimum_ligand_atoms)
        ]
        if df.empty:
            raise ValueError("No binding sites found for the given UniProt ID with the specified criteria.")
        self._sites = df[list(column_name_map.values())].sort_values("res_start_num").reset_index(drop=True)
        return self._sites

    @property
    def sequence_length(self) -> int:
        if self._sequence_length is not None:
            return self._sequence_length
        self.sites  # Trigger sites loading to set sequence_length
        return self._sequence_length

    @property
    def site_residues(self) -> pd.DataFrame:
        """Residue numbers and their prevalence in the binding sites.

        Returns
        -------
        DataFrame with columns:
        - `unp_residue_number`: UniProt residue number in the binding site.
        - `prevalence`: Prevalence of the residue across all binding sites.
          as a fraction of the total number of sites.
        """
        if self._site_residues is not None:
            return self._site_residues
        site_counts = self.sites.value_counts(["res_start_num", "res_end_num"]).rename("prevalence").reset_index()
        site_counts["prevalence"] = site_counts["prevalence"] / self.sites.groupby(["lig_id", "pdb_id"]).ngroups
        lengths = (site_counts["res_end_num"] - site_counts["res_start_num"] + 1).astype("int64")
        expanded = site_counts.loc[site_counts.index.repeat(lengths)].copy()
        expanded["unp_residue_number"] = expanded["res_start_num"] + expanded.groupby(level=0).cumcount()
        residue_df = expanded[["unp_residue_number", "prevalence"]]
        self._site_residues = residue_df.sort_values("prevalence", ascending=False).reset_index(drop=True)
        return self._site_residues

    @property
    def modified_residues(self) -> pd.DataFrame:
        """Modified residues in compatible PDB entries."""
        if self._modified_residues is not None:
            return self._modified_residues
        modified_residues_rows = self._pdbe.pdb_modified_residues(self.compatible_pdb_ids, explode=True)
        self._modified_residues = pd.DataFrame(modified_residues_rows)
        return self._modified_residues

    @property
    def mutated_residues(self) -> pd.DataFrame:
        """Mutated residues in compatible PDB entries."""
        if self._mutated_residues is not None:
            return self._mutated_residues
        mutated_residues_rows = self._pdbe.pdb_mutated_residues(self.compatible_pdb_ids, explode=True)
        self._mutated_residues = pd.DataFrame(mutated_residues_rows)
        return self._mutated_residues

    @property
    def residue_weights(self) -> pd.DataFrame:
        """Weights for each residue in the sequence based on binding site prevalence.

        The weights are directly taken from `site_residues` where available.
        For other residues, the weight is determined by the closest residue(s)
        present in `site_residues`. In case of equal distances, the site with the
        highest prevalence is selected.

        Parameters
        ----------
        site_residues : pd.DataFrame
            Must contain columns:
            - 'unp_residue_number' (int): residue indices (1-based).
            - 'prevalence' (float): prevalence value for that residue.
        sequence_length : int
            Total length of the sequence. Residues are assumed numbered [1, sequence_length].

        Returns
        -------
        pd.DataFrame
            A dataframe with:
            - 'unp_residue_number': all integers from 1 to `sequence_length`.
            - 'weight': float weight assigned per the described rule.

        Raises
        ------
        ValueError
            If `site_residues` is empty or missing required columns.

        Notes
        -----
        For residues not in `site_residues`, the closest residue by index
        is used. If multiple are equally close, the one with the largest
        prevalence is chosen. The weight is then computed as:

            prevalence / distance

        where `distance` is the absolute difference in residue numbers.
        """
        if self._residue_weights is not None:
            return self._residue_weights
        # Clean up input
        site_residues = self.site_residues.copy()
        site_residues = site_residues.sort_values("unp_residue_number").reset_index(drop=True)

        residues = np.arange(1, self.sequence_length + 1)
        site_positions = site_residues["unp_residue_number"].to_numpy()
        site_prevalences = site_residues["prevalence"].to_numpy()

        weights = []
        for r in residues:
            if r in site_positions:
                # Direct prevalence
                weight = site_prevalences[np.where(site_positions == r)[0][0]]
            else:
                distances = np.abs(site_positions - r)
                min_dist = np.min(distances)
                nearest_idxs = np.where(distances == min_dist)[0]
                # Resolve ties by highest prevalence
                best_idx = nearest_idxs[np.argmax(site_prevalences[nearest_idxs])]
                weight = site_prevalences[best_idx] / min_dist
            weights.append(weight)
        self._residue_weights = pd.DataFrame({"unp_residue_number": residues, "weight": weights})
        return self._residue_weights

    @property
    def all_pdb_ids(self) -> list[str]:
        """List of all PDB IDs associated with the UniProt ID, and with a bound ligand."""
        if self._all_pdb_ids is not None:
            return self._all_pdb_ids
        self._all_pdb_ids = np.strings.upper(self.sites["pdb_id"].unique().astype(str)).tolist()
        return self._all_pdb_ids

    @property
    def compatible_pdb_ids(self) -> list[str]:
        """Subset of `self.all_pdb_ids` that are compatible with the PDB file format."""
        if self._compatible_pdb_ids is not None:
            return self._compatible_pdb_ids
        query = rcsbapi.data.DataQuery(
            input_type="entries",
            input_ids=self.all_pdb_ids,
            return_data_list=["pdbx_database_status.pdb_format_compatible"]
        )
        query.exec()
        results = query.get_response()
        self._compatible_pdb_ids = []
        for entry in results["data"]["entries"]:
            compatibility_flag = entry["pdbx_database_status"]["pdb_format_compatible"]
            if compatibility_flag not in ("Y", "N"):
                raise ValueError(
                    f"Unexpected compatibility flag '{compatibility_flag}' for PDB ID {entry['rcsb_id']}. "
                    "Expected 'Y' or 'N'."
                )
            if compatibility_flag == "Y":
                self._compatible_pdb_ids.append(entry["rcsb_id"])
        return self._compatible_pdb_ids

    def residue_map(self, pdb_id: str) -> pd.DataFrame:
        """Get the SIFTS residue mapping for a given PDB ID."""
        residue_map = self._residue_map.get(pdb_id)
        if residue_map is not None:
            return residue_map
        residue_map_rows = self._pdbe.sifts_pdb_uniprot(pdb_id=pdb_id, explode=True, expand=True)
        residue_map = pd.DataFrame(residue_map_rows)
        required_columns = {"accession", "chain_id", "unp_residue_number", "pdb_residue_number"}
        missing_columns = required_columns - set(residue_map.columns)
        if missing_columns:
            raise ValueError(f"Residue map dataframe missing required columns: {sorted(missing_columns)}")
        residue_map = residue_map[residue_map["accession"] == self._uniprot]
        self._residue_map[pdb_id] = residue_map
        return residue_map

    def _score_coverage(self) -> pd.DataFrame:
        """Calculate coverage scores for all compatible PDB entries."""
        rows = []
        for pdb_id in self.compatible_pdb_ids:
            chain_scores = self._calculate_coverage_score(pdb_id=pdb_id)
            rows.extend(chain_scores)
        return pd.DataFrame(rows)

    def _score_modification(self) -> pd.DataFrame:
        """Calculate residue modification scores for all compatible PDB entries."""
        rows = []
        for pdb_id in self.compatible_pdb_ids:
            chain_scores = self._calculate_modification_score(pdb_id=pdb_id, mutation=False)
            rows.extend(chain_scores)
        return pd.DataFrame(rows)

    def _score_mutation(self) -> pd.DataFrame:
        """Calculate residue mutation scores for all compatible PDB entries."""
        rows = []
        for pdb_id in self.compatible_pdb_ids:
            chain_scores = self._calculate_modification_score(pdb_id=pdb_id, mutation=True)
            rows.extend(chain_scores)
        return pd.DataFrame(rows)

    def _calculate_coverage_score(self, pdb_id: str) -> list[dict[str, float | None]]:
        """Compute chain-wise coverage score as weighted average of observed residue ratios.

        For each unique `chain_id` in `self.residue_map(pdb_id)`,
        this method computes the weighted average of observed residue ratios in that chain.
        Weights are determined by mapping each listing row's `(chain_id, pdb_residue_number)` to a
        `unp_residue_number` via `residue_map`, then looking up `prevalence` for that
        `unp_residue_number` in `site_residues`. If no prevalence is available for a row,
        use half of the minimum prevalence found in `site_residues`.

        Parameters
        ----------
        pdb_id
            PDB ID of the structure being analyzed.

        Returns
        -------
        list of dict
            One dict per unique `chain_id` found in `self.residue_map(pdb_id)`, each:
            `{"chain_id": <chain_id>, "completeness_score": <float|None>}`.
            If a chain yields no valid weighted observations or total weight is zero,
            `completeness_score` will be `None`.

        Notes
        ------
        - Deterministic order: results are sorted by `chain_id` (ascending).
        - If a `(chain_id, pdb_residue_number)` maps to multiple `unp_residue_number`s in
          `residue_map`, rows will be duplicated during the merge (each mapping contributes).
          If your data should be one-to-one, ensure `residue_map` uniqueness prior to calling.
        - Rows with non-finite `observed_ratio` (NaN/inf) are excluded from the average.
        """
        def _safe_divide(n: float, d: float) -> float | None:
            return (n / d) if d and np.isfinite(n) and np.isfinite(d) else None

        # Get residue listing containing observed ratios for each residue in the PDB structure
        listing_rows = self._pdbe.pdb_residue_listing(pdb_id=pdb_id, explode=True)
        listing = pd.DataFrame(listing_rows)
        required_columns = {"chain_id", "residue_number", "observed_ratio"}
        missing_columns = required_columns - set(listing.columns)
        if missing_columns:
            raise ValueError(f"Residue listing dataframe missing required columns: {sorted(missing_columns)}")
        listing.rename(
            columns={"residue_number": "pdb_residue_number"},
            inplace=True,
        )

        # Merge to attach unp_residue_number to the listing via (chain_id, pdb_residue_number)
        residue_map = self.residue_map(pdb_id=pdb_id)
        listing = listing.merge(
            residue_map[["chain_id", "pdb_residue_number", "unp_residue_number"]],
            on=["chain_id", "pdb_residue_number"],
            how="left",
            validate="m:m",  # allow potential duplication; see Notes
        )

        # Merge with weights
        listing = listing.merge(
            self.residue_weights[["unp_residue_number", "weight"]],
            on="unp_residue_number",
            how="left",
            validate="m:1",
        )

        # Replace missing weights with 0
        listing["weight"] = pd.to_numeric(listing["weight"], errors="coerce")
        listing.loc[~np.isfinite(listing["weight"]), "weight"] = 0

        # Exclude non-finite observed_ratio from contributing
        listing["observed_ratio"] = pd.to_numeric(listing["observed_ratio"], errors="coerce")
        listing = listing[np.isfinite(listing["observed_ratio"])]

        # If there is no data left, return all chains with None
        unique_chains = sorted(pd.unique(residue_map["chain_id"]))
        if listing.empty:
            return [{"chain_id": cid, "score_coverage": None} for cid in unique_chains]

        # Compute weighted averages per chain
        listing["wx"] = listing["weight"] * listing["observed_ratio"]
        grouped = listing.groupby("chain_id", dropna=False, sort=True).agg(
            wsum=("wx", "sum"),
            wtot=("weight", "sum"),
        )

        scores_map: dict[str, float | None] = {
            chain_id: _safe_divide(row["wsum"], row["wtot"]) for chain_id, row in grouped.iterrows()
        }

        # Ensure all chains that appear in residue_map are represented
        result: list[dict[str, float]] = []
        for cid in unique_chains:
            score = scores_map.get(cid, None)
            # Normalize to float (or None) if present
            if isinstance(score, (np.floating,)):
                score = float(score)  # type: ignore[assignment]
            result.append({"pdb_id": pdb_id, "chain_id": cid, "score_coverage": score})
        return result

    def _calculate_modification_score(self, pdb_id: str, mutation: bool) -> list[dict[str, float | None]]:
        def _safe_divide(n: float, d: float) -> float | None:
            return (n / d) if d and np.isfinite(n) and np.isfinite(d) else None

        mod_type = "mutation" if mutation else "modification"

        residue_map = self.residue_map(pdb_id=pdb_id).copy()
        modified_residues = self.mutated_residues if mutation else self.modified_residues
        modified_residues = modified_residues[modified_residues["pdb_id"] == pdb_id].copy()

        # Add `not_modified` column to residue_map indicating whether a residue is (not) modified
        residue_map["key"] = list(zip(residue_map["chain_id"], residue_map["pdb_residue_number"]))
        modified_residues["key"] = list(zip(modified_residues["chain_id"], modified_residues["residue_number"]))
        residue_map["not_modified"] = (~residue_map["key"].isin(modified_residues["key"])).astype(int)
        residue_map = residue_map.drop(columns="key")

        # Merge with weights
        listing = residue_map.merge(
            self.residue_weights[["unp_residue_number", "weight"]],
            on="unp_residue_number",
            how="left",
            validate="m:1",
        )

        # Replace missing weights with 0
        listing["weight"] = pd.to_numeric(listing["weight"], errors="coerce")
        listing.loc[~np.isfinite(listing["weight"]), "weight"] = 0

        # If there is no data left, return all chains with None
        unique_chains = sorted(pd.unique(residue_map["chain_id"]))
        if listing.empty:
            return [{"chain_id": cid, f"score_{mod_type}": None} for cid in unique_chains]

        # Compute weighted averages per chain
        listing["wx"] = listing["weight"] * listing["not_modified"]
        grouped = listing.groupby("chain_id", dropna=False, sort=True).agg(
            wsum=("wx", "sum"),
            wtot=("weight", "sum"),
        )

        scores_map: dict[str, float | None] = {
            chain_id: _safe_divide(row["wsum"], row["wtot"]) for chain_id, row in grouped.iterrows()
        }

        # Ensure all chains that appear in residue_map are represented
        result: list[dict[str, float]] = []
        for cid in unique_chains:
            score = scores_map.get(cid, None)
            # Normalize to float (or None) if present
            if isinstance(score, (np.floating,)):
                score = float(score)  # type: ignore[assignment]
            result.append({"pdb_id": pdb_id, "chain_id": cid, f"score_{mod_type}": score})
        return result
