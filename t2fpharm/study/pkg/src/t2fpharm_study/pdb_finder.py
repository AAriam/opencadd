"""PDB structure finder based on UniProt ID."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import rcsbapi.data
import rcsbapi.search
import scipy.spatial
import pyserials
import sciapi
import scifile
import t2fpharm

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
    "OLA",  # oleic acid (from membrane)
    "OLB",  # oleic acid (from membrane)
    "OLC",  # 1-oleoyl-R-glycerol (from membrane)
    "PEG",
    "PGE",
    "PO4",
    "SO4",
    "TLA",
    "TRS",
    "WAT",
    "ZN",
)


class PDBFinder:
    def __init__(
        self,
        uniprot: str,
        *,
        artifact_ligand_ids: Sequence[str] = ARTIFACT_LIGAND_IDS,
        ligand_min_carbons: int = 6,
        ligand_min_unique_elements: int = 4,
        ligand_min_dist_threshold: float = 4,
        ligand_max_dist_threshold: float = 7,
        min_pocket_residues: int = 5,
        min_pocket_residue_prevalence: float = 0.8,
        resolution_range: tuple[float, float] = (0.48, 70),
        score_weight_coverage: float = 1,
        score_weight_modification: float = 1,
        score_weight_mutation: float = 1,
        score_weight_outlier: float = 1,
        score_weight_resolution: float = 1,
    ):
        self._uniprot = uniprot
        self._artifact_ligand_ids = set(artifact_ligand_ids)
        self._ligand_min_carbons = ligand_min_carbons
        self._ligand_min_unique_elements = ligand_min_unique_elements
        self._ligand_min_dist_threshold = ligand_min_dist_threshold
        self._ligand_max_dist_threshold = ligand_max_dist_threshold
        self._min_pocket_residues = min_pocket_residues
        self._min_pocket_residue_prevalence = min_pocket_residue_prevalence
        self._resolution_range = resolution_range
        self._score_weight = {
            "score_coverage": score_weight_coverage,
            "score_modification": score_weight_modification,
            "score_mutation": score_weight_mutation,
            "score_outlier": score_weight_outlier,
            "score_resolution": score_weight_resolution,
        }

        self._pdbe = sciapi.pdbe()

        self._scores = None
        self._sites = None
        self._sequence_length = None
        self._site_residues = None
        self._selected_site_residues = None
        self._residue_weights = None
        self._modified_residues = None
        self._mutated_residues = None
        self._outlier_residues = None
        self._pdb_entry_details = None
        self._all_pdb_ids = None
        self._compatible_pdb_ids = None
        self._default_residue_weight = None
        self._best_entry = None
        self._similar_entries = None
        self._residue_map: dict[str, pd.DataFrame] = {}
        self._pdb_str: dict[str, str] = {}
        return

    @property
    def best_entry(self) -> tuple[str, str, str]:
        """Best PDB ID based on the total score."""
        if self._best_entry is not None:
            return self._best_entry
        best = self.scores.iloc[0]
        self._best_entry = {
            "pdb_id": best["pdb_id"],
            "struct_asym_id": best["struct_asym_id"],
            "chain_id": best["chain_id"],
            "ligand_chain_id": best["ligand_chain_id"],
            "ligand_id": best["ligand_id"],
            "ligand_res_num": int(best["ligand_res_num"]),
        }
        return self._best_entry

    @property
    def similar_entries(self):
        if self._similar_entries is not None:
            return self._similar_entries
        best_entry = self.best_entry
        best_pdb_id = best_entry["pdb_id"]

        resmap = self.residue_map(best_pdb_id)
        site_residues = resmap[
            resmap["unp_residue_number"].isin(self.selected_site_residues)
        ].sort_values("weight", ascending=False).reset_index(drop=True).convert_dtypes()

        residue_list = []
        for _, residue in site_residues.iterrows():
            if len(residue_list) == 10:
                # PDB structure motif API has a limit of 10 residues per query
                break
            res_num = residue["pdb_residue_number"]
            if pd.isna(res_num):
                continue
            res = rcsbapi.search.StructMotifResidue(
                chain_id=residue["struct_asym_id"],
                label_seq_id=res_num,
                struct_oper_id="1",
            )
            residue_list.append(res)
        allowed_pdb_ids = self.scores[self.scores["ligand_score"]]["pdb_id"].unique().tolist()
        allowed_pdb_ids.remove(best_pdb_id)
        query = rcsbapi.search.StructMotifQuery(
            entry_id=best_pdb_id,
            residue_ids=residue_list,
            backbone_distance_tolerance=1,
            side_chain_distance_tolerance=1,
            angle_tolerance=1,
            rmsd_cutoff=2,
            atom_pairing_scheme="ALL",
            motif_pruning_strategy="KRUSKAL",
            allowed_structures=allowed_pdb_ids,
        )
        results = query(return_type="assembly", results_verbosity="verbose")
        results_flat = []
        for result in results:
            pdb_id, assembly_id = result["identifier"].split(".")
            score = result["score"]
            for service in result["services"]:
                service_type = service["service_type"]
                for node in nodes:
                    node_id = node["node_id"]
                    original_score = node["original_score"]
                    norm_score = node["norm_score"]
                    for match in node["match_context"]:
                        match_score = match["score"]
                        transformation = match["transformation"]
                        residue_types = match["residue_types"]
                        label_asym_ids = [elem["label_asym_id"] for elem in match["residue_ids"]]
                        assert all([label_asym_ids[0] == laid for laid in label_asym_ids[1:]])
                        assert score == norm_score
                        results_flat.append({
                            "pdb_id": pdb_id,
                            "struct_asym_id": label_asym_ids[0],
                            "score_match": match_score,
                        })
        df = pd.DataFrame(results_flat)
        df = (
            df.sort_values("score_match", ascending=False)
            .drop_duplicates(["pdb_id", "struct_asym_id"])
        )
        df = df.merge(
            self.scores,
            on=["pdb_id", "struct_asym_id"],
            how="left",
            validate="1:1"
        )
        df = df[(df["score_total"].notna()) & (df["score_ligand"] == True)]
        df = (
            df.sort_values(["score_total", "score_match"], ascending=[False, False])
            .groupby("pdb_id", as_index=False)
            .first()
        )
        if df.empty:
            raise ValueError("No similar entries found for the best PDB ID.")
        self._similar_entries = df
        return self._similar_entries

    @property
    def scores(self) -> pd.DataFrame:
        """Quality scores for PDB entries associated with the UniProt ID."""
        if self._scores is not None:
            return self._scores
        scores = pd.concat(
            [
                df.set_index(["pdb_id", "chain_id"])
                for df in (
                    self._score_coverage(),
                    self._score_modification(),
                    self._score_mutation(),
                    self._score_outlier(),
                    self._score_resolution(),
                    self._score_ligand(),
                )
            ],
            axis=1
        ).reset_index()
        scores["score_total"] = (
            sum(scores[col] * self._score_weight[col] for col in self._score_weight) / sum(self._score_weight.values())
        )
        scores = scores.sort_values(
            ["score_ligand", "score_total", "pdb_id", "struct_asym_id"],
            ascending=[False, False, True, True],
        ).reset_index(drop=True)
        cols = [
            "pdb_id",
            "struct_asym_id",
            "chain_id",
            "ligand_chain_id",
            "ligand_id",
            "ligand_res_num",
            "score_total",
            "score_coverage",
            "score_modification",
            "score_mutation",
            "score_outlier",
            "score_resolution",
            "score_ligand",
            "ligand_dist_min",
            "ligand_dist_mean",
            "ligand_dist_max",
        ]
        self._scores = scores[cols]
        return self._scores

    @property
    def sites(self) -> pd.DataFrame:
        """Binding site information."""
        def filter_ligands(df: pd.DataFrame) -> pd.Index:
            """Get cc_id values whose groups meet element-count criteria.

            This function groups rows by ``cc_id``
            and returns the unique ``cc_id`` values for which:
            1. there are at least 5 rows where ``element == "C"``, and
            2. there are more than 3 unique ``element`` values in that group.

            Parameters
            ----------
            df
                Input dataframe with at least the columns ``'cc_id'`` and ``'element'``.
                ``'cc_id'`` may be any hashable dtype (e.g., int, str, category).
                ``'element'`` is compared as a string to the literal ``"C"`` (case-sensitive).

            Returns
            -------
            Subset of available ``cc_id`` values that satisfy both criteria.
            """
            carbon_count = df["element"].eq("C").groupby(df["cc_id"], sort=False).sum()
            unique_element_count = df.groupby("cc_id", sort=False)["element"].nunique(dropna=True)
            mask = carbon_count.ge(self._ligand_min_carbons) & unique_element_count.ge(self._ligand_min_unique_elements)
            return set(mask.index[mask])

        if self._sites is not None:
            return self._sites
        column_name_map = {
            "startIndex": "res_start_num",
            "endIndex": "res_end_num",
            "accession": "lig_id",
            "numAtoms": "lig_atom_count",
            "pdbId": "pdb_id",
            "chainIds": "chain_id",
        }
        sites = self._pdbe.ligand_sites(self._uniprot, explode=True)
        self._sequence_length = sites["length"]
        df = pd.DataFrame(sites["data"])
        df.rename(columns=column_name_map, inplace=True)
        df = df[
            (df["indexType"] == "UNIPROT") &
            (~df["lig_id"].isin(self._artifact_ligand_ids)) &
            (df["lig_atom_count"] >= self._ligand_min_carbons)
        ]
        lig_atoms_rows = self._pdbe.compound_atoms(df["lig_id"].unique(), explode=True)
        lig_atoms = pd.DataFrame(lig_atoms_rows)
        acceptable_ligand_ids = filter_ligands(lig_atoms)
        df = df[df["lig_id"].isin(acceptable_ligand_ids)]
        if df.empty:
            raise ValueError("No binding sites found for the given UniProt ID with the specified criteria.")
        df["pdb_id"] = df["pdb_id"].str.upper()
        df["unp_residue_number"] = df.apply(lambda row: list(range(row["res_start_num"], row["res_end_num"]+1)), axis=1)
        df = df.drop(columns=["res_start_num", "res_end_num"]).explode("unp_residue_number")
        self._sites = df[
            ["unp_residue_number", "lig_id", "pdb_id", "chain_id"]
        ].sort_values("unp_residue_number").reset_index(drop=True)
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
        df = self.sites.value_counts("unp_residue_number").rename("prevalence").reset_index()
        df["prevalence"] = df["prevalence"] / self.sites.groupby(["lig_id", "pdb_id"]).ngroups
        df = df[["unp_residue_number", "prevalence"]]
        self._site_residues = df.sort_values("prevalence", ascending=False).reset_index(drop=True)
        return self._site_residues

    @property
    def modified_residues(self) -> pd.DataFrame:
        """Modified residues in compatible PDB entries."""
        if self._modified_residues is not None:
            return self._modified_residues
        modified_residues_rows = self._pdbe.pdb_modified_residues(self.compatible_pdb_ids, explode=True)
        self._modified_residues = pd.DataFrame(
            modified_residues_rows,
            columns=[
                "pdb_id",
                "chain_id",
                "author_residue_number",
                "author_insertion_code",
                "chem_comp_id",
                "alternate_conformers",
                "entity_id",
                "struct_asym_id",
                "residue_number",
                "chem_comp_name",
                "description",
                "weight",
            ]
        )
        return self._modified_residues

    @property
    def mutated_residues(self) -> pd.DataFrame:
        """Mutated residues in compatible PDB entries."""
        if self._mutated_residues is not None:
            return self._mutated_residues
        mutated_residues_rows = self._pdbe.pdb_mutated_residues(self.compatible_pdb_ids, explode=True)
        self._mutated_residues = pd.DataFrame(
            mutated_residues_rows,
            columns=[
                "pdb_id",
                "entity_id",
                "chain_id",
                "author_residue_number",
                "author_insertion_code",
                "chem_comp_id",
                "struct_asym_id",
                "residue_number",
                "mutation_from",
                "mutation_to",
                "mutation_type",
            ]
        )
        return self._mutated_residues

    @property
    def outlier_residues(self) -> pd.DataFrame:
        """Outlier residues in compatible PDB entries."""
        if self._outlier_residues is not None:
            return self._outlier_residues
        outlier_rows = self._pdbe.validation_residuewise_outlier_summary(self.compatible_pdb_ids, explode=True)
        df = pd.DataFrame(outlier_rows)
        required_columns = [
            'pdb_id', 'entity_id', 'chain_id', 'struct_asym_id', 'model_id',
            'residue_number', 'author_residue_number', 'author_insertion_code',
            'alt_code', 'outlier_type',
        ]
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")
        has_single_model = df.groupby("pdb_id")["model_id"].nunique() == 1
        if not has_single_model.all():
            raise ValueError(f"Multiple models found for PDB IDs: {has_single_model[~has_single_model].index.tolist()}")

        keys = [c for c in required_columns if c != 'outlier_type']
        df = (
            df.groupby(keys, dropna=False, sort=False)
            .size()
            .rename('outlier_type_ratio')
            .reset_index()
        )
        df['outlier_type_ratio'] = df['outlier_type_ratio'] / 11  # Normalize by total number of possible outlier types
        self._outlier_residues = df[keys + ['outlier_type_ratio']]
        return self._outlier_residues

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
    def selected_site_residues(self) -> np.ndarray:
        """UniProt residue numbers of selected residues in the binding sites."""
        if self._selected_site_residues is not None:
            return self._selected_site_residues
        len_by_prevalence = len(self.site_residues[self.site_residues["prevalence"] >= self._min_pocket_residue_prevalence])
        min_residues = max(self._min_pocket_residues, len_by_prevalence)
        self._selected_site_residues = self.site_residues[:min_residues]["unp_residue_number"].to_numpy()
        return self._selected_site_residues

    @property
    def compatible_pdb_ids(self) -> list[str]:
        """Subset of `self.all_pdb_ids` that are compatible with the PDB file format."""
        if self._compatible_pdb_ids is not None:
            return self._compatible_pdb_ids
        df = self.pdb_entry_details
        compatibles = df[
            df["pdbx_database_status.pdb_format_compatible"] &
            (df["rcsb_entry_info.deposited_model_count"] == 1) &
            (df["rcsb_entry_info.resolution_combined"].notna())
        ]
        if compatibles.empty:
            raise ValueError("No compatible PDB entries found with the specified criteria.")
        self._compatible_pdb_ids = compatibles["pdb_id"].tolist()
        return self._compatible_pdb_ids

    @property
    def pdb_entry_details(self) -> pd.DataFrame:
        """Details of compatible PDB entries."""
        if self._pdb_entry_details is not None:
            return self._pdb_entry_details
        data_attributes = [
            "pdbx_database_status.pdb_format_compatible",
            "rcsb_entry_info.deposited_model_count",
            "rcsb_entry_info.resolution_combined",
        ]
        query = rcsbapi.data.DataQuery(
            input_type="entries",
            input_ids=self.all_pdb_ids,
            return_data_list=list(data_attributes)
        )
        query.exec()
        results = query.get_response()
        df = pd.DataFrame(pyserials.flatten(results["data"]["entries"]))
        required_columns = set(data_attributes + ["rcsb_id"])
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"Bound entry details dataframe missing required columns: {sorted(missing_columns)}. "
                f"Required columns: {sorted(required_columns)}. "
                f"Available columns: {sorted(df.columns)}"
            )
        compatibility_flags = df["pdbx_database_status.pdb_format_compatible"].to_numpy(dtype=str, copy=False)
        flag_is_known = np.isin(compatibility_flags, ["Y", "N"])
        if not flag_is_known.all():
            raise ValueError(
                "Unexpected values in 'pdbx_database_status.pdb_format_compatible'. "
                f"Expected only 'Y' and 'N', got: {compatibility_flags[~flag_is_known].tolist()} "
                f"for PDB IDs: {df.loc[~flag_is_known, 'rcsb_id'].tolist()}"
            )
        df["pdbx_database_status.pdb_format_compatible"] = df["pdbx_database_status.pdb_format_compatible"] == "Y"
        # If multiple resolutions are present for an entry, take the mean.
        df['rcsb_entry_info.resolution_combined'] = df['rcsb_entry_info.resolution_combined'].apply(
            lambda v: float(np.mean(v)) if isinstance(v, (list, tuple, np.ndarray)) else v
        ).astype(float)
        df.rename(columns={"rcsb_id": "pdb_id"}, inplace=True)
        self._pdb_entry_details = df
        return self._pdb_entry_details

    def residue_map(self, pdb_id: str) -> pd.DataFrame:
        """Get the SIFTS residue mapping for a given PDB ID."""
        pdb_id = pdb_id.upper()
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
        residue_map = residue_map[
            ["entity_id", "chain_id", "struct_asym_id", "unp_residue_number", "pdb_residue_number"]
        ]
        residue_map = residue_map.merge(
            self.residue_weights,
            on="unp_residue_number",
            how="left",
            validate="m:1",
        )
        # Some proteins (like UniProt P00734) have chains with different residues (e.g., short and long variants).
        # In such cases, we need to select the chain that contains the most pocket residues.
        pocket_cov_by_chain = residue_map.groupby("chain_id")["weight"].sum()
        max_cov_mask = (pocket_cov_by_chain - pocket_cov_by_chain.max()).abs() <= 1e-6  # Allow small numerical errors
        max_cov_chain_ids = set(pocket_cov_by_chain.index[max_cov_mask])
        residue_map = residue_map[residue_map["chain_id"].isin(max_cov_chain_ids)]

        self._residue_map[pdb_id] = residue_map.convert_dtypes()
        return residue_map

    def _score_coverage(self) -> pd.DataFrame:
        """Calculate coverage scores for all compatible PDB entries."""
        rows = []
        for pdb_id in tqdm(self.compatible_pdb_ids, desc="Calculating coverage scores", unit="entry"):
            chain_scores = self._calculate_coverage_score(pdb_id=pdb_id)
            rows.extend(chain_scores)
        return pd.DataFrame(rows)

    def _score_modification(self) -> pd.DataFrame:
        """Calculate residue modification scores for all compatible PDB entries."""
        rows = []
        for pdb_id in tqdm(self.compatible_pdb_ids, desc="Calculating modification scores", unit="entry"):
            chain_scores = self._calculate_modification_score(pdb_id=pdb_id, mutation=False)
            rows.extend(chain_scores)
        return pd.DataFrame(rows)

    def _score_mutation(self) -> pd.DataFrame:
        """Calculate residue mutation scores for all compatible PDB entries."""
        rows = []
        for pdb_id in tqdm(self.compatible_pdb_ids, desc="Calculating mutation scores", unit="entry"):
            chain_scores = self._calculate_modification_score(pdb_id=pdb_id, mutation=True)
            rows.extend(chain_scores)
        return pd.DataFrame(rows)

    def _score_outlier(self) -> pd.DataFrame:
        """Calculate residue outlier scores for all compatible PDB entries."""
        rows = []
        for pdb_id in tqdm(self.compatible_pdb_ids, desc="Calculating outlier scores", unit="entry"):
            chain_scores = self._calculate_outlier_score(pdb_id=pdb_id)
            rows.extend(chain_scores)
        return pd.DataFrame(rows)

    def _score_resolution(self) -> pd.DataFrame:
        """Calculate resolution scores for all compatible PDB entries."""
        rows = []
        for pdb_id in tqdm(self.compatible_pdb_ids, desc="Calculating resolution scores", unit="entry"):
            chain_scores = self._calculate_resolution_score(pdb_id=pdb_id)
            rows.extend(chain_scores)
        return pd.DataFrame(rows)

    def _score_ligand(self) -> pd.DataFrame:
        """Calculate ligand scores for all compatible PDB entries."""
        rows = []
        for pdb_id in tqdm(self.compatible_pdb_ids, desc="Calculating ligand scores", unit="entry"):
            try:
                chain_scores = self._calculate_ligand_score(pdb_id=pdb_id)
            except Exception as e:
                raise RuntimeError(f"Error calculating ligand score for PDB ID {pdb_id}: {e}") from e
            rows.extend(chain_scores)
        df = pd.DataFrame(rows)
        df["score_ligand"] = (
            (df["ligand_dist_min"] <= self._ligand_min_dist_threshold) &
            (df["ligand_dist_max"] <= self._ligand_max_dist_threshold)
        )
        return df

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
            residue_map[["chain_id", "pdb_residue_number", "unp_residue_number", "weight"]],
            on=["chain_id", "pdb_residue_number"],
            how="left",
            validate="m:m",  # allow potential duplication; see Notes
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
        listing = residue_map.drop(columns="key")

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

    def _calculate_outlier_score(self, pdb_id: str) -> list[dict[str, float | None]]:
        def _safe_divide(n: float, d: float) -> float | None:
            return (n / d) if d and np.isfinite(n) and np.isfinite(d) else None

        residue_map = self.residue_map(pdb_id=pdb_id).copy()

        outlier_residues = self.outlier_residues[self.outlier_residues["pdb_id"] == pdb_id].copy()
        outlier_residues = (
            outlier_residues[["chain_id", "residue_number", "outlier_type_ratio"]]
            .rename(columns={"residue_number": "pdb_residue_number"})
        )

        merged = residue_map.merge(
            outlier_residues,
            on=["chain_id", "pdb_residue_number"],
            how="left",
            copy=False
        )
        merged["nonoutlier_ratio"] = 1.0 - merged["outlier_type_ratio"].fillna(0.0)
        listing = merged.drop(columns=["outlier_type_ratio"])

        # Replace missing weights with 0
        listing["weight"] = pd.to_numeric(listing["weight"], errors="coerce")
        listing.loc[~np.isfinite(listing["weight"]), "weight"] = 0

        # If there is no data left, return all chains with None
        unique_chains = sorted(pd.unique(residue_map["chain_id"]))
        if listing.empty:
            return [{"chain_id": cid, f"score_outlier": None} for cid in unique_chains]

        # Compute weighted averages per chain
        listing["wx"] = listing["weight"] * listing["nonoutlier_ratio"]
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
            result.append({"pdb_id": pdb_id, "chain_id": cid, f"score_outlier": score})
        return result

    def _calculate_resolution_score(self, pdb_id: str) -> list[dict[str, float | None]]:
        def score(res):
            lower, upper = self._resolution_range
            if res < lower:
                return 1.0
            if res > upper:
                return 0.0
            return (upper - res) / (upper - lower)
        entry_df = self.pdb_entry_details
        resolution = entry_df[entry_df["pdb_id"] == pdb_id]["rcsb_entry_info.resolution_combined"].iloc[0]
        return [
            {
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "score_resolution": score(resolution),
            }
            for chain_id in self.residue_map(pdb_id=pdb_id)["chain_id"].unique()
        ]

    def _calculate_ligand_score(self, pdb_id: str) -> tuple[tuple[str, str, int], np.ndarray]:
        """Return lig_id of the (lig_id, chain_id) group with most range hits.

        A row is a "hit" if at least one integer from `common_nums` lies within
        the inclusive interval [res_start_num, res_end_num]. Rows are grouped by
        (lig_id, chain_id); we count hits per group and return the `lig_id` of the
        group with the highest count. In case of ties, the first occurring group
        in Pandas' order is chosen. If `sites` is empty, return None.

        Parameters
        ----------
        sites
            DataFrame with columns:
            - 'res_start_num' (int): inclusive start of the residue range.
            - 'res_end_num' (int): inclusive end of the residue range.
            - 'lig_id': ligand identifier (any hashable/label-like).
            - 'chain_id': chain identifier (any hashable/label-like).
        common_nums
            Iterable of integers to test for membership within row intervals.
            Duplicates are ignored; order does not matter.

        Returns
        -------
        ligand_chain_id, ligand_id, ligand_res_num
            The `lig_id` of the (lig_id, chain_id) group with the largest number
            of hit rows, or None if `sites` has no rows.

        Notes
        -----
        Implementation uses a vectorized binary search:
        For each row, find the first `common_num` >= start; if it exists and is
        <= end, the row is a hit. Complexity ~ O(N log M), with N rows and
        M unique `common_nums`.
        """
        def pdb_atoms(pdb_id: str) -> pd.DataFrame:
            """Fetch the PDB atom content for a given PDB ID."""
            pdb_id = pdb_id.upper()
            mmcif_str = sciapi.pdb.file.entry(pdb_id, file_format="cif").decode("utf-8")
            cif = MMCIF2Dict(io.StringIO(mmcif_str))
            df = pd.DataFrame(
                {
                    "entity_id": cif["_atom_site.label_entity_id"],
                    "struct_asym_id": cif["_atom_site.label_asym_id"],
                    "chain_id": cif["_atom_site.auth_asym_id"],
                    "res_name": cif["_atom_site.label_comp_id"],
                    "author_res_name": cif["_atom_site.auth_comp_id"],
                    "pdb_residue_number": cif["_atom_site.label_seq_id"],
                    "pdb_author_residue_number": cif["_atom_site.auth_seq_id"],
                    "x": cif["_atom_site.Cartn_x"],
                    "y": cif["_atom_site.Cartn_y"],
                    "z": cif["_atom_site.Cartn_z"],
                }
            )
            df["entity_id"] = df["entity_id"].astype(int)
            df.loc[df["pdb_residue_number"] == ".", "pdb_residue_number"] = -1
            df["pdb_residue_number"] = df["pdb_residue_number"].astype(int)
            df["pdb_author_residue_number"] = df["pdb_author_residue_number"].astype(int)
            df["x"] = df["x"].astype(float)
            df["y"] = df["y"].astype(float)
            df["z"] = df["z"].astype(float)
            return df.convert_dtypes()
        sites = self.sites[self.sites["pdb_id"] == pdb_id]
        unique_lig_ids = sites["lig_id"].unique()
        atom = pdb_atoms(pdb_id)
        resmap = self.residue_map(pdb_id)
        site_res = resmap[resmap["unp_residue_number"].isin(self.selected_site_residues)]
        site_atoms = site_res.merge(
            atom,
            on=["entity_id", "chain_id", "struct_asym_id", "pdb_residue_number"],
            how="left"
        )
        site_atoms = site_atoms[site_atoms[["x", "y", "z"]].notna().all(axis=1)]

        lig_atom = atom[atom["res_name"].isin(unique_lig_ids)]
        lig_groups = lig_atom.groupby(["chain_id", "author_res_name", "pdb_author_residue_number"], sort=False)

        result: list[dict] = []
        for chain_id, site_df in site_atoms.groupby("chain_id", sort=False):
            struct_asym_id = site_df["struct_asym_id"].iloc[0]

            best_key: tuple[str, str, int] | None = None
            best_dist_mean = np.inf
            best_dist_min = None
            best_dist_max = None
            for key, lig_df in lig_groups:
                lig_atom_coords = lig_df.loc[:, ["x", "y", "z"]].to_numpy(dtype=float, copy=False)
                lig_atom_kdtree = scipy.spatial.KDTree(lig_atom_coords)
                dists = []
                for _, res in site_df.groupby("pdb_residue_number"):
                    res_coords = res[["x","y","z"]].to_numpy(dtype=float)
                    res_dists, _ = lig_atom_kdtree.query(res_coords, k=1)
                    dists.append(np.min(res_dists))
                dists = np.array(dists, dtype=float)
                mean = np.mean(dists)
                if mean < best_dist_mean:
                    best_key = key
                    best_dist_mean = mean
                    best_dist_min = np.min(dists)
                    best_dist_max = np.max(dists)

            if best_key is None or not np.isfinite(best_dist_mean):
                raise ValueError("No valid groups with finite coordinates were found.")
            result.append({
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "struct_asym_id": struct_asym_id,
                "ligand_chain_id": best_key[0],
                "ligand_id": best_key[1],
                "ligand_res_num": int(best_key[2]),
                "ligand_dist_min": best_dist_min,
                "ligand_dist_mean": best_dist_mean,
                "ligand_dist_max": best_dist_max,
            })
        return result
