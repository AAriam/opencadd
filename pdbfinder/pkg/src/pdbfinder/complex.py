"""PDB complex structure finder based on UniProt ID."""

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


class ComplexFinder:
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
        if min_pocket_residues < 3:
            raise ValueError("min_pocket_residues must be at least 3.")
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
        best_struct_asym_id = best_entry["struct_asym_id"]

        resmap = self.residue_map(best_pdb_id)
        site_residues = resmap[
            (resmap["struct_asym_id"] == best_struct_asym_id) &
            (resmap["unp_residue_number"].isin(self.selected_site_residues)) &
            (resmap["pdb_residue_number"].notna())
        ].sort_values("weight", ascending=False).reset_index(drop=True).convert_dtypes()

        residue_list = []
        for _, residue in site_residues.iterrows():
            if len(residue_list) == 10:
                # PDB structure motif API has a limit of 10 residues per query
                break
            res = rcsbapi.search.StructMotifResidue(
                chain_id=best_struct_asym_id,
                label_seq_id=residue["pdb_residue_number"],
                struct_oper_id="1",
            )
            residue_list.append(res)
        allowed_pdb_ids = self.scores[
            (self.scores["score_ligand"] == True) &
            (self.scores["observed_site_residues"] >= 2)
        ]["pdb_id"].unique().tolist()
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
            pdb_id, assembly_id = result["identifier"].split("-")
            score = result["score"]
            for service in result["services"]:
                service_type = service["service_type"]
                for node in service["nodes"]:
                    node_id = node["node_id"]
                    original_score = node["original_score"]
                    norm_score = node["norm_score"]
                    for match in node["match_context"]:
                        match_score = match["score"]
                        transformation = match["transformation"]
                        residue_types = match["residue_types"]
                        label_asym_ids = [elem["label_asym_id"] for elem in match["residue_ids"]]
                        if not all([label_asym_ids[0] == laid for laid in label_asym_ids[1:]]):
                            continue
                        assert score == norm_score
                        results_flat.append({
                            "pdb_id": pdb_id,
                            "struct_asym_id": label_asym_ids[0],
                            "score_match": norm_score,
                        })
        df = pd.DataFrame(results_flat).drop_duplicates()
        # Different transformations may give different scores for the same chain; keep only the best score
        df = df.loc[
            df.groupby(["pdb_id", "struct_asym_id"])["score_match"].idxmax()
        ].reset_index(drop=True)

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

        rows_coverage = []
        rows_modification = []
        rows_mutation = []
        rows_outlier = []
        rows_resolution = []
        rows_ligand = []

        for pdb_id in tqdm(self.compatible_pdb_ids, desc="Calculating scores", unit="entry"):
            try:
                rows_coverage.extend(self._score_coverage(pdb_id))
                rows_modification.extend(self._score_modification(pdb_id, mutation=False))
                rows_mutation.extend(self._score_modification(pdb_id, mutation=True))
                rows_outlier.extend(self._score_outlier(pdb_id))
                rows_resolution.extend(self._score_resolution(pdb_id))
                rows_ligand.extend(self._score_ligand(pdb_id))
            except Exception as e:
                raise RuntimeError(f"Error calculating score for PDB ID {pdb_id}: {e}") from e

        scores = pd.concat(
            [
                df.set_index(["pdb_id", "struct_asym_id", "chain_id"])
                for df in (
                    pd.DataFrame(rows) for rows in (
                        rows_coverage,
                        rows_modification,
                        rows_mutation,
                        rows_outlier,
                        rows_resolution,
                        rows_ligand
                    )
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
            "observed_site_residues",
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
        self._selected_site_residues = np.sort(self.site_residues[:min_residues]["unp_residue_number"].to_numpy())
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
        df = self._residue_map.get(pdb_id)
        if df is not None:
            return df
        residue_map_rows = self._pdbe.sifts_pdb_uniprot(pdb_id=pdb_id, explode=True, expand=True)
        df = pd.DataFrame(residue_map_rows)
        required_columns = {"accession", "chain_id", "unp_residue_number", "pdb_residue_number"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Residue map dataframe missing required columns: {sorted(missing_columns)}")
        df = df[df["accession"] == self._uniprot]
        df = df[
            ["entity_id", "chain_id", "struct_asym_id", "unp_residue_number", "pdb_residue_number"]
        ]


        invariant_cols: list[str] = ["entity_id", "chain_id"]

        # Sanity check invariants: ensure per-chain constancy.
        varying: dict[str, Iterable[str]] = {}
        g = df.groupby("struct_asym_id", dropna=False)
        for col in invariant_cols:
            # Count distinct values per chain for the column.
            nunq = g[col].nunique(dropna=False)
            bad_ids = nunq[nunq > 1].index.tolist()
            if bad_ids:
                varying[col] = bad_ids
        if varying:
            details = "; ".join(
                f"{col} varies for chains {bad_ids}" for col, bad_ids in varying.items()
            )
            raise ValueError(
                "Invariant column(s) vary within struct_asym_id groups: " + details
            )

        # Deduplicate potential repeated (chain, unp) pairs.
        base = (
            df.sort_index()
            .drop_duplicates(subset=["struct_asym_id", "unp_residue_number"], keep="first")
        )

        # Per-chain invariant row (take first).
        per_chain = (
            base.groupby("struct_asym_id", dropna=False)[invariant_cols]
            .first()
            .reset_index()
        )

        # Full grid of (struct_asym_id, unp_residue_number) for 1..sequence_length.
        chains = per_chain["struct_asym_id"].unique()
        expanded = pd.MultiIndex.from_product(
            [chains, range(1, self.sequence_length + 1)],
            names=["struct_asym_id", "unp_residue_number"],
        ).to_frame(index=False)

        # Attach invariants to the grid (repeats per chain).
        expanded = expanded.merge(per_chain, on="struct_asym_id", how="left")

        # Bring over pdb_residue_number from original data.
        pdb_map = base[["struct_asym_id", "unp_residue_number", "pdb_residue_number"]]
        expanded = expanded.merge(
            pdb_map,
            on=["struct_asym_id", "unp_residue_number"],
            how="left",
        )

        # Enforce dtypes: unp as int (no NA), pdb as nullable Int64 (allows <NA>).
        expanded["unp_residue_number"] = expanded["unp_residue_number"].astype(int)
        # Cast last to preserve Int64 even if all present.
        expanded["pdb_residue_number"] = expanded["pdb_residue_number"].astype("Int64")

        # Reorder columns to match input order where possible.
        expanded = expanded[["entity_id", "chain_id", "struct_asym_id", "unp_residue_number", "pdb_residue_number"]]

        # Stable sort
        expanded = expanded.sort_values(
            by=["struct_asym_id", "unp_residue_number"],
            kind="mergesort",
            ignore_index=True,
        )

        residue_map = expanded.merge(
            self.residue_weights,
            on="unp_residue_number",
            how="left",
            validate="m:1",
        )
        self._residue_map[pdb_id] = residue_map.convert_dtypes()
        return residue_map

    def _score_coverage(self, pdb_id: str) -> list[dict[str, float | None]]:
        resmap = self.residue_map(pdb_id=pdb_id)

        # Get residue listing containing observed ratios for each available residue in the PDB structure
        listing_rows = self._pdbe.pdb_residue_listing(pdb_id=pdb_id, explode=True)
        listing = pd.DataFrame(listing_rows)
        required_columns = {"chain_id", "struct_asym_id", "residue_number", "observed_ratio"}
        missing_columns = required_columns - set(listing.columns)
        if missing_columns:
            raise ValueError(f"Residue listing dataframe missing required columns: {sorted(missing_columns)}")
        listing.rename(
            columns={"residue_number": "pdb_residue_number"},
            inplace=True,
        )

        merged = resmap.merge(
            listing,
            on=["entity_id", "chain_id", "struct_asym_id", "pdb_residue_number"],
            how="left",
        )
        # set observed_ratio to 0 for residues not available in the listing
        merged.fillna({"observed_ratio": 0}, inplace=True)

        merged["site_residue_is_observed"] = (
            merged["unp_residue_number"].isin(self.selected_site_residues) &
            merged["observed_ratio"].gt(0)
        ).astype(int)

        # Compute weighted averages per chain
        merged["wx"] = merged["weight"] * merged["observed_ratio"]
        grouped = merged.groupby(["struct_asym_id", "chain_id"], dropna=False, sort=True).agg(
            wsum=("wx", "sum"),
            wtot=("weight", "sum"),
            observed_site_residues=("site_residue_is_observed", "sum"),
        )

        result: list[dict[str, float]] = []
        for (struct_asym_id, chain_id), row in grouped.iterrows():
            result.append({
                "pdb_id": pdb_id,
                "struct_asym_id": struct_asym_id,
                "chain_id": chain_id,
                "score_coverage": float(row["wsum"] / row["wtot"]),
                "observed_site_residues": int(row["observed_site_residues"]),
            })
        return result

    def _score_modification(self, pdb_id: str, mutation: bool) -> list[dict[str, float | None]]:
        mod_type = "mutation" if mutation else "modification"

        resmap = self.residue_map(pdb_id=pdb_id).copy()
        modres = self.mutated_residues if mutation else self.modified_residues
        modres = modres[modres["pdb_id"] == pdb_id].copy()

        # Add `not_modified` column to residue_map indicating whether a residue is (not) modified
        resmap["key"] = list(zip(resmap["chain_id"], resmap["pdb_residue_number"]))
        modres["key"] = list(zip(modres["chain_id"], modres["residue_number"]))
        resmap["not_modified"] = (~resmap["key"].isin(modres["key"])).astype(int)

        # Compute weighted averages per chain
        resmap["wx"] = resmap["weight"] * resmap["not_modified"]
        grouped = resmap.groupby(["struct_asym_id", "chain_id"], dropna=False, sort=True).agg(
            wsum=("wx", "sum"),
            wtot=("weight", "sum"),
        )

        result: list[dict[str, float]] = []
        for (struct_asym_id, chain_id), row in grouped.iterrows():
            result.append({
                "pdb_id": pdb_id,
                "struct_asym_id": struct_asym_id,
                "chain_id": chain_id,
                f"score_{mod_type}": float(row["wsum"] / row["wtot"]),
            })
        return result

    def _score_outlier(self, pdb_id: str) -> list[dict[str, float | None]]:
        resmap = self.residue_map(pdb_id=pdb_id).copy()

        outres = self.outlier_residues[self.outlier_residues["pdb_id"] == pdb_id].copy()
        outres = (
            outres[["chain_id", "residue_number", "outlier_type_ratio"]]
            .rename(columns={"residue_number": "pdb_residue_number"})
        )

        merged = resmap.merge(
            outres,
            on=["chain_id", "pdb_residue_number"],
            how="left",
            copy=False
        )
        merged["nonoutlier_ratio"] = 1.0 - merged["outlier_type_ratio"].fillna(0.0)

        # Compute weighted averages per chain
        merged["wx"] = merged["weight"] * merged["nonoutlier_ratio"]
        grouped = merged.groupby(["struct_asym_id", "chain_id"], dropna=False, sort=True).agg(
            wsum=("wx", "sum"),
            wtot=("weight", "sum"),
        )

        result: list[dict[str, float]] = []
        for (struct_asym_id, chain_id), row in grouped.iterrows():
            result.append({
                "pdb_id": pdb_id,
                "struct_asym_id": struct_asym_id,
                "chain_id": chain_id,
                "score_outlier": float(row["wsum"] / row["wtot"]),
            })
        return result

    def _score_resolution(self, pdb_id: str) -> list[dict[str, float | None]]:
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
                "struct_asym_id": struct_asym_id,
                "chain_id": chain_id,
                "score_resolution": score(resolution),
            }
            for struct_asym_id, chain_id in self.residue_map(pdb_id=pdb_id)[
                ["struct_asym_id", "chain_id"]
            ].drop_duplicates().to_records(index=False)
        ]

    def _score_ligand(self, pdb_id: str) -> tuple[tuple[str, str, int], np.ndarray]:
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
                "score_ligand": best_dist_min <= self._ligand_min_dist_threshold and best_dist_max <= self._ligand_max_dist_threshold,
            })
        return result
