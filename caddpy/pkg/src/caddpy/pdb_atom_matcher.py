from __future__ import annotations

from collections.abc import Sequence, Mapping
from typing import Literal
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment



class PDBAtomMatcher:
    """Merge ATOM rows with CCD chem_comp_atom rows.

    This is done via robust per-residue variant selection
    and multi-alias atom-name matching.
    The algorithm builds a compact lookup from
    (comp_id, any of the CCD name aliases) -> canonical atom_id.
    For each residue instance in `atom`,
    it scores all candidate CCD variants that share the same `main_comp_id`,
    selects the best one by coverage criteria,
    maps each ATOM row to the canonical CCD atom_id,
    and finally joins the full CCD row (all columns).
    Try to pair remaining unmatched ATOM rows with unmatched CCD atoms
    using rigid-fit geometry + connectivity (Hungarian assignment).

    Parameters
    ----------
    atom
        DataFrame for the ATOM table; must contain columns:
        - `atom_res_key_col` (e.g., 'res_num'): unique residue-instance ID,
        - `atom_res_name_col` (e.g., 'res_name'): residue name, e.g. 'GLU',
        - `atom_name_col` (e.g., 'name'): PDB atom name (ID) inside the residue.
    ccd_atom
        DataFrame for CCD's chem_comp_atom data (with protonation variants) with at least:
        - `ccd_main_comp_col` (e.g., 'main_comp_id'): maps to `atom_res_name_col`,
        - `ccd_comp_id_col` (e.g., 'comp_id'): unique chemical component ID,
        - `ccd_atom_id_cols` (e.g., 'atom_id', 'alt_atom_id', 'pdbx_component_atom_id'): atom name aliases mapping to `atom_name_col`.
    ccd_bond
        DataFrame for CCD's chem_comp_bond data.
    atom_res_key_col
        Column in `atom` that uniquely identifies each residue instance in the structure.
    atom_res_name_col
        Column in `atom` holding residue names (maps to `main_comp_id`).
    atom_name_col
        Column in `atom` holding atom names to be matched (maps to `ccd_atom_id_cols`).
    atom_elem_col
        Column in `atom` holding element symbols.
    atom_xyz_cols
        Column in `atom` holding (x,y,z) coordinates.
    ccd_main_comp_col
        Column in `ccd` holding main component ID (maps from `atom_res_name_col`).
    ccd_comp_id_col
        Column in `ccd` for the unique component/variant ID (maps from `atom_res_key_col`).
    ccd_atom_id_cols
        The CCD atom name columns to match against `atom_name_col` in `atom`.
        The first column must be the cannonical name (e.g., 'atom_id').
    ccd_elem_col
        Column in `ccd` holding element symbols.
    ccd_xyz_cols
        Sequence of sequences of column names in `ccd_atom` holding (x,y,z) coordinates.
        The selected set, if any, is the one with the least number of NaNs.
    col_merge_suffix
        Suffix to append to overlapping columns in `atom` when merging with `ccd_atom`.
        No columns in `atom` or `ccd_atom` must already end with this suffix.
    geom_thresh
        Max allowed distance (Å) for candidate pruning after rigid fit.
    w_*
        Weights for cost components. Tune per your data.
    strictness
        What to do when issues are found (e.g., no CCD variants for a residue name):
        - "error": Raise errors.
        - "warn": Emit `warnings.warn` messages.
        - "ignore": Silently ignore issues.
    refine_kwargs
        Optional keyword arguments forwarded to `refine_unmatched_atom_mapping`
        (e.g., thresholds/weights).

    Returns
    -------
    merged_atom : pd.DataFrame
        A left-join of `atom` with matching `ccd_atom` rows (all CCD columns are appended).
        Unmatched `atom` rows get NaNs for CCD fields.
    diagnostics : pd.DataFrame
        One row per residue instance with selection metrics:
        - atom_residue_key, residue_name, best_comp_id,
        - n_atom_names, coverage, missing_count, extra_count,
        - missing_names (list), extras_sample (small sample),
        - match_type ('exact' if missing_count==0 else 'partial'),
        - refined_pairs (int, number of unmatched pairs resolved by refinement).
    res_matching_details : list[tuple]
        List of (residue_key, comp_id, details) for residues where refinement was attempted.

    Raises
    ------
    KeyError
        If required columns are missing.

    Notes
    ------
    Performance strategy:
    - Build a minimal `(comp_id, normalized_name) -> atom_id` map once,
    not a fully 'melted' CCD carrying all CCD columns.
    - Score candidates using set operations; number of residue instances is
    small relative to CCD size.
    - After selecting `comp_id`s, filter CCD down to just those variants, then
    do a single join on (comp_id, atom_id) to bring **all** CCD columns.
    """
    def __init__(
        self,
        atom: pd.DataFrame,
        ccd_atom: pd.DataFrame,
        ccd_bond: pd.DataFrame,
        *,
        atom_res_key_col: str = "res_idx",
        atom_res_name_col: str = "res_name",
        atom_name_col: str = "name",
        atom_elem_col: str = "element",
        atom_xyz_cols: Sequence[str] = ("x", "y", "z"),
        ccd_main_comp_col: str = "main_comp_id",
        ccd_comp_id_col: str = "comp_id",
        ccd_atom_id_cols: Sequence[str] = ("atom_id", "alt_atom_id", "pdbx_component_atom_id"),
        ccd_elem_col: str = "type_symbol",
        ccd_xyz_cols: Sequence[Sequence[str]] = (
            ("pdbx_model_Cartn_x_ideal", "pdbx_model_Cartn_y_ideal", "pdbx_model_Cartn_z_ideal"),
            ("model_Cartn_x", "model_Cartn_y", "model_Cartn_z"),
        ),
        col_merge_suffix: str = "_input",
        geom_thresh: float = 1.25,
        w_geom: float = 1.0,
        w_ngh_geom: float = 0.6,
        w_degree: float = 0.8,
        w_bond_order: float = 0.5,
        w_aromatic: float = 0.5,
        w_ngh_elem: float = 0.3,
        w_chiral: float = 3.0,
        allow_fallback_greedy: bool = True,
        strictness: Literal["error", "warn", "ignore"] = "warn",
    ):

        self._tmp_atom_id_alias_col = "_atom_id_alias"
        self._tmp_canon_atom_id_col = "_canon_atom_id"
        self._tmp_comp_id_col = "_comp_id"

        self.atom = atom
        self.ccd_atom = ccd_atom
        self.ccd_bond = ccd_bond
        self.atom_res_key_col = atom_res_key_col
        self.atom_res_name_col = atom_res_name_col
        self.atom_name_col = atom_name_col
        self.atom_elem_col = atom_elem_col
        self.atom_xyz_cols = atom_xyz_cols
        self.ccd_main_comp_col = ccd_main_comp_col
        self.ccd_comp_id_col = ccd_comp_id_col
        self.ccd_atom_id_cols = ccd_atom_id_cols
        self.ccd_canon_atom_id_col = ccd_atom_id_cols[0]
        self.ccd_elem_col = ccd_elem_col
        self.ccd_xyz_cols = ccd_xyz_cols
        self.col_merge_suffix = col_merge_suffix
        self.geom_thresh = geom_thresh
        self.w_geom = w_geom
        self.w_ngh_geom = w_ngh_geom
        self.w_degree = w_degree
        self.w_bond_order = w_bond_order
        self.w_aromatic = w_aromatic
        self.w_ngh_elem = w_ngh_elem
        self.w_chiral = w_chiral
        self.allow_fallback_greedy = allow_fallback_greedy
        self.strictness = strictness

        self._validate_inputs()
        self.ccd_atom, self.ccd_bond = self._prune_ccd()
        self.name_lookup = self._create_atom_name_lookup()
        self.canon_atom_id_to_all_aliases = self._create_canon_atom_id_to_all_aliases()
        self.main_to_compids = self._create_main_to_compid_map()
        self.res_match = self._find_best_match_per_residue()
        self.atom_merged = self._merge_atom_with_canon_atom_ids()
        if (self.res_match["match_type"] == "partial").any():
            self.atom_merged, self.res_matching_details = self._refine_partial_matches()
        else:
            self.res_matching_details = []
        self.atom_merged = self._merge_atom_with_ccd()
        return

    def _validate_inputs(self) -> None:
        """Validate input DataFrames and required columns."""
        atom_required_cols = {
            self.atom_res_key_col,
            self.atom_res_name_col,
            self.atom_name_col,
            self.atom_elem_col,
            *self.atom_xyz_cols,
        }
        ccd_atom_required_cols = {
            self.ccd_main_comp_col,
            self.ccd_comp_id_col,
            *self.ccd_atom_id_cols,
            self.ccd_canon_atom_id_col,
            self.ccd_elem_col,
            *(col for cols in self.ccd_xyz_cols for col in cols),
        }
        forbidden_cols = {
            self._tmp_atom_id_alias_col,
            self._tmp_canon_atom_id_col,
            self._tmp_comp_id_col,
        }
        for df_name, df, required_col_names in (
            ("atom", self.atom, atom_required_cols),
            ("ccd_atom", self.ccd_atom, ccd_atom_required_cols),
        ):
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"`{df_name}` must be a pandas DataFrame; got {type(df)}")
            df_col_names = set(df.columns)
            missing = required_col_names - df_col_names
            if missing:
                raise KeyError(f"`{df_name}` DataFrame is missing required columns: {sorted(missing)}")
            forbidden = forbidden_cols & df_col_names
            if forbidden:
                raise KeyError(f"`{df_name}` DataFrame has forbidden columns: {sorted(forbidden)}")
            forrbiden_suffix = [col for col in df_col_names if col.endswith(self.col_merge_suffix)]
            if forrbiden_suffix:
                raise KeyError(
                    f"`{df_name}` DataFrame has columns that end with the reserved suffix "
                    f"'{self.col_merge_suffix}': {sorted(forrbiden_suffix)}"
                )
        return

    def _prune_ccd(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Prune CCD data to only relevant residues for speedup."""
        all_unique_res_names_in_atom = self.atom[self.atom_res_name_col].unique()
        ccd_atom = self.ccd_atom[self.ccd_atom[self.ccd_main_comp_col].isin(all_unique_res_names_in_atom)].copy()
        ccd_bond = self.ccd_bond[self.ccd_bond[self.ccd_comp_id_col].isin(ccd_atom[self.ccd_comp_id_col])]
        if self.strictness != "ignore":
            # Check for residues in atom not in CCD
            all_unique_res_names_in_ccd = ccd_atom[self.ccd_main_comp_col].unique()
            res_names_with_no_ccd = set(all_unique_res_names_in_atom) - set(all_unique_res_names_in_ccd)
            if res_names_with_no_ccd:
                msg = (
                    f"Found residues with no CCD entries: {sorted(res_names_with_no_ccd)}. "
                    "All instances of these residues will remain unmatched."
                )
                if self.strictness == "error":
                    raise ValueError(msg)
                elif self.strictness == "warn":
                    warnings.warn(msg, RuntimeWarning)
        return ccd_atom, ccd_bond

    def _create_atom_name_lookup(self) -> pd.DataFrame:
        """Create a minimal lookup DataFrame for (comp_id, any alias) -> canonical atom_id."""
        # Build minimal name lookup: (comp_id, normalized alias) -> canonical atom_id
        alias_frames = []
        for idcol in self.ccd_atom_id_cols:
            tmp = self.ccd_atom[[self.ccd_comp_id_col, idcol]].rename(
                columns={
                    self.ccd_comp_id_col: self._tmp_comp_id_col,
                    idcol: self._tmp_atom_id_alias_col,
                }
            )
            # carry canonical atom_id for the eventual join
            tmp[self._tmp_canon_atom_id_col] = self.ccd_atom[self.ccd_canon_atom_id_col].values
            alias_frames.append(tmp)

        # Add extra known aliases (from non-standard naming conventions used by other tools)

        # 1. Some tools (e.g. openmm/pdbfixer) use "H" instead of "H1"
        #    (seen in N-terminal protonated amino acids, i.e. those with comp_id_suffix "LSN3").
        #    Here, we add this lookup for all components who have an "H1" atom_id but no "H" atom_id or alias.

        # Per-row flags
        has_H1_row = self.ccd_atom[self.ccd_canon_atom_id_col].eq("H1")
        any_H_row = self.ccd_atom[list(self.ccd_atom_id_cols)].eq("H").any(axis=1)
        # Group-level aggregation
        grp = (
            self.ccd_atom
            .assign(_has_H1=has_H1_row, _any_H=any_H_row)
            .groupby(self.ccd_comp_id_col, sort=False)
            .agg(has_H1=("_has_H1", "any"), any_H=("_any_H", "any"))
        )
        alias_frame = pd.DataFrame({
            self._tmp_comp_id_col: grp.index[grp["has_H1"] & ~grp["any_H"]],
            self._tmp_atom_id_alias_col: "H",
            self._tmp_canon_atom_id_col: "H1",
        })
        alias_frames.append(alias_frame)

        # Create the final lookup DataFrame
        name_lookup = (
            pd.concat(alias_frames, ignore_index=True)
            .dropna(subset=[self._tmp_atom_id_alias_col])
            .drop_duplicates(
                subset=[self._tmp_comp_id_col, self._tmp_atom_id_alias_col],
                keep="first"
            )
            .reset_index(drop=True)
        )
        return name_lookup

    def _create_canon_atom_id_to_all_aliases(self) -> dict[str, dict[str, set[str]]]:
        # Synonym sets per comp_id, grouped by canonical atom_id
        canon_atom_id_to_all_aliases: dict[str, dict[str, set[str]]] = {}
        for comp_atom_id, group in (
            self.name_lookup.groupby(
                [self._tmp_comp_id_col, self._tmp_canon_atom_id_col]
            )[self._tmp_atom_id_alias_col]
        ):
            comp_id, canon_atom_id = comp_atom_id
            canon_atom_id_to_all_aliases.setdefault(comp_id, {})[canon_atom_id] = set(group.tolist())
        return canon_atom_id_to_all_aliases

    def _create_main_to_compid_map(self) -> dict[str, list[str]]:
        """Map main_comp_id -> list of comp_id candidates."""
        main_to_compids = (
            self.ccd_atom[[self.ccd_main_comp_col, self.ccd_comp_id_col]]
            .drop_duplicates()
            .groupby(self.ccd_main_comp_col)[self.ccd_comp_id_col]
            .agg(list)
        ).to_dict()
        return main_to_compids

    def _find_best_match_per_residue(self) -> pd.DataFrame:
        """Find the best-matching CCD component ID per residue instance in `self.atom`."""
        residues = self.atom.groupby(self.atom_res_key_col, sort=False)
        res_keys: list = []
        res_names: list[str] = []
        atom_name_sets: list[set[str]] = []
        for res_num, res in residues:
            res_keys.append(res_num)
            res_names.append(res[self.atom_res_name_col].iloc[0])
            atom_name_sets.append(set(res[self.atom_name_col].dropna().tolist()))

        best_comp_ids: list[str | None] = []
        coverages: list[int] = []
        n_missing_list: list[int] = []
        n_extra_list: list[int] = []
        missing_list: list[list[str]] = []
        extra_list: list[list[str]] = []
        match_type_list: list[str] = []

        for res_name, atom_names in zip(res_names, atom_name_sets):
            candidates = self.main_to_compids.get(res_name, [])
            if not candidates:
                # No CCD entries for this residue name
                best_comp_ids.append(None)
                coverages.append(0)
                n_missing_list.append(len(atom_names))
                n_extra_list.append(0)
                missing_list.append(sorted(atom_names))
                extra_list.append([])
                match_type_list.append("none")
                continue

            # Score each candidate with synonym-set semantics
            results: list[tuple[int, int, int, str, set[str], set[str]]] = []
            for comp_id in candidates:
                n_missing, n_extra, n_match, missings, extras = self._score_candidate(atom_names, comp_id)
                results.append((n_missing, n_extra, -n_match, comp_id, missings, extras))

            results.sort()
            n_missing, n_extra, n_match_neg, best_comp_id, missings, extras = results[0]
            best_comp_ids.append(best_comp_id)
            coverages.append(-n_match_neg)
            n_missing_list.append(n_missing)
            n_extra_list.append(n_extra)
            missing_list.append(sorted(list(missings)))
            extra_list.append(sorted(list(extras)))
            if n_missing == 0 and n_extra == 0:
                match_type = "exact"
            elif n_missing == 0:
                match_type = "extra"
            elif n_extra == 0:
                match_type = "missing"
            else:
                match_type = "partial"
            match_type_list.append(match_type)

            if match_type != "exact":
                message = (
                    f"Residue '{res_name}' has no exact CCD variant; chose '{best_comp_id}' "
                    f"with coverage={-n_match_neg}/{len(atom_names)}, missing={n_missing}, extras={n_extra}."
                )
                if self.strictness == "error":
                    raise ValueError(message)
                elif self.strictness == "warn":
                    warnings.warn(message, RuntimeWarning)

        return pd.DataFrame(
            {
                "res_key": res_keys,
                "res_name": res_names,
                "match_type": match_type_list,
                "comp_id": best_comp_ids,
                "n_atoms": [len(s) for s in atom_name_sets],
                "n_match": coverages,
                "n_missing": n_missing_list,
                "n_extra": n_extra_list,
                "missing": missing_list,
                "extra": extra_list,
            }
        )

    def _merge_atom_with_canon_atom_ids(self) -> pd.DataFrame:
        # Attach best component ID to each atom row
        atom = self.atom.copy()
        atom[self._tmp_comp_id_col] = atom[self.atom_res_key_col].map(
            dict(zip(self.res_match["res_key"], self.res_match["comp_id"]))
        )
        # Map each atom row to its canonical CCD atom ID
        atom = atom.merge(
            self.name_lookup,
            left_on=[self._tmp_comp_id_col, self.atom_name_col],
            right_on=[self._tmp_comp_id_col, self._tmp_atom_id_alias_col],
            how="left",
        ).drop(columns=[self._tmp_atom_id_alias_col])
        return atom

    def _refine_partial_matches(self) -> tuple[pd.DataFrame, list[tuple]]:
        "Perform refinement of partially matched residues."
        partial_matches = self.res_match[self.res_match["match_type"] == "partial"]
        needed_comp_ids = partial_matches["comp_id"].unique()
        ccd_atom_by_comp = {
            cid: df for cid, df in
            self.ccd_atom[self.ccd_atom[self.ccd_comp_id_col].isin(needed_comp_ids)].groupby(self.ccd_comp_id_col)
        }
        ccd_bond_by_comp = {
            cid: df for cid, df in
            self.ccd_bond[self.ccd_bond[self.ccd_comp_id_col].isin(needed_comp_ids)].groupby(self.ccd_comp_id_col)
        }
        atom = self.atom_merged.copy()
        atom_unmatched = atom[atom[self.atom_res_key_col].isin(partial_matches["res_key"])]

        reskey_to_matches: dict[object, int] = {}
        res_matching_details = []

        for res_key, res in atom_unmatched.groupby(self.atom_res_key_col, sort=False):
            comp_id = res[self._tmp_comp_id_col].iloc[0]

            # seeds: already matched pairs
            seeds = res.loc[res[self._tmp_canon_atom_id_col].notna(), [self.atom_name_col, self._tmp_canon_atom_id_col]]

            new_pairs, details = self._refine_unmatched_atom_mapping(
                atom=res.copy(),
                ccd_atom=ccd_atom_by_comp.get(comp_id),
                ccd_bond=ccd_bond_by_comp.get(comp_id),
                matched_atom_name_to_canon_id=dict(
                    zip(seeds[self.atom_name_col], seeds[self._tmp_canon_atom_id_col])
                ),
            )
            res_matching_details.append((res_key, comp_id, details))
            if not new_pairs:
                continue

            # apply new matches into atom_with_lookup['ccd_atom_id'] for this residue
            mask_res = atom[self.atom_res_key_col] == res_key
            for atom_name, atom_id in new_pairs.items():
                mask_row = mask_res & (atom[self.atom_name_col] == atom_name)
                atom.loc[mask_row, self._tmp_canon_atom_id_col] = atom_id

            reskey_to_matches[res_key] = new_pairs

        if reskey_to_matches:
            self.res_match["refined_pairs"] = self.res_match["res_key"].map(reskey_to_matches)
        return atom, res_matching_details

    def _merge_atom_with_ccd(self) -> pd.DataFrame:
        # Final bring-in of all CCD columns using (comp_id, atom_id)
        merged = self.atom_merged.merge(
            self.ccd_atom,
            how="left",
            left_on=[self._tmp_comp_id_col, self._tmp_canon_atom_id_col],
            right_on=[self.ccd_comp_id_col, self.ccd_canon_atom_id_col],
            suffixes=(self.col_merge_suffix, ""),
        ).drop(columns=[self._tmp_comp_id_col, self._tmp_canon_atom_id_col])

        # Delete overlapping columns that got suffixed
        for col_name_orig in self.atom_merged.columns.intersection(self.ccd_atom.columns):
            col_name_suffixed = col_name_orig + self.col_merge_suffix
            if all(col_name in merged.columns for col_name in (col_name_orig, col_name_suffixed)):
                atom_is_na = merged[col_name_suffixed].isna()
                ccd_is_na = merged[col_name_orig].isna()
                is_equal = merged[col_name_orig].equals(merged[col_name_suffixed])
                if is_equal or atom_is_na.all():
                    merged = merged.drop(columns=[col_name_suffixed])
                elif ccd_is_na.all():
                    merged = merged.drop(columns=[col_name_orig])
                    merged = merged.rename(columns={col_name_suffixed: col_name_orig})
                elif (is_equal | atom_is_na | ccd_is_na).all():
                    # where one is NA, take the other
                    merged[col_name_orig] = merged[col_name_orig].combine_first(merged[col_name_suffixed])
                    merged = merged.drop(columns=[col_name_suffixed])
                elif self.strictness != "ignore":
                    msg = (
                        f"Column '{col_name_orig}' has conflicting values between ATOM and CCD data. "
                        "Please resolve manually."
                    )
                    if self.strictness == "error":
                        raise ValueError(msg)
                    elif self.strictness == "warn":
                        warnings.warn(msg, RuntimeWarning)
        return merged

    def _score_candidate(self, atom_names: set[str], comp_id: str) -> tuple[int, int, int, set[str], set[str]]:
        """Helper to score a candidate variant against one residue."""
        synsets = self.canon_atom_id_to_all_aliases[comp_id]
        used_canon: set[str] = set()
        matched_atoms: set[str] = set()
        for aname in atom_names:
            for canon, syns in synsets.items():
                if aname in syns:
                    used_canon.add(canon)
                    matched_atoms.add(aname)
                    break
        missing = atom_names - matched_atoms
        extra_canon = set(synsets.keys()) - used_canon
        return (len(missing), len(extra_canon), len(matched_atoms), missing, extra_canon)

    def _refine_unmatched_atom_mapping(
        self,
        atom: pd.DataFrame,
        ccd_atom: pd.DataFrame,
        ccd_bond: pd.DataFrame,
        matched_atom_name_to_canon_id: Mapping[str, str],
    ) -> tuple[dict[str, str], pd.DataFrame]:
        """Find a best 1-to-1 pairing between currently unmatched ATOM rows and unmatched CCD atoms.

        The method:
        1) Rigidly fits CCD ideal coordinates into the PDB frame using *already matched* pairs.
        2) Builds candidate pairs filtered by element (and geometry threshold).
        3) Scores each candidate by geometry, neighbor-geometry consistency vs matched neighbors,
        graph signature similarity (degree, bond orders, aromatic count, neighbor-element multiset),
        and optional chirality sign at tetrahedral centers.
        4) Solves a min-cost assignment to pick the final pairing (with dummy 'unmatched' if sizes differ).

        Parameters
        ----------
        atom
            ATOM rows for a single residue instance.
        ccd_atom
            CCD rows (chem_comp_atom) for the chosen comp_id of that residue.
        ccd_bond
            CCD bond rows for the same comp_id (chem_comp_bond).
        matched_atom_name_to_canon_id
            Mapping of *already matched* ATOM names → CCD atom_ids (canonical).
            These seed the rigid fit and neighbor checks.

        Returns
        -------
        new_pairs, details
            `new_pairs`: dict[atom_name -> ccd_atom_id] for the newly matched pairs only.
            `details`: per-pair scoring diagnostics (DataFrame) including chosen pairs
            and costs, helpful for auditing and thresholds.

        Notes
        -----
        - Costs are *additive*; geometry terms are squared distances (Å²).
        - Chirality check applies only if the center has >=3 already matched neighbors.
        """

        # Trivial case of single atom
        if len(atom) == 1:
            atom_name = atom[self.atom_name_col].iloc[0]
            atom_elem = atom[self.atom_elem_col].iloc[0]
            if len(ccd_atom) == 1:
                # Single CCD atom too;
                # Only match if elements agree
                ccd_atom_id = ccd_atom[self.ccd_canon_atom_id_col].iloc[0]
                ccd_elem = ccd_atom[self.ccd_elem_col].iloc[0]
                if atom_elem == ccd_elem:
                    return {atom_name: ccd_atom_id}, pd.DataFrame(
                        {
                            "atom_name": [atom_name],
                            "atom_id": [ccd_atom_id],
                            "match_type": ["single-atom"],
                        }
                    )
            matching_elem = ccd_atom[ccd_atom[self.ccd_elem_col] == atom_elem]
            if len(matching_elem) == 1:
                # Multiple CCD atoms but exactly one matching element;
                # accept that match
                ccd_atom_id = matching_elem[self.ccd_canon_atom_id_col].iloc[0]
                return {atom_name: ccd_atom_id}, pd.DataFrame(
                    {
                        "atom_name": [atom_name],
                        "atom_id": [ccd_atom_id],
                        "match_type": ["single-atom-element-match"],
                    }
                )
            # No match possible
            return {}, pd.DataFrame(
                {
                    "atom_name": [atom_name],
                    "atom_id": [],
                    "match_type": ["single-atom-element-mismatch"],
                }
            )

        # Trivial case of single CCD atom
        if len(ccd_atom) == 1:
            ccd_atom_id = ccd_atom[self.ccd_canon_atom_id_col].iloc[0]
            ccd_elem = ccd_atom[self.ccd_elem_col].iloc[0]
            matching_elem = atom[atom[self.atom_elem_col] == ccd_elem]
            if len(matching_elem) == 1:
                # Multiple ATOM rows but exactly one matching element;
                # accept that match
                atom_name = matching_elem[self.atom_name_col].iloc[0]
                return {atom_name: ccd_atom_id}, pd.DataFrame(
                    {
                        "atom_name": [atom_name],
                        "atom_id": [ccd_atom_id],
                        "match_type": ["single-atom-element-match"],
                    }
                )
            # No match possible
            return {}, pd.DataFrame(
                {
                    "atom_name": [],
                    "atom_id": [ccd_atom_id],
                    "match_type": ["single-atom-element-mismatch"],
                }
            )


        # --- Split matched / unmatched sets
        atom_coords = atom.set_index(self.atom_name_col)[list(self.atom_xyz_cols)].astype(float)
        atom_elems = atom.set_index(self.atom_name_col)[self.atom_elem_col].astype(str)

        ccd_xyz_notna = [ccd_atom[list(ccd_xyz_cols)].notna().sum().sum() for ccd_xyz_cols in self.ccd_xyz_cols]
        xyz_cols = self.ccd_xyz_cols[np.argmax(ccd_xyz_notna)]
        ccd_coords = ccd_atom.set_index(self.ccd_canon_atom_id_col)[list(xyz_cols)].astype(float)
        ccd_elems = ccd_atom.set_index(self.ccd_canon_atom_id_col)[self.ccd_elem_col].astype(str)

        matched_pairs = [
            (atom_name, atom_id) for atom_name, atom_id in matched_atom_name_to_canon_id.items()
            if atom_name in atom_coords.index and atom_id in ccd_coords.index
        ]

        # Need at least 3 non-collinear points for a stable rigid fit; otherwise skip geometry terms
        use_geometry = len(matched_atom_name_to_canon_id) >= 3
        R = None
        t = None
        if use_geometry:
            P = np.vstack([atom_coords.loc[a].values for a, _ in matched_pairs])
            Q = np.vstack([ccd_coords.loc[c].values for _, c in matched_pairs])
            R, t = _kabsch(P, Q)

        # Transform CCD coords if possible
        if use_geometry:
            ccd_coords_tf = (R @ ccd_coords.values.T).T + t  # (N,3)
            ccd_coords_tf = pd.DataFrame(ccd_coords_tf, index=ccd_coords.index, columns=["x_tf", "y_tf", "z_tf"])
        else:
            ccd_coords_tf = pd.DataFrame(index=ccd_coords.index, columns=["x_tf", "y_tf", "z_tf"], dtype=float)

        matched_atom_names = {a for a, _ in matched_pairs}
        matched_ccd_ids = {c for _, c in matched_pairs}

        atom_unmatched = [a for a in atom_coords.index if a not in matched_atom_names]
        ccd_unmatched = [c for c in ccd_coords.index if c not in matched_ccd_ids]

        # --- CCD graph signatures
        # Build adjacency and bond properties for quick lookup
        adj: dict[str, list[tuple[str, str, str, str]]] = {k: [] for k in ccd_coords.index}
        for row in ccd_bond.itertuples(index=False):
            a1 = getattr(row, "atom_id_1")
            a2 = getattr(row, "atom_id_2")
            order = getattr(row, "value_order", None) or "UNKN"
            arom = getattr(row, "pdbx_aromatic_flag", None)
            stereo = getattr(row, "pdbx_stereo_config", None)
            if a1 in adj and a2 in adj:
                adj[a1].append((a2, order, str(bool(arom)), stereo))
                adj[a2].append((a1, order, str(bool(arom)), stereo))

        def _sig(ccd_id: str) -> tuple[int, dict[str, int], int, dict[str, int]]:
            """(degree, bond_count_by_order, aromatic_count, neighbor_element_multiset)"""
            neighbors = adj.get(ccd_id, [])
            degree = len(neighbors)
            bond_count: dict[str, int] = {}
            arom_count = 0
            ngh_elem: dict[str, int] = {}
            for nb, order, arom, _st in neighbors:
                bond_count[order] = bond_count.get(order, 0) + 1
                arom_count += 1 if arom == "True" else 0
                el = ccd_elems.get(nb, "?")
                ngh_elem[el] = ngh_elem.get(el, 0) + 1
            return degree, bond_count, arom_count, ngh_elem

        sig_cache: dict[str, tuple[int, dict[str, int], int, dict[str, int]]] = {cid: _sig(cid) for cid in ccd_unmatched}

        # Distances to already matched neighbors (for neighbor geometry term)
        matched_ccd_ids_list = list(matched_ccd_ids)
        atom_coords_np = atom_coords.astype(float)
        ccd_coords_tf_np = ccd_coords_tf.astype(float)

        # --- Build cost matrix
        M = len(atom_unmatched)
        N = len(ccd_unmatched)
        cost = np.full((M, N), fill_value=10e10, dtype=float)
        reasons: dict[tuple[int, int], dict[str, float]] = {}

        for i, aname in enumerate(atom_unmatched):
            atom_elem = atom_elems.loc[aname]
            a_xyz = atom_coords_np.loc[aname].values
            for j, cid in enumerate(ccd_unmatched):
                ccd_elem = ccd_elems.loc[cid]
                if atom_elem != ccd_elem:
                    continue  # hard prune

                # base geometry
                geom_term = 0.0
                if use_geometry:
                    c_xyz_tf = ccd_coords_tf_np.loc[cid].values
                    if np.any(np.isnan(c_xyz_tf)):
                        pass
                    else:
                        d = float(np.linalg.norm(a_xyz - c_xyz_tf))
                        if d > self.geom_thresh:
                            continue  # prune
                        geom_term = d * d

                # neighbor geometry term
                ngh_geom_term = 0.0
                if use_geometry:
                    # For every CCD neighbor already matched, compare PDB vs transformed CCD neighbor distances
                    for nb in adj.get(cid, []):
                        nb_id = nb[0]
                        if nb_id in matched_ccd_ids:
                            # find corresponding ATOM neighbor name
                            # reverse lookup: CCD id -> ATOM name (from pre_matched_name_to_ccd_id)
                            a_nb = next((a0 for a0, c0 in matched_pairs if c0 == nb_id), None)
                            if a_nb is None:
                                continue
                            a_nb_xyz = atom_coords_np.loc[a_nb].values
                            c_nb_xyz_tf = ccd_coords_tf_np.loc[nb_id].values
                            if not (np.any(np.isnan(c_nb_xyz_tf)) or np.any(np.isnan(c_xyz_tf))):
                                ngh_geom_term += (float(np.linalg.norm(a_xyz - a_nb_xyz)) -
                                                float(np.linalg.norm(c_xyz_tf - c_nb_xyz_tf))) ** 2

                # graph signature penalties
                deg_c, bondc_c, arom_c, ngh_elem_c = sig_cache[cid]
                # Estimate ATOM-side degree by counting how many of the CCD neighbors are already matched (proxy)
                deg_a = sum(1 for nb in adj.get(cid, []) if nb[0] in matched_ccd_ids)  # proxy only
                degree_term = (deg_c - deg_a) ** 2

                # bond order counts vs matched neighbors only
                bond_term = 0.0
                for k, v in bondc_c.items():
                    # only consider those edges whose neighbor is already matched (proxy again)
                    pass  # leave as soft prior; bond order influences chirality/aromatic indirectly

                aromatic_term = float(arom_c)  # if center is aromatic but neighbors not seen yet, keep small prior

                # neighbor element multiset overlap: penalize missing expected neighbor elements among matched neighbors
                ngh_elem_term = 0.0
                if matched_ccd_ids:
                    # Build multiset of neighbor elements among matched neighbors only
                    elem_counts_seen: dict[str, int] = {}
                    for nb in adj.get(cid, []):
                        nb_id = nb[0]
                        if nb_id in matched_ccd_ids:
                            el = ccd_elems.loc[nb_id]
                            elem_counts_seen[el] = elem_counts_seen.get(el, 0) + 1
                    # penalty: expected minus seen (only positives)
                    for el, cnt in ngh_elem_c.items():
                        lack = max(0, cnt - elem_counts_seen.get(el, 0))
                        ngh_elem_term += lack

                # chirality term (only if at least three matched neighbors define a chiral frame)
                chiral_term = 0.0
                if use_geometry:
                    # Find three matched neighbors
                    tri = [nb[0] for nb in adj.get(cid, []) if nb[0] in matched_ccd_ids][:3]
                    if len(tri) == 3:
                        # PDB signed volume vs transformed CCD signed volume
                        a_nb_xyz = [atom_coords_np.loc[next(a0 for a0, c0 in matched_pairs if c0 == nb)].values for nb in tri]
                        # center positions
                        c_xyz_tf = ccd_coords_tf_np.loc[cid].values
                        vols_ccd = _signed_tetra_volume(
                            ccd_coords_tf_np.loc[tri[0]].values,
                            ccd_coords_tf_np.loc[tri[1]].values,
                            ccd_coords_tf_np.loc[tri[2]].values,
                            c_xyz_tf,
                        )
                        vols_pdb = _signed_tetra_volume(a_nb_xyz[0], a_nb_xyz[1], a_nb_xyz[2], a_xyz)
                        if vols_ccd * vols_pdb < 0:
                            chiral_term = 1.0  # mismatch; weight controls impact

                total = (
                    self.w_geom * geom_term
                    + self.w_ngh_geom * ngh_geom_term
                    + self.w_degree * degree_term
                    + self.w_bond_order * bond_term
                    + self.w_aromatic * aromatic_term
                    + self.w_ngh_elem * ngh_elem_term
                    + self.w_chiral * chiral_term
                )
                cost[i, j] = total
                reasons[(i, j)] = {
                    "geom": geom_term,
                    "ngh_geom": ngh_geom_term,
                    "degree": degree_term,
                    "bond_order": bond_term,
                    "aromatic": aromatic_term,
                    "ngh_elem": ngh_elem_term,
                    "chiral": chiral_term,
                }
        row_ind, col_ind = linear_sum_assignment(cost)

        # Extract matches
        new_pairs: dict[str, str] = {}
        rows = []
        for i, j in zip(row_ind, col_ind):
            aname = atom_unmatched[i]
            cid = ccd_unmatched[j]
            new_pairs[aname] = cid
            rows.append({
                "atom_name": aname,
                "ccd_atom_id": cid,
                "cost": float(cost[i, j]),
                **reasons.get((i, j), {}),
            })

        details = pd.DataFrame(rows).sort_values("cost", ascending=True).reset_index(drop=True)
        return new_pairs, details


def _kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Optimal rigid transform (R, t) with no scaling: R @ Q + t ≈ P.

    Parameters
    ----------
    P
        (N,3) target points (PDB frame).
    Q
        (N,3) source points (CCD ideal), same N and ordering as P.

    Returns
    -------
    R, t
        Rotation (3,3) and translation (3,) so that R@Q + t ≈ P.
    """
    Pc = P.mean(axis=0)
    Qc = Q.mean(axis=0)
    P0 = P - Pc
    Q0 = Q - Qc
    H = Q0.T @ P0
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0.0:  # ensure right-handed
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = Pc - R @ Qc
    return R, t


def _signed_tetra_volume(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    """Signed volume of tetrahedron (a,b,c,d)."""
    return float(np.dot(np.cross(b - a, c - a), d - a) / 6.0)

