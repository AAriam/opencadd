from collections.abc import Sequence
import warnings

import numpy as np
import pandas as pd



def merge_atom_with_ccd(
    atom: pd.DataFrame,
    ccd: pd.DataFrame,
    *,
    atom_residue_key: str = "res_num",
    atom_residue_name_col: str = "res_name",
    atom_name_col: str = "name",
    ccd_main_comp_col: str = "main_comp_id",
    ccd_comp_id_col: str = "comp_id",
    ccd_id_cols: Sequence[str] = ("atom_id", "alt_atom_id", "pdbx_component_atom_id"),
    prefer_order: Sequence[str] = ("atom_id", "alt_atom_id", "pdbx_component_atom_id"),
    issue_warnings: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge ATOM rows with CCD chem_comp_atom rows.

    This is done via robust per-residue variant selection
    and multi-alias atom-name matching.
    The algorithm builds a compact lookup from
    (comp_id, any of the CCD name aliases)
    -> canonical atom_id (preferring `atom_id` over `alt_atom_id` over `pdbx_component_atom_id`).
    For each residue instance in `atom`,
    it scores all candidate CCD variants that share the same `main_comp_id`,
    selects the best one by coverage criteria,
    maps each ATOM row to the canonical CCD atom_id,
    and finally joins the full CCD row (all columns).

    Parameters
    ----------
    atom
        DataFrame for the ATOM table; must contain columns:
        - `atom_residue_key` (e.g., 'res_num'): unique residue-instance id,
        - `atom_residue_name_col` (e.g., 'res_name'): residue name, e.g. 'GLU',
        - `atom_name_col` (e.g., 'name'): PDB atom name inside the residue.
    ccd
        DataFrame for chem_comp_atom (+ protonation variants) with at least:
        - `ccd_main_comp_col` (e.g., 'main_comp_id'): maps to `res_name`,
        - `ccd_comp_id_col`  (e.g., 'comp_id'): unique variant id,
        - three id columns `ccd_id_cols`: ('atom_id', 'alt_atom_id',
          'pdbx_component_atom_id'), plus any number of other CCD columns
          which will be added to the merged output.
    atom_residue_key
        Column in `atom` that uniquely identifies a residue instance.
    atom_residue_name_col
        Column in `atom` holding the residue name (maps to CCD `main_comp_id`).
    atom_name_col
        Column in `atom` holding the atom name to be matched.
    ccd_main_comp_col
        Column in `ccd` for the main component id (maps from `res_name`).
    ccd_comp_id_col
        Column in `ccd` for the unique component/variant id.
    ccd_id_cols
        The three CCD name columns to match against, in *any* order.
        Must contain 'atom_id', 'alt_atom_id', 'pdbx_component_atom_id'.
    prefer_order
        Preference order among the three CCD id columns when the same ATOM
        name could match multiple aliases. Default: atom_id > alt_atom_id > pdbx.
    issue_warnings
        If True, emit `warnings.warn` for residues where no CCD variant fully
        covers all ATOM names.

    Returns
    -------
    merged_atom : pd.DataFrame
        A left-join of `atom` with matching CCD rows (all CCD columns are
        appended). Includes helper columns:
        - 'best_comp_id' : chosen CCD variant per residue instance,
        - 'ccd_atom_id'  : canonical CCD atom_id used for the join.
        Unmatched ATOM rows get NaNs for CCD fields.
    diagnostics : pd.DataFrame
        One row per residue instance with selection metrics:
        - atom_residue_key, residue_name, best_comp_id,
        - n_atom_names, coverage, missing_count, extra_count,
        - missing_names (list), extras_sample (small sample),
        - status ('exact' if missing_count==0 else 'partial').

    Raises
    ------
    KeyError
        If required columns are missing.
    ValueError
        If `ccd_id_cols` is invalid or `prefer_order` is not a permutation
        of it.

    Notes
    ------
    Performance strategy:
    - Build a minimal `(comp_id, normalized_name) -> atom_id` map once,
      not a fully 'melted' CCD carrying all CCD columns.
    - Score candidates using set operations; number of residue instances is
      small relative to CCD size.
    - After selecting `comp_id`s, filter CCD down to just those variants, then
      do a single join on (comp_id, atom_id) to bring **all** CCD columns.

    Examples
    --------
    >>> merged, diag = merge_atom_with_ccd(atom_df, ccd_df)
    >>> diag.query("status != 'exact'")[["res_num", "res_name", "missing_names"]]
    """
    # ---- Validation ---------------------------------------------------------
    required_atom = {atom_residue_key, atom_residue_name_col, atom_name_col}
    missing_atom = required_atom - set(atom.columns)
    if missing_atom:
        raise KeyError(f"`atom` missing required columns: {sorted(missing_atom)}")

    required_ccd = {ccd_main_comp_col, ccd_comp_id_col, *ccd_id_cols}
    missing_ccd = required_ccd - set(ccd.columns)
    if missing_ccd:
        raise KeyError(f"`ccd` missing required columns: {sorted(missing_ccd)}")

    ccd_id_cols = tuple(ccd_id_cols)
    prefer_order = tuple(prefer_order)
    if set(prefer_order) != set(ccd_id_cols) or len(prefer_order) != 3:
        raise ValueError("`prefer_order` must be a permutation of `ccd_id_cols`.")

    # ---- Normalization ------------------------------------------------------
    def _norm_name(x: object) -> str | None:
        """Normalize atom names: uppercase and strip spaces; preserve None/NaN."""
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        s = str(x).upper().replace(" ", "")
        return s if s else None

    # Do not mutate inputs
    atom_local = atom.copy()
    ccd_local = ccd.copy()

    # Normalize residue names and atom names
    atom_local["_resname_norm"] = atom_local[atom_residue_name_col].astype(str).str.upper()
    atom_local["_name_norm"] = atom_local[atom_name_col].map(_norm_name)

    # Keep only CCD rows for residue names present in atom (huge speedup)
    resnames_needed = atom_local["_resname_norm"].unique()
    ccd_local["_main_norm"] = ccd_local[ccd_main_comp_col].astype(str).str.upper()
    ccd_filtered = ccd_local[ccd_local["_main_norm"].isin(resnames_needed)].copy()

    # Build minimal name lookup: (comp_id, normalized alias) -> canonical atom_id
    alias_frames = []
    priority_map = {col: i for i, col in enumerate(prefer_order)}
    for idcol in ccd_id_cols:
        tmp = ccd_filtered[[ccd_comp_id_col, idcol]].copy()
        tmp["_match_norm"] = tmp[idcol].map(_norm_name)
        tmp["_priority"] = priority_map[idcol]
        # carry canonical atom_id for the eventual join; canonical is 'atom_id'
        tmp["_canonical_atom_id"] = ccd_filtered["atom_id"].values
        alias_frames.append(tmp[[ccd_comp_id_col, "_match_norm", "_priority", "_canonical_atom_id"]])

    name_lookup = (
        pd.concat(alias_frames, ignore_index=True)
        .dropna(subset=["_match_norm"])
        .sort_values(["_priority"])
        .drop_duplicates(subset=[ccd_comp_id_col, "_match_norm"], keep="first")
        .reset_index(drop=True)
    )

    # Synonym sets per comp_id, grouped by canonical atom_id
    comp_to_synsets: dict[str, dict[str, set[str]]] = {}
    for comp_id, group in (
        name_lookup.groupby([ccd_comp_id_col, "_canonical_atom_id"])["_match_norm"]
    ):
        cid, canon = comp_id
        comp_to_synsets.setdefault(cid, {})[canon] = set(group.tolist())

    # Helper to score a candidate variant against one residue
    def _score_candidate(atom_names: set[str], comp_id: str) -> tuple[int, int, int, set[str], set[str]]:
        synsets = comp_to_synsets.get(comp_id, {})
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


    # Map main_comp_id -> list of comp_id candidates
    main_to_compids = (
        ccd_filtered[[ccd_main_comp_col, ccd_comp_id_col]]
        .drop_duplicates()
        .groupby(ccd_main_comp_col)[ccd_comp_id_col]
        .agg(list)
    )
    # Use normalized key
    main_to_compids.index = main_to_compids.index.str.upper()
    main_to_compids = main_to_compids.to_dict()

    # ---- Per-residue selection ---------------------------------------------
    # Precompute atom name sets per residue instance
    grp = atom_local.groupby(atom_residue_key, sort=False)
    res_nums: list = []
    res_names_norm: list[str] = []
    atom_name_sets: list[set[str]] = []

    for res_num, sub in grp:
        res_nums.append(res_num)
        res_names_norm.append(sub["_resname_norm"].iloc[0])
        atom_name_sets.append(set(sub["_name_norm"].dropna().tolist()))

    best_comp_ids: list[str | None] = []
    coverage_list: list[int] = []
    missing_cnt_list: list[int] = []
    extra_cnt_list: list[int] = []
    missing_names_list: list[list[str]] = []
    extras_sample_list: list[list[str]] = []
    status_list: list[str] = []

    for rname_norm, atom_names in zip(res_names_norm, atom_name_sets):
        candidates = main_to_compids.get(rname_norm, [])
        if not candidates:
            # No CCD entries for this residue name
            best_comp_ids.append(None)
            coverage_list.append(0)
            missing_cnt_list.append(len(atom_names))
            extra_cnt_list.append(0)
            missing_names_list.append(sorted(atom_names))
            extras_sample_list.append([])
            status_list.append("no_ccd")
            continue

        # Score each candidate with synonym-set semantics
        results: list[tuple[int, int, int, str, set[str], set[str]]] = []
        for comp_id in candidates:
            missing_n, extra_n, cov, miss, extra = _score_candidate(atom_names, comp_id)
            results.append((missing_n, extra_n, -cov, comp_id, miss, extra))

        results.sort()
        missing_n, extra_n, neg_cov, best, miss, extra = results[0]
        best_comp_ids.append(best)
        coverage_list.append(-neg_cov)
        missing_cnt_list.append(missing_n)
        extra_cnt_list.append(extra_n)
        missing_names_list.append(sorted(list(miss))[:10])
        extras_sample_list.append(sorted(list(extra))[:10])
        status_list.append("exact" if missing_n == 0 else "partial")

        if issue_warnings and missing_n > 0:
            warnings.warn(
                f"Residue '{rname_norm}' has no exact CCD variant; chose '{best}' "
                f"with coverage={-neg_cov}/{len(atom_names)}, missing={missing_n}, extras={extra_n}.",
                RuntimeWarning,
            )

    diagnostics = pd.DataFrame(
        {
            atom_residue_key: res_nums,
            "residue_name": res_names_norm,
            "best_comp_id": best_comp_ids,
            "n_atom_names": [len(s) for s in atom_name_sets],
            "coverage": coverage_list,
            "missing_count": missing_cnt_list,
            "extra_count": extra_cnt_list,
            "missing_names": missing_names_list,
            "extras_sample": extras_sample_list,
            "status": status_list,
        }
    )

    # ---- Row-level merge ----------------------------------------------------
    # Attach best_comp_id to each ATOM row
    resnum_to_best = dict(zip(res_nums, best_comp_ids, strict=False))
    atom_local["best_comp_id"] = atom_local[atom_residue_key].map(resnum_to_best)

    # Map each ATOM row (best_comp_id, name_norm) -> canonical CCD atom_id
    # Only attempt where comp_id is known
    atom_with_lookup = atom_local.merge(
        name_lookup.rename(columns={"_match_norm": "_name_norm"}),
        how="left",
        left_on=["best_comp_id", "_name_norm"],
        right_on=[ccd_comp_id_col, "_name_norm"],
        suffixes=("", "_lkp"),
    )

    atom_with_lookup.rename(columns={"_canonical_atom_id": "ccd_atom_id"}, inplace=True)
    # We don't need the lookup key columns after this point
    atom_with_lookup.drop(columns=[ccd_comp_id_col, "_priority"], inplace=True, errors="ignore")

    # Filter CCD to just the chosen comp_ids for the whole structure (big perf win)
    chosen_comp_ids = pd.Series(best_comp_ids, dtype="object").dropna().unique().tolist()
    if chosen_comp_ids:
        ccd_subset = ccd_local[ccd_local[ccd_comp_id_col].isin(chosen_comp_ids)].copy()
    else:
        ccd_subset = ccd_local.iloc[0:0].copy()  # empty

    # Final bring-in of all CCD columns using (comp_id, atom_id)
    merged = atom_with_lookup.merge(
        ccd_subset,
        how="left",
        left_on=["best_comp_id", "ccd_atom_id"],
        right_on=[ccd_comp_id_col, "atom_id"],
        suffixes=("", "_ccd"),
    )

    # Clean helpers
    merged.drop(columns=["_resname_norm", "_name_norm", "_main_norm"], inplace=True, errors="ignore")

    return merged, diagnostics
