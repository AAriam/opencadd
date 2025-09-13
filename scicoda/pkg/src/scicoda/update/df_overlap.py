from __future__ import annotations

import pandas as pd
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_timedelta64_dtype, is_bool_dtype, is_numeric_dtype, is_string_dtype,
)
from pandas.util import hash_pandas_object


class _MissingSentinel:
    __slots__ = ()
    def __repr__(self) -> str:
        return "<MISSING>"


_MISSING = _MissingSentinel()


def find_nonunique_ids(
    df: pd.DataFrame,
    *,
    group_col: str = "comp_id",
    id_cols: list[str] = ["atom_id", "alt_atom_id", "pdbx_component_atom_id"],
    cross_column: bool = True,
) -> pd.DataFrame:
    """Report rows that violate uniqueness constraints.

    Finds duplicates of identifiers within each `group_col`.
    Two modes:
    - cross_column=True: the union of all `id_cols` must be unique per group.
      I.e., if an identifier appears in any of the `id_cols` of one row, it
      must not appear in any of the `id_cols` of any other row in that same group.
    - cross_column=False: enforce uniqueness *per column* within each group.

    Parameters
    ----------
    df
        Input DataFrame. Must contain `group_col` and `id_cols`.
    group_col
        Column name indicating the grouping key (default: "comp_id").
    id_cols
        List of column names whose values must be unique per group.
    cross_column
        If True, enforce uniqueness across the union of `id_cols`.
        If False, enforce uniqueness separately per column.

    Returns
    -------
    pd.DataFrame
        A DataFrame describing violations. Empty if none.
        For cross_column=True: columns include [group_col, "row_index", "which",
        "identifier", "count_in_group"].
        For cross_column=False: columns include [group_col, "row_index", "which",
        "identifier", "count_in_group"] with `which` indicating the offending column.

    Raises
    ------
    KeyError
        If required columns are missing.

    Notes
    ------
    - NaNs are ignored.
    - Use this function to *inspect* problems; it does not mutate `atom`.
    """
    # Validate columns
    missing = {group_col, *id_cols} - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    # Keep a stable row reference
    df = df.copy()
    df = df.reset_index(drop=False).rename(columns={"index": "row_index"})

    if cross_column:
        # Long form over the union of id_cols
        long = (
            df[[group_col, "row_index", *id_cols]]
            .melt(
                id_vars=[group_col, "row_index"],
                value_vars=id_cols,
                var_name="which",
                value_name="identifier",
            )
            .dropna(subset=["identifier"])
        )

        # Count identifier occurrences per group across all columns
        counts = (
            long.groupby([group_col, "identifier"], as_index=False)
            .size()
            .rename(columns={"size": "count_in_group"})
        )

        # Flag identifiers occurring more than once in the same group
        dup_ids = counts[counts["count_in_group"] > 1]

        # Join back to list all violating rows/columns for those identifiers
        violations = long.merge(
            dup_ids, on=[group_col, "identifier"], how="inner"
        ).sort_values([group_col, "identifier", "row_index", "which"], kind="stable")

        return violations[[group_col, "row_index", "which", "identifier", "count_in_group"]]

    # Per-column uniqueness within each group
    pieces: list[pd.DataFrame] = []
    for col in id_cols:
        sub = df[[group_col, "row_index", col]].dropna(subset=[col]).rename(
            columns={col: "identifier"}
        )
        counts = (
            sub.groupby([group_col, "identifier"], as_index=False)
            .size()
            .rename(columns={"size": "count_in_group"})
        )
        dup_ids = counts[counts["count_in_group"] > 1]
        viol = (
            sub.merge(dup_ids, on=[group_col, "identifier"], how="inner")
            .assign(which=col)
            .sort_values([group_col, "identifier", "row_index"], kind="stable")
        )
        pieces.append(viol)

    if not pieces:
        return pd.DataFrame(columns=[group_col, "row_index", "which", "identifier", "count_in_group"])

    out = pd.concat(pieces, ignore_index=True)
    return out[[group_col, "row_index", "which", "identifier", "count_in_group"]]


def same_rows_on_overlap(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    exclude: list[str] | None = None,
) -> bool:
    """Check whether two DataFrames have the same data in their overlapping sections.

    Return True iff, on overlapping columns (excluding `exclude`),
    the multiset of rows in df2 is a subset of the multiset of rows in df1.
    Order is ignored; duplicates matter. Extra rows in df1 are allowed.
    """
    ex = set(exclude or [])
    common = df1.columns.intersection(df2.columns).difference(ex)
    a = _canon_df(df1, common)
    b = _canon_df(df2, common)
    c1 = _row_hash_counts(a)
    c2 = _row_hash_counts(b)
    keys = c1.index.union(c2.index, sort=False)
    c1a = c1.reindex(keys, fill_value=0)
    c2a = c2.reindex(keys, fill_value=0)
    return (c1a >= c2a).all()


def diff_rows_on_overlap(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    exclude: list[str] | None = None,
) -> pd.DataFrame:
    ex = set(exclude or [])
    common = df1.columns.intersection(df2.columns).difference(ex)
    a = _canon_df(df1, common)
    b = _canon_df(df2, common)

    h1 = _row_hash_counts(a)
    h2 = _row_hash_counts(b)
    keys = h1.index.union(h2.index, sort=False)

    def first_row_by_hash(df: pd.DataFrame) -> dict[int, tuple]:
        if df.empty:
            return {}
        h = hash_pandas_object(df, index=False)
        first = ~h.duplicated()
        return {int(hv): row for hv, row in zip(h[first], df[first].itertuples(index=False, name=None))}

    rep = first_row_by_hash(a)
    rep |= {int(k): v for k, v in first_row_by_hash(b).items() if int(k) not in rep}

    out = pd.DataFrame({"df1": h1.reindex(keys, fill_value=0),
                    "df2": h2.reindex(keys, fill_value=0)})

    # Positive means df1 has at least as many as df2 (OK or surplus);
    # Negative means df1 is short vs df2 (NOT OK under subset semantics).
    out["delta"] = out["df1"] - out["df2"]

    # Keep only shortages (rows where df2 has more than df1).
    out = out.loc[out["df1"] < out["df2"]]

    if out.empty:
        return out  # nothing violates the subset condition

    # Optional: make the shortage explicit and easy to read
    out["shortfall"] = out["df2"] - out["df1"]

    # Replace hash index with the actual row values
    tuples = [rep[int(h)] for h in out.index]
    out.index = pd.MultiIndex.from_tuples(tuples, names=list(a.columns))

    # Nice column order
    return out.loc[:, ["df1", "df2", "shortfall", "delta"]].sort_index()


def _canon_df(df: pd.DataFrame, cols: pd.Index) -> pd.DataFrame:
    """Canonicalize selected columns with deterministic order + unify missing."""
    cols = pd.Index(sorted(cols))
    out = df.loc[:, cols].copy()

    for c in cols:
        out[c] = _canon_col(out[c])

    # FINAL SWEEP: identical missing token across all columns/dtypes
    for c in cols:
        col = out[c].astype(object)
        mask = pd.isna(col)
        if mask.any():
            col.loc[mask] = _MISSING
        out[c] = col
    return out


def _canon_col(s: pd.Series) -> pd.Series:
    """Canonicalize a column so logically equal values hash the same."""
    s = s.copy()
    dt = s.dtype

    if isinstance(dt, pd.CategoricalDtype):
        s = s.astype(object)
        dt = s.dtype

    if isinstance(dt, pd.DatetimeTZDtype):
        return s.dt.tz_convert("UTC").dt.tz_localize(None).view("int64")
    if is_datetime64_any_dtype(dt):
        return s.view("int64")
    if is_timedelta64_dtype(dt):
        return s.view("int64")
    if is_bool_dtype(dt):
        return s.astype("boolean").astype("Int8")
    if is_numeric_dtype(dt):
        return pd.to_numeric(s, errors="coerce").astype("float64")

    if dt == object or is_string_dtype(dt):
        # Normalize to StringDtype to stabilize NA handling
        s = s.where(~pd.isna(s), pd.NA).astype("string")
    return s


def _row_hash_counts(df: pd.DataFrame) -> pd.Series:
    h = hash_pandas_object(df, index=False)  # stable row hash
    return h.value_counts(sort=False)


