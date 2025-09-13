from __future__ import annotations

from typing import Any
import pandas as pd


def assert_group_uniques(
    df: pd.DataFrame,
    group_by: str | list[str],
    unique_cols: list[str],
    *,
    na_as_value: bool = False,
) -> bool:
    """Make sure that specified columns are unique *within* each group.

    This validates that, for every group defined by ``group_by``,
    each column in ``unique_cols`` contains no repeated values *within that group*.
    If any group has duplicates in any of those columns,
    a ``ValueError`` is raised that lists every offending group,
    the specific columns, and the duplicated values with counts.

    Parameters
    ----------
    df
        Input DataFrame to validate.
    group_by
        Column name or list of column names that define the grouping.
        Groups are formed exactly as in ``df.groupby(group_by, dropna=False)``.
    unique_cols
        Columns that must be unique *within* each group. All must exist in ``df``.
    na_as_value
        If ``True``, ``NaN``/``NA`` is treated as a normal value and can violate
        uniqueness (e.g., two ``NaN``s in a group will be reported as duplicates).
        If ``False`` (default), missing values are ignored for the uniqueness check.

    Raises
    ------
    KeyError
        If any of ``group_by`` or ``unique_cols`` columns are missing from ``df``.
    ValueError
        If duplicates are found. The error message details all violating groups,
        the columns, and the duplicated values with their multiplicities.

    Notes
    -----
    - Values must be hashable for accurate duplicate detection (as required by
      ``value_counts``). If your data contains unhashable values (e.g., lists),
      normalize them beforehand (e.g., convert to tuples or strings).

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     "project": ["A", "A", "A", "B", "B"],
    ...     "user":    [  1,   2,   1,   3,   3],
    ...     "email":   ["x@a","y@a","x@a","z@b","z@b"],
    ... })
    >>> assert_group_uniques(data, group_by="project", unique_cols=["user", "email"])
    Traceback (most recent call last):
        ...
    ValueError: Duplicate values found within groups:
    - Group ('A',):
        - column 'user': 1×2
        - column 'email': 'x@a'×2
    - Group ('B',):
        - column 'user': 3×2
        - column 'email': 'z@b'×2
    """
    # Normalize arguments
    gb_cols: list[str] = [group_by] if isinstance(group_by, str) else list(group_by)
    check_cols: list[str] = list(unique_cols)

    # Column existence checks
    missing_gb = [c for c in gb_cols if c not in df.columns]
    missing_uc = [c for c in check_cols if c not in df.columns]
    if missing_gb or missing_uc:
        parts = []
        if missing_gb:
            parts.append(f"missing group_by columns: {missing_gb}")
        if missing_uc:
            parts.append(f"missing unique_cols columns: {missing_uc}")
        raise KeyError("; ".join(parts))

    # Collect violations: {group_key: {col: {value: count, ...}, ...}, ...}
    violations: dict[tuple[Any, ...], dict[str, dict[Any, int]]] = {}

    for gkey, gdf in df.groupby(gb_cols, dropna=False, sort=False):
        # Ensure tuple key for consistent formatting
        gkey_tup = gkey if isinstance(gkey, tuple) else (gkey,)
        for col in check_cols:
            s = gdf[col]
            # Decide whether to include missing values in duplicate accounting
            vc = s.value_counts(dropna=not na_as_value)  # include NaNs if na_as_value=True
            dup = vc[vc > 1]
            if not dup.empty:
                col_map = violations.setdefault(gkey_tup, {})
                # Preserve the original value objects as keys; convert counts to int
                col_map[col] = {val: int(cnt) for val, cnt in dup.items()}

    if not violations:
        return True
    lines: list[str] = ["Duplicate values found within groups:"]
    for gkey_tup, col_map in violations.items():
        lines.append(f"- Group {gkey_tup}:")
        for col, dup_map in col_map.items():
            # Render values with repr to disambiguate types/NaNs
            rendered = ", ".join(f"{repr(val)}×{cnt}" for val, cnt in dup_map.items())
            lines.append(f"    - column '{col}': {rendered}")
    raise ValueError("\n".join(lines))
