from typing import Self, Literal
import functools

import pandas as pd


_VALUE_ORDERS = {
    "SING": 1,
    "DOUB": 2,
    "TRIP": 3,
    "QUAD": 4,

    "AROM": 1.5,
    "DELO": 1.5,

    "DIRECTED": 0,
    "PI": 0,
    "POLY": 0,
}



class Bond:
    def __init__(
        self,
        df: pd.DataFrame,
        validate: bool = True,
    ):
        if validate:
            required = {
                "atom_id_1", "atom_id_2", "value_order", "pdbx_aromatic_flag",
                "pdbx_stereo_config", "pdbx_ordinal"
            }
            missing = required - set(df.columns)
            if missing:
                raise KeyError(f"Missing required columns: {sorted(missing)}")
            if not df["value_order"].isin(_VALUE_ORDERS).all():
                invalid = df.loc[~df["value_order"].isin(_VALUE_ORDERS), "value_order"].unique()
                raise ValueError(f"Invalid value_order entries: {sorted(invalid)}")

            if "bond_order" not in df.columns:
                df["bond_order"] = df["value_order"].map(_VALUE_ORDERS).astype(float)

        self._df = df

        self._exploded = None
        self._comp_view = {}
        return

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def exploded(self) -> pd.DataFrame:
        if self._exploded is None:
            payload_cols = self._df.columns.difference(
                {"atom_id_1", "atom_id_2"}
            ).tolist()
            out_cols = ["atom_id", "partner_atom_id", *payload_cols]
            left = self._df.rename(
                columns={"atom_id_1": "atom_id", "atom_id_2": "partner_atom_id"}
            )[out_cols]
            right = self._df.rename(
                    columns={"atom_id_2": "atom_id", "atom_id_1": "partner_atom_id"}
            )[out_cols]
            self._exploded = pd.concat([left, right], ignore_index=True).sort_values(
                by=["comp_id", "pdbx_ordinal", "atom_id"]
            ).reset_index(drop=True)
        return self._exploded

    def __call__(self, comp_id: str) -> Self:
        comp_bond = self._comp_view.get(comp_id)
        if comp_bond:
            return comp_bond
        comp_bond = self._comp_view[comp_id] = Bond(self._df[self._df["comp_id"] == comp_id])
        return comp_bond

    def select(self,operation: Literal["&", "|"] = "&", **kwargs) -> Self:
        conditions = []
        for col_name, value in kwargs.items():
            if col_name not in self._df.columns:
                raise KeyError(f"Column '{col_name}' not in DataFrame")
            col = self._df[col_name]
            if isinstance(value, str | int | float | bool):
                func = col.eq
            else:
                func = col.isin
            conditions.append(func(value))
        if not conditions:
            raise ValueError("No selection criteria provided")
        reduce_fun = {
            "&": lambda x, y: x & y,
            "|": lambda x, y: x | y,
        }.get(operation)
        if reduce_fun is None:
            raise ValueError(f"Unsupported operation '{operation}'. Use '&' or '|'.")
        mask = functools.reduce(reduce_fun, conditions)
        return Bond(self._df[mask])
