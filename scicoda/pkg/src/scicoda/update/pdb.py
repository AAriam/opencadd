from io import StringIO
from pathlib import Path

import pandas as pd
import sciapi
from openmm.app.internal.pdbx.reader.PdbxReader import PdbxReader

from scicoda.update.io import write_df
from scicoda.data import _data_dir
from scicoda.update.df_overlap import same_rows_on_overlap, diff_rows_on_overlap, find_nonunique_ids


def update_all(data_dir: Path | str | None = None):
    if data_dir is None:
        data_dir = _data_dir
    ccd(data_dir=data_dir)
    return


def ccd(
    data_dir: Path | str | None = None,
    basepath: str = "pdb/ccd-"
) -> dict[Path, pd.DataFrame]:
    """Download and store the Chemical Component Dictionary (CCD) of the PDB.

    This function retrieves the CCD from the RCSB PDB,
    processes it into individual tables, and saves each table as a Parquet file.

    Parameters
    ----------
    data_dir
        Data directory of the package.
    basepath
        Base path for storing the CCD tables. Each table will be saved
        with this base path followed by the table name and a `.parquet` extension.

    Returns
    -------
    A dictionary mapping file paths to their corresponding DataFrames.
    """
    if data_dir is None:
        data_dir = _data_dir

    ccd_dfs, aa_dfs = (
        _cif_to_dfs(file_content) for file_content in (
            sciapi.pdb.file.chemical_component_dictionary(
                variant=variant
            ).decode()
            for variant in ("main", "protonation")
        )
    )

    dfs = {}
    for df_name in set(ccd_dfs.keys()).union(set(aa_dfs.keys())):
        ccd_df = ccd_dfs.get(df_name)
        aa_df = aa_dfs.get(df_name)
        if ccd_df is None:
            aa_df["is_aa_variant"] = True
            dfs[df_name] = aa_df.convert_dtypes()
        elif aa_df is None:
            ccd_df["is_aa_variant"] = False
            dfs[df_name] = ccd_df.convert_dtypes()
        else:
            aa_df["is_aa_variant"] = True
            ccd_df["is_aa_variant"] = False
            dfs[df_name] = _merge_dfs(ccd_df=ccd_df, aa_df=aa_df, df_name=df_name).convert_dtypes()

    # Sanity checks
    # 1. Make sure that for each compound in the "chem_comp_atom" table,
    #    each atom ID variant is only used for one specific atom;
    #    i.e., no two atoms in the same compound share the same atom ID variant.
    viol = find_nonunique_ids(
        dfs["chem_comp_atom"],
        group_col="comp_id",
        id_cols=["atom_id", "alt_atom_id", "pdbx_component_atom_id"],
        cross_column=True,
    )
    if not viol.empty:
        raise ValueError(f"Found non-unique atom IDs in chem_comp_atom:\n{viol}")

    dirpath = Path(data_dir)
    out = {}
    for name, df in dfs.items():
        filepath = (dirpath / f"{basepath}{name}").with_suffix(".parquet")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        write_df(df, filepath=filepath)
        out[filepath] = df
    return out


def _merge_dfs(ccd_df: pd.DataFrame, aa_df: pd.DataFrame, df_name: str) -> pd.DataFrame:

    # Select the dataframe with more columns as the main one
    col_dif = list(aa_df.columns.difference(ccd_df.columns))
    if len(col_dif) == 0:
        df1 = ccd_df
        df2 = aa_df
    else:
        df1 = aa_df
        df2 = ccd_df

    # Select common components in both dataframes
    id_col = "id" if df_name == "chem_comp" else "comp_id"
    df1_mask = df1[id_col].isin(df2[id_col])
    df2_mask = df2[id_col].isin(df1[id_col])
    df1_common = df1[df1_mask]
    df2_common = df2[df2_mask]

    # Make sure the main dataframe has more rows than the other dataframe
    exclude = ["pdbx_modified_date", "is_aa_variant"]
    if not same_rows_on_overlap(df1_common, df2_common, exclude=exclude):
        raise ValueError(diff_rows_on_overlap(df1_common, df2_common, exclude=exclude))

    # Merge two dataframes
    sub_df2 = df2[~df2_mask]
    if sub_df2.empty:
        return df1
    return pd.concat([df1, sub_df2]).reset_index(drop=True)


def _cif_to_dfs(file_content: str) -> dict[str, pd.DataFrame]:
    """Read a CIF file and return a dictionary of DataFrames."""
    data = _read_cif(file_content)
    df_list = {}
    for d in data:
        for obj_name in d.getObjNameList():
            obj = d.getObj(obj_name)
            _, column_names, rows = obj.get()
            df = pd.DataFrame(rows, columns=column_names)
            df_list.setdefault(obj_name, []).append(df)

    dfs = {}
    for name, dfs_ in df_list.items():
        df = (
            pd.concat(dfs_, ignore_index=True, copy=False)
            .replace("?", pd.NA)
            .convert_dtypes()
        )
        for col in df.select_dtypes(include=["object", "string"]).columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                if df[col].dropna().isin(["Y", "N"]).all():
                    df[col] = (
                        df[col]
                        .map({"Y": True, "N": False})
                        .astype("boolean")
                    )
        dfs[name] = df
    return dfs


def _read_cif(file_content: str) -> list:
    """Read a CIF file and return a list of data block objects."""
    reader = PdbxReader(StringIO(file_content))
    data = []
    reader.read(data)
    return data