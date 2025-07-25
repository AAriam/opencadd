from pathlib import Path

import pandas as pd


def read_pharm_df(
    dirpath: Path | str,
    pdb_id: str,
    job_idx: int,
) -> pd.DataFrame:
    """Read a pharmacophore DataFrame from a file."""
    dirpath = Path(dirpath)
    filepath = dirpath / _pharm_df_filename(pdb_id, job_idx)
    df = read_df(filepath=filepath)
    for col_name in ["label", "target_label"]:
        if col_name in df.columns and df[col_name].dtype == "object":
            df[col_name] = df[col_name].apply(tuple)
    return df


def read_df(filepath: Path | str) -> pd.DataFrame:
    filepath = Path(filepath).resolve()
    if not filepath.is_file():
        raise FileNotFoundError(
            f"DataFrame file not found at '{filepath}'."
        )
    return pd.read_parquet(filepath, engine="pyarrow")


def write_pharm_df(
    df: pd.DataFrame,
    dirpath: Path | str,
    pdb_id: str,
    job_idx: int,
) -> None:
    """Write a pharmacophore DataFrame to a file."""
    return write_df(
        df=df,
        filepath=Path(dirpath) / _pharm_df_filename(pdb_id, job_idx),
    )


def write_df(
    df: pd.DataFrame,
    filepath: Path | str,
) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(
        path=filepath,
        engine="pyarrow",
        compression="zstd",
        compression_level=3,
        index=False,
    )
    return


def _pharm_df_filename(pdb_id: str, job_idx: int) -> str:
    """Generate a filename for a pharmacophore DataFrame."""
    return f"{pdb_id}_{job_idx}.parquet"
