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
    if not filepath.is_file():
        raise FileNotFoundError(
            f"DataFrame file not found at '{filepath}'."
        )
    df = pd.read_parquet(filepath, engine="pyarrow")
    for col_name in ["label", "target_label"]:
        if col_name in df.columns and df[col_name].dtype == "object":
            df[col_name] = df[col_name].apply(tuple)
    return df


def write_pharm_df(
    df: pd.DataFrame,
    dirpath: Path | str,
    pdb_id: str,
    job_idx: int,
) -> None:
    """Write a pharmacophore DataFrame to a file."""
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    df.to_parquet(
        path=dirpath / _pharm_df_filename(pdb_id, job_idx),
        engine="pyarrow",
        compression="zstd",
        compression_level=3,
        index=False,
    )
    return


def _pharm_df_filename(pdb_id: str, job_idx: int) -> str:
    """Generate a filename for a pharmacophore DataFrame."""
    return f"{pdb_id}_{job_idx}.parquet"
