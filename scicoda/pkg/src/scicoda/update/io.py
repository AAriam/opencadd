from pathlib import Path

import pandas as pd


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