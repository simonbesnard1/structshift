from __future__ import annotations

from pathlib import Path
import pandas as pd


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_inputs(parquet_path: Path, biomass_cols: list[str]) -> pd.DataFrame:
    use_cols = [
        "latitude",
        "longitude",
        "time",
        "harvest",
        "wind_bark_beetle",
        "forest_fraction",
        *biomass_cols,
    ]
    return pd.read_parquet(parquet_path, columns=use_cols)
