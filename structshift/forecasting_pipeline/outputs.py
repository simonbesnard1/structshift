from __future__ import annotations

import pandas as pd
import geopandas as gpd
from pathlib import Path


def write_outputs(
    *,
    outdir: Path,
    df_boxplot: pd.DataFrame,
    summary_box: pd.DataFrame,
    hex_grid: gpd.GeoDataFrame,
    forecast_results_df: pd.DataFrame,
    filenames: dict[str, str],
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    df_boxplot.to_parquet(outdir / filenames["boxplot_parquet"], index=False)
    summary_box.to_csv(outdir / filenames["summary_csv"], index=False)

    hex_grid_out = hex_grid[["geometry", "delta_biomass_bark", "delta_biomass_harvest"]].copy()
    hex_grid_out.to_file(outdir / filenames["hex_gpkg"], driver="GPKG")

    forecast_results_df.to_csv(outdir / filenames["forecast_csv"], index=False)
