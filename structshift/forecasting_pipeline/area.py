from __future__ import annotations

import pandas as pd
import geopandas as gpd


def annual_area_by_disturbance(
    gdf_joined: gpd.GeoDataFrame,
    disturbance_column: str,
    *,
    start_year: int,
    end_year: int,
    min_forest_frac: float,
    min_dist_frac: float,
    km2_to_Mha: float = 1 / 10_000,
) -> pd.DataFrame:
    subset = gdf_joined[
        (gdf_joined["year"].between(start_year, end_year))
        & (gdf_joined[disturbance_column] >= min_dist_frac)
        & (gdf_joined["forest_fraction"] >= min_forest_frac)
    ].copy()

    if subset.empty:
        return pd.DataFrame(columns=["index_right", "area_Mha", "year", "disturbance"])

    subset["area_Mha"] = (
        subset["forest_fraction"] * subset[disturbance_column] * subset["pixel_area_km2"] * km2_to_Mha
    ).astype("float32")

    out = (
        subset.groupby(["year", "index_right"])["area_Mha"]
        .sum()
        .reset_index()
    )
    out["disturbance"] = disturbance_column
    return out


def pivot_area_timeseries(area_df: pd.DataFrame, years: pd.Index) -> pd.DataFrame:
    pivot = area_df.pivot_table(
        index="index_right",
        columns="year",
        values="area_Mha",
        fill_value=0.0,
        aggfunc="sum",
    )
    return pivot.reindex(columns=years, fill_value=0.0)
