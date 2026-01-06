from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd


def get_consistent_valid_hex_ids(
    gdf_joined: gpd.GeoDataFrame,
    disturbance_column: str,
    period1: tuple[int, int],
    period2: tuple[int, int],
    *,
    min_pixels: int,
    min_forest_frac: float,
    min_dist_frac: float,
) -> pd.Index:
    """
    Hexagons that have >= min_pixels “disturbed” pixels in both periods.
    Disturbed pixel = forest_fraction >= min_forest_frac AND disturbance >= min_dist_frac.
    """

    def valid_in_period(df_period: gpd.GeoDataFrame) -> pd.Index:
        m = (df_period["forest_fraction"] >= min_forest_frac) & (df_period[disturbance_column] >= min_dist_frac)
        counts = df_period.loc[m].groupby("index_right").size()
        return counts.index[counts >= min_pixels]

    df1 = gdf_joined[gdf_joined["year"].between(*period1)]
    df2 = gdf_joined[gdf_joined["year"].between(*period2)]

    v1 = valid_in_period(df1)
    v2 = valid_in_period(df2)

    return v1.intersection(v2)


def biomass_by_disturbance_with_mask(
    gdf_joined: gpd.GeoDataFrame,
    disturbance_column: str,
    start_year: int,
    end_year: int,
    valid_hex_ids: pd.Index,
    biomass_cols: list[str],
    *,
    min_forest_frac: float,
    min_dist_frac: float,
) -> pd.DataFrame:
    subset = gdf_joined[
        (gdf_joined["year"].between(start_year, end_year))
        & (gdf_joined[disturbance_column] >= min_dist_frac)
        & (gdf_joined["forest_fraction"] >= min_forest_frac)
        & (gdf_joined["index_right"].isin(valid_hex_ids))
    ]

    if subset.empty:
        return pd.DataFrame(index=pd.Index([], name="index_right"), columns=biomass_cols, dtype=float)

    # median per hex, per biomass model
    return subset.groupby("index_right")[biomass_cols].quantile(0.5)


def attach_biomass_quantiles_to_hexgrid(
    hex_grid: gpd.GeoDataFrame,
    disturbance: str,
    bm_early: pd.DataFrame,
    bm_late: pd.DataFrame,
    biomass_cols: list[str],
) -> gpd.GeoDataFrame:
    hg = hex_grid
    for i, col in enumerate(biomass_cols):
        hg[f"bm_{disturbance}_early_m{i}"] = bm_early[col]
        hg[f"bm_{disturbance}_late_m{i}"] = bm_late[col]
    return hg
