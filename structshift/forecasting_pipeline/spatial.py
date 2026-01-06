from __future__ import annotations

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import box

from structshift.analysis.hexgrid import make_hex_grid


def load_europe_boundary(world_gpk: str, crs_europe: str) -> gpd.GeoDataFrame:
    world = gpd.read_file(world_gpk).to_crs(crs_europe)

    europe = world[
        (world["CONTINENT"] == "Europe") & (~world["ISO_A3"].isin(["RUS", "ISL"]))
    ].copy()

    bbox = gpd.GeoDataFrame(
        geometry=[box(-20, 32, 45, 71)],
        crs="EPSG:4326",
    ).to_crs(crs_europe)

    return gpd.clip(europe, bbox)


def to_points_gdf(df: pd.DataFrame, crs_in: str, crs_out: str) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs=crs_in,
    )
    return gdf.to_crs(crs_out)


def make_hex_and_join_points(
    gdf_points: gpd.GeoDataFrame,
    europe: gpd.GeoDataFrame,
    hex_diameter_m: int,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    hex_grid = make_hex_grid(
        gdf_points,
        hex_diameter=hex_diameter_m,
        clip_geometry=europe,
        return_index=True,
    )
    gdf_joined = gpd.sjoin(gdf_points, hex_grid, how="inner", predicate="within")
    return hex_grid, gdf_joined


def attach_regions(hex_grid: gpd.GeoDataFrame, eco_path: str, crs: str) -> gpd.GeoDataFrame:
    eco = gpd.read_file(eco_path).to_crs(crs)

    hg = gpd.sjoin(
        hex_grid,
        eco[["code", "geometry"]],
        how="left",
        predicate="intersects",
    ).rename(columns={"code": "region_id"})

    # mode across overlaps
    hg["region_id"] = (
        hg.groupby(hg.index)["region_id"]
          .transform(lambda x: x.mode().iat[0] if not x.mode().empty else np.nan)
    )

    # de-dup index
    hg = hg[~hg.index.duplicated(keep="first")].copy()

    # drop sjoin baggage if present
    if "index_right" in hg.columns:
        hg = hg.drop(columns=["index_right"])

    return hg
