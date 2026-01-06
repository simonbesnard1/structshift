from __future__ import annotations

import pandas as pd
import geopandas as gpd

from structshift.utils.area_calc import pixel_area_latlon_km2


def preprocess_df(
    df: pd.DataFrame,
    biomass_cols: list[str],
    years_min: int,
) -> pd.DataFrame:
    # downcast
    for col in ["harvest", "wind_bark_beetle", "forest_fraction"]:
        df[col] = df[col].astype("float32")
    for col in biomass_cols:
        df[col] = df[col].astype("float32")

    df["time"] = pd.to_datetime(df["time"])
    df["year"] = df["time"].dt.year
    df = df[df["year"] >= years_min].copy()

    # biomass > 0 then convert to carbon
    df[biomass_cols] = df[biomass_cols].where(df[biomass_cols] > 0)
    df[biomass_cols] *= 0.47

    return df


def apply_year_polygons_filter(
    df: pd.DataFrame,
    polys_gpkg: str,
    crs: str = "EPSG:4326",
    drop: bool = False,
) -> pd.DataFrame:
    """
    Reproduces your “year-specific polygons” join and identifies points to drop.
    By default `drop=False` (same as your current code where drop is commented out).
    """
    polys = gpd.read_file(polys_gpkg)
    polys = polys.set_crs(crs) if polys.crs is None else polys.to_crs(crs)

    gdf_pts = gpd.GeoDataFrame(
        df[["latitude", "longitude", "year"]].copy(),
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=crs,
    )

    years_in_polys = polys["year"].unique()
    pts_sub = gdf_pts[gdf_pts["year"].isin(years_in_polys)][["geometry", "year"]]
    polys_sub = polys[["geometry", "year"]]

    joined = gpd.sjoin(pts_sub, polys_sub, how="left", predicate="within")
    to_drop_idx = joined.index[
        joined["index_right"].notna() & (joined["year_left"] == joined["year_right"])
    ]

    if drop and len(to_drop_idx) > 0:
        return df.drop(index=to_drop_idx).reset_index(drop=True)

    return df


def add_pixel_area_km2(
    df: pd.DataFrame,
    pixel_res_deg: float,
) -> pd.DataFrame:
    df = df.copy()
    df["pixel_area_km2"] = pixel_area_latlon_km2(df["latitude"], pixel_res_deg, pixel_res_deg)
    df["pixel_area_km2"] = df["pixel_area_km2"].astype("float32")
    return df
