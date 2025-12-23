# analysis/spatial_filter.py
from __future__ import annotations
import geopandas as gpd
import pandas as pd


class YearPolygonExcluder:
    """
    Exclude points falling inside year-specific polygons.
    """

    def __init__(self, gpkg_path: str, crs: str = "EPSG:4326"):
        self.polys = gpd.read_file(gpkg_path)
        self.polys = self.polys.set_crs(crs) if self.polys.crs is None else self.polys.to_crs(crs)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.longitude, df.latitude),
            crs=self.polys.crs,
        )

        years = self.polys["year"].unique()
        pts = gdf[gdf["year"].isin(years)][["geometry", "year"]]
        polys = self.polys[["geometry", "year"]]

        joined = gpd.sjoin(pts, polys, how="left", predicate="within")
        to_drop = joined.index[
            joined["index_right"].notna()
            & (joined["year_left"] == joined["year_right"])
        ]

        return df.drop(index=to_drop).reset_index(drop=True)
