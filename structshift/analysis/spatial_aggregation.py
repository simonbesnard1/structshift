import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.stats import energy_distance


def aggregate_by_cell(
    gdf: gpd.GeoDataFrame,
    cell_col: str,
    value_col: str,
    period: tuple[int, int],
    disturbance_col: str | None = None,
    forest_fraction_col: str = "forest_fraction",
    forest_fraction_min: float = 0.3,
    disturbance_min: float = 0.5,
    min_pixels: int = 1,
    reducer: str = "median",
) -> pd.Series:
    """
    Aggregate a numeric variable per spatial cell.

    Returns a Series indexed by cell id.
    """

    subset = gdf[gdf["year"].between(*period)]

    subset = subset[subset[forest_fraction_col] >= forest_fraction_min]

    if disturbance_col is not None:
        subset = subset[subset[disturbance_col] >= disturbance_min]

    subset = subset[[cell_col, value_col]].dropna()

    counts = subset.groupby(cell_col)[value_col].count()
    valid_cells = counts[counts >= min_pixels].index

    grouped = subset[subset[cell_col].isin(valid_cells)].groupby(cell_col)[value_col]

    if reducer == "median":
        return grouped.median()
    elif reducer == "mean":
        return grouped.mean()
    elif reducer.startswith("quantile"):
        q = float(reducer.split(":")[1])
        return grouped.quantile(q)
    else:
        raise ValueError(f"Unknown reducer: {reducer}")



def distribution_distance_by_cell(
    gdf: gpd.GeoDataFrame,
    cell_col: str,
    value_col: str,
    period1: tuple[int, int],
    period2: tuple[int, int],
    disturbance_col: str | None = None,
    forest_fraction_col: str = "forest_fraction",
    forest_fraction_min: float = 0.3,
    disturbance_min: float = 0.5,
    min_pixels: int = 50,
    metric: str = "energy",
) -> pd.Series:
    """
    Compute a distributional distance between two periods per spatial cell.
    """

    def _subset(period):
        s = gdf[gdf["year"].between(*period)]
        s = s[s[forest_fraction_col] >= forest_fraction_min]
        if disturbance_col is not None:
            s = s[s[disturbance_col] >= disturbance_min]
        return s[[cell_col, value_col]].dropna()

    df1 = _subset(period1)
    df2 = _subset(period2)

    cells = set(df1[cell_col]).union(df2[cell_col])
    out = {}

    for cid in cells:
        a = df1[df1[cell_col] == cid][value_col].values
        b = df2[df2[cell_col] == cid][value_col].values

        if len(a) >= min_pixels and len(b) >= min_pixels:
            if metric == "energy":
                out[cid] = energy_distance(a, b)
            else:
                raise ValueError(f"Unknown metric: {metric}")
        else:
            out[cid] = np.nan

    return pd.Series(out)
