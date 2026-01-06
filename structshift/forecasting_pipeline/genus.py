from __future__ import annotations

import pandas as pd
import geopandas as gpd


def compute_hex_genus_fractions(
    gdf_joined: gpd.GeoDataFrame,
    genus_groups: dict[str, list[int]],
    *,
    genus_col: str = "genus",
    forest_frac_col: str = "forest_fraction",
    pixel_area_col: str = "pixel_area_km2",
) -> pd.DataFrame:
    """
    Compute per-hex genus fractions (area-weighted) + dominant genus.

    Returns DataFrame indexed by hex_id ('index_right') with:
      - one column per genus group (fraction)
      - dominant_genus (string)
      - dominant_frac (float)
    """
    genus_to_group = {gid: gname for gname, ids in genus_groups.items() for gid in ids}

    tmp = gdf_joined[[genus_col, forest_frac_col, pixel_area_col, "index_right"]].copy()
    tmp["genus_group"] = tmp[genus_col].map(genus_to_group)
    tmp = tmp[tmp["genus_group"].notna()]

    # area-weighted by forested area
    tmp["area_km2"] = tmp[forest_frac_col] * tmp[pixel_area_col]

    agg = (
        tmp.groupby(["index_right", "genus_group"], observed=True)["area_km2"]
        .sum()
        .reset_index()
    )

    total_area = agg.groupby("index_right", observed=True)["area_km2"].sum().rename("area_total")
    agg = agg.join(total_area, on="index_right")
    agg["frac"] = agg["area_km2"] / agg["area_total"]

    frac_wide = (
        agg.pivot(index="index_right", columns="genus_group", values="frac")
        .fillna(0.0)
    )

    idx_max = agg.groupby("index_right", observed=True)["area_km2"].idxmax()
    dom = (
        agg.loc[idx_max, ["index_right", "genus_group", "frac"]]
        .set_index("index_right")
        .rename(columns={"genus_group": "dominant_genus", "frac": "dominant_frac"})
    )

    return frac_wide.join(dom)
