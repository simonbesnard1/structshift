import numpy as np

def aggregate_genus_biomass(
    df,
    genus_groups,
    biomass_cols,
    disturbance_col,
    period,
    forest_fraction_min=0.3,
    disturbance_min=0.5,
):
    """
    Aggregate disturbed biomass and area per genus group.

    Returns dict:
      genus → (median, p5, p95, area_mha)
    """
    start, end = period
    out = {}

    for genus, genus_ids in genus_groups.items():
        sub = df[
            (df["genus"].isin(genus_ids)) &
            (df["year"].between(start, end)) &
            (df["forest_fraction"] >= forest_fraction_min) &
            (df[disturbance_col] >= disturbance_min)
        ]

        if sub.empty:
            out[genus] = (np.nan, np.nan, np.nan, 0.0)
            continue

        # Ensemble biomass loss (TgC)
        vals = [
            (
                sub[bcol]
                * sub["forest_fraction"]
                * sub["pixel_area_km2"]
                * 100
                * sub[disturbance_col]
            ).sum() / 1e6
            for bcol in biomass_cols
        ]

        # Disturbed area (Mha)
        area_mha = (
            sub["forest_fraction"]
            * sub[disturbance_col]
            * sub["pixel_area_km2"]
        ).sum() / 1e4

        out[genus] = (
            np.median(vals),
            np.percentile(vals, 5),
            np.percentile(vals, 95),
            area_mha,
        )

    return out
