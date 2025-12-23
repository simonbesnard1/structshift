
def cv_by_cell(
    gdf,
    cell_col,
    value_col,
    period,
    disturbance_col,
    forest_fraction_min=0.3,
    disturbance_min=0.5,
    min_pixels=25,
):
    """
    Compute coefficient of variation (std/mean) per spatial cell.
    """
    start, end = period

    subset = gdf[
        gdf["year"].between(start, end)
        & (gdf["forest_fraction"] >= forest_fraction_min)
        & (gdf[disturbance_col] >= disturbance_min)
    ][[cell_col, value_col]].dropna()

    counts = subset.groupby(cell_col)[value_col].count()
    valid = counts[counts >= min_pixels].index

    grouped = subset[subset[cell_col].isin(valid)].groupby(cell_col)[value_col]
    return grouped.std() / grouped.mean()


def cv_by_cell_and_genus(
    gdf,
    cell_col,
    value_col,
    period,
    disturbance_col,
    genus_ids,
    forest_fraction_min=0.3,
    disturbance_min=0.5,
    min_pixels=25,
):
    subset = gdf[
        gdf["genus"].isin(genus_ids)
        & gdf["year"].between(*period)
        & (gdf["forest_fraction"] >= forest_fraction_min)
        & (gdf[disturbance_col] >= disturbance_min)
    ][[cell_col, value_col]].dropna()

    counts = subset.groupby(cell_col)[value_col].count()
    valid = counts[counts >= min_pixels].index

    grouped = subset[subset[cell_col].isin(valid)].groupby(cell_col)[value_col]
    return grouped.std() / grouped.mean()
