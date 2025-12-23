import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon


def make_hex_grid(
    gdf: gpd.GeoDataFrame,
    hex_diameter: float,
    clip_geometry: gpd.GeoDataFrame | None = None,
    return_index: bool = True,
) -> gpd.GeoDataFrame:
    """
    Create a regular hexagonal grid covering the extent of a GeoDataFrame.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input data in a projected CRS (units = meters).
    hex_diameter : float
        Flat-to-flat hex diameter in CRS units (meters).
    clip_geometry : GeoDataFrame, optional
        Geometry used to clip the hex grid (e.g. Europe boundary).
    return_index : bool
        If True, adds a stable hex_id column.

    Returns
    -------
    GeoDataFrame
        Hex grid.
    """

    if gdf.crs is None or not gdf.crs.is_projected:
        raise ValueError("GeoDataFrame must be in a projected CRS (meters).")

    xmin, ymin, xmax, ymax = gdf.total_bounds

    r = hex_diameter / 2.0
    h = np.sqrt(3) * r
    dx = 3.0 / 2.0 * r
    dy = h

    hexes = []

    n_cols = int((xmax - xmin) / dx) + 3
    n_rows = int((ymax - ymin) / dy) + 3

    for col in range(n_cols):
        for row in range(n_rows):
            x = xmin + col * dx
            y = ymin + row * dy + (dy / 2 if col % 2 else 0)

            hexes.append(
                Polygon([
                    (x + r * np.cos(theta), y + r * np.sin(theta))
                    for theta in np.linspace(0, 2 * np.pi, 7)[:-1]
                ])
            )

    hex_grid = gpd.GeoDataFrame(
        geometry=hexes,
        crs=gdf.crs,
    )

    # Optional clipping
    if clip_geometry is not None:
        clip_geometry = clip_geometry.to_crs(gdf.crs)
        hex_grid = gpd.clip(hex_grid, clip_geometry)

    # Stable ID
    if return_index:
        hex_grid["hex_id"] = np.arange(len(hex_grid))

    return hex_grid
