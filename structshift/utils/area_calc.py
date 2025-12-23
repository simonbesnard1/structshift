import numpy as np

def pixel_area_latlon_km2(lat, res_lat, res_lon):
    """
    Compute pixel area (km²) on a lat/lon grid.
    """
    ER = 6_378_160  # Earth radius (m)
    lats1 = lat - res_lat / 2
    lats2 = lat + res_lat / 2
    area = (
        (np.pi / 180) * ER**2
        * np.abs(np.sin(np.deg2rad(lats1)) - np.sin(np.deg2rad(lats2)))
        * np.abs(res_lon)
    )
    return area / 1e6
