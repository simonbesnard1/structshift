import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

def AreaGridlatlon(lats,lons,res_lat,res_lon):
     ER          = 6378160 #Earth radius (m)
     londel      = np.abs(res_lon)
     lats1       = lats - res_lat/2.
     lats2       = lats + res_lat/2.
     areavec     = (np.pi/180)*ER**2 * np.abs(np.sin(lats1 * np.pi/180)-np.sin(lats2 * np.pi/180))*londel
     
     return areavec

# Step 1: Load your disturbance data
use_cols = ['latitude', 'longitude', 'time','harvest', 'wind_bark_beetle', 'forest_fraction'] + [f'forest_age_gami_2010_m{i}' for i in range(20)]
df= pd.read_parquet('/misc/glm1/person/besnard/coupling_demography_dist/data/disturbance_data_combined_v2025-11-1.parquet', columns=use_cols)
df['time'] = pd.to_datetime(df['time'])
df["year"] = pd.to_datetime(df["time"]).dt.year
df = df[df["year"] >= 2011]

# Filter out points in 2018 or 2023 above 67°N
df = df[~(((df['year'].isin([2018, 2023])) & (df['latitude'] >= 65)))]

# Load polygons from GeoPackage
polys = gpd.read_file("/misc/glm1/person/besnard/coupling_demography_dist/data/bounding_box_filter_years_4326.gpkg")
if polys.crs is None:
    polys = polys.set_crs("EPSG:4326")
else:
    polys = polys.to_crs("EPSG:4326")

gdf_pts = gpd.GeoDataFrame(
    df.copy(),
    geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
    crs="EPSG:4326"
)

years_in_polys = polys["year"].unique()
pts_sub = gdf_pts[gdf_pts["year"].isin(years_in_polys)][["geometry", "year"]]
polys_sub = polys[["geometry", "year"]]

joined = gpd.sjoin(pts_sub, polys_sub, how="left", predicate="within")
to_drop_idx = joined.index[
    joined["index_right"].notna() & (joined["year_left"] == joined["year_right"])
]

df = df.drop(index=to_drop_idx).reset_index(drop=True)

forest_age_cols = [col for col in df.columns if col.startswith("forest_age")]
df[forest_age_cols] = df[forest_age_cols].where(df[forest_age_cols] > 0)
df["forest_age"] = df[forest_age_cols].median(axis=1, skipna=True)
df['pixel_area_km2'] = AreaGridlatlon(df.latitude, df.longitude, 0.0008888888888888889, 0.0008888888888888889) / 1e6

# 2. Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")

# 3. Create hex grid
def create_true_hex_grid(gdf, hex_diameter=1.0):
    xmin, ymin, xmax, ymax = gdf.total_bounds
    hex_radius = hex_diameter / 2
    hex_height = np.sqrt(3) * hex_radius
    dx = 3/2 * hex_radius
    dy = hex_height
    cols = int((xmax - xmin) / dx) + 2
    rows = int((ymax - ymin) / dy) + 2
    hexes = []
    for row in range(rows):
        for col in range(cols):
            x = xmin + col * dx
            y = ymin + row * dy
            if col % 2 == 1:
                y += dy / 2
            hex = Polygon([(x + hex_radius * np.cos(theta), y + hex_radius * np.sin(theta))
                           for theta in np.linspace(0, 2*np.pi, 7)[:-1]])
            hexes.append(hex)
    return gpd.GeoDataFrame({'geometry': hexes}, crs="EPSG:4326")

hex_grid = create_true_hex_grid(gdf, hex_diameter=1.0)

# 4. Spatial join
gdf_joined = gpd.sjoin(gdf, hex_grid, how="inner", predicate="within")

# 5. Aggregate biomass per hex for each period and disturbance type
km2_to_Mha = 1 / 10_000

def area_by_age_class(gdf_joined, disturbance_column, start_year, end_year, age_min, age_max):
    subset = gdf_joined[
        (gdf_joined["year"] >= start_year) &
        (gdf_joined["year"] <= end_year) &
        (gdf_joined['forest_fraction'] >= 0.3) &
        (gdf_joined[disturbance_column] >= 0.5) &
        (gdf_joined["forest_age"] > age_min) &
        (gdf_joined["forest_age"] <= age_max)
    ].copy()

    subset['area'] = (
        subset['forest_fraction'] * subset[disturbance_column] * subset["pixel_area_km2"] * km2_to_Mha
    )

    return subset.groupby("index_right")[["area"]].sum()

young_harvest_early = area_by_age_class(gdf_joined, "harvest", 2011, 2016, 0, 60)
young_harvest_late  = area_by_age_class(gdf_joined, "harvest", 2017, 2023, 0, 60)
young_bark_early = area_by_age_class(gdf_joined, "wind_bark_beetle", 2011, 2016, 0, 60)
young_bark_late  = area_by_age_class(gdf_joined, "wind_bark_beetle", 2017, 2023, 0, 60)
old_harvest_early = area_by_age_class(gdf_joined, "harvest", 2011, 2016, 61, 300)
old_harvest_late  = area_by_age_class(gdf_joined, "harvest", 2017, 2023, 61, 300)
old_bark_early = area_by_age_class(gdf_joined, "wind_bark_beetle", 2011, 2016, 61, 300)
old_bark_late  = area_by_age_class(gdf_joined, "wind_bark_beetle", 2017, 2023, 61, 300)

# Creating a dictionary where each key corresponds to the total area value from the DataFrame
summary_dict = {
    'young_harvest_early': young_harvest_early['area'].sum(),
    'young_harvest_late': young_harvest_late['area'].sum(),
    'young_bark_early': young_bark_early['area'].sum(),
    'young_bark_late': young_bark_late['area'].sum(),
    'old_harvest_early': old_harvest_early['area'].sum(),
    'old_harvest_late': old_harvest_late['area'].sum(),
    'old_bark_early': old_bark_early['area'].sum(),
    'old_bark_late': old_bark_late['area'].sum(),
}

# Convert dictionary to DataFrame
summary_df = pd.DataFrame.from_dict(summary_dict, orient='index', columns=['total_area_Mha'])

# Save to CSV
csv_path = "/misc/glm1/person/besnard/coupling_demography_dist/figs/area_by_age_class_totals.csv"
summary_df.to_csv(csv_path)

