#%% Load library
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, box

params = {
    # font
    'font.family': 'serif',
    # 'font.serif': 'Times', #'cmr10',
    'font.size': 16,
    # axes
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'axes.linewidth': 0.5,
    # ticks
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'xtick.major.width': 0.3,
    'ytick.major.width': 0.3,
    'xtick.minor.width': 0.3,
    'ytick.minor.width': 0.3,
    # legend
    'legend.fontsize': 14,
    # tex
    'text.usetex': True,
}

mpl.rcParams.update(params)

def AreaGridlatlon(lats,lons,res_lat,res_lon):
     ER          = 6378160 #Earth radius (m)
     londel      = np.abs(res_lon)
     lats1       = lats - res_lat/2.
     lats2       = lats + res_lat/2.
     areavec     = (np.pi/180)*ER**2 * np.abs(np.sin(lats1 * np.pi/180)-np.sin(lats2 * np.pi/180))*londel
     
     return areavec

def create_true_hex_grid(gdf, hex_diameter=200000):  # 20km in meters
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
                           for theta in np.linspace(0, 2 * np.pi, 7)[:-1]])
            hexes.append(hex)
    return gpd.GeoDataFrame({'geometry': hexes}, crs="EPSG:3035")

# Load and preprocess
use_cols = ['latitude', 'longitude', 'time', 'genus', 'harvest', 'wind_bark_beetle', 'forest_fraction'] + [f'biomass_m{i}' for i in range(20)]
data_= pd.read_parquet('/misc/glm1/person/besnard/coupling_demography_dist/data/disturbance_data_combined_v2025-11-1.parquet', columns=use_cols)
data_['time'] = pd.to_datetime(data_['time'])
data_["year"] = pd.to_datetime(data_["time"]).dt.year
data_ = data_[data_["year"] >= 2011]
data_ = data_[~(((data_['year'].isin([2018, 2023])) & (data_['latitude'] >= 65)))]

# Load polygons from GeoPackage
polys = gpd.read_file("/misc/glm1/person/besnard/coupling_demography_dist/data/bounding_box_filter_years_4326.gpkg")
if polys.crs is None:
    polys = polys.set_crs("EPSG:4326")
else:
    polys = polys.to_crs("EPSG:4326")

gdf_pts = gpd.GeoDataFrame(
    data_.copy(),
    geometry=gpd.points_from_xy(data_["longitude"], data_["latitude"]),
    crs="EPSG:4326"
)

years_in_polys = polys["year"].unique()
pts_sub = gdf_pts[gdf_pts["year"].isin(years_in_polys)][["geometry", "year"]]
polys_sub = polys[["geometry", "year"]]

joined = gpd.sjoin(pts_sub, polys_sub, how="left", predicate="within")
to_drop_idx = joined.index[
    joined["index_right"].notna() & (joined["year_left"] == joined["year_right"])
]

data_ = data_.drop(index=to_drop_idx).reset_index(drop=True)

data_ = data_[data_["forest_fraction"] >= 0.3]
biomass_cols = [col for col in data_.columns if col.startswith("biomass")]
data_[biomass_cols] = data_[biomass_cols].where(data_[biomass_cols] > 0)
for i in range(20):
    biomass_col = f"biomass_m{i}"
    data_[biomass_col] = data_[biomass_col] * 0.47
data_["biomass"] = data_[biomass_cols].mean(axis=1, skipna=True)

# Calculate pixel area
data_['pixel_area_km2'] = AreaGridlatlon(data_.latitude, data_.longitude, 0.0008888888888888889, 0.0008888888888888889) / 1e6
 

world = gpd.read_file('/misc/glm1/person/besnard/coupling_demography_dist/data/ne_10m_admin_0_countries.zip')
europe_gdf = world[world['CONTINENT'] == 'Europe'].to_crs("EPSG:4326")
bbox = box(-20, 32, 45, 71)

# Ensure the bounding box and europe_gdf are both in EPSG:3035
bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs="EPSG:4326")
bbox_3035 = bbox_gdf.to_crs("EPSG:3035")
europe_gdf = europe_gdf.to_crs("EPSG:3035")

# Clip using reprojected bounding box
europe_gdf = gpd.clip(europe_gdf, bbox_3035)

# Exclude unwanted countries
exclude_countries = ['RUS', 'ISL']
europe_gdf = europe_gdf[~europe_gdf['ISO_A3'].isin(exclude_countries)]

genus_groups = {
    "Spruce": [1],
    "Other needleleaf": [0, 2, 5],  # Larix, Pinus, Other needleleaf
    "Broadleaf": [3, 4, 6]          # Fagus, Quercus, Other broadleaf
}

# 2. Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(data_, geometry=gpd.points_from_xy(data_.longitude, data_.latitude), crs="EPSG:4326")

# Reproject your data to EPSG:3035
gdf = gdf.to_crs(epsg=3035)

del data_, gdf_pts, pts_sub, joined, polys_sub
import gc; gc.collect()

# 3. Create hex grid
hex_grid = create_true_hex_grid(gdf, hex_diameter=100000)

# 4. Spatial join
gdf_joined = gpd.sjoin(gdf, hex_grid, how="inner", predicate="within")

def delta_biomass_by_genus(gdf_joined, disturbance_column, period1, period2, genus_ids, min_pixels=10):
    delta_biomass = {}
    
    # Filter once per period
    df1 = gdf_joined[
        (gdf_joined['genus'].isin(genus_ids)) &
        (gdf_joined["year"].between(*period1)) &
        (gdf_joined['forest_fraction'] >= 0.3) &
        (gdf_joined[disturbance_column] >= 0.5)
    ][["index_right", "biomass"]].copy()
    df2 = gdf_joined[
        (gdf_joined['genus'].isin(genus_ids)) &
        (gdf_joined["year"].between(*period2)) &
        (gdf_joined['forest_fraction'] >= 0.3) &
        (gdf_joined[disturbance_column] >= 0.5)
    ][["index_right", "biomass"]].copy()
    
    # Loop over all hexes that appear in either period
    all_hex_ids = set(df1["index_right"]).union(df2["index_right"])
    
    for hex_id in all_hex_ids:
        a = df1[df1["index_right"] == hex_id]["biomass"].dropna().values
        b = df2[df2["index_right"] == hex_id]["biomass"].dropna().values
      
        if len(a) >= min_pixels and len(b) >= min_pixels:
            delta_biomass[hex_id] = np.nanmean(b) - np.nanmean(a) 
        else:
            delta_biomass[hex_id] = np.nan  # Not enough data for comparison

    return pd.Series(delta_biomass)


bm_harvest_spruce = delta_biomass_by_genus(gdf_joined, "harvest", (2011, 2016), (2017, 2023), genus_ids= genus_groups['Spruce'], min_pixels=25)
bm_harvest_broadleaf  = delta_biomass_by_genus(gdf_joined, "harvest", (2011, 2016), (2017, 2023),  genus_ids= genus_groups['Broadleaf'], min_pixels=25)
bm_harvest_needleleaf  = delta_biomass_by_genus(gdf_joined, "harvest", (2011, 2016), (2017, 2023), genus_ids= genus_groups["Other needleleaf"], min_pixels=25)

bm_bark_spruce = delta_biomass_by_genus(gdf_joined, "wind_bark_beetle", (2011, 2016), (2017, 2023), genus_ids= genus_groups['Spruce'], min_pixels=25)
bm_bark_broadleaf  = delta_biomass_by_genus(gdf_joined, "wind_bark_beetle", (2011, 2016), (2017, 2023),  genus_ids= genus_groups['Broadleaf'], min_pixels=25)
bm_bark_needleleaf  = delta_biomass_by_genus(gdf_joined, "wind_bark_beetle", (2011, 2016), (2017, 2023), genus_ids= genus_groups["Other needleleaf"], min_pixels=25)

hex_grid["bm_harvest_spruce"] = bm_harvest_spruce
hex_grid["bm_harvest_broadleaf"] = bm_harvest_broadleaf
hex_grid["bm_harvest_needleleaf"] = bm_harvest_needleleaf
hex_grid["bm_bark_spruce"] = bm_bark_spruce
hex_grid["bm_bark_broadleaf"] = bm_bark_broadleaf
hex_grid["bm_bark_needleleaf"] = bm_bark_needleleaf

#%% --- Plot Map of Expansion Hexagons ---

# Define map details
columns = [
    "bm_bark_spruce", "bm_bark_broadleaf", "bm_bark_needleleaf",
    "bm_harvest_spruce", "bm_harvest_broadleaf", "bm_harvest_needleleaf",
    
]
titles = [
    "Natural Disturbance – Spruce", "Natural Disturbance – Broadleaf", "Natural Disturbance – Other Needleleaf",
    "Harvest – Spruce", "Harvest – Broadleaf", "Harvest – Other Needleleaf",
]

# Create subplots
fig, axs = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

for ax, col, title in zip(axs.flat, columns, titles):
    # Plot background
    europe_gdf.boundary.plot(ax=ax, linewidth=0.5, color='lightgrey')

    # Plot hex grid values
    hex_grid.dropna(subset=[col]).plot(
        column=col,
        ax=ax,
        cmap="RdBu_r",
        edgecolor="none",
        vmin=-20, vmax=20,
        legend=True if col == "bm_bark_spruce" else False,  # Legend on first plot only
        legend_kwds={
            "label": r"$\Delta$Biomass [MgC ha$^{-1}$]",
            "shrink": 0.6
        } if col == "bm_bark_spruce" else None
    )

    ax.set_title(title, fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
        
subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
for idx, ax in enumerate(axs.flat):
   ax.text(-0.1, 1.05, subplot_labels[idx], transform=ax.transAxes,
           fontsize=18, fontweight='bold', va='bottom')

plt.savefig('/misc/glm1/person/besnard/coupling_demography_dist/figs/Extended_fig5.png', dpi=300)
