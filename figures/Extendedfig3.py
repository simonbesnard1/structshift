#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 16:09:53 2025

@author: simon
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
import matplotlib as mpl
from shapely.geometry import box


params = {
    # font
    'font.family': 'serif',
    # 'font.serif': 'Times', #'cmr10',
    'font.size': 18,
    # axes
    'axes.titlesize': 14,
    'axes.labelsize': 14,
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

# Step 1: Load your disturbance data
use_cols = ['latitude', 'longitude', 'time','harvest', 'wind_bark_beetle', 'forest_fraction']
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

# Calculate pixel area
df['pixel_area_km2'] = AreaGridlatlon(df.latitude, df.longitude, 0.0008888888888888889, 0.0008888888888888889) / 1e6

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

# 2. Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")

# Reproject your data to EPSG:3035
gdf_3035 = gdf.to_crs(epsg=3035)

del df, gdf_pts, pts_sub, joined, polys_sub
import gc; gc.collect()

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

# Create hex grid directly in EPSG:3035
hex_grid = create_true_hex_grid(gdf_3035, hex_diameter=100000)

# Spatial join: assign each point to its hex
gdf_joined = gpd.sjoin(gdf_3035, hex_grid, how="inner", predicate="within")

# 5. Aggregate biomass per hex for each period and disturbance type
km2_to_Mha = 1 / 10_000

def aggregate_by_disturbance(gdf_joined, disturbance_column, start_year, end_year):
    subset = gdf_joined[
        (gdf_joined["year"] >= start_year) &
        (gdf_joined["year"] <= end_year) &
        (gdf_joined['forest_fraction'] >= 0.3) &
        (gdf_joined[disturbance_column] >= 0.5)
    ]
    subset = subset.copy()
    subset['area'] = (
        subset['forest_fraction'] * subset[disturbance_column] * subset["pixel_area_km2"] * km2_to_Mha
    )
    return subset.groupby("index_right")[["area"]].sum()


bm_harvest_early = aggregate_by_disturbance(gdf_joined, "harvest", 2011, 2016)
bm_harvest_late  = aggregate_by_disturbance(gdf_joined, "harvest", 2017, 2023)
bm_bark_early = aggregate_by_disturbance(gdf_joined, "wind_bark_beetle", 2011, 2016)
bm_bark_late  = aggregate_by_disturbance(gdf_joined, "wind_bark_beetle", 2017, 2023)

# 6. Compute deltas and add to hex grid
hex_grid["delta_area_harvest"] = (bm_harvest_late - bm_harvest_early)["area"]
hex_grid["delta_area_bark"] = (bm_bark_late - bm_bark_early)["area"]

# 8. Plot: maps + bar charts
fig, axs = plt.subplots(2, 2, figsize=(8, 7), constrained_layout=True)

europe_gdf.boundary.plot(ax=axs[0,1], linewidth=0.5, color='lightgrey')  # <-- background

# (a) Unplanned disturbance map
hex_grid.dropna(subset=["delta_area_bark"]).plot(
    column="delta_area_bark", ax=axs[0, 1], cmap="RdBu_r", edgecolor="none",
    legend=True, vmin=-0.02, vmax=0.02, legend_kwds={"label": r"$\Delta$Area [Mha]", "shrink": 0.6}
)
axs[0, 1].set_title("Natural Disturbance", fontsize=14)
axs[0, 1].set_xticks([])
axs[0, 1].set_yticks([])
axs[0, 1].set_xlabel("")
axs[0, 1].set_ylabel("")
axs[0, 1].spines['top'].set_visible(False)
axs[0, 1].spines['right'].set_visible(False)
axs[0, 1].spines['bottom'].set_visible(False)
axs[0, 1].spines['left'].set_visible(False)


# (b) Planned disturbance map
europe_gdf.boundary.plot(ax=axs[1,1], linewidth=0.5, color='lightgrey')  # <-- background

hex_grid.dropna(subset=["delta_area_harvest"]).plot(
    column="delta_area_harvest", ax=axs[1, 1], cmap="RdBu_r", edgecolor="none",
    legend=True, vmin=-0.02, vmax=0.02, legend_kwds={"label": r"$\Delta$Area [Mha]", "shrink": 0.6}
)
axs[1, 1].set_title("Harvest", fontsize=14)
axs[1, 1].set_xticks([])
axs[1, 1].set_yticks([])
axs[1, 1].set_xlabel("")
axs[1, 1].set_ylabel("")
axs[1, 1].spines['top'].set_visible(False)
axs[1, 1].spines['right'].set_visible(False)
axs[1, 1].spines['bottom'].set_visible(False)
axs[1, 1].spines['left'].set_visible(False)

# # Bar values
periods = ['2011–2016', '2017–2023', r"$\Delta$(P2-P1) "]
x = np.arange(len(periods))

# # (c) Unplanned total bar plot
early_sum = bm_bark_early['area'].sum()
late_sum = bm_bark_late['area'].sum()
relative_increase_percent = ((late_sum - early_sum) / early_sum) * 100

unplanned_means = [
    early_sum,
    late_sum,
    late_sum - early_sum
]
print(unplanned_means)

axs[0, 0].bar(x, unplanned_means, color=['#a6cee3','#1f78b4','#b2df8a'])
axs[0, 0].set_title("Natural Disturbance", fontsize=14)
axs[0, 0].set_xticks(x)
axs[0, 0].set_xticklabels(periods, fontsize=12)
axs[0, 0].set_ylabel("Total Area Disturbed [Mha]", fontsize=14)
axs[0, 0].spines['top'].set_visible(False)
axs[0, 0].spines['right'].set_visible(False)
early_sum = bm_harvest_early['area'].sum()
late_sum = bm_harvest_late['area'].sum()
relative_increase_percent = ((late_sum - early_sum) / early_sum) * 100
planned_means = [
    early_sum,
    late_sum,
    late_sum - early_sum
]
print(planned_means)



# # (d) Planned total bar plot
axs[1, 0].bar(x, planned_means, color=['#a6cee3','#1f78b4','#b2df8a'])
axs[1, 0].set_title("Harvest", fontsize=14)
axs[1, 0].set_xticks(x)
axs[1, 0].set_xticklabels(periods, fontsize=12)
axs[1, 0].set_ylabel("Total Area Disturbed [Mha]", fontsize=14)
axs[1, 0].spines['top'].set_visible(False)
axs[1, 0].spines['right'].set_visible(False)

# Optional: Add subplot labels (a, b, c, d)
subplot_labels = ['(a)', '(b)', '(c)', '(d)']
for idx, ax in enumerate(axs.flat):
   ax.text(-0.1, 1.05, subplot_labels[idx], transform=ax.transAxes,
           fontsize=16, fontweight='bold', va='bottom')
# Add row-level titles
#fig.text(0.5, 0.96, "Unplanned Disturbance", ha='center', va='center', fontsize=16, fontweight='bold')
#fig.text(0.5, 0.48, "Planned Disturbance", ha='center', va='center', fontsize=16, fontweight='bold')

plt.savefig('/misc/glm1/person/besnard/coupling_demography_dist/figs/Extended_fig3.png', dpi=300)
