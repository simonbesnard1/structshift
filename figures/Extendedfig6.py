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
import matplotlib as mpl
from scipy.stats import linregress
from shapely.geometry import Polygon, box
from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr

# --- Plot Settings ---
params = {
    'font.family': 'serif',
    'font.size': 18,
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'axes.linewidth': 0.5,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'text.usetex': True,
}
mpl.rcParams.update(params)


# --- Load Data ---
use_cols = ['latitude', 'longitude', 'time', 'harvest', 'wind_bark_beetle', 'forest_fraction', 'genus'] + [f'biomass_m{i}' for i in range(20)]
df= pd.read_parquet('/misc/glm1/person/besnard/coupling_demography_dist/data/disturbance_data_combined_v2025-11-1.parquet', columns=use_cols)

df['time'] = pd.to_datetime(df['time'])
df['year'] = df['time'].dt.year
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

biomass_cols = [col for col in df.columns if col.startswith("biomass")]
df[biomass_cols] = df[biomass_cols].where(df[biomass_cols] > 0)
for i in range(20):
    biomass_col = f"biomass_m{i}"
    df[biomass_col] = df[biomass_col] * 0.47

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

# --- GeoDataFrame Setup ---
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")

# Reproject your data to EPSG:3035
gdf_3035 = gdf.to_crs(epsg=3035)

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
gdf_joined = gpd.sjoin(gdf_3035, hex_grid, how="inner", predicate="within") 

def annual_cv_summary_across_genus(gdf, disturbance_column, biomass_cols, genus_ids, min_pixels=50):
    summary_records = []

    for year in range(2011, 2024):
        df_year = gdf[
            (gdf['genus'].isin(genus_ids)) &
            (gdf["year"] == year) &
            (gdf["forest_fraction"] >= 0.3) &
            (gdf[disturbance_column] >= 0.5)
        ]

        medians_per_member = []
        for member in biomass_cols:
            grouped = df_year.groupby("index_right")[member]
            count = grouped.count()
            valid = count[count >= min_pixels].index

            cv = (grouped.std() / grouped.mean())[valid]
            median_cv = cv.median()
            if not np.isnan(median_cv):
                medians_per_member.append(median_cv)

        if medians_per_member:
            medians_per_member = np.array(medians_per_member)
            summary_records.append({
                "year": year,
                "disturbance": disturbance_column,
                "cv_median": np.median(medians_per_member),
                "cv_q5": np.quantile(medians_per_member, 0.05),
                "cv_q95": np.quantile(medians_per_member, 0.95)
            })

    return pd.DataFrame(summary_records)

def annual_cv_summary(gdf, disturbance_column, biomass_cols, min_pixels=50):
    summary_records = []

    for year in range(2011, 2024):
        df_year = gdf[
            (gdf["year"] == year) &
            (gdf["forest_fraction"] >= 0.3) &
            (gdf[disturbance_column] >= 0.5)
        ]

        medians_per_member = []
        for member in biomass_cols:
            grouped = df_year.groupby("index_right")[member]
            count = grouped.count()
            valid = count[count >= min_pixels].index

            cv = (grouped.std() / grouped.mean())[valid]
            median_cv = cv.median()
            if not np.isnan(median_cv):
                medians_per_member.append(median_cv)

        if medians_per_member:
            medians_per_member = np.array(medians_per_member)
            summary_records.append({
                "year": year,
                "disturbance": disturbance_column,
                "cv_median": np.median(medians_per_member),
                "cv_q5": np.quantile(medians_per_member, 0.05),
                "cv_q95": np.quantile(medians_per_member, 0.95)
            })

    return pd.DataFrame(summary_records)


genus_groups = {
    "Spruce": [1],
    "Other needleleaf": [0, 2, 5],  # Larix, Pinus, Other needleleaf
    "Broadleaf": [3, 4, 6]          # Fagus, Quercus, Other broadleaf
}

# --- time series of CV ---
df_cv_unplanned_broadleaf = annual_cv_summary_across_genus(gdf_joined, "wind_bark_beetle", biomass_cols, genus_groups['Broadleaf'], min_pixels=25)
df_cv_planned_broadleaf = annual_cv_summary_across_genus(gdf_joined, "harvest", biomass_cols, genus_groups['Broadleaf'], min_pixels=25)

df_cv_unplanned_spruce = annual_cv_summary(gdf_joined, "wind_bark_beetle", biomass_cols, min_pixels=25)
df_cv_planned_spruce = annual_cv_summary(gdf_joined, "harvest", biomass_cols, min_pixels=25)

df_cv_unplanned_needleleaf = annual_cv_summary_across_genus(gdf_joined, "wind_bark_beetle", biomass_cols, genus_groups["Other needleleaf"], min_pixels=25)
df_cv_planned_needleleaf = annual_cv_summary_across_genus(gdf_joined, "harvest", biomass_cols, genus_groups["Other needleleaf"], min_pixels=25)


#%% --- Plot Map of Expansion Hexagons ---
fig, axs = plt.subplots(
    2, 2,
    figsize=(12, 9),
    gridspec_kw={"wspace": 0},  # Increase horizontal spacing (default is ~0.2)
    constrained_layout=True
)

# (d) Time series of CV biomass

# Regression for unplanned
slope_unplanned, intercept_unplanned, r_value_u, p_value_u, std_err_u = linregress(df_cv_unplanned_spruce["year"].values, df_cv_unplanned_spruce["cv_median"].values)

# Regression for planned
slope_planned, intercept_planned, r_value_p, p_value_p, std_err_p = linregress(df_cv_planned_spruce["year"].values, df_cv_planned_spruce["cv_median"].values)

# Plot unplanned disturbance trend
axs[0, 0].plot(df_cv_unplanned_spruce["year"], df_cv_unplanned_spruce["cv_median"], label="Natural Disturbance", color="#1b9e77", lw=2)
axs[0, 0].fill_between(df_cv_unplanned_spruce["year"],
                       df_cv_unplanned_spruce["cv_q5"],
                       df_cv_unplanned_spruce["cv_q95"],
                       color="#1b9e77", alpha=0.3)

# Plot planned disturbance trend
axs[0, 0].plot(df_cv_planned_spruce["year"], df_cv_planned_spruce["cv_median"], label="Harvest", color="black", lw=2)
axs[0, 0].fill_between(df_cv_planned_spruce["year"],
                       df_cv_planned_spruce["cv_q5"],
                       df_cv_planned_spruce["cv_q95"],
                       color="black", alpha=0.2)

# Labels and legend
axs[0, 0].set_xlabel("Years")
axs[0, 0].set_ylabel("CV Biomass [adimensional]")
axs[0, 0].set_title("All Species", fontsize=14)
axs[0, 0].legend(frameon=False)
axs[0, 0].spines['top'].set_visible(False)
axs[0, 0].spines['right'].set_visible(False)
axs[0, 0].set_ylim(0.2, 0.6)
axs[0, 0].xaxis.set_major_locator(MaxNLocator(integer=True))

# --- Unplanned regression line ---
x_unplanned = df_cv_unplanned_spruce["year"].values
y_unplanned = df_cv_unplanned_spruce["cv_median"].values
slope_unplanned, intercept_unplanned = np.polyfit(x_unplanned, y_unplanned, 1)
axs[0, 0].plot(x_unplanned, intercept_unplanned + slope_unplanned * x_unplanned,
               color="#1b9e77", linestyle="--", linewidth=1.8)

# --- Planned regression line ---
x_planned = df_cv_planned_spruce["year"].values
y_planned = df_cv_planned_spruce["cv_median"].values
slope_planned, intercept_planned = np.polyfit(x_planned, y_planned, 1)
axs[0, 0].plot(x_planned, intercept_planned + slope_planned * x_planned,
               color="black", linestyle="--", linewidth=1.8)

# Assuming y_unplanned and y_planned are 1D NumPy arrays of equal length
r_value, p_value = pearsonr(y_unplanned, y_planned)

print(f"Pearson correlation coefficient: {r_value:.3f}")
print(f"P-value: {p_value:.3e}")

# --- Add statistical annotations ---

# Annotation for unplanned
annotation_unplanned = (
    f"Slope = {slope_unplanned:.4f}\n"
    f"p = {p_value_u:.3f}"
)
axs[0, 0].text(
    0.03, 0.25, annotation_unplanned,
    transform=axs[0, 0].transAxes,
    fontsize=13, color="#1b9e77",
    verticalalignment='top',
    horizontalalignment='left',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="#1b9e77", alpha=0.6)
)

# Annotation for planned
annotation_planned = (
    f"Slope = {slope_planned:.4f}\n"
    f"p = {p_value_p:.3f}"
)
axs[0, 0].text(
    0.03, 0.97, annotation_planned,
    transform=axs[0, 0].transAxes,
    fontsize=13, color="black",
    verticalalignment='top',
    horizontalalignment='left',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.6)
)


# Regression for unplanned
slope_unplanned, intercept_unplanned, r_value_u, p_value_u, std_err_u = linregress(df_cv_unplanned_broadleaf["year"].values, df_cv_unplanned_broadleaf["cv_median"].values)

# Regression for planned
slope_planned, intercept_planned, r_value_p, p_value_p, std_err_p = linregress(df_cv_planned_broadleaf["year"].values, df_cv_planned_broadleaf["cv_median"].values)

# Plot unplanned disturbance trend
axs[0, 1].plot(df_cv_unplanned_broadleaf["year"], df_cv_unplanned_broadleaf["cv_median"], label="Natural Disturbance", color="#1b9e77", lw=2)
axs[0, 1].fill_between(df_cv_unplanned_broadleaf["year"],
                       df_cv_unplanned_broadleaf["cv_q5"],
                       df_cv_unplanned_broadleaf["cv_q95"],
                       color="#1b9e77", alpha=0.3)

# Plot planned disturbance trend
axs[0, 1].plot(df_cv_planned_broadleaf["year"], df_cv_planned_broadleaf["cv_median"], label="Harvest", color="black", lw=2)
axs[0, 1].fill_between(df_cv_planned_broadleaf["year"],
                       df_cv_planned_broadleaf["cv_q5"],
                       df_cv_planned_broadleaf["cv_q95"],
                       color="black", alpha=0.2)

# Labels and legend
axs[0, 1].set_xlabel("Years")
axs[0, 1].set_ylabel("CV Biomass [adimensional]")
axs[0, 1].set_title("Broadleaf", fontsize=14)
axs[0, 1].legend(frameon=False)
axs[0, 1].spines['top'].set_visible(False)
axs[0, 1].spines['right'].set_visible(False)
axs[0, 1].set_ylim(0.25, 0.6)
axs[0, 1].xaxis.set_major_locator(MaxNLocator(integer=True))

# --- Unplanned regression line ---
x_unplanned = df_cv_unplanned_broadleaf["year"].values
y_unplanned = df_cv_unplanned_broadleaf["cv_median"].values
slope_unplanned, intercept_unplanned = np.polyfit(x_unplanned, y_unplanned, 1)
axs[0, 1].plot(x_unplanned, intercept_unplanned + slope_unplanned * x_unplanned,
               color="#1b9e77", linestyle="--", linewidth=1.8)

# --- Planned regression line ---
x_planned = df_cv_planned_broadleaf["year"].values
y_planned = df_cv_planned_broadleaf["cv_median"].values
slope_planned, intercept_planned = np.polyfit(x_planned, y_planned, 1)
axs[0, 1].plot(x_planned, intercept_planned + slope_planned * x_planned,
               color="black", linestyle="--", linewidth=1.8)

# Assuming y_unplanned and y_planned are 1D NumPy arrays of equal length
r_value, p_value = pearsonr(y_unplanned, y_planned)

print(f"Pearson correlation coefficient: {r_value:.3f}")
print(f"P-value: {p_value:.3e}")

# --- Add statistical annotations ---

# Annotation for unplanned
annotation_unplanned = (
    f"Slope = {slope_unplanned:.4f}\n"
    f"p = {p_value_u:.3f}"
)
axs[0, 1].text(
    0.03, 0.25, annotation_unplanned,
    transform=axs[0, 1].transAxes,
    fontsize=13, color="#1b9e77",
    verticalalignment='top',
    horizontalalignment='left',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="#1b9e77", alpha=0.6)
)

# Annotation for planned
annotation_planned = (
    f"Slope = {slope_planned:.4f}\n"
    f"p = {p_value_p:.3f}"
)
axs[0, 1].text(
    0.03, 0.97, annotation_planned,
    transform=axs[0, 1].transAxes,
    fontsize=13, color="black",
    verticalalignment='top',
    horizontalalignment='left',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.6)
)


# Regression for unplanned
slope_unplanned, intercept_unplanned, r_value_u, p_value_u, std_err_u = linregress(df_cv_unplanned_needleleaf["year"].values, df_cv_unplanned_needleleaf["cv_median"].values)

# Regression for planned
slope_planned, intercept_planned, r_value_p, p_value_p, std_err_p = linregress(df_cv_planned_needleleaf["year"].values, df_cv_planned_needleleaf["cv_median"].values)

# Plot unplanned disturbance trend
axs[1, 0].plot(df_cv_unplanned_needleleaf["year"], df_cv_unplanned_needleleaf["cv_median"], label="Natural Disturbance", color="#1b9e77", lw=2)
axs[1, 0].fill_between(df_cv_unplanned_needleleaf["year"],
                       df_cv_unplanned_needleleaf["cv_q5"],
                       df_cv_unplanned_needleleaf["cv_q95"],
                       color="#1b9e77", alpha=0.3)

# Plot planned disturbance trend
axs[1, 0].plot(df_cv_planned_needleleaf["year"], df_cv_planned_needleleaf["cv_median"], label="Harvest", color="black", lw=2)
axs[1, 0].fill_between(df_cv_planned_needleleaf["year"],
                       df_cv_planned_needleleaf["cv_q5"],
                       df_cv_planned_needleleaf["cv_q95"],
                       color="black", alpha=0.2)

# Labels and legend
axs[1, 0].set_xlabel("Years")
axs[1, 0].set_ylabel("CV Biomass [adimensional]")
axs[1, 0].set_title("Other Needleleaf", fontsize=14)
axs[1, 0].legend(frameon=False)
axs[1, 0].spines['top'].set_visible(False)
axs[1, 0].spines['right'].set_visible(False)
axs[1, 0].set_ylim(0.2, 0.55)
axs[1, 0].xaxis.set_major_locator(MaxNLocator(integer=True))

# --- Unplanned regression line ---
x_unplanned = df_cv_unplanned_needleleaf["year"].values
y_unplanned = df_cv_unplanned_needleleaf["cv_median"].values
slope_unplanned, intercept_unplanned = np.polyfit(x_unplanned, y_unplanned, 1)
axs[1, 0].plot(x_unplanned, intercept_unplanned + slope_unplanned * x_unplanned,
               color="#1b9e77", linestyle="--", linewidth=1.8)

# --- Planned regression line ---
x_planned = df_cv_planned_needleleaf["year"].values
y_planned = df_cv_planned_needleleaf["cv_median"].values
slope_planned, intercept_planned = np.polyfit(x_planned, y_planned, 1)
axs[1, 0].plot(x_planned, intercept_planned + slope_planned * x_planned,
               color="black", linestyle="--", linewidth=1.8)

# Assuming y_unplanned and y_planned are 1D NumPy arrays of equal length
r_value, p_value = pearsonr(y_unplanned, y_planned)

print(f"Pearson correlation coefficient: {r_value:.3f}")
print(f"P-value: {p_value:.3e}")

# --- Add statistical annotations ---

# Annotation for unplanned
annotation_unplanned = (
    f"Slope = {slope_unplanned:.4f}\n"
    f"p = {p_value_u:.3f}"
)
axs[1, 0].text(
    0.03, 0.25, annotation_unplanned,
    transform=axs[1, 0].transAxes,
    fontsize=13, color="#1b9e77",
    verticalalignment='top',
    horizontalalignment='left',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="#1b9e77", alpha=0.6)
)

# Annotation for planned
annotation_planned = (
    f"Slope = {slope_planned:.4f}\n"
    f"p = {p_value_p:.3f}"
)
axs[1, 0].text(
    0.03, 0.97, annotation_planned,
    transform=axs[1, 0].transAxes,
    fontsize=13, color="black",
    verticalalignment='top',
    horizontalalignment='left',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.6)
)

subplot_labels = ['(a)', '(b)', '(c)', '(d)']
for idx, ax in enumerate(axs.flat):
   ax.text(-0.1, 1.05, subplot_labels[idx], transform=ax.transAxes,
           fontsize=18, fontweight='bold', va='bottom')
fig.delaxes(axs[1,1])
plt.savefig('/misc/glm1/person/besnard/coupling_demography_dist/figs/Extended_fig6.png', dpi=300)
