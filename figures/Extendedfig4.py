#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 15:22:16 2023

@author: simon
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pandas as pd
import geopandas as gpd

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

# Step 1: Load your disturbance data
use_cols = ['latitude', 'longitude', 'time','harvest', 'wind_bark_beetle', 'country', 'forest_fraction'] + [f'forest_age_gami_2010_m{i}' for i in range(20)]
df= pd.read_parquet('/misc/glm1/person/besnard/coupling_demography_dist/data/disturbance_data_combined_v2025-11-1.parquet', columns=use_cols)
df['time'] = pd.to_datetime(df['time'])
df["year"] = df["time"].dt.year
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

forest_age_cols = [col for col in df.columns if col.startswith("forest_age_gami_2010")]
df[forest_age_cols] = df[forest_age_cols].where(df[forest_age_cols] > 0)
df["forest_age"] = df[forest_age_cols].median(axis=1, skipna=True) 
df = df.dropna()
region_map = {
    "Northern Europe": [
        "Finland", "Sweden", "Norway", "Denmark", "Estonia", "Latvia", "Lithuania"
    ],
    "Western Europe": [
        "United Kingdom", "Ireland", "France", "Belgium", "Netherlands", "Luxembourg"
    ],
    "Central Europe": [
        "Germany", "Switzerland", "Austria", "Czechia", "Poland"
    ],
    "Eastern and Southeastern Europe": [
        "Hungary", "Slovakia", "Slovenia", "Ukraine", "Turkey",
        "Croatia", "Bosnia and Herzegovina", "Serbia", "Montenegro",
        "Kosovo", "Albania", "North Macedonia", "Moldova", "Bulgaria", "Romania"
    ]
}

#%% Create 2D heatmaps of age vs biomass bins per disturbance type
disturbance_types = {'Natural Disturbance': 'wind_bark_beetle','Harvest':'harvest'}

#%% analysis per region
fig, axes = plt.subplots(4, 2, figsize=(12, 13), constrained_layout=True)
for region_idx, region_name in enumerate(region_map):
    
    region_data = df[df['country'].isin(region_map[region_name])]
    
    # Define the years of interest
    years = range(2011, 2024)
    
    # Defining age class boundaries
    age_classes = {
        '1-60': (1, 60),
        '$>$60': (61, np.inf)
    }
    
    # Collect results per model variant
    fractions_per_year_age_all_models = {
        i: {year: {disturbance: {age_class: 0 for age_class in age_classes} 
            for disturbance in disturbance_types} for year in years} for i in range(20)
    }
    
    # Calculate fractions for each year, disturbance type, and age class
    for i in range(20):
        age_col = f"forest_age_gami_2010_m{i}"
    
        for year in years:
            for disturbance_name, disturbance_column in disturbance_types.items():
                subset = region_data[(region_data["year"] == year) & (region_data['forest_fraction'] >= 0.3) & (region_data[disturbance_column] >= 0.5) & (region_data[age_col] > 0)]
                total_pixels = len(subset)
    
                if total_pixels > 0:
                    for age_class_name, (age_min, age_max) in age_classes.items():
                        age_mask = (subset[age_col] >= age_min) & (subset[age_col] <= age_max)
                        fraction = age_mask.sum() / total_pixels
                        fractions_per_year_age_all_models[i][year][disturbance_name][age_class_name] = fraction
    
    # Initialize structure to hold summary stats
    fractions_summary = {
        year: {
            disturbance: {
                age_class: {"median": 0, "p5": 0, "p95": 0}
                for age_class in age_classes
            } for disturbance in disturbance_types
        } for year in years
    }
    
    # Aggregate
    for year in years:
        for disturbance in disturbance_types:
            for age_class in age_classes:
                vals = [
                    fractions_per_year_age_all_models[i][year][disturbance][age_class]
                    for i in range(20)
                ]
                fractions_summary[year][disturbance][age_class]["median"] = np.median(vals)
                fractions_summary[year][disturbance][age_class]["p5"] = np.percentile(vals, 5)
                fractions_summary[year][disturbance][age_class]["p95"] = np.percentile(vals, 95)
    
    
    # Remaining plots: Line Plots for Each Disturbance Type Over Time
    age_class_colors = {
        '1-60': '#fc8d62',
        '$>$60': '#8da0cb'
    }
    
    for col_idx, disturbance in enumerate(disturbance_types):
        ax = axes[region_idx, col_idx]
        
        for age_class in age_classes:
            color = age_class_colors[age_class]
            median_vals = [fractions_summary[year][disturbance][age_class]["median"] for year in years]
            p5_vals = [fractions_summary[year][disturbance][age_class]["p5"] for year in years]
            p95_vals = [fractions_summary[year][disturbance][age_class]["p95"] for year in years]
    
            # Optional: smoothing
            median_vals = pd.Series(median_vals).rolling(3, center=True, min_periods=1).mean()
            p5_vals = pd.Series(p5_vals).rolling(3, center=True, min_periods=1).mean()
            p95_vals = pd.Series(p95_vals).rolling(3, center=True, min_periods=1).mean()
    
            ax.plot(years, median_vals, marker='o', label=age_class, color=color)
            ax.fill_between(years, p5_vals, p95_vals, alpha=0.2, color=color)
    
        ax.set_title(f"{region_name} - {disturbance}", fontsize=16, fontweight='bold')
        ax.set_ylabel("Fraction [adimensional]", fontsize=14)
        ax.set_ylim(0, 1)
        ax.legend(title="Age class", frameon=False, loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
for idx, ax in enumerate(axes.flat):
   ax.text(-0.1, 1.05, subplot_labels[idx], transform=ax.transAxes,
           fontsize=18, fontweight='bold', va='bottom')
plt.savefig('/misc/glm1/person/besnard/coupling_demography_dist/figs/Extended_fig4.png', dpi=300)
