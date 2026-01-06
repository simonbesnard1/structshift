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
from matplotlib.ticker import MaxNLocator

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
use_cols = ['latitude', 'longitude', 'time','harvest', 'wind_bark_beetle', 'forest_fraction'] + [f'forest_age_gami_2010_m{i}' for i in range(20)] + [f'biomass_m{i}' for i in range(20)]
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

biomass_cols = [col for col in df.columns if col.startswith("biomass")]
df[biomass_cols] = df[biomass_cols].where(df[biomass_cols] > 0)
for i in range(20):
    biomass_col = f"biomass_m{i}"
    df[biomass_col] = df[biomass_col] * 0.47
df["biomass"] = df[biomass_cols].median(axis=1, skipna=True) 
forest_age_cols = [col for col in df.columns if col.startswith("forest_age_gami_2010")]
df[forest_age_cols] = df[forest_age_cols].where(df[forest_age_cols] > 0)
df["forest_age"] = df[forest_age_cols].median(axis=1, skipna=True) 
df = df.dropna()

#%% Create 2D heatmaps of age vs biomass bins per disturbance type
disturbance_types = {'Natural Disturbance': 'wind_bark_beetle','Harvest':'harvest'}

# Re-define the age and biomass bins with labels and bounds
age_classes = {
    '0-20': (0, 20),
    '21-40': (20, 40),
    '41-60': (40, 60),
    '61-80': (60, 80),
    '81-100': (80, 100),
    '101-120': (100, 120),
    '$>$120': (120, np.inf)
}
agb_classes = {
    '0-20':   (0, 20),
    '21-40':  (20, 40),
    '41-60':  (40, 60),
    '61-80':  (60, 80),
    '81-100':  (80, 100),
    '101-120':  (100, 120),
    '$>$120':  (120, np.inf)
}

age_labels = list(age_classes.keys())
agb_labels = list(agb_classes.keys())

#%% Compute all 6 matrices
time_slices = {
    "2011–2016": (2011, 2016),
    "2017–2023": (2017, 2023),
}

# nested dictionary: [time_slice][disturbance_type]
fraction_matrix = {
    tlabel: {
        dlabel: np.full((len(agb_classes), len(age_classes)), np.nan)
        for dlabel in disturbance_types
    } for tlabel in time_slices
}

for tlabel, (start_year, end_year) in time_slices.items():
    
    for disturbance_name, disturbance_column in disturbance_types.items():
        subset = df[(df["year"] >= start_year) &
                    (df["year"] <= end_year) &
                    (df['forest_fraction'] >= 0.3) &
                    (df[disturbance_column] > 0.5)]
        subset = subset.copy()
        total_pixels = len(subset.index)
        age_2010 = subset["forest_age"].values
        biomass_2010 = subset["biomass"].values

        if total_pixels > 0:
            for i, (agb_label, (agb_min, agb_max)) in enumerate(agb_classes.items()):
                if np.isinf(agb_max):
                    agb_mask = (biomass_2010 > agb_min)
                else:
                    agb_mask = (biomass_2010 >= agb_min) & (biomass_2010 <= agb_max)

                for j, (age_label, (age_min, age_max)) in enumerate(age_classes.items()):
                    if np.isinf(age_max):
                        age_mask = (age_2010 > age_min)
                    else:
                        age_mask = (age_2010 > age_min) & (age_2010 <= age_max)

                    count = np.sum(agb_mask & age_mask)
                    frac = (count / total_pixels) * 100
                    fraction_matrix[tlabel][disturbance_name][i, j] = frac

#%% Recreate the plot with vertical colorbars on the side and asymmetric change colorbar
fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True,
                         gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.02})
subplot_labels = ['a', 'b', 'c', 'd']
rows = ['2011–2016', '2017–2023']
im_diff = []

for idx, row_type in enumerate(rows):
    for jdx, disturbance in enumerate(disturbance_types):
        ax = axes[idx, jdx]
        data = fraction_matrix[row_type][disturbance]

        masked_data = np.ma.masked_invalid(data)

        im = ax.imshow(masked_data, cmap='afmhot_r', origin='lower', aspect='auto', vmin=0, vmax=6.5)
        im_diff.append(im)

        ax.set_xticks(np.arange(len(age_labels)))
        ax.set_xticklabels(age_labels, rotation=45)
        ax.set_xlabel("Age class [years]", fontsize=16)

        ax.set_yticks(np.arange(len(agb_labels)))
        ax.set_yticklabels(agb_labels)
        ax.set_ylabel("AGC class [MgC ha$^{-1}$]", fontsize=16)

        ax.set_title(f"{disturbance} – {row_type}", fontsize=14, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

cbar2 = fig.colorbar(im_diff[0], ax=axes[0, :], shrink=0.75, location='right', pad=0.02)
cbar2.set_label("Change in disturbance [$\\%$]", fontsize=15)
cbar2.ax.tick_params(labelsize=12)
cbar2.ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

subplot_labels = ['(a)', '(b)', '(c)', '(d)']
for idx, ax in enumerate(axes.flat):
   ax.text(-0.1, 1.05, subplot_labels[idx], transform=ax.transAxes,
           fontsize=18, fontweight='bold', va='bottom')
plt.savefig('/misc/glm1/person/besnard/coupling_demography_dist/figs/Extended_fig2.png', dpi=300)
