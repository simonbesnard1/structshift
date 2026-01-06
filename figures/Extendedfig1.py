#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 11:48:35 2025

@author: simon
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import numpy as np

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


#%% 1. calculate disturbance shit 
df = pd.read_csv('/misc/glm1/person/besnard/coupling_demography_dist/data/disturbance_timeseries_database.csv')
df = df[df["Year"] >= 1960]

# Only keep wind and bark beetle
df_unplanned = df[df["disturbance_driver"].isin(["Wind", "Bark Beetles"])]
df_grouped = (
    df_unplanned
    .groupby(["Year", "disturbance_driver"])["Reported_GapFilled_m3"]
    .sum()
    .reset_index()
)

df_pivot = df_grouped.pivot(index="Year", columns="disturbance_driver", values="Reported_GapFilled_m3").fillna(0)

# Compute total unplanned volume per year
df_pivot["total"] = df_pivot.sum(axis=1)

# Compute relative fractions
df_frac = df_pivot.div(df_pivot["total"], axis=0).drop(columns="total")


#%% 2. Climate climate extreme event
event_stats = pd.read_csv("/misc/glm1/person/besnard/coupling_demography_dist/data/MergedEventStats_landonly_int.csv")
event_stats['time'] = pd.to_datetime(event_stats['end_time'])
event_stats['year'] = event_stats['time'].dt.year
event_stats = event_stats[(event_stats['year'] >= 1960)]
event_stats['longitude_min'] = event_stats['longitude_min'].apply(lambda x: x - 360 if x > 180 else x)
event_stats['longitude_max'] = event_stats['longitude_max'].apply(lambda x: x - 360 if x > 180 else x)

#valid = (event_stats['longitude_min'] < event_stats['longitude_max']) & (event_stats['latitude_min'] < event_stats['latitude_max'])
#event_stats = event_stats[valid].copy()
event_stats['geometry'] = event_stats.apply(lambda row: box(row['longitude_min'], row['latitude_min'],
                                                            row['longitude_max'], row['latitude_max']), axis=1)
event_gdf = gpd.GeoDataFrame(event_stats, geometry='geometry', crs='EPSG:4326')

# --- Create Europe bounding box ---
world = gpd.read_file('/misc/glm1/person/besnard/coupling_demography_dist/data/ne_10m_admin_0_countries.zip')
europe_gdf = world[(world['CONTINENT'] == 'Europe') & (~world['ISO_A3'].isin(['RUS', 'ISL']))].to_crs("EPSG:4326")
bbox = box(-20, 34, 45, 71)  # Approx. continental Europe extent (W, S, E, N)
europe_gdf = gpd.clip(europe_gdf, bbox)
event_gdf = gpd.clip(event_gdf, bbox)
event_gdf = (
    event_gdf
    .groupby(["year"])["area"]
    .sum()
    .reset_index()
)

# Prepare x and y data
x = event_gdf['year'].values
y = event_gdf['area'].values

# Fit a 2nd-degree polynomial
coeffs = np.polyfit(x, y, deg=2)
poly_func = np.poly1d(coeffs)

# Generate smooth x-values for plotting the curve
x_fit = np.linspace(x.min(), x.max(), 200)
y_fit = poly_func(x_fit)

# Plotting
fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True, sharex=True)
colors = ["#8da0cb", "#fc8d62"]

# First subplot: relative fractions
df_frac.plot(kind='area', stacked=True, alpha=0.8, ax=axes[0], color=colors)
axes[0].set_title("Shift in Natural Disturbance Types", fontsize=16)
axes[0].set_ylabel("Fraction [-]", fontsize=16)
axes[0].set_xlabel("")
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].legend(title="", frameon=True, loc='upper left', bbox_to_anchor=(0.05, 0.95))


import numpy as np

# Your data
x = event_gdf['year'].values
y = event_gdf['area'].values

# Define a breakpoint year
breakpoint = 1997

# Split the data
mask1 = x < breakpoint
mask2 = x >= breakpoint

# Fit linear regression on each segment
coeffs1 = np.polyfit(x[mask1], y[mask1], deg=1)
coeffs2 = np.polyfit(x[mask2], y[mask2], deg=1)

# Create poly1d functions
line1 = np.poly1d(coeffs1)
line2 = np.poly1d(coeffs2)

# Generate x values for smooth plotting
x_fit1 = np.linspace(x[mask1].min(), x[mask1].max(), 100)
x_fit2 = np.linspace(x[mask2].min(), x[mask2].max(), 100)

# Extract slopes
slope1 = coeffs1[0]
slope2 = coeffs2[0]

# Plot
axes[1].scatter(x, y, alpha=0.6, s=50)
axes[1].plot(x_fit1, line1(x_fit1), '--', color='black', linewidth=3)
axes[1].plot(x_fit2, line2(x_fit2), '--', color='red', linewidth=3)

axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].set_ylabel("Volume-to-Duration Ratio [-]", fontsize=16)
axes[1].set_title("Temporal Trend in Area Impacted by Climate Extremes", fontsize=16)

# Annotate slopes
axes[1].text(0.02, 0.88,
             f"Slope: {slope1:.2f} $\\mathrm{{year^{{-1}}}}$", fontsize=14,
             transform=axes[1].transAxes,  
             va='top', ha='left',
)

axes[1].text(0.02, 0.78,
             f"Slope: {slope2:.2f} $\\mathrm{{year^{{-1}}}}$", fontsize=14,
             transform=axes[1].transAxes,  
             va='top', ha='left', color='red'
)

plt.savefig('/misc/glm1/person/besnard/coupling_demography_dist/figs/Extended_fig1.png', dpi=300)
