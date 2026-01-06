#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 11:08:30 2025

@author: simon
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from shapely.geometry import box
import numpy as np

# ----------------------------------------------------------------------
# Matplotlib params
# ----------------------------------------------------------------------
params = {
    # font
    'font.family': 'serif',
    # 'font.serif': 'Times',
    'font.size': 18,
    # axes
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'axes.linewidth': 0.5,
    # ticks
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'xtick.major.width': 0.3,
    'ytick.major.width': 0.3,
    'xtick.minor.width': 0.3,
    'ytick.minor.width': 0.3,
    # legend
    'legend.fontsize': 16,
    # tex
    'text.usetex': True,
}

mpl.rcParams.update(params)

def moving_average(series, window=10):
    """Centered moving average (NumPy-only, edge-handled)."""
    x = np.asarray(series, dtype=float)
    n = x.size
    if window <= 1 or n == 0:
        return x

    pad_left = window // 2
    pad_right = window - 1 - pad_left
    x_padded = np.pad(x, (pad_left, pad_right), mode="edge")

    kernel = np.ones(window, dtype=float)
    out = np.convolve(x_padded, kernel, mode="valid") / window
    return out


# paths adapted to your local setup
base = "/home/simon/hpc_home/projects/coupling_demography_dist/outputs/"

df_boxplot = pd.read_parquet(base + "cumulative_biomass_loss_flux_v11_full.parquet")
final_df   = pd.read_csv(base + "forecast_results_1985_2040_v11.csv")
hex_grid   = gpd.read_file(base + "hex_delta_biomass_v11.gpkg")


# Load background map
world = gpd.read_file(
    '/home/simon/hpc_home/projects/coupling_demography_dist/data/ancillary/ne_10m_admin_0_countries.zip'
)
europe_gdf = world[world['CONTINENT'] == 'Europe'].to_crs("EPSG:4326")
bbox = box(-20, 32, 45, 71)

bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs="EPSG:4326")
bbox_3035 = bbox_gdf.to_crs("EPSG:3035")
europe_gdf = europe_gdf.to_crs("EPSG:3035")

# Clip using bbox & exclude unwanted countries
europe_gdf = gpd.clip(europe_gdf, bbox_3035)
exclude_countries = ['RUS', 'ISL']
europe_gdf = europe_gdf[~europe_gdf['ISO_A3'].isin(exclude_countries)]


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(14.5, 10.5), constrained_layout=True)

# Top Left: Delta biomass (natural disturbance)
europe_gdf.boundary.plot(ax=axs[0, 0], linewidth=0.5, color='lightgrey')
hex_grid[
    hex_grid["delta_biomass_bark"].notna() &
    (hex_grid["delta_biomass_bark"] != 0)
].plot(
    column="delta_biomass_bark", ax=axs[0, 0],
    cmap="RdBu", edgecolor="none", vmin=-1, vmax=1,
    legend=True,
    legend_kwds={"label": r"$\Delta$Biomass [TgC]", "shrink": 0.6}
)
axs[0, 0].set_title("Biomass Loss: Natural Disturbance (Late - Early)",
                    fontsize=16)
#axs[0, 0].set_axis_off()

# Top Right: Delta biomass (harvest)
europe_gdf.boundary.plot(ax=axs[0, 1], linewidth=0.5, color='lightgrey')
hex_grid[
    hex_grid["delta_biomass_harvest"].notna() &
    (hex_grid["delta_biomass_harvest"] != 0)
].plot(
    column="delta_biomass_harvest", ax=axs[0, 1],
    cmap="RdBu", edgecolor="none", vmin=-1, vmax=1,
    legend=True,
    legend_kwds={"label": r"$\Delta$Biomass [TgC]", "shrink": 0.6}
)

axs[0, 1].set_title("Biomass Loss: Harvest (Late - Early)",
             fontsize=16)
#axs[0, 1].set_axis_off()

# Bottom Left: Boxplot
ax_box = axs[1, 0]
group_order = [
    ("Natural Disturbance", "Early"),
    ("Natural Disturbance", "Late"),
    ("Harvest", "Early"),
    ("Harvest", "Late"),
]

box_data = [
    df_boxplot.query("disturbance == @dist and period == @per")["cumulative_loss"].values
    for dist, per in group_order
]

colors_box = ["#66c2a5", "#fc8d62", "#66c2a5", "#fc8d62"]
positions = [0.9, 1.1, 1.9, 2.1]
width = 0.15

bp = ax_box.boxplot(box_data, positions=positions, widths=width, patch_artist=True)
for patch, color in zip(bp["boxes"], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

for whisker in bp['whiskers']:
    whisker.set_color('gray')
for cap in bp['caps']:
    cap.set_color('gray')
for median in bp['medians']:
    median.set_color('black')
    median.set_linewidth(2)

ax_box.set_xticks([1.0, 2.0])
ax_box.set_xticklabels(["Natural Disturbance", "Harvest"], fontsize=16)
legend_patches = [plt.Line2D([0], [0], color=c, lw=6) for c in ["#66c2a5", "#fc8d62"]]
ax_box.legend(legend_patches, ["Early", "Late"], title="", frameon=False)
ax_box.set_ylabel("Biomass Loss [TgC year$^{-1}$]", fontsize=16)
ax_box.set_title("Biomass Loss Forecast by Disturbance", fontsize=16)
ax_box.spines['top'].set_visible(False)
ax_box.spines['right'].set_visible(False)

# Bottom Right: historical & forecasted disturbed area
colors_area = {
    "Natural Disturbance": "#1f77b4",
    "Harvest": "#ff7f0e",
}
area_types = {
    'Natural Disturbance': 'wind_bark_beetle',
    'Harvest': 'harvest',
}
years_future = np.arange(2024, 2041)
axs[1, 1].set_title("Pan-European Disturbed Area", fontsize=16)
for label, color in colors_area.items():
    sub = (
        final_df
        .loc[final_df["label"] == label]
        .sort_values("time")
    )

    # Historical part
    hist = sub.dropna(subset=["hist"])
    if not hist.empty:
        axs[1, 1].plot(
            hist["time"],
            moving_average(hist["hist"].values, window=5),
            "o",
            alpha=0.6,
            color=color
        )

    # Forecast part
    fut = sub.dropna(subset=["median"])
    if not fut.empty:
        axs[1, 1].plot(
            fut["time"],
            fut["median"],
            color=color,
            label=f"{label}"
        )
        axs[1, 1].fill_between(
            fut["time"],
            fut["p5"],
            fut["p95"],
            color=color,
            alpha=0.2
        )


axs[1, 1].set_ylabel("Disturbed Area [Mha]", fontsize=16)
axs[1, 1].legend(frameon=False, loc='upper left')
axs[1, 1].spines['top'].set_visible(False)
axs[1, 1].spines['right'].set_visible(False)

for ax in [axs[0,0], axs[0,1]]:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)


subplot_labels = ['(a)', '(b)', '(c)', '(d)']
for idx, ax_sub in enumerate(axs.flat):
   ax_sub.text(
       -0.1, 1.05, subplot_labels[idx],
       transform=ax_sub.transAxes,
       fontsize=18, fontweight='bold', va='bottom'
   )

plt.savefig(
    '/fig5.png',
    dpi=300
)

