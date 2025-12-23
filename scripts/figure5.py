#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 5 — Biomass loss and disturbed area forecasts

Panels:
(a) Δ biomass (natural disturbance)
(b) Δ biomass (harvest)
(c) Biomass loss boxplots
(d) Pan-European disturbed area (historical + forecast)

Author: Simon
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from shapely.geometry import box


# ----------------------------------------------------------------------
# Matplotlib params (unchanged)
# ----------------------------------------------------------------------
params = {
    "font.family": "serif",
    "font.size": 18,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "axes.linewidth": 0.5,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "xtick.major.width": 0.3,
    "ytick.major.width": 0.3,
    "xtick.minor.width": 0.3,
    "ytick.minor.width": 0.3,
    "legend.fontsize": 16,
    "text.usetex": True,
}
mpl.rcParams.update(params)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def moving_average(series, window=10):
    x = np.asarray(series, dtype=float)
    if window <= 1 or x.size == 0:
        return x

    pad_left = window // 2
    pad_right = window - 1 - pad_left
    x_pad = np.pad(x, (pad_left, pad_right), mode="edge")
    return np.convolve(x_pad, np.ones(window), mode="valid") / window


def load_europe_background(world):
    bbox = box(-20, 32, 45, 71)
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs="EPSG:4326").to_crs("EPSG:3035")

    europe = world[
        (world["CONTINENT"] == "Europe")
        & (~world["ISO_A3"].isin(["RUS", "ISL"]))
    ].to_crs("EPSG:3035")

    return gpd.clip(europe, bbox_gdf)


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
BASE = "/home/simon/hpc_home/projects/coupling_demography_dist/outputs/"

DF_BOXPLOT = BASE + "cumulative_biomass_loss_flux_v11_full.parquet"
FINAL_DF   = BASE + "forecast_results_1985_2040_v11.csv"
HEX_GRID   = BASE + "hex_delta_biomass_v11.gpkg"

WORLD_PATH = (
    "/home/simon/hpc_home/projects/coupling_demography_dist/data/ancillary/"
    "ne_10m_admin_0_countries.zip"
)

OUTFIG = (
    "fig5.png"
)


# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------
df_boxplot = pd.read_parquet(DF_BOXPLOT)
final_df   = pd.read_csv(FINAL_DF)
hex_grid   = gpd.read_file(HEX_GRID)

world = gpd.read_file(WORLD_PATH)
europe_gdf = load_europe_background(world)


# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(14.5, 10.5), constrained_layout=True)

# ----------------------------------------------------------------------
# (a) Δ biomass — natural disturbance
# ----------------------------------------------------------------------
europe_gdf.boundary.plot(ax=axs[0, 0], linewidth=0.5, color="lightgrey")

hex_grid[
    hex_grid["delta_biomass_bark"].notna()
    & (hex_grid["delta_biomass_bark"] != 0)
].plot(
    column="delta_biomass_bark",
    ax=axs[0, 0],
    cmap="RdBu",
    edgecolor="none",
    vmin=-1,
    vmax=1,
    legend=True,
    legend_kwds={"label": r"$\Delta$Biomass [TgC]", "shrink": 0.6},
)

axs[0, 0].set_title(
    "Biomass Loss: Natural Disturbance (Late − Early)", fontsize=16
)


# ----------------------------------------------------------------------
# (b) Δ biomass — harvest
# ----------------------------------------------------------------------
europe_gdf.boundary.plot(ax=axs[0, 1], linewidth=0.5, color="lightgrey")

hex_grid[
    hex_grid["delta_biomass_harvest"].notna()
    & (hex_grid["delta_biomass_harvest"] != 0)
].plot(
    column="delta_biomass_harvest",
    ax=axs[0, 1],
    cmap="RdBu",
    edgecolor="none",
    vmin=-1,
    vmax=1,
    legend=True,
    legend_kwds={"label": r"$\Delta$Biomass [TgC]", "shrink": 0.6},
)

axs[0, 1].set_title(
    "Biomass Loss: Harvest (Late − Early)", fontsize=16
)


# ----------------------------------------------------------------------
# (c) Biomass loss boxplots
# ----------------------------------------------------------------------
ax_box = axs[1, 0]

group_order = [
    ("Natural Disturbance", "Early"),
    ("Natural Disturbance", "Late"),
    ("Harvest", "Early"),
    ("Harvest", "Late"),
]

box_data = [
    df_boxplot.query(
        "disturbance == @d and period == @p"
    )["cumulative_loss"].values
    for d, p in group_order
]

positions = [0.9, 1.1, 1.9, 2.1]
colors = ["#66c2a5", "#fc8d62"] * 2

bp = ax_box.boxplot(
    box_data, positions=positions, widths=0.15, patch_artist=True
)

for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

for key in ["whiskers", "caps"]:
    for item in bp[key]:
        item.set_color("gray")

for med in bp["medians"]:
    med.set_color("black")
    med.set_linewidth(2)

ax_box.set_xticks([1.0, 2.0])
ax_box.set_xticklabels(["Natural Disturbance", "Harvest"], fontsize=16)
ax_box.set_ylabel("Biomass Loss [TgC year$^{-1}$]", fontsize=16)
ax_box.set_title("Biomass Loss Forecast by Disturbance", fontsize=16)

ax_box.legend(
    handles=[
        plt.Line2D([0], [0], color="#66c2a5", lw=6),
        plt.Line2D([0], [0], color="#fc8d62", lw=6),
    ],
    labels=["Early", "Late"],
    frameon=False,
)

ax_box.spines["top"].set_visible(False)
ax_box.spines["right"].set_visible(False)


# ----------------------------------------------------------------------
# (d) Pan-European disturbed area
# ----------------------------------------------------------------------
ax_area = axs[1, 1]

colors_area = {
    "Natural Disturbance": "#1f77b4",
    "Harvest": "#ff7f0e",
}

ax_area.set_title("Pan-European Disturbed Area", fontsize=16)

for label, color in colors_area.items():
    sub = final_df.query("label == @label").sort_values("time")

    hist = sub.dropna(subset=["hist"])
    if not hist.empty:
        ax_area.plot(
            hist["time"],
            moving_average(hist["hist"].values, 5),
            "o",
            alpha=0.6,
            color=color,
        )

    fut = sub.dropna(subset=["median"])
    if not fut.empty:
        ax_area.plot(fut["time"], fut["median"], color=color, label=label)
        ax_area.fill_between(
            fut["time"], fut["p5"], fut["p95"], color=color, alpha=0.2
        )

ax_area.set_ylabel("Disturbed Area [Mha]", fontsize=16)
ax_area.legend(frameon=False)
ax_area.spines["top"].set_visible(False)
ax_area.spines["right"].set_visible(False)


# ----------------------------------------------------------------------
# Cosmetics
# ----------------------------------------------------------------------
for ax in axs.flat:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

subplot_labels = ["(a)", "(b)", "(c)", "(d)"]
for i, ax in enumerate(axs.flat):
    ax.text(
        -0.1,
        1.05,
        subplot_labels[i],
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        va="bottom",
    )


# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------
plt.savefig(OUTFIG, dpi=300)
print("Figure 5 written to:", OUTFIG)
