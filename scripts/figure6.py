#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 6 — Genus-resolved biomass loss forecasts

Panels:
(a) Natural disturbance (unplanned)
(b) Harvest (planned)

Data source:
- cumulative_biomass_loss_forecast_by_genus.csv

Author: Simon Besnard
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


# ----------------------------------------------------------------------
# Matplotlib params
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
# Paths & config
# ----------------------------------------------------------------------
DATA_PATH = (
    "/home/simon/hpc_home/projects/coupling_demography_dist/outputs/"
    "cumulative_biomass_loss_forecast_by_genus.csv"
)

OUTFIG = (
    "/home/simon/glm1/person/besnard/coupling_demography_dist/figs/"
    "fig6.png"
)

genus_names = ["Spruce", "Other needleleaf", "Broadleaf"]

disturbance_panels = [
    ("Natural Disturbance", "Unplanned"),
    ("Harvest", "Planned"),
]

period_order = ["Early", "Late"]
period_colors = {
    "Early": "#66c2a5",
    "Late": "#fc8d62",
}

# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
fig, axs = plt.subplots(
    1, 2,
    figsize=(10.5, 5),
    constrained_layout=True,
)

for ax, (dist_label, _) in zip(axs, disturbance_panels):

    x_base = np.arange(len(genus_names))
    width = 0.3

    positions = []
    box_data = []
    colors_box = []

    # track max y per genus (for optional annotations)
    genus_ymax = {g: -np.inf for g in genus_names}

    for i, gname in enumerate(genus_names):
        for j, period in enumerate(period_order):

            xpos = x_base[i] + (j - 0.5) * width

            subset = df.query(
                "disturbance == @dist_label and "
                "period == @period and "
                "genus == @gname"
            )["cumulative_loss"].values

            if subset.size == 0:
                continue

            positions.append(xpos)
            box_data.append(subset)
            colors_box.append(period_colors[period])

            genus_ymax[gname] = max(genus_ymax[gname], subset.max())

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=width,
        patch_artist=True,
        showfliers=False,
    )

    # styling
    for patch, c in zip(bp["boxes"], colors_box):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    for key in ["whiskers", "caps"]:
        for item in bp[key]:
            item.set_color("gray")

    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(2)

    ax.set_xticks(x_base)
    ax.set_xticklabels(genus_names)
    ax.set_ylabel("Biomass loss [TgC year$^{-1}$]")
    ax.set_title(dist_label)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ----------------------------------------------------------------------
# Legend & panel labels
# ----------------------------------------------------------------------
handles = [
    mpl.lines.Line2D([0], [0], color=period_colors["Early"], lw=6),
    mpl.lines.Line2D([0], [0], color=period_colors["Late"], lw=6),
]

axs[0].legend(
    handles,
    ["Early", "Late"],
    loc="lower left",
    frameon=False,
    fontsize=14,
)

subplot_labels = ["(a)", "(b)"]
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
print("Figure 6 written to:", OUTFIG)
