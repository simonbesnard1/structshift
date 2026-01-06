#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 3 — Genus-specific biomass losses under planned and unplanned disturbances

Author: Simon Besnard

"""

from pathlib import Path
import pandas as pd

from structshift.analysis.ensemble import EnsembleReducer
from structshift.analysis.spatial_filter import YearPolygonExcluder
from structshift.analysis.genus_aggregation import aggregate_genus_biomass
from structshift.utils.area_calc import pixel_area_latlon_km2
from structshift.plotting.stats import violins, cohens_d


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATA_PATH = Path(
    "/misc/glm1/person/besnard/coupling_demography_dist/data/"
    "disturbance_data_combined_v2025-11-1.parquet"
)

POLYGON_MASK = Path(
    "/misc/glm1/person/besnard/coupling_demography_dist/data/"
    "bounding_box_filter_years_4326.gpkg"
)

START_YEAR = 2011
PIXEL_RES = 0.0008888888888888889  # ~100 m

PERIODS = {
    "2011-2016": (2011, 2016),
    "2017-2023": (2017, 2023),
}

DISTURBANCES = {
    "Natural Disturbance": "wind_bark_beetle",
    "Harvest": "harvest",
}

GENUS_GROUPS = {
    "Spruce": [1],
    "Other needleleaf": [0, 2, 5],
    "Broadleaf": [3, 4, 6],
}


# ---------------------------------------------------------------------
# Load & preprocess (shared logic)
# ---------------------------------------------------------------------

def load_data():
    use_cols = (
        ["latitude", "longitude", "time", "genus",
         "harvest", "wind_bark_beetle", "forest_fraction"]
        + [f"biomass_m{i}" for i in range(20)]
    )

    df = pd.read_parquet(DATA_PATH, columns=use_cols)
    df["time"] = pd.to_datetime(df["time"])
    df["year"] = df["time"].dt.year
    df = df[df["year"] >= START_YEAR]

    # Arctic exclusion
    df = df[~((df["year"].isin([2018, 2023])) & (df["latitude"] >= 65))]

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------
# Main analysis routine
# ---------------------------------------------------------------------

def run_figure3_analysis():

    df = load_data()
    df = YearPolygonExcluder(POLYGON_MASK).apply(df)

    # Forest fraction
    df = df[df["forest_fraction"] >= 0.3]

    # Ensemble biomass
    reducer = EnsembleReducer()
    biomass_cols = reducer.biomass_cols
    df["biomass"] = reducer.median_biomass(df)

    # Pixel area
    df["pixel_area_km2"] = pixel_area_latlon_km2(
        df["latitude"], PIXEL_RES, PIXEL_RES
    )

    # --------------------------------------------------------------
    # Genus-level aggregation
    # --------------------------------------------------------------
    genus_totals = {}

    for dist_label, dist_col in DISTURBANCES.items():
        genus_totals[dist_label] = {}

        for period_label, period in PERIODS.items():
            genus_totals[dist_label][period_label] = aggregate_genus_biomass(
                df,
                genus_groups=GENUS_GROUPS,
                biomass_cols=biomass_cols,
                disturbance_col=dist_col,
                period=period,
            )

    # Flatten table (for CSV / plotting)
    rows = []
    for dist, periods in genus_totals.items():
        for period, genus_data in periods.items():
            for genus, (med, q5, q95, area) in genus_data.items():
                rows.append({
                    "Disturbance": dist,
                    "Period": period,
                    "Genus": genus,
                    "Median Biomass Loss (TgC)": med,
                    "P5": q5,
                    "P95": q95,
                    "Area (Mha)": area,
                })

    df_summary = pd.DataFrame(rows)

    return {
        "raw_data": df,
        "genus_totals": genus_totals,
        "summary_table": df_summary,
    }


# ---------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------

results = run_figure3_analysis()

# ---------------------------------------------------------------------
# Plotting — Figure 3
# ---------------------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 16,
    "axes.labelsize": 12,
    "axes.linewidth": 0.5,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "text.usetex": True,
})

def plot_figure3(results, out_path):

    df = results["raw_data"]
    genus_totals = results["genus_totals"]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.5),
                             constrained_layout=True,
                             gridspec_kw={"hspace": 0.1})

    period_colors = {
        "2011-2016": "#66c2a5",
        "2017-2023": "#fc8d62",
    }

    disturbance_labels = ["Natural Disturbance", "Harvest"]
    subplot_labels = ["a", "b"]

    # --------------------------------------------------------------
    # (a,b) violin-style scatter plots
    # --------------------------------------------------------------
    for idx, dist in enumerate(disturbance_labels):
        ax = axes[0, idx]
        xpos = 0
        xticks = []

        for genus, ids in GENUS_GROUPS.items():
            period_vals = []

            for period in PERIODS:
                sub = df[
                    (df["genus"].isin(ids)) &
                    (df["year"].between(*PERIODS[period])) &
                    (df[DISTURBANCES[dist]] >= 0.5)
                ]

                vals = sub["biomass"].dropna().values
                if len(vals) == 0:
                    period_vals.append(np.array([]))
                    xpos += 1
                    continue

                pointx_pos, pointy_pos, _, _ = violins(
                    vals,
                    pos=xpos,
                    spread=0.3,
                    max_num_points=2000,
                )
                
                ax.scatter(
                    pointy_pos,
                    pointx_pos,
                    color=period_colors[period],
                    alpha=0.2,
                    marker=".",
                    linewidths=0,
                )


                ax.scatter(
                    xpos,
                    np.median(vals),
                    color="black",
                    marker="d",
                    s=80,
                )

                period_vals.append(vals)
                xpos += 1

            # Effect size
            if all(len(v) > 10 for v in period_vals):
                d = cohens_d(*period_vals)
                ax.text(
                    xpos - 1,
                    max(np.max(v) for v in period_vals) + 2,
                    f"d={d:.2f}",
                    ha="center",
                    fontsize=11,
                )

            xticks.append((xpos - 1, genus))
            xpos += 1

        ax.set_xticks([p for p, _ in xticks])
        ax.set_xticklabels([g for _, g in xticks])
        ax.set_ylabel("Biomass [MgC ha$^{-1}$]")
        ax.set_title(dist)
        ax.text(-0.06, 1.08, f"({subplot_labels[idx]})",
                transform=ax.transAxes,
                fontsize=16, fontweight="bold")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # --------------------------------------------------------------
    # (c,d) bar plots of total biomass loss
    # --------------------------------------------------------------
    subplot_labels = ["c", "d"]

    for idx, dist in enumerate(disturbance_labels):
        ax = axes[1, idx]
        bar_data = genus_totals[dist]

        genus_names = list(GENUS_GROUPS.keys())
        x = np.arange(len(genus_names))
        width = 0.35

        early = [bar_data["2011-2016"][g] for g in genus_names]
        late  = [bar_data["2017-2023"][g] for g in genus_names]

        ax.bar(x - width/2, [e[0] for e in early],
               yerr=[[e[0]-e[1] for e in early], [e[2]-e[0] for e in early]],
               width=width, color=period_colors["2011-2016"], capsize=4)

        ax.bar(x + width/2, [l[0] for l in late],
               yerr=[[l[0]-l[1] for l in late], [l[2]-l[0] for l in late]],
               width=width, color=period_colors["2017-2023"], capsize=4)

        ax.set_xticks(x)
        ax.set_xticklabels(genus_names)
        ax.set_ylabel("Total biomass disturbed [TgC]")
        ax.set_title(dist)
        ax.text(-0.06, 1.08, f"({subplot_labels[idx]})",
                transform=ax.transAxes,
                fontsize=16, fontweight="bold")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.savefig(out_path, dpi=300)
    plt.close(fig)

plot_figure3(
    results,
    "fig3.png"
)
