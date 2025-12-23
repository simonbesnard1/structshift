#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 4 — Structural homogenisation of disturbed forests
Analysis only. No plotting.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

from structshift.analysis.ensemble import EnsembleReducer
from structshift.analysis.spatial_filter import YearPolygonExcluder
from structshift.analysis.hexgrid import make_hex_grid
from structshift.analysis.biomass_variability import (
    cv_by_cell,
    cv_by_cell_and_genus,
)
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
PIXEL_RES = 0.0008888888888888889
HEX_DIAMETER_M = 100_000

PERIOD_EARLY = (2011, 2016)
PERIOD_LATE  = (2017, 2023)

GENUS_GROUPS = {
    "Spruce": [1],
    "Other needleleaf": [0, 2, 5],
    "Broadleaf": [3, 4, 6],
}

DISTURBANCES = {
    "Natural": "wind_bark_beetle",
    "Harvest": "harvest",
}

# ---------------------------------------------------------------------
# Load & preprocess
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

def run_figure4_analysis():

    df = load_data()
    df = YearPolygonExcluder(POLYGON_MASK).apply(df)
    df = df[df["forest_fraction"] >= 0.3]

    # Biomass ensemble
    reducer = EnsembleReducer()
    df["biomass"] = reducer.median_biomass(df)

    # GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326",
    ).to_crs(3035)

    # Europe boundary
    world = gpd.read_file(
        "/misc/glm1/person/besnard/coupling_demography_dist/data/"
        "ne_10m_admin_0_countries.zip"
    ).to_crs(3035)

    europe = world[
        (world["CONTINENT"] == "Europe")
        & (~world["ISO_A3"].isin(["RUS", "ISL"]))
    ]

    bbox = gpd.GeoDataFrame(
        geometry=[box(-20, 32, 45, 71)],
        crs="EPSG:4326"
    ).to_crs(3035)

    europe = gpd.clip(europe, bbox)

    # Hex grid
    hex_grid = make_hex_grid(
        gdf,
        hex_diameter=HEX_DIAMETER_M,
        clip_geometry=europe,
        return_index=True,
    )

    gdf_hexed = gpd.sjoin(gdf, hex_grid, predicate="within")

    # --------------------------------------------------------------
    # CV changes per hex
    # --------------------------------------------------------------
    cv_bark_early = cv_by_cell(
        gdf_hexed, "hex_id", "biomass",
        PERIOD_EARLY, "wind_bark_beetle"
    )
    cv_bark_late = cv_by_cell(
        gdf_hexed, "hex_id", "biomass",
        PERIOD_LATE, "wind_bark_beetle"
    )

    cv_harvest_early = cv_by_cell(
        gdf_hexed, "hex_id", "biomass",
        PERIOD_EARLY, "harvest"
    )
    cv_harvest_late = cv_by_cell(
        gdf_hexed, "hex_id", "biomass",
        PERIOD_LATE, "harvest"
    )

    hex_grid["delta_cv_bark"] = cv_bark_late - cv_bark_early
    hex_grid["delta_cv_harvest"] = cv_harvest_late - cv_harvest_early

    # --------------------------------------------------------------
    # CV distributions by genus (natural disturbance)
    # --------------------------------------------------------------
    cv_vals = {}
    for genus, ids in GENUS_GROUPS.items():
        cv_vals[genus] = {
            "2011–2016": cv_by_cell_and_genus(
                gdf_hexed, "hex_id", "biomass",
                PERIOD_EARLY, "wind_bark_beetle", ids
            ),
            "2017–2023": cv_by_cell_and_genus(
                gdf_hexed, "hex_id", "biomass",
                PERIOD_LATE, "wind_bark_beetle", ids
            ),
        }

    # --------------------------------------------------------------
    # Annual CV time series (Spruce)
    # --------------------------------------------------------------
    def annual_cv(gdf, disturbance):
        records = []
        for year in range(2011, 2024):
            subset = gdf[
                (gdf["year"] == year)
                & (gdf["genus"].isin(GENUS_GROUPS["Spruce"]))
                & (gdf["forest_fraction"] >= 0.3)
                & (gdf[disturbance] >= 0.5)
            ]

            cvs = []
            for col in reducer.biomass_cols:
                grouped = subset.groupby("hex_id")[col]
                valid = grouped.count()[grouped.count() >= 25].index
                cv = (grouped.std() / grouped.mean())[valid]
                if not cv.empty:
                    cvs.append(cv.median())

            if cvs:
                cvs = np.array(cvs)
                records.append({
                    "year": year,
                    "disturbance": disturbance,
                    "cv_median": np.median(cvs),
                    "cv_q5": np.quantile(cvs, 0.05),
                    "cv_q95": np.quantile(cvs, 0.95),
                })
        return pd.DataFrame(records)

    df_cv = pd.concat([
        annual_cv(gdf_hexed, "wind_bark_beetle"),
        annual_cv(gdf_hexed, "harvest"),
    ])

    return {
        "hex_grid": hex_grid,
        "europe_gdf": europe,
        "cv_vals": cv_vals,
        "cv_timeseries": df_cv,
    }


# ---------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------

results = run_figure4_analysis()

# ---------------------------------------------------------------------
# Plotting — Figure 4
# ---------------------------------------------------------------------

from scipy.stats import linregress
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import matplotlib as mpl

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

def plot_figure4(results, out_path):

    hex_grid = results["hex_grid"]
    europe_gdf = results["europe_gdf"]
    cv_vals = results["cv_vals"]
    df_cv = results["cv_timeseries"]

    fig, axs = plt.subplots(2, 2, figsize=(12, 9),
                            constrained_layout=True)

    # --------------------------------------------------------------
    # (a,b) Maps
    # --------------------------------------------------------------
    for ax, col, title in [
        (axs[0, 0], "delta_cv_bark", "Natural Disturbance"),
        (axs[0, 1], "delta_cv_harvest", "Harvest"),
    ]:
        europe_gdf.boundary.plot(ax=ax, linewidth=0.5, color="lightgrey")
        hex_grid.dropna(subset=[col]).plot(
            column=col, ax=ax, cmap="RdBu_r",
            vmin=-0.3, vmax=0.3, edgecolor="none",
            legend=True,
            legend_kwds={"label": r"$\Delta$CV", "shrink": 0.6},
        )
        ax.set_title(title)
        ax.set_axis_off()

    # --------------------------------------------------------------
    # (c) CV distributions by genus
    # --------------------------------------------------------------
    species = list(cv_vals.keys())
    periods = ["2011–2016", "2017–2023"]
    colors = {"2011–2016": "#66c2a5", "2017–2023": "#fc8d62"}

    for i, sp in enumerate(species):
        for j, period in enumerate(periods):
            xpos = i + j * 0.3 - 0.15
            vals = np.asarray(cv_vals[sp][period].dropna())
            if len(vals) == 0:
                continue

            q1, q3 = np.percentile(vals, [25, 75])
            vals = vals[(vals > q1) & (vals < q3)]

            px, py, *_ = violins(vals, pos=xpos, spread=0.25, max_num_points=2000)
            axs[1, 0].scatter(py, px, color=colors[period], alpha=0.3, marker=".")
            axs[1, 0].scatter(xpos, np.median(vals), color="black", marker="d", s=80)

        d = cohens_d(
            cv_vals[sp]["2011–2016"].dropna(),
            cv_vals[sp]["2017–2023"].dropna(),
        )
        axs[1, 0].text(i, np.max(vals) + 0.02, f"d={d:.2f}",
                       ha="center", fontsize=12)

    axs[1, 0].set_xticks(range(len(species)))
    axs[1, 0].set_xticklabels(species)
    axs[1, 0].set_ylabel("CV Biomass")
    axs[1, 0].set_ylim(0.15, 0.55)
    axs[1, 0].legend(
        [plt.Line2D([], [], color=c) for c in colors.values()],
        colors.keys(), frameon=False
    )

    # --------------------------------------------------------------
    # (d) Time series
    # --------------------------------------------------------------
    for label, color in [
        ("wind_bark_beetle", "#1b9e77"),
        ("harvest", "black"),
    ]:
        sub = df_cv[df_cv["disturbance"] == label]
        axs[1, 1].plot(sub["year"], sub["cv_median"],
                       label=label, color=color)
        axs[1, 1].fill_between(sub["year"],
                               sub["cv_q5"], sub["cv_q95"],
                               color=color, alpha=0.3)

        slope, intercept, *_ = linregress(sub["year"], sub["cv_median"])
        axs[1, 1].plot(sub["year"],
                       intercept + slope * sub["year"],
                       color=color, linestyle="--")

    axs[1, 1].set_xlabel("Year")
    axs[1, 1].set_ylabel("CV Biomass")
    axs[1, 1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1, 1].legend(frameon=False)

    # --------------------------------------------------------------
    # Panel labels
    # --------------------------------------------------------------
    for lab, ax in zip(["(a)", "(b)", "(c)", "(d)"], axs.flat):
        ax.text(-0.1, 1.05, lab, transform=ax.transAxes,
                fontsize=18, fontweight="bold")

    plt.savefig(out_path, dpi=300)
    plt.close(fig)

plot_figure4(
    results,
    "/misc/glm1/person/besnard/coupling_demography_dist/figs/fig4_v2.png"
)
