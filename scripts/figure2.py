#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 2 — Spatial redistribution and structural shifts of disturbed forests

Analysis pipeline:
1. Load and preprocess disturbance data
2. Apply year-specific spatial exclusions
3. Reduce forest age ensemble
4. Aggregate disturbed forest age on a hex grid
5. Quantify structural redistribution using energy distance
6. Compute per-hex age shifts between periods

No plotting is performed here.
"""

from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

from structshift.analysis.ensemble import EnsembleReducer
from structshift.analysis.spatial_filter import YearPolygonExcluder
from structshift.analysis.hexgrid import make_hex_grid
from structshift.analysis.spatial_aggregation import (
    aggregate_by_cell,
    distribution_distance_by_cell,
)

# ---------------------------------------------------------------------
# Configuration (paper-frozen)
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
FOREST_FRACTION_MIN = 0.3
DISTURBANCE_MIN = 0.5

PERIOD_EARLY = (2011, 2016)
PERIOD_LATE  = (2017, 2023)

HEX_DIAMETER_M = 100_000     # 100 km
MIN_PIXELS_MEDIAN = 25
MIN_PIXELS_ED = 50


# ---------------------------------------------------------------------
# Load & preprocess (shared with Fig. 1)
# ---------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    use_cols = (
        ["latitude", "longitude", "time",
         "harvest", "wind_bark_beetle", "forest_fraction"]
        + [f"forest_age_gami_2010_m{i}" for i in range(20)]
    )

    df = pd.read_parquet(DATA_PATH, columns=use_cols)
    df["time"] = pd.to_datetime(df["time"])
    df["year"] = df["time"].dt.year
    df = df[df["year"] >= START_YEAR]

    # Arctic exclusion (paper-specific)
    df = df[~((df["year"].isin([2018, 2023])) & (df["latitude"] >= 65))]

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------
# Main analysis routine
# ---------------------------------------------------------------------

def run_figure2_analysis() -> dict:

    # --------------------------------------------------------------
    # Load + spatial filtering
    # --------------------------------------------------------------
    df = load_data()
    df = YearPolygonExcluder(POLYGON_MASK).apply(df)

    # --------------------------------------------------------------
    # Reduce ensemble (forest age)
    # --------------------------------------------------------------
    reducer = EnsembleReducer()
    df["forest_age"] = reducer.median_age(df)
    df = df.dropna(subset=["forest_age"])

    # --------------------------------------------------------------
    # Convert to projected GeoDataFrame
    # --------------------------------------------------------------
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326",
    ).to_crs(3035)

    # --------------------------------------------------------------
    # Europe boundary (for clipping + plotting background)
    # --------------------------------------------------------------
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

    # --------------------------------------------------------------
    # Hex grid + spatial join
    # --------------------------------------------------------------
    hex_grid = make_hex_grid(
        gdf,
        hex_diameter=HEX_DIAMETER_M,
        clip_geometry=europe,
        return_index=True,
    )

    gdf_hexed = gpd.sjoin(
        gdf,
        hex_grid,
        how="inner",
        predicate="within",
    )

    # --------------------------------------------------------------
    # Per-hex median forest age (early vs late)
    # --------------------------------------------------------------
    hex_grid["bm_bark_early"] = aggregate_by_cell(
        gdf_hexed,
        cell_col="hex_id",
        value_col="forest_age",
        period=PERIOD_EARLY,
        disturbance_col="wind_bark_beetle",
        forest_fraction_min=FOREST_FRACTION_MIN,
        disturbance_min=DISTURBANCE_MIN,
        min_pixels=MIN_PIXELS_MEDIAN,
        reducer="median",
    )

    hex_grid["bm_bark_late"] = aggregate_by_cell(
        gdf_hexed,
        cell_col="hex_id",
        value_col="forest_age",
        period=PERIOD_LATE,
        disturbance_col="wind_bark_beetle",
        forest_fraction_min=FOREST_FRACTION_MIN,
        disturbance_min=DISTURBANCE_MIN,
        min_pixels=MIN_PIXELS_MEDIAN,
        reducer="median",
    )

    hex_grid["delta_forest_age_bark"] = (
        hex_grid["bm_bark_late"] - hex_grid["bm_bark_early"]
    )

    # --------------------------------------------------------------
    # Distributional shift (energy distance)
    # --------------------------------------------------------------
    hex_grid["energy_dist_bark"] = distribution_distance_by_cell(
        gdf_hexed,
        cell_col="hex_id",
        value_col="forest_age",
        period1=PERIOD_EARLY,
        period2=PERIOD_LATE,
        disturbance_col="wind_bark_beetle",
        forest_fraction_min=FOREST_FRACTION_MIN,
        disturbance_min=DISTURBANCE_MIN,
        min_pixels=MIN_PIXELS_ED,
        metric="energy",
    )

    return {
        "hex_grid": hex_grid,
        "europe_gdf": europe,
        "periods": {
            "early": PERIOD_EARLY,
            "late": PERIOD_LATE,
        },
    }


# ---------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------

results = run_figure2_analysis()

# ---------------------------------------------------------------------
# Plotting — Figure 2 (consumes `results`)
# ---------------------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import linregress

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 16,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "axes.linewidth": 0.5,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "text.usetex": True,
})


def plot_figure2(results, out_path):
    hex_grid = results["hex_grid"]
    europe_gdf = results["europe_gdf"]

    fig, axs = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)

    # ===============================================================
    # (a) Natural disturbance — energy distance map
    # ===============================================================
    europe_gdf.boundary.plot(ax=axs[0, 0], linewidth=0.5, color="lightgrey")

    hex_grid.dropna(subset=["energy_dist_bark"]).plot(
        column="energy_dist_bark",
        ax=axs[0, 0],
        cmap="afmhot_r",
        edgecolor="none",
        legend=True,
        vmin=0,
        vmax=6,
        legend_kwds={
            "label": r"$\mathrm{ED}(P_{2011{-}2016},\ P_{2017{-}2023})\ \mathrm{[years]}$",
            "shrink": 0.6,
        },
    )

    axs[0, 0].set_title("Natural Disturbance")
    axs[0, 0].set_axis_off()

    # ===============================================================
    # (b) Harvest — energy distance map
    # ===============================================================
    europe_gdf.boundary.plot(ax=axs[0, 1], linewidth=0.5, color="lightgrey")

    hex_grid.dropna(subset=["energy_dist_harvest"]).plot(
        column="energy_dist_harvest",
        ax=axs[0, 1],
        cmap="afmhot_r",
        edgecolor="none",
        legend=True,
        vmin=0,
        vmax=6,
        legend_kwds={
            "label": r"$\mathrm{ED}(P_{2011{-}2016},\ P_{2017{-}2023})\ \mathrm{[years]}$",
            "shrink": 0.6,
        },
    )

    axs[0, 1].set_title("Harvest")
    axs[0, 1].set_axis_off()

    # ===============================================================
    # (c) ΔAge vs energy distance (Natural disturbance)
    # ===============================================================
    ax_pdf = axs[1, 0]

    hex_data = hex_grid.dropna(
        subset=["energy_dist_bark", "delta_forest_age_bark"]
    )

    pos = hex_data[hex_data["delta_forest_age_bark"] > 0]
    neg = hex_data[hex_data["delta_forest_age_bark"] < 0]

    hb = ax_pdf.hexbin(
        hex_data["energy_dist_bark"],
        hex_data["delta_forest_age_bark"],
        gridsize=70,
        cmap="YlGnBu",
        linewidths=0.2,
        mincnt=1,
        bins="log",
    )

    cb = plt.colorbar(hb, ax=ax_pdf)
    cb.set_label("Log-scaled Count of Hexagons [-]")

    # Regression arrows
    for data, linestyle in [(pos, "-"), (neg, "--")]:
        slope, intercept, *_ = linregress(
            data["energy_dist_bark"], data["delta_forest_age_bark"]
        )
        x0, x1 = data["energy_dist_bark"].quantile([0.0, 0.95])
        y0, y1 = slope * x0 + intercept, slope * x1 + intercept

        ax_pdf.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="->",
                linewidth=2.5,
                color="black",
                linestyle=linestyle,
            ),
        )

    ax_pdf.axhline(0, color="red", linestyle="dashed", linewidth=2)
    ax_pdf.set_xlim(0, 7)
    ax_pdf.set_ylim(-70, 70)

    ax_pdf.set_xlabel(
        r"$\mathrm{ED}(P_{2011{-}2016},\ P_{2017{-}2023})\ \mathrm{[years]}$"
    )
    ax_pdf.set_ylabel(r"$\Delta$ Forest Age [years]")
    ax_pdf.set_title("Natural Disturbance")

    ax_pdf.annotate(
        "Older forests targeted",
        xy=(6, 5),
        xytext=(6, 30),
        arrowprops=dict(arrowstyle="<-", linewidth=2, color="#d95f02"),
        ha="center",
        color="#d95f02",
        fontweight="bold",
    )

    ax_pdf.annotate(
        "Younger forests targeted",
        xy=(6, -5),
        xytext=(6, -35),
        arrowprops=dict(arrowstyle="<-", linewidth=2, color="#7570b3"),
        ha="center",
        color="#7570b3",
        fontweight="bold",
    )

    # ===============================================================
    # (d) Early vs late forest age (Natural disturbance)
    # ===============================================================
    ax_scatter = axs[1, 1]

    hex_data = hex_grid.dropna(subset=["bm_bark_early", "bm_bark_late"])

    min_val = min(hex_data["bm_bark_early"].min(), hex_data["bm_bark_late"].min())
    max_val = max(hex_data["bm_bark_early"].max(), hex_data["bm_bark_late"].max())

    ax_scatter.plot(
        [min_val, max_val], [min_val, max_val],
        color="black", linewidth=1.5
    )

    hb = ax_scatter.hexbin(
        hex_data["bm_bark_early"],
        hex_data["bm_bark_late"],
        gridsize=50,
        cmap="YlGnBu",
        linewidths=0.2,
        mincnt=1,
        bins="log",
    )

    cb = plt.colorbar(hb, ax=ax_scatter)
    cb.set_label("Log-scaled Count of Hexagons [-]")

    above_frac = (hex_data["bm_bark_late"] > hex_data["bm_bark_early"]).sum()
    below_frac = (hex_data["bm_bark_late"] < hex_data["bm_bark_early"]).sum()
    
    ax_scatter.text(
            0.6, 0.1,
            rf"$\begin{{array}}{{c}} \mathrm{{{above_frac}\ above\ 1:1\ (increase)}} \\ \mathrm{{{below_frac}\ below\ 1:1\ (decrease)}} \end{{array}}$",
            transform=ax_scatter.transAxes,
            fontsize=11,
            bbox=dict(facecolor='white', edgecolor='black')
        )


    ax_scatter.set_xlabel("2010 Forest Age disturbed in 2011-2016 [years]")
    ax_scatter.set_ylabel("2010 Forest Age disturbed in 2017-2023 [years]")
    ax_scatter.set_title("Natural Disturbance")

    # ---------------------------------------------------------------
    # Panel labels
    # ---------------------------------------------------------------
    for label, ax in zip(["(a)", "(b)", "(c)", "(d)"], axs.flat):
        ax.text(
            -0.1,
            1.05,
            label,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            va="bottom",
        )

    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    
plot_figure2(results, "fig2.png")
