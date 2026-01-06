#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 1 — Analysis pipeline

Computes:
1. Age × biomass disturbance fraction matrices (early vs late period)
2. Difference matrices (late − early)
3. Age-selectivity time series with ensemble uncertainty

Author: Simon Besnard

"""

from pathlib import Path
import pandas as pd
import numpy as np

from structshift.analysis.ensemble import EnsembleReducer
from structshift.analysis.spatial_filter import YearPolygonExcluder
from structshift.analysis.age_biomass_bins import AgeBiomassBinner
from structshift.analysis.age_selectivity import AgeSelectivityAnalyzer


# ---------------------------------------------------------------------
# Configuration (paper-frozen)
# ---------------------------------------------------------------------

DATA_PATH = Path(
    "/home/simon/hpc_home/projects/coupling_demography_dist/data/data_extraction/chunks/disturbance_chunk_91.csv"
)

POLYGON_MASK = Path(
    "/home/simon/glm1/person/besnard/coupling_demography_dist/data/bounding_box_filter_years_4326.gpkg"
)

START_YEAR = 2011
FOREST_FRACTION_MIN = 0.3
DISTURBANCE_MIN = 0.5

DISTURBANCE_TYPES = {
    "Natural Disturbance": "wind_bark_beetle",
    "Harvest": "harvest",
}

TIME_SLICES = {
    "2011–2016": (2011, 2016),
    "2017–2023": (2017, 2023),
}

AGE_BINS = {
    "0–20": (0, 20),
    "21–40": (20, 40),
    "41–60": (40, 60),
    "61–80": (60, 80),
    "81–100": (80, 100),
    "100–120": (100, 120),
    ">120": (120, np.inf),
}

BIOMASS_BINS = AGE_BINS  # same binning for Fig. 1

AGE_CLASSES_TS = {
    "1–60": (1, 60),
    ">60": (61, np.inf),
}


# ---------------------------------------------------------------------
# Load & preprocess
# ---------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """Load disturbance table and apply basic filters."""
    use_cols = (
        ["latitude", "longitude", "time", "harvest", "wind_bark_beetle", "forest_fraction"]
        + [f"forest_age_gami_2010_m{i}" for i in range(20)]
        + [f"biomass_m{i}" for i in range(20)]
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

def run_figure1_analysis() -> dict:
    """
    Run all analysis required for Figure 1.

    Returns
    -------
    dict
        {
            "fraction_matrices": nested dict [period][disturbance],
            "difference_matrices": dict [disturbance],
            "age_selectivity_ts": dict [disturbance]
        }
    """

    # Load data
    df = load_data()

    # Apply spatial/year exclusion
    excluder = YearPolygonExcluder(POLYGON_MASK)
    df = excluder.apply(df)

    # Reduce ensemble
    reducer = EnsembleReducer()
    df["biomass"] = reducer.median_biomass(df)
    df["forest_age"] = reducer.median_age(df)

    # Drop invalid rows
    df = df.dropna(subset=["biomass", "forest_age"])

    # ------------------------------------------------------------------
    # Age × biomass binning
    # ------------------------------------------------------------------

    binner = AgeBiomassBinner(
        age_bins=AGE_BINS,
        biomass_bins=BIOMASS_BINS,
    )

    fraction_matrices = {
        period: {
            label: binner.fraction_matrix(
                df,
                disturbance_col=col,
                year_range=years,
                forest_fraction_min=FOREST_FRACTION_MIN,
                disturbance_min=DISTURBANCE_MIN,
            )
            for label, col in DISTURBANCE_TYPES.items()
        }
        for period, years in TIME_SLICES.items()
    }

    # Difference matrices (late − early)
    diff_matrices = {
        label: (
            fraction_matrices["2017–2023"][label]
            - fraction_matrices["2011–2016"][label]
        )
        for label in DISTURBANCE_TYPES
    }

    # ------------------------------------------------------------------
    # Age selectivity through time
    # ------------------------------------------------------------------

    selectivity = AgeSelectivityAnalyzer(age_classes=AGE_CLASSES_TS)

    age_selectivity_ts = {
        label: selectivity.compute(
            df,
            disturbance_col=col,
            forest_fraction_min=FOREST_FRACTION_MIN,
            disturbance_min=DISTURBANCE_MIN,
        )
        for label, col in DISTURBANCE_TYPES.items()
    }

    return {
        "fraction_matrices": fraction_matrices,
        "difference_matrices": diff_matrices,
        "age_selectivity_ts": age_selectivity_ts,
    }


# ---------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------

results = run_figure1_analysis()

# ---------------------------------------------------------------------
# Plotting — Figure 1 (consumes `results`)
# ---------------------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator


# -----------------------------
# Plot style (paper-frozen)
# -----------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 16,
    "axes.titlesize": 15,
    "axes.labelsize": 15,
    "axes.linewidth": 0.5,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "text.usetex": True,
})


# -----------------------------
# Unpack results
# -----------------------------
diff_matrices = results["difference_matrices"]
age_ts = results["age_selectivity_ts"]

age_labels = list(AGE_BINS.keys())
biomass_labels = list(BIOMASS_BINS.keys())
disturbances = list(diff_matrices.keys())


# -----------------------------
# Create figure
# -----------------------------
norm = TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)

fig, axes = plt.subplots(
    2, 2,
    figsize=(12, 10),
    constrained_layout=True,
    gridspec_kw={"height_ratios": [1, 1]},
)

# =============================
# Top row: heatmaps
# =============================
for col, disturbance in enumerate(disturbances):
    ax = axes[0, col]
    data = np.ma.masked_invalid(diff_matrices[disturbance])

    im = ax.imshow(
        data,
        cmap="RdBu_r",
        norm=norm,
        origin="lower",
        aspect="auto",
    )

    # Hotspot contours
    hotspots = np.where(np.abs(data) > 2.5, 1, np.nan)
    ax.contour(hotspots, levels=[1], colors="black", linewidths=1.1)

    ax.set_xticks(range(len(age_labels)))
    ax.set_xticklabels(age_labels, rotation=45)
    ax.set_xlabel("Age class [years]")

    ax.set_yticks(range(len(biomass_labels)))
    ax.set_yticklabels(biomass_labels)
    ax.set_ylabel("AGC class [MgC ha$^{-1}$]")

    ax.set_title(disturbance, fontweight="bold")


# Colorbar
cbar = fig.colorbar(
    im,
    ax=axes[0, :],
    shrink=0.8,
    location="right",
    pad=0.02,
)
cbar.set_label("Change in disturbance [$\\%$]")
cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


# =============================
# Bottom row: age selectivity time series
# =============================
age_colors = {
    "1–60": "#969696",
    ">60": "#252525",
}

for col, disturbance in enumerate(disturbances):
    ax = axes[1, col]
    ts = age_ts[disturbance]
    years = sorted(ts.keys())

    for age_class, color in age_colors.items():
        med = [ts[y][age_class]["median"] for y in years]
        p5 = [ts[y][age_class]["p5"] for y in years]
        p95 = [ts[y][age_class]["p95"] for y in years]

        # same smoothing as original script
        med = pd.Series(med).rolling(3, center=True, min_periods=1).mean()
        p5 = pd.Series(p5).rolling(3, center=True, min_periods=1).mean()
        p95 = pd.Series(p95).rolling(3, center=True, min_periods=1).mean()

        ax.plot(years, med, marker="o", color=color, label=age_class)
        ax.fill_between(years, p5, p95, color=color, alpha=0.2)

    ax.set_ylim(0, 1)
    ax.set_xlabel("Year")
    ax.set_ylabel("Fraction [–]")
    ax.set_title(disturbance, fontweight="bold")
    ax.legend(frameon=False, loc="lower left")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# -----------------------------
# Panel labels
# -----------------------------
for label, ax in zip(["(a)", "(b)", "(c)", "(d)"], axes.flat):
    ax.text(
        -0.1, 1.05, label,
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        va="bottom",
    )


# -----------------------------
# Save figure
# -----------------------------
#plt.savefig("fig1.png", dpi=300)
#plt.close(fig)

