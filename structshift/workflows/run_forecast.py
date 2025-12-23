# workflows/run_forecast.py
# ================================================================
"""
Run full disturbance → biomass forecasting workflow.

Handles:
- I/O
- preprocessing
- orchestration
"""
# ================================================================

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from structshift.utils.area_calc import pixel_area_latlon_km2

from structshift.analysis.hexgrid import make_hex_grid
from analysis.forecasting import (
    fit_taylor_temporal,
    fit_taylor_by_region,
    forecast_disturbance_area,
)

# ------------------------------------------------
# Config
# ------------------------------------------------

N_SIM = 1000
N_MODELS = 20

YEARS_ALL = np.arange(1985, 2025)
YEARS_HIST = np.arange(2008, 2025)
YEARS_FUTURE = np.arange(2025, 2041)

MIN_FOREST_FRAC = 0.3
MIN_DIST_FRAC = 0.5

HEX_DIAMETER_M =  100_000

# ------------------------------------------------
# Paths
# ------------------------------------------------

BASE = Path("/home/besnard/projects/coupling_demography_dist")
PARQUET = BASE / "data/data_extraction/disturbance_data_combined_v2025-12.parquet"
ECO = BASE / "data/ancillary/biogeo_EU_2016.gpkg"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

# ------------------------------------------------
# Main workflow
# ------------------------------------------------

def run_forecast():

    biomass_cols = [f"biomass_m{i}" for i in range(N_MODELS)]

    # --------------------------------------------------------------
    # Load & preprocess
    # --------------------------------------------------------------
    df = pd.read_parquet(PARQUET)
    df["year"] = pd.to_datetime(df["time"]).dt.year
    df = df[df["year"] >= YEARS_ALL[0]]

    # biomass → carbon
    df[biomass_cols] = df[biomass_cols].where(df[biomass_cols] > 0) * 0.47

    # pixel area
    df["pixel_area_km2"] = pixel_area_latlon_km2(
        df.latitude.values, df.longitude.values
    )

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326",
    ).to_crs(3035)

    del df
    gc.collect()

    # --------------------------------------------------------------
    # Europe boundary
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
        crs="EPSG:4326",
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

    gdf = gpd.sjoin(gdf, hex_grid, how="inner", predicate="within")

    # --------------------------------------------------------------
    # Annual disturbed area per hex
    # --------------------------------------------------------------
    def annual_area(dist):
        sub = gdf[
            (gdf[dist] >= MIN_DIST_FRAC)
            & (gdf["forest_fraction"] >= MIN_FOREST_FRAC)
        ].copy()

        sub["area_Mha"] = (
            sub["forest_fraction"]
            * sub[dist]
            * sub["pixel_area_km2"]
            / 10_000
        )

        return (
            sub.groupby(["index_right", "year"])["area_Mha"]
            .sum()
            .unstack(fill_value=0)
        )

    area_bark = annual_area("wind_bark_beetle")
    area_harv = annual_area("harvest")

    # --------------------------------------------------------------
    # Taylor’s law (global + regional)
    # --------------------------------------------------------------
    A_bark, b_bark, *_ = fit_taylor_temporal(area_bark, YEARS_HIST)
    A_harv, b_harv, *_ = fit_taylor_temporal(area_harv, YEARS_HIST)

    taylor_bark_by_region = fit_taylor_by_region(
        area_bark, YEARS_HIST, hex_grid
    )
    taylor_harv_by_region = fit_taylor_by_region(
        area_harv, YEARS_HIST, hex_grid
    )

    taylor_global = {
        "wind_bark_beetle": (A_bark, b_bark),
        "harvest": (A_harv, b_harv),
    }

    taylor_regional = {
        "wind_bark_beetle": taylor_bark_by_region,
        "harvest": taylor_harv_by_region,
    }

    # --------------------------------------------------------------
    # Monte-Carlo forecasting (HEX LEVEL)
    # --------------------------------------------------------------
    T = len(YEARS_FUTURE)
    S = N_SIM

    global_sims = {
        "wind_bark_beetle": np.zeros((T, S), dtype=np.float32),
        "harvest": np.zeros((T, S), dtype=np.float32),
    }

    for disturbance, area_hex in [
        ("wind_bark_beetle", area_bark),
        ("harvest", area_harv),
    ]:

        for hex_id, row in area_hex.iterrows():

            hist = row.reindex(YEARS_HIST, fill_value=0).values
            if hist.sum() == 0:
                continue

            # --- select Taylor parameters ---
            region_id = hex_grid.loc[hex_id, "region_id"]

            if (
                pd.isna(region_id)
                or region_id not in taylor_regional[disturbance]
            ):
                A, b = taylor_global[disturbance]
            else:
                A, b, *_ = taylor_regional[disturbance][region_id]

            mean, sims, _ = forecast_disturbance_area(
                YEARS_HIST,
                hist,
                YEARS_FUTURE,
                smoothing_window=5,
                taylor_params=(A, b),
                n_sim=S,
            )

            global_sims[disturbance] += sims

    # --------------------------------------------------------------
    # Output structure
    # --------------------------------------------------------------
    results = {
        "Natural Disturbance": {
            "sims": global_sims["wind_bark_beetle"],
        },
        "Harvest": {
            "sims": global_sims["harvest"],
        },
    }

    print("Forecast completed successfully.")
    return results

run_forecast()
