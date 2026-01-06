from __future__ import annotations

import gc
import logging
import numpy as np
import pandas as pd

from forecasting_pipeline.config import ForecastConfig
from forecasting_pipeline.io import ensure_outdir, read_inputs
from forecasting_pipeline.preprocessing import preprocess_df, apply_year_polygons_filter, add_pixel_area_km2
from forecasting_pipeline.spatial import to_points_gdf, make_hex_and_join_points, attach_regions
from forecasting_pipeline.biomass import (
    get_consistent_valid_hex_ids,
    biomass_by_disturbance_with_mask,
    attach_biomass_quantiles_to_hexgrid,
)
from forecasting_pipeline.area import annual_area_by_disturbance, pivot_area_timeseries
from forecasting_pipeline.montecarlo import fit_taylor_models
from forecasting_pipeline.genus import compute_hex_genus_fractions
from forecasting_pipeline.montecarlo_genus import run_hex_level_simulations_by_genus


log = logging.getLogger(__name__)


def main(cfg: ForecastConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # genus mapping (keep here or move into config)
    genus_groups = {
        "Spruce": [1],
        "Other needleleaf": [0, 2, 5],
        "Broadleaf": [3, 4, 6],
    }
    genus_names = list(genus_groups.keys())

    ensure_outdir(cfg.outdir)
    biomass_cols = cfg.biomass_cols()

    # ---- load with genus column included
    log.info("Loading inputs")
    df = read_inputs(cfg.parquet_path, biomass_cols)
    if "genus" not in df.columns:
        raise ValueError("Missing 'genus' column. Use a parquet that includes genus.")

    # ---- preprocess
    log.info("Preprocessing")
    df = preprocess_df(df, biomass_cols, years_min=int(cfg.years_all_hist.min()))
    df = apply_year_polygons_filter(df, str(cfg.polys_gpkg), crs=cfg.crs_latlon, drop=True)
    df = add_pixel_area_km2(df, cfg.pixel_res_deg)

    # ---- project to 3035 points
    gdf_points = to_points_gdf(df, cfg.crs_latlon, cfg.crs_europe)
    del df
    gc.collect()

    # ---- hex + join
    log.info("Hex grid + join")
    # If you want the Europe clip like the other runner, call load_europe_boundary and pass clip geometry.
    # Here we keep it minimal: you can still pass a clip geometry if needed.
    # make_hex_and_join_points in the earlier refactor expects europe geometry; if your version requires it, reuse that.
    # For now, assume your make_hex_and_join_points signature is (gdf_points, europe, hex_diameter_m).
    #
    # If your current module requires europe: import load_europe_boundary and create it.
    from forecasting_pipeline.spatial import load_europe_boundary
    europe = load_europe_boundary(str(cfg.world_gpk), cfg.crs_europe)

    hex_grid, gdf_joined = make_hex_and_join_points(gdf_points, europe, cfg.hex_diameter_m)
    del gdf_points
    gc.collect()

    # ---- genus fractions per hex
    log.info("Computing dominant genus per hex")
    hex_genus = compute_hex_genus_fractions(gdf_joined, genus_groups)

    # ---- biomass medians (same as non-genus)
    log.info("Biomass medians (early vs late) with consistent valid-hex masks")
    period_early = (2011, 2016)
    period_late = (2017, 2023)

    for disturbance in cfg.disturbance_types:
        valid_hex = get_consistent_valid_hex_ids(
            gdf_joined,
            disturbance,
            period_early,
            period_late,
            min_pixels=cfg.min_pixels_per_period,
            min_forest_frac=cfg.min_forest_frac,
            min_dist_frac=cfg.min_dist_frac,
        )

        bm_early = biomass_by_disturbance_with_mask(
            gdf_joined,
            disturbance,
            period_early[0],
            period_early[1],
            valid_hex,
            biomass_cols,
            min_forest_frac=cfg.min_forest_frac,
            min_dist_frac=cfg.min_dist_frac,
        )
        bm_late = biomass_by_disturbance_with_mask(
            gdf_joined,
            disturbance,
            period_late[0],
            period_late[1],
            valid_hex,
            biomass_cols,
            min_forest_frac=cfg.min_forest_frac,
            min_dist_frac=cfg.min_dist_frac,
        )
        hex_grid = attach_biomass_quantiles_to_hexgrid(hex_grid, disturbance, bm_early, bm_late, biomass_cols)

    # ---- regions (for regional Taylor fallback)
    log.info("Attach ecoregions")
    hex_grid = attach_regions(hex_grid, str(cfg.eco_path), cfg.crs_europe)

    # ---- annual area and pivots
    log.info("Annual disturbed area per hex")
    area_frames = {}
    for disturbance in cfg.disturbance_types:
        area_frames[disturbance] = annual_area_by_disturbance(
            gdf_joined,
            disturbance,
            start_year=1985,
            end_year=2023,
            min_forest_frac=0.0,
            min_dist_frac=0.0,
        )

    # free pixel join ASAP
    del gdf_joined
    gc.collect()

    years_all_hist = pd.Index(np.arange(1985, 2024))
    years_hist_fit = np.arange(2008, 2024)
    years_future = np.arange(2024, 2041)

    area_pivots_fit = {}
    for disturbance in cfg.disturbance_types:
        pivot_full = pivot_area_timeseries(area_frames[disturbance], years_all_hist)
        area_pivots_fit[disturbance] = pivot_full.loc[:, years_hist_fit]
        del pivot_full
    gc.collect()

    # ---- Taylor fits
    log.info("Fitting Taylor (global + by region)")
    taylor_global = {}
    taylor_by_region = {}
    for disturbance in cfg.disturbance_types:
        tg, tr = fit_taylor_models(area_pivots_fit[disturbance], years_hist_fit, hex_grid)
        taylor_global[disturbance] = tg
        taylor_by_region[disturbance] = tr

    # ---- Monte Carlo by genus
    log.info("Monte Carlo by genus (dominant genus routing)")
    rng = np.random.default_rng(cfg.seed)
    S = cfg.n_sim
    T_future = len(years_future)

    global_early_genus = {d: {g: np.zeros((T_future, S), dtype=np.float32) for g in genus_names} for d in cfg.disturbance_types}
    global_late_genus  = {d: {g: np.zeros((T_future, S), dtype=np.float32) for g in genus_names} for d in cfg.disturbance_types}

    for disturbance in cfg.disturbance_types:
        bm_ids = rng.integers(0, cfg.n_biomass_models, size=S, endpoint=False)

        early, late = run_hex_level_simulations_by_genus(
            disturbance=disturbance,
            area_pivot=area_pivots_fit[disturbance],
            years_hist_fit=years_hist_fit,
            years_future=years_future,
            hex_grid=hex_grid,
            hex_genus=hex_genus,
            genus_names=genus_names,
            biomass_model_ids=bm_ids,
            biomass_cols_count=cfg.n_biomass_models,
            taylor_global=taylor_global[disturbance],
            taylor_by_region=taylor_by_region[disturbance],
            n_sim=S,
        )
        for g in genus_names:
            global_early_genus[disturbance][g] = early[g]
            global_late_genus[disturbance][g]  = late[g]

    # ---- write genus CSV (matches your current output)
    log.info("Writing genus-resolved CSV")
    rows = []
    period_len = int(years_future[-1] - years_future[0])

    label_map = {
        "wind_bark_beetle": "Natural Disturbance",
        "harvest": "Harvest",
    }

    for disturbance, label in label_map.items():
        for gname in genus_names:
            ge = global_early_genus[disturbance][gname]
            gl = global_late_genus[disturbance][gname]

            cum_early = ge.sum(axis=0)
            cum_late = gl.sum(axis=0)

            for sim in range(S):
                rows.append(dict(simulation=sim, disturbance=label, period="Early", genus=gname, cumulative_loss=float(cum_early[sim])))
                rows.append(dict(simulation=sim, disturbance=label, period="Late",  genus=gname, cumulative_loss=float(cum_late[sim])))

    df_out = pd.DataFrame(rows)
    df_out["cumulative_loss"] /= period_len

    out_path = cfg.outdir / "cumulative_biomass_loss_forecast_by_genus.csv"
    df_out.to_csv(out_path, index=False)
    log.info("Done. Wrote: %s", out_path)


if __name__ == "__main__":
    main(ForecastConfig())
