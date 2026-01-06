from __future__ import annotations

import gc
import logging
import numpy as np
import pandas as pd

from structshift.forecasting_pipeline import ForecastConfig
from structshift.forecasting_pipeline.io import ensure_outdir, read_inputs
from structshift.forecasting_pipeline.preprocessing import (
    preprocess_df,
    apply_year_polygons_filter,
    add_pixel_area_km2,
)
from structshift.forecasting_pipeline.spatial import (
    load_europe_boundary,
    to_points_gdf,
    make_hex_and_join_points,
    attach_regions,
)
from structshift.forecasting_pipeline.biomass import (
    get_consistent_valid_hex_ids,
    biomass_by_disturbance_with_mask,
    attach_biomass_quantiles_to_hexgrid,
)
from structshift.forecasting_pipeline.area import (
    annual_area_by_disturbance,
    pivot_area_timeseries,
)
from structshift.forecasting_pipeline.montecarlo import (
    fit_taylor_models,
    run_hex_level_simulations,
    make_boxplot_frames,
)
from structshift.forecasting_pipeline.outputs import write_outputs
from structshift.analysis.forecasting import forecast_disturbance_area


log = logging.getLogger(__name__)


def build_forecast_results_df(
    area_combined: pd.DataFrame,
    years_all_hist: np.ndarray,
    years_hist_fit: np.ndarray,
    years_future: np.ndarray,
    taylor_by_agent: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    area_types = {
        "Natural Disturbance": "wind_bark_beetle",
        "Harvest": "harvest",
    }

    all_results = []

    for label, disturbance_type in area_types.items():
        area_series_full = (
            area_combined.query(f"disturbance == '{disturbance_type}'")
            .groupby("year")["area_Mha"]
            .sum()
            .reindex(years_all_hist, fill_value=0.0)
        )

        area_hist_for_fit = area_series_full.reindex(years_hist_fit, fill_value=0.0).values

        forecast_mean, sims_area, _, _ = forecast_disturbance_area(
            years_hist_fit,
            area_hist_for_fit,
            years_future,
            n_sim=1000,
            smoothing_window=5,
            criterion="AIC",
            fit_start_year=int(years_hist_fit[0]),
            fit_end_year=int(years_hist_fit[-1]),
            clip_max=None,
            taylor_params=taylor_by_agent[disturbance_type],
        )

        forecast_summary = pd.DataFrame(
            {
                "median": np.median(sims_area, axis=1),
                "p5": np.percentile(sims_area, 5, axis=1),
                "p95": np.percentile(sims_area, 95, axis=1),
            },
            index=years_future,
        )

        hist_df = area_series_full.reset_index()
        hist_df.columns = ["time", "hist"]
        hist_df["time"] = hist_df["time"].astype(int)

        summary_df = forecast_summary.reset_index().rename(columns={"index": "time"})
        summary_df["time"] = summary_df["time"].astype(int)

        years_future_idx = pd.Index(years_future, name="time").astype(int)
        fm = pd.Series(forecast_mean, index=years_future_idx, name="forecast_mean")

        merged = hist_df.merge(summary_df, on="time", how="outer")
        merged = merged.merge(fm.reset_index(), on="time", how="left")
        merged = merged.sort_values("time").reset_index(drop=True)
        merged["label"] = label
        all_results.append(merged)

    return pd.concat(all_results, ignore_index=True)


def main(cfg: ForecastConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    ensure_outdir(cfg.outdir)
    biomass_cols = cfg.biomass_cols()

    log.info("Loading inputs")
    df = read_inputs(cfg.parquet_path, biomass_cols)

    log.info("Preprocessing")
    df = preprocess_df(df, biomass_cols, years_min=int(cfg.years_all_hist.min()))
    df = apply_year_polygons_filter(df, str(cfg.polys_gpkg), crs=cfg.crs_latlon, drop=False)
    df = add_pixel_area_km2(df, cfg.pixel_res_deg)

    log.info("Loading Europe boundary + projecting points")
    europe = load_europe_boundary(str(cfg.world_gpk), cfg.crs_europe)

    gdf_points = to_points_gdf(df, cfg.crs_latlon, cfg.crs_europe)
    del df
    gc.collect()

    log.info("Hex grid + join")
    hex_grid, gdf_joined = make_hex_and_join_points(gdf_points, europe, cfg.hex_diameter_m)
    del gdf_points
    gc.collect()

    # -------------------------
    # Biomass quantiles per hex
    # -------------------------
    log.info("Computing valid hex masks + biomass medians")
    period_early = (2011, 2017)
    period_late = (2018, 2024)

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

    log.info("Attach ecoregions")
    hex_grid = attach_regions(hex_grid, str(cfg.eco_path), cfg.crs_europe)

    # -------------------------
    # Annual disturbed area
    # -------------------------
    log.info("Computing annual disturbed area per hex")
    area_frames = {}
    for disturbance in cfg.disturbance_types:
        area_frames[disturbance] = annual_area_by_disturbance(
            gdf_joined,
            disturbance,
            start_year=int(cfg.years_all_hist.min()),
            end_year=int(cfg.years_all_hist.max() - 1),
            min_forest_frac=0.0,
            min_dist_frac=0.0,
        )

    area_combined = pd.concat(list(area_frames.values()), ignore_index=True)

    # drop heavy pixel-level joined data
    del gdf_joined
    gc.collect()

    log.info("Pivot area time series")
    years_all_hist = pd.Index(cfg.years_all_hist)
    area_pivots_full = {
        d: pivot_area_timeseries(area_frames[d], years_all_hist)
        for d in cfg.disturbance_types
    }
    area_pivots_fit = {
        d: area_pivots_full[d].loc[:, cfg.years_hist_fit]
        for d in cfg.disturbance_types
    }
    del area_pivots_full
    gc.collect()

    # -------------------------
    # Taylor fits
    # -------------------------
    log.info("Fitting Taylor (global + by region)")
    taylor_by_agent: dict[str, tuple[float, float]] = {}
    taylor_by_region: dict[str, dict] = {}

    for disturbance in cfg.disturbance_types:
        t_global, t_regional = fit_taylor_models(area_pivots_fit[disturbance], cfg.years_hist_fit, hex_grid)
        taylor_by_agent[disturbance] = t_global
        taylor_by_region[disturbance] = t_regional

    # -------------------------
    # Monte Carlo (hex-wise)
    # -------------------------
    log.info("Running Monte Carlo per hex (streaming)")
    rng = np.random.default_rng(cfg.seed)

    global_early = {}
    global_late = {}
    hex_records_all: list[dict] = []

    for disturbance in cfg.disturbance_types:
        biomass_model_ids = rng.integers(0, cfg.n_biomass_models, size=cfg.n_sim, endpoint=False)

        ge, gl, hex_records = run_hex_level_simulations(
            disturbance=disturbance,
            area_pivot=area_pivots_fit[disturbance],
            years_hist_fit=cfg.years_hist_fit,
            years_future=cfg.years_future,
            hex_grid=hex_grid,
            biomass_model_ids=biomass_model_ids,
            biomass_cols_count=cfg.n_biomass_models,
            taylor_global=taylor_by_agent[disturbance],
            taylor_by_region=taylor_by_region[disturbance],
            n_sim=cfg.n_sim,
        )
        global_early[disturbance] = ge
        global_late[disturbance] = gl
        hex_records_all.extend(hex_records)

    # -------------------------
    # Hex deltas for mapping
    # -------------------------
    biomass_sums = pd.DataFrame(hex_records_all)
    biomass_pivot = biomass_sums.pivot(index="hex_id", columns="disturbance")

    hex_grid["delta_biomass_bark"] = (
        biomass_pivot.get(("biomass_late_sum", "wind_bark_beetle"), pd.Series(0.0))
        - biomass_pivot.get(("biomass_early_sum", "wind_bark_beetle"), pd.Series(0.0))
    )

    hex_grid["delta_biomass_harvest"] = (
        biomass_pivot.get(("biomass_late_sum", "harvest"), pd.Series(0.0))
        - biomass_pivot.get(("biomass_early_sum", "harvest"), pd.Series(0.0))
    )

    # -------------------------
    # Boxplot frames
    # -------------------------
    df_boxplot, summary_box = make_boxplot_frames(global_early, global_late, cfg.years_future)

    # -------------------------
    # Pan-EU area forecast dataframe
    # -------------------------
    forecast_results_df = build_forecast_results_df(
        area_combined=area_combined,
        years_all_hist=cfg.years_all_hist,
        years_hist_fit=cfg.years_hist_fit,
        years_future=cfg.years_future,
        taylor_by_agent=taylor_by_agent,
    )

    # -------------------------
    # Write outputs
    # -------------------------
    write_outputs(
        outdir=cfg.outdir,
        df_boxplot=df_boxplot,
        summary_box=summary_box,
        hex_grid=hex_grid,
        forecast_results_df=forecast_results_df,
        filenames={
            "boxplot_parquet": cfg.out_boxplot_parquet,
            "summary_csv": cfg.out_summary_csv,
            "hex_gpkg": cfg.out_hex_gpkg,
            "forecast_csv": cfg.out_forecast_csv,
        },
    )

    log.info("Done. Outputs written to: %s", cfg.outdir)


if __name__ == "__main__":
    main(ForecastConfig())
