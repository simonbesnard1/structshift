from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd

from structshift.analysis.forecasting import (
    fit_taylor_temporal,
    fit_taylor_by_region,
    forecast_disturbance_area,
)


def fit_taylor_models(
    area_pivot: pd.DataFrame,
    years_hist_fit: np.ndarray,
    hex_grid: gpd.GeoDataFrame,
) -> tuple[tuple[float, float], dict]:
    # global temporal Taylor
    A, b, *_ = fit_taylor_temporal(area_pivot, years=years_hist_fit)

    # regional Taylor
    by_region = fit_taylor_by_region(area_pivot, years_hist_fit, hex_grid)

    return (A, b), by_region


def run_hex_level_simulations(
    *,
    disturbance: str,
    area_pivot: pd.DataFrame,           # index=hex, columns=years_hist_fit
    years_hist_fit: np.ndarray,
    years_future: np.ndarray,
    hex_grid: gpd.GeoDataFrame,
    biomass_model_ids: np.ndarray,      # shape (S,)
    biomass_cols_count: int,
    taylor_global: tuple[float, float],
    taylor_by_region: dict,
    n_sim: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Returns:
      global_early: (T_future, S)
      global_late:  (T_future, S)
      hex_records: list of dicts for mapping deltas
    """
    S = n_sim
    T_future = len(years_future)

    global_early = np.zeros((T_future, S), dtype=np.float32)
    global_late = np.zeros((T_future, S), dtype=np.float32)
    hex_records: list[dict] = []

    for hex_id, row in area_pivot.iterrows():
        area_hist = row.reindex(years_hist_fit, fill_value=0.0).values
        if area_hist.sum() == 0:
            continue

        region_val = hex_grid.loc[hex_id, "region_id"]
        if isinstance(region_val, pd.Series):
            region_val = region_val.iloc[0]
        region_id = region_val

        if pd.isna(region_id):
            A, b = taylor_global
        else:
            params = taylor_by_region.get(region_id)
            if params is None:
                A, b = taylor_global
            else:
                A, b, *_ = params

        _, sims, _, _ = forecast_disturbance_area(
            years_hist_fit,
            area_hist,
            years_future,
            n_sim=S,
            smoothing_window=5,
            criterion="AIC",
            fit_start_year=int(years_hist_fit[0]),
            fit_end_year=int(years_hist_fit[-1]),
            clip_max=None,
            taylor_params=(A, b),
        )
        # sims: (T_future, S)

        # read biomass medians for this hex for all biomass models
        bm_early_all = np.array(
            [hex_grid.loc[hex_id, f"bm_{disturbance}_early_m{i}"] for i in range(biomass_cols_count)],
            dtype=float,
        )
        bm_late_all = np.array(
            [hex_grid.loc[hex_id, f"bm_{disturbance}_late_m{i}"] for i in range(biomass_cols_count)],
            dtype=float,
        )
        if np.all(np.isnan(bm_early_all)) and np.all(np.isnan(bm_late_all)):
            continue

        bm_early = np.nan_to_num(bm_early_all[biomass_model_ids], nan=0.0)  # (S,)
        bm_late = np.nan_to_num(bm_late_all[biomass_model_ids], nan=0.0)    # (S,)

        biomass_early = (sims * bm_early[None, :]).astype(np.float32)
        biomass_late = (sims * bm_late[None, :]).astype(np.float32)

        global_early += biomass_early
        global_late += biomass_late

        med_early = np.median(biomass_early, axis=1)
        med_late = np.median(biomass_late, axis=1)

        hex_records.append(
            dict(
                hex_id=hex_id,
                disturbance=disturbance,
                biomass_early_sum=float(med_early.sum()),
                biomass_late_sum=float(med_late.sum()),
            )
        )

    return global_early, global_late, hex_records


def make_boxplot_frames(
    global_early: dict[str, np.ndarray],
    global_late: dict[str, np.ndarray],
    years_future: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    period_len = int(years_future[-1] - years_future[0])

    label_map = {
        "wind_bark_beetle": "Natural Disturbance",
        "harvest": "Harvest",
    }

    for disturbance, label in label_map.items():
        ge = global_early[disturbance]
        gl = global_late[disturbance]
        S = ge.shape[1]

        cum_early = ge.sum(axis=0)
        cum_late = gl.sum(axis=0)

        for sim in range(S):
            rows.append(dict(simulation=sim, disturbance=label, period="Early", cumulative_loss=float(cum_early[sim])))
            rows.append(dict(simulation=sim, disturbance=label, period="Late", cumulative_loss=float(cum_late[sim])))

    df_box = pd.DataFrame(rows)
    df_box["cumulative_loss"] /= period_len

    summary = (
        df_box.groupby(["disturbance", "period"])["cumulative_loss"]
        .agg(
            median="median",
            q5=lambda x: x.quantile(0.05),
            q95=lambda x: x.quantile(0.95),
        )
        .reset_index()
    )
    return df_box, summary
