from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import pandas as pd
import geopandas as gpd

from structshift.analysis.forecasting import forecast_disturbance_area


def run_hex_level_simulations_by_genus(
    *,
    disturbance: str,
    area_pivot: pd.DataFrame,           # index=hex, columns=years_hist_fit
    years_hist_fit: np.ndarray,
    years_future: np.ndarray,
    hex_grid: gpd.GeoDataFrame,
    hex_genus: pd.DataFrame,            # index=hex_id, has dominant_genus (and optionally fractions)
    genus_names: list[str],
    biomass_model_ids: np.ndarray,      # shape (S,)
    biomass_cols_count: int,
    taylor_global: tuple[float, float],
    taylor_by_region: dict,
    n_sim: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Returns:
      early: genus -> (T_future, S)
      late:  genus -> (T_future, S)
    """
    S = int(n_sim)
    T_future = len(years_future)

    early = {g: np.zeros((T_future, S), dtype=np.float32) for g in genus_names}
    late  = {g: np.zeros((T_future, S), dtype=np.float32) for g in genus_names}

    # optional sanity (cheap): ensure columns are the expected years
    # if not np.array_equal(area_pivot.columns.to_numpy(), years_hist_fit):
    #     raise ValueError("area_pivot columns must equal years_hist_fit")

    for hex_id, row in area_pivot.iterrows():
        # If area_pivot is already restricted to years_hist_fit, this is enough:
        area_hist = row.to_numpy(dtype=float)
        if area_hist.sum() == 0:
            continue

        # genus routing
        if hex_id not in hex_genus.index:
            continue
        dom_genus = hex_genus.at[hex_id, "dominant_genus"]
        if not isinstance(dom_genus, str) or dom_genus not in early:
            continue

        # region Taylor params (fallback to global)
        region_val = hex_grid.loc[hex_id, "region_id"]
        if isinstance(region_val, pd.Series):
            region_val = region_val.iloc[0]

        if pd.isna(region_val):
            A, b = taylor_global
        else:
            params = taylor_by_region.get(region_val)
            if params is None:
                A, b = taylor_global
            else:
                A, b, *_ = params

        # forecast area sims: (T_future, S)
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

        # biomass per model for this hex
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
        bm_late  = np.nan_to_num(bm_late_all[biomass_model_ids],  nan=0.0)  # (S,)

        biomass_early = (sims * bm_early[None, :]).astype(np.float32)
        biomass_late  = (sims * bm_late[None, :]).astype(np.float32)

        # dominant genus accumulation
        early[dom_genus] += biomass_early
        late[dom_genus]  += biomass_late

        # Fraction-weighted alternative (replace the two lines above with):
        # for g in genus_names:
        #     w = float(hex_genus.at[hex_id, g]) if g in hex_genus.columns else 0.0
        #     if w > 0:
        #         early[g] += biomass_early * w
        #         late[g]  += biomass_late * w

    return early, late
