# analysis/forecasting_genus.py
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Biomass application
# ---------------------------------------------------------------------
def apply_biomass_projection(
    sims: np.ndarray,                 # (T, S)
    bm_early_all: np.ndarray,          # (N_MODELS,)
    bm_late_all: np.ndarray,           # (N_MODELS,)
    biomass_model_ids: np.ndarray,     # (S,)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert disturbed-area simulations into biomass-loss simulations.

    Returns
    -------
    biomass_early, biomass_late : arrays (T, S)
    """
    bm_early = np.nan_to_num(
        bm_early_all[biomass_model_ids], nan=0.0
    )
    bm_late = np.nan_to_num(
        bm_late_all[biomass_model_ids], nan=0.0
    )

    biomass_early = sims * bm_early[None, :]
    biomass_late  = sims * bm_late[None, :]

    return biomass_early.astype(np.float32), biomass_late.astype(np.float32)


# ---------------------------------------------------------------------
# Genus-resolved forecast
# ---------------------------------------------------------------------
def forecast_by_genus(
    *,
    area_pivot: pd.DataFrame,                  # hex × year
    hex_grid: pd.DataFrame,                    # must contain region_id
    hex_genus: pd.DataFrame,                   # dominant_genus or fractions
    disturbance: str,
    years_hist: np.ndarray,
    years_future: np.ndarray,
    n_sim: int,
    biomass_model_ids: np.ndarray,
    n_models: int,
    taylor_global: tuple[float, float],
    taylor_by_region: dict,
    forecast_area_fn,
    genus_mode: str = "dominant",               # "dominant" | "fractional"
) -> dict[str, np.ndarray]:
    """
    Run genus-resolved biomass-loss forecasts for one disturbance type.

    Returns
    -------
    dict genus -> array (T_future, S)
    """

    T_future = len(years_future)
    S = n_sim

    # initialise output
    genus_names = hex_genus["dominant_genus"].unique()
    out = {
        g: np.zeros((T_future, S), dtype=np.float32)
        for g in genus_names
    }

    for hex_id, row in area_pivot.iterrows():

        area_hist = row.reindex(years_hist, fill_value=0.0).values
        if area_hist.sum() == 0:
            continue

        # --------------------------------------------------
        # Taylor parameters (regional → global fallback)
        # --------------------------------------------------
        region_id = hex_grid.loc[hex_id, "region_id"]

        if pd.isna(region_id):
            A, b = taylor_global
        else:
            params = taylor_by_region.get(region_id)
            A, b = params[:2] if params is not None else taylor_global

        # --------------------------------------------------
        # Area forecast
        # --------------------------------------------------
        mean, sims, *_ = forecast_area_fn(
            years_hist,
            area_hist,
            years_future,
            n_sim=S,
            smoothing_window=5,
            taylor_params=(A, b),
        )

        # --------------------------------------------------
        # Biomass per model
        # --------------------------------------------------
        bm_early_all = np.array(
            [hex_grid.loc[hex_id, f"bm_{disturbance}_early_m{i}"]
             for i in range(n_models)]
        )
        bm_late_all = np.array(
            [hex_grid.loc[hex_id, f"bm_{disturbance}_late_m{i}"]
             for i in range(n_models)]
        )

        if np.all(np.isnan(bm_early_all)) and np.all(np.isnan(bm_late_all)):
            continue

        biomass_early, biomass_late = apply_biomass_projection(
            sims,
            bm_early_all,
            bm_late_all,
            biomass_model_ids,
        )

        # --------------------------------------------------
        # Genus attribution
        # --------------------------------------------------
        if hex_id not in hex_genus.index:
            continue

        g_row = hex_genus.loc[hex_id]

        if genus_mode == "dominant":
            gname = g_row["dominant_genus"]
            out[gname] += biomass_late

        elif genus_mode == "fractional":
            for gname in out:
                frac = g_row.get(gname, 0.0)
                if frac > 0:
                    out[gname] += frac * biomass_late

        else:
            raise ValueError(f"Unknown genus_mode: {genus_mode}")

    return out
