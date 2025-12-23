# workflows/run_forecast_by_genus.py
import numpy as np

from analysis.forecasting import forecast_disturbance_area_best_fit
from analysis.forecasting_genus import forecast_by_genus


def run_forecast_by_genus(
    *,
    area_pivot_bark,
    area_pivot_harv,
    hex_grid,
    hex_genus,
    years_hist,
    years_future,
    n_sim,
    n_models,
    taylor_bark_global,
    taylor_harv_global,
    taylor_bark_by_region,
    taylor_harv_by_region,
):

    disturbance_types = {
        "wind_bark_beetle": (
            area_pivot_bark,
            taylor_bark_global,
            taylor_bark_by_region,
        ),
        "harvest": (
            area_pivot_harv,
            taylor_harv_global,
            taylor_harv_by_region,
        ),
    }

    # one biomass model per simulation
    biomass_model_ids = {
        dist: np.random.choice(n_models, size=n_sim, replace=True)
        for dist in disturbance_types
    }

    results = {}

    for disturbance, (area_pivot, taylor_global, taylor_region) in disturbance_types.items():

        results[disturbance] = forecast_by_genus(
            area_pivot=area_pivot,
            hex_grid=hex_grid,
            hex_genus=hex_genus,
            disturbance=disturbance,
            years_hist=years_hist,
            years_future=years_future,
            n_sim=n_sim,
            biomass_model_ids=biomass_model_ids[disturbance],
            n_models=n_models,
            taylor_global=taylor_global,
            taylor_by_region=taylor_region,
            forecast_area_fn=forecast_disturbance_area_best_fit,
            genus_mode="dominant",   # or "fractional"
        )

    print("✔ Genus-resolved forecast completed.")
    return results
