# analysis/forecasting.py
# ================================================================
"""
Core forecasting logic for disturbance-driven biomass loss.

Pure numerical / geospatial routines:
- hex aggregation
- Taylor's law
- area forecasting
- Monte Carlo propagation

No I/O, no plotting.
"""
# ================================================================

import numpy as np
import pandas as pd

from scipy.optimize import curve_fit
from scipy.stats import linregress


# ----------------------------------------------------------------
# Area & smoothing
# ----------------------------------------------------------------

def moving_average(x, window):
    x = np.asarray(x, dtype=float)
    if window <= 1 or x.size == 0:
        return x
    pad = window // 2
    x_pad = np.pad(x, (pad, window - pad - 1), mode="edge")
    return np.convolve(x_pad, np.ones(window), mode="valid") / window


# ----------------------------------------------------------------
# Taylor's law
# ----------------------------------------------------------------

def fit_taylor_temporal(area_hex_by_year, years):
    M = area_hex_by_year.loc[:, years]
    mean = M.mean(axis=1)
    var = M.var(axis=1, ddof=1)

    mask = (mean > 0) & (var > 0)
    if mask.sum() < 2:
        return 1.0, 1.0, 0.0, mask.sum()

    slope, intercept, r, *_ = linregress(
        np.log(mean[mask]),
        np.log(var[mask]),
    )
    return np.exp(intercept), slope, r**2, mask.sum()


def fit_taylor_by_region(area_pivot, years, hex_grid, region_col="region_id"):
    out = {}
    for region, idx in hex_grid.groupby(region_col).groups.items():
        if pd.isna(region):
            continue
        sub = area_pivot.loc[area_pivot.index.intersection(hex_grid.index[idx])]
        if sub.empty:
            continue
        out[region] = fit_taylor_temporal(sub, years)
    return out


# ----------------------------------------------------------------
# Forecast models
# ----------------------------------------------------------------

def decay_func(x, a, b, c, x0):
    return a * np.exp(-b * (x - x0)) + c


def compute_aic(y_true, y_pred, k):
    resid = y_true - y_pred
    sse = max(np.sum(resid**2), 1e-12)
    n = len(y_true)
    return n * np.log(sse / n) + 2 * k


def meanvar_to_lognormal(mean, var):
    m2 = mean**2
    mu = np.log(m2 / np.sqrt(var + m2))
    sigma = np.sqrt(np.log1p(var / m2))
    return mu, sigma


def forecast_disturbance_area(
    years_hist,
    area_hist,
    years_future,
    *,
    smoothing_window,
    taylor_params,
    n_sim,
):
    y = moving_average(area_hist, smoothing_window)
    x = np.asarray(years_hist)

    # --- linear ---
    coef = np.polyfit(x, y, 1)
    y_fit = np.polyval(coef, x)
    y_fore_lin = np.polyval(coef, years_future)
    aic_lin = compute_aic(y, y_fit, 2)

    models = {"linear": y_fore_lin}
    scores = {"linear": aic_lin}

    # --- decay ---
    try:
        x0 = x[-1]
        popt, _ = curve_fit(
            lambda t, a, b, c: decay_func(t, a, b, c, x0),
            x, y, maxfev=10000
        )
        y_fit_d = decay_func(x, *popt, x0)
        y_fore_d = decay_func(years_future, *popt, x0)
        if np.all(np.diff(y_fore_d) <= 0):
            scores["decay"] = compute_aic(y, y_fit_d, 3)
            models["decay"] = y_fore_d
    except RuntimeError:
        pass

    best = min(scores, key=scores.get)
    mean_forecast = np.clip(models[best], 1e-12, None)

    # --- Taylor variance ---
    A, b = taylor_params
    var = np.maximum(A * mean_forecast**b, 1e-18)
    mu, sigma = meanvar_to_lognormal(mean_forecast, var)

    sims = np.random.lognormal(
        mu[:, None], sigma[:, None],
        size=(len(years_future), n_sim)
    )

    return mean_forecast, sims, best
