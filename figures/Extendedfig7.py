import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
import matplotlib as mpl
from shapely.geometry import box
from scipy.optimize import curve_fit
from scipy.stats import linregress

params = {
    # font
    'font.family': 'serif',
    # 'font.serif': 'Times', #'cmr10',
    'font.size': 18,
    # axes
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'axes.linewidth': 0.5,
    # ticks
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'xtick.major.width': 0.3,
    'ytick.major.width': 0.3,
    'xtick.minor.width': 0.3,
    'ytick.minor.width': 0.3,
    # legend
    'legend.fontsize': 16,
    # tex
    'text.usetex': True,
}

mpl.rcParams.update(params)

def AreaGridlatlon(lats,lons,res_lat,res_lon):
     ER          = 6378160 #Earth radius (m)
     londel      = np.abs(res_lon)
     lats1       = lats - res_lat/2.
     lats2       = lats + res_lat/2.
     areavec     = (np.pi/180)*ER**2 * np.abs(np.sin(lats1 * np.pi/180)-np.sin(lats2 * np.pi/180))*londel
     
     return areavec
 
def moving_average(series, window=10):
    """Centered moving average with edge handling. Works on 1D array-like."""
    s = pd.Series(series, dtype=float)
    return s.rolling(window=window, center=True, min_periods=1).mean().to_numpy()

def fit_taylor_temporal(area_hex_by_year: pd.DataFrame, years=None):
    """
    Temporal Taylor across hexes:
    - Input: rows = hex_id, cols = years, values = area (>=0)
    - For each hex: temporal mean & var across 'years' (all columns if None)
    - Regress log(var) ~ log(mean) across hexes.
    Returns: (A, b, r2, n)
    """
    M = area_hex_by_year if years is None else area_hex_by_year.loc[:, years]
    means = M.mean(axis=1)
    variances = M.var(axis=1, ddof=1)

    mask = (means > 0) & (variances > 0) & np.isfinite(means) & np.isfinite(variances)
    if mask.sum() < 2:
        # fallback: Poisson-ish
        return 1.0, 1.0, 0.0, int(mask.sum())

    log_mean = np.log(means[mask].to_numpy())
    log_var  = np.log(variances[mask].to_numpy())

    slope, intercept, r, _, _ = linregress(log_mean, log_var)
    A = np.exp(intercept)
    b = slope
    return A, b, r**2, int(mask.sum())

def meanvar_to_lognormal_params(mean_arr, var_arr):
    """Vectorized conversion of mean/variance to lognormal (mu, sigma)."""
    m = np.asarray(mean_arr, dtype=float)
    v = np.asarray(var_arr, dtype=float)
    m2 = m * m
    mu = np.log(m2 / np.sqrt(v + m2))
    sigma = np.sqrt(np.log1p(v / m2))
    return mu, sigma

def decay_func(x, a, b, c, x0):
    return a * np.exp(-b * (x - x0)) + c

def logistic_func(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

def gompertz_func(x, a, b, c, x0):
    return a * np.exp(-b * np.exp(-c * (x - x0)))

def compute_aic_bic(y_true, y_pred, k):
    resid = y_true - y_pred
    sse = np.sum(resid**2)
    n = len(y_true)
    if sse == 0:
        sse = 1e-10
    aic = n * np.log(sse / n) + 2 * k
    bic = n * np.log(sse / n) + k * np.log(n)
    return aic, bic


def forecast_disturbance_area_best_fit(
    years_hist,
    area_hist_raw,
    years_future,
    n_sim=1000,
    smoothing_window=5,
    criterion="AIC",
    clip_max=None,
    fit_start_year=None,
    fit_end_year=None,
    # NEW ↓ either provide taylor_params OR a per-hex matrix to compute them
    taylor_params=None,                    # tuple (A, b)
    area_hex_by_year_for_taylor=None,      # DataFrame: index=hex, cols=year
):
    """
    area_hist_raw: 1D array of historical totals (e.g., EU-wide or per-hex) for years_hist
    Taylor law is used only for uncertainty mapping (variance = A * mean^b).
    """
    # --- Smooth & slice ---
    area_hist_full = moving_average(area_hist_raw, window=smoothing_window)
    x_full = np.asarray(years_hist, dtype=int)

    if fit_start_year is not None or fit_end_year is not None:
        mask = np.ones_like(x_full, dtype=bool)
        if fit_start_year is not None: mask &= (x_full >= fit_start_year)
        if fit_end_year   is not None: mask &= (x_full <= fit_end_year)
        x, y = x_full[mask], area_hist_full[mask]
    else:
        x, y = x_full, area_hist_full

    x0 = x[-1]

    # --- Candidate models ---
    models = {}
    criteria_scores = {}
    residuals_all = {}

    # linear
    coeffs = np.polyfit(x, y, deg=1)
    y_fit = np.polyval(coeffs, x)
    y_fore = np.polyval(coeffs, years_future)
    aic, bic = compute_aic_bic(y, y_fit, k=2)
    models["linear"] = y_fore
    criteria_scores["linear"] = aic if criterion.upper()=="AIC" else bic
    residuals_all["linear"] = y - y_fit

    # decay (monotone decreasing)
    try:
        popt, _ = curve_fit(lambda xx, a, b, c: decay_func(xx, a, b, c, x0),
                            x, y, p0=(0.005, 0.05, 0.004))
        y_fit = decay_func(x, *popt, x0)
        y_fore = decay_func(np.asarray(years_future), *popt, x0)
        if np.all(np.diff(y_fore) <= 0):
            aic, bic = compute_aic_bic(y, y_fit, k=3)
            models["decay"] = y_fore
            criteria_scores["decay"] = aic if criterion.upper()=="AIC" else bic
            residuals_all["decay"] = y - y_fit
        else:
            criteria_scores["decay"] = np.inf
    except RuntimeError:
        criteria_scores["decay"] = np.inf

    # --- pick best ---
    best_model = min(criteria_scores, key=criteria_scores.get)
    mean_forecast = np.clip(models[best_model], 1e-12, clip_max if clip_max is not None else np.inf)

    # --- Taylor parameters ---
    if taylor_params is not None:
        A, b = taylor_params
    elif area_hex_by_year_for_taylor is not None:
        # Compute once from the provided hex×year matrix, using the *historical* window
        A, b, r2_taylor, n_pts = fit_taylor_temporal(area_hex_by_year_for_taylor, years=years_hist)
        # Optional: print/log r2_taylor, n_pts for QA
    else:
        # Sensible fallback (Poisson-ish)
        A, b = 1.0, 1.0

    variance = A * (mean_forecast ** b)
    variance = np.maximum(variance, 1e-18)  # guard

    # --- Lognormal sims ---
    mu, sigma = meanvar_to_lognormal_params(mean_forecast, variance)
    sims = np.random.lognormal(mean=mu[:, None], sigma=sigma[:, None], size=(len(years_future), n_sim))
    if clip_max is not None:
        sims = np.clip(sims, 0, clip_max)

    return mean_forecast, sims, best_model, criteria_scores


def create_true_hex_grid(gdf, hex_diameter=200000):  # 20km in meters
    xmin, ymin, xmax, ymax = gdf.total_bounds
    hex_radius = hex_diameter / 2
    hex_height = np.sqrt(3) * hex_radius
    dx = 3/2 * hex_radius
    dy = hex_height
    cols = int((xmax - xmin) / dx) + 2
    rows = int((ymax - ymin) / dy) + 2
    hexes = []
    for row in range(rows):
        for col in range(cols):
            x = xmin + col * dx
            y = ymin + row * dy
            if col % 2 == 1:
                y += dy / 2
            hex = Polygon([(x + hex_radius * np.cos(theta), y + hex_radius * np.sin(theta))
                           for theta in np.linspace(0, 2 * np.pi, 7)[:-1]])
            hexes.append(hex)
    return gpd.GeoDataFrame({'geometry': hexes}, crs="EPSG:3035")


# Step 1: Load your disturbance data
use_cols = ['latitude', 'longitude', 'time','harvest', 'wind_bark_beetle', 'forest_fraction'] + [f'biomass_m{i}' for i in range(20)]
df= pd.read_parquet('/misc/glm1/person/besnard/coupling_demography_dist/data/disturbance_data_combined.parquet', columns=use_cols)
df['time'] = pd.to_datetime(df['time'])
df["year"] = pd.to_datetime(df["time"]).dt.year
df = df[df["year"] >= 1985]
df = df[~(((df['year'].isin([2018, 2023])) & (df['latitude'] >= 65)))]

# Load polygons from GeoPackage
polys = gpd.read_file("/misc/glm1/person/besnard/coupling_demography_dist/data/bounding_box_filter_years_4326.gpkg")
if polys.crs is None:
    polys = polys.set_crs("EPSG:4326")
else:
    polys = polys.to_crs("EPSG:4326")

gdf_pts = gpd.GeoDataFrame(
    df.copy(),
    geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
    crs="EPSG:4326"
)

years_in_polys = polys["year"].unique()
pts_sub = gdf_pts[gdf_pts["year"].isin(years_in_polys)][["geometry", "year"]]
polys_sub = polys[["geometry", "year"]]

joined = gpd.sjoin(pts_sub, polys_sub, how="left", predicate="within")
to_drop_idx = joined.index[
    joined["index_right"].notna() & (joined["year_left"] == joined["year_right"])
]

df = df.drop(index=to_drop_idx).reset_index(drop=True)

biomass_cols = [col for col in df.columns if col.startswith("biomass")]
df[biomass_cols] = df[biomass_cols].where(df[biomass_cols] > 0)
for i in range(20):
    biomass_col = f"biomass_m{i}"
    df[biomass_col] = df[biomass_col] * 0.47
    
# Calculate pixel area
df['pixel_area_km2'] = AreaGridlatlon(df.latitude, df.longitude, 0.0008888888888888889, 0.0008888888888888889) / 1e6
      
world = gpd.read_file('/misc/glm1/person/besnard/coupling_demography_dist/data/ne_10m_admin_0_countries.zip')
europe_gdf = world[world['CONTINENT'] == 'Europe'].to_crs("EPSG:4326")
bbox = box(-20, 32, 45, 71)

# Ensure the bounding box and europe_gdf are both in EPSG:3035
bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs="EPSG:4326")
bbox_3035 = bbox_gdf.to_crs("EPSG:3035")
europe_gdf = europe_gdf.to_crs("EPSG:3035")

# Clip using reprojected bounding box
europe_gdf = gpd.clip(europe_gdf, bbox_3035)

# Exclude unwanted countries
exclude_countries = ['RUS', 'ISL']
europe_gdf = europe_gdf[~europe_gdf['ISO_A3'].isin(exclude_countries)]

# 2. Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")

# Reproject your data to EPSG:3035
gdf = gdf.to_crs(epsg=3035)

del df, gdf_pts, pts_sub, joined, polys_sub
import gc; gc.collect()

# 3. Create hex grid
hex_grid = create_true_hex_grid(gdf, hex_diameter=100000)

# 4. Spatial join
gdf_joined = gpd.sjoin(gdf, hex_grid, how="inner", predicate="within")

#gdf_joined = gpd.sjoin(gdf_3035, hex_grid, how="inner", predicate="within")

def get_consistent_valid_hex_ids(gdf_joined, disturbance_column, period1, period2,
                                  min_pixels=25, min_forest_frac=0.3):
    """
    Identify hexagons that contain at least `min_pixels` of disturbed pixels
    with forest_fraction ≥ `min_forest_frac` in both time periods.
    """
    def filter_hexes(df):
        filtered = df[df["forest_fraction"] >= min_forest_frac]
        pixel_count = filtered.groupby("index_right").size()
        valid = pixel_count >= min_pixels
        return valid[valid].index

    df1 = gdf_joined[
        (gdf_joined["year"].between(*period1)) &
        (gdf_joined['forest_fraction'] >= min_forest_frac) &
        (gdf_joined[disturbance_column] >= 0.5)
    ]
    df2 = gdf_joined[
        (gdf_joined["year"].between(*period2)) &
        (gdf_joined['forest_fraction'] > min_forest_frac) &
        (gdf_joined[disturbance_column] >= 0.5)
    ]

    valid1 = filter_hexes(df1)
    valid2 = filter_hexes(df2)
    return valid1.intersection(valid2)


def biomass_by_disturbance_with_mask(gdf_joined, disturbance_column, start_year, end_year, valid_hex_ids):
    """
    Compute the median biomass per hexagon for disturbed pixels with sufficient forest cover,
    restricted to valid hexagons identified beforehand.
    """
    subset = gdf_joined[
        (gdf_joined["year"] >= start_year) &
        (gdf_joined["year"] <= end_year) &
        (gdf_joined[disturbance_column] >= 0.5) &  # high disturbance threshold
        (gdf_joined["forest_fraction"] >= 0.3) &  # match hex-level threshold
        (gdf_joined["index_right"].isin(valid_hex_ids))
    ]
    
    grouped = subset.groupby("index_right")
    output = {}
    for i in range(20):
        biomass_col = f"biomass_m{i}"
        output[biomass_col] = grouped[biomass_col].quantile(0.5)

    return pd.DataFrame(output)

# Step 1: Get valid hex IDs for wind_bark_beetle and harvest
valid_hex_bark = get_consistent_valid_hex_ids(gdf_joined, "wind_bark_beetle", (2011, 2016), (2017, 2023))
valid_hex_harv = get_consistent_valid_hex_ids(gdf_joined, "harvest", (2011, 2016), (2017, 2023))

# Step 2: Use those in your biomass quantile calculations
bm_bark_early = biomass_by_disturbance_with_mask(gdf_joined, "wind_bark_beetle", 2011, 2016, valid_hex_bark)
bm_bark_late  = biomass_by_disturbance_with_mask(gdf_joined, "wind_bark_beetle", 2017, 2023, valid_hex_bark)

bm_harv_early = biomass_by_disturbance_with_mask(gdf_joined, "harvest", 2011, 2016, valid_hex_harv)
bm_harv_late  = biomass_by_disturbance_with_mask(gdf_joined, "harvest", 2017, 2023, valid_hex_harv)

# Add to hex_grid with clear column naming
for i in range(20):
    hex_grid[f"bm_wind_bark_beetle_early_m{i}"] = bm_bark_early[f"biomass_m{i}"]
    hex_grid[f"bm_wind_bark_beetle_late_m{i}"]  = bm_bark_late[f"biomass_m{i}"]
    hex_grid[f"bm_harvest_early_m{i}"]          = bm_harv_early[f"biomass_m{i}"]
    hex_grid[f"bm_harvest_late_m{i}"]           = bm_harv_late[f"biomass_m{i}"]

# 6. Calculate annual total disturbed area per hex for each year and disturbance type
def annual_area_by_disturbance(
    gdf_joined,
    disturbance_column,
    start_year=2003,
    end_year=2023,
    km2_to_Mha=1 / 10_000
):
    annual_records = []

    for year_ in range(start_year, end_year + 1):
        subset = gdf_joined[
            (gdf_joined["year"] == year_) &
            (gdf_joined[disturbance_column] > 0) &  # high disturbance threshold
      	    (gdf_joined["forest_fraction"] >= 0)  # match hex-level threshold
        ].copy()

        if subset.empty:
            continue

        # Area in Mha (million hectares)
        subset["area_Mha"] = (
            subset["forest_fraction"] *
            subset[disturbance_column] *
            subset["pixel_area_km2"] *
            km2_to_Mha
        )

        yearly_area = (
            subset.groupby("index_right")["area_Mha"]
            .sum()
            .reset_index()
            .assign(year=year_, disturbance=disturbance_column)
        )

        annual_records.append(yearly_area)

    if not annual_records:
        return pd.DataFrame(columns=["index_right", "area_Mha", "year", "disturbance"])

    return pd.concat(annual_records, ignore_index=True)

# Run for both disturbance types
area_bark = annual_area_by_disturbance(gdf_joined, "wind_bark_beetle", start_year=1985, end_year=2023)
area_harvest = annual_area_by_disturbance(gdf_joined, "harvest", start_year=1985, end_year=2023)
area_combined = pd.concat([area_bark, area_harvest], ignore_index=True)

# 7. Run prognostic model

# Parameters
years_hist = np.arange(2015, 2024)
years_future = np.arange(2024, 2041)
n_sim = 1000

# Initialize output dictionary
forecast_results = []

# prepare a pivoted area timeseries: index=hex_id, columns=year, values=area_Mha
area_pivot_bark = area_bark.pivot_table(index='index_right', columns='year',
                                        values='area_Mha', fill_value=0, aggfunc='sum')
area_pivot_harv = area_harvest.pivot_table(index='index_right', columns='year',
                                           values='area_Mha', fill_value=0, aggfunc='sum')
                                                                                  

# Temporal Taylor across hexes (each hex contributes one point: its temporal mean & var)
A_bark, b_bark, _, _ = fit_taylor_temporal(area_pivot_bark, years_hist)
A_harv, b_harv, _, _ = fit_taylor_temporal(area_pivot_harv, years_hist)

taylor_by_agent = {
    'wind_bark_beetle': (A_bark, b_bark),
    'harvest':          (A_harv, b_harv),
}

n_models = 20
disturbance_types = ['wind_bark_beetle', 'harvest']
all_sims = []

# Assign one biomass model per simulation per disturbance type
biomass_model_ids = {
    disturbance: np.random.choice(n_models, size=n_sim, replace=True)
    for disturbance in disturbance_types
}

all_sims = []

for disturbance in disturbance_types:
    area_pivot = area_pivot_bark if disturbance == "wind_bark_beetle" else area_pivot_harv

    for hex_id, row in area_pivot.iterrows():
        area_hist_raw = row.reindex(years_hist, fill_value=0).values
        if area_hist_raw.sum() == 0:
            continue

        mean_forecast, sims, best_model, criteria_scores = forecast_disturbance_area_best_fit(
            years_hist, area_hist_raw, years_future,
            n_sim=1000, smoothing_window=5, criterion="AIC",
            fit_start_year=2015, fit_end_year=2023,
            clip_max=None,
            taylor_params=taylor_by_agent[disturbance]
        )

        # Biomass model lookup
        bme_values = [hex_grid.loc[hex_id, f"bm_{disturbance}_early_m{i}"] for i in range(n_models)]
        bml_values = [hex_grid.loc[hex_id, f"bm_{disturbance}_late_m{i}"]  for i in range(n_models)]

        for i, year in enumerate(years_future):
            for j in range(n_sim):
                m = biomass_model_ids[disturbance][j]
                bme = np.nan_to_num(bme_values[m], nan=0.0)
                bml = np.nan_to_num(bml_values[m], nan=0.0)
                all_sims.append({
                    "hex_id": hex_id,
                    "disturbance": disturbance,
                    "year": year,
                    "simulation": j,
                    "area_Mha": sims[i, j],
                    "biomass_early": sims[i, j] * bme,
                    "biomass_late":  sims[i, j] * bml,
                    "biomass_model": m,
                })

df_sims = pd.DataFrame(all_sims)
#df_sims.to_parquet("/misc/glm1/person/besnard/coupling_demography_dist/data/mcmc_prognostic_disturbance.parquet", index=False)

# Global summary: group by year and simulation, then get percentiles
summary_early = (
    df_sims
    .groupby(["year", "simulation", "disturbance"])["biomass_early"]
    .sum()
    .unstack("simulation")  # columns = simulations
    .pipe(lambda df: pd.DataFrame({
        "median": df.median(axis=1),
        "p5":     df.quantile(0.25, axis=1),
        "p95":    df.quantile(0.75, axis=1),
    }))
)

summary_late = (
    df_sims
    .groupby(["year", "simulation", "disturbance"])["biomass_late"]
    .sum()
    .unstack("simulation")
    .pipe(lambda df: pd.DataFrame({
        "median": df.median(axis=1),
        "p5":     df.quantile(0.25, axis=1),
        "p95":    df.quantile(0.75, axis=1),
    }))
)

# 1. Compute median per hex, year, disturbance across simulations
median_biomass = (
    df_sims
    .groupby(["hex_id", "year", "disturbance"])
    .agg({
        "biomass_early": "median",
        "biomass_late": "median"
    })
    .reset_index()
)

# 2. Ensure all future years are present (2024–2050)
full_years = pd.DataFrame({'year': np.arange(2024, 2041)})

# 3. Loop and sum across years for each hexagon & disturbance
records = []
for disturbance in ['wind_bark_beetle', 'harvest']:
    df_d = median_biomass[median_biomass["disturbance"] == disturbance].copy()

    for hex_id, group in df_d.groupby("hex_id"):
        merged = full_years.merge(group, on="year", how="left")
        bme_sum = merged["biomass_early"].fillna(0).sum()
        bml_sum = merged["biomass_late"].fillna(0).sum()
        records.append({
            "hex_id": hex_id,
            "disturbance": disturbance,
            "biomass_early_sum": bme_sum,
            "biomass_late_sum":  bml_sum,
        })

biomass_sums = pd.DataFrame(records)

# Pivot to align early and late per disturbance
biomass_pivot = biomass_sums.pivot(index="hex_id", columns="disturbance")

# Add columns with .get() to avoid KeyError in case of missing disturbance
hex_grid["delta_biomass_bark"] = (
    biomass_pivot.get(("biomass_late_sum", "wind_bark_beetle"), pd.Series(0)) -
    biomass_pivot.get(("biomass_early_sum", "wind_bark_beetle"), pd.Series(0))
)

hex_grid["delta_biomass_harvest"] = (
    biomass_pivot.get(("biomass_late_sum", "harvest"), pd.Series(0)) -
    biomass_pivot.get(("biomass_early_sum", "harvest"), pd.Series(0))
)

#%% Plot data
fig, axs = plt.subplots(2, 2, figsize=(14.5, 10.5), constrained_layout=True)

# --- Top Left: Delta biomass (wind_bark_beetle) ---
europe_gdf.boundary.plot(ax=axs[0, 0], linewidth=0.5, color='lightgrey')
hex_grid[hex_grid["delta_biomass_bark"].notna() & (hex_grid["delta_biomass_bark"] != 0)].plot(
    column="delta_biomass_bark", ax=axs[0, 0],
    cmap="RdBu", edgecolor="none", vmin=-1, vmax=1,
    legend=True, legend_kwds={"label": r"$\Delta$Biomass [TgC]", "shrink": 0.6}
)
axs[0, 0].set_title("Biomass Loss: Natural Disturbance (Late - Early)", fontsize=16)
axs[0, 0].set_axis_off()

# --- Top Right: Delta biomass (harvest) ---
europe_gdf.boundary.plot(ax=axs[0, 1], linewidth=0.5, color='lightgrey')
hex_grid[hex_grid["delta_biomass_harvest"].notna() & (hex_grid["delta_biomass_harvest"] != 0)].plot(
    column="delta_biomass_harvest", ax=axs[0, 1],
    cmap="RdBu", edgecolor="none", vmin=-1, vmax=1,
    legend=True, legend_kwds={"label": r"$\Delta$Biomass [TgC]", "shrink": 0.6}
)
axs[0, 1].set_title("Biomass Loss: Harvest (Late - Early)",  fontsize=16)
axs[0, 1].set_axis_off()

# --- Bottom Left: Boxplot of cumulative biomass loss ---

# Step 1: Compute cumulative loss per simulation
df_boxplot = (
    df_sims
    .groupby(["simulation", "disturbance"])[["biomass_early", "biomass_late"]]
    .sum()
    .reset_index()
    .melt(id_vars=["simulation", "disturbance"],
          value_vars=["biomass_early", "biomass_late"],
          var_name="period",
          value_name="cumulative_loss")
)

# Rename labels
df_boxplot["period"] = df_boxplot["period"].map({
    "biomass_early": "Early",
    "biomass_late": "Late"
})

df_boxplot["disturbance"] = df_boxplot["disturbance"].map({
    "wind_bark_beetle": "Natural Disturbance",
    "harvest": "Harvest"
})

# Convert to PgC
df_boxplot["cumulative_loss"] /= (2040-2024)  # to express as a flux

# Group by disturbance and period, compute median, 5th percentile, 95th percentile
summary = (
    df_boxplot.groupby(["disturbance", "period"])["cumulative_loss"]
    .agg(median="median", q5=lambda x: x.quantile(0.05), q95=lambda x: x.quantile(0.95))
    .reset_index()
)

summary.to_csv("/misc/glm1/person/besnard/coupling_demography_dist/figs/cumulative_biomass_loss_flux_2015_2040.csv", index=False)

ax = axs[1, 0]

# Prepare data grouped by (disturbance, period)
group_order = [("Natural Disturbance", "Early"), ("Natural Disturbance", "Late"),
               ("Harvest", "Early"), ("Harvest", "Late")]

box_data = [
    df_boxplot.query("disturbance == @dist and period == @per")["cumulative_loss"].values
    for dist, per in group_order
]

# Define colors for periods
colors = ["#66c2a5", "#fc8d62", "#66c2a5", "#fc8d62"]

# X tick positions with spacing between groups
positions = [0.9, 1.1, 1.9, 2.1]  # Group 1: 0.9,1.1 | Group 2: 1.9,2.1
width = 0.15

# Boxplot
bp = ax.boxplot(box_data, positions=positions, widths=width, patch_artist=True)

# Set box colors
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Optional: style other elements
for whisker in bp['whiskers']:
    whisker.set_color('gray')
for cap in bp['caps']:
    cap.set_color('gray')
for median in bp['medians']:
    median.set_color('black')
    median.set_linewidth(2)

# X-axis labels
ax.set_xticks([1.0, 2.0])
ax.set_xticklabels(["Natural Disturbance", "Harvest"], fontsize=16)

# Legend manually
legend_patches = [plt.Line2D([0], [0], color=c, lw=6) for c in ["#66c2a5", "#fc8d62"]]
ax.legend(legend_patches, ["Early", "Late"], title="", frameon=False)

# Labels and formatting
ax.set_ylabel("Biomass Loss [TgC year$^{-1}$]", fontsize=16)
ax.set_title("Biomass Loss Forecast by Disturbance", fontsize=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- Prepare historical time series ---
area_types = {
    'Natural Disturbance': 'wind_bark_beetle',
    'Harvest': 'harvest',
}

forecast_results = {}

for label, disturbance_type in area_types.items():
    area_series = (
        area_combined
        .query(f"disturbance == '{disturbance_type}'")
        .groupby("year")["area_Mha"]
        .sum()
        .reindex(years_hist, fill_value=0)
    )
    
    forecast_mean, sims, best_model, criteria_scores = forecast_disturbance_area_best_fit(
                                                                                            years_hist, area_series.values, years_future,
                                                                                            n_sim=1000, smoothing_window=5, criterion="AIC",
                                                                                            fit_start_year=2015, fit_end_year=2023,
                                                                                            clip_max=None,
                                                                                            taylor_params=taylor_by_agent[disturbance_type]   # <<— agent-specific
                                                                                        )
      
    forecast_summary = pd.DataFrame({
        "median": np.median(sims, axis=1),
        "p5":     np.percentile(sims, 25, axis=1),
        "p95":    np.percentile(sims, 75, axis=1),
    }, index=years_future)

    forecast_results[label] = {
        "hist": area_series,
        "mean": forecast_mean,
        "summary": forecast_summary
    }

# --- Plot historical and forecasted area for both disturbance types ---
colors = {
    "Natural Disturbance": "#1f77b4",  # blue
    "Harvest": "#ff7f0e",    # orange
}

axs[1, 1].set_title("Pan-European Disturbed Area", fontsize=16)

for label in area_types.keys():
    hist = forecast_results[label]["hist"]
    
    summary = forecast_results[label]["summary"]
    color = colors[label]
    axs[1, 1].plot(hist.index, moving_average(hist.values, window=5), 'o',alpha=0.6, color=color)
    axs[1, 1].plot(years_future, summary["median"], color=color, label=f"{label}")
    axs[1, 1].fill_between(years_future, summary["p5"], summary["p95"],
                           color=color, alpha=0.2)

axs[1, 1].set_ylabel("Disturbed Area [Mha]", fontsize=16)
axs[1, 1].legend(frameon=False)
axs[1, 1].spines['top'].set_visible(False)
axs[1, 1].spines['right'].set_visible(False)

subplot_labels = ['(a)', '(b)', '(c)', '(d)']
for idx, ax in enumerate(axs.flat):
   ax.text(-0.1, 1.05, subplot_labels[idx], transform=ax.transAxes,
           fontsize=18, fontweight='bold', va='bottom')
plt.savefig('/misc/glm1/person/besnard/coupling_demography_dist/figs/Extended_fig7.png', dpi=300)

all_results = []

for label, data in forecast_results.items():
    # --- hist -> DF ---
    hist_df = data["hist"].reset_index()
    hist_df.columns = ["time", "hist"]
    hist_df["time"] = hist_df["time"].astype(int)

    # --- summary -> DF (ensure correct quantiles + dtype) ---
    # If you already fixed p5/p95 upstream, you can drop this comment.
    summary_df = pd.DataFrame(data["summary"]).reset_index()
    summary_df = summary_df.rename(columns={"index": "time"})
    summary_df["time"] = summary_df["time"].astype(int)

    # --- forecast mean -> Series indexed by years_future ---
    years_future = pd.Index(years_future, name="time").astype(int)  # ensure int + named index
    fm = pd.Series(data["mean"], index=years_future, name="forecast_mean")

    # --- join by time ---
    merged = hist_df.merge(summary_df, on="time", how="outer")
    merged = merged.merge(fm.reset_index(), on="time", how="left")

    # cosmetics / guarantees
    merged = merged.sort_values("time").reset_index(drop=True)
    merged["label"] = label

    all_results.append(merged)

final_df = pd.concat(all_results, ignore_index=True)

# Export to CSV
final_df.to_csv("/misc/glm1/person/besnard/coupling_demography_dist/figs/forecast_results_2015_2040.csv", index=False)

