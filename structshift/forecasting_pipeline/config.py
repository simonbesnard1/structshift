from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class ForecastConfig:
    # -----------------------
    # Simulation + models
    # -----------------------
    n_sim: int = 1000
    n_biomass_models: int = 20
    seed: int = 42

    # -----------------------
    # Time axes
    # -----------------------
    years_all_hist: np.ndarray = field(default_factory=lambda: np.arange(1985, 2025))
    years_hist_fit: np.ndarray = field(default_factory=lambda: np.arange(2008, 2025))
    years_future: np.ndarray = field(default_factory=lambda: np.arange(2025, 2041))

    # -----------------------
    # Thresholds / masks
    # -----------------------
    min_forest_frac: float = 0.3
    min_dist_frac: float = 0.5
    min_pixels_per_period: int = 50

    # -----------------------
    # Spatial
    # -----------------------
    hex_diameter_m: int = 100_000
    pixel_res_deg: float = 0.0008888888888888889  # ~100 m
    crs_latlon: str = "EPSG:4326"
    crs_europe: str = "EPSG:3035"

    # -----------------------
    # Disturbance types
    # -----------------------
    disturbance_types: tuple[str, ...] = ("wind_bark_beetle", "harvest")

    # -----------------------
    # Paths
    # -----------------------
    parquet_path: Path = Path(
        "/home/besnard/projects/coupling_demography_dist/data/data_extraction/"
        "disturbance_data_combined_v2025-12.parquet"
    )
    polys_gpkg: Path = Path(
        "/home/besnard/projects/coupling_demography_dist/data/ancillary/"
        "bounding_box_filter_years_4326.gpkg"
    )
    eco_path: Path = Path(
        "/home/besnard/projects/coupling_demography_dist/data/ancillary/"
        "biogeo_EU_2016.gpkg"
    )
    world_gpk: Path = Path(
        "/misc/glm1/person/besnard/coupling_demography_dist/data/"
        "ne_10m_admin_0_countries.zip"
    )
    outdir: Path = Path("/home/besnard/projects/coupling_demography_dist/outputs/")

    # -----------------------
    # Output filenames
    # -----------------------
    out_boxplot_parquet: str = "cumulative_biomass_loss_flux_v11_full.parquet"
    out_summary_csv: str = "cumulative_biomass_loss_flux_v11_summary.csv"
    out_hex_gpkg: str = "hex_delta_biomass_v11.gpkg"
    out_forecast_csv: str = "forecast_results_1985_2040_v11.csv"

    def biomass_cols(self) -> list[str]:
        return [f"biomass_m{i}" for i in range(self.n_biomass_models)]
