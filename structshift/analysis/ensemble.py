# analysis/ensemble.py
from __future__ import annotations
import pandas as pd

class EnsembleReducer:
    """
    Reduce ensemble model variants to robust summary statistics.
    """

    def __init__(
        self,
        biomass_prefix: str = "biomass_m",
        age_prefix: str = "forest_age_gami_2010_m",
        n_models: int = 20,
        carbon_fraction: float = 0.47,
    ):
        self.biomass_cols = [f"{biomass_prefix}{i}" for i in range(n_models)]
        self.age_cols = [f"{age_prefix}{i}" for i in range(n_models)]
        self.carbon_fraction = carbon_fraction

    def median_biomass(self, df: pd.DataFrame) -> pd.Series:
        biomass = df[self.biomass_cols].where(df[self.biomass_cols] > 0)
        biomass = biomass * self.carbon_fraction
        return biomass.median(axis=1, skipna=True)

    def median_age(self, df: pd.DataFrame) -> pd.Series:
        age = df[self.age_cols].where(df[self.age_cols] >= 0)
        return age.median(axis=1, skipna=True)
