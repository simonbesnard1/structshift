# analysis/age_biomass_bins.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple


class AgeBiomassBinner:
    """
    Compute age × biomass disturbance fractions.
    """

    def __init__(
        self,
        age_bins: Dict[str, Tuple[float, float]],
        biomass_bins: Dict[str, Tuple[float, float]],
    ):
        self.age_bins = age_bins
        self.biomass_bins = biomass_bins

    def fraction_matrix(
        self,
        df: pd.DataFrame,
        disturbance_col: str,
        year_range: Tuple[int, int],
        forest_fraction_min: float = 0.3,
        disturbance_min: float = 0.5,
    ) -> np.ndarray:

        subset = df[
            (df["year"] >= year_range[0])
            & (df["year"] <= year_range[1])
            & (df["forest_fraction"] >= forest_fraction_min)
            & (df[disturbance_col] >= disturbance_min)
        ]

        total = len(subset)
        mat = np.full((len(self.biomass_bins), len(self.age_bins)), np.nan)

        if total == 0:
            return mat

        age = subset["forest_age"].values
        biomass = subset["biomass"].values

        for i, (_, (bmin, bmax)) in enumerate(self.biomass_bins.items()):
            bmask = biomass > bmin if np.isinf(bmax) else (biomass > bmin) & (biomass <= bmax)

            for j, (_, (amin, amax)) in enumerate(self.age_bins.items()):
                amask = age > amin if np.isinf(amax) else (age > amin) & (age <= amax)
                mat[i, j] = (bmask & amask).sum() / total * 100

        return mat
