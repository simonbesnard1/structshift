# analysis/age_selectivity.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple


class AgeSelectivityAnalyzer:
    """
    Compute age-class disturbance fractions with ensemble uncertainty.
    """

    def __init__(
        self,
        age_classes: Dict[str, Tuple[float, float]],
        age_prefix: str = "forest_age_gami_2010_m",
        n_models: int = 20,
    ):
        self.age_classes = age_classes
        self.age_cols = [f"{age_prefix}{i}" for i in range(n_models)]

    def compute(
        self,
        df: pd.DataFrame,
        disturbance_col: str,
        forest_fraction_min: float = 0.3,
        disturbance_min: float = 0.5,
    ) -> Dict:

        years = sorted(df["year"].unique())
        out = {}

        for year in years:
            out[year] = {}
            for age_class, (amin, amax) in self.age_classes.items():
                vals = []

                for col in self.age_cols:
                    subset = df[
                        (df["year"] == year)
                        & (df[disturbance_col] >= disturbance_min)
                        & (df["forest_fraction"] >= forest_fraction_min)
                        & (df[col] > 0)
                    ]

                    if len(subset) == 0:
                        vals.append(np.nan)
                        continue

                    mask = subset[col] >= amin if np.isinf(amax) else (
                        (subset[col] >= amin) & (subset[col] <= amax)
                    )
                    vals.append(mask.sum() / len(subset))

                vals = np.array(vals)
                out[year][age_class] = {
                    "median": np.nanmedian(vals),
                    "p5": np.nanpercentile(vals, 5),
                    "p95": np.nanpercentile(vals, 95),
                }

        return out
