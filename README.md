# Forest Disturbance–Demography Forecasting

This repository contains the **analysis and figure-generation code** used in the manuscript:

> **Coupling forest disturbance regimes and stand demography reveals emerging risks to Europe’s carbon sink**  
> *Simon Besnard et al.*  
> EGUsphere preprint (2025)  
> 👉 https://egusphere.copernicus.org/preprints/2025/egusphere-2025-6288/

The code implements the full analytical workflow used in the paper, including:

- spatial aggregation on hexagonal grids,
- disturbance-specific and genus-specific biomass loss attribution,
- Taylor’s law–based uncertainty propagation,
- trend-based forecasting of disturbed area and biomass loss,
- and reproduction of all main figures.

This repository is provided **for transparency, reproducibility, and reuse**, and reflects the state of the analysis at the time of manuscript submission.

---

## Repository structure

```
.
├── analysis/
│   ├── forecasting.py
│   ├── forecasting_genus.py
│   └── helpers.py
├── workflows/
│   ├── run_forecast.py
│   └── run_forecast_genus.py
├── figures/
│   ├── figure5.py
│   └── figure6.py
├── outputs/
├── README.md
└── LICENSE
```

---

## Data availability

All input data required to run the analyses are publicly available via Zenodo:

👉 https://zenodo.org/records/17977435

This repository does **not** redistribute the data itself.

---

## Citation

If you use this code, please cite:

Besnard, S. et al. (2025).  
*Coupling forest disturbance regimes and stand demography reveals emerging risks to Europe’s carbon sink.*  
EGUsphere preprint.  
https://egusphere.copernicus.org/preprints/2025/egusphere-2025-6288/
