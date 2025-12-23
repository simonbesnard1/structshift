# Forest Disturbance-Structure Analytics

This repository contains the **analysis and figure-generation code** used in the manuscript:

> **Natural disturbances increasingly affect Europe’s most mature and carbon-rich forests**  
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

## Installation

```bash
pip install git+https://github.com/simonbesnard1/structshift.git
```

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
*Natural disturbances increasingly affect Europe’s most mature and carbon-rich forests.*  
EGUsphere preprint.  
https://egusphere.copernicus.org/preprints/2025/egusphere-2025-6288/

⚠️ Important:
This repository does not redistribute the data itself. Users must download the data separately from Zenodo and update local paths accordingly in the workflow scripts.

## 🚧 Work in progress

This repository reflects an **active effort to consolidate the analytical workflows from the manuscript into a reusable Python package**.  
The core analytical logic and numerical results corresponding to the paper are stable, but the codebase is currently being refactored and modularized (e.g. separation into analysis modules and figure-specific workflows).

Minor cleanup, restructuring, and documentation improvements may occur, but these changes are not expected to alter the scientific conclusions of the manuscript.



