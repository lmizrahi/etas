# ETAS: Epidemic-Type Aftershock Sequence

[![DOI](https://zenodo.org/badge/341629005.svg)](https://zenodo.org/badge/latestdoi/341629005)

A Python implementation of the ETAS model: parameter inversion (including
time- and space-varying completeness, fixed parameters, free background and
free productivity), simulation of synthetic catalogs and forecasts,
evaluation on a test window, completeness and b-value estimation, and plots
of the model fit.

## Installation

```bash
pip install git+https://github.com/lmizrahi/etas
```

Optional extras: `plot` (matplotlib, for `etas.plots`), `hermes` (hermes
entrypoints in `etas.oef`), `docs`, `test`. Python ≥ 3.12.

## Quick example

```python
from etas.inversion import ETASParameterCalculation

calculation = ETASParameterCalculation({
    "fn_catalog": "input_data/example_catalog.csv",
    "shape_coords": "input_data/california_shape.npy",
    "auxiliary_start": "1971-01-01 00:00:00",
    "timewindow_start": "1981-01-01 00:00:00",
    "timewindow_end": "2007-01-01 00:00:00",
    "mc": 3.6,
    "delta_m": 0.1,
    "coppersmith_multiplier": 100,
})
calculation.prepare()
parameters = calculation.invert()
calculation.store_results("output_data/", store_pij=True)
```

## Documentation

The documentation (quickstart, model, all configuration options, API
reference) is in `docs/`. To view it locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

## Contents

- `etas/`: the package
  - `inversion.py`: parameter inversion (`ETASParameterCalculation`)
  - `simulation.py`: catalog simulation and forecasts (`generate_catalog`,
    `ETASSimulation`)
  - `evaluation.py`: log-likelihood scores on a test window
    (`ETASLikelihoodCalculation`)
  - `mc_b_est.py`: completeness magnitude and b-value estimation
  - `plots.py`: plots of the model fit (`ETASFitVisualisation`)
  - `download.py`: download of the Swiss catalog
  - `oef/`: entrypoints for the hermes forecasting platform
- `runnable_code/`: example scripts, configured by the files in `config/`
  - `invert_etas.py`: estimate ETAS parameters (options for varying mc and
    fixed parameters)
  - `simulate_catalog.py`: simulate a synthetic catalog
  - `simulate_catalog_continuation.py`: simulate a continuation of a
    catalog after an inversion; run many times, this is a forecast
    (requires `invert_etas.py` first)
  - `visualise_fit.py`: plots of the model fit (requires `invert_etas.py`
    with `store_pij=True`)
  - `predict_etas.py`: evaluate the model with the event-based
    log-likelihood on a test window
  - `ch_forecast.py`: parameters and 100 simulations for the Swiss catalog
  - `estimate_mc.py`: completeness magnitude of a set of magnitudes
- `input_data/`: example data
  - `example_catalog.csv`, `example_catalog_mc_var.csv` (with varying mc):
    synthetic catalogs for `invert_etas.py`
  - `california_shape.npy`, `ch_rect.npy`: regions (lat, lon)
  - `ch_catalog.csv`: Swiss catalog 1972–2021, for `ch_forecast.py`
  - `magnitudes.npy`: magnitudes for `estimate_mc.py`
- `output_data/`: your output goes here
- `tests/`: test suite (`pytest`; `pytest -m "not slow"` for the fast part)
- `docs/`: documentation sources

## Citing

This code was written for:

Mizrahi, L., Nandan, S., & Wiemer, S. (2021). The Effect of Declustering on
the Size Distribution of Mainshocks. *Seismological Research Letters*.
https://doi.org/10.1785/0220200231

The option for (space-time-)varying completeness magnitude in the parameter
inversion is described in:

Mizrahi, L., Nandan, S., & Wiemer, S. (2021). Embracing Data Incompleteness
for Better Earthquake Forecasting (Section 3.1). *Journal of Geophysical
Research: Solid Earth*. https://doi.org/10.1029/2021JB022379

The parameter inversion uses the expectation-maximization algorithm for
ETAS models of:

Veen, A., & Schoenberg, F. P. (2008). Estimation of Space-Time Branching
Process Models in Seismology Using an EM-Type Algorithm. *Journal of the
American Statistical Association*, 103(482), 614-624.
https://doi.org/10.1198/016214508000000148

To cite the code, please use its
[DOI](https://zenodo.org/badge/latestdoi/341629005), and cite the relevant
article(s). For Probabilistic, Epidemic-Type Aftershock Incompleteness, see
[PETAI](https://github.com/lmizrahi/petai).

In case of questions or comments, contact leila.mizrahi@sed.ethz.ch.
