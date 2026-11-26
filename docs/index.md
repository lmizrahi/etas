# etas

A Python implementation of the **Epidemic-Type Aftershock Sequence (ETAS)**
model for earthquake catalogs:

- **parameter inversion** with the expectation-maximization algorithm
  (Veen and Schoenberg, 2008),
  including time- and space-varying completeness, ETAS-positive
  (`mc = "positive"`), fixed parameters, free background and free
  productivity (flETAS);
- **simulation** of synthetic catalogs and of catalog continuations
  (forecasts);
- **evaluation** of a fitted model on a test window (event-based
  log-likelihood);
- **completeness magnitude and b-value** estimation;
- **plots** of the model fit.

## Installation

```bash
pip install etas
```

The development version: `pip install git+https://github.com/lmizrahi/etas`.

Optional extras:

| Extra | Adds | Needed for |
|---|---|---|
| `plot` | matplotlib | `etas.plots` |
| `hermes` | hermes-model, seismostats | the hermes entrypoints in `etas.oef`, `ETASSimulation.simulate_to_df` |
| `test` | pytest, flake8 | running the test suite |

e.g. `pip install "etas[plot]"`.
Python ≥ 3.12 is required.

## Where to start

- [Quickstart](quickstart.md): invert, simulate, evaluate, plot.
- [The model](model.md): equations, parameters and units.
- [Configuration](configuration.md): all settings of an inversion.
- API reference: every public function and class.

## Citing

To cite the code, please use its
[DOI](https://zenodo.org/badge/latestdoi/341629005), and cite the relevant
article(s):

- Mizrahi, L., Nandan, S., & Wiemer, S. (2021). The Effect of Declustering
  on the Size Distribution of Mainshocks. *Seismological Research Letters*.
  <https://doi.org/10.1785/0220200231>
- For the inversion with (space-time-)varying completeness magnitude:
  Mizrahi, L., Nandan, S., & Wiemer, S. (2021). Embracing Data
  Incompleteness for Better Earthquake Forecasting. *Journal of Geophysical
  Research: Solid Earth*, Section 3.1. <https://doi.org/10.1029/2021JB022379>

The parameter inversion uses the expectation-maximization algorithm for
ETAS models of Veen, A., & Schoenberg, F. P. (2008). Estimation of
Space-Time Branching Process Models in Seismology Using an EM-Type
Algorithm. *Journal of the American Statistical Association*, 103(482),
614-624. <https://doi.org/10.1198/016214508000000148>

For Probabilistic, Epidemic-Type Aftershock Incompleteness, see
[PETAI](https://github.com/lmizrahi/petai).
