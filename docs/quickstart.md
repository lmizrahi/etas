# Quickstart

The examples use the data in `input_data/` and run from the root of the
repository. The same steps exist as scripts in `runnable_code/`, configured
by the JSON files in `config/`.

## 1. Invert ETAS parameters

```python
import logging

from etas import set_up_logger
from etas.inversion import ETASParameterCalculation

set_up_logger(level=logging.INFO)

config = {
    "fn_catalog": "input_data/example_catalog.csv",
    "shape_coords": "input_data/california_shape.npy",
    "auxiliary_start": "1971-01-01 00:00:00",
    "timewindow_start": "1981-01-01 00:00:00",
    "timewindow_end": "2007-01-01 00:00:00",
    "mc": 3.6,
    "delta_m": 0.1,
    "coppersmith_multiplier": 100,
    "theta_0": {
        "log10_mu": -5.8, "log10_k0": -2.6, "a": 1.8, "log10_c": -2.5,
        "omega": -0.02, "log10_tau": 3.5, "log10_d": -0.85,
        "gamma": 1.3, "rho": 0.66,
    },
    "id": "example",
}

calculation = ETASParameterCalculation(config)
calculation.prepare()
parameters = calculation.invert()
print(parameters)

calculation.store_results("output_data/", store_pij=True)
```

The catalog is a csv file with the columns `id`, `latitude`, `longitude`,
`time`, `magnitude`; it must be complete above `mc`. Events between
`auxiliary_start` and `timewindow_start` can trigger events but are not
modeled themselves; events in the primary window
`[timewindow_start, timewindow_end)` are both. All settings are described in
[Configuration](configuration.md).

`store_results` writes `parameters_example.json` (settings and estimated
parameters) and tables of the triggering and background probabilities to
`output_data/`.

## 2. Simulate a catalog continuation (forecast)

```python
import json

from etas.inversion import ETASParameterCalculation
from etas.simulation import ETASSimulation

with open("output_data/parameters_example.json") as f:
    inversion_output = json.load(f)
calculation = ETASParameterCalculation.load_calculation(inversion_output)

simulation = ETASSimulation(calculation)
simulation.prepare()
# 10 catalogs, 30 days after timewindow_end, appended to a csv file
simulation.simulate_to_csv(
    "output_data/forecast_example.csv", forecast_n_days=30, n_simulations=10)
```

Each simulated catalog has its `catalog_id`. Many simulations together form
a probabilistic forecast.

## 3. Evaluate on a test window

```python
import json

from etas.evaluation import ETASLikelihoodCalculation

with open("output_data/parameters_example.json") as f:
    inversion_output = json.load(f)
inversion_output["testwindow_end"] = "2010-01-01 00:00:00"

evaluation = ETASLikelihoodCalculation(inversion_output)
evaluation.prepare(n=100000)
evaluation.evaluate_baseline_poisson_model()
scores = evaluation.evaluate()  # {"nll": ..., "tll": ..., "sll": ...}
```

The scores are averages over the events in the test window
`[timewindow_end, testwindow_end]`: the negative log-likelihood (`nll`) and
its temporal (`tll`) and spatial (`sll`) parts, with `-nll = tll + sll`.
The same scores of a homogeneous Poisson model are in
`evaluation.Poisson_scores`.

## 4. Plot the fit

Requires the `plot` extra and `store_pij=True` in step 1.

```python
import json

import numpy as np

from etas.plots import ETASFitVisualisation

with open("output_data/parameters_example.json") as f:
    out = json.load(f)

visualisation = ETASFitVisualisation({
    "fn_catalog": out["fn_ip"],
    "fn_pij": out["fn_pij"],
    "delta_m": out["delta_m"],
    "mc": out["m_ref"],
    "parameters": out["final_parameters"],
    "label": out["name"],
    "magnitude_list": np.arange(3.8, 6.0, out["delta_m"]),
    "store_path": "output_data/",
})
visualisation.all_plots()  # time, space and productivity fits, as pdf
```

## Other tasks

- Synthetic catalog from scratch: `etas.simulation.generate_catalog`
  (see `runnable_code/simulate_catalog.py`).
- Completeness magnitude: `etas.mc_b_est.estimate_mc`
  (see `runnable_code/estimate_mc.py`).
- Swiss catalog: `etas.download.download_catalog_sed`.
