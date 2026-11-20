"""End-to-end check: simulate a catalog with known parameters, invert it,
and recover those parameters. This is the main guard that performance work
on the inversion does not change the statistics. Marked slow."""
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from etas.inversion import ETASParameterCalculation
from etas.mc_b_est import round_half_up
from etas.simulation import generate_catalog
from tests.conftest import INPUT

TRUE = {
    "log10_mu": -7.4, "log10_k0": -2.49, "a": 1.69, "log10_c": -2.95,
    "omega": -0.03, "log10_tau": 3.99, "log10_d": -0.35, "gamma": 1.22,
    "rho": 0.51,
}
# generous: one realization, and some parameters (c, tau) are hard to
# constrain; tighter checks on the well-constrained ones
ATOL = {
    "log10_mu": 0.2, "log10_k0": 0.5, "a": 0.4, "log10_c": 0.8,
    "omega": 0.2, "log10_tau": 1.5, "log10_d": 0.4, "gamma": 0.4,
    "rho": 0.2,
}


@pytest.mark.slow
def test_simulate_then_invert_recovers_parameters(tmp_path):
    shape = INPUT / "california_shape.npy"
    synthetic = generate_catalog(
        polygon=Polygon(np.load(shape)),
        timewindow_start=pd.to_datetime("1970-01-01"),
        timewindow_end=pd.to_datetime("2020-01-01"),
        parameters=TRUE, mc=3.6, beta_main=np.log(10), delta_m=0.1)
    synthetic["magnitude"] = round_half_up(synthetic["magnitude"], 1)
    synthetic.index.name = "id"
    fn = tmp_path / "synthetic.csv"
    synthetic[["latitude", "longitude", "time", "magnitude"]].query(
        "time >= '1985-01-01'").to_csv(fn)

    calc = ETASParameterCalculation({
        "fn_catalog": str(fn),
        "auxiliary_start": "1985-01-01 00:00:00",
        "timewindow_start": "1990-01-01 00:00:00",
        "timewindow_end": "2020-01-01 00:00:00",
        "theta_0": {
            "log10_mu": -5.8, "log10_k0": -2.6, "a": 1.8, "log10_c": -2.5,
            "omega": -0.02, "log10_tau": 3.5, "log10_d": -0.85,
            "gamma": 1.3, "rho": 0.66},
        "mc": 3.6, "delta_m": 0.1, "coppersmith_multiplier": 100,
        "shape_coords": str(shape), "id": "recovery",
    })
    calc.prepare()
    est = calc.invert()

    assert calc.beta == pytest.approx(np.log(10), rel=0.1)
    errors = {k: est[k] - TRUE[k] for k in TRUE}
    bad = {k: round(v, 3) for k, v in errors.items() if abs(v) > ATOL[k]}
    assert not bad, f"parameters not recovered: {bad} (all: {errors})"
