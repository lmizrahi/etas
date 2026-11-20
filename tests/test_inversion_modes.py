"""Inversion variants beyond the standard ETAS model: free productivity
(flETAS), an additional background term (``bg_term``), and both together.
Run on three years of the example catalog. Marked slow until the
inversion speed-up (rework phase 4)."""
import numpy as np
import pandas as pd
import pytest

from etas.inversion import ETASParameterCalculation
from tests.conftest import INPUT, SHORT_INVERSION_CONFIG

PARAM_NAMES = ["log10_mu", "log10_k0", "a", "log10_c", "omega",
               "log10_tau", "log10_d", "gamma", "rho"]


@pytest.fixture(scope="module")
def catalog():
    cat = pd.read_csv(INPUT / "example_catalog.csv", index_col=0,
                      parse_dates=["time"])
    cat["bg"] = np.random.default_rng(0).uniform(0.5, 1.5, len(cat))
    return cat


def _config(catalog, **kwargs):
    return dict(
        SHORT_INVERSION_CONFIG,
        fn_catalog=None,
        catalog=catalog,
        theta_0=dict(SHORT_INVERSION_CONFIG["theta_0"], log10_iota=-7.0),
        timewindow_end="1984-01-01 00:00:00",
        **kwargs,
    )


@pytest.mark.parametrize(
    "free_productivity,bg_term",
    [pytest.param(True, None, marks=pytest.mark.slow),
     pytest.param(False, "bg", marks=pytest.mark.slow),
     pytest.param(True, "bg", marks=pytest.mark.slow)],
    ids=["free_productivity", "bg_term", "free_productivity+bg_term"])
def test_inversion_variant_runs(catalog, free_productivity, bg_term):
    calc = ETASParameterCalculation(_config(
        catalog, free_productivity=free_productivity, bg_term=bg_term))
    calc.prepare()
    theta = calc.invert()

    for name in PARAM_NAMES:
        assert np.isfinite(theta[name]), name
    if bg_term is None:
        assert theta["log10_iota"] is None
    else:
        assert np.isfinite(theta["log10_iota"])
        te = calc.target_events
        np.testing.assert_allclose(
            te["P_triggered"] + te["P_background"] + te["P_induced"], 1,
            atol=1e-8)
    if free_productivity:
        assert (calc.source_events["source_kappa"] >= 0).all()


def test_prepare_twice_does_not_redo_work(catalog, caplog):
    calc = ETASParameterCalculation(_config(catalog))
    calc.prepare()
    distances = calc.distances
    calc.prepare()
    assert calc.distances is distances
    assert "already done" in caplog.text
