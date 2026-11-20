"""Tests for catalog simulation.

The simulation currently reseeds the global RNG internally, so results are
checked statistically rather than event by event.
"""
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

from etas.inversion import branching_ratio, parameter_dict2array
from etas.mc_b_est import estimate_beta_tinti
from etas.simulation import generate_catalog
from tests.conftest import INPUT

PARAMS = {
    "log10_mu": -6.5, "log10_k0": -2.49, "a": 1.69, "log10_c": -2.95,
    "omega": -0.03, "log10_tau": 3.99, "log10_d": -0.35, "gamma": 1.22,
    "rho": 0.51,
}
MC, BETA = 3.6, np.log(10)


@pytest.fixture(scope="module")
def region():
    return Polygon(np.load(INPUT / "california_shape.npy"))


@pytest.fixture(scope="module")
def synthetic(region):
    # module-scoped fixtures run before the per-test seeding in conftest
    np.random.seed(42)
    return generate_catalog(
        polygon=region,
        timewindow_start=pd.to_datetime("2000-01-01"),
        timewindow_end=pd.to_datetime("2010-01-01"),
        parameters=PARAMS, mc=MC, beta_main=BETA, delta_m=0)


def test_catalog_columns_and_window(synthetic):
    for col in ["latitude", "longitude", "time", "magnitude"]:
        assert col in synthetic.columns
    assert synthetic["time"].min() >= pd.Timestamp("2000-01-01")
    assert synthetic["time"].max() <= pd.Timestamp("2010-01-01")
    assert synthetic["magnitude"].min() >= MC


def test_magnitudes_follow_gutenberg_richter(synthetic):
    beta = estimate_beta_tinti(synthetic["magnitude"], mc=MC)
    se = beta / np.sqrt(len(synthetic))
    assert abs(beta - BETA) < 4 * se


def test_background_events_inside_region(synthetic, region):
    bg = synthetic.query("generation == 0")
    inside = [region.contains(Point(la, lo))
              for la, lo in zip(bg["latitude"], bg["longitude"])]
    assert np.all(inside)


def test_event_count_consistent_with_branching_ratio(synthetic):
    eta = branching_ratio(parameter_dict2array(PARAMS), BETA)
    assert 0 < eta < 1
    n_bg = (synthetic["generation"] == 0).sum()
    # background events within the window, Poisson
    assert len(synthetic) >= n_bg
    # boundary effects make the total smaller than n_bg / (1 - eta)
    assert len(synthetic) <= 1.3 * n_bg / (1 - eta)
