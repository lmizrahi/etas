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


@pytest.mark.parametrize("omega, log10_tau, n", [
    (-0.04, 3.7, 4000),   # fast path (gammainccinv)
    (0.2, np.inf, 4000),  # no taper, closed form
    (0.1, 3.0, 400),      # numerical inverse (slow)
])
def test_window_time_sampler_matches_truncated_kernel(omega, log10_tau, n):
    """Delays drawn within a window follow the time kernel restricted to
    that window (KS test against the exact CDF)."""
    from scipy.stats import kstest

    from etas.inversion import _time_integral
    from etas.simulation import _simulate_aftershock_time_window

    c, tau = 10 ** -3.1, 10 ** log10_tau
    start, end = 400.0, 800.0
    delays = _simulate_aftershock_time_window(
        -3.1, omega, log10_tau, np.full(n, start), np.full(n, end))
    assert ((delays >= start) & (delays <= end)).all()
    total = _time_integral(c, omega, tau, start, end)

    def cdf(t):
        return _time_integral(c, omega, tau, start, np.asarray(t)) / total

    assert kstest(delays, cdf).pvalue > 0.01


def test_induced_events_without_background_events(region):
    """Induced events get valid ids also when the window has no background
    events (previously: NaN ids, and a crash when they had aftershocks)."""
    import datetime as dt

    from etas.simulation import simulate_catalog_continuation

    params = dict(PARAMS, log10_mu=-12.0)  # no background events
    auxiliary = pd.DataFrame({
        "latitude": [36.0, 37.0], "longitude": [-120.0, -121.0],
        "time": pd.to_datetime(["2009-06-01", "2009-12-01"]),
        "magnitude": [5.0, 4.5], "xi_plus_1": [1, 1]})
    np.random.seed(0)
    continuation = simulate_catalog_continuation(
        auxiliary, auxiliary_start=dt.datetime(2009, 1, 1),
        auxiliary_end=dt.datetime(2010, 1, 1), polygon=region,
        simulation_end=dt.datetime(2010, 1, 31), parameters=params,
        mc=3.55, beta_main=2.3,
        induced_lats=pd.Series([38.75]), induced_lons=pd.Series([-122.75]),
        induced_term=pd.Series([1.0]), induced_bsla=0.1, induced_bslo=0.1,
        n_induced=5.0, filter_polygon=False)
    induced = continuation[
        continuation["is_background"].astype(str) == "induced"]
    assert len(induced) > 0
    assert not continuation.index.hasnans
    assert continuation.index.is_unique
    assert not continuation["parent"].isna().any()
