"""Convergence and fixed parameters of the EM inversion (rework phase 2b)."""
import logging

import numpy as np
import pytest

from etas.inversion import (ETASParameterCalculation, _convergence_difference,
                            calc_diff_to_before)
from tests.conftest import SHORT_INVERSION_CONFIG

THREE_YEARS = dict(SHORT_INVERSION_CONFIG,
                   timewindow_end="1984-01-01 00:00:00")

# ten days after the largest event (M7.8) of the example catalog, in a
# 1x1 degree box around it: almost all events are aftershocks, and the
# background rate goes to zero
SEQUENCE_ONLY = dict(
    SHORT_INVERSION_CONFIG,
    shape_coords=str([[37.9, -121.4], [38.9, -121.4],
                      [38.9, -120.4], [37.9, -120.4]]),
    auxiliary_start="1996-08-01 00:00:00",
    timewindow_start="1996-08-06 06:20:00",
    timewindow_end="1996-08-16 00:00:00",
)


def _invert(config, **kwargs):
    calc = ETASParameterCalculation(dict(config, **kwargs))
    calc.prepare()
    theta = calc.invert()
    return calc, theta


# --- convergence -----------------------------------------------------------

def test_convergence_difference_unchanged_for_normal_background():
    old = np.array([-7.0, None, -2.5, 1.7, -3, 0, 4, -0.3, 1.2, 0.5],
                   dtype=object)
    new = old.copy()
    new[0], new[4] = -7.01, -3.02
    # 50 expected background events: log10 change is the smaller one
    volume_time = 50 / 10 ** -7.0
    assert _convergence_difference(old, new, volume_time) == pytest.approx(
        calc_diff_to_before(old, new))


def test_convergence_difference_vanishing_background():
    old = np.array([-140.0, None, -2.5, 1.7, -3, 0, 4, -0.3, 1.2, 0.5],
                   dtype=object)
    new = old.copy()
    new[0] = -140.17
    assert calc_diff_to_before(old, new) == pytest.approx(0.17)
    assert _convergence_difference(old, new, 1e6) < 1e-100


def test_inversion_stops_when_background_rate_vanishes():
    calc, theta = _invert(SEQUENCE_ONLY)
    assert calc.i < 100
    assert calc.n_hat < 0.01
    for name in ["log10_k0", "a", "log10_c", "omega", "log10_d", "rho"]:
        assert np.isfinite(theta[name])


def test_max_iterations(caplog):
    with caplog.at_level(logging.WARNING, logger="etas.inversion"):
        calc, _ = _invert(SEQUENCE_ONLY, max_iterations=3)
    assert calc.i == 3
    assert "max_iterations" in caplog.text


# --- fixed parameters ------------------------------------------------------

@pytest.mark.slow
def test_fixed_parameters_are_exact():
    _, theta = _invert(THREE_YEARS, fixed_parameters={
        "log10_tau": 4, "omega": -0.16})
    assert theta["log10_tau"] == 4
    assert theta["omega"] == -0.16


def test_fixed_parameter_outside_default_range_is_used(caplog):
    with caplog.at_level(logging.WARNING, logger="etas.inversion"):
        _, theta = _invert(THREE_YEARS, fixed_parameters={"log10_tau": 13})
    assert theta["log10_tau"] == 13
    assert "outside the default range" in caplog.text


@pytest.mark.slow
def test_fixed_background_rate_is_used():
    _, theta = _invert(THREE_YEARS, fixed_parameters={"log10_mu": -7.0})
    assert theta["log10_mu"] == -7.0


@pytest.mark.slow
def test_alpha_fixed_to_beta():
    calc, theta = _invert(THREE_YEARS, fixed_parameters={
        "alpha": "beta", "log10_tau": 4, "omega": -0.16})
    alpha = theta["a"] - theta["rho"] * theta["gamma"]
    assert alpha == pytest.approx(calc.beta, abs=1e-6)
    assert theta["log10_tau"] == 4


def test_inconsistent_alpha_raises():
    calc = ETASParameterCalculation(dict(THREE_YEARS, fixed_parameters={
        "alpha": 2.0, "a": 1.5, "rho": 0.5, "gamma": 1.0}))
    with pytest.raises(ValueError, match="imply alpha"):
        calc.prepare()


@pytest.mark.parametrize("fixed", [{"a": 1.5}, {"log10_k0": -2},
                                   {"alpha": "beta"}])
def test_free_productivity_rejects_derived_parameters(fixed):
    calc = ETASParameterCalculation(dict(
        THREE_YEARS, free_productivity=True, fixed_parameters=fixed))
    with pytest.raises(ValueError, match="free-productivity"):
        calc.prepare()


def test_free_productivity_with_fixed_parameters():
    _, theta = _invert(THREE_YEARS, free_productivity=True,
                       fixed_parameters={"omega": -0.1, "rho": 0.6})
    assert theta["omega"] == -0.1
    assert theta["rho"] == 0.6
    assert np.isfinite(theta["a"]) and np.isfinite(theta["log10_k0"])


# --- no taper (log10_tau = inf) --------------------------------------------

def test_likelihood_continuous_without_taper():
    from etas.inversion import neg_log_likelihood
    calc, _ = _invert(THREE_YEARS, max_iterations=1)
    mc = calc.m_ref - calc.delta_m / 2

    def nll(log10_tau):
        theta = [-2.6, 1.8, -2.5, 0.1, log10_tau, -0.85, 1.3, 0.66]
        return neg_log_likelihood(theta, calc.pij, calc.source_events, mc)

    assert np.isfinite(nll(np.inf))
    # convergence to the limit is slow (~ tau^-omega), so compare with a
    # very large tau
    assert nll(np.inf) == pytest.approx(nll(200.0), rel=1e-6)
    assert nll(20.0) < nll(30.0) < nll(np.inf)


@pytest.mark.slow
def test_fixed_infinite_tau_matches_very_large_tau():
    _, theta_inf = _invert(THREE_YEARS, fixed_parameters={
        "log10_tau": np.inf, "omega": 0.1})
    _, theta_large = _invert(THREE_YEARS, fixed_parameters={
        "log10_tau": 200, "omega": 0.1})
    assert theta_inf["log10_tau"] == np.inf
    for name in ["log10_mu", "log10_k0", "a", "log10_c", "log10_d",
                 "gamma", "rho"]:
        assert theta_inf[name] == pytest.approx(theta_large[name], abs=0.01)


def test_infinite_tau_requires_positive_omega():
    calc = ETASParameterCalculation(dict(THREE_YEARS, fixed_parameters={
        "log10_tau": np.inf, "omega": -0.1}))
    with pytest.raises(ValueError, match="omega > 0"):
        calc.prepare()


def test_integer_initial_values():
    """Integer values, e.g. from a JSON config, must not break np.power."""
    theta_0 = dict(THREE_YEARS["theta_0"], log10_c=-3, log10_tau=4)
    calc, theta = _invert(THREE_YEARS, theta_0=theta_0, max_iterations=2)
    assert np.isfinite(theta["log10_c"])


def test_free_productivity_expected_aftershocks_without_taper():
    """Without taper, the free-productivity helper equals the standard one
    for kappa = k0 * exp(a * (m - mc)), and the limit of very large tau."""
    import pandas as pd
    from etas.inversion import (expected_aftershocks,
                                expected_aftershocks_free_prod)
    log10_k0, a, mc = -2.6, 1.8, 3.55
    m = pd.Series([4.0, 5.0, 6.5])
    t_start = pd.Series([0.0, 3.0, 0.0])
    t_end = pd.Series([100.0, 1000.0, 10.0])
    kappa = 10 ** log10_k0 * np.exp(a * (m - mc))
    rest = [-2.5, 0.1, np.inf, -0.85, 1.3, 0.66]

    free = expected_aftershocks_free_prod([m, kappa, t_start, t_end],
                                          [rest, mc])
    standard = expected_aftershocks([m, t_start, t_end],
                                    [[log10_k0, a] + rest, mc])
    assert np.all(np.isfinite(free))
    np.testing.assert_allclose(free, standard, rtol=1e-12)

    large_tau = list(rest)
    large_tau[2] = 200.0
    np.testing.assert_allclose(
        free, expected_aftershocks_free_prod([m, kappa, t_start, t_end],
                                             [large_tau, mc]), rtol=1e-6)
    # integration to infinity / from the source time
    assert np.all(np.isfinite(expected_aftershocks_free_prod(
        [m, kappa], [rest, mc], no_start=True, no_end=True)))


@pytest.mark.parametrize("value", [0, -1, 2.5, True])
def test_invalid_max_iterations(value):
    with pytest.raises(ValueError, match="max_iterations"):
        ETASParameterCalculation(dict(THREE_YEARS, max_iterations=value))


def test_fixed_background_rate_rejected_with_free_background():
    calc = ETASParameterCalculation(dict(
        THREE_YEARS, free_background=True,
        fixed_parameters={"log10_mu": -7.0}))
    with pytest.raises(ValueError, match="free_background"):
        calc.prepare()
