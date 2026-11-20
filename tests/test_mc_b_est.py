"""Tests for magnitude of completeness and b-value estimation."""
import numpy as np
import pytest

from etas.mc_b_est import (estimate_beta_positive, estimate_beta_tinti,
                           estimate_mc, round_half_up, simulate_magnitudes)

BETA = np.log(10)


def test_round_half_up():
    np.testing.assert_array_equal(
        round_half_up(np.array([0.05, 0.15, 0.25, -0.05]), 1),
        [0.1, 0.2, 0.3, 0.0])
    assert round_half_up(2.5) == 3


def test_round_half_up_reexported_from_inversion():
    from etas.inversion import round_half_up as rhu
    assert rhu is round_half_up


@pytest.mark.parametrize("delta_m", [0, 0.1])
def test_tinti_recovers_beta(delta_m):
    mc = 2.0
    mags = simulate_magnitudes(50000, BETA, mc - delta_m / 2)
    if delta_m:
        mags = round_half_up(mags / delta_m) * delta_m
    beta = estimate_beta_tinti(mags, mc=mc, delta_m=delta_m)
    assert beta == pytest.approx(BETA, rel=0.03)


def test_b_positive_recovers_beta():
    mags = simulate_magnitudes(50000, BETA, 2.0)
    assert estimate_beta_positive(mags) == pytest.approx(BETA, rel=0.05)


def test_estimate_mc_finds_completeness():
    rng = np.random.default_rng(1)
    complete = simulate_magnitudes(4000, BETA, 2.95)
    # incomplete part below 3.0: thinned out
    low = simulate_magnitudes(20000, BETA, 1.95)
    low = low[(low < 2.95) & (rng.uniform(size=len(low)) < 0.1)]
    mags = round_half_up(np.concatenate([complete, low]) / 0.1) * 0.1
    mcs = np.round(np.arange(2.0, 3.5, 0.1), 1)
    result = estimate_mc(mags, mcs, delta_m=0.1, p_pass=0.05,
                         n_samples=1000)
    best_mc = result[3]
    assert 2.8 <= best_mc <= 3.2
