"""The fast maximization-step objective equals the public likelihood
functions, and its gradient equals finite differences."""
import numpy as np
import pandas as pd
import pytest

from etas.inversion import (_MStepObjective, _time_integral,
                            expected_aftershocks, neg_log_likelihood,
                            neg_log_likelihood_free_prod)

MC_MIN = 2.95


@pytest.fixture(scope="module")
def mstep_data():
    rng = np.random.default_rng(0)
    n_sources, n_pairs = 200, 3000
    source_mag = MC_MIN + rng.exponential(1 / 2.3, n_sources)
    # half of the sources inside the window (start 0), half before it
    start = np.where(np.arange(n_sources) % 2 == 0, 0.0,
                     rng.uniform(0, 500, n_sources))
    source_events = pd.DataFrame({
        "l_hat": rng.exponential(1.0, n_sources),
        "source_magnitude": source_mag,
        "pos_source_to_start_time_distance": start,
        "source_to_end_time_distance": start + rng.uniform(10, 3000,
                                                           n_sources),
    }, index=pd.Index(np.arange(n_sources), name="source_id"))
    src = rng.integers(0, n_sources, n_pairs)
    pij = pd.DataFrame({
        "Pij": rng.uniform(0, 1, n_pairs),
        "zeta_plus_1": rng.choice([1.0, 1.5, 3.2], n_pairs),
        "time_distance": rng.exponential(30, n_pairs),
        "spatial_distance_squared": rng.exponential(200, n_pairs),
        "source_magnitude": source_mag[src],
    }, index=pd.MultiIndex.from_arrays(
        [src, np.arange(n_pairs)], names=("source_id", "target_id")))
    return pij, source_events


THETAS = [
    [-2.6, 1.8, -2.5, -0.02, 3.5, -0.85, 1.3, 0.66],
    [-2.2, 1.2, -3.0, 0.3, 2.0, 0.2, 0.5, 1.5],
    [-3.0, 2.2, -1.5, -0.5, 4.5, -1.5, 1.8, 0.3],
    [-2.6, 1.8, -2.5, 0.2, np.inf, -0.85, 1.3, 0.66],  # no taper
]


def _finite_differences(f, theta, h=1e-6):
    grad = np.zeros(len(theta))
    for i in range(len(theta)):
        if np.isinf(theta[i]):
            continue
        e = np.zeros(len(theta))
        e[i] = h
        grad[i] = (f(theta + e) - f(theta - e)) / (2 * h)
    return grad


@pytest.mark.parametrize("theta", THETAS)
def test_value_and_gradient(mstep_data, theta):
    pij, source_events = mstep_data
    theta = np.array(theta)
    objective = _MStepObjective(pij, source_events, MC_MIN)
    value, grad = objective(theta)
    assert value == pytest.approx(
        neg_log_likelihood(theta, pij, source_events, MC_MIN), rel=1e-12)
    fd = _finite_differences(lambda t: objective(t)[0], theta)
    np.testing.assert_allclose(grad, fd, rtol=1e-6, atol=1e-4)
    if np.isinf(theta[4]):
        assert grad[4] == 0


@pytest.mark.parametrize("theta", THETAS)
def test_value_and_gradient_free_productivity(mstep_data, theta):
    pij, source_events = mstep_data
    theta = np.array(theta)[2:]
    objective = _MStepObjective(pij, source_events, MC_MIN,
                                free_productivity=True)
    value, grad = objective(theta)
    assert value == pytest.approx(neg_log_likelihood_free_prod(
        theta, None, pij, source_events, None, None, None, None, MC_MIN),
        rel=1e-12)
    fd = _finite_differences(lambda t: objective(t)[0], theta)
    np.testing.assert_allclose(grad, fd, rtol=1e-6, atol=1e-4)


def test_objective_does_not_modify_inputs(mstep_data):
    pij, source_events = mstep_data
    pij_before, se_before = pij.copy(), source_events.copy()
    _MStepObjective(pij, source_events, MC_MIN)(np.array(THETAS[0]))
    pd.testing.assert_frame_equal(pij, pij_before)
    pd.testing.assert_frame_equal(source_events, se_before)


@pytest.mark.parametrize("omega, log10_tau", [(-0.3, 3.0), (0.4, 2.0),
                                              (0.2, np.inf)])
def test_time_integral_matches_expected_aftershocks(omega, log10_tau):
    # with k0 = 1, a = 0 and the area factor divided out, the expected
    # number of aftershocks is the time integral
    c, tau = 10 ** -2.5, 10 ** log10_tau
    start, end = np.array([0.0, 5.0]), np.array([100.0, 2000.0])
    theta = [0.0, 0.0, -2.5, omega, log10_tau, 0.0, 0.0, 1.0]
    area = np.pi
    expected = expected_aftershocks(
        [np.array([MC_MIN, MC_MIN]), start, end], [theta, MC_MIN]) / area
    np.testing.assert_allclose(
        _time_integral(c, omega, tau, start, end), expected, rtol=1e-12)
