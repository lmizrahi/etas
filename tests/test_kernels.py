"""Unit tests for the mathematical building blocks of the ETAS model."""
import numpy as np
import pytest
from scipy import integrate, stats
from scipy.special import exp1

from etas.inversion import (branching_integral, branching_ratio,
                            haversine, parameter_array2dict,
                            parameter_dict2array, polygon_surface,
                            upper_gamma_ext)
from etas.simulation import (inv_time_cdf_approx, simulate_aftershock_radius,
                             simulate_aftershock_time,
                             simulate_aftershock_time_rejection)

PARAMS = {
    "log10_mu": -7.5, "log10_iota": None, "log10_k0": -2.49, "a": 1.69,
    "log10_c": -2.95, "omega": -0.03, "log10_tau": 3.99, "log10_d": -0.35,
    "gamma": 1.22, "rho": 0.51,
}


@pytest.mark.parametrize("a", [-0.5, -0.03, 0.0, 0.3, 1.5])
@pytest.mark.parametrize("x", [1e-4, 0.1, 2.0])
def test_upper_gamma_ext_matches_integral(a, x):
    expected, _ = integrate.quad(lambda t: t ** (a - 1) * np.exp(-t),
                                 x, np.inf, limit=200)
    assert upper_gamma_ext(a, x) == pytest.approx(expected, rel=1e-6)
    if a == 0:
        assert upper_gamma_ext(a, x) == pytest.approx(exp1(x))


def test_parameter_dict_array_round_trip():
    arr = parameter_dict2array(PARAMS)
    assert parameter_array2dict(arr) == PARAMS
    with_alpha = dict(PARAMS, alpha=2.0)
    assert parameter_array2dict(parameter_dict2array(with_alpha)) == \
        with_alpha


def test_branching_integral():
    assert branching_integral(-0.5) == pytest.approx(2.0)
    assert branching_integral(0, dm_max=3) == 3
    val, _ = integrate.quad(lambda m: np.exp(0.4 * m), 0, 2)
    assert branching_integral(0.4, dm_max=2) == pytest.approx(val)
    with pytest.raises(AssertionError):
        branching_integral(0.1)


def test_branching_ratio_matches_numerical_integration():
    """eta = int beta e^{-beta m} k(m) int_t int_x g(t,x,m) dx dt dm."""
    p = PARAMS
    beta = np.log(10)
    k0, c, tau, d = (10 ** p["log10_k0"], 10 ** p["log10_c"],
                     10 ** p["log10_tau"], 10 ** p["log10_d"])
    a, omega, gamma, rho = p["a"], p["omega"], p["gamma"], p["rho"]

    def f(t):
        return np.exp(-t / tau) * (t + c) ** (-1 - omega)
    time_int = (integrate.quad(f, 0, tau, limit=500, points=[c, 1])[0]
                + integrate.quad(f, tau, np.inf, limit=500)[0])

    def mag_space(m):
        dm = d * np.exp(gamma * m)
        space = np.pi / rho * dm ** -rho      # int over x of space kernel
        return beta * np.exp(-beta * m) * k0 * np.exp(a * m) * space

    mag_int, _ = integrate.quad(mag_space, 0, 60)  # integrand ~ 0 by 60
    eta = branching_ratio(parameter_dict2array(p), beta)
    assert eta == pytest.approx(time_int * mag_int, rel=1e-4)


def test_haversine_known_distance():
    # Zurich - Geneva, ~224 km great-circle distance
    zrh = np.radians([47.3769, 8.5417])
    gva = np.radians([46.2044, 6.1432])
    dist = haversine(zrh[0], gva[0], zrh[1], gva[1])
    assert dist == pytest.approx(224, abs=2)
    assert haversine(zrh[0], zrh[0], zrh[1], zrh[1]) == 0


def test_polygon_surface_one_degree_box_at_equator():
    from shapely.geometry import Polygon
    # coordinates are (lat, lon) as used throughout the package
    box = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    assert polygon_surface(box) == pytest.approx(111.2 ** 2, rel=0.01)


def _time_cdf(t, c, tau, omega):
    """Exact CDF of the tapered Omori kernel, by numerical integration."""
    def f(s):
        return np.exp(-s / tau) * (s + c) ** (-1 - omega)
    norm = (integrate.quad(f, 0, tau, limit=500, points=[c])[0]
            + integrate.quad(f, tau, np.inf, limit=500)[0])
    return np.array([integrate.quad(f, 0, ti, limit=500)[0] / norm
                     for ti in np.atleast_1d(t)])


@pytest.mark.parametrize("omega", [-0.3, 0.0, 0.2])
def test_rejection_time_sampler_matches_kernel(omega):
    log10_c, log10_tau = -2.0, 2.0
    c, tau = 10 ** log10_c, 10 ** log10_tau
    t = simulate_aftershock_time_rejection(log10_c, omega, log10_tau,
                                           size=4000)
    qs = np.quantile(t, [0.1, 0.25, 0.5, 0.75, 0.9])
    np.testing.assert_allclose(_time_cdf(qs, c, tau, omega),
                               [0.1, 0.25, 0.5, 0.75, 0.9], atol=0.03)


@pytest.mark.parametrize("omega", [-0.3, 0.2])
def test_default_time_sampler_matches_kernel(omega):
    t = simulate_aftershock_time(-2.0, omega, 2.0, size=4000)
    qs = np.quantile(t, [0.25, 0.5, 0.75])
    np.testing.assert_allclose(_time_cdf(qs, 1e-2, 1e2, omega),
                               [0.25, 0.5, 0.75], atol=0.03)


def test_inv_time_cdf_approx_monotone_and_continuous():
    p = np.linspace(0, 0.999, 2001)
    t = inv_time_cdf_approx(p, 1e-2, 1e2, 0.1)
    assert np.all(np.isfinite(t))
    assert np.all(np.diff(t) >= 0)


def test_aftershock_radius_distribution():
    """r^2 follows the space kernel: P(R^2 > s) = (d_m / (s + d_m))^rho."""
    log10_d, gamma, rho, mc = -0.35, 1.22, 0.51, 3.6
    mi = np.full(20000, 4.6)
    r2 = simulate_aftershock_radius(log10_d, gamma, rho, mi, mc) ** 2
    d_m = 10 ** log10_d * np.exp(gamma * 1.0)
    for s in [0.1 * d_m, d_m, 10 * d_m]:
        assert np.mean(r2 > s) == pytest.approx(
            (d_m / (s + d_m)) ** rho, abs=0.015)


def test_gutenberg_richter_sampling_ks():
    from etas.mc_b_est import simulate_magnitudes
    mags = simulate_magnitudes(5000, np.log(10), 3.0)
    assert stats.kstest(mags - 3.0, "expon",
                        args=(0, 1 / np.log(10))).pvalue > 0.01


def test_polygon_coordinates_are_lat_lon():
    """Shapes are given as (latitude, longitude) pairs throughout the
    package. A 1x1 degree box at 60-61 N must have about half the area of
    one at the equator; with swapped axes it would not."""
    from shapely.geometry import Polygon
    box = Polygon([(60, 0), (60, 1), (61, 1), (61, 0)])
    expected = 111.2 * 111.3 * np.cos(np.radians(60.5))
    assert polygon_surface(box) == pytest.approx(expected, rel=0.01)


def test_example_shape_file_is_lat_lon():
    from tests.conftest import INPUT
    coords = np.load(INPUT / "california_shape.npy")
    assert np.all((coords[:, 0] > 30) & (coords[:, 0] < 45))      # lat
    assert np.all((coords[:, 1] > -130) & (coords[:, 1] < -110))  # lon
