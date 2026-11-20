#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Inversion of ETAS parameters with the expectation-maximization algorithm.

The EM algorithm for ETAS models was introduced by Veen and Schoenberg
(2008), Estimation of Space-Time Branching Process Models in Seismology
Using an EM-Type Algorithm, Journal of the American Statistical
Association, https://doi.org/10.1198/016214508000000148. This
implementation is described by Mizrahi et al. (2021), The Effect of
Declustering on the Size Distribution of Mainshocks, Seismological Research
Letters, https://doi.org/10.1785/0220200231, and, for time- and
space-varying completeness, by Mizrahi et al. (2021), Embracing Data
Incompleteness for Better Earthquake Forecasting, JGR Solid Earth,
Section 3.1, https://doi.org/10.1029/2021JB022379.

The model
---------
A catalog consists of events $e_1, \dots, e_N$; event $e_i$ has the
location $(x_i, y_i)$ (latitude, longitude), the occurrence time $t_i$
(days) and the magnitude $m_i$. The conditional intensity, that is, the
expected number of events per km² and day at location $(x, y)$ and time
$t$, given the events before $t$, is

$$
\lambda(x, y, t) = \mu + \sum_{i: t_i < t} g(m_i, r_i(x, y), t - t_i),
$$

where $\mu$ is the background rate (per km² and day), the sum goes over all
events $e_i$ before $t$, and $r_i(x, y)$ is the distance (km) between
$(x_i, y_i)$ and $(x, y)$. The triggering kernel $g$ is the rate of direct
aftershocks of an event of magnitude $m$, at a distance $r$ (km) and a time
delay $\Delta t$ (days),

$$
g(m, r, \Delta t) = k_0 \, e^{a (m - m_{\mathrm{ref}})}
    \cdot \frac{1}{(r^2 + d \, e^{\gamma (m - m_{\mathrm{ref}})})^{1 + \rho}}
    \cdot \frac{e^{-\Delta t / \tau}}{(\Delta t + c)^{1 + \omega}},
$$

where $m_{\mathrm{ref}}$ is the reference magnitude, the lower edge of the
lowest magnitude bin (``m_ref - delta_m / 2`` in the code); $k_0$ and $a$
set the number of aftershocks and its increase with magnitude; $d$ (km²),
$\gamma$ and $\rho$ are the size of the aftershock zone of an event of
magnitude $m_{\mathrm{ref}}$, its increase with magnitude, and the decay of
the rate with distance; and $c$ (days), $\omega$ and $\tau$ (days) are the
offset, exponent and taper time of the Omori-Utsu law. Magnitudes at or
above $m_{\mathrm{ref}}$ follow the Gutenberg-Richter law with parameter
$\beta = b \ln 10$, where $b$ is the b-value. With a constant completeness
magnitude $m_c$, $m_{\mathrm{ref}} = m_c$; with varying completeness
(``mc = "var"`` or ``"positive"``), $m_{\mathrm{ref}}$ is at most the
smallest completeness. The documentation page "The model" explains each
term with example values. For 3D catalogs, distances are in units of
``space_unit_in_meters`` instead of km.

Parameters are handled either as a dict or as an array ``theta`` in the
order log10_mu, log10_iota, log10_k0, a, log10_c, omega, log10_tau,
log10_d, gamma, rho (see `parameter_dict2array`); log10_iota is the
rate of the optional additional background term (``bg_term``).

Coordinates: regions and catalogs use (latitude, longitude) in degrees,
in this order.

Main entry point: `ETASParameterCalculation`.
"""

import datetime as dt
import json
import logging
import os
import pprint
import uuid

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely.ops as ops
from scipy.optimize import NonlinearConstraint, linprog, minimize
from scipy.spatial import ConvexHull
from scipy.special import exp1
from scipy.special import gamma as gamma_func
from scipy.special import gammaincc, gammaln
from shapely.geometry import Polygon

from etas.mc_b_est import (estimate_beta_positive, estimate_beta_tinti,
                           round_half_up)

logger = logging.getLogger(__name__)

# ranges for parameters
LOG10_MU_RANGE = (-10, 0)
LOG10_IOTA_RANGE = (-10, 0)
LOG10_K0_RANGE = (-20, 10)
A_RANGE = (0.01, 20)
LOG10_C_RANGE = (-8, 0)
OMEGA_RANGE = (-0.99, 1)
LOG10_TAU_RANGE = (0.01, 12.26)
LOG10_D_RANGE = (-4, 3)
GAMMA_RANGE = (-1, 5.0)
RHO_RANGE = (0.01, 5.0)
RANGES = (
    LOG10_MU_RANGE,
    LOG10_IOTA_RANGE,
    LOG10_K0_RANGE,
    A_RANGE,
    LOG10_C_RANGE,
    OMEGA_RANGE,
    LOG10_TAU_RANGE,
    LOG10_D_RANGE,
    GAMMA_RANGE,
    RHO_RANGE,
)

# order of the parameters in theta (the array form of the parameters)
_PARAMETER_NAMES = (
    "log10_mu",
    "log10_iota",
    "log10_k0",
    "a",
    "log10_c",
    "omega",
    "log10_tau",
    "log10_d",
    "gamma",
    "rho",
)

# the EM algorithm stops when the parameters change by less than this
CONVERGENCE_TOLERANCE = 0.001
DEFAULT_MAX_ITERATIONS = 1000


def _convergence_difference(theta_old, theta_new, volume_time):
    """Change between two EM iterations, used as stopping criterion.

    Sum of absolute changes of all parameters, as ``calc_diff_to_before``.
    For the background rates (log10_mu, log10_iota), the change is the
    smaller of the change in log10 and the change in the expected *number*
    of background events, ``volume_time * |mu_new - mu_old|``. When the
    background rate goes to zero, log10_mu keeps decreasing by a similar
    amount in every iteration, so the log10 change alone never falls below
    the tolerance; the change in the number of events does. The two
    measures only differ when fewer than 1 / ln(10) ~ 0.43 background events
    are expected, so the criterion is unchanged in all other cases.
    """
    diff = calc_diff_to_before(theta_old[2:], theta_new[2:])
    for k in (0, 1):
        a, b = theta_old[k], theta_new[k]
        if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)):
            continue
        diff += min(abs(a - b),
                    volume_time * abs(np.power(10.0, a) - np.power(10.0, b)))
    return diff


def coppersmith(mag, fault_type):
    """
    Returns the result of coppersmith in km.

    Parameters
    ----------
    mag : float
        Magnitude
    fault_type : int
        1: strike slip fault, 2: reverse fault 3: normal fault 4: oblique fault

    Returns
    -------
    dict
        containing results for SRL, SSRL, RW, RA and AD
    """

    def log_reg(a, b, mag=mag):
        return np.power(10, (a * mag + b))

    if fault_type == 1:
        # surface rupture length
        SRL = log_reg(0.74, -3.55)
        # subsurface rupture length
        SSRL = log_reg(0.62, -2.57)
        # rupture width
        RW = log_reg(0.27, -0.76)
        # rupture area
        RA = log_reg(0.9, -3.42)
        # average slip
        AD = log_reg(0.9, -6.32)

    elif fault_type == 2:
        # surface rupture length
        SRL = log_reg(0.63, -2.86)
        # subsurface rupture length
        SSRL = log_reg(0.58, -2.42)
        # rupture width
        RW = log_reg(0.41, -1.61)
        # rupture area
        RA = log_reg(0.98, -3.99)
        # average slip
        AD = log_reg(0.08, -0.74)

    elif fault_type == 3:
        # surface rupture length
        SRL = log_reg(0.5, -2.01)
        # subsurface rupture length
        SSRL = log_reg(0.5, -1.88)
        # rupture width
        RW = log_reg(0.35, -1.14)
        # rupture area
        RA = log_reg(0.82, -2.87)
        # average slip
        AD = log_reg(0.63, -4.45)

    elif fault_type == 4:
        # surface rupture length
        SRL = log_reg(0.69, -3.22)
        # subsurface rupture length
        SSRL = log_reg(0.59, -2.44)
        # rupture width
        RW = log_reg(0.32, -1.01)
        # rupture area
        RA = log_reg(0.91, -3.49)
        # average slip
        AD = log_reg(0.69, -4.80)

    return {"SRL": SRL, "SSRL": SSRL, "RW": RW, "RA": RA, "AD": AD}


def rectangle_surface(lat1, lat2, lon1, lon2):
    """Area in km² of the lat/lon rectangle with the given corners.

    Parameters
    ----------
    lat1, lat2 : float
        Latitudes of two opposite corners, in degrees.
    lon1, lon2 : float
        Longitudes of two opposite corners, in degrees.

    Returns
    -------
    float
        Area in km², computed in an Albers equal-area projection.
    """
    vertices = [[lat1, lon1], [lat2, lon1], [lat2, lon2], [lat1, lon2]]
    polygon = Polygon(vertices)

    proj_wgs84 = pyproj.CRS('EPSG:4326')
    proj_aea = pyproj.CRS(
        proj="aea", lat_1=polygon.bounds[0], lat_2=polygon.bounds[2])

    transformer = pyproj.Transformer.from_crs(
        proj_wgs84, proj_aea)

    geom_area = ops.transform(transformer.transform, polygon)

    return geom_area.area / 1e6


def polygon_surface(polygon):
    """Area in km² of a polygon on the Earth's surface.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        Polygon with vertices given as (latitude, longitude) in degrees, the
        convention used throughout the package.

    Returns
    -------
    float
        Area in km², computed in an Albers equal-area projection with
        standard parallels at the polygon's latitude bounds.
    """
    proj_wgs84 = pyproj.CRS('EPSG:4326')
    proj_aea = pyproj.CRS(
        proj="aea", lat_1=polygon.bounds[0], lat_2=polygon.bounds[2])

    transformer = pyproj.Transformer.from_crs(
        proj_wgs84, proj_aea)

    geom_area = ops.transform(transformer.transform, polygon)

    return geom_area.area / 1e6


def in_hull(points, x):
    """Whether point ``x`` lies inside the convex hull of ``points``.

    Solved as a linear feasibility problem (is ``x`` a convex combination of
    the points?), so it also works in 3D.

    Parameters
    ----------
    points : np.ndarray, shape (n, k)
        Points spanning the hull.
    x : np.ndarray, shape (k,)
        Point to test.

    Returns
    -------
    bool
    """
    n_points = len(points)
    c = np.zeros(n_points)
    A = np.r_[points.T, np.ones((1, n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success


def hav(theta):
    """Haversine of an angle, ``sin²(theta / 2)``.

    Parameters
    ----------
    theta : float or np.ndarray
        Angle in radians.
    """
    return np.square(np.sin(theta / 2))


def haversine(lat_rad_1,
              lat_rad_2,
              lon_rad_1,
              lon_rad_2,
              earth_radius=6.3781e3):
    """
    Calculates the distance on a sphere.
    """
    d = (
        2
        * earth_radius
        * np.arcsin(
            np.sqrt(
                hav(lat_rad_1 - lat_rad_2)
                + np.cos(lat_rad_1) * np.cos(lat_rad_2)
                * hav(lon_rad_1 - lon_rad_2)
            )
        )
    )
    return d


def branching_integral(alpha_minus_beta, dm_max=None):
    """Integral of ``exp((alpha - beta) * m)`` over magnitudes above the
    reference magnitude.

    Parameters
    ----------
    alpha_minus_beta : float
        ``alpha - beta``, where ``alpha = a - rho * gamma``.
    dm_max : float, optional
        Upper integration limit, as magnitude above the reference magnitude.
        If None, the integral goes to infinity, which requires
        ``alpha < beta``.

    Returns
    -------
    float
    """
    if dm_max is None:
        assert alpha_minus_beta < 0, (
            "for unlimited magnitudes, " "alpha minus beta has to be negative"
        )
        return -1 / alpha_minus_beta
    else:
        if alpha_minus_beta == 0:
            return dm_max
        else:
            return (np.exp(alpha_minus_beta * dm_max) - 1) / (alpha_minus_beta)


def branching_ratio(theta, beta, dm_max=None):
    """Branching ratio: expected number of direct aftershocks per event.

    Averaged over a Gutenberg-Richter distribution of source magnitudes with
    parameter ``beta``, integrated over all space and time.

    Parameters
    ----------
    theta : array-like
        Parameters in the order of ``parameter_dict2array``
        (log10_mu, log10_iota, log10_k0, a, log10_c, omega, log10_tau,
        log10_d, gamma, rho).
    beta : float
        Gutenberg-Richter parameter in natural-log units
        (``beta = b * ln(10)``).
    dm_max : float, optional
        Maximum magnitude above the reference magnitude. If None, magnitudes
        are unbounded, which requires ``alpha = a - rho * gamma < beta``.

    Returns
    -------
    float
        The branching ratio; values >= 1 mean a supercritical process.
    """
    (
        log10_mu,
        log10_iota,
        log10_k0,
        a,
        log10_c,
        omega,
        log10_tau,
        log10_d,
        gamma,
        rho,
    ) = theta
    k0 = np.power(10, log10_k0)
    c = np.power(10, log10_c)
    d = np.power(10, log10_d)
    tau = np.power(10, log10_tau)

    alpha = a - rho * gamma
    mag_int = branching_integral(alpha - beta, dm_max)
    time_int = (
        np.power(tau, -omega) * np.exp(c / tau)
        * upper_gamma_ext(-omega, c / tau)
        if tau != np.inf
        else np.power(c, -omega) / omega
    )
    k_factor = k0 * np.pi / rho * np.power(d, -rho)

    eta = beta * time_int * mag_int * k_factor
    return eta


def to_days(timediff):
    """Convert a time difference to (fractional) days.

    Parameters
    ----------
    timediff : datetime.timedelta, pd.Timedelta or pd.Series of timedeltas

    Returns
    -------
    float or pd.Series
    """
    return timediff / dt.timedelta(days=1)


def upper_gamma_ext(a, x):
    r"""Upper incomplete gamma function, extended to non-positive ``a``.

    $$
    \Gamma(a, x) = \int_x^\infty t^{a - 1} e^{-t} \, dt
    $$

    where $a$ is the shape parameter, $x$ the lower limit of the integral,
    and $t$ the integration variable. In the ETAS model, it appears in the
    integral of the tapered time kernel, with $a = -\omega$ and
    $x = c / \tau$.

    For ``a > 0`` this is ``gammaincc(a, x) * gamma(a)``, for ``a = 0`` the
    exponential integral ``E1(x)``, and for ``a < 0`` it is obtained by the
    recurrence ``Gamma(a, x) = (Gamma(a + 1, x) - x^a e^-x) / a``.

    Parameters
    ----------
    a : float
        Shape parameter (any real number).
    x : float or np.ndarray
        Lower integration limit, ``x > 0``.

    Returns
    -------
    float or np.ndarray
    """
    if a > 0:
        return gammaincc(a, x) * gamma_func(a)
    elif a == 0:
        return exp1(x)
    else:
        return (upper_gamma_ext(a + 1, x) - np.power(x, a) * np.exp(-x)) / a


def parameter_array2dict(theta):
    """Convert an array of ETAS parameters to a dict.

    Parameters
    ----------
    theta : array-like
        Parameters in the order log10_mu, log10_iota, log10_k0, a, log10_c,
        omega, log10_tau, log10_d, gamma, rho. If it has 11 entries, the first
        one is ``alpha``.

    Returns
    -------
    dict
        Parameter name -> value.
    """
    if len(theta) > 10:
        return dict(
            zip(
                [
                    "alpha",
                    "log10_mu",
                    "log10_iota",
                    "log10_k0",
                    "a",
                    "log10_c",
                    "omega",
                    "log10_tau",
                    "log10_d",
                    "gamma",
                    "rho",
                ],
                theta,
            )
        )

    return dict(
        zip(
            [
                "log10_mu",
                "log10_iota",
                "log10_k0",
                "a",
                "log10_c",
                "omega",
                "log10_tau",
                "log10_d",
                "gamma",
                "rho",
            ],
            theta,
        )
    )


def parameter_dict2array(parameters):
    """Convert a dict of ETAS parameters to an array.

    Inverse of `parameter_array2dict`. Missing parameters become
    ``None``; if ``"alpha"`` is a key, it is the first entry.

    Parameters
    ----------
    parameters : dict
        Parameter name -> value.

    Returns
    -------
    np.ndarray
        Values in the order log10_mu, log10_iota, log10_k0, a, log10_c, omega,
        log10_tau, log10_d, gamma, rho (preceded by alpha if given).
    """
    order = [
        "log10_mu",
        "log10_iota",
        "log10_k0",
        "a",
        "log10_c",
        "omega",
        "log10_tau",
        "log10_d",
        "gamma",
        "rho",
    ]

    if "alpha" in parameters:
        order.insert(0, "alpha")

    return np.array([parameters.get(key, None) for key in order])


def create_initial_values(ranges=RANGES):
    """Random initial parameter values, uniformly within ``ranges``.

    Parameters
    ----------
    ranges : sequence of (low, high)
        One range per parameter, in the order of ``parameter_dict2array``.

    Returns
    -------
    list of float
    """
    return [np.random.uniform(*r) for r in ranges]


def triggering_kernel(metrics, params):
    """
    Given time distance in days and squared space distance in square km and
    magnitude of target event, calculate the (not normalized) likelihood,
    that source event triggered target event.
    """
    time_distance, spatial_distance_squared, m, source_kappa = metrics
    theta, mc = params

    (
        log10_mu,
        log10_iota,
        log10_k0,
        a,
        log10_c,
        omega,
        log10_tau,
        log10_d,
        gamma,
        rho,
    ) = theta

    if source_kappa is None:
        k0 = np.power(10, log10_k0)
    c = np.power(10, log10_c)
    tau = np.power(10, log10_tau)
    d = np.power(10, log10_d)
    aftershock_number = (
        source_kappa if source_kappa is not None else k0 * np.exp(a * (m - mc))
    )
    time_decay = np.exp(-time_distance / tau) / np.power(
        (time_distance + c), (1 + omega)
    )
    space_decay = 1 / np.power(
        (spatial_distance_squared + d * np.exp(gamma * (m - mc))), (1 + rho)
    )

    res = aftershock_number * time_decay * space_decay
    return res


def responsibility_factor(theta, beta, delta_mc):
    """Correction for triggering by undetected events (varying completeness).

    An event observed above its local completeness ``mc_i`` stands for all
    events between ``m_ref`` and ``mc_i`` that were not detected. This factor,
    ``1 + xi = exp(-(a - beta - gamma * rho) * delta_mc)``, is the ratio of the
    triggering by all events above ``m_ref`` to the triggering by those
    observed (Mizrahi et al. 2021, JGR, Sect. 3.1).

    Parameters
    ----------
    theta : array-like
        Parameters in the order of ``parameter_dict2array``.
    beta : float
        Gutenberg-Richter parameter (natural-log units).
    delta_mc : float or array-like
        Completeness of the source event above ``m_ref``,
        ``mc_current - m_ref``.

    Returns
    -------
    float or array-like
        ``1 + xi``; equal to 1 where ``delta_mc = 0``.
    """
    (
        log10_mu,
        log10_iota,
        log10_k0,
        a,
        log10_c,
        omega,
        log10_tau,
        log10_d,
        gamma,
        rho,
    ) = theta

    xi_plus_1 = 1 / (np.exp((a - beta - gamma * rho) * delta_mc))

    return xi_plus_1


def observation_factor(beta, delta_mc):
    """Number of events above ``m_ref`` per event observed above
    ``mc_current``.

    ``1 + zeta = exp(beta * delta_mc)``, from the Gutenberg-Richter law
    (Mizrahi et al. 2021, JGR, Sect. 3.1).

    Parameters
    ----------
    beta : float
        Gutenberg-Richter parameter (natural-log units).
    delta_mc : float or array-like
        Completeness above the reference magnitude,
        ``mc_current - m_ref``.

    Returns
    -------
    float or array-like
    """
    zeta_plus_1 = np.exp(beta * delta_mc)
    return zeta_plus_1


def expected_aftershocks(event, params, no_start=False, no_end=False):
    """Expected number of aftershocks of events within a time window.

    Integral of the triggering kernel over the whole plane and over the time
    window, for events with productivity ``k0 * exp(a * (m - mc))``.

    Parameters
    ----------
    event : list
        ``[magnitude, time_to_start, time_to_end]``: source magnitudes, and the
        time in days from the source to the start and to the end of the window
        (``time_to_start`` is 0 for sources inside the window). With
        ``no_start=True`` the ``time_to_start`` entry is omitted (integration
        from the source time), with ``no_end=True`` the ``time_to_end`` entry
        (integration to infinity). Entries may be arrays or pandas Series.
    params : list
        ``[theta, mc]``, where ``theta`` holds the eight parameters log10_k0,
        a, log10_c, omega, log10_tau, log10_d, gamma, rho, and ``mc`` is the
        reference magnitude of the kernel (``m_ref - delta_m / 2`` in the
        inversion).
    no_start, no_end : bool
        See ``event``.

    Returns
    -------
    float or array-like
        Expected number of aftershocks per source event.
    """
    theta, mc = params

    log10_k0, a, log10_c, omega, log10_tau, log10_d, gamma, rho = theta
    k0 = np.power(10, log10_k0)
    c = np.power(10, log10_c)
    tau = np.power(10, log10_tau)
    d = np.power(10, log10_d)

    if no_start:
        if no_end:
            event_magnitude = event
        else:
            event_magnitude, event_time_to_end = event
    else:
        if no_end:
            event_magnitude, event_time_to_start = event
        else:
            event_magnitude, event_time_to_start, event_time_to_end = event

    number_factor = k0 * np.exp(a * (event_magnitude - mc))
    area_factor = (
        np.pi * np.power(d * np.exp(gamma
                         * (event_magnitude - mc)), -1 * rho) / rho
    )

    time_factor = np.exp(c / tau) * np.power(tau,
                                             - omega)  # * gamma_func(-omega)

    if no_start:
        if tau == np.inf:
            time_factor = np.power(c, -omega) / omega
        else:
            time_fraction = upper_gamma_ext(-omega, c / tau)
    else:
        if tau == np.inf:
            time_factor = np.power(event_time_to_start + c, -omega) / omega
        else:
            time_fraction = upper_gamma_ext(-omega,
                                            (event_time_to_start + c) / tau)
    if not no_end:
        if tau == np.inf:
            time_factor = time_factor - \
                np.power(event_time_to_end + c, -omega) / omega
        else:
            time_fraction = time_fraction - upper_gamma_ext(
                -omega, (event_time_to_end + c) / tau
            )

    if tau != np.inf:
        time_factor = time_factor * time_fraction

    return number_factor * area_factor * time_factor


def _log_time_normalization(c, omega, tau):
    """Log of the normalization of the time kernel.

    ``-log(int_0^inf exp(-t / tau) (t + c)^-(1 + omega) dt)`` minus
    ``omega * log(tau)`` cancelled against ``exp(c / tau)``, i.e.
    ``omega * log(tau) - log(Gamma(-omega, c / tau))``. Without a taper
    (``tau = inf``) this is the limit ``log(omega) + omega * log(c)``, which
    requires ``omega > 0``.
    """
    if np.isinf(tau):
        return np.log(omega) + omega * np.log(c)
    return omega * np.log(tau) - np.log(upper_gamma_ext(-omega, c / tau))


def ll_aftershock_term(l_hat, g):
    """Poisson log-likelihood of the expected number of aftershocks.

    ``l_hat * log(g) - g - log(l_hat!)``, with ``log(0)`` replaced by -300.

    Parameters
    ----------
    l_hat : array-like
        Estimated number of aftershocks per source (from the expectation step).
    g : array-like
        Expected number of aftershocks per source under the current
        parameters (see `expected_aftershocks`).

    Returns
    -------
    array-like
    """
    mask = g != 0
    term = -1 * gammaln(l_hat + 1) - g
    term = term + l_hat * np.where(mask, np.log(g), -300)
    return term


def neg_log_likelihood(theta, Pij, source_events, mc_min):
    """Objective of the maximization step: negative expected complete-data
    log-likelihood of the triggering parameters.

    Sum of the Poisson log-likelihood of the number of aftershocks per source
    (`ll_aftershock_term`) and of the space-time distribution of the
    aftershocks, weighted by the triggering probabilities ``Pij``. The
    background rate is estimated separately, in closed form.

    Parameters
    ----------
    theta : array-like
        The eight parameters log10_k0, a, log10_c, omega, log10_tau,
        log10_d, gamma, rho.
    Pij : pd.DataFrame
        Output of the expectation step, indexed by
        (``source_id``, ``target_id``), with columns ``Pij``, ``zeta_plus_1``,
        ``time_distance``, ``spatial_distance_squared``, ``source_magnitude``.
    source_events : pd.DataFrame
        Indexed by ``source_id``, with columns ``l_hat``,
        ``source_magnitude``, ``pos_source_to_start_time_distance``,
        ``source_to_end_time_distance``.
    mc_min : float
        Reference magnitude of the kernel, ``m_ref - delta_m / 2``.

    Returns
    -------
    float
        The inputs are not modified.
    """
    assert Pij.index.names == ("source_id", "target_id"), logger.error(
        'Pij must have multiindex with names "source_id", "target_id"'
    )
    assert source_events.index.name == "source_id", logger.error(
        'source_events must have index with name "source_id"'
    )

    log10_k0, a, log10_c, omega, log10_tau, log10_d, gamma, rho = theta

    c = np.power(10, log10_c)
    tau = np.power(10, log10_tau)
    d = np.power(10, log10_d)

    G = expected_aftershocks(
        [
            source_events["source_magnitude"],
            source_events["pos_source_to_start_time_distance"],
            source_events["source_to_end_time_distance"],
        ],
        [theta, mc_min],
    )

    aftershock_term = ll_aftershock_term(source_events["l_hat"], G).sum()

    # space time distribution term
    likelihood_term = (
        (
            _log_time_normalization(c, omega, tau)
            + np.log(rho)
            + rho * np.log(d * np.exp(gamma
                           * (Pij["source_magnitude"] - mc_min)))
        )
        - (
            (1 + rho)
            * np.log(
                Pij["spatial_distance_squared"]
                + (d * np.exp(gamma * (Pij["source_magnitude"] - mc_min)))
            )
        )
        - (1 + omega) * np.log(Pij["time_distance"] + c)
        - (Pij["time_distance"] + c) / tau
        - np.log(np.pi)
    )
    distribution_term = Pij["Pij"].mul(
        Pij["zeta_plus_1"]).mul(likelihood_term).sum()

    total = aftershock_term + distribution_term

    return -1 * total


def expected_aftershocks_free_prod(
        event, params, no_start=False, no_end=False):
    """Expected number of aftershocks in free-productivity mode.

    Like `expected_aftershocks`, but the productivity of each source is
    its own value ``kappa`` instead of ``k0 * exp(a * (m - mc))``.

    Parameters
    ----------
    event : list
        ``[magnitude, kappa, time_to_start, time_to_end]``; ``time_to_start``
        is omitted with ``no_start=True``, ``time_to_end`` with
        ``no_end=True``.
    params : list
        ``[theta, mc]``, where ``theta`` holds the six parameters log10_c,
        omega, log10_tau, log10_d, gamma, rho.
    no_start, no_end : bool
        See `expected_aftershocks`.

    Returns
    -------
    float or array-like
    """
    theta, mc = params

    log10_c, omega, log10_tau, log10_d, gamma, rho = theta
    c = np.power(10, log10_c)
    tau = np.power(10, log10_tau)
    d = np.power(10, log10_d)

    if no_start:
        if no_end:
            event_magnitude, event_kappa = event
        else:
            event_magnitude, event_kappa, event_time_to_end = event
    else:
        if no_end:
            event_magnitude, event_kappa, event_time_to_start = event
        else:
            (event_magnitude,
             event_kappa,
             event_time_to_start,
             event_time_to_end) = event

    number_factor = event_kappa
    area_factor = (
        np.pi * np.power(d * np.exp(gamma
                         * (event_magnitude - mc)), -1 * rho) / rho
    )

    if tau == np.inf:
        # no taper: integral of (t + c)^-(1 + omega), requires omega > 0
        start = c if no_start else event_time_to_start + c
        time_factor = np.power(start, -omega) / omega
        if not no_end:
            time_factor = time_factor - \
                np.power(event_time_to_end + c, -omega) / omega
        return number_factor * area_factor * time_factor

    time_factor = np.exp(c / tau) * np.power(tau,
                                             - omega)  # * gamma_func(-omega)

    if no_start:
        time_fraction = upper_gamma_ext(-omega, c / tau)
    else:
        time_fraction = upper_gamma_ext(-omega,
                                        (event_time_to_start + c) / tau)
    if not no_end:
        time_fraction = time_fraction - upper_gamma_ext(
            -omega, (event_time_to_end + c) / tau
        )

    time_factor = time_factor * time_fraction

    return number_factor * area_factor * time_factor


def neg_log_likelihood_free_prod(
    theta,
    n_hat,
    Pij,
    source_events,
    timewindow_length,
    timewindow_start,
    area,
    beta,
    mc_min,
):
    """Objective of the maximization step in free-productivity mode.

    Only the space-time distribution of the aftershocks enters (weighted by
    the triggering probabilities); the productivities are updated separately
    (see `ETASParameterCalculation.update_source_kappa`).

    Parameters
    ----------
    theta : array-like
        The six parameters log10_c, omega, log10_tau, log10_d, gamma, rho.
    n_hat, source_events, timewindow_length, timewindow_start, area, beta
        Not used; kept for backwards compatibility of the signature.
    Pij : pd.DataFrame
        As in `neg_log_likelihood`.
    mc_min : float
        Reference magnitude of the kernel, ``m_ref - delta_m / 2``.

    Returns
    -------
    float
    """
    assert Pij.index.names == (
        "source_id",
        "target_id",
    ), "Pij must have multiindex with names 'source_id', 'target_id'"
    assert (
        source_events.index.name == "source_id"
    ), "source_events must have index with name 'source_id'"

    log10_c, omega, log10_tau, log10_d, gamma, rho = theta
    c = np.power(10, log10_c)
    tau = np.power(10, log10_tau)
    d = np.power(10, log10_d)

    # space time distribution term
    likelihood_term = (
        (
            _log_time_normalization(c, omega, tau)
            + np.log(rho)
            + rho * np.log(d * np.exp(gamma
                           * (Pij["source_magnitude"] - mc_min)))
        )
        - (
            (1 + rho)
            * np.log(
                Pij["spatial_distance_squared"]
                + (d * np.exp(gamma * (Pij["source_magnitude"] - mc_min)))
            )
        )
        - (1 + omega) * np.log(Pij["time_distance"] + c)
        - (Pij["time_distance"] + c) / tau
        - np.log(np.pi)
    )
    distribution_term = Pij["Pij"].mul(
        Pij["zeta_plus_1"]).mul(likelihood_term).sum()

    total = distribution_term

    return -1 * total


def prod_neg_log_lik(a, args):
    """Objective for estimating ``a`` from individual productivities.

    Absolute value of the derivative with respect to ``a`` of the (weighted)
    Poisson log-likelihood of ``kappa ~ k0 * exp(a * m_diff)``, with ``k0``
    profiled out. Its minimum (zero) is the maximum-likelihood ``a``.

    Parameters
    ----------
    a : float or array of one float
    args : list
        ``[kappa, m_diff, weights]``, see `calc_a_k0_from_kappa`.

    Returns
    -------
    float
    """
    sk, md, weights = args
    k_0 = np.sum(weights * sk) / (weights * np.exp(a * md)).sum()
    ll = (weights * md * (sk - k_0 * np.exp(a * md))).sum()
    return np.abs(ll)


def calc_a_k0_from_kappa(kappa, m_diff, weights=1):
    """Fit the productivity law to individual productivities.

    Finds ``a`` and ``k0`` such that ``kappa ≈ k0 * exp(a * m_diff)``, by
    maximum likelihood (see `prod_neg_log_lik`); ``a`` is searched in
    [0, 5].

    Parameters
    ----------
    kappa : array-like
        Productivity (expected number of direct aftershocks) per event.
    m_diff : array-like
        Magnitude above the reference magnitude, per event.
    weights : array-like or float, default 1
        Weight per event.

    Returns
    -------
    tuple of float
        ``(a, log10_k0)``.
    """
    res = minimize(
        prod_neg_log_lik,
        x0=1.5,
        args=[kappa, m_diff, weights],
        bounds=[(0, 5)]
    )
    a = res.x[0]
    log10_k0 = np.log10(np.sum(kappa * weights)
                        / (np.exp(a * m_diff) * weights).sum())
    return a, log10_k0


def read_shape_coords(shape_coords):
    """Read region coordinates from the different accepted formats.

    Parameters
    ----------
    shape_coords : None, str or array-like
        - None: no region (the whole globe).
        - path to a ``.npy`` file with an array of shape (n, 2).
        - a string representation of the coordinates, e.g. as written to
          ``parameters_<id>.json`` by ``store_results``. It is evaluated with
          ``eval``, so it must come from a trusted source.
        - an array-like of (latitude, longitude) pairs (or (x, y, z) points
          for 3D inversions).

    Returns
    -------
    np.ndarray or None
    """
    if shape_coords is None:
        return None
    if isinstance(shape_coords, str):
        if shape_coords[-4:] == ".npy":
            # input is the path to a -npy file containing the coordinates
            coordinates = np.load(shape_coords, allow_pickle=True)
        else:
            from numpy import array  # noqa

            coordinates = np.array(eval(shape_coords))
    else:
        coordinates = np.array(shape_coords)
    return coordinates


def calc_diff_to_before(a, b):
    """Sum of absolute differences between two parameter vectors.

    Entries that are None or infinite in either vector are skipped.

    Parameters
    ----------
    a, b : array-like
        Parameter vectors of equal length.

    Returns
    -------
    float
    """
    assert len(a) == len(b), "a and b must have the same length."

    return np.sum(
        np.abs(
            [
                a[i] - b[i]
                for i in range(len(a))
                if (
                    a[i] is not None
                    and b[i] is not None
                    and not np.isinf(a[i])
                    and not np.isinf(b[i])
                )
            ]
        )
    )


class ETASParameterCalculation:
    def __init__(self, metadata: dict):
        """
        Estimate ETAS parameters from an earthquake catalog.

        Usage:

            calc = ETASParameterCalculation(metadata)
            calc.prepare()
            parameters = calc.invert()
            calc.store_results("output/", store_pij=True)

        Parameters
        ----------
        metadata : dict
            A dict with stored metadata.

            Necessary attributes are:

            - fn_catalog: Path to the catalog. Catalog is expected to be a csv
                    file with the following columns:
                        id, latitude, longitude, time, magnitude
                    id needs to contain a unique identifier for each event
                    time contains datetime of event occurrence
                    see example_catalog.csv for an example
                    Either 'fn_catalog' or 'catalog' need to be defined.
            - catalog: Dataframe with a catalog, same requirements as for the
                    csv above apply.
                    Either 'fn_catalog' or 'catalog' need to be defined.
            - auxiliary_start (str or datetime): Start date of the auxiliary
                    catalog. Events of the auxiliary catalog act as sources,
                    not as targets.
            - timewindow_start: Start date of the primary catalog , end date
                    of auxiliary catalog (str or datetime). Events of the
                    primary catalog act as sources and as targets.
            - timewindow_end: End date of the primary catalog (str or datetime)
            - mc: Cutoff magnitude. Catalog needs to be complete above mc.
                    if mc == 'var', m_ref is required, and the catalog needs to
                    contain a column named 'mc_current'.
                    if mc == 'positive', m_ref is required, and the catalog
                    will be automatically filtered to contain only events with
                    magnitudes greater than that of the previous event,
                    following the idea of Van der Elst 2021
                    (J Geophysical Research: Solid Earth, Vol 126, Issue 2).
            - m_ref: Reference magnitude when mc is variable. Not required
                    unless mc == 'var' or mc == 'positive'. Must be less
                    than or equal to the smallest mc_current in the
                    filtered catalog.
            - delta_m: Size of magnitude bins
            - coppersmith_multiplier: Events further apart from each other than
                    coppersmith subsurface rupture length * this multiplier
                    are considered to be uncorrelated (to reduce size of
                    distance matrix).
            - shape_coords: Coordinates of the boundary of the region to
                    consider, or path to a .npy file containing the coordinates
                    (list of lists, i.e. ``[[lat1, lon1], [lat2, lon2],
                    [lat3, lon3]]``). If None, the whole globe is
                    considered.
            - inner_shape_coords: optional, a region inside shape_coords.
                    Only events inside it are targets; all events inside
                    shape_coords can be sources. Default is None (same as
                    shape_coords).
            - beta: optional. If provided, beta will be fixed to this value.
                    If set to 'positive', beta will be estimated using the
                    b-positive method. Default is None.
            - three_dim: optional, if True, the inversion will be done in 3D.
                    In this case, columns "x", "y", "z" need to be present in
                    the catalog, and shape_coords requires a list of
                    coordinates whose convex hull defines the considered
                    region.
                    Default is False.
            - space_unit_in_meters: optional, unit of space in meters. Default
                    is 1000. This is only relevant if three_dim is True.
                    Otherwise, latitude and longitude are used and distances
                    given in km.
            - theta_0: optional, initial guess for parameters. Does not affect
                    final parameters, but with a good initial guess
                    the algorithm converges faster.
            - free_background: optional, allow free_background during
                    inversion (flETAS)
                default: False
            - free_productivity: optional, allow free_productivity during
                    inversion (flETAS)
                default: False
            - bw_sq: optional, squared bandwidth of Gaussian kernel used for
                    free_background/free_productivity mode
                default: 1
            - fixed_parameters: optional, dict of parameters that are not
                    estimated but fixed to the given value, e.g.
                    ``{"omega": -0.1, "log10_tau": 4}``. Any parameter of
                    theta can be fixed, including log10_mu. Additionally,
                    ``"alpha"`` (= a - rho * gamma) can be fixed to a value
                    or to ``"beta"``. Values outside the default range
                    (``RANGES``) are used as given, with a warning. In
                    free_productivity mode, log10_k0, a and alpha cannot be
                    fixed. Default is None.
            - bg_term: optional, name of a catalog column with a covariate
                    (e.g. injection rate) that drives an additional
                    background term with rate log10_iota (needs an initial
                    value for log10_iota in theta_0). Default is None.
            - max_iterations: optional, maximum number of EM iterations
                    (None for no limit). Default is 1000.
            - earth_radius: optional, in km. Default is 6378.1.
            - testwindow_end: optional, end of the test window, used by
                    ``etas.evaluation``.
            - name: optional, give the model a name
            - id: optional, give the model an ID
        """

        self.logger = logging.getLogger(__name__)
        self.name = metadata.get("name", "NoName ETAS Model")
        self.id = metadata.get("id", uuid.uuid4())
        self.logger.info("INITIALIZING...")
        self.logger.info(
            "  model is named {}, has ID {}".format(self.name, self.id))
        self.shape_coords = read_shape_coords(
            metadata.get("shape_coords", None))
        self.inner_shape_coords = read_shape_coords(
            metadata.get("inner_shape_coords", None)
        )
        if self.inner_shape_coords is not None:
            self.logger.debug("  using inner shape coords!")
        self.fn_catalog = metadata.get("fn_catalog", None)
        self.catalog = metadata.get("catalog", None)

        self.delta_m = metadata["delta_m"]
        self.mc = metadata["mc"]
        self.m_ref = (
            metadata["m_ref"]
            if (self.mc == "var" or self.mc == "positive")
            else self.mc
        )
        self.coppersmith_multiplier = metadata["coppersmith_multiplier"]
        self.earth_radius = metadata.get("earth_radius", 6.3781e3)
        self.bw_sq = metadata.get("bw_sq", 1)
        self.beta = metadata.get("beta", None)
        self.b_positive = None

        self.three_dim = metadata.get("three_dim", False)
        self.space_unit_in_meters = metadata.get("space_unit_in_meters", 1000)

        self.auxiliary_start = pd.to_datetime(metadata["auxiliary_start"])
        self.timewindow_start = pd.to_datetime(metadata["timewindow_start"])
        self.timewindow_end = pd.to_datetime(metadata["timewindow_end"])
        try:
            self.testwindow_end = pd.to_datetime(metadata["testwindow_end"])
        except (KeyError, TypeError, ValueError):
            # testwindow_end is optional (only needed for evaluation)
            self.testwindow_end = None

        self.timewindow_length = to_days(
            self.timewindow_end - self.timewindow_start)
        self.calculation_date = dt.datetime.now()

        self.free_background = metadata.get("free_background", False)
        self.free_productivity = metadata.get("free_productivity", False)
        self.bg_term = metadata.get("bg_term", None)
        self.max_iterations = metadata.get(
            "max_iterations", DEFAULT_MAX_ITERATIONS)
        if self.max_iterations is not None and (
                isinstance(self.max_iterations, bool)
                or not float(self.max_iterations).is_integer()
                or self.max_iterations < 1):
            raise ValueError(
                "max_iterations must be a positive integer or None, got "
                "{!r}".format(self.max_iterations))

        self.logger.info(
            "  Time Window: \n      {} (aux start)\n      {} "
            "(start)\n      {} (end).".format(
                self.auxiliary_start,
                self.timewindow_start,
                self.timewindow_end
            )
        )

        self.logger.info(
            "  free_productivity: {}, free_background: {}".format(
                self.free_productivity, self.free_background
            )
        )

        self.preparation_done = False
        self.inversion_done = False

        if not isinstance(self.catalog, pd.DataFrame):
            self.catalog = pd.read_csv(
                self.fn_catalog,
                index_col=0,
                parse_dates=["time"],
                dtype={"url": str, "alert": str},
            )
            self.catalog["time"] = pd.to_datetime(
                self.catalog["time"], format="ISO8601"
            )

        self.distances = None
        self.source_events = None
        self.target_events = None

        self.area = None
        self.__theta_0 = None
        self.theta_0 = metadata.get("theta_0")
        self.__fixed_parameters = None
        self.fixed_parameters = metadata.get("fixed_parameters", None)
        self.__theta = None
        self.alpha = None
        self.constraints = None
        self.pij = None
        self.n_hat = None
        self.i_hat = None
        self.i = metadata.get("n_iterations")

    @classmethod
    def load_calculation(cls, metadata: dict):
        """Recreate a calculation from stored results.

        Parameters
        ----------
        metadata : dict
            Content of the ``parameters_<id>.json`` file written by
            `store_results`. The catalog is reloaded (from
            ``fn_catalog``) and filtered; source events, target events,
            ``Pij`` and distances are reloaded if their file names
            (``fn_src``, ``fn_ip``, ``fn_pij``, ``fn_dist``) are in
            ``metadata``.

        Returns
        -------
        ETASParameterCalculation
            With ``preparation_done`` and ``inversion_done`` set, and the
            parameters in ``theta``.
        """
        obj = cls.__new__(cls)

        obj.logger = logging.getLogger(__name__)
        obj.name = metadata["name"]
        obj.id = metadata["id"]

        obj.logger.info("Loading Calculation...")
        obj.logger.info(
            "  model is named {}, has ID {}".format(obj.name, obj.id))

        obj.shape_coords = read_shape_coords(metadata["shape_coords"])

        obj.fn_catalog = metadata["fn_catalog"]

        obj.delta_m = metadata["delta_m"]
        obj.mc = metadata["mc"]
        obj.m_ref = metadata["m_ref"]
        obj.coppersmith_multiplier = metadata["coppersmith_multiplier"]
        obj.earth_radius = metadata["earth_radius"]
        obj.bw_sq = metadata["bw_sq"]
        obj.b_positive = metadata["b_positive"]
        obj.three_dim = metadata.get("three_dim", False)
        obj.space_unit_in_meters = metadata.get("space_unit_in_meters", 1000)

        obj.auxiliary_start = pd.to_datetime(metadata["auxiliary_start"])
        obj.timewindow_start = pd.to_datetime(metadata["timewindow_start"])
        obj.timewindow_end = pd.to_datetime(metadata["timewindow_end"])
        obj.timewindow_length = metadata["timewindow_length"]
        obj.calculation_date = metadata["calculation_date"]

        obj.free_background = metadata["free_background"]
        obj.free_productivity = metadata["free_productivity"]
        obj.bg_term = metadata["bg_term"]

        obj.logger.info(
            "  Time Window: \n      {} (aux start)\n      {} "
            "(start)\n      {} (end).".format(
                obj.auxiliary_start, obj.timewindow_start, obj.timewindow_end
            )
        )

        obj.logger.info(
            "  free_productivity: {}, free_background: {}".format(
                obj.free_productivity, obj.free_background
            )
        )

        obj.preparation_done = True
        obj.inversion_done = True

        if "oef_setting" in metadata:
            obj.oef_setting = metadata["oef_setting"]
        else:
            obj.oef_setting = None

        if obj.fn_catalog is not None:
            obj.catalog = pd.read_csv(
                obj.fn_catalog,
                index_col=0,
                parse_dates=["time"],
                dtype={"url": str, "alert": str},
            )
            obj.catalog["time"] = pd.to_datetime(
                obj.catalog["time"], format="ISO8601")
        else:
            obj.catalog = None
            if not obj.oef_setting:
                obj.logger.warning("Catalog could not be loaded. \
                                    Only ok to proceed in specific use cases.")

        obj.area = metadata["area"]
        obj.beta = metadata["beta"]
        obj.theta_0 = metadata["initial_values"]
        obj.theta = metadata["final_parameters"]

        obj.n_hat = metadata["n_hat"]
        obj.i_hat = metadata["i_hat"]
        obj.i = metadata["n_iterations"]

        if obj.catalog is not None:
            obj.catalog = obj.filter_catalog(obj.catalog)

        if "fn_src" in metadata:
            obj.source_events = pd.read_csv(
                metadata["fn_src"],
                index_col=0
            )
        else:
            if not obj.oef_setting:
                obj.logger.warning("Sources could not be loaded. \
                                    Only ok to proceed in specific use cases.")

        if "fn_ip" in metadata:
            obj.target_events = pd.read_csv(
                metadata["fn_ip"],
                index_col=0,
                parse_dates=["time"]
            )
        else:
            if not obj.oef_setting:
                obj.logger.warning("Targets could not be loaded. \
                                    Only ok to proceed in specific use cases.")

        if "fn_pij" in metadata:
            obj.pij = pd.read_csv(
                metadata["fn_pij"],
                index_col=["source_id", "target_id"],
                parse_dates=["target_time"],
            )
        else:
            if not obj.oef_setting:
                obj.logger.warning("Pij could not be loaded.")

        if "fn_dist" in metadata:
            obj.distances = pd.read_csv(
                metadata["fn_dist"],
                index_col=["source_id", "target_id"],
                parse_dates=["target_time"],
            )
        else:
            if not obj.oef_setting:
                obj.logger.warning("Distances could not be loaded.")

        return obj

    def prepare(self):
        """Prepare the data for the inversion.

        Filters the catalog (time window, region, completeness), sets
        initial parameter values (random within ``RANGES`` if ``theta_0`` was
        not given), calculates the distances between possibly related events,
        prepares source and target events, estimates ``beta`` (unless fixed),
        and sets up fixed parameters. Does nothing if already prepared.
        """
        if self.preparation_done:
            self.logger.warning("Preparation already done, aborting...")
            return

        self.logger.info("PREPARING {}".format(self.name))
        self.logger.info("  filtering catalog...")
        self.catalog = self.filter_catalog(self.catalog)

        if self.__theta_0 is not None:
            self.logger.info("  using input initial values for theta")
        else:
            self.logger.info("  randomly chosing initial values for theta")
            self.__theta_0 = create_initial_values()

        self.logger.info("  calculating distances...")
        self.distances = self.calculate_distances()

        self.logger.info("  preparing source and target events..")
        self.target_events = self.prepare_target_events()
        self.source_events = self.prepare_source_events()

        if isinstance(self.beta, float):
            self.b_positive = False
            self.logger.info(
                "  beta of primary catalog is fixed to {}".format(self.beta)
            )
        elif self.beta == "positive" or self.mc == "positive":
            self.beta = estimate_beta_positive(
                self.target_events["magnitude"], delta_m=self.delta_m
            )
            self.b_positive = True
            self.logger.info(
                "  beta of primary catalog is {}, "
                "estimated with b-positive".format(
                    self.beta
                )
            )
        else:
            self.beta = estimate_beta_tinti(
                self.target_events["magnitude"]
                - self.target_events["mc_current"],
                mc=0,
                delta_m=self.delta_m,
            )
            self.b_positive = False
            self.logger.info(
                "  beta of primary catalog is {}".format(self.beta))

        if self.free_productivity:
            self.source_events["source_kappa"] = np.exp(
                self.theta_0["a"]
                * (
                    self.source_events["source_magnitude"]
                    - self.m_ref
                    - self.delta_m / 2
                )
            )
        if self.free_background:
            self.target_events["P_background"] = 0.1

        self._set_up_fixed_parameters()

        self.preparation_done = True

    def _set_up_fixed_parameters(self):
        """Validate fixed parameters and set up the alpha constraint.

        Fixed parameters are not optimized: they are inserted with exactly
        the given value (see ``_optimizer_layout``), also when the value lies
        outside the default range of the parameter. The only constraint left
        for the optimizer is alpha = a - rho * gamma, which is fixed either
        to a number or to beta.
        """
        self.constraints = []
        self._fixed_values = {}
        if not self.fixed_parameters:
            return

        fixed = {k: v for k, v in self.fixed_parameters.items()
                 if v is not None}
        alpha = fixed.pop("alpha", None)
        if self.free_background and "log10_mu" in fixed:
            raise ValueError(
                "log10_mu cannot be fixed with free_background=True: the "
                "background rate is then estimated per event from the "
                "background probabilities, and a fixed log10_mu would not "
                "be used.")
        # e.g. JSON integers: np.power(10, 25) overflows for int
        fixed = {k: float(v) for k, v in fixed.items()}

        if self.free_productivity:
            derived = {"log10_k0", "a"} & set(fixed)
            if alpha is not None:
                derived.add("alpha")
            if derived:
                raise ValueError(
                    "In free-productivity mode, log10_k0 and a are derived "
                    "from the productivities of the individual events and "
                    "cannot be fixed (nor can alpha). Fixed parameters "
                    "given: {}".format(sorted(derived)))

        if self.__theta_0 is not None and not isinstance(
                self.__theta_0, np.ndarray):
            self.__theta_0 = np.array(self.__theta_0, dtype=object)
        for name, value in fixed.items():
            idx = _PARAMETER_NAMES.index(name)
            low, high = RANGES[idx]
            if not low <= value <= high:
                self.logger.warning(
                    "  fixed {} = {} is outside the default range [{}, {}]; "
                    "it is used as given.".format(name, value, low, high))
            # start from the fixed value, so that the first expectation step
            # is already consistent with it
            self.__theta_0[idx] = value
        if np.isinf(fixed.get("log10_tau", 0)) and \
                fixed.get("omega", 1) <= 0:
            raise ValueError(
                "Without taper (log10_tau = inf), the time kernel can only "
                "be normalized for omega > 0; omega is fixed to {}".format(
                    fixed["omega"]))
        self._fixed_values = fixed
        self.logger.info("  fixed parameters: {}".format(fixed))

        if alpha is None:
            return
        self.alpha = self.beta if alpha == "beta" else alpha
        self.logger.info(
            "  Alpha has been constrained to {}".format(self.alpha))

        if all(n in fixed for n in ("a", "gamma", "rho")):
            implied = fixed["a"] - fixed["rho"] * fixed["gamma"]
            if not np.isclose(implied, self.alpha):
                raise ValueError(
                    "alpha is fixed to {}, but the fixed a, rho and gamma "
                    "imply alpha = a - rho * gamma = {}".format(
                        self.alpha, implied))
            return

        def alpha_constant(x_free):
            full = self._expand_free_parameters(x_free)
            names = self._optimizer_layout()[1]
            p = dict(zip(names, full))
            return p["a"] - p["rho"] * p["gamma"] - self.alpha

        self.constraints.append(NonlinearConstraint(alpha_constant, 0, 0))

    def _optimizer_layout(self):
        """Which entries of theta the optimizer works on.

        Returns ``(first, names, free_idx)``: the optimizer sees
        ``theta[first:]`` (the background rates are estimated in closed
        form; in free-productivity mode also log10_k0 and a are derived),
        ``names`` are those parameters' names, and ``free_idx`` the positions
        within them that are not fixed.
        """
        first = 4 if self.free_productivity else 2
        names = _PARAMETER_NAMES[first:]
        fixed = getattr(self, "_fixed_values", {})
        free_idx = [i for i, n in enumerate(names) if n not in fixed]
        return first, names, free_idx

    def _expand_free_parameters(self, x_free):
        """Insert fixed values into the vector of free parameters."""
        _, names, free_idx = self._optimizer_layout()
        fixed = getattr(self, "_fixed_values", {})
        full = np.array([fixed.get(n, np.nan) for n in names], dtype=float)
        full[free_idx] = x_free
        return full

    @property
    def theta_0(self):
        """getter"""
        return (
            parameter_array2dict(
                self.__theta_0) if self.__theta_0 is not None else None
        )

    @theta_0.setter
    def theta_0(self, t):
        if t is not None:
            # integers (e.g. from JSON) would break np.power(10, -3)
            t = {k: float(v) if isinstance(v, (int, float)) else v
                 for k, v in t.items()}
        self.__theta_0 = parameter_dict2array(t) if t is not None else None

    @property
    def fixed_parameters(self):
        """getter"""
        return (
            parameter_array2dict(self.__fixed_parameters)
            if self.__fixed_parameters is not None
            else None
        )

    @fixed_parameters.setter
    def fixed_parameters(self, t):
        self.__fixed_parameters = parameter_dict2array(
            t) if t is not None else None

    @property
    def theta(self):
        """getter"""
        return parameter_array2dict(self.__theta) if \
            self.__theta is not None else None

    @theta.setter
    def theta(self, t):
        self.__theta = parameter_dict2array(t) if t is not None else None

    def invert(self):
        """Estimate the ETAS parameters with the expectation-maximization
        algorithm (Veen and Schoenberg, 2008; Mizrahi et al., 2021).

        Starts from an initial guess of the parameters (``theta_0``, random
        if not given) and alternates between two steps. The expectation
        step calculates, given the current parameters, the probabilities
        $p_{ij}$ that event $e_j$ was triggered by event $e_i$, and the
        expected number of background events. The maximization step
        updates the parameters to maximize the likelihood of this
        branching structure. This is repeated until the parameters
        change by less than ``CONVERGENCE_TOLERANCE`` (see
        `_convergence_difference`) or ``max_iterations`` is reached.
        Requires `prepare`.

        Returns
        -------
        dict
            The estimated parameters (``self.theta``). Afterwards,
            ``self.pij``, ``self.source_events``, ``self.target_events``,
            ``self.n_hat`` (expected number of background events) and
            ``self.i`` (number of iterations) hold the results of the last
            expectation step.
        """

        if self.inversion_done:
            self.logger.warning("Inversion already done, aborting...")
            return self.theta

        self.logger.info("START INVERSION")
        diff_to_before = 100
        i = 0
        theta_old = self.__theta_0.copy()

        max_iterations = getattr(
            self, "max_iterations", DEFAULT_MAX_ITERATIONS)
        while diff_to_before >= CONVERGENCE_TOLERANCE:
            if max_iterations is not None and i >= max_iterations:
                self.logger.warning(
                    "  stopping after max_iterations = {} iterations without "
                    "convergence (last difference: {}).".format(
                        max_iterations, diff_to_before))
                break
            self.logger.info("  iteration {}".format(i))

            self.logger.debug("    expectation step")
            (
                self.pij,
                self.target_events,
                self.source_events,
                self.n_hat,
                self.i_hat,
            ) = self.expectation_step(theta_old, self.m_ref - self.delta_m / 2)

            self.logger.debug("      n_hat: {}".format(self.n_hat))
            self.logger.debug("      i_hat: {}".format(self.i_hat))

            self.logger.debug("    optimizing parameters")
            self.__theta = self.optimize_parameters(theta_old)
            if self.free_productivity:
                self.calc_a_k0_from_kappa()

            self.logger.info("    new parameters:")
            self.logger.info(
                pprint.pformat(parameter_array2dict(self.__theta), indent=4)
            )

            diff_to_before = _convergence_difference(
                theta_old, self.__theta,
                self.area * self.timewindow_length)
            self.logger.info(
                "    difference to previous: {}".format(diff_to_before))

            try:
                br = branching_ratio(theta_old, self.beta)
                self.logger.debug("    branching ratio: {}".format(br))
            except Exception:
                # e.g. alpha >= beta (infinite branching ratio), or a and
                # log10_k0 not yet known in free-productivity mode
                self.logger.debug("    branching ratio not calculated")
            theta_old = self.__theta.copy()
            if self.free_productivity:
                self.logger.debug("    updating source kappa")
                self.update_source_kappa()
            i += 1

        if diff_to_before < CONVERGENCE_TOLERANCE:
            self.logger.info(
                "  stopping here. converged after {} iterations.".format(i))
        self.i = i

        self.logger.info("    last expectation step")
        (
            self.pij,
            self.target_events,
            self.source_events,
            self.n_hat,
            self.i_hat,
        ) = self.expectation_step(theta_old, self.m_ref - self.delta_m / 2)
        self.logger.info("    n_hat: {}".format(self.n_hat))

        self.inversion_done = True

        return self.theta

    def filter_catalog(self, catalog):
        """Select the events used in the inversion.

        Keeps events with ``auxiliary_start <= time < timewindow_end`` inside
        the region (``shape_coords``), rounds magnitudes to ``delta_m``, sets
        the column ``mc_current`` (constant ``mc``, the given column for ``mc =
        'var'``, or the previous magnitude + ``delta_m`` for ``mc =
        'positive'``), and keeps events with ``magnitude >= mc_current``. Also
        sets ``self.area`` (km², or the volume for 3D inversions) and, if
        needed, lowers ``m_ref`` to the smallest ``mc_current``.

        Parameters
        ----------
        catalog : pd.DataFrame
            With columns ``latitude``, ``longitude`` (or ``x``, ``y``, ``z``),
            ``time``, ``magnitude``.

        Returns
        -------
        pd.DataFrame
            The filtered catalog, sorted by time.
        """
        len_full_catalog = catalog.shape[0]

        filtered_catalog = catalog.copy()

        # filter for events in relevant timewindow
        filtered_catalog = filtered_catalog.query(
            "time >= @ self.auxiliary_start and time < @ self.timewindow_end",
        )
        self.logger.info(
            "  {} out of {} events are within time window.".format(
                filtered_catalog.shape[0], len_full_catalog
            )
        )

        # filter for events in region of interest
        if self.shape_coords is not None:
            self.shape_coords = read_shape_coords(self.shape_coords)

            self.logger.info(
                "  Coordinates of region: {}".format(list(self.shape_coords))
            )

            if not self.three_dim:
                poly = Polygon(self.shape_coords)
                self.area = polygon_surface(poly)
                gdf = gpd.GeoDataFrame(
                    filtered_catalog,
                    geometry=gpd.points_from_xy(
                        filtered_catalog.latitude, filtered_catalog.longitude
                    ),
                )
                filtered_catalog = gdf[gdf.intersects(poly)].copy()
                filtered_catalog = filtered_catalog.drop("geometry", axis=1)
                self.logger.info("Region has {} square km".format(self.area))
            else:
                hull = ConvexHull(self.shape_coords)
                self.area = hull.volume
                # filter for events within convex hull
                # this is probably very inefficient
                in_hull_test = []
                for i, row in filtered_catalog.iterrows():
                    in_hull_test.append(
                        in_hull(self.shape_coords, row[["x", "y", "z"]].values)
                    )
                filtered_catalog = filtered_catalog[in_hull_test].copy()
                self.logger.info("Volume is {} units cubed".format(self.area))
        else:
            self.area = 6.3781e3**2 * 4 * np.pi

        self.logger.info(
            "{} events lie within target region.".format(len(filtered_catalog))
        )

        # filter for events above cutoff magnitude - delta_m/2
        # first sort by time, in case ETAS-positive is used
        filtered_catalog = filtered_catalog.sort_values(by="time")
        if self.delta_m > 0:
            filtered_catalog["magnitude"] = (
                round_half_up(filtered_catalog["magnitude"] / self.delta_m)
                * self.delta_m
            )
        if self.mc == "var":
            assert "mc_current" in filtered_catalog.columns, self.logger.error(
                'Need column "mc_current" in '
                'catalog when mc is set to "var".'
            )
        elif self.mc == "positive":
            filtered_catalog["mc_current"] = (
                filtered_catalog["magnitude"].shift(1) + self.delta_m
            )
            if self.delta_m > 0:
                filtered_catalog["mc_current"] = (
                    round_half_up(
                        filtered_catalog["mc_current"] / self.delta_m)
                    * self.delta_m
                )
        else:
            filtered_catalog["mc_current"] = self.mc
        self.logger.debug("  Checking for rounding issues...")
        if (np.abs(
            filtered_catalog["magnitude"] - filtered_catalog["mc_current"]
        ) < self.delta_m / 2).sum() == (
            filtered_catalog["magnitude"] == filtered_catalog["mc_current"]
        ).sum():
            self.logger.debug("  No rounding issues found.")
        else:
            self.logger.warning(
                "  Rounding issues found. Check if delta_m and"
                " mc_current are set correctly."
            )
        filtered_catalog = filtered_catalog.query("magnitude >= mc_current")
        self.logger.info(
            "{} events are above completeness.".format(len(filtered_catalog))
        )
        if self.mc in ["var", "positive"]:
            if filtered_catalog["mc_current"].min() < self.m_ref:
                self.logger.warning(
                    "  mc_current is below m_ref. Setting m_ref to "
                    "smallest mc_current: {}".format(
                        filtered_catalog["mc_current"].min()
                    )
                )
                self.m_ref = filtered_catalog["mc_current"].min()

        return filtered_catalog

    def prepare_target_events(self):
        """Events that can be triggered: in the primary time window, inside
        ``inner_shape_coords`` (if given), above completeness.

        Returns
        -------
        pd.DataFrame
            Indexed by ``target_id``, with the catalog columns plus
            ``mc_current_above_ref`` (and ``bg_term``, normalized to a mean of
            1, if ``bg_term`` is used).
        """
        target_events = self.catalog.query("magnitude >= mc_current").copy()
        if self.inner_shape_coords is not None:
            inner_poly = Polygon(self.inner_shape_coords)
            gdf = gpd.GeoDataFrame(
                target_events,
                geometry=gpd.points_from_xy(
                    target_events.latitude, target_events.longitude
                ),
            )
            target_events = gdf[gdf.intersects(inner_poly)].copy()
            target_events = target_events.drop("geometry", axis=1)
        target_events = target_events.query("time > @ self.timewindow_start")
        target_events["mc_current_above_ref"] = \
            target_events["mc_current"] - self.m_ref

        if self.bg_term is not None:
            target_events["bg_term"] = target_events[self.bg_term]
            target_events["bg_term"] = (
                target_events["bg_term"]
                / target_events["bg_term"].sum()
                # * self.timewindow_length
                # * self.area
                * len(target_events)
            )

        target_events.index.name = "target_id"
        return target_events

    def prepare_source_events(self):
        """Events that can trigger: all events above completeness, from
        ``auxiliary_start`` on.

        Returns
        -------
        pd.DataFrame
            Indexed by ``source_id``, with columns ``source_magnitude``,
            ``source_completeness_above_ref``, ``source_to_end_time_distance``
            and ``pos_source_to_start_time_distance`` (days, 0 for sources
            inside the primary window).
        """
        source_columns = [
            "source_magnitude",
            "source_completeness_above_ref",
            "source_to_end_time_distance",
            "pos_source_to_start_time_distance",
        ]

        sources = self.catalog.query("magnitude >= mc_current").copy()
        sources["source_to_end_time_distance"] = to_days(
            self.timewindow_end - sources["time"]
        )
        sources["pos_source_to_start_time_distance"] = np.clip(
            to_days(self.timewindow_start - sources["time"]),
            a_min=0, a_max=None
        )
        sources["source_magnitude"] = sources["magnitude"]
        sources["source_completeness_above_ref"] = (
            sources["mc_current"] - self.m_ref
        )
        sources.index.name = "source_id"

        return sources[source_columns]

    def optimize_parameters(self, theta_0, ranges=RANGES):
        """Maximization step: new parameters given the current ``Pij``.

        The background rate is ``n_hat / (area * timewindow_length)`` (unless
        fixed); the triggering parameters minimize `neg_log_likelihood`
        (or `neg_log_likelihood_free_prod`) within ``ranges``. Fixed
        parameters are not optimized but inserted with their fixed value.

        Parameters
        ----------
        theta_0 : array-like
            Current parameters (starting point), in the order of
            ``parameter_dict2array``.
        ranges : sequence of (low, high), default RANGES
            Bounds per parameter.

        Returns
        -------
        np.ndarray
            New parameters, in the order of ``parameter_dict2array``.
        """
        start_calc = dt.datetime.now()

        # estimate mu independently and remove from parameters
        fixed = getattr(self, "_fixed_values", {})
        with np.errstate(divide="ignore"):
            log10_mu = fixed.get("log10_mu", np.log10(
                self.n_hat / (self.area * self.timewindow_length)))
            if self.bg_term is not None:
                log10_iota = fixed.get("log10_iota", np.log10(
                    self.i_hat / (self.area * self.timewindow_length)))

        # the optimizer only sees the parameters that are not fixed
        first, names, free_idx = self._optimizer_layout()
        x0 = np.array(theta_0[first:], dtype=float)[free_idx]
        bounds = [ranges[first:][i] for i in free_idx]
        if np.isinf(fixed.get("log10_tau", 0)) and "omega" not in fixed:
            # without taper, the time kernel needs omega > 0
            k = [names[i] for i in free_idx].index("omega")
            bounds[k] = (max(bounds[k][0], 1e-3), bounds[k][1])
            x0[k] = np.clip(x0[k], *bounds[k])

        def minimize_free(objective, args):
            if len(free_idx) == 0:
                return self._expand_free_parameters(x0)

            def objective_free(x, *a):
                return objective(self._expand_free_parameters(x), *a)

            res = minimize(
                objective_free,
                x0=x0,
                bounds=bounds,
                args=args,
                tol=1e-12,
                constraints=self.constraints or (),
            )
            return self._expand_free_parameters(res.x)

        if self.free_productivity:
            new_theta_without_mu = minimize_free(
                neg_log_likelihood_free_prod,
                args=(
                    self.n_hat,
                    self.pij,
                    self.source_events,
                    self.timewindow_length,
                    self.timewindow_start,
                    self.area,
                    self.beta,
                    self.m_ref - self.delta_m / 2,
                ),
            )
            # log10_k0 and a are derived from the productivities afterwards
            new_theta = [log10_mu,
                         log10_iota if self.bg_term is not None else None,
                         None, None, *new_theta_without_mu]

        else:
            new_theta_without_mu = minimize_free(
                neg_log_likelihood,
                args=(self.pij, self.source_events,
                      self.m_ref - self.delta_m / 2),
            )
            # expected number of aftershocks per source at the new parameters
            self.source_events["G"] = expected_aftershocks(
                [
                    self.source_events["source_magnitude"],
                    self.source_events["pos_source_to_start_time_distance"],
                    self.source_events["source_to_end_time_distance"],
                ],
                [new_theta_without_mu, self.m_ref - self.delta_m / 2],
            )
            new_theta = [log10_mu,
                         log10_iota if self.bg_term is not None else None,
                         *new_theta_without_mu]

        self.logger.debug(
            "    optimization step took {}".format(
                dt.datetime.now() - start_calc)
        )

        return np.array(new_theta)

    def store_results(
            self, data_path="", store_pij=False, store_distances=False):
        """Write the results of the inversion to ``data_path``.

        Files (``<id>`` is ``self.id``):

        - ``parameters_<id>.json``: settings, estimated parameters and file
          names; can be passed to `load_calculation` and
          ``etas.evaluation.ETASLikelihoodCalculation``.
        - ``trig_and_bg_probs_<id>.csv``: target events with background and
          triggering probabilities.
        - ``sources_<id>.csv``: source events with expected (``G``) and
          estimated (``l_hat``) numbers of aftershocks.
        - ``pij_<id>.csv`` (if ``store_pij``): triggering probabilities per
          pair, needed for ``etas.plots``.
        - ``distances_<id>.csv`` (if ``store_distances``).
        - ``catalog_<id>.csv``, if the catalog was passed as a DataFrame.

        Parameters
        ----------
        data_path : str, default ""
            Directory, ending with a path separator; current directory if
            empty.
        store_pij, store_distances : bool, default False
        """
        if data_path == "":
            data_path = os.getcwd() + "/"

        self.logger.info("  Data will be stored in {}".format(data_path))

        fn_parameters = data_path + "parameters_{}.json".format(self.id)
        fn_ip = data_path + "trig_and_bg_probs_{}.csv".format(self.id)
        fn_src = data_path + "sources_{}.csv".format(self.id)
        fn_dist = data_path + "distances_{}.csv".format(self.id)
        fn_pij = data_path + "pij_{}.csv".format(self.id)

        os.makedirs(os.path.dirname(fn_ip), exist_ok=True)
        os.makedirs(os.path.dirname(fn_src), exist_ok=True)
        self.target_events.to_csv(fn_ip)
        self.source_events.to_csv(fn_src)

        if self.fn_catalog is None:
            self.fn_catalog = data_path + "catalog_{}.csv".format(self.id)
            self.catalog.to_csv(self.fn_catalog)

        all_info = {
            "name": self.name,
            "id": str(self.id),
            "fn_catalog": self.fn_catalog,
            "auxiliary_start": str(self.auxiliary_start),
            "timewindow_start": str(self.timewindow_start),
            "timewindow_end": str(self.timewindow_end),
            "testwindow_end": str(self.testwindow_end),
            "timewindow_length": self.timewindow_length,
            "shape_coords": str(list(self.shape_coords)),
            "delta_m": self.delta_m,
            "mc": self.mc,
            "m_ref": self.m_ref,
            "coppersmith_multiplier": self.coppersmith_multiplier,
            "earth_radius": self.earth_radius,
            "bw_sq": self.bw_sq,
            "free_productivity": self.free_productivity,
            "free_background": self.free_background,
            "bg_term": self.bg_term,
            "preparation_done": self.preparation_done,
            "inversion_done": self.inversion_done,
            "n_target_events": len(self.target_events),
            "area": self.area,
            "log10_mu_range": LOG10_MU_RANGE,
            "log10_iota_range": LOG10_IOTA_RANGE,
            "log10_k0_range": LOG10_K0_RANGE,
            "a_range": A_RANGE,
            "log10_c_range": LOG10_C_RANGE,
            "omega_range": OMEGA_RANGE,
            "log10_tau_range": LOG10_TAU_RANGE,
            "log10_d_range": LOG10_D_RANGE,
            "gamma_range": GAMMA_RANGE,
            "rho_range": RHO_RANGE,
            "beta": self.beta,
            "b_positive": self.b_positive,
            "three_dim": self.three_dim,
            "space_unit_in_meters": self.space_unit_in_meters,
            "n_hat": self.n_hat,
            "i_hat": self.i_hat,
            "calculation_date": str(self.calculation_date),
            "initial_values": self.theta_0,
            "final_parameters": self.theta,
            "n_iterations": self.i,
            "max_iterations": getattr(self, "max_iterations", None),
            "fn_ip": fn_ip,
            "fn_src": fn_src,
        }

        if store_pij:
            os.makedirs(os.path.dirname(fn_pij), exist_ok=True)
            self.pij.to_csv(fn_pij)
            all_info["fn_pij"] = fn_pij

        if store_distances:
            os.makedirs(os.path.dirname(fn_dist), exist_ok=True)
            self.distances.to_csv(fn_dist)
            all_info["fn_dist"] = fn_dist

        with open(fn_parameters, "w") as f:
            f.write(json.dumps(all_info))

    def calculate_distances(self):
        """
        Precalculates distances in time and space between events that are
        potentially related to each other.
        """

        calc_start = dt.datetime.now()

        # only use data above completeness magnitude
        if self.delta_m > 0:
            self.catalog["magnitude"] = (
                round_half_up(
                    self.catalog["magnitude"] / self.delta_m) * self.delta_m
            )
        relevant = self.catalog.query("magnitude >= mc_current").copy()
        # sorting by time is not needed anymore,
        # because it now happens when filtering
        # for completeness (for ETAS-positive)
        # relevant.sort_values(by="time", inplace=True)

        # all entries can be sources, but targets only after timewindow start
        targets = relevant.query("time>=@self.timewindow_start").copy()
        if self.inner_shape_coords is not None:
            inner_poly = Polygon(self.inner_shape_coords)
            self.area = polygon_surface(inner_poly)
            gdf = gpd.GeoDataFrame(
                targets,
                geometry=gpd.points_from_xy(
                    targets.latitude, targets.longitude),
            )
            targets = gdf[gdf.intersects(inner_poly)].copy()
            targets = targets.drop("geometry", axis=1)

        if self.mc == "positive":
            beta = estimate_beta_tinti(
                targets["magnitude"] - (targets["mc_current"] - 0.1),
                mc=self.delta_m,
                delta_m=self.delta_m,
            )
        elif self.b_positive:
            beta = estimate_beta_positive(
                targets["magnitude"], delta_m=self.delta_m)
        else:
            beta = estimate_beta_tinti(
                targets["magnitude"] - targets["mc_current"],
                mc=0,
                delta_m=self.delta_m
            )
        logger.info("    beta is {}".format(beta))

        # calculate max distance so that only events closer than
        # distance_range are considered possibly related
        # this is done to reduce the size of the distance matrix
        relevant["distance_range_squared"] = np.square(
            coppersmith(relevant["magnitude"], 4)[
                "SSRL"] * self.coppersmith_multiplier
        )
        if self.three_dim:
            # translate to the space unit of the catalog
            units_per_km = 1000 / self.space_unit_in_meters
            relevant["distance_range_squared"] = (
                relevant["distance_range_squared"] * units_per_km
            )

        # calculate distances to timewindow boundaries
        relevant["source_to_end_time_distance"] = to_days(
            self.timewindow_end - relevant["time"]
        )
        relevant["pos_source_to_start_time_distance"] = np.clip(
            to_days(self.timewindow_start - relevant["time"]),
            a_min=0,
            a_max=None
        )

        if not self.three_dim:
            # translate target lat, lon to radians for spherical distance
            # calculation
            targets["target_lat_rad"] = np.radians(targets["latitude"])
            targets["target_lon_rad"] = np.radians(targets["longitude"])
        targets["target_time"] = targets["time"]
        targets["target_id"] = targets.index
        targets["target_time"] = targets["time"]
        targets["target_completeness_above_ref"] = targets["mc_current"]
        # columns that are needed later
        targets["source_id"] = "i"
        targets["source_magnitude"] = 0.0
        targets["source_completeness_above_ref"] = 0.0
        targets["time_distance"] = 0.0
        targets["spatial_distance_squared"] = 0.0
        targets["source_to_end_time_distance"] = 0.0
        targets["pos_source_to_start_time_distance"] = 0.0

        targets = targets.sort_values(by="time")

        # define index and columns that are later going to be needed
        if pd.__version__ >= "0.24.0":
            index = pd.MultiIndex(levels=[[], []],
                                  names=["source_id", "target_id"],
                                  codes=[[], []]
                                  )
        else:
            index = pd.MultiIndex(levels=[[], []],
                                  names=["source_id", "target_id"],
                                  labels=[[], []]
                                  )

        columns = [
            "target_time",
            "source_magnitude",
            "source_completeness_above_ref",
            "target_completeness_above_ref",
            "spatial_distance_squared",
            "time_distance",
            "source_to_end_time_distance",
            "pos_source_to_start_time_distance",
        ]
        res_df = pd.DataFrame(index=index, columns=columns)

        df_list = []

        logger.info("  number of sources: {}".format(len(relevant.index)))
        logger.info("  number of targets: {}".format(len(targets.index)))

        if self.three_dim:
            logger.info("    assuming 3D Euclidian coordinates.")
        else:
            logger.info("    assuming 2D lat/long coordinates.")
        for source in relevant.itertuples():
            stime = source.time

            # filter potential targets
            if source.time < self.timewindow_start:
                potential_targets = targets.copy()
            else:
                potential_targets = targets.query("time>@stime").copy()
            targets = potential_targets.copy()

            if potential_targets.shape[0] == 0:
                continue

            # calculate spatial distance from source to target event
            if self.three_dim:
                sx = source.x
                sy = source.y
                sz = source.z

                potential_targets["spatial_distance_squared"] = (
                    np.square((sx - potential_targets["x"]))
                    + np.square((sy - potential_targets["y"]))
                    + np.square((sz - potential_targets["z"]))
                )
            else:
                slatrad = np.radians(source.latitude)
                slonrad = np.radians(source.longitude)

                # calculate spatial distance from source to target event
                potential_targets["spatial_distance_squared"] = np.square(
                    haversine(
                        slatrad,
                        potential_targets["target_lat_rad"],
                        slonrad,
                        potential_targets["target_lon_rad"],
                        self.earth_radius,
                    )
                )

            # filter for only small enough distances
            drs = source.distance_range_squared  # noqa
            potential_targets = potential_targets.query(
                "spatial_distance_squared <= @drs")

            # get source id and info of target events
            potential_targets["source_id"] = source.Index
            potential_targets["source_magnitude"] = source.magnitude
            potential_targets["source_completeness_above_ref"] = \
                source.mc_current

            # calculate time distance from source to target event
            potential_targets["time_distance"] = to_days(
                potential_targets["target_time"] - stime
            )

            # calculate time distance from source event to timewindow
            # boundaries for integration later
            potential_targets["source_to_end_time_distance"] = (
                source.source_to_end_time_distance
            )
            potential_targets["pos_source_to_start_time_distance"] = (
                source.pos_source_to_start_time_distance
            )

            # append to resulting dataframe
            df_list.append(potential_targets)

        res_df = (
            pd.concat(df_list)[["source_id", "target_id"] + columns]
            .reset_index()
            .set_index(["source_id", "target_id"])
        )
        res_df["source_completeness_above_ref"] = (
            res_df["source_completeness_above_ref"] - self.m_ref
        )
        res_df["target_completeness_above_ref"] = (
            res_df["target_completeness_above_ref"] - self.m_ref
        )

        logger.debug(
            "  took {} to prepare the data".format(
                dt.datetime.now() - calc_start)
        )

        return res_df

    def expectation_step(self, theta, mc_min):
        r"""Expectation step: triggering and background probabilities.

        For each pair of a possible triggering event (source) $e_i$ and a
        possible aftershock (target) $e_j$, the probability that $e_j$ was
        triggered by $e_i$ is

        $$
        p_{ij} = \frac{g_{ij}}{\mu_j + \sum_k (1 + \xi_k) \, g_{kj}},
        $$

        where $g_{ij}$ is the triggering kernel evaluated for the magnitude
        of $e_i$, the distance between $e_i$ and $e_j$ and the time between
        them (column ``gij``); $\mu_j$ is the background rate at $e_j$
        (constant, or per event with ``free_background``, plus the
        ``bg_term`` contribution if used); the sum goes over all possible
        sources $e_k$ of $e_j$; and $1 + \xi_k$ is the
        `responsibility_factor` of $e_k$, which accounts for triggering by
        undetected events (1 for a constant completeness).
        The denominator is the total rate at $e_j$ (column ``tot_rates``).
        Summing $p_{ij}$ over all sources $e_i$ gives the probability that
        $e_j$ was triggered (``P_triggered``); summing over all targets $e_j$,
        weighted by the `observation_factor` $1 + \zeta_j$ of each target,
        gives the expected number of direct aftershocks of $e_i$
        (``l_hat``).

        Parameters
        ----------
        theta : array-like
            Current parameters, in the order of ``parameter_dict2array``.
        mc_min : float
            Reference magnitude of the kernel, ``m_ref - delta_m / 2``.

        Returns
        -------
        tuple
            ``(Pij, target_events, source_events, n_hat, i_hat)``: the pair
            table with columns ``gij``, ``Pij``, ``tot_rates``, ``xi_plus_1``,
            ``zeta_plus_1``; target events with ``P_triggered``,
            ``P_background`` (and ``P_induced``); source events with ``l_hat``;
            the expected number of background events; and of induced events (0
            without ``bg_term``).
        """
        calc_start = dt.datetime.now()
        log10_mu = theta[0]
        mu = np.power(10, log10_mu)

        if self.bg_term is not None:
            log10_iota = theta[1]
            iota = np.power(10, log10_iota)

        # calculate the triggering density values gij
        logger.debug("    calculating gij")
        Pij_0 = self.distances.copy()
        source_kappa = (
            pd.merge(
                Pij_0[[]],
                self.source_events["source_kappa"],
                left_index=True,
                right_index=True,
            )
            .copy()
            .fillna(0)
            if self.free_productivity
            else {"source_kappa": None}
        )
        Pij_0["gij"] = triggering_kernel(
            [
                Pij_0["time_distance"],
                Pij_0["spatial_distance_squared"],
                Pij_0["source_magnitude"],
                source_kappa["source_kappa"],
            ],
            [theta, mc_min],
        )

        # responsibility factor for invisible triggering events
        Pij_0["xi_plus_1"] = responsibility_factor(
            theta, self.beta, Pij_0["source_completeness_above_ref"]
        )
        Pij_0["zeta_plus_1"] = observation_factor(
            self.beta, Pij_0["target_completeness_above_ref"]
        )
        # calculate muj for each target. currently constant, could be improved
        target_events_0 = self.target_events.copy()
        target_events_0["mu"] = mu
        if self.free_background:
            target_events_0["mu"] = (
                (
                    (
                        (
                            np.exp(
                                -1 / 2
                                * Pij_0["spatial_distance_squared"]
                                / self.bw_sq
                            )
                            / (self.bw_sq * 2 * np.pi)
                        ).mul(target_events_0["P_background"], level=0)
                    )
                    .groupby(level=1)
                    .sum()
                    + target_events_0["P_background"]
                    / (self.bw_sq * 2 * np.pi)
                )
                / (
                    self.timewindow_length
                    # TODO: divide by tw_length minus
                    # target_to_end_time_distance
                )
            ).fillna(0)
        else:
            target_events_0["mu"] = mu

        if self.bg_term is not None:
            target_events_0["ind"] = iota * target_events_0["bg_term"]

        # calculate triggering probabilities Pij
        logger.debug("    calculating Pij")
        Pij_0["tot_rates"] = 0
        Pij_0["tot_rates"] = (
            Pij_0["tot_rates"]
            .add((Pij_0["gij"] * Pij_0["xi_plus_1"]).groupby(level=1).sum())
            .add(target_events_0["mu"])
        )
        if self.bg_term is not None:
            Pij_0["tot_rates"] = Pij_0["tot_rates"].add(target_events_0["ind"])
        Pij_0["Pij"] = Pij_0["gij"].div(Pij_0["tot_rates"])

        tot_rates_by_target = (
            Pij_0.groupby(level=1).first()["tot_rates"]
            .reindex(target_events_0.index)
        )
        if self.bg_term is not None:
            tot_rates_by_target = tot_rates_by_target.fillna(
                target_events_0["mu"] + target_events_0["ind"]
            )
        else:
            tot_rates_by_target = tot_rates_by_target.fillna(
                target_events_0["mu"]
            )

        # calculate probabilities of being triggered or background
        target_events_0["P_triggered"] = 0
        target_events_0["P_triggered"] = (
            target_events_0["P_triggered"]
            .add(Pij_0["Pij"].groupby(level=1).sum())
            .fillna(0)
        )
        target_events_0["P_background"] = target_events_0["mu"].div(
            tot_rates_by_target
        )

        if self.bg_term is not None:
            target_events_0["P_induced"] = target_events_0["ind"].div(
                tot_rates_by_target
            )
        target_events_0["zeta_plus_1"] = observation_factor(
            self.beta, target_events_0["mc_current_above_ref"]
        )

        # calculate expected number of background events
        logger.debug("    calculating n_hat and l_hat")
        n_hat_0 = (
            target_events_0["P_background"] * target_events_0["zeta_plus_1"]
        ).sum()
        i_hat_0 = 0
        if self.bg_term is not None:
            i_hat_0 = (
                target_events_0["P_induced"] * target_events_0["zeta_plus_1"]
            ).sum()

        # calculate aftershocks per source event
        source_events_0 = self.source_events.copy()
        source_events_0["l_hat"] = (
            (Pij_0["Pij"] * Pij_0["zeta_plus_1"]).groupby(level=0).sum()
        )
        # filling NaN with 0 to indicate
        # that those sources have no aftershocks (yet)
        source_events_0["l_hat"] = source_events_0["l_hat"].fillna(0)

        logger.debug(
            "    expectation step took {}".format(
                dt.datetime.now() - calc_start)
        )
        return Pij_0, target_events_0, source_events_0, n_hat_0, i_hat_0

    def update_source_kappa(self):
        """Update the individual productivities (free-productivity mode).

        Fixed-point update ``kappa <- kappa * l_hat / G``, where ``G`` is the
        expected number of aftershocks under the current kappa and ``l_hat``
        the estimated number from the expectation step.
        """
        self.source_events["G"] = expected_aftershocks_free_prod(
            [
                self.source_events["source_magnitude"],
                self.source_events["source_kappa"],
                self.source_events["pos_source_to_start_time_distance"],
                self.source_events["source_to_end_time_distance"],
            ],
            [self.__theta[4:], self.m_ref - self.delta_m / 2],
        )
        # filling nan with 0 because I believe the only way this can be
        # undefined is when G is zero, which only happens when source_kappa
        # is zero. so should be fine.
        self.source_events["source_kappa"] = (
            self.source_events["source_kappa"]
            * self.source_events["l_hat"]
            / self.source_events["G"]
        ).fillna(0)

    def calc_a_k0_from_kappa(self):
        """Set ``a`` and ``log10_k0`` in ``theta`` by fitting the
        productivity law to the individual productivities of the events in
        the primary window (free-productivity mode). See
        `calc_a_k0_from_kappa`.
        """
        prim_mags = self.catalog.query(
            "time >=@self.timewindow_start")["magnitude"]
        kappas_estimated = pd.merge(
            prim_mags,
            self.source_events[["source_kappa"]],
            left_index=True,
            right_index=True,
            how="left",
        ).fillna(0)
        self.__theta[3], self.__theta[2] = calc_a_k0_from_kappa(
            kappas_estimated["source_kappa"],
            kappas_estimated["magnitude"] - (self.m_ref - self.delta_m / 2),
        )
