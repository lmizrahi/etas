#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################################################################
# inversion of ETAS parameters
#
# as described by Mizrahi et al., 2021
# Leila Mizrahi, Shyam Nandan, Stefan Wiemer;
# The Effect of Declustering on the Size Distribution of Mainshocks.
# Seismological Research Letters 2021; doi: https://doi.org/10.1785/0220200231
##############################################################################

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
    proj_wgs84 = pyproj.CRS('EPSG:4326')
    proj_aea = pyproj.CRS(
        proj="aea", lat_1=polygon.bounds[0], lat_2=polygon.bounds[2])

    transformer = pyproj.Transformer.from_crs(
        proj_wgs84, proj_aea)

    geom_area = ops.transform(transformer.transform, polygon)

    return geom_area.area / 1e6


def in_hull(points, x):
    n_points = len(points)
    c = np.zeros(n_points)
    A = np.r_[points.T, np.ones((1, n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success


def hav(theta):
    return np.square(np.sin(theta / 2))


def haversine(lat_rad_1, lat_rad_2, lon_rad_1, lon_rad_2, earth_radius=6.3781e3):
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
    return timediff / dt.timedelta(days=1)


def upper_gamma_ext(a, x):
    if a > 0:
        return gammaincc(a, x) * gamma_func(a)
    elif a == 0:
        return exp1(x)
    else:
        return (upper_gamma_ext(a + 1, x) - np.power(x, a) * np.exp(-x)) / a


def parameter_array2dict(theta):
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
    zeta_plus_1 = np.exp(beta * delta_mc)
    return zeta_plus_1


def expected_aftershocks(event, params, no_start=False, no_end=False):
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


def ll_aftershock_term(l_hat, g):
    mask = g != 0
    term = -1 * gammaln(l_hat + 1) - g
    term = term + l_hat * np.where(mask, np.log(g), -300)
    return term


def neg_log_likelihood(theta, Pij, source_events, mc_min):
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

    source_events["G"] = expected_aftershocks(
        [
            source_events["source_magnitude"],
            source_events["pos_source_to_start_time_distance"],
            source_events["source_to_end_time_distance"],
        ],
        [theta, mc_min],
    )

    aftershock_term = ll_aftershock_term(
        source_events["l_hat"],
        source_events["G"],
    ).sum()

    # space time distribution term
    Pij["likelihood_term"] = (
        (
            omega * np.log(tau)
            - np.log(upper_gamma_ext(-omega, c / tau))
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
    distribution_term = Pij["Pij"].mul(Pij["likelihood_term"]).sum()

    total = aftershock_term + distribution_term

    return -1 * total


def expected_aftershocks_free_prod(event, params, no_start=False, no_end=False):
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
            event_magnitude, event_kappa, event_time_to_start, event_time_to_end = event

    number_factor = event_kappa
    area_factor = (
        np.pi * np.power(d * np.exp(gamma
                         * (event_magnitude - mc)), -1 * rho) / rho
    )

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

    source_events["G"] = expected_aftershocks_free_prod(
        [
            source_events["source_magnitude"],
            source_events["source_kappa"],
            source_events["pos_source_to_start_time_distance"],
            source_events["source_to_end_time_distance"],
        ],
        [theta, mc_min],
    )

    # space time distribution term
    Pij["likelihood_term"] = (
        (
            omega * np.log(tau)
            - np.log(upper_gamma_ext(-omega, c / tau))
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
    distribution_term = Pij["Pij"].mul(Pij["likelihood_term"]).sum()

    total = distribution_term

    return -1 * total


def prod_neg_log_lik(a, args):
    sk, md, weights = args
    k_0 = np.sum(weights * sk) / (weights * np.exp(a * md)).sum()
    ll = (weights * md * (sk - k_0 * np.exp(a * md))).sum()
    return np.abs(ll)


def calc_a_k0_from_kappa(kappa, m_diff, weights=1):
    res = minimize(
        prod_neg_log_lik, x0=1.5, args=[kappa, m_diff, weights], bounds=[(0, 5)]
    )
    a = res.x[0]
    log10_k0 = np.log10(np.sum(kappa * weights)
                        / (np.exp(a * m_diff) * weights).sum())
    return a, log10_k0


def read_shape_coords(shape_coords):
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
        Class to invert ETAS parameters.


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
                    [lat3, lon3]]``). Necessary unless globe=True when calling
                    invert_etas_params(), i.e. `invert_etas_params(
                    inversion_config, globe=True)`. In this case, the whole
                    globe is considered.
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
                default: 2
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
        except:
            self.testwindow_end = None

        self.timewindow_length = to_days(
            self.timewindow_end - self.timewindow_start)
        self.calculation_date = dt.datetime.now()

        self.free_background = metadata.get("free_background", False)
        self.free_productivity = metadata.get("free_productivity", False)
        self.bg_term = metadata.get("bg_term", None)

        self.logger.info(
            "  Time Window: \n      {} (aux start)\n      {} "
            "(start)\n      {} (end).".format(
                self.auxiliary_start, self.timewindow_start, self.timewindow_end
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

        if obj.fn_catalog:
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
            obj.logger.warning("Sources could not be loaded. \
                               Only ok to proceed in specific use cases.")

        if "fn_ip" in metadata:
            obj.target_events = pd.read_csv(
                metadata["fn_ip"],
                index_col=0,
                parse_dates=["time"]
        )
        else:
            obj.logger.warning("Targets could not be loaded. \
                               Only ok to proceed in specific use cases.")

        if "fn_pij" in metadata:
            obj.pij = pd.read_csv(
                metadata["fn_pij"],
                index_col=["source_id", "target_id"],
                parse_dates=["target_time"],
            )
        else:
            obj.logger.warning("Pij could not be loaded.")

        if "fn_dist" in metadata:
            obj.distances = pd.read_csv(
                metadata["fn_dist"],
                index_col=["source_id", "target_id"],
                parse_dates=["target_time"],
            )
        else:
            obj.logger.warning("Distances could not be loaded.")

        return obj

    def prepare(self):
        if self.preparation_done:
            self.logger.warning("Preparation already done, aborting...")
            pass

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
                "  beta of primary catalog is {}, estimated with b-positive".format(
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

        if self.fixed_parameters:
            self.constraints = []
            starting_index = 2

            if "alpha" in self.fixed_parameters:
                if self.fixed_parameters["alpha"] == "beta":
                    self.alpha = self.beta
                else:
                    self.alpha = self.fixed_parameters["alpha"]

                starting_index = 3

                def alpha_constant(x):
                    return x[1] - x[6] * x[7] - self.alpha
                self.constraints.append(
                    NonlinearConstraint(alpha_constant, 0, 0))
                self.logger.info(
                    "  Alpha has been constrained to {}".format(self.alpha)
                )

            idx_fixed = [
                k
                for k, a in enumerate(self.__fixed_parameters[starting_index:])
                if a is not None
            ]
            if len(idx_fixed) > 0:
                def param_constant(x): return np.array(
                    [x[k] for k in idx_fixed]
                ) - np.array(
                    [self.__fixed_parameters[starting_index:][k]
                        for k in idx_fixed]
                )
                self.constraints.append(
                    NonlinearConstraint(param_constant, 0, 0))

            self.logger.info(
                "  {} other constraints have been set up".format(
                    len(idx_fixed))
            )

        self.preparation_done = True

    @property
    def theta_0(self):
        """getter"""
        return (
            parameter_array2dict(
                self.__theta_0) if self.__theta_0 is not None else None
        )

    @theta_0.setter
    def theta_0(self, t):
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
        return parameter_array2dict(self.__theta) if self.__theta is not None else None

    @theta.setter
    def theta(self, t):
        self.__theta = parameter_dict2array(t) if t is not None else None

    def invert(self):
        """
        Invert the ETAS (or flETAS) parameters.
        """

        if self.inversion_done:
            self.logger.warning("Inversion already done, aborting...")
            return self.theta

        self.logger.info("START INVERSION")
        diff_to_before = 100
        i = 0
        theta_old = self.__theta_0[:]

        while diff_to_before >= 0.001:
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

            diff_to_before = calc_diff_to_before(theta_old, self.__theta)
            self.logger.info(
                "    difference to previous: {}".format(diff_to_before))

            try:
                br = branching_ratio(theta_old, self.beta)
                self.logger.debug("    branching ratio: {}".format(br))
            except:
                self.logger.debug("    branching ratio not calculated")
            theta_old = self.__theta[:]
            if self.free_productivity:
                self.logger.debug("    updating source kappa")
                self.update_source_kappa()
            i += 1

        self.logger.info(
            "  stopping here. converged after " "{} iterations.".format(i))
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
        len_full_catalog = catalog.shape[0]

        filtered_catalog = catalog.copy()

        # filter for events in relevant timewindow
        filtered_catalog.query(
            "time >= @ self.auxiliary_start and time < @ self.timewindow_end",
            inplace=True,
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
                filtered_catalog.drop("geometry", axis=1, inplace=True)
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
        filtered_catalog.sort_values(by="time", inplace=True)
        if self.delta_m > 0:
            filtered_catalog["magnitude"] = (
                round_half_up(filtered_catalog["magnitude"] / self.delta_m)
                * self.delta_m
            )
        if self.mc == "var":
            assert "mc_current" in filtered_catalog.columns, self.logger.error(
                'Need column "mc_current" in ' 'catalog when mc is set to "var".'
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
        filtered_catalog.query("magnitude >= mc_current", inplace=True)
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
            target_events.drop("geometry", axis=1, inplace=True)
        target_events.query("time > @ self.timewindow_start", inplace=True)
        target_events["mc_current_above_ref"] = target_events["mc_current"] - self.m_ref

        if self.bg_term is not None:
            target_events["bg_term"] = target_events[self.bg_term]
            target_events["bg_term"] = (
                target_events["bg_term"]
                / target_events["bg_term"].sum()
                * self.timewindow_length
                * self.area
            )

        target_events.index.name = "target_id"
        return target_events

    def prepare_source_events(self):
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
        start_calc = dt.datetime.now()

        # estimate mu independently and remove from parameters
        mu_hat = self.n_hat / (self.area * self.timewindow_length)
        if self.bg_term is not None:
            iota_hat = self.i_hat / (self.area * self.timewindow_length)

        if self.free_productivity:
            # select values from theta needed in free prod mode
            theta_0_without_mu = theta_0[4:]
            bounds = ranges[4:]

            res = minimize(
                neg_log_likelihood_free_prod,
                x0=theta_0_without_mu,
                bounds=bounds,
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
                tol=1e-12,
                constraints=self.constraints,
            )

            new_theta_without_mu = res.x
            if self.bg_term is not None:
                new_theta = [
                    np.log10(mu_hat),
                    np.log10(iota_hat),
                    None,
                    None * new_theta_without_mu,
                ]
            else:
                new_theta = [np.log10(mu_hat), None, None,
                             None, *new_theta_without_mu]

        else:
            theta_0_without_mu = theta_0[2:]
            bounds = ranges[2:]

            res = minimize(
                neg_log_likelihood,
                x0=theta_0_without_mu,
                bounds=bounds,
                args=(self.pij, self.source_events,
                      self.m_ref - self.delta_m / 2),
                tol=1e-12,
                constraints=self.constraints,
            )

            new_theta_without_mu = res.x
            if self.bg_term is not None:
                new_theta = [
                    np.log10(mu_hat),
                    np.log10(iota_hat),
                    *new_theta_without_mu,
                ]
            else:
                new_theta = [np.log10(mu_hat), None, *new_theta_without_mu]

        self.logger.debug(
            "    optimization step took {}".format(
                dt.datetime.now() - start_calc)
        )

        return np.array(new_theta)

    def store_results(self, data_path="", store_pij=False, store_distances=False):
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
            "log10_mu_range": RANGES[0],
            "log10_k0_range": RANGES[1],
            "a_range": RANGES[2],
            "log10_c_range": RANGES[3],
            "omega_range": RANGES[4],
            "log10_tau_range": RANGES[5],
            "log10_d_range": RANGES[6],
            "gamma_range": RANGES[7],
            "rho_range": RANGES[8],
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
        # because it now happens when filtering for completeness (for ETAS-positive)
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
            targets.drop("geometry", axis=1, inplace=True)

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
                targets["magnitude"] - targets["mc_current"], mc=0, delta_m=self.delta_m
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
            to_days(self.timewindow_start - relevant["time"]), a_min=0, a_max=None
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
            index = pd.MultiIndex(
                levels=[[], []], names=["source_id", "target_id"], codes=[[], []]
            )
        else:
            index = pd.MultiIndex(
                levels=[[], []], names=["source_id", "target_id"], labels=[[], []]
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
            potential_targets.query(
                "spatial_distance_squared <= @drs", inplace=True)

            # get source id and info of target events
            potential_targets["source_id"] = source.Index
            potential_targets["source_magnitude"] = source.magnitude
            potential_targets["source_completeness_above_ref"] = source.mc_current

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
                                * Pij_0["spatial_distance_squared"] / self.bw_sq
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

        # calculate probabilities of being triggered or background
        target_events_0["P_triggered"] = 0
        target_events_0["P_triggered"] = (
            target_events_0["P_triggered"]
            .add(Pij_0["Pij"].groupby(level=1).sum())
            .fillna(0)
        )
        target_events_0["P_background"] = (
            target_events_0["mu"] / Pij_0.groupby(level=1).first()["tot_rates"]
        ).fillna(1)

        if self.bg_term is not None:
            target_events_0["P_induced"] = (
                target_events_0["ind"]
                / Pij_0.groupby(level=1).first()["tot_rates"]
            )
        # do we also want do add .fillna(1) here?
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
