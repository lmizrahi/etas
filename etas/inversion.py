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

import logging
from scipy.optimize import minimize
from scipy.special import gamma as gamma_func, gammaln, gammaincc, exp1

import pandas as pd
import numpy as np
import geopandas as gpd
import datetime as dt
import json
import os
import uuid
import pprint

from functools import partial
import pyproj
from shapely.geometry import Polygon
import shapely.ops as ops

from etas.mc_b_est import round_half_up, estimate_beta_tinti

logger = logging.getLogger(__name__)

# ranges for parameters
LOG10_MU_RANGE = (-10, 0)
LOG10_K0_RANGE = (-4, 0)
A_RANGE = (0.01, 5.)
LOG10_C_RANGE = (-8, 0)
OMEGA_RANGE = (-0.99, 1)
LOG10_TAU_RANGE = (0.01, 5)
LOG10_D_RANGE = (-4, 3)
GAMMA_RANGE = (0.01, 5.)
RHO_RANGE = (0.01, 5.)
RANGES = LOG10_MU_RANGE, LOG10_K0_RANGE, A_RANGE, LOG10_C_RANGE, \
    OMEGA_RANGE, LOG10_TAU_RANGE, LOG10_D_RANGE, GAMMA_RANGE, RHO_RANGE


def coppersmith(mag, fault_type):
    '''
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
    '''

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

    return {
        'SRL': SRL,
        'SSRL': SSRL,
        'RW': RW,
        'RA': RA,
        'AD': AD
    }


def rectangle_surface(lat1, lat2, lon1, lon2):
    vertices = [[lat1, lon1],
                [lat2, lon1],
                [lat2, lon2],
                [lat1, lon2]]
    polygon = Polygon(vertices)

    geom_area = ops.transform(
        partial(pyproj.transform,
                pyproj.Proj('EPSG:4326'),
                pyproj.Proj(proj='aea',
                            lat1=polygon.bounds[0],
                            lat2=polygon.bounds[2])),
        polygon)

    return geom_area.area / 1e6


def polygon_surface(polygon):
    geom_area = ops.transform(
        partial(
            pyproj.transform,
            pyproj.Proj('EPSG:4326'),
            pyproj.Proj(
                proj='aea',
                lat_1=polygon.bounds[0],
                lat_2=polygon.bounds[2])),
        polygon)
    return geom_area.area / 1e6


def hav(theta):
    return np.square(np.sin(theta / 2))


def haversine(lat_rad_1,
              lat_rad_2,
              lon_rad_1,
              lon_rad_2,
              earth_radius=6.3781e3):
    '''
    Calculates the distance on a sphere.
    '''
    d = 2 * earth_radius * np.arcsin(
        np.sqrt(hav(lat_rad_1 - lat_rad_2)
                + np.cos(lat_rad_1)
                * np.cos(lat_rad_2)
                * hav(lon_rad_1 - lon_rad_2)
                )
    )
    return d


def branching_ratio(theta, beta):
    log10_mu, log10_k0, a, log10_c, omega, log10_tau, log10_d, gamma, rho = \
        theta
    k0 = np.power(10, log10_k0)
    c = np.power(10, log10_c)
    d = np.power(10, log10_d)
    tau = np.power(10, log10_tau)

    eta = beta * k0 * np.pi * np.power(d, -rho) * np.power(tau, -omega) \
        * np.exp(c / tau) * upper_gamma_ext(-omega, c / tau) \
        / (rho * (-a + beta + gamma * rho))
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
    return dict(zip(['log10_mu', 'log10_k0', 'a', 'log10_c',
                'omega', 'log10_tau', 'log10_d', 'gamma', 'rho'], theta))


def parameter_dict2array(parameters):
    order = [
        'log10_mu',
        'log10_k0',
        'a',
        'log10_c',
        'omega',
        'log10_tau',
        'log10_d',
        'gamma',
        'rho']
    return np.array([
        parameters.get(key, None) for key in order
    ])


def create_initial_values(ranges=RANGES):
    return [np.random.uniform(*r) for r in ranges]


def triggering_kernel(metrics, params):
    '''
    Given time distance in days and squared space distance in square km and
    magnitude of target event, calculate the (not normalized) likelihood,
    that source event triggered target event.
    '''
    time_distance, spatial_distance_squared, m, source_kappa = metrics
    theta, mc = params

    log10_mu, log10_k0, a, log10_c, omega, log10_tau, log10_d, gamma, rho = \
        theta

    if source_kappa is None:
        k0 = np.power(10, log10_k0)
    c = np.power(10, log10_c)
    tau = np.power(10, log10_tau)
    d = np.power(10, log10_d)
    aftershock_number = source_kappa \
        if source_kappa is not None else k0 * np.exp(a * (m - mc))
    time_decay = np.exp(-time_distance / tau) / \
        np.power((time_distance + c), (1 + omega))
    space_decay = 1 / np.power(
        (spatial_distance_squared + d * np.exp(gamma * (m - mc))),
        (1 + rho)
    )

    res = aftershock_number * time_decay * space_decay
    return res


def responsibility_factor(theta, beta, delta_mc):
    log10_mu, log10_k0, a, log10_c, omega, log10_tau, log10_d, gamma, rho = \
        theta

    xi_plus_1 = 1 / (np.exp(
        (a - beta - gamma * rho) * delta_mc
    ))

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
    area_factor = np.pi * np.power(
        d * np.exp(gamma * (event_magnitude - mc)),
        -1 * rho
    ) / rho

    time_factor = np.exp(c / tau) * np.power(tau,
                                             - omega)  # * gamma_func(-omega)

    if no_start:
        time_fraction = upper_gamma_ext(-omega, c / tau)
    else:
        time_fraction = upper_gamma_ext(-omega,
                                        (event_time_to_start + c) / tau)
    if not no_end:
        time_fraction = time_fraction - \
            upper_gamma_ext(-omega, (event_time_to_end + c) / tau)

    time_factor = time_factor * time_fraction

    return number_factor * area_factor * time_factor


def ll_aftershock_term(l_hat, g):
    mask = g != 0
    term = -1 * gammaln(l_hat + 1) - g
    term = term + l_hat * np.where(mask, np.log(g, where=mask), -300)
    return term


def neg_log_likelihood(theta, Pij, source_events, mc_min):

    assert Pij.index.names == ('source_id', 'target_id'), logger.error(
        'Pij must have multiindex with names "source_id", "target_id"')
    assert source_events.index.name == 'source_id', \
        logger.error('source_events must have index with name "source_id"')

    log10_k0, a, log10_c, omega, log10_tau, log10_d, gamma, rho = theta

    c = np.power(10, log10_c)
    tau = np.power(10, log10_tau)
    d = np.power(10, log10_d)

    source_events['G'] = expected_aftershocks(
        [
            source_events['source_magnitude'],
            source_events['pos_source_to_start_time_distance'],
            source_events['source_to_end_time_distance']
        ],
        [theta, mc_min]
    )

    aftershock_term = ll_aftershock_term(
        source_events['l_hat'],
        source_events['G'],
    ).sum()

    # space time distribution term
    Pij['likelihood_term'] = (
        (
            omega * np.log(tau) - np.log(upper_gamma_ext(-omega, c / tau))
            + np.log(rho) + rho * np.log(
                d * np.exp(gamma * (Pij['source_magnitude'] - mc_min))
            )
        ) - (
            (1 + rho) * np.log(
                Pij['spatial_distance_squared'] + (
                    d * np.exp(gamma * (Pij['source_magnitude'] - mc_min))
                )
            )
        )
        - (1 + omega) * np.log(Pij['time_distance'] + c)
        - (Pij['time_distance'] + c) / tau
        - np.log(np.pi)

    )
    distribution_term = Pij['Pij'].mul(Pij['likelihood_term']).sum()

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
    area_factor = np.pi * np.power(
        d * np.exp(gamma * (event_magnitude - mc)),
        -1 * rho
    ) / rho

    time_factor = np.exp(c / tau) * np.power(tau, -omega)  # * gamma_func(-omega)

    if no_start:
        time_fraction = upper_gamma_ext(-omega, c / tau)
    else:
        time_fraction = upper_gamma_ext(-omega, (event_time_to_start + c) / tau)
    if not no_end:
        time_fraction = time_fraction - upper_gamma_ext(-omega, (event_time_to_end + c) / tau)

    time_factor = time_factor * time_fraction

    return number_factor * area_factor * time_factor


def neg_log_likelihood_free_prod(theta, n_hat, Pij, source_events, timewindow_length, timewindow_start, area, beta, mc_min):

    assert Pij.index.names == ("source_id", "target_id"), "Pij must have multiindex with names 'source_id', 'target_id'"
    assert source_events.index.name == "source_id", "source_events must have index with name 'source_id'"

    log10_c, omega, log10_tau, log10_d, gamma, rho = theta
    c = np.power(10, log10_c)
    tau = np.power(10, log10_tau)
    d = np.power(10, log10_d)

    source_events["G"] = expected_aftershocks_free_prod(
        [
            source_events["source_magnitude"],
            source_events["source_kappa"],
            source_events["pos_source_to_start_time_distance"],
            source_events["source_to_end_time_distance"]
        ],
        [theta, mc_min]
    )

    # space time distribution term
    Pij["likelihood_term"] = (
            (omega * np.log(tau) - np.log(upper_gamma_ext(-omega, c / tau))
             + np.log(rho) + rho * np.log(
                        d * np.exp(gamma * (Pij["source_magnitude"] - mc_min))
                    ))
            - ((1 + rho) * np.log(
        Pij["spatial_distance_squared"] + (
                d * np.exp(gamma * (Pij["source_magnitude"] - mc_min))
        )
    ))
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
        prod_neg_log_lik, x0=1.5,
        args=[kappa, m_diff, weights],
        bounds=[(0, 5)]
    )
    a = res.x[0]
    log10_k0 = np.log10(
        np.sum(kappa * weights) / (np.exp(a*m_diff) * weights).sum()
    )
    return a, log10_k0


def read_shape_coords(shape_coords):
    if shape_coords is None:
        return None
    if isinstance(shape_coords, str):
        if shape_coords[-4:] == '.npy':
            # input is the path to a -npy file containing the coordinates
            coordinates = np.load(shape_coords)
        else:
            from numpy import array  # noqa
            coordinates = np.array(eval(shape_coords))
    else:
        coordinates = np.array(shape_coords)
    return coordinates


def calc_diff_to_before(a, b):
    assert len(a) == len(b), "a and b must have the same length."

    return np.sum(np.abs(
        [a[i] - b[i] for i in range(len(a)) if (a[i] is not None and b[i] is not None)]))


class ETASParameterCalculation:
    def __init__(self, metadata: dict):
        '''
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
            - m_ref: Reference magnitude when mc is variable. Not required
                    unless mc == 'var'.
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
            - theta_0: optional, initial guess for parameters. Does not affect
                    final parameters, but with a good initial guess
                    the algorithm converges faster.
            - free_background: optional, allow free_background during inversion (flETAS)
                default: False
            - free_productivity: optional, allow free_productivity during inversion (flETAS)
                default: False
            - bw_sq: optional, squared bandwidth of Gaussian kernel used for free_background/free_productivity mode
                default: 2
            - name: optional, give the model a name
            - id: optional, give the model an ID
        '''

        self.logger = logging.getLogger(__name__)
        self.name = metadata.get('name', 'NoName ETAS Model')
        self.id = metadata.get('id', uuid.uuid4())
        self.logger.info('INITIALIZING...')
        self.logger.info('  model is named {}, has ID {}'.format(self.name, self.id))
        self.shape_coords = read_shape_coords(
            metadata.get('shape_coords', None))
        self.fn_catalog = metadata.get('fn_catalog', None)
        self.catalog = metadata.get('catalog', None)

        self.delta_m = metadata['delta_m']
        self.mc = metadata['mc']
        self.m_ref = metadata['m_ref'] if self.mc == 'var' else self.mc
        self.coppersmith_multiplier = metadata['coppersmith_multiplier']
        self.earth_radius = metadata.get('earth_radius', 6.3781e3)
        self.bw_sq = metadata.get('bw_sq', 1)

        self.auxiliary_start = pd.to_datetime(metadata['auxiliary_start'])
        self.timewindow_start = pd.to_datetime(metadata['timewindow_start'])
        self.timewindow_end = pd.to_datetime(metadata['timewindow_end'])
        self.timewindow_length = to_days(self.timewindow_end - self.timewindow_start)

        self.free_background = metadata.get('free_background', False)
        self.free_productivity = metadata.get('free_productivity', False)

        self.logger.info('  Time Window: \n      {} (aux start)\n      {} '
                         '(start)\n      {} (end).'
                         .format(self.auxiliary_start,
                                 self.timewindow_start,
                                 self.timewindow_end))

        self.logger.info('  free_productivity: {}, free_background: {}'
                         .format(self.free_productivity,
                                 self.free_background))

        self.preparation_done = metadata.get('preparation_done', False)
        self.inversion_done = metadata.get('inversion_done', False)

        if not isinstance(self.catalog, pd.DataFrame):
            self.catalog = pd.read_csv(
                self.fn_catalog,
                index_col=0,
                parse_dates=['time'],
                dtype={'url': str, 'alert': str})

        self.distances = None
        self.source_events = None
        self.target_events = None

        self.area = metadata.get('area')
        self.beta = metadata.get('beta')
        self.__theta_0 = None
        self.theta_0 = metadata.get('theta_0')
        self.__theta = None
        self.theta = metadata.get('final_parameters')
        self.pij = None
        self.n_hat = None
        self.i = metadata.get('n_iterations')

        if self.inversion_done:
            fn_ip = metadata.get('fn_ip')
            fn_src = metadata.get('fn_src')
            self.sources = pd.read_csv(fn_src, index_col=0)
            self.targets = pd.read_csv(fn_ip, index_col=0, parse_dates=['time'])

    def prepare(self):
        self.logger.info('PREPARING {}'.format(self.name))
        self.logger.info('  filtering catalog...')
        self.catalog = self.filter_catalog(self.catalog)

        if self.__theta_0 is not None:
            self.logger.info('  using input initial values for theta')
        else:
            self.logger.info('  randomly chosing initial values for theta')
            self.__theta_0 = create_initial_values()

        self.logger.info('  calculating distances...')
        self.distances = self.calculate_distances()

        self.logger.info('  preparing source and target events..')
        self.target_events = self.prepare_target_events()
        self.source_events = self.prepare_source_events()

        self.beta = estimate_beta_tinti(
            self.target_events['magnitude'] - self.target_events['mc_current'],
            mc=0,
            delta_m=self.delta_m
        )
        self.logger.info('  beta of primary catalog is {}'.format(self.beta))

        if self.free_productivity:
            self.source_events["source_kappa"] = np.exp(
                self.theta_0['a'] * (self.source_events["source_magnitude"] - self.m_ref - self.delta_m / 2)
            )
        if self.free_background:
            self.target_events["P_background"] = 0.1

        self.preparation_done = True

    @property
    def theta_0(self):
        ''' getter '''
        return parameter_array2dict(self.__theta_0) \
            if self.__theta_0 is not None else None

    @theta_0.setter
    def theta_0(self, t):
        self.__theta_0 = parameter_dict2array(t) if t is not None else None

    @property
    def theta(self):
        ''' getter '''
        return parameter_array2dict(self.__theta)\
            if self.__theta is not None else None

    @theta.setter
    def theta(self, t):
        self.__theta = parameter_dict2array(t) if t is not None else None

    def invert(self):
        '''
        Invert the ETAS (or flETAS) parameters.
        '''
        self.logger.info('START INVERSION')
        diff_to_before = 100
        i = 0
        theta_old = self.__theta_0[:]

        while diff_to_before >= 0.001:
            self.logger.debug('  iteration {}'.format(i))

            self.logger.debug('    expectation step')
            self.pij, self.target_events, self.source_events, self.n_hat = \
                self.expectation_step(theta_old, self.m_ref - self.delta_m / 2)

            self.logger.debug('      n_hat: {}'.format(self.n_hat))

            self.logger.debug('    optimizing parameters')
            self.__theta = self.optimize_parameters(theta_old)
            if self.free_productivity:
                self.calc_a_k0_from_kappa()

            self.logger.debug('    new parameters:')
            self.logger.debug(
                pprint.pformat(
                    parameter_array2dict(
                        self.__theta),
                    indent=4))

            diff_to_before = calc_diff_to_before(theta_old, self.__theta)
            self.logger.debug(
                '    difference to previous: {}'.format(diff_to_before))

            if not self.free_productivity:
                br = branching_ratio(theta_old, self.beta)
                self.logger.debug('    branching ratio: {}'.format(br))
            theta_old = self.__theta[:]
            if self.free_productivity:
                self.logger.debug(
                    '    updating source kappa')
                self.update_source_kappa()
            i += 1

        self.logger.info('  stopping here. converged after '
                         '{} iterations.'.format(i))
        self.i = i

        self.logger.info('    last expectation step')
        self.pij, self.target_events, self.source_events, self.n_hat = \
            self.expectation_step(theta_old, self.m_ref - self.delta_m / 2)
        self.logger.info('    n_hat: {}'.format(self.n_hat))

        self.inversion_done = True

        return self.theta

    def filter_catalog(self, catalog):
        len_full_catalog = catalog.shape[0]
        # filter for events in region of interest
        if self.shape_coords is not None:
            self.shape_coords = read_shape_coords(self.shape_coords)

            self.logger.info(
                '  Coordinates of region: {}'.format(list(self.shape_coords)))

            poly = Polygon(self.shape_coords)
            self.area = polygon_surface(poly)
            gdf = gpd.GeoDataFrame(
                catalog, geometry=gpd.points_from_xy(
                    catalog.latitude, catalog.longitude))
            filtered_catalog = gdf[gdf.intersects(poly)].copy()
            filtered_catalog.drop('geometry', axis=1, inplace=True)
        else:
            filtered_catalog = catalog.copy()
            self.area = 6.3781e3 ** 2 * 4 * np.pi
        self.logger.info('Region has {} square km'.format(self.area))
        self.logger.info('{} out of {} events lie within target region.'
                         .format(len(self.catalog), len_full_catalog))

        # filter for events above cutoff magnitude - delta_m/2
        if self.delta_m > 0:
            filtered_catalog['magnitude'] = round_half_up(
                filtered_catalog['magnitude'] / self.delta_m) * self.delta_m
        if self.mc == 'var':
            assert 'mc_current' in filtered_catalog.columns, \
                self.logger.error(
                    'Need column "mc_current" in '
                    'catalog when mc is set to "var".')
        else:
            filtered_catalog['mc_current'] = self.mc
        filtered_catalog.query('magnitude >= mc_current', inplace=True)

        # filter for events in relevant timewindow
        filtered_catalog.query(
            'time >= @ self.auxiliary_start and time < @ self.timewindow_end',
            inplace=True)
        self.logger.info(
            '  {} events are within time window.'.format(
                filtered_catalog.shape[0]))
        return filtered_catalog

    def prepare_target_events(self):
        target_events = self.catalog.query(
            'magnitude >= mc_current').copy()
        target_events.query('time > @ self.timewindow_start', inplace=True)
        target_events['mc_current_above_ref'] = target_events['mc_current'] \
            - self.m_ref
        target_events.index.name = 'target_id'
        return target_events

    def prepare_source_events(self):
        source_columns = [
            'source_magnitude',
            'source_completeness_above_ref',
            'source_to_end_time_distance',
            'pos_source_to_start_time_distance'
        ]

        return pd.DataFrame(self.distances[source_columns]
                            .groupby('source_id').first())

    def optimize_parameters(self, theta_0, ranges=RANGES):
        start_calc = dt.datetime.now()

        # estimate mu independently and remove from parameters
        mu_hat = self.n_hat / \
            (self.area * self.timewindow_length)

        if self.free_productivity:
            # select values from theta needed in free prod mode
            theta_0_without_mu = theta_0[3:]
            bounds = ranges[3:]

            res = minimize(
                neg_log_likelihood_free_prod,
                x0=theta_0_without_mu,
                bounds=bounds,
                args=(self.n_hat, self.pij, self.source_events, self.timewindow_length, self.timewindow_start, self.area, self.beta, self.m_ref - self.delta_m / 2),
                tol=1e-12,
            )

            new_theta_without_mu = res.x
            new_theta = [np.log10(mu_hat), None, None, *new_theta_without_mu]
        else:
            theta_0_without_mu = theta_0[1:]
            bounds = ranges[1:]

            res = minimize(
                neg_log_likelihood,
                x0=theta_0_without_mu,
                bounds=bounds,
                args=(self.pij, self.source_events, self.m_ref - self.delta_m / 2),
                tol=1e-12,
            )

            new_theta_without_mu = res.x
            new_theta = [np.log10(mu_hat), *new_theta_without_mu]

        self.logger.debug(
            '    optimization step took {}'.format(dt.datetime.now()
                                                   - start_calc))

        return np.array(new_theta)

    def store_results(self, data_path='', store_pij=False):

        if data_path == '':
            data_path = os.getcwd() + '/'

        self.logger.info('  Data will be stored in {}'.format(data_path))

        fn_parameters = data_path + 'parameters_{}.json'.format(self.id)
        fn_ip = data_path + 'trig_and_bg_probs_{}.csv'.format(self.id)
        fn_src = data_path + 'sources_{}.csv'.format(self.id)
        fn_dist = data_path + 'distances_{}.csv'.format(self.id)
        fn_pij = data_path + 'pij_{}.csv'.format(self.id)

        os.makedirs(os.path.dirname(fn_ip), exist_ok=True)
        os.makedirs(os.path.dirname(fn_src), exist_ok=True)
        self.target_events.to_csv(fn_ip)
        self.source_events.to_csv(fn_src)

        all_info = {
            'name': self.name,
            'id': str(self.id),
            'fn_catalog': self.fn_catalog,
            'auxiliary_start': str(self.auxiliary_start),
            'timewindow_start': str(self.timewindow_start),
            'timewindow_end': str(self.timewindow_end),
            'timewindow_length': self.timewindow_length,
            'shape_coords': str(list(self.shape_coords)),
            'delta_m': self.delta_m,
            'mc': self.mc,
            'm_ref': self.m_ref,
            'coppersmith_multiplier': self.coppersmith_multiplier,
            'earth_radius': self.earth_radius,
            'bq_sq': self.bw_sq,
            'free_productivity': self.free_productivity,
            'free_background': self.free_background,
            'preparation_done': self.preparation_done,
            'inversion_done': self.inversion_done,
            'n_target_events': len(self.target_events),
            'area': self.area,
            'log10_mu_range': RANGES[0],
            'log10_k0_range': RANGES[1],
            'a_range': RANGES[2],
            'log10_c_range': RANGES[3],
            'omega_range': RANGES[4],
            'log10_tau_range': RANGES[5],
            'log10_d_range': RANGES[6],
            'gamma_range': RANGES[7],
            'rho_range': RANGES[8],
            'beta': self.beta,
            'n_hat': self.n_hat,
            'calculation_date': str(dt.datetime.now()),
            'initial_values': self.theta_0,
            'final_parameters': self.theta,
            'n_iterations': self.i,
            'fn_dist': fn_dist,
            'fn_ip': fn_ip,
            'fn_src': fn_src,
        }
        with open(fn_parameters, 'w') as f:
            f.write(json.dumps(all_info))

        if store_pij:
            os.makedirs(os.path.dirname(fn_pij), exist_ok=True)
            self.pij.to_csv(fn_pij)

    def calculate_distances(self):
        '''
        Precalculates distances in time and space between events that are
        potentially related to each other.
        '''

        calc_start = dt.datetime.now()

        # only use data above completeness magnitude
        if self.delta_m > 0:
            self.catalog['magnitude'] = round_half_up(
                self.catalog['magnitude'] / self.delta_m) * self.delta_m
        relevant = self.catalog.query('magnitude >= mc_current').copy()
        relevant.sort_values(by='time', inplace=True)

        # all entries can be sources, but targets only after timewindow start
        targets = relevant.query('time>=@self.timewindow_start').copy()

        beta = estimate_beta_tinti(
            targets['magnitude']
            - targets['mc_current'],
            mc=0,
            delta_m=self.delta_m)
        logger.info('    beta is {}'.format(beta))

        # calculate some source stuff
        relevant['distance_range_squared'] = np.square(coppersmith(
            relevant['magnitude'], 4)['SSRL'] * self.coppersmith_multiplier)
        relevant['source_to_end_time_distance'] = to_days(
            self.timewindow_end - relevant['time'])
        relevant['pos_source_to_start_time_distance'] = np.clip(
            to_days(self.timewindow_start - relevant['time']),
            a_min=0,
            a_max=None
        )

        # translate target lat, lon to radians for spherical distance
        # calculation
        targets['target_lat_rad'] = np.radians(targets['latitude'])
        targets['target_lon_rad'] = np.radians(targets['longitude'])
        targets['target_time'] = targets['time']
        targets['target_id'] = targets.index
        targets['target_time'] = targets['time']
        targets['target_completeness_above_ref'] = targets['mc_current']
        # columns that are needed later
        targets['source_id'] = 'i'
        targets['source_magnitude'] = 0.0
        targets['source_completeness_above_ref'] = 0.0
        targets['time_distance'] = 0.0
        targets['spatial_distance_squared'] = 0.0
        targets['source_to_end_time_distance'] = 0.0
        targets['pos_source_to_start_time_distance'] = 0.0

        targets = targets.sort_values(by='time')

        # define index and columns that are later going to be needed
        if pd.__version__ >= '0.24.0':
            index = pd.MultiIndex(
                levels=[[], []],
                names=['source_id', 'target_id'],
                codes=[[], []]
            )
        else:
            index = pd.MultiIndex(
                levels=[[], []],
                names=['source_id', 'target_id'],
                labels=[[], []]
            )

        columns = [
            'target_time',
            'source_magnitude',
            'source_completeness_above_ref',
            'target_completeness_above_ref',
            'spatial_distance_squared',
            'time_distance',
            'source_to_end_time_distance',
            'pos_source_to_start_time_distance'
        ]
        res_df = pd.DataFrame(index=index, columns=columns)

        df_list = []

        logger.info('  number of sources: {}'.format(len(relevant.index)))
        logger.info('  number of targets: {}'.format(len(targets.index)))
        for source in relevant.itertuples():
            stime = source.time

            # filter potential targets
            if source.time < self.timewindow_start:
                potential_targets = targets.copy()
            else:
                potential_targets = targets.query('time>@stime').copy()
            targets = potential_targets.copy()

            if potential_targets.shape[0] == 0:
                continue

            # get values of source event
            slatrad = np.radians(source.latitude)
            slonrad = np.radians(source.longitude)
            drs = source.distance_range_squared  # noqa

            # get source id and info of target events
            potential_targets['source_id'] = source.Index
            potential_targets['source_magnitude'] = source.magnitude
            potential_targets['source_completeness_above_ref'] = \
                source.mc_current

            # calculate space and time distance from source to target event
            potential_targets['time_distance'] = to_days(
                potential_targets['target_time'] - stime)

            potential_targets['spatial_distance_squared'] = np.square(
                haversine(
                    slatrad,
                    potential_targets['target_lat_rad'],
                    slonrad,
                    potential_targets['target_lon_rad'],
                    self.earth_radius
                )
            )

            # filter for only small enough distances
            potential_targets.query('spatial_distance_squared <= @drs',
                                    inplace=True)

            # calculate time distance from source event to timewindow
            # boundaries for integration later
            potential_targets['source_to_end_time_distance'] = \
                source.source_to_end_time_distance
            potential_targets['pos_source_to_start_time_distance'] = \
                source.pos_source_to_start_time_distance

            # append to resulting dataframe
            df_list.append(potential_targets)

        res_df = pd.concat(df_list)[['source_id', 'target_id'] + columns]\
            .reset_index().set_index(['source_id', 'target_id'])
        res_df['source_completeness_above_ref'] = \
            res_df['source_completeness_above_ref'] - self.m_ref
        res_df['target_completeness_above_ref'] = \
            res_df['target_completeness_above_ref'] - self.m_ref

        logger.debug(
            '  took {} to prepare the data'.format(
                dt.datetime.now()
                - calc_start))

        return res_df

    def expectation_step(self, theta, mc_min):

        calc_start = dt.datetime.now()
        log10_mu = theta[0]
        mu = np.power(10, log10_mu)

        # calculate the triggering density values gij
        logger.debug('    calculating gij')
        Pij_0 = self.distances.copy()
        source_kappa = pd.merge(
            Pij_0[[]], self.source_events["source_kappa"], left_index=True, right_index=True).copy().fillna(0) \
            if self.free_productivity else {'source_kappa': None}
        Pij_0['gij'] = triggering_kernel(
            [
                Pij_0['time_distance'],
                Pij_0['spatial_distance_squared'],
                Pij_0['source_magnitude'],
                source_kappa['source_kappa']
            ],
            [theta, mc_min]
        )

        # responsibility factor for invisible triggering events
        Pij_0['xi_plus_1'] = responsibility_factor(
            theta, self.beta, Pij_0['source_completeness_above_ref'])
        Pij_0['zeta_plus_1'] = observation_factor(
            self.beta, Pij_0['target_completeness_above_ref'])
        # calculate muj for each target. currently constant, could be improved
        target_events_0 = self.target_events.copy()
        target_events_0['mu'] = mu
        if self.free_background:
            target_events_0["mu"] = (
                    (((np.exp(-1 / 2 * Pij_0["spatial_distance_squared"] / self.bw_sq) / (self.bw_sq * 2 * np.pi)).mul(
                        target_events_0["P_background"], level=0)).groupby(level=1).sum() + target_events_0["P_background"] / (
                             self.bw_sq * 2 * np.pi)) / (
                        self.timewindow_length  # TODO: divide by tw_length minus target_to_end_time_distance
                    )
            ).fillna(0)
        else:
            target_events_0["mu"] = mu

        # calculate triggering probabilities Pij
        logger.debug('    calculating Pij')
        Pij_0['tot_rates'] = 0
        Pij_0['tot_rates'] = Pij_0['tot_rates'].add(
            (Pij_0['gij']
             * Pij_0['xi_plus_1']).groupby(
                level=1).sum()).add(
            target_events_0['mu'])
        Pij_0['Pij'] = Pij_0['gij'].div(Pij_0['tot_rates'])

        # calculate probabilities of being triggered or background
        target_events_0['P_triggered'] = 0
        target_events_0['P_triggered'] = target_events_0['P_triggered'].add(
            Pij_0['Pij'].groupby(level=1).sum()).fillna(0)
        target_events_0['P_background'] = target_events_0['mu'] / \
            Pij_0.groupby(level=1).first()['tot_rates']
        target_events_0['zeta_plus_1'] = observation_factor(
            self.beta, target_events_0['mc_current_above_ref'])

        # calculate expected number of background events
        logger.debug('    calculating n_hat and l_hat')
        n_hat_0 = target_events_0['P_background'].sum()

        # calculate aftershocks per source event
        source_events_0 = self.source_events.copy()
        source_events_0['l_hat'] = (Pij_0['Pij'] * Pij_0['zeta_plus_1']) \
            .groupby(level=0).sum()

        logger.debug('    expectation step took {}'.format(
            dt.datetime.now() - calc_start))
        return Pij_0, target_events_0, source_events_0, n_hat_0

    def update_source_kappa(self):
        self.source_events["G"] = expected_aftershocks_free_prod(
            [
                self.source_events["source_magnitude"],
                self.source_events["source_kappa"],
                self.source_events["pos_source_to_start_time_distance"],
                self.source_events["source_to_end_time_distance"]
            ],
            [self.__theta[3:], self.m_ref - self.delta_m / 2]
        )
        # filling nan with 0 because I believe the only way this can be undefined is when G is zero, which only
        # happens when source_kappa is zero. so should be fine.
        self.source_events["source_kappa"] = (self.source_events["source_kappa"] * self.source_events["l_hat"] / self.source_events["G"]).fillna(0)

    def calc_a_k0_from_kappa(self):
        prim_mags = self.catalog.query("time >=@self.timewindow_start")["magnitude"]
        kappas_estimated = pd.merge(
            prim_mags,
            self.source_events[["source_kappa"]],
            left_index=True,
            right_index=True,
            how='left'
        ).fillna(0)
        self.__theta[2], self.__theta[1] = calc_a_k0_from_kappa(
            kappas_estimated["source_kappa"],
            kappas_estimated["magnitude"] - (self.m_ref - self.delta_m/2)
        )
