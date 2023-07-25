#!/usr/bin/env python
# coding: utf-8

##############################################################################
# simulation of earthquake catalogs using ETAS
#
# as described by Mizrahi et al., 2021
# Leila Mizrahi, Shyam Nandan, Stefan Wiemer;
# The Effect of Declustering on the Size Distribution of Mainshocks.
# Seismological Research Letters 2021; doi: https://doi.org/10.1785/0220200231
##############################################################################

import datetime as dt
import logging
import os
import pprint

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.special import gamma as gamma_func
from scipy.special import gammainccinv
from shapely.geometry import Polygon

from etas.inversion import (ETASParameterCalculation, branching_ratio,
                            expected_aftershocks, haversine,
                            parameter_dict2array, round_half_up, to_days,
                            upper_gamma_ext, branching_integral)
from etas.mc_b_est import simulate_magnitudes, simulate_magnitudes_from_zone

logger = logging.getLogger(__name__)


def inverse_upper_gamma_ext(a, y):
    # TODO: find a more elegant way to do this
    if a > 0:
        return gammainccinv(a, y / gamma_func(a))
    else:
        import warnings

        from pynverse import inversefunc
        from scipy.optimize import minimize

        uge = (lambda x: upper_gamma_ext(a, x))

        # numerical inverse
        def num_inv(a, y):
            def diff(x, xhat):
                xt = upper_gamma_ext(a, x)
                return (xt - xhat) ** 2

            x = np.zeros(len(y))
            for idx, y_value in enumerate(y):
                res = minimize(diff,
                               1.0,
                               args=(y_value),
                               method='Nelder-Mead',
                               tol=1e-6)
                x[idx] = res.x[0]

            return x

        warnings.filterwarnings("ignore")
        result = inversefunc(uge, y)
        warnings.filterwarnings("default")

        # where inversefunc was unable to calculate a result, calculate
        # numerical approximation
        nan_idxs = np.argwhere(np.isnan(result)).flatten()
        if len(nan_idxs) > 0:
            num_res = num_inv(a, y[nan_idxs])
            result[nan_idxs] = num_res

        return result


def transform_parameters(par, beta, delta_m, dm_max_orig=None):
    """
    Transform the ETAS parameters to a different reference magnitude.

    Args:
        par (dict): A dictionary containing original parameter values.
        beta (float): The beta value used in the transformation.
        delta_m (float): The difference in reference magnitude
            (m_ref_new - m_ref_old)
        dm_max_orig (float): Difference between max magnitude and
            m_ref in original parameters. only required if alpha-beta >= 0

    Returns:
        dict: A dictionary with the transformed parameter values.

    """
    par_corrected = par.copy()
    if delta_m == 0:
        return par_corrected
    if dm_max_orig is None:
        alpha_minus_beta = par["a"] - par["rho"] * par["gamma"] - beta
        assert alpha_minus_beta < 0, "for unlimited magnitudes, " \
                                     "alpha-beta must be negative."
        branching_integral_orig = branching_integral(
            alpha_minus_beta,
            dm_max_orig
        )
        branching_integral_new = branching_integral(
            alpha_minus_beta,
            (dm_max_orig + delta_m if dm_max_orig is not None else None)
        )
        branching_integral_ratio = branching_integral_new \
                                   / branching_integral_orig
    else:
        branching_integral_ratio = 1

    par_corrected["log10_mu"] -= delta_m * beta / np.log(10)
    par_corrected["log10_d"] += delta_m * par_corrected["gamma"] / np.log(10)
    par_corrected["log10_k0"] += delta_m * par_corrected["gamma"] * \
         par_corrected["rho"] / np.log(10) - \
         np.log10(branching_integral_ratio)

    return par_corrected


def parameters_from_standard_formulation(
        par_st, par_here, delta_m_ref=0, beta=np.log(10), dm_max_st=None):
    """
    Convert parameters of standard ETAS formulation (without spatial kernel)
    to parameters used here.

    Args:
        par_st (dict): A dictionary containing the parameters
            in standard formulation.
        par_here (dict): A dictionary containing spatial parameters
            in the formulation used here (rho, gamma, log10_d).
        delta_m_ref (float, optional) : reference magnitude difference.
            target m_ref minus m_ref of standard formulation.
        dm_max_st (float, optional): The possible magnitude range
            (above m_ref of the standard formulation).
            Used in case alpha-beta is non-negative when ensuring
            branching ratio stays the same when transforming
            to other reference magnitudes.

    Returns:
        dict: A dictionary with the transformed parameters.

    """
    here = par_here.copy()
    # first transform the "here" parameters to m_ref of standard formulation
    result = transform_parameters(
        here, beta=beta, delta_m=-delta_m_ref,
        dm_max_orig=(dm_max_st - delta_m_ref if dm_max_st is not None else None))

    # define parameters based on standard formulation
    result["log10_c"] = par_st["log10_c"]
    result["log10_k0"] = par_st["a"] \
                         - np.log10(np.pi / par_here["rho"]) \
                         + (par_here["rho"] * par_here["log10_d"]) \
                         + par_st["alpha"] * delta_m_ref
    result["omega"] = par_st["p"] - 1
    result["log10_tau"] = 12.26 if result["omega"] <= 0 else np.inf
    result["a"] = par_st["alpha"] * np.log(10) + par_here["rho"] * par_here[
        "gamma"]

    # transform back to reference magnitude of interest
    result = transform_parameters(
        result, beta=beta, delta_m=delta_m_ref,
        dm_max_orig=dm_max_st
    )
    return result


def parameters_from_etes_formulation(etes_par, par, delta_m_ref = 0):
    """
    Convert parameters of standard ETAS formulation (without spatial kernel)
    to parameters used here.

    Args:
        etes_par (dict): A dictionary containing the parameters
            in ETES formulation.
        par (dict): A dictionary containing parameters
            in the formulation used here.
        delta_m_ref (float, optional) : reference magnitude difference.
            target m_ref minus m_ref of standard formulation.

    Returns:
        dict: A dictionary with the transformed parameters.

    """
    result = par.copy()
    result["log10_c"] = etes_par["log10_c"]
    result["log10_k0"] = etes_par["a"] \
                         - np.log10(np.pi / par["rho"]) \
                         + (par["rho"] * par["log10_d"]) \
                         + etes_par["alpha"] * delta_m_ref
    result["omega"] = etes_par["p"] - 1
    result["log10_tau"] = 12.26 if result["omega"] <= 0 else np.inf
    result["a"] = etes_par["alpha"] * np.log(10) + par["rho"] * par[
        "gamma"]
    return result


def simulate_aftershock_time(log10_c, omega, log10_tau, size=1):
    # time delay in days

    # this function makes sense. I have panicked and checked several times.
    # if you plot a histogram of simulated times with logarithmic bins,
    # make sure to account for bin width!

    c = np.power(10, log10_c)
    tau = np.power(10, log10_tau)
    y = np.random.uniform(size=size)

    return inverse_upper_gamma_ext(
        -omega,
        (1 - y) * upper_gamma_ext(-omega, c / tau)) * tau - c


def simulate_aftershock_time_untapered(log10_c, omega, size=1):
    # time delay in days

    # TODO: find a way to sample y values with higher precision that 1e-15
    # otherwise there is a maximum time delay that will be sampled...

    c = np.power(10, log10_c)
    y = np.random.uniform(size=size)

    return np.power((1 - y), -1/omega) * c - c


def inv_time_cdf_approx(p, c, tau, omega):
    part_a = -1 / omega * (np.power(tau + c, -omega) - np.power(c, -omega))
    part_b = np.exp(1) * np.power(tau + c, -(1 + omega)) * (tau / np.exp(1))
    k1 = 1 / (part_a + part_b)
    k2 = k1 * np.exp(1) / np.power(tau + c, 1 + omega)

    res_a = np.power(np.power(c, -omega) - omega * p / k1, -1 / omega) - c
    res_b = np.log(
        (p - (part_a / (part_a + part_b))) * (-1) / (tau * k2) + np.exp(
            -1)) * (-tau)

    return np.where(p < tau, res_a, res_b)


def simulate_aftershock_time_approx(log10_c, omega, log10_tau, size=1):
    # time delay in days
    c = np.power(10, log10_c)
    tau = np.power(10, log10_tau)
    y = np.random.uniform(size=size)

    return inv_time_cdf_approx(y, c, tau, omega)


def simulate_aftershock_place(log10_d, gamma, rho, mi, mc):
    # x and y offset in km
    d = np.power(10, log10_d)
    d_g = d * np.exp(gamma * (mi - mc))
    y_r = np.random.uniform(size=len(mi))
    r = np.sqrt(np.power(1 - y_r, -1 / rho) * d_g - d_g)
    phi = np.random.uniform(0, 2 * np.pi, size=len(mi))

    x = r * np.sin(phi)
    y = r * np.cos(phi)

    return x, y


def simulate_aftershock_radius(log10_d, gamma, rho, mi, mc):
    # x and y offset in km
    d = np.power(10, log10_d)
    d_g = d * np.exp(gamma * (mi - mc))
    y_r = np.random.uniform(size=len(mi))
    r = np.sqrt(np.power(1 - y_r, -1 / rho) * d_g - d_g)

    return r


def simulate_background_location(
        latitudes, longitudes, background_probs, scale=0.1,
        grid=False, bsla=None, bslo=None,
        n=1
):
    np.random.seed()
    assert np.max(background_probs) <= 1, "background_probs cannot exceed 1"

    keep_idxs = []
    while sum(keep_idxs) == 0:
        keep_idxs = background_probs >= np.random.uniform(
            size=len(background_probs))

    sample_lats = latitudes[keep_idxs]
    sample_lons = longitudes[keep_idxs]

    choices = np.floor(np.random.uniform(0, len(sample_lats), size=n)).astype(
        int)

    if grid:
        lats = sample_lats.iloc[choices] + np.random.uniform(0, bsla,
                                                             size=n) - bsla / 2
        lons = sample_lons.iloc[choices] + np.random.uniform(0, bslo,
                                                             size=n) - bslo / 2
    else:
        lats = sample_lats.iloc[choices] + np.random.normal(loc=0, scale=scale,
                                                            size=n)
        lons = sample_lons.iloc[choices] + np.random.normal(loc=0, scale=scale,
                                                            size=n)

    return lats, lons


def generate_background_events(polygon, timewindow_start, timewindow_end,
                               parameters, beta, mc, delta_m=0, m_max=None,
                               background_lats=None, background_lons=None,
                               background_probs=None, gaussian_scale=None,
                               bsla=None, bslo=None, grid=False,
                               mfd_zones=None, zones_from_latlon=None
                               ):
    from etas.inversion import polygon_surface, to_days

    theta = parameter_dict2array(parameters)
    theta_without_mu = theta[2:]

    area = polygon_surface(polygon)
    timewindow_length = to_days(timewindow_end - timewindow_start)

    # area of surrounding rectangle
    min_lat, min_lon, max_lat, max_lon = polygon.bounds
    coords = [[min_lat, min_lon],
              [max_lat, min_lon],
              [max_lat, max_lon],
              [min_lat, max_lon]]
    rectangle = Polygon(coords)
    rectangle_area = polygon_surface(rectangle)

    # number of background events
    expected_n_background = np.power(
        10, parameters["log10_mu"]) * area * timewindow_length
    n_background = np.random.poisson(lam=expected_n_background)

    # generate too many events, afterwards filter those that are in the polygon
    n_generate = int(np.round(n_background * rectangle_area / area * 1.2))

    logger.info(f"  number of background events needed: {n_background}")
    if n_background == 0:
        return pd.DataFrame()
    logger.info(
        f"  generating {n_generate} to throw away those outside the polygon")

    # define dataframe with background events
    catalog = pd.DataFrame(
        None,
        columns=[
            "latitude",
            "longitude",
            "time",
            "magnitude",
            "parent",
            "generation"])

    # generate lat, long
    if background_probs is not None:
        catalog["latitude"], catalog["longitude"] = \
            simulate_background_location(background_lats,
                                         background_lons,
                                         background_probs=background_probs,
                                         scale=gaussian_scale,
                                         bsla=bsla, bslo=bslo, grid=grid,
                                         n=n_generate
                                         )
    else:
        catalog["latitude"] = np.random.uniform(
            min_lat, max_lat, size=n_generate)
        catalog["longitude"] = np.random.uniform(
            min_lon, max_lon, size=n_generate)

    catalog = gpd.GeoDataFrame(
        catalog, geometry=gpd.points_from_xy(
            catalog.latitude, catalog.longitude))
    catalog = catalog[catalog.intersects(polygon)].head(n_background)

    # if not enough events fell into the polygon, do it again...
    while len(catalog) != n_background:
        logger.info("  didn't create enough events. trying again..")

        # define dataframe with background events
        catalog = pd.DataFrame(
            None,
            columns=[
                "latitude",
                "longitude",
                "time",
                "magnitude",
                "parent",
                "generation"])

        # generate lat, long
        catalog["latitude"] = np.random.uniform(
            min_lat, max_lat, size=n_generate)
        catalog["longitude"] = np.random.uniform(
            min_lon, max_lon, size=n_generate)

        catalog = gpd.GeoDataFrame(
            catalog, geometry=gpd.points_from_xy(
                catalog.latitude, catalog.longitude))
        catalog = catalog[catalog.intersects(polygon)].head(n_background)

    # generate time, magnitude
    catalog["time"] = [
        timewindow_start
        + dt.timedelta(days=d) for d in np.random.uniform(
            0,
            timewindow_length,
            size=n_background)]

    if mfd_zones is not None:
        zones = zones_from_latlon(catalog["latitude"], catalog["longitude"])
        catalog["magnitude"] = simulate_magnitudes_from_zone(zones, mfd_zones)
    else:
        catalog["magnitude"] = simulate_magnitudes(
            n_background, beta=beta, mc=mc - delta_m / 2,
            m_max=m_max + delta_m / 2 if m_max is not None else None)

    # info about origin of event
    catalog["generation"] = 0
    catalog["parent"] = 0
    catalog["is_background"] = True

    # reindexing
    catalog = catalog.sort_values(by="time").reset_index(drop=True)
    catalog.index += 1
    catalog["gen_0_parent"] = catalog.index

    # simulate number of aftershocks
    catalog["expected_n_aftershocks"] = expected_aftershocks(
        catalog["magnitude"],
        params=[theta_without_mu, mc - delta_m / 2],
        no_start=True,
        no_end=True,
        # axis=1
    )
    catalog["n_aftershocks"] = np.random.poisson(
        lam=catalog["expected_n_aftershocks"])

    return catalog.drop("geometry", axis=1)


def generate_aftershocks(sources,
                         generation,
                         parameters,
                         beta,
                         mc,
                         timewindow_end,
                         timewindow_length,
                         auxiliary_end=None,
                         delta_m=0,
                         m_max=None,
                         earth_radius=6.3781e3,
                         polygon=None,
                         approx_times=False,
                         mfd_zones=None,
                         zones_from_latlon=None):
    theta = parameter_dict2array(parameters)
    theta_without_mu = theta[2:]

    # random timedeltas for all aftershocks
    total_n_aftershocks = sources["n_aftershocks"].sum()

    if parameters["log10_tau"] == np.inf:
        all_deltas = simulate_aftershock_time_untapered(
            log10_c=parameters["log10_c"],
            omega=parameters["omega"],
            size=total_n_aftershocks
        )
    elif approx_times:
        all_deltas = simulate_aftershock_time_approx(
            log10_c=parameters["log10_c"],
            omega=parameters["omega"],
            log10_tau=parameters["log10_tau"],
            size=total_n_aftershocks
        )
    else:
        all_deltas = simulate_aftershock_time(
            log10_c=parameters["log10_c"],
            omega=parameters["omega"],
            log10_tau=parameters["log10_tau"],
            size=total_n_aftershocks
        )

    aftershocks = sources.loc[sources.index.repeat(sources.n_aftershocks)]

    keep_columns = ["time", "latitude", "longitude", "magnitude"]
    aftershocks["parent"] = aftershocks.index

    for col in keep_columns:
        aftershocks["parent_" + col] = aftershocks[col]

    # time of aftershock
    aftershocks = aftershocks[[
        col for col in aftershocks.columns if "parent" in col]] \
        .reset_index(drop=True)
    aftershocks["time_delta"] = all_deltas
    aftershocks.query("time_delta <= @ timewindow_length", inplace=True)

    aftershocks["time"] = aftershocks["parent_time"] + \
                          pd.to_timedelta(aftershocks["time_delta"], unit='d')
    aftershocks.query("time <= @ timewindow_end", inplace=True)
    if auxiliary_end is not None:
        aftershocks.query("time > @ auxiliary_end", inplace=True)

    # location of aftershock
    aftershocks["radius"] = simulate_aftershock_radius(
        parameters["log10_d"],
        parameters["gamma"],
        parameters["rho"],
        aftershocks["parent_magnitude"],
        mc=mc)
    aftershocks["angle"] = np.random.uniform(
        0, 2 * np.pi, size=len(aftershocks))
    aftershocks["degree_lon"] = haversine(
        np.radians(aftershocks["parent_latitude"]),
        np.radians(aftershocks["parent_latitude"]),
        np.radians(0),
        np.radians(1),
        earth_radius
    )
    aftershocks["degree_lat"] = haversine(
        np.radians(aftershocks["parent_latitude"] - 0.5),
        np.radians(aftershocks["parent_latitude"] + 0.5),
        np.radians(0),
        np.radians(0),
        earth_radius
    )
    aftershocks["latitude"] = aftershocks["parent_latitude"] + (
            aftershocks["radius"] * np.cos(aftershocks["angle"])
    ) / aftershocks["degree_lat"]
    aftershocks["longitude"] = aftershocks["parent_longitude"] + (
            aftershocks["radius"] * np.sin(aftershocks["angle"])
    ) / aftershocks["degree_lon"]

    as_cols = [
        "parent",
        "gen_0_parent",
        "time",
        "latitude",
        "longitude"
    ]
    if polygon is not None:
        aftershocks = gpd.GeoDataFrame(
            aftershocks, geometry=gpd.points_from_xy(
                aftershocks.latitude, aftershocks.longitude))
        aftershocks = aftershocks[aftershocks.intersects(polygon)]

    aadf = aftershocks[as_cols].reset_index(drop=True)

    # magnitudes
    n_total_aftershocks = len(aadf.index)
    if mfd_zones is not None:
        zones = zones_from_latlon(aadf["latitude"], aadf["longitude"])
        aadf["magnitude"] = simulate_magnitudes_from_zone(zones, mfd_zones)
    else:
        aadf["magnitude"] = simulate_magnitudes(
            n_total_aftershocks, beta=beta, mc=mc - delta_m / 2,
            m_max=m_max + delta_m / 2 if m_max is not None else None)

    # info about generation and being background
    aadf["generation"] = generation + 1
    aadf["is_background"] = False

    # info for next generation
    aadf["expected_n_aftershocks"] = expected_aftershocks(
        aadf["magnitude"],
        params=[theta_without_mu, mc - delta_m / 2],
        no_start=True,
        no_end=True,
    )
    aadf["n_aftershocks"] = np.random.poisson(
        lam=aadf["expected_n_aftershocks"])

    return aadf


def prepare_auxiliary_catalog(auxiliary_catalog, parameters, mc, delta_m=0):
    theta = parameter_dict2array(parameters)
    theta_without_mu = theta[2:]

    catalog = auxiliary_catalog.copy()

    catalog.loc[:, "generation"] = 0
    catalog.loc[:, "parent"] = 0
    catalog.loc[:, "is_background"] = False

    # reindexing
    catalog["evt_id"] = catalog.index.values
    catalog = catalog.sort_values(by="time").reset_index(drop=True)
    catalog.index += 1
    catalog["gen_0_parent"] = catalog.index

    # simulate number of aftershocks
    catalog["expected_n_aftershocks"] = expected_aftershocks(
        catalog["magnitude"],
        params=[theta_without_mu, mc - delta_m / 2],
        no_start=True,
        no_end=True,
        # axis=1
    )
    catalog["expected_n_aftershocks"] = catalog["expected_n_aftershocks"] \
                                        * catalog["xi_plus_1"]

    catalog["n_aftershocks"] = catalog["expected_n_aftershocks"].apply(
        np.random.poisson,
        # axis = 1
    )

    return catalog


def generate_catalog(polygon,
                     timewindow_start,
                     timewindow_end,
                     parameters,
                     mc,
                     beta_main,
                     beta_aftershock=None,
                     delta_m=0,
                     m_max=None,
                     background_lats=None,
                     background_lons=None,
                     background_probs=None,
                     gaussian_scale=None,
                     approx_times=False):
    """
    Simulates an earthquake catalog.

    Optionally use coordinates and independence probabilities
    of observed events to simulate locations of background events.

    Parameters
    ----------
    polygon : Polygon
        Coordinates of boundaries in which catalog is generated.
    timewindow_start : datetime
        Simulation start.
    timewindow_end : datetime
         Simulation end.
    parameters : dict
        As estimated in the ETAS EM inversion.
    mc : float
        Completeness magnitude. If delta_m > 0, magnitudes are
        simulated above mc-delta_m/2.
    beta_main : float
        Beta used to generate background event magnitudes.
    beta_aftershock : float, optional
        Beta used to generate aftershock magnitudes. If none,
        beta_main is used.
    delta_m : float, default 0
        Bin size of magnitudes.
    m_max : float, default None
        Maximum simulated magnitude bin (m_max + delta_m/2 can be simulated).
    background_lats : list, optional
        list of latitudes
    background_lons : list, optional
        list of longitudes
    background_probs : list, optional
        list of independence probabilities
        these three lists are assumed to be sorted
        such that corresponding entries belong to the same event
    gaussian_scale : float, optional
        sigma to be used when background locations are generated
    approx_times : bool, optional
        if True, times are simulated using an approximation,
        making it much faster.
    """

    if beta_aftershock is None:
        beta_aftershock = beta_main

    # generate background events
    logger.info("generating background events..")
    catalog = generate_background_events(
        polygon,
        timewindow_start,
        timewindow_end,
        parameters,
        beta=beta_main,
        mc=mc,
        delta_m=delta_m,
        m_max=m_max,
        background_lats=background_lats,
        background_lons=background_lons,
        background_probs=background_probs,
        gaussian_scale=gaussian_scale)

    theta = parameter_dict2array(parameters)
    br = branching_ratio(theta, beta_main)

    logger.info(f'  number of background events: {len(catalog.index)}')
    logger.info(f'\n  branching ratio: {br}')
    logger.info(f'  expected total number of events (if time were infinite):'
                f' {len(catalog) / (1 - br)}')

    generation = 0
    timewindow_length = to_days(timewindow_end - timewindow_start)

    while True:
        logger.info(f'\n\nsimulating aftershocks of generation {generation}..')
        sources = catalog.query(
            "generation == @generation and n_aftershocks > 0").copy()

        # if no aftershocks are produced by events of this generation, stop
        logger.info(
            f'  number of events with aftershocks: {len(sources.index)}')

        if len(sources.index) == 0:
            break

        # an array with all aftershocks. to be appended to the catalog
        aftershocks = generate_aftershocks(
            sources,
            generation,
            parameters,
            beta_aftershock,
            mc,
            delta_m=delta_m,
            m_max=m_max,
            timewindow_end=timewindow_end,
            timewindow_length=timewindow_length,
            approx_times=approx_times
        )

        aftershocks.index += catalog.index.max() + 1

        logger.info(
            f'  number of generated aftershocks: {len(aftershocks.index)}')

        catalog = pd.concat([
            catalog, aftershocks
        ], ignore_index=False, sort=True)

        generation = generation + 1

    logger.info(f'\n\ntotal events simulated: {len(catalog)}')
    catalog = gpd.GeoDataFrame(
        catalog, geometry=gpd.points_from_xy(
            catalog.latitude, catalog.longitude))
    catalog = catalog[catalog.intersects(polygon)]
    logger.info(f'inside the polygon: {len(catalog)}')

    return catalog.drop("geometry", axis=1)


def simulate_catalog_continuation(auxiliary_catalog,
                                  auxiliary_start,
                                  auxiliary_end,
                                  polygon,
                                  simulation_end,
                                  parameters,
                                  mc,
                                  beta_main,
                                  beta_aftershock=None,
                                  delta_m=0,
                                  m_max=None,
                                  background_lats=None,
                                  background_lons=None,
                                  background_probs=None,
                                  gaussian_scale=None,
                                  bsla=None,
                                  bslo=None,
                                  bg_grid=False,
                                  mfd_zones=None,
                                  zones_from_latlon=None,
                                  filter_polygon=True,
                                  approx_times=False,
                                  induced_lats=None,
                                  induced_lons=None,
                                  induced_term=None,
                                  induced_bsla=None,
                                  induced_bslo=None,
                                  n_induced=None,
                                  ):
    """
    auxiliary_catalog : pd.DataFrame
        Catalog used for aftershock generation in simulation period
    auxiliary_start : datetime
        Start time of auxiliary catalog.
    auxiliary_end : datetime
        End time of auxiliary_catalog. start of simulation period.
    polygon : Polygon
        Polygon in which events are generated.
    simulation_end : datetime
        End time of simulation period.
    parameters : dict
        ETAS parameters
    mc : float
        Reference mc for ETAS parameters.
    beta_main : float
        Beta for main shocks. can be a map for spatially variable betas.
    beta_aftershock : float, optional
        Beta for aftershocks. if None, is set to be same as main shock beta.
    delta_m : float, default 0
        Bin size for discrete magnitudes.
    m_max : float, default None
        Maximum simulated magnitude bin (m_max + delta_m/2 can be simulated).
    background_lats : list, optional
        Latitudes of background events.
    background_lons : list, optional
        Longitudes of background events.
    background_probs : list, optional
        Independence probabilities of background events.
    gaussian_scale : float, optional
        Extent of background location smoothing.
    bsla : float, optional
        Latitude bin size of background grid.
    bslo : float, optional
        Longitude bin size of background grid.
    bg_grid : bool, optional
        if True, background events are simulated assuming that
        background_lats, background_lons, and background_probs
        define a grid.
        if False, it is assumed that they define locations of
        past background earthquakes which will be sampled
    # TODO: add description here
    mfd_zones:
    zones_from_latlon:
    approx_times : bool, optional
        if True, times are simulated using an approximation,
        making it much faster.
    induced_lats : list, optional
        Latitudes of induced events.
    induced_lons : list, optional
        Longitudes of induced events.
    induced_term : list, optional
        Term proportiaonal to rate of induced events.
    induced_bsla : float, optional
        Latitude bin size of induced grid term.
    induced_bslo : float, optional
        Longitude bin size of induced grid term.
    n_induced : float, optional
        Expected number of induced earthquakes.
    """
    # preparing betas
    if beta_aftershock is None:
        beta_aftershock = beta_main

    background = generate_background_events(
        polygon,
        auxiliary_end,
        simulation_end,
        parameters,
        beta_main,
        mc,
        delta_m,
        m_max=m_max,
        background_lats=background_lats,
        background_lons=background_lons,
        background_probs=background_probs,
        gaussian_scale=gaussian_scale,
        bsla=bsla,
        bslo=bslo,
        grid=bg_grid,
        mfd_zones=mfd_zones,
        zones_from_latlon=zones_from_latlon,
    )
    background["evt_id"] = ''
    background["xi_plus_1"] = 1

    if induced_lats is not None:
        from etas.inversion import polygon_surface
        parameters_induced = parameters.copy()
        area = polygon_surface(polygon)
        timewindow_length = to_days(simulation_end - auxiliary_end)
        mu_induced = n_induced/(timewindow_length * area)
        parameters_induced["log10_mu"] = np.log10(mu_induced)
        induced = generate_background_events(
            polygon,
            auxiliary_end,
            simulation_end,
            parameters_induced,
            beta_main,
            mc,
            delta_m,
            m_max=m_max,
            background_lats=induced_lats,
            background_lons=induced_lons,
            background_probs=induced_term,
            bsla=induced_bsla, bslo=induced_bslo, grid=True,
        )
        induced['is_background'] = 'induced'
        induced["evt_id"] = ''
        induced["xi_plus_1"] = 1
    else:
        induced = pd.DataFrame()
    logger.debug(f'number of induced events: {len(induced.index)}')
    auxiliary_catalog = prepare_auxiliary_catalog(
        auxiliary_catalog=auxiliary_catalog, parameters=parameters, mc=mc,
        delta_m=delta_m,
    )
    background.index += auxiliary_catalog.index.max() + 1
    background["evt_id"] = background.index.values

    induced.index += background.index.max() + 1
    induced["evt_id"] = induced.index.values

    catalog = pd.concat(
        [a for a in [background, induced, auxiliary_catalog]
         if len(a) != 0], sort=True)

    logger.debug(f'number of background events: {len(background.index)}')
    logger.debug(
        f'number of auxiliary events: {len(auxiliary_catalog.index)}')
    generation = 0
    timewindow_length = to_days(simulation_end - auxiliary_start)

    while True:
        logger.debug(f'generation {generation}')
        sources = catalog.query(
            "generation == @generation and n_aftershocks > 0").copy()

        # if no aftershocks are produced by events of this generation, stop
        logger.debug(
            f'number of events with aftershocks: {len(sources.index)}')
        if len(sources.index) == 0:
            break

        # an array with all aftershocks. to be appended to the catalog
        aftershocks = generate_aftershocks(
            sources,
            generation,
            parameters,
            beta_aftershock,
            mc,
            delta_m=delta_m,
            m_max=m_max,
            timewindow_end=simulation_end,
            timewindow_length=timewindow_length,
            auxiliary_end=auxiliary_end,
            approx_times=approx_times,
            mfd_zones=mfd_zones,
            zones_from_latlon=zones_from_latlon)

        aftershocks.index += catalog.index.max() + 1
        aftershocks.query("time>@auxiliary_end", inplace=True)

        logger.debug(f'number of aftershocks: {len(aftershocks.index)}')
        logger.debug('their number of aftershocks should be:'
                     f'{aftershocks["n_aftershocks"].sum()}')
        aftershocks["xi_plus_1"] = 1
        catalog = pd.concat([
            catalog, aftershocks
        ], ignore_index=False, sort=True)

        generation = generation + 1
    if filter_polygon:
        catalog = gpd.GeoDataFrame(
            catalog, geometry=gpd.points_from_xy(
                catalog.latitude, catalog.longitude))
        catalog = catalog[catalog.intersects(polygon)]
        return catalog.drop("geometry", axis=1)
    else:
        return catalog


class ETASSimulation:
    def __init__(self, inversion_params: ETASParameterCalculation,
                 gaussian_scale: float = 0.1,
                 approx_times: bool = False,
                 m_max: float = None,
                 induced_info: list = None):

        self.logger = logging.getLogger(__name__)

        self.inversion_params = inversion_params

        self.forecast_start_date = None
        self.forecast_end_date = None

        self.catalog = None
        self.target_events = None
        self.source_events = None

        self.polygon = None

        self.m_max = m_max
        self.gaussian_scale = gaussian_scale
        self.approx_times = approx_times

        self.mfd_zones = None
        self.zones_from_latlon = None

        self.background_lats = None
        self.background_lons = None
        self.background_probs = None
        self.bg_grid = False
        self.bsla = None
        self.bslo = None

        self.induced = (induced_info is not None)
        if self.induced:
            self.induced_lats, self.induced_lons, self.induced_term, \
            self.induced_bsla, self.induced_bslo, \
            self.n_induced = induced_info
        else:
            self.induced_lats, self.induced_lons, self.induced_term, \
            self.induced_bsla, self.induced_bslo, \
            self.n_induced = [None, None, None, None, None, None]

        self.logger.debug('using parameters calculated on {}\n'.format(
            inversion_params.calculation_date))
        self.logger.debug(pprint.pformat(self.inversion_params.theta))

        self.logger.info(
            'm_ref: {}, min magnitude in training catalog: {}'.format(
                self.inversion_params.m_ref,
                self.inversion_params.catalog['magnitude'].min()))

    def prepare(self):
        self.polygon = Polygon(self.inversion_params.shape_coords)
        # Xi_plus_1 is aftershock productivity inflation factor.
        # If not used, set to 1.
        self.source_events = self.inversion_params.source_events.copy()
        if 'xi_plus_1' not in self.source_events.columns:
            self.source_events['xi_plus_1'] = 1

        self.catalog = pd.merge(
            self.source_events,
            self.inversion_params.catalog[["latitude",
                                           "longitude", "time", "magnitude"]],
            left_index=True,
            right_index=True,
            how='left',
        )
        assert len(self.catalog) == len(self.source_events), \
            "lost/found some sources in the merge! " \
            f"{len(self.catalog)} -- " \
            f"{len(self.source_events)}"

        np.testing.assert_allclose(
            self.catalog.magnitude.min(),
            self.inversion_params.m_ref,
            err_msg="smallest magnitude in sources is "
                    f"{self.catalog.magnitude.min()} "
                    f"but I am supposed to simulate "
                    f"above {self.inversion_params.m_ref}")

        self.target_events = self.inversion_params.target_events.query(
            "magnitude>=@self.inversion_params.m_ref "
            "-@self.inversion_params.delta_m/2")
        self.target_events = gpd.GeoDataFrame(
            self.target_events, geometry=gpd.points_from_xy(
                self.target_events.latitude,
                self.target_events.longitude))
        self.target_events = self.target_events[
            self.target_events.intersects(self.polygon)]

        self.background_lats = self.target_events['latitude']
        self.background_lons = self.target_events['longitude']
        self.background_probs = self.target_events['P_background'] * (
                    self.target_events['zeta_plus_1']
                    / self.target_events['zeta_plus_1'].max())

    def simulate(self, forecast_n_days: int, n_simulations: int,
                 m_threshold: float = None, chunksize: int = 100,
                 info_cols: list = ['is_background'],
                 i_start: int = 0) -> None:
        start = dt.datetime.now()
        np.random.seed()
        logger.debug('induced info: {}'.format(self.induced))

        if m_threshold is None:
            m_threshold = self.inversion_params.m_ref

        # columns returned in resulting DataFrame
        cols = ['latitude', 'longitude',
                'magnitude', 'time'] + info_cols
        if n_simulations != 1:
            cols.append('catalog_id')

        # end of training period is start of forecasting period
        self.forecast_start_date = self.inversion_params.timewindow_end
        self.forecast_end_date = self.forecast_start_date \
                                 + dt.timedelta(days=forecast_n_days)

        simulations = pd.DataFrame()
        for sim_id in np.arange(i_start, n_simulations):
            continuation = simulate_catalog_continuation(
                self.catalog,
                auxiliary_start=self.inversion_params.auxiliary_start,
                auxiliary_end=self.forecast_start_date,
                polygon=self.polygon,
                simulation_end=self.forecast_end_date,
                parameters=self.inversion_params.theta,
                mc=self.inversion_params.m_ref
                   - self.inversion_params.delta_m / 2,
                m_max=self.m_max + self.inversion_params.delta_m / 2
                if self.m_max is not None else None,
                beta_main=self.inversion_params.beta,
                background_lats=self.background_lats,
                background_lons=self.background_lons,
                background_probs=self.background_probs,
                bg_grid=self.bg_grid,
                bsla=self.bsla,
                bslo=self.bslo,
                gaussian_scale=self.gaussian_scale,
                filter_polygon=False,
                approx_times=self.approx_times,
                mfd_zones=self.mfd_zones,
                zones_from_latlon=self.zones_from_latlon,
                induced_lats=self.induced_lats,
                induced_lons=self.induced_lons,
                induced_term=self.induced_term,
                induced_bsla=self.induced_bsla,
                induced_bslo=self.induced_bslo,
                n_induced=self.n_induced,
            )

            continuation["catalog_id"] = sim_id
            simulations = pd.concat([simulations, continuation],
                                    ignore_index=False)

            if sim_id % chunksize == 0 or sim_id == n_simulations - 1:
                simulations.query(
                    'time>=@self.forecast_start_date and '
                    'time<=@self.forecast_end_date and '
                    'magnitude>=@m_threshold-@self.inversion_params.delta_m/2',
                    inplace=True)
                simulations.magnitude = round_half_up(simulations.magnitude, 1)
                simulations.index.name = 'id'
                self.logger.debug(
                    "storing simulations up to {}".format(sim_id))
                self.logger.debug(
                    f'took {dt.datetime.now() - start} to simulate '
                    f'{sim_id + 1} catalogs.')

                # now filter polygon
                simulations = gpd.GeoDataFrame(
                    simulations, geometry=gpd.points_from_xy(
                        simulations.latitude, simulations.longitude))
                simulations = simulations[simulations.intersects(self.polygon)]

                yield simulations[cols]

                simulations = pd.DataFrame()
        self.logger.info("DONE simulating!")

    def simulate_to_csv(self, fn_store: str, forecast_n_days: int,
                        n_simulations: int, m_threshold: float = None,
                        chunksize: int = 100, info_cols: list = [],
                        i_start: int = 0) -> None:

        i_end = i_start + n_simulations

        os.makedirs(os.path.dirname(fn_store), exist_ok=True)

        if not os.path.exists(fn_store):
            # create new file for first chunk
            generator = self.simulate(
                forecast_n_days,
                i_end,
                m_threshold,
                chunksize, info_cols,
                i_start=i_start)

            next(generator).to_csv(fn_store, mode='w', header=True,
                                   index=False)
        else:
            logger.info('file already exists.')
            with open(fn_store, 'r') as f:
                lines = f.read().splitlines()
                first_line = lines[0]
                first_line = first_line.split(",")
                if 'catalog_id' in first_line:
                    cat_id_index = first_line.index('catalog_id')
                    last_line = lines[-1]
                    last_line = last_line.split(",")
                    last_index = int(float(last_line[cat_id_index]))
                    logger.debug(
                        "simulations were stored until index {}".format(
                            last_index))
                else:
                    logger.info("no column 'catalog_id' in this file.")
                    last_index = -1

            max_store_incomplete = ((
                n_simulations - 1) // chunksize) * chunksize
            if last_index > max_store_incomplete:
                logger.debug("all done, nothing left to do.")
                exit()
            else:
                chunks_done = (last_index // chunksize)
                if last_index % chunksize > 0:
                    # last simulation done
                    # didn't have any events
                    chunks_done += 1

                i_next = chunks_done * chunksize + 1
                logger.debug(
                    "will continue from simulation {}.".format(i_next))

                generator = self.simulate(forecast_n_days,
                                          i_end,
                                          m_threshold,
                                          chunksize, info_cols,
                                          i_start=i_next)

        # append rest of chunks to file
        for chunk in generator:
            chunk.to_csv(fn_store, mode='a', header=False,
                         index=False)

    def simulate_to_df(self, forecast_n_days: int,
                       n_simulations: int, m_threshold: float = None,
                       chunksize: int = 100, info_cols: list = []) \
            -> pd.DataFrame:
        store = pd.DataFrame()
        for chunk in self.simulate(forecast_n_days,
                                   n_simulations,
                                   m_threshold,
                                   chunksize, info_cols):
            store = pd.concat([store, chunk], ignore_index=False)

        return store
