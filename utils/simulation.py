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

import pandas as pd
import numpy as np
import datetime as dt
import geopandas as gpd
from scipy.special import gammainccinv, gamma as gamma_func

from utils.inversion import parameter_dict2array, to_days, branching_ratio, \
    haversine, expected_aftershocks, upper_gamma_ext
from utils.mc_b_est import simulate_magnitudes


from shapely.geometry import Polygon


def inverse_upper_gamma_ext(a, y):
    # TODO: find a more elegant way to do this
    if a > 0:
        return gammainccinv(a, y/gamma_func(a))
    else:
        from pynverse import inversefunc
        import warnings
        from scipy.optimize import minimize

        uge = (lambda x: upper_gamma_ext(a, x))

        # numerical inverse
        def num_inv(a, y):
            def diff(x, xhat):
                xt = upper_gamma_ext(a, x)
                return (xt - xhat)**2
            x = np.zeros(len(y))
            for idx, y_value in enumerate(y):
                res = minimize(diff, 1.0, args=(y_value), method='Nelder-Mead', tol=1e-6)
                x[idx] = res.x[0]

            return x

        warnings.filterwarnings("ignore")
        result = inversefunc(uge, y)
        warnings.filterwarnings("default")


        # where inversefunc was unable to calculate a result, calculate numerical approximation
        nan_idxs = np.argwhere(np.isnan(result)).flatten()
        if len(nan_idxs) > 0:
            num_res = num_inv(a, y[nan_idxs])
            result[nan_idxs] = num_res

        return result


def simulate_aftershock_time(log10_c, omega, log10_tau, size=1):
    # time delay in days
    c = np.power(10, log10_c)
    tau = np.power(10, log10_tau)
    y = np.random.uniform(size=size)

    return inverse_upper_gamma_ext(-omega, (1 - y) * upper_gamma_ext(-omega, c / tau)) * tau - c


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


def simulate_background_location(latitudes, longitudes, background_probs, scale=0.1, n=1):
    np.random.seed()
    keep_idxs = background_probs >= np.random.uniform(size=len(background_probs))

    sample_lats = latitudes[keep_idxs]
    sample_lons = longitudes[keep_idxs]

    choices = np.floor(np.random.uniform(0, len(sample_lats), size=n)).astype(int)

    lats = sample_lats.iloc[choices] + np.random.normal(loc=0, scale=scale, size=n)
    lons = sample_lons.iloc[choices] + np.random.normal(loc=0, scale=scale, size=n)

    return lats, lons


def generate_background_events(polygon, timewindow_start, timewindow_end,
                               parameters, beta, mc, delta_m=0,
                               background_lats=None, background_lons=None,
                               background_probs=None, gaussian_scale=None
                               ):
    from utils.inversion import polygon_surface, to_days

    theta_without_mu = parameters["log10_k0"], parameters["a"], parameters["log10_c"], parameters["omega"], \
                       parameters["log10_tau"], parameters["log10_d"], parameters["gamma"], parameters["rho"]

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
    expected_n_background = np.power(10, parameters["log10_mu"]) * area * timewindow_length
    n_background = np.random.poisson(lam=expected_n_background)

    # generate too many events, afterwards filter those that are in the polygon
    n_generate = int(np.round(n_background * rectangle_area / area * 1.2))

    print("  number of background events needed:", n_background)
    print("  generating", n_generate, "to throw away those outside the polygon")

    # define dataframe with background events
    catalog = pd.DataFrame(None, columns=["latitude", "longitude", "time", "magnitude", "parent", "generation"])

    # generate lat, long
    if background_probs is not None:
        catalog["latitude"], catalog["longitude"] = simulate_background_location(
            background_lats,
            background_lons,
            background_probs=background_probs,
            scale=gaussian_scale,
            n=n_generate
        )
    else:
        catalog["latitude"] = np.random.uniform(min_lat, max_lat, size=n_generate)
        catalog["longitude"] = np.random.uniform(min_lon, max_lon, size=n_generate)

    catalog = gpd.GeoDataFrame(catalog, geometry=gpd.points_from_xy(catalog.latitude, catalog.longitude))
    catalog = catalog[catalog.intersects(polygon)].head(n_background)

    # if not enough events fell into the polygon, do it again...
    while len(catalog) != n_background:
        print("  didn't create enough events. trying again..")

        # define dataframe with background events
        catalog = pd.DataFrame(None, columns=["latitude", "longitude", "time", "magnitude", "parent", "generation"])

        # generate lat, long
        catalog["latitude"] = np.random.uniform(min_lat, max_lat, size=n_generate)
        catalog["longitude"] = np.random.uniform(min_lon, max_lon, size=n_generate)

        catalog = gpd.GeoDataFrame(catalog, geometry=gpd.points_from_xy(catalog.latitude, catalog.longitude))
        catalog = catalog[catalog.intersects(polygon)].head(n_background)

    # generate time, magnitude
    catalog["time"] = [
        timewindow_start
        + dt.timedelta(days=d) for d in np.random.uniform(0, timewindow_length, size=n_background)
    ]

    catalog["magnitude"] = simulate_magnitudes(n_background, beta=beta, mc=mc - delta_m / 2)

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
    catalog["n_aftershocks"] = np.random.poisson(lam=catalog["expected_n_aftershocks"])

    return catalog.drop("geometry", axis=1)


def generate_aftershocks(sources, generation, parameters, beta, mc, timewindow_end, timewindow_length,
                         delta_m=0, earth_radius=6.3781e3,
                         polygon=None
                         ):
    theta = parameter_dict2array(parameters)
    theta_without_mu = parameters["log10_k0"], parameters["a"], parameters["log10_c"], parameters["omega"], \
                       parameters["log10_tau"], parameters["log10_d"], parameters["gamma"], parameters["rho"]

    all_aftershocks = []

    # random timedeltas for all aftershocks
    total_n_aftershocks = sources["n_aftershocks"].sum()

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
    aftershocks = aftershocks[[col for col in aftershocks.columns if "parent" in col]].reset_index(drop=True)
    aftershocks["time_delta"] = all_deltas
    aftershocks.query("time_delta <= @ timewindow_length", inplace=True)

    aftershocks["time"] = aftershocks["parent_time"] + pd.to_timedelta(aftershocks["time_delta"], unit='d')
    aftershocks.query("time <= @ timewindow_end", inplace=True)

    # location of aftershock
    aftershocks["radius"] = simulate_aftershock_radius(
        parameters["log10_d"], parameters["gamma"], parameters["rho"], aftershocks["parent_magnitude"], mc=mc
    )
    aftershocks["angle"] = np.random.uniform(0, 2 * np.pi, size=len(aftershocks))
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
            aftershocks,
            geometry=gpd.points_from_xy(aftershocks.latitude, aftershocks.longitude)
        )
        aftershocks = aftershocks[aftershocks.intersects(polygon)]

    aadf = aftershocks[as_cols].reset_index(drop=True)

    # magnitudes
    n_total_aftershocks = len(aadf.index)
    aadf["magnitude"] = simulate_magnitudes(n_total_aftershocks, beta=beta, mc=mc - delta_m / 2)

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
    aadf["n_aftershocks"] = np.random.poisson(lam=aadf["expected_n_aftershocks"])

    return aadf


def prepare_auxiliary_catalog(auxiliary_catalog, parameters, mc, delta_m=0):
    theta_without_mu = parameters["log10_k0"], parameters["a"], parameters["log10_c"], parameters["omega"], \
                       parameters["log10_tau"], parameters["log10_d"], parameters["gamma"], parameters["rho"]

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

    catalog["n_aftershocks"] = catalog["expected_n_aftershocks"].apply(
        np.random.poisson,
        # axis = 1
    )

    return catalog


def generate_catalog(
        polygon, timewindow_start, timewindow_end,
        parameters, mc, beta_main, beta_aftershock=None, delta_m=0,
        background_lats=None, background_lons=None, background_probs=None, gaussian_scale=None
):
    """
    Simulates an earthquake catalog.
        polygon: lat lon coordinates in which catalog is generated
        timewindow_start: datetime of simulation start
        timewindow_end: datetime of simulation end
        parameters: as estimated in the ETAS EM inversion
        mc: completeness magnitude. if delta_m > 0, magnitudes are simulated above mc-delta_m/2
        beta_main: beta used to generate background event magnitudes,
        beta_aftershock: beta used to generate aftershock magnitudes. if none, beta_main is used
        delta_m: bin size of magnitudes

        optional: use coordinates and independence probabilities
        of observed events to simulate locations of background events
        background_lats: list of latitudes
        background_lons: list of longitudes
        background_probs: list of independence probabilities
            these three lists are assumed to be sorted
            such that corresponding entries belong to the same event
        gaussian_scale: sigma to be used when background loations are generated
    """

    if beta_aftershock is None:
        beta_aftershock = beta_main

    # generate background events
    print("generating background events..")
    catalog = generate_background_events(
        polygon, timewindow_start, timewindow_end, parameters, beta=beta_main, mc=mc, delta_m=delta_m,
        background_lats=background_lats, background_lons=background_lons,
        background_probs=background_probs, gaussian_scale=gaussian_scale
    )

    theta = parameters["log10_mu"], parameters["log10_k0"], parameters["a"], parameters["log10_c"], parameters["omega"], \
            parameters["log10_tau"], parameters["log10_d"], parameters["gamma"], parameters["rho"]
    br = branching_ratio(theta, beta_main)

    print('  number of background events:', len(catalog.index))
    print('\n  branching ratio:', br)
    print('  expected total number of events (if time were infinite):', len(catalog.index) * 1 / (1 - br))

    generation = 0
    timewindow_length = to_days(timewindow_end - timewindow_start)

    while True:
        print('\n\nsimulating aftershocks of generation', generation, '..')
        sources = catalog.query("generation == @generation and n_aftershocks > 0").copy()

        # if no aftershocks are produced by events of this generation, stop
        print('  number of events with aftershocks:', len(sources.index))

        if len(sources.index) == 0:
            break

        # an array with all aftershocks. to be appended to the catalog
        aftershocks = generate_aftershocks(
            sources, generation, parameters, beta_aftershock, mc, delta_m=delta_m,
            timewindow_end=timewindow_end, timewindow_length=timewindow_length,
        )

        aftershocks.index += catalog.index.max() + 1

        print('  number of generated aftershocks:', len(aftershocks.index))

        catalog = catalog.append(aftershocks, ignore_index=False, sort=True)

        generation = generation + 1

    print('\n\ntotal events simulated:', len(catalog))
    catalog = gpd.GeoDataFrame(catalog, geometry=gpd.points_from_xy(catalog.latitude, catalog.longitude))
    catalog = catalog[catalog.intersects(polygon)]
    print('inside the polygon:', len(catalog))

    return catalog.drop("geometry", axis=1)


def simulate_catalog_continuation(
        auxiliary_catalog, auxiliary_start, auxiliary_end,
        polygon, simulation_end,
        parameters, mc, beta_main, beta_aftershock=None, delta_m=0, verbose=False,
        background_lats=None, background_lons=None, background_probs=None, gaussian_scale=None,
):
    # auxiliary_catalog: catalog used for aftershock generation in simulation period
    # auxiliary_start: start time of auxiliary catalog
    # auxiliary_end: end time of auxiliary_catalog. start of simulation period
    # polygon: polygon in which events are generated
    # simulation_end: end time of simulation period
    # parameters: ETAS parameters
    # mc: reference mc for ETAS parameters
    # beta_main: beta for main shocks. can be a map for spatially variable betas
    # beta_aftershock: beta for aftershocks. if None, is set to be same as main shock beta
    # delta_m: bin size for discrete magnitudes
    # background_lats: latitudes of background events
    # background_lons: longitudes of background events
    # background_probs: independence probabilities of background events
    # gaussian_scale: extent of background location smoothing

    # preparing betas
    if beta_aftershock is None:
        beta_aftershock = beta_main

    background = generate_background_events(
        polygon, auxiliary_end, simulation_end, parameters, beta_main, mc, delta_m,
        background_lats=background_lats, background_lons=background_lons,
        background_probs=background_probs, gaussian_scale=gaussian_scale,
    )
    background["evt_id"] = ''
    background["xi_plus_1"] = 1
    auxiliary_catalog = prepare_auxiliary_catalog(
        auxiliary_catalog=auxiliary_catalog, parameters=parameters, mc=mc,
        delta_m=delta_m,
    )
    background.index += auxiliary_catalog.index.max() + 1
    background["evt_id"] = background.index.values

    catalog = background.append(auxiliary_catalog, sort=True)

    if verbose:
        print('number of background events:', len(background.index))
        print('number of auxiliary events:', len(auxiliary_catalog.index))
    generation = 0
    timewindow_length = to_days(simulation_end - auxiliary_start)

    while True:
        if verbose:
            print('generation', generation)
        sources = catalog.query("generation == @generation and n_aftershocks > 0").copy()

        # if no aftershocks are produced by events of this generation, stop
        if verbose:
            print('number of events with aftershocks:', len(sources.index))
        if len(sources.index) == 0:
            break

        # an array with all aftershocks. to be appended to the catalog
        aftershocks = generate_aftershocks(sources, generation, parameters, beta_aftershock, mc, delta_m=delta_m,
                                           timewindow_end=simulation_end, timewindow_length=timewindow_length)

        aftershocks.index += catalog.index.max() + 1
        aftershocks.query("time>@auxiliary_end", inplace=True)
        if verbose:
            print('number of aftershocks:', len(aftershocks.index))
            print('their number of aftershocks should be:', aftershocks["n_aftershocks"].sum())
        aftershocks["xi_plus_1"] = 1
        catalog = catalog.append(aftershocks, ignore_index=False, sort=True)

        generation = generation + 1

    catalog = gpd.GeoDataFrame(catalog, geometry=gpd.points_from_xy(catalog.latitude, catalog.longitude))
    catalog = catalog[catalog.intersects(polygon)]
    return catalog.drop("geometry", axis=1)

