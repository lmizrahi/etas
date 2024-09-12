import json
from importlib import resources

import numpy as np
import pandas as pd

from etas.inversion import (parameter_dict2array, responsibility_factor,
                            round_half_up)

try:
    from hermes_model import ModelInput, validate_entrypoint
except ImportError:
    raise ImportError(
        "The hermes package is required to run the model. "
        "Please install this package with the 'hermes' extra requirements.")

from seismostats import Catalog, ForecastCatalog
from shapely import wkt
from shapely.wkt import loads

from etas.inversion import ETASParameterCalculation
from etas.simulation import ETASSimulation


@validate_entrypoint(induced=False)
def entrypoint_suiETAS(model_input: ModelInput) -> list[ForecastCatalog]:
    """
    Introduces a standardized interface to run the suiETAS model.

    More information under https://gitlab.seismo.ethz.ch/indu/hermes-model

    """

    # Prepare seismic data from QuakeML
    catalog = Catalog.from_quakeml(model_input.seismicity_observation)

    # Prepare model input
    polygon = np.array(
        wkt.loads(model_input.bounding_polygon).exterior.coords)
    model_parameters = model_input.model_parameters
    model_parameters['shape_coords'] = polygon
    model_parameters['catalog'] = catalog
    model_parameters['timewindow_end'] = model_input.forecast_start
    model_parameters['b_positive'] = True

    # Run ETAS Parameter Inversion
    etas_parameters = ETASParameterCalculation(model_parameters)
    etas_parameters.prepare()
    etas_parameters.invert()

    # Run ETAS Simulation
    simulation = ETASSimulation(etas_parameters, m_max=7.6)
    simulation.prepare()

    # prepare background grid for simulation of locations
    with resources.open_binary("etas.oef.data", "SUIhaz2015_rates.csv") as f:
        bg_grid = pd.read_csv(f, index_col=0)

    background_lats = bg_grid.query("in_poly")["latitude"].copy()
    background_lons = bg_grid.query("in_poly")["longitude"].copy()
    background_probs = 1000 * bg_grid.query("in_poly")["rate_2.5"].copy()

    simulation.bg_grid = True
    simulation.background_lats = background_lats
    simulation.background_lons = background_lons
    simulation.background_probs = background_probs
    simulation.bsla = round_half_up(
        np.min(np.diff(np.sort(np.unique(background_lats)))),
        4
    )
    simulation.bslo = round_half_up(
        np.min(np.diff(np.sort(np.unique(background_lons)))),
        4
    )

    forecast_duration = model_input.forecast_end - model_input.forecast_start

    results = simulation.simulate_to_df(
        forecast_duration.days, model_parameters['n_simulations'])

    # Add required additional columns and attributes
    results['depth'] = 0
    results.starttime = model_input.forecast_start
    results.endtime = model_input.forecast_end
    results.n_catalogs = model_parameters['n_simulations']
    results.bounding_polygon = model_input.bounding_polygon
    results.depth_min = model_input.depth_min
    results.depth_max = model_input.depth_max
    return [results]


@validate_entrypoint(induced=False)
def entrypoint_europe(model_input: ModelInput) -> list[ForecastCatalog]:
    # Prepare seismic data from QuakeML
    catalog = Catalog.from_quakeml(model_input.seismicity_observation)

    # Prepare model input
    model_parameters = model_input.model_parameters
    model_parameters["catalog"] = catalog

    etas_parameters = ETASParameterCalculation(model_parameters)

    etas_parameters.timewindow_end = model_input.forecast_start
    etas_parameters.catalog["mc_current"] = model_parameters["mc"]
    etas_parameters.theta = model_parameters["theta"]
    etas_parameters.m_ref = model_parameters["m_ref"]
    mc_above_ref = etas_parameters.catalog["mc_current"] - \
        model_parameters["m_ref"]

    theta = parameter_dict2array(etas_parameters.theta)
    etas_parameters.catalog["xi_plus_1"] = responsibility_factor(
        theta,
        etas_parameters.beta,
        mc_above_ref
    )
    etas_parameters.catalog = etas_parameters.catalog[["longitude",
                                                       "latitude",
                                                       "time",
                                                       "magnitude",
                                                       "mc_current",
                                                       "xi_plus_1"]].copy()

    etas_parameters.catalog["magnitude"] = round_half_up(
        etas_parameters.catalog.magnitude / 0.2) * 0.2
    etas_parameters.source_events = etas_parameters.catalog

    # prepare background grid for simulation of locations
    with resources.open_binary("etas.oef.data", "europe_rate_map.csv") as f:
        bg_grid = pd.read_csv(f, index_col=0)

    background_lats = bg_grid.latitude
    background_lons = bg_grid.longitude
    background_probs = bg_grid.total / bg_grid.total.max()

    # Run ETAS Simulation
    simulation = ETASSimulation(etas_parameters, m_max=10.0)
    simulation.bg_grid = True
    simulation.background_lats = background_lats
    simulation.background_lons = background_lons
    simulation.background_probs = background_probs
    simulation.bsla = round_half_up(
        np.min(np.diff(np.sort(np.unique(background_lats)))),
        4
    )
    simulation.bslo = round_half_up(
        np.min(np.diff(np.sort(np.unique(background_lons)))),
        4
    )
    simulation.catalog = etas_parameters.catalog
    simulation.source_events = etas_parameters.source_events
    simulation.forecast_start_date = model_input.forecast_start
    simulation.forecast_end_date = model_input.forecast_end
    simulation.polygon = loads(model_input.bounding_polygon)

    forecast_duration = model_input.forecast_end - model_input.forecast_start

    results = simulation.simulate_to_df(
        forecast_duration.days, model_parameters['n_simulations'])

    # Add required additional columns and attributes
    results['depth'] = 0
    results.starttime = model_input.forecast_start
    results.endtime = model_input.forecast_end
    results.n_catalogs = model_parameters['n_simulations']
    results.bounding_polygon = model_input.bounding_polygon
    results.depth_min = model_input.depth_min
    results.depth_max = model_input.depth_max

    return [results]
