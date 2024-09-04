import json
from datetime import datetime

import numpy as np
import pandas as pd

from etas.inversion import round_half_up

try:
    from hermes_model import ModelInput, validate_entrypoint
except ImportError:
    raise ImportError(
        "The hermes package is required to run the model. "
        "Please install this package with the 'hermes' extra requirements.")

from seismostats import Catalog, ForecastCatalog
from shapely import wkt

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
    catalog.rename_axis(None, inplace=True)

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
    simulation = ETASSimulation(etas_parameters)
    simulation.prepare()

    # prepare background grid for simulation of locations
    bg_grid = pd.read_csv(model_parameters["fn_bg_grid"], index_col=0)
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


def entrypoint_europe(model_input: ModelInput) -> list[ForecastCatalog]:
    # bla
    return []
