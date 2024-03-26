from datetime import datetime

import numpy as np
import pandas as pd

try:
    from ramsis_model import ModelInput, validate_entrypoint
except ImportError:
    raise ImportError(
        "The ramsis package is required to run the model. "
        "Please install this package with the 'ramsis' extra requirements.")

from seismostats import Catalog
from shapely import wkt

from etas.inversion import ETASParameterCalculation
from etas.simulation import ETASSimulation


@validate_entrypoint(induced=False)
def entrypoint(model_input: ModelInput) -> pd.DataFrame:
    """
    Introduces a standardized interface to run this model.

    More information under https://gitlab.seismo.ethz.ch/indu/ramsis-model

    """

    # Prepare seismic data from QuakeML
    catalog = Catalog.from_quakeml(model_input.seismic_catalog)
    catalog.set_index('time', inplace=True, drop=False)
    catalog.rename_axis(
        None, inplace=True)
    catalog['mc_current'] = np.where(
        catalog.index < datetime(
            1992, 1, 1), 2.7, 2.3)

    # Prepare model input
    polygon = np.array(
        wkt.loads(model_input.geometry.bounding_polygon).exterior.coords)
    model_parameters = model_input.model_parameters
    model_parameters['shape_coords'] = polygon
    model_parameters['catalog'] = catalog
    model_parameters['timewindow_end'] = model_input.forecast_start

    # Run ETAS Parameter Inversion
    etas_parameters = ETASParameterCalculation(model_parameters)
    etas_parameters.prepare()
    etas_parameters.invert()

    # Run ETAS Simulation
    simulation = ETASSimulation(etas_parameters)
    simulation.prepare()

    forecast_duration = model_input.forecast_end - model_input.forecast_start

    results = simulation.simulate_to_df(
        forecast_duration.days, model_parameters['n_simulations'])

    # Add required additional columns and attributes
    results['depth'] = 0
    results.starttime = model_input.forecast_start
    results.endtime = model_input.forecast_end

    return [results]
