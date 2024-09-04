from datetime import datetime, timedelta

import numpy as np
from seismostats import Catalog
from seismostats.io.client import FDSNWSEventClient
from shapely.geometry import Polygon
from shapely.wkt import dumps

from etas.entrypoint import entrypoint


def main():
    """
    Requires to install the package with 'hermes' extras.
    pip install -e .[hermes]

    Makes use of the standardized interface to run the model.

    More information under https://gitlab.seismo.ethz.ch/indu/hermes-model
    """

    format = '%Y-%m-%d %H:%M:%S'
    auxiliary_start = datetime.strptime("2015-01-01 00:00:00", format)
    timewindow_start = datetime.strptime("2015-01-01 00:00:00", format)
    timewindow_end = datetime.strptime("2022-01-01 00:00:00", format)

    min_longitude = -30
    max_longitude = 50
    min_latitude = 30
    max_latitude = 75

    min_magnitude = 4.5
    url = 'https://www.seismicportal.eu/fdsnws/event/1/query'
    client = FDSNWSEventClient(url)
    catalog = client.get_events(
        start_time=timewindow_start,
        end_time=timewindow_end,
        min_magnitude=min_magnitude,
        min_longitude=min_longitude,
        max_longitude=max_longitude,
        min_latitude=min_latitude,
        max_latitude=max_latitude)

    polygon = Polygon(np.load('../input_data/ch_rect.npy'))

    forecast_duration = 30  # days

    model_input = {
        'forecast_start': timewindow_end,
        'forecast_end': timewindow_end + timedelta(days=forecast_duration),
        'bounding_polygon': dumps(polygon),
        'depth_min': 0,
        'depth_max': 1,                         # always in WGS84
        'seismicity_observation': catalog.to_quakeml(),
        'model_parameters': {
            "theta_0": {
                "log10_mu": -6.21,
                "log10_k0": -2.75,
                "a": 1.13,
                "log10_c": -2.85,
                "omega": -0.13,
                "log10_tau": 3.57,
                "log10_d": -0.51,
                "gamma": 0.15,
                "rho": 0.63
            },
            "mc": 4.6,
            "delta_m": 0.2,
            "coppersmith_multiplier": 100,
            "earth_radius": 6.3781e3,
            "auxiliary_start": auxiliary_start,
            "timewindow_start": timewindow_start,
            "n_simulations": 100
        },
        'invert_parameters': False,
        'fn_parameters': '../output_data/parameters_ch.json'
    }

    results = entrypoint(model_input)
    print(results)


if __name__ == "__main__":
    main()
