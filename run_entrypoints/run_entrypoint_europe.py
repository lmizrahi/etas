from datetime import datetime, timedelta

import numpy as np
from seismostats.io.client import FDSNWSEventClient
from shapely.geometry import Polygon
from shapely.wkt import dumps

from etas.oef import entrypoint_europe


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
    timewindow_end = datetime.strptime("2023-02-06 00:00:00", format)
    # timewindow_end at later points causes an error in to_quakeml()

    min_longitude = -38
    max_longitude = 48
    min_latitude = 25
    max_latitude = 73

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

    polygon = Polygon(np.load('../etas/oef/data/europe_shape_r.npy'))

    forecast_duration = 30  # days

    model_input = {
        'forecast_start': timewindow_end,
        'forecast_end': timewindow_end + timedelta(days=forecast_duration),
        'bounding_polygon': dumps(polygon),
        'depth_min': 0,
        'depth_max': 1,                         # always in WGS84
        'seismicity_observation': catalog.to_quakeml(),
        'model_parameters': {
            "theta_0": {},
            "mc": 4.6,
            "m_ref": 3.6,
            "delta_m": 0.2,
            "coppersmith_multiplier": 100,
            "earth_radius": 6.3781e3,
            "auxiliary_start": auxiliary_start,
            "timewindow_start": timewindow_start,
            "timewindow_end": timewindow_end,
            "n_simulations": 100,
            "beta": 2.8414508,
            "theta": {
                "log10_mu": -7.036605653217294,
                "log10_k0": -1.9810161198259735,
                "a": 3.393445580204645,
                "log10_c": -2.3188629487442154,
                "omega": -0.027169946471281122,
                "log10_tau": 3.8028947342444486,
                "log10_d": 0.698787394760153,
                "gamma": 1.2242424408776438,
                "rho": 0.8910493957656155
            }
        }
    }

    results = entrypoint_europe(model_input)
    results[0].to_csv('forecast.csv')


if __name__ == "__main__":
    main()
