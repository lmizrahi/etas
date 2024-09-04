from datetime import datetime, timedelta

import numpy as np
from seismostats import Catalog
from seismostats.io.client import FDSNWSEventClient
from shapely.geometry import Polygon
from shapely.wkt import dumps

from etas.oef import entrypoint_suiETAS


def main():
    """
    Requires to install the package with 'hermes' extras.
    pip install -e .[hermes]

    Makes use of the standardized interface to run the model.

    More information under https://gitlab.seismo.ethz.ch/indu/hermes-model
    """

    format = '%Y-%m-%d %H:%M:%S'
    auxiliary_start = datetime.strptime("1975-01-01 00:00:00", format)
    timewindow_start = datetime.strptime("1980-01-01 00:00:00", format)
    timewindow_end = datetime.now()

    min_longitude = 5
    max_longitude = 11
    min_latitude = 45
    max_latitude = 48
    min_magnitude = 2

    url = 'http://eida.ethz.ch/fdsnws/event/1/query'
    client = FDSNWSEventClient(url)
    catalog = client.get_events(
        start_time=auxiliary_start,
        end_time=timewindow_end,
        min_magnitude=min_magnitude,
        min_longitude=min_longitude,
        max_longitude=max_longitude,
        min_latitude=min_latitude,
        max_latitude=max_latitude,
        include_all_magnitudes=True,
        event_type='earthquake',
        include_uncertainty=True,
        include_ids=True)

    polygon = Polygon(np.load('../etas/oef/data/ch_shape_buffer.npy'))

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
            "mc": 2.2,
            "m_ref": 2.2,
            "delta_m": 0.1,
            "coppersmith_multiplier": 100,
            "earth_radius": 6.3781e3,
            "auxiliary_start": auxiliary_start,
            "timewindow_start": timewindow_start,
            "m_max": 7.6,
            "fn_bg_grid": "SUIhaz2015_rates.csv",
            "m_thr": 2.5,
            "n_simulations": 100
        },
    }

    results = entrypoint_suiETAS(model_input)
    print(results)


if __name__ == "__main__":
    main()
