from datetime import datetime, timedelta

import numpy as np
from seismostats import Catalog
from shapely.geometry import Polygon
from shapely.wkt import dumps

from etas.entrypoint import entrypoint


def main():
    """
    Requires to install the package with 'ramsis' extras.
    pip install -e .[ramsis]

    Makes use of the standardized interface to run the model.

    More information under https://gitlab.seismo.ethz.ch/indu/ramsis-model
    """

    format = '%Y-%m-%d %H:%M:%S'
    auxiliary_start = datetime.strptime("1992-01-01 00:00:00", format)
    timewindow_start = datetime.strptime("1997-01-01 00:00:00", format)
    timewindow_end = datetime.strptime("2022-01-01 00:00:00", format)

    catalog = Catalog.from_quakeml('../input_data/catalog.xml')
    polygon = Polygon(np.load('../input_data/ch_rect.npy'))

    forecast_duration = 30 * 24 * 60 * 60  # seconds

    extent = {
        'bounding_polygon': dumps(polygon),
        'altitude_min': 0,
        'altitude_max': 1
    }

    model_input = {
        'forecast_start': timewindow_end,
        'forecast_end': timewindow_end + timedelta(seconds=forecast_duration),
        'geometry': extent,                         # always in WGS84
        'seismic_catalog': catalog.to_quakeml(),
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
            "mc": 2.3,
            "delta_m": 0.1,
            "coppersmith_multiplier": 100,
            "earth_radius": 6.3781e3,
            "auxiliary_start": auxiliary_start,
            "timewindow_start": timewindow_start,
            "n_simulations": 100
        }}

    results = entrypoint(model_input)
    print(results)


if __name__ == "__main__":
    main()
