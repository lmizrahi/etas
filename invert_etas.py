import numpy as np
import datetime as dt

from inversion import invert_etas_params

if __name__ == '__main__':
    theta_0 = {
        'log10_mu': -5.8,
        'log10_k0': -2.6,
        'a': 1.8,
        'log10_c': -2.5,
        'omega': -0.02,
        'log10_tau': 3.5,
        'log10_d': -0.85,
        'gamma': 1.3,
        'rho': 0.66
    }

    inversion_meta = {
        "fn_catalog": "synthetic_catalog.csv",
        "data_path": "",
        "auxiliary_start": dt.datetime(1971, 1, 1),
        "timewindow_start": dt.datetime(1981, 1, 1),
        "timewindow_end": dt.datetime(2021, 1, 1),
        "theta_0": theta_0,
        "mc": 3.6,
        "delta_m": 0.1,
        "coppersmith_multiplier": 100,
        "shape_coords": [list([a[1], a[0]]) for a in (np.load("california_shape.npy"))],
    }

    parameters = invert_etas_params(
        inversion_meta,
    )
