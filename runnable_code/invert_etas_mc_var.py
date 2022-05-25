import numpy as np
import datetime as dt

from code.inversion import invert_etas_params

if __name__ == '__main__':

    """
        TO RUN ETAS INVERSION WITH VARYING MC:
            "mc" needs to be set to "var" in the metadata below,
            the catalog stored in "fn_catalog" needs to have such a column "mc_current" (example described below),
            and a reference magnitude "m_ref" needs to be provided.
                this can, but does not have to be the minimum mc_current.
                
        example_catalog_mc_var.csv contains a synthetic catalog. 
        it has an additional column named "mc_current", 
            which for each event contains the completeness magnitude (mc) valid at the time and location of the event.
            in Sothern California (latitude < 37),
                mc = 3.0 if time <= 1981/1/1,
                     2.7 if 1981/1/1 < time <= 2010/1/1
                     2.5 if time > 2010/1/1
            in Northern California (latitude >= 37),
                mc = 3.1 if time <= 1981/1/1,
                     2.8 if 1981/1/1 < time <= 2010/1/1
                     2.6 if time > 2010/1/1
        this is an example of space-time varying mc, and is not intended to reflect reality.
    """
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
        "fn_catalog": "example_catalog_mc_var.csv",
        "data_path": "mc_var_",
        "auxiliary_start": dt.datetime(1971, 1, 1),
        "timewindow_start": dt.datetime(1981, 1, 1),
        "timewindow_end": dt.datetime(2022, 1, 1),
        "theta_0": theta_0,
        "mc": 'var',
        "m_ref": 2.5,
        "delta_m": 0.1,
        "coppersmith_multiplier": 100,
        "shape_coords": np.load("california_shape.npy"),
    }

    parameters = invert_etas_params(
        inversion_meta
    )
