import numpy as np
import datetime as dt
import json

from utils.inversion import invert_etas_params

if __name__ == '__main__':

    with open("../config/invert_etas_config.json", 'r') as f:
        inversion_config = json.load(f)

    parameters = invert_etas_params(
        inversion_config
    )

    """
        Inverts ETAS parameters.
        config data is stored in '../config/invert_etas_config..json'
        necessary attributes are:
            fn_catalog: filename of the catalog (absolute path or filename in current directory)
                        catalog is expected to be a csv file with the following columns:
                        id, latitude, longitude, time, magnitude
                        id needs to contain a unique identifier for each event
                        time contains datetime of event occurrence
                        see example_catalog.csv for an example
            data_path: path where output data will be stored
            auxiliary_start: start date of the auxiliary catalog (str or datetime).
                             events of the auxiliary catalog act as sources, not as targets
            timewindow_start: start date of the primary catalog , end date of auxiliary catalog (str or datetime).
                             events of the primary catalog act as sources and as targets
            timewindow_end: end date of the primary catalog (str or datetime)
            mc: cutoff magnitude. catalog needs to be complete above mc.
                if mc == 'var', m_ref is required, and the catalog needs to contain a column named "mc_current".
            m_ref: reference magnitude when mc is variable. not required unless mc == 'var'.
            delta_m: size of magnitude bins
            coppersmith_multiplier: events further apart from each other than
                                    coppersmith subsurface rupture length * this multiplier
                                    are considered to be uncorrelated (to reduce size of distance matrix)
            shape_coords: coordinates of the boundary of the region to consider,
                          or path to a .npy file containing the coordinates.
                          (list of lists, i.e. [[lat1, lon1], [lat2, lon2], [lat3, lon3]])
                          
                          necessary unless globe=True when calling invert_etas_params(), i.e.
                          invert_etas_params(inversion_config, globe=True). 
                          in this case, the whole globe is considered
        accepted attributes are:
            theta_0: initial guess for parameters. does not affect final parameters,
                     but with a good initial guess the algorithm converges faster.

    """
