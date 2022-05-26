# inversion of ETAS parameters
#
# as described by Mizrahi et al., 2021
# Leila Mizrahi, Shyam Nandan, Stefan Wiemer;
# The Effect of Declustering on the Size Distribution of Mainshocks.
# Seismological Research Letters 2021; doi: https://doi.org/10.1785/0220200231
#
# for varying mc, as described by Mizrahi et al., 2021
# Leila Mizrahi, Shyam Nandan, Stefan Wiemer;
# Embracing Data Incompleteness for Better Earthquake Forecasting.
# Journal of Geophysical Research: Solid Earth.
# doi: https://doi.org/10.1029/2021JB022379


import json

from utils.inversion import invert_etas_params

if __name__ == '__main__':

    # reads configuration for example ETAS parameter inversion
    with open("../config/invert_etas_config.json", 'r') as f:
        inversion_config = json.load(f)

    # to run varying mc example inversion, uncomment this (explanations below):
    #
    # with open("../config/invert_etas_mc_var_config.json", 'r') as f:
    #     inversion_config = json.load(f)

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


    WHEN RUNNING ETAS INVERSION WITH VARYING MC:
            "mc" needs to be set to "var" in the config data file in '../config/invert_etas_mc_var_config.json',
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
