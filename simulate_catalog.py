import numpy as np
import datetime as dt
from shapely.geometry import Polygon


from simulation import generate_catalog
from inversion import round_half_up

if __name__ == '__main__':
    fn_store = 'my_synthetic_catalog.csv'
    shape_coords = np.load("california_shape.npy")
    caliregion = Polygon(shape_coords)
    burn_start = dt.datetime(1871, 1, 1)
    primary_start = dt.datetime(1971, 1, 1)
    end = dt.datetime(2021, 1, 1)

    delta_m = 0.1
    mc = 3.6
    beta = np.log(10)

    parameters = {
        'log10_mu': -7.5,
        'log10_k0': -2.49,
        'a': 1.69,
        'log10_c': -2.95,
        'omega': -0.03,
        'log10_tau': 3.99,
        'log10_d': -0.35,
        'gamma': 1.22,
        'rho': 0.51
    }

    # np.random.seed(777)

    synthetic = generate_catalog(
        polygon=caliregion,
        timewindow_start=burn_start,
        timewindow_end=end,
        parameters=parameters,
        mc=mc,
        beta_main=beta,
        delta_m=delta_m
    )

    synthetic.magnitude = round_half_up(synthetic.magnitude, 1)
    synthetic.index.name = 'id'
    print("store catalog..")
    synthetic[["latitude", "longitude", "time", "magnitude"]].query("time>=@primary_start").to_csv(fn_store)
    print("\nDONE!")
