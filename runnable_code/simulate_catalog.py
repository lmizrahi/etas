import json
import datetime as dt
import pandas as pd
import numpy as np
from shapely.geometry import Polygon


from utils.simulation import generate_catalog
from utils.inversion import round_half_up

if __name__ == '__main__':

    # reads configuration for example ETAS parameter inversion
    with open("../config/simulate_catalog_config.json", 'r') as f:
        simulation_config = json.load(f)

    region = Polygon(np.load(simulation_config["shape_coords"]))

    # np.random.seed(777)

    synthetic = generate_catalog(
        polygon=region,
        timewindow_start=pd.to_datetime(simulation_config["burn_start"]),
        timewindow_end=pd.to_datetime(simulation_config["end"]),
        parameters=simulation_config["parameters"],
        mc=simulation_config["mc"],
        beta_main=simulation_config["beta"],
        delta_m=simulation_config["delta_m"]
    )

    synthetic.magnitude = round_half_up(synthetic.magnitude, 1)
    synthetic.index.name = 'id'
    print("store catalog..")
    primary_start = simulation_config['primary_start']
    fn_store = simulation_config['fn_store']
    synthetic[["latitude", "longitude", "time", "magnitude"]].query("time>=@primary_start").to_csv(fn_store)
    print("\nDONE!")
