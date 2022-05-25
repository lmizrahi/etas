#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
# simulation of catalog continuation (for forecasting)
#
# as described by Mizrahi et al., 2021
# Leila Mizrahi, Shyam Nandan, Stefan Wiemer;
# Embracing Data Incompleteness for Better Earthquake Forecasting.
# Journal of Geophysical Research: Solid Earth.
# doi: https://doi.org/10.1029/2021JB022379
###############################################################################


import pandas as pd
import numpy as np
from numpy import array
import datetime as dt
import json
import geopandas as gpd
from shapely.geometry import Polygon
import pprint

from utils.simulation import simulate_catalog_continuation
from utils.inversion import parameter_dict2array, round_half_up

if __name__ == '__main__':

	# read configuration in '../config/simulate_catalog_continuation_config.json'
	with open('../config/simulate_catalog_continuation_config.json', 'r') as f:
		simulation_config = json.load(f)
	# this should contain paths to the output files that are produced when running invert_etas.py
	# this information is then read and processed below to simulate a continuation
	# of the catalog that was the input of the inversion

	# read parameters
	with open(simulation_config["fn_parameters"], 'r') as f:
		parameters_dict = json.load(f)

	aux_start = pd.to_datetime(parameters_dict["auxiliary_start"])
	prim_start = pd.to_datetime(parameters_dict["timewindow_start"])
	# end of training period is start of forecasting period
	forecast_start_date = pd.to_datetime(parameters_dict["timewindow_end"])
	forecast_end_date = forecast_start_date + dt.timedelta(days=int(simulation_config["forecast_duration"]))

	coordinates = np.array(
		[np.array(a) for a in eval(parameters_dict["shape_coords"])]
	)
	poly = Polygon(coordinates)

	fn_train_catalog = parameters_dict["fn"]
	delta_m = parameters_dict["delta_m"]
	m_ref = (parameters_dict["m_ref"])
	beta = parameters_dict["beta"]

	# read in correct ETAS parameters to be used for simulation
	parameters = eval(parameters_dict["final_parameters"])
	theta = parameter_dict2array(parameters)
	theta_without_mu = theta[1:]
	print("using parameters calculated on", parameters_dict["calculation_date"], "\n")
	pprint.pprint(parameters)

	# read training catalog and source info (contains current rate needed for inflation factor calculation)
	catalog = pd.read_csv(fn_train_catalog, index_col=0, parse_dates=["time"], dtype={"url": str, "alert": str})
	sources = pd.read_csv(simulation_config["fn_src"], index_col=0)
	# xi_plus_1 is aftershock productivity inflation factor. not used here.
	sources["xi_plus_1"] = 1

	catalog = pd.merge(
		sources,
		catalog[["latitude", "longitude", "time", "magnitude"]],
		left_index=True,
		right_index=True,
		how='left',
	)
	assert len(catalog) == len(sources), "lost/found some sources in the merge! " + str(len(catalog)) + " -- " + str(
		len(sources))
	assert catalog.magnitude.min() == m_ref, "smallest magnitude in sources is " + str(
		catalog.magnitude.min()) + " but I am supposed to simulate above " + str(m_ref)

	# background rates
	ip = pd.read_csv(simulation_config["fn_ip"], index_col=0)
	ip.query("magnitude>=@m_ref -@delta_m/2", inplace=True)
	ip = gpd.GeoDataFrame(ip, geometry=gpd.points_from_xy(ip.latitude, ip.longitude))
	ip = ip[ip.intersects(poly)]

	# other constants
	coppersmith_multiplier = parameters_dict["coppersmith_multiplier"]
	earth_radius = parameters_dict["earth_radius"]

	print("m ref:", m_ref, "min magnitude in training catalog:", catalog["magnitude"].min())

	start = dt.datetime.now()

	continuation = simulate_catalog_continuation(
		catalog,
		auxiliary_start=aux_start,
		auxiliary_end=forecast_start_date,
		polygon=poly,
		simulation_end=forecast_end_date,
		parameters=parameters,
		mc=m_ref - delta_m / 2,
		beta_main=beta,
		verbose=False,
		background_lats=ip["latitude"],
		background_lons=ip["longitude"],
		background_probs=ip["P_background"],
		gaussian_scale=0.1
	)
	continuation.query(
		"time>=@forecast_start_date and time<=@forecast_end_date and magnitude >= @m_ref-@delta_m/2",
		inplace=True
	)

	print("took", dt.datetime.now() - start, "to simulate 1 catalog containing", len(continuation), "events.")

	continuation.magnitude = round_half_up(continuation.magnitude, 1)
	continuation.index.name = 'id'
	print("store catalog..")
	continuation[["latitude", "longitude", "time", "magnitude", "is_background"]].sort_values(by="time").to_csv(
		simulation_config["fn_store_simulation"]
	)
	print("\nDONE!")
