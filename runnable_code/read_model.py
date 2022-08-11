import logging
import pandas as pd
import numpy as np
from numpy import array  # noqa
import datetime as dt
import json
import geopandas as gpd
from shapely.geometry import Polygon
import pprint
from etas import set_up_logger

from etas.simulation import simulate_catalog_continuation
from etas.inversion import ETASParameterCalculation

set_up_logger(level=logging.INFO)

if __name__ == '__main__':

	logger = logging.getLogger(__name__)
	# read configuration in
	# '../config/simulate_catalog_continuation_config.json'
	with open('../output_data/parameters_f8b8fa13-2355-498e-860a-d14da357451f.json', 'r') as f:
		metadata = json.load(f)

	model = ETASParameterCalculation(metadata)

	logger.info('parameters are {}'.format(model.theta))
