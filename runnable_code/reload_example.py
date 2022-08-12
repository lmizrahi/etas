import json
import logging
from etas import set_up_logger
from pprint import pprint
from etas.inversion import ETASParameterCalculation

set_up_logger(level=logging.DEBUG)

logger = logging.getLogger(__name__)

if __name__ == '__main__':

    # reads configuration for example ETAS parameter inversion
    with open('../config/ch_forecast_config.json', 'r') as f:
        forecast_config = json.load(f)

    etas_invert = ETASParameterCalculation(forecast_config)

    etas_invert.prepare()

    theta = etas_invert.invert()

    etas_invert.store_results(forecast_config['data_path'], True, True)

    pprint(etas_invert.__dict__)

    with open('../output_data/ch_parameters_ch.json', 'r') as f:
        forecast_config_reload = json.load(f)

    etas_reload = ETASParameterCalculation.load_calculation(
        forecast_config_reload)

    pprint(etas_reload.__dict__)
