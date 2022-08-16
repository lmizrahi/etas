import json
import logging

from etas import set_up_logger
from etas.inversion import ETASParameterCalculation
from etas.simulation import ETASSimulation

set_up_logger(level=logging.INFO)

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # reads configuration for example ETAS parameter inversion
    with open('../config/ch_forecast_config.json', 'r') as f:
        forecast_config = json.load(f)
    etas_invert = ETASParameterCalculation(forecast_config)
    etas_invert.prepare()
    theta = etas_invert.invert()
    etas_invert.store_results(forecast_config['data_path'], True)

    # # reads a previously created parameter calculation from file
    # with open('../output_data/parameters_ch.json', 'r') as f:
    #     forecast_params = json.load(f)
    # etas_invert = ETASParameterCalculation.load_calculation(
    #     forecast_params)

    simulation = ETASSimulation(etas_invert)
    simulation.prepare()
    fn_store_simulation = forecast_config['fn_store_simulation']
    forecast_duration = forecast_config['forecast_duration']
    n_simulations = forecast_config['n_simulations']

    simulation.simulate_many(fn_store_simulation, forecast_duration,
                             n_simulations)
