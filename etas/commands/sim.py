import argparse
import os
import logging
import json
import numpy

from etas import set_up_logger
from etas.simulation import ETASSimulation
from etas.inversion import ETASParameterCalculation

set_up_logger(level=logging.INFO)


def parse_config(config_fn):
    with open(config_fn, 'r') as f:
        config = json.load(f)
    return True if 'fn_catalog' in config else False


def sim(parameter_fn, output_fn='simulation.csv',
        forecast_duration=365, n_sims=1, **kwargs):

    if isinstance(parameter_fn, str):
        with open(parameter_fn, 'r') as config_file:
            config_dict = json.load(config_file)

    output_fn = os.path.join(os.path.abspath(os.path.dirname(output_fn)), output_fn)

    etas_inversion_reload = ETASParameterCalculation.load_calculation(
        config_dict)

    simulation = ETASSimulation(etas_inversion_reload)
    simulation.prepare()

    simulation.simulate_to_csv(output_fn, forecast_duration, n_sims, **kwargs)


def sim_catalog(config, seed=None, **kwargs):
    exist_auxcat = parse_config(config)

    if seed:
        numpy.random.seed(seed)

    sim(config, **kwargs)


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('parameter_fn',
                        help='Configuration file or parameter file'
                             ' of the simulation', type=str)
    parser.add_argument('-o', '--output_fn', help='Output filename', type=str)
    parser.add_argument('-t', '--forecast_duration',
                        help='Duration of the forecast (in days)', type=int)
    parser.add_argument('-n', '--n_sims',
                        help='Number of synthetic catalogs', type=int)
    parser.add_argument('-mt', '--m_threshold',
                        help='Magnitude threshold of the simulation',
                        type=float)
    parser.add_argument('-s', '--seed',
                        help='Seed for pseudo-random number generation',
                        type=int)
    parser.add_argument('-f', '--fmt',
                        help='Output format',
                        type=str)
    args = parser.parse_args()
    sim(**vars(args))


if __name__ == '__main__':
    main()
