import argparse
import logging
import json
from etas import set_up_logger
from etas.inversion import ETASParameterCalculation

set_up_logger(level=logging.INFO)


def invert_etas(config, **kwargs):

    if isinstance(config, str):
        with open(config, 'r') as config_file:
            config_dict = json.load(config_file)

    calculation = ETASParameterCalculation(config_dict, **kwargs)
    calculation.prepare()
    calculation.invert()
    calculation.store_results()


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('config', help='Configuration file of the inversion',
                        type=str)
    parser.add_argument('-i', '--fn_catalog', help='Input catalog',
                        type=str)
    parser.add_argument('-o', '--data_path', help='Output file path',
                        type=str)
    parser.add_argument('-s', '--shape_coords', help='Input region',
                        type=str)
    parser.add_argument('-aux', '--auxiliary_start',
                        help='Catalog auxiliary start',
                        type=str)
    parser.add_argument('-start', '--timewindow_start',
                        help='Training start window',
                        type=str)
    parser.add_argument('-end', '--timewindow_end',
                        help='Training start window',
                        type=str)
    parser.add_argument('-mc', help='Completeness magnitude',
                        type=float)
    args = parser.parse_args()
    invert_etas(**vars(args))


if __name__ == '__main__':
    main()
