import json
import logging
import sys

from etas.evaluation import ETASLikelihoodCalculation
from __init__ import set_up_logger

set_up_logger(level=logging.DEBUG)

config = sys.argv[1]

if __name__ == '__main__':
    # reads configuration for example ETAS parameter inversion
    with open("../config/"+ config+".json", 'r') as f:
        inversion_config = json.load(f)

    with open(inversion_config["data_path"]+"/parameters_0.json", 'r') as f:
        inversion_output = json.load(f)

    calculation = ETASLikelihoodCalculation(inversion_output)
    calculation.prepare(n=1000000)
    calculation.evaluate_baseline_poisson_model()
    nll, sll, tll = calculation.evaluate()
    calculation.store_results(inversion_config['data_path'])