import argparse
import numpy
from etas.mc_b_est import estimate_mc, round_half_up


def est_mc(magnitudes: (numpy.ndarray, str), mc_min: float = 2.0, mc_max: float = 5.5, mc_dm: float = 0.1,
           delta_m: float = 0.1, p_val: float = 0.05, n_samples: int = 1000, verbose=True):
    """
    magnitudes: array of magnitudes
    mcs: values of mc you want to test. make sure they are rounded correctly,
    because 3.19999999 will produce weird results.
    delta_m: magnitude bin size
    p_pass: p_value above which the catalog is accepted to be complete
    stop_when_passed: if True, remaining mc values will not be tested anymore
    verbose: if True, stuff will be printed while the code is running
    n_samples: number of samples that are simulated to obtain the p-value


    see the paper below for details on the method:

    Leila Mizrahi, Shyam Nandan, Stefan Wiemer 2021;
    The Effect of Declustering on the Size Distribution of Mainshocks.
    Seismological Research Letters; doi: https://doi.org/10.1785/0220200231
    """

    if isinstance(magnitudes, str):
        if magnitudes.endswith('npy'):
            magnitudes = numpy.load(magnitudes)
        elif magnitudes.endswith('txt'):
            with open(magnitudes, 'r') as file_:
                magnitudes = numpy.genfromtxt(magnitudes)

    mcs = round_half_up(numpy.arange(mc_min, mc_max, mc_dm), 1)
    mcs_tested, ks_distances, p_values, mc_winner, beta_winner = estimate_mc(
        magnitudes,
        mcs,
        delta_m=delta_m,
        p_pass=p_val,
        stop_when_passed=True,
        verbose=verbose,
        n_samples=n_samples
    )


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('magnitudes', help='file containing magnitudes')
    parser.add_argument('-min', '--mc_min', help='min search mc', type=float)
    parser.add_argument('-max', '--mc_max', help='min search mc', type=float)

    parser.add_argument('-d', '--mc_dm', help='delta search magnitudes', type=float)
    parser.add_argument('-p', '--p_val', help='p value to pass ks test', type=float)
    parser.add_argument('-dm', '--delta_m', help='magnitude bin', type=float)
    parser.add_argument('-n', '--n_samples', help='sample number', type=int)
    parser.add_argument('-v', '--verbose', help='output log', type=bool)
    args = parser.parse_args()
    est_mc(**vars(args))


if __name__ == '__main__':
    main()
