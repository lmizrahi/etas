import numpy as np
from utils.mc_b_est import round_half_up, estimate_mc

if __name__ == '__main__':
	magnitude_sample = np.load("../input_data/magnitudes.npy")

	mcs = round_half_up(np.arange(2.0, 5.5, 0.1), 1)
	mcs_tested, ks_distances, p_values, mc_winner, beta_winner = estimate_mc(
		magnitude_sample,
		mcs,
		delta_m=0.1,
		p_pass=0.05,
		stop_when_passed=True,
		verbose=True,
		n_samples=1000
	)

	'''
	magnitude_sample: array of magnitudes
	mcs: values of mc you want to test. make sure they are rounded correctly, because 3.19999999 will produce weird results.
	delta_m: magnitude bin size
	p_pass: p_value above which the catalog is accepted to be complete
	stop_when_passed: if True, the remaining mc values will not be tested anymore
	verbose: if True, stuff will be printed while the code is running
	n_samples: number of samples that are simulated to obtain the p-value
	
	
	see the paper below for details on the method:
	
	Leila Mizrahi, Shyam Nandan, Stefan Wiemer 2021;
	The Effect of Declustering on the Size Distribution of Mainshocks.
	Seismological Research Letters; doi: https://doi.org/10.1785/0220200231
	'''