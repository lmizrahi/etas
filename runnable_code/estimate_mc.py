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
		stop_when_passed=False,
		verbose=True,
		n_samples=1000
	)