#!/usr/bin/env python
# coding: utf-8
"""Magnitude of completeness and b-value estimation, and magnitude
sampling.

Joint estimation of beta and the completeness magnitude using the p-value
of the Kolmogorov-Smirnov distance to the fitted Gutenberg-Richter law
(`estimate_mc`), as described by Mizrahi et al. (2021), The Effect of
Declustering on the Size Distribution of Mainshocks, Seismological Research
Letters, https://doi.org/10.1785/0220200231; inspired by the method of
Clauset et al. (2009).

``beta`` is the Gutenberg-Richter parameter in natural-log units,
``beta = b * ln(10)``.
"""
import numpy as np

# mc is the binned completeness magnitude,
# so the 'true' completeness magnitude is mc - delta_m / 2


def round_half_up(n, decimals=0):
    # this is because numpy does weird rounding.
    """Round half up (``0.15 -> 0.2``), unlike numpy's round-half-to-even.

    Parameters
    ----------
    n : float or array-like
    decimals : int, default 0

    Returns
    -------
    float or array-like
    """
    multiplier = 10 ** decimals
    return np.floor(n * multiplier + 0.5) / multiplier


def estimate_beta_tinti(magnitudes, mc, weights=None, axis=None, delta_m=0):
    """
    Tinti, S., & Mulargia, F. (1987). Confidence intervals of b values
    for grouped magnitudes. Bulletin of the Seismological Society of
    America, 77(6), 2125-2134.
    """

    if delta_m > 0:
        p = (1 + (delta_m / (np.average(
            magnitudes - mc, weights=weights, axis=axis))))
        beta = 1 / delta_m * np.log(p)
    else:
        beta = 1 / np.average((magnitudes - (mc - delta_m / 2)),
                              weights=weights, axis=axis)
    return beta


def estimate_beta_positive(magnitudes: np.ndarray, delta_m: float = 0
                           ) -> tuple[float, float]:
    """ returns the b-value estimation using the positive differences of the
    Magnitudes

    Source:
        Van der Elst 2021 (J Geophysical Research: Solid Earth, Vol 126, Issue
        2)

    Args:
        magnitudes: vector of magnitudes differences, sorted in time (first
                    entry is the earliest earthquake)
        delta_m:    discretization of magnitudes. default is no discretization

    Returns:
        beta:       maximum likelihood beta (b_value = beta * log10(e))
    """
    mag_diffs = np.diff(magnitudes)
    # only take the values where the next earthquake is larger
    mag_diffs = abs(mag_diffs[mag_diffs > 0])
    beta = estimate_beta_tinti(mag_diffs, mc=delta_m, delta_m=delta_m)

    return beta


def simulate_magnitudes(n, beta, mc, m_max=None):
    """Sample magnitudes from the (truncated) Gutenberg-Richter law.

    Parameters
    ----------
    n : int
        Number of magnitudes.
    beta : float
        Gutenberg-Richter parameter (natural-log units, ``b * ln(10)``).
    mc : float
        Minimum magnitude.
    m_max : float, optional
        Maximum magnitude (truncation).

    Returns
    -------
    np.ndarray
        Continuous magnitudes (not binned).
    """
    if m_max is not None:
        norm_factor = (1 - np.exp(-beta * (m_max - mc)))
    else:
        norm_factor = 1
    mags = np.random.uniform(size=n)
    mags = (-1 * np.log(1 - norm_factor * mags) / beta) + mc
    return mags


def simulate_magnitudes_from_zone(zones, mfds):

    """Sample magnitudes from zone-dependent magnitude distributions.

    Parameters
    ----------
    zones : array-like
        Zone of each event (index labels of ``mfds``).
    mfds : pd.DataFrame
        One row per zone, one column per magnitude bin (column labels are the
        magnitudes), values are the cumulative distribution function.

    Returns
    -------
    np.ndarray
        For each event, the smallest magnitude whose CDF value is at least a
        uniform random number.
    """
    y = np.random.uniform(size=len(zones))
    mags = (y <= mfds.loc[zones].T).idxmax()
    return mags.values


def fitted_cdf_discrete(sample, mc, delta_m, x_max=None, beta=None):
    """CDF of the discretized Gutenberg-Richter law fitted to a sample.

    Parameters
    ----------
    sample : array-like
        Binned magnitudes.
    mc : float
        Completeness magnitude.
    delta_m : float
        Bin size.
    x_max : float, optional
        Largest magnitude to evaluate; default is the sample maximum.
    beta : float, optional
        If None, estimated from the sample with `estimate_beta_tinti`.

    Returns
    -------
    tuple of np.ndarray
        ``(x, y)``: magnitude bins and CDF values.
    """
    if beta is None:
        beta = estimate_beta_tinti(sample, mc=mc, delta_m=delta_m)

    if x_max is None:
        sample_bin_n = (sample.max() - mc) / delta_m
    else:
        sample_bin_n = (x_max - mc) / delta_m
    bins = np.arange(sample_bin_n + 1)
    cdf = 1 - np.exp(-beta * delta_m * (bins + 1))
    x, y = mc + bins * delta_m, cdf

    x, y_count = np.unique(x, return_counts=True)
    return x, y[np.cumsum(y_count) - 1]


def empirical_cdf(sample, weights=None):
    # accept pandas objects as well as arrays
    """Empirical (optionally weighted) CDF of a sample.

    Parameters
    ----------
    sample : array-like or pd.Series
    weights : array-like or pd.Series, optional

    Returns
    -------
    tuple of np.ndarray
        ``(x, y)``: sorted unique values and CDF values.
    """
    sample = getattr(sample, "values", sample)
    weights = getattr(weights, "values", weights)

    sample_idxs_sorted = np.argsort(sample)
    sample_sorted = sample[sample_idxs_sorted]
    if weights is not None:
        weights_sorted = weights[sample_idxs_sorted]
        x, y = sample_sorted, np.cumsum(weights_sorted) / weights_sorted.sum()
    else:
        x, y = sample_sorted, np.arange(1, len(sample) + 1) / len(sample)

    # only return one value per bin
    x, y_count = np.unique(x, return_counts=True)
    return x, y[np.cumsum(y_count) - 1]


def ks_test_gr(sample, mc, delta_m, ks_ds=None, n_samples=10000, beta=None):
    """Kolmogorov-Smirnov test of a sample against the fitted, discretized
    Gutenberg-Richter law.

    The p-value is obtained by simulation: the KS distance of the sample is
    compared to those of ``n_samples`` synthetic samples of the same size.

    Parameters
    ----------
    sample : array-like
        Binned magnitudes above ``mc``.
    mc : float
    delta_m : float
    ks_ds : list, optional
        Precomputed KS distances of synthetic samples (to reuse).
    n_samples : int, default 10000
    beta : float, optional
        If None, estimated from the sample.

    Returns
    -------
    tuple
        ``(original_ks_d, p_value, ks_ds)``.
    """
    sample = sample[sample >= mc - delta_m / 2]
    if len(sample) == 0:
        print("no sample")
        return 1, 0, []
    if len(np.unique(sample)) == 1:
        print("sample contains only one value")
        return 1, 0, []
    if beta is None:
        beta = estimate_beta_tinti(sample, mc=mc, delta_m=delta_m)

    if ks_ds is None:
        ks_ds = []

        n_sample = len(sample)
        simulated_all = round_half_up(
            simulate_magnitudes(
                mc=mc - delta_m / 2,
                beta=beta,
                n=n_samples * n_sample) / delta_m) * delta_m

        x_max = np.max(simulated_all)
        x_fit, y_fit = fitted_cdf_discrete(
            sample, mc=mc, delta_m=delta_m, x_max=x_max, beta=beta)

        for i in range(n_samples):
            simulated = simulated_all[n_sample * i:n_sample * (i + 1)].copy()
            x_emp, y_emp = empirical_cdf(simulated)
            y_fit_int = np.interp(x_emp, x_fit, y_fit)

            ks_d = np.max(np.abs(y_emp - y_fit_int))
            ks_ds.append(ks_d)
    else:
        x_fit, y_fit = fitted_cdf_discrete(
            sample, mc=mc, delta_m=delta_m, beta=beta)

    x_emp, y_emp = empirical_cdf(sample)
    y_emp_int = np.interp(x_fit, x_emp, y_emp)

    orig_ks_d = np.max(np.abs(y_fit - y_emp_int))

    return orig_ks_d, sum(ks_ds >= orig_ks_d) / len(ks_ds), ks_ds


def estimate_mc(sample,
                mcs_test,
                delta_m,
                p_pass,
                stop_when_passed=True,
                verbose=False,
                beta=None,
                n_samples=10000):
    """
    Estimates mc.

    Parameters
    ----------
    sample : np.array
        Magnitudes to test.
    mcs_test : np.array
        Completeness magnitudes to test.
    delta_m : float
        Magnitude bins (sample has to be rounded to bins beforehand).
    p_pass : float
        P-value with which the test is passed.
    stop_when_passed : bool
        Stop calculations when first mc passes the test.
    verbose : bool
        Verbose
    beta : float
        If beta is 'known', only estimate mc.
    n_samples : int
        Number of magnitude samples to be generated in p-value
        calculation of KS distance.
    """

    ks_ds = []
    ps = []
    i = 0
    for mc in mcs_test:
        if verbose:
            print('\ntesting mc', mc)
        ks_d, p, _ = ks_test_gr(
            sample, mc=mc, delta_m=delta_m, n_samples=n_samples, beta=beta)

        ks_ds.append(ks_d)
        ps.append(p)

        i += 1
        if verbose:
            print('..p-value: ', p)

        if p >= p_pass and stop_when_passed:
            break
    ps = np.array(ps)
    if np.any(ps >= p_pass):
        best_mc = mcs_test[np.argmax(ps >= p_pass)]
        if beta is None:
            beta = estimate_beta_tinti(
                sample[sample >= best_mc - delta_m / 2],
                mc=best_mc, delta_m=delta_m)
        if verbose:
            print("\n\nFirst mc to pass the test:",
                  best_mc, "\nwith a beta of:", beta)
    else:
        best_mc = None
        beta = None
        if verbose:
            print("None of the mcs passed the test.")

    return mcs_test, ks_ds, ps, best_mc, beta
