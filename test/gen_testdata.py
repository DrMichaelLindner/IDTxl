"""Providing sets of code to generate test data for unit- and systemtests"""

import numpy as np
from idtxl.idtxl_utils import calculate_mi
import random as rn
from idtxl.estimators_python import PythonDiscreteCMI
from idtxl.data import Data

def _get_gauss_data(n=10000, covariance=0.4, expand=True, seed=None):
    """Generate correlated and uncorrelated Gaussian variables.

    Generate two sets of random normal data, where one set has a given
    covariance and the second is uncorrelated.
    """
    np.random.seed(seed)
    corr_expected = covariance / (1 * np.sqrt(covariance**2 + (1-covariance)**2))
    expected_mi = calculate_mi(corr_expected)
    src_corr = [rn.normalvariate(0, 1) for r in range(n)]  # correlated src
    src_uncorr = [rn.normalvariate(0, 1) for r in range(n)]  # uncorrelated src
    target = [sum(pair) for pair in zip(
                    [covariance * y for y in src_corr[0:n]],
                    [(1-covariance) * y for y in [
                        rn.normalvariate(0, 1) for r in range(n)]])]
    # Make everything numpy arrays so jpype understands it. Add an additional
    # axis if requested (MI/CMI estimators accept 2D arrays, TE/AIS only 1D).
    if expand:
        src_corr = np.expand_dims(np.array(src_corr), axis=1)
        src_uncorr = np.expand_dims(np.array(src_uncorr), axis=1)
        target = np.expand_dims(np.array(target), axis=1)
    else:
        src_corr = np.array(src_corr)
        src_uncorr = np.array(src_uncorr)
        target = np.array(target)
    return expected_mi, src_corr, src_uncorr, target

def _get_ar_data(n=10000, expand=False, seed=None):
    """Simulate a process with memory using an AR process of order 2.

    Return data with memory and random data without memory.
    """
    order = 2
    source1 = np.zeros(n + order)
    source1[0:order] = np.random.normal(size=(order))
    term_1 = 0.95 * np.sqrt(2)
    for n in range(order, n + order):
        source1[n] = (term_1 * source1[n - 1] - 0.9025 * source1[n - 2] +
                      np.random.normal())
    source2 = np.random.randn(n + order)
    if expand:
        return np.expand_dims(source1, axis=1), np.expand_dims(source2, axis=1)
    else:
        return source1, source2

def _get_freq_data(sample_rate=10000, duration=1.0, hz=40, lag=0, noise=0.2, seed=None):
	"""Generate correlated and uncorrelated Frequency variables."""
	
	np.random.seed(seed)

	n_samples = int(sample_rate * duration)
	t = np.linspace(0, duration, n_samples, endpoint=False)
	
	# create signal with noise
	signal1 = np.sin(2 * np.pi * hz * t) + noise * np.random.randn(n_samples)
	signal2 = np.sin(2 * np.pi * hz * t) + noise * np.random.randn(n_samples)
	signal3 = np.sin(2 * np.pi * hz*0.7 * t) + noise * np.random.randn(n_samples)

	# add lag
	n = n_samples - lag
	signal1 = signal1[:n]
	signal2 = signal2[lag:]
	signal3 = signal3[lag:]

	return signal1, signal2, signal3

def _get_mem_binary_data(n=10000, expand=False):
    """Simulate simple binary process with memory.

    Return data with memory and random data without memory.
    """
    source1 = np.zeros(n + 2)
    source1[0:2] = np.random.randint(2, size=(2))
    for n in range(2, n + 2):
        source1[n] = np.logical_xor(source1[n - 1], np.random.rand() > 0.15)
    source1 = source1.astype(int)
    source2 = np.random.randint(2, size=(n + 2))


    if expand:
        return np.expand_dims(source1[:n], axis=1), np.expand_dims(source2[:n], axis=1)
    else:
        return source1[:n], source2[:n]

###################################################################### TODO
def _get_cte_test_data(n=10000):
	"""Generate example data for testing cte

	0 -> 1, u = 2, 0,95 
    1 -> 2, u = 1, 0,95
    3 -> 2, u = 1,
    4 rand
	"""
	x = np.zeros((4, n + 3, n_replications))




def _get_discrete_gauss_data(
    covariance=0.4, n=10000, delay=1, normalise=False, seed=None):

    # Generate two coupled Gaussian time series
    np.random.seed(seed)
    source = np.random.normal(0, 1, size=n)
    target = covariance * source + (1 - covariance) * np.random.normal(0, 1, size=n)
    source = source[delay:]
    target = target[:-delay]

    # Discretise data for speed
    settings = {"discretise_method": "equal", "n_discrete_bins": 5}
    est = PythonDiscreteCMI(settings)
    source_dis, target_dis = est._discretise_vars(var1=source, var2=target)
    return Data(
        np.vstack((source_dis, target_dis)), dim_order="ps", normalise=normalise
    )


def _generate_mute_data(n_samples=10000, n_replications=10):
    """Generate example data for a 6-process network.
	
    0 -> 1, u = 2 (non-linear)
    0 -> 2, u = 3
    0 -> 3, u = 2 (non-linear)
    3 -> 4, u = 1
    4 -> 3, u = 1
	
	5 is pure noise
    """
    n_processes = 6

    x = np.zeros((n_processes, n_samples + 3, n_replications))
    x[:, 0:3, :] = np.random.normal(size=(n_processes, 3, n_replications))
    term_1 = 0.95 * np.sqrt(2)
    term_2 = 0.25 * np.sqrt(2)
    term_3 = -0.25 * np.sqrt(2)
    for r in range(n_replications):
        for n in range(3, n_samples + 3):
            x[0, n, r] = (
                term_1 * x[0, n - 1, r]
                - 0.9025 * x[0, n - 2, r]
                + np.random.normal()
            )
            x[1, n, r] = 0.5 * x[0, n - 2, r] ** 2 + np.random.normal()
            x[2, n, r] = -0.4 * x[0, n - 3, r] + np.random.normal()
            x[3, n, r] = (
                -0.5 * x[0, n - 2, r] ** 2
                + term_2 * x[3, n - 1, r]
                + term_2 * x[4, n - 1, r]
                + np.random.normal()
            )
            x[4, n, r] = (
                term_3 * x[3, n - 1, r]
                + term_2 * x[4, n - 1, r]
                + np.random.normal()
            )
            x[5, n, r] = np.random.normal()

    return x



