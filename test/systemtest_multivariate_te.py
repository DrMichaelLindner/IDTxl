"""System test for multivariate TE using the JIDT Kraskov estimator."""
import os
import random as rn
import numpy as np
from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data
from idtxl.idtxl_utils import calculate_mi


def test_multivariate_te_corr_gaussian(estimator=None):
    """Test multivariate TE estimation on correlated Gaussians.

    Run the multivariate TE algorithm on two sets of random Gaussian data with
    a given covariance. The second data set is shifted by one sample creating
    a source-target delay of one sample. This example is modeled after the
    JIDT demo 4 for transfer entropy. The resulting TE can be compared to the
    analytical result (but expect some error in the estimate).

    The simulated delay is 1 sample, i.e., the algorithm should find
    significant TE from sample (0, 1), a sample in process 0 with lag/delay 1.
    The final target sample should always be (1, 1), the mandatory sample at
    lat 1, because there is no memory in the process.

    Note:
        This test runs considerably faster than other system tests.
        This produces strange small values for non-coupled sources.  TODO
    """
    if estimator is None:
        estimator = "JidtKraskovCMI"

    n = 1000
    cov = 0.4
    source = [rn.normalvariate(0, 1) for r in range(n)]
    target = [
        sum(pair)
        for pair in zip(
            [cov * y for y in source],
            [(1 - cov) * y for y in [rn.normalvariate(0, 1) for r in range(n)]],
        )
    ]
    # Cast everything to numpy so the idtxl estimator understands it.
    source = np.expand_dims(np.array(source), axis=1)
    target = np.expand_dims(np.array(target), axis=1)

    data = Data(normalise=True)
    data.set_data(np.vstack((source[1:].T, target[:-1].T)), "ps")
    settings = {
        "cmi_estimator": estimator,
        "max_lag_sources": 5,
        "min_lag_sources": 1,
        "max_lag_target": 5,
        "n_perm_max_stat": 21,
        "n_perm_min_stat": 21,
        "n_perm_omnibus": 21,
        "n_perm_max_seq": 21,
    }
    random_analysis = MultivariateTE()
    results = random_analysis.analyse_single_target(settings, data, 1)

    # Assert that there are significant conditionals from the source for target
    # 1. For 500 repetitions I got mean errors of 0.02097686 and 0.01454073 for
    # examples 1 and 2 respectively. The maximum errors were 0.093841 and
    # 0.05833172 repectively. This inspired the following error boundaries.
    corr_expected = cov / (1 * np.sqrt(cov**2 + (1 - cov) ** 2))
    expected_res = calculate_mi(corr_expected)
    estimated_res = results.get_single_target(1, fdr=False).omnibus_te
    diff = np.abs(estimated_res - expected_res)
    print("Expected source sample: (0, 1)\nExpected target sample: (1, 1)")
    print(
        (
            "Estimated TE: {0:5.4f}, analytical result: {1:5.4f}, error:" "{2:2.2f} % "
        ).format(estimated_res, expected_res, diff / expected_res)
    )
    assert diff < 0.1, (
        "Multivariate TE calculation for correlated "
        "Gaussians failed (error larger 0.1: {0}, expected: "
        "{1}, actual: {2}).".format(diff, expected_res, estimated_res)
    )


def test_multivariate_te_lagged_copies():
    """Test multivariate TE estimation on a lagged copy of random data.

    Run the multivariate TE algorithm on two sets of random data, where the
    second set is a lagged copy of the first. This test should find no
    significant conditionals at all (neither in the target's nor in the
    source's past).

    Note:
        This test takes several hours and may take one to two days on some
        machines.
    """
    lag = 3
    d_0 = np.random.rand(1, 1000, 20)
    d_1 = np.hstack((np.random.rand(1, lag, 20), d_0[:, lag:, :]))

    data = Data()
    data.set_data(np.vstack((d_0, d_1)), "psr")
    settings = {
        "cmi_estimator": "JidtKraskovCMI",
        "max_lag_sources": 5,
        "n_perm_max_stat": 21,
        "n_perm_min_stat": 21,
        "n_perm_omnibus": 500,
        "n_perm_max_seq": 500,
    }
    random_analysis = MultivariateTE()
    # Assert that there are no significant conditionals in either direction
    # other than the mandatory single sample in the target's past (which
    # ensures that we calculate a proper TE at any time in the algorithm).
    for t in range(2):
        results = random_analysis.analyse_single_target(settings, data, t)
        assert (
            len(results.get_single_target(t, fdr=False).selected_vars_full) == 1
        ), "Conditional contains more/less than 1 variables."
        assert not results.get_single_target(
            t, fdr=False
        ).selected_vars_sources, "Conditional sources is not empty."
        assert (
            len(results.get_single_target(t, fdr=False).selected_vars_target) == 1
        ), "Conditional target contains more/less than 1 variable."
        assert (
            results.get_single_target(t, fdr=False).selected_sources_pval is None
        ), "Conditional p-value is not None."
        assert (
            results.get_single_target(t, fdr=False).omnibus_pval is None
        ), "Omnibus p-value is not None."
        assert (
            results.get_single_target(t, fdr=False).omnibus_sign is None
        ), "Omnibus significance is not None."
        assert (
            results.get_single_target(t, fdr=False).selected_sources_te is None
        ), "Conditional TE values is not None."


def test_multivariate_te_random():
    """Test multivariate TE estimation on two random data sets.

    Run the multivariate TE algorithm on two sets of random data with no
    coupling. This test should find no significant conditionals at all (neither
    in the target's nor in the source's past).

    Note:
        This test takes several hours and may take one to two days on some
        machines.
    """
    d = np.random.rand(2, 1000, 20)
    data = Data()
    data.set_data(d, "psr")
    settings = {
        "cmi_estimator": "JidtKraskovCMI",
        "max_lag_sources": 5,
        "min_lag_sources": 1,
        "n_perm_max_stat": 200,
        "n_perm_min_stat": 200,
        "n_perm_omnibus": 500,
        "n_perm_max_seq": 500,
    }
    random_analysis = MultivariateTE()
    # Assert that there are no significant conditionals in either direction
    # other than the mandatory single sample in the target's past (which
    # ensures that we calculate a proper TE at any time in the algorithm).
    for t in range(2):
        results = random_analysis.analyse_single_target(settings, data, t)
        assert (
            len(results.get_single_target(t, fdr=False).selected_vars_full) == 1
        ), "Conditional contains more/less than 1 variables."
        assert not results.get_single_target(
            t, fdr=False
        ).selected_vars_sources, "Conditional sources is not empty."
        assert (
            len(results.get_single_target(t, fdr=False).selected_vars_target) == 1
        ), "Conditional target contains more/less than 1 variable."
        assert (
            results.get_single_target(t, fdr=False).selected_sources_pval is None
        ), "Conditional p-value is not None."
        assert (
            results.get_single_target(t, fdr=False).omnibus_pval is None
        ), "Omnibus p-value is not None."
        assert (
            results.get_single_target(t, fdr=False).omnibus_sign is None
        ), "Omnibus significance is not None."
        assert (
            results.get_single_target(t, fdr=False).selected_sources_te is None
        ), "Conditional TE values is not None."


def test_multivariate_te_lorenz_2():
    """Test multivariate TE estimation on bivariately couled Lorenz systems.

    Run the multivariate TE algorithm on two Lorenz systems with a coupling
    from first to second system with delay u = 45 samples. Both directions are
    analyzed, the algorithm should not find a coupling from system two to one.

    Note:
        This test takes several hours and may take one to two days on some
        machines.
    """
    # load simulated data from 2 coupled Lorenz systems 1->2, u = 45 ms
    d = np.load(
        os.path.join(os.path.dirname(__file__), "data/lorenz_2_exampledata.npy")
    )
    data = Data()
    data.set_data(d, "psr")
    settings = {
        "cmi_estimator": "JidtKraskovCMI",
        "max_lag_sources": 47,
        "min_lag_sources": 42,
        "max_lag_target": 20,
        "tau_target": 2,
        "n_perm_max_stat": 21,  # 200
        "n_perm_min_stat": 21,  # 200
        "n_perm_omnibus": 21,
        "n_perm_max_seq": 21,  # this should be equal to the min stats b/c we
        # reuse the surrogate table from the min stats
    }
    lorenz_analysis = MultivariateTE()
    # FOR DEBUGGING: add the whole history for k = 20, tau = 2 to the
    # estimation, this makes things faster, b/c these don't have to be
    # tested again. Note conditionals are specified using lags.
    settings["add_conditionals"] = [
        (1, 19),
        (1, 17),
        (1, 15),
        (1, 13),
        (1, 11),
        (1, 9),
        (1, 7),
        (1, 5),
        (1, 3),
        (1, 1),
    ]

    settings["max_lag_sources"] = 60
    settings["min_lag_sources"] = 31
    settings["tau_sources"] = 2
    settings["max_lag_target"] = 1
    settings["tau_target"] = 1

    # Just analyse the direction of coupling
    results = lorenz_analysis.analyse_single_target(settings, data, target=1)
    print(results._single_target)
    adj_matrix = results.get_adjacency_matrix(weights="binary", fdr=False)
    adj_matrix.print_matrix()


def test_multivariate_te_mute():
    """Test multivariate TE estimation on the MUTE example network.

    Test data comes from a network that is used as an example in the paper on
    the MuTE toolbox (Montalto, PLOS ONE, 2014, eq. 14). The network has the
    following (non-linear) couplings:

    0 -> 1, u = 2
    0 -> 2, u = 3
    0 -> 3, u = 2 (non-linear)
    3 -> 4, u = 1
    4 -> 3, u = 1

    The maximum order of any single AR process is never higher than 2.
    """
    data = Data()
    data.generate_mute_data(n_samples=1000, n_replications=10)
    settings = {
        "cmi_estimator": "JidtKraskovCMI",
        "max_lag_sources": 3,
        "min_lag_sources": 1,
        "max_lag_target": 3,
        "n_perm_max_stat": 21,
        "n_perm_min_stat": 21,
        "n_perm_omnibus": 21,
        "n_perm_max_seq": 21,
    }  # this should be equal to the min stats b/c we
    # reuse the surrogate table from the min stats

    network_analysis = MultivariateTE()
    network_analysis.analyse_network(settings, data, targets=[1, 2])


def test_multivariate_te_multiple_runs():
    """Test TE estimation using multiple runs on the GPU.

    Test if data is correctly split over multiple runs, if the problem size
    exceeds the GPU global memory and thus requires multiple runs. Using a
    number of permutations of 7000 requires two runs on a GPU with global
    memory of about 6 GB.
    """
    data = Data()
    data.generate_mute_data(n_samples=1000, n_replications=10)
    settings = {
        "cmi_estimator": "OpenCLKraskovCMI",
        "max_lag_sources": 3,
        "min_lag_sources": 1,
        "max_lag_target": 3,
        "n_perm_max_stat": 7000,
        "n_perm_min_stat": 7000,
        "n_perm_omnibus": 21,
        "n_perm_max_seq": 21,
    }  # this should be equal to the min stats b/c we
    # reuse the surrogate table from the min stats

    network_analysis = MultivariateTE()
    network_analysis.analyse_network(settings, data, targets=[1, 2])


if __name__ == "__main__":
    test_multivariate_te_mute()
    test_multivariate_te_lorenz_2()
    test_multivariate_te_random()
    test_multivariate_te_lagged_copies()
    test_multivariate_te_multiple_runs()
    test_multivariate_te_corr_gaussian()
    test_multivariate_te_corr_gaussian("OpenCLKraskovCMI")
