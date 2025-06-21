"""Provide unit tests for multivariate TE estimation."""


import numpy as np
import pytest
import copy
from test_estimators_jidt import _get_gauss_data
from idtxl.data import Data
from idtxl.multivariate_te import MultivariateTE

SEED = 0


def test_gauss_data():
    """Test nonlinear granger estimation from correlated Gaussians."""
    # Generate data and add a delay one one sample.
    expected_mi, source, source_uncorr, target = _get_gauss_data(seed=SEED)
    source = source[1:]
    source_uncorr = source_uncorr[1:]
    target = target[:-1]
    data = Data(
        np.hstack((source, source_uncorr, target)), dim_order="sp", normalise=False
    )
    settings = {
        "cmi_estimator": "JidtGaussianCMI",
        "n_perm_max_stat": 21,
        "n_perm_min_stat": 21,
        "n_perm_max_seq": 21,
        "n_perm_omnibus": 21,
        "max_lag_sources": 2,
        "min_lag_sources": 1,
        'noise_level': 0
    }
    # perform JidtGaussianCMI WITHOUT nonlinear data
    nw = MultivariateTE()
    results1 = nw.analyse_single_target(settings, data, target=2, sources=[0, 1])
    te1 = results1.get_single_target(2, fdr=False)["te"][0]
    sources1 = results1.get_target_sources(2, fdr=False)

    # prepare data object for nonlinear analysis
    settings["target"] = 2
    settings, data = data.prepare_nonlinear(settings, data)

    # perform JidtGaussianCMI WITH nonlinear data
    nonlin_analysis = MultivariateTE()
    results2 = nonlin_analysis.analyse_single_target(settings, data,
                                                    target=settings["nonlinear_settings"][
                                                        "nonlinear_target_predictors"],
                                                    sources=settings["nonlinear_settings"][
                                                        "nonlinear_source_predictors"])

    te2 = results2.get_single_target(2, fdr=False)["te"][0]
    sources2 = results2.get_target_sources(2, fdr=False)

    # Assert that only the correlated source was detected.
    assert sources1[0] == 0, "Wrong inferred source in standard estimation: {0}.".format(sources1[0])
    assert sources1[0] == 0, "Wrong inferred source in standard estimation: {0}.".format(sources1[0])
    assert sources2[0] == 0, "Wrong inferred source in nonlinear estimation: {0}.".format(sources2[0])
    assert sources2[0] == 0, "Wrong inferred source in nonlinear estimation: {0}.".format(sources2[0])

    # Assert that only the correlated source was detected.
    assert np.isclose(te1, expected_mi, atol=0.05), (
        "Estimated TE {0:0.6f} differs from expected TE {1:0.6f}.".format(te1, expected_mi))

    assert np.isclose(te2, expected_mi, atol=0.05), (
        "Estimated TE using nonlinear analysis {0:0.6f} differs from expected TE {1:0.6f}.".format(
            te2, expected_mi))

    assert np.isclose(te1, te2, atol=0.005), (
        "Standard estimated TE {0:0.6f} differs from estimated TE using nonlinear analysis {1:0.6f} (expected: "
        "MI {2:0.6f}).".format(te1, te2, expected_mi))

    #assert sources1[0] == sources2[0], (
    #    "sources of standard TE estimation {0:0.0f} differs from sources of nonlinear TE estimation {1:0.0f}."
    #        .format(sources1[0], sources2[0]))


def test_flags_and_result_output():
    """Test nonlinear granger estimation from nonlinear coupled AR processes."""
    data = Data(normalise=False)  # initialise an empty data object
    data.generate_nonlinear_data(n_samples=1000, n_replications=1)

    # prepare settings
    settings = {
        "target": 1,  # mandatory for analyse_single_target (nonlinear granger)
        "cmi_estimator": "JidtGaussianCMI",
        "n_perm_max_stat": 21,
        "n_perm_min_stat": 21,
        "n_perm_omnibus": 21,
        "n_perm_max_seq": 21,
        "max_lag_sources": 3,
        "min_lag_sources": 1,
    }

    # prepare data object for nonlinear analysis
    settings, data = data.prepare_nonlinear(settings, data)

    # check if flags in data and settings were set and the if the nonlinear_settings exist in the settings
    assert "nonlinear_prepared" in settings, "Flag nonlinear_prepared was not set in settings"
    assert "nonlinear_settings" in settings, "Nonlinear_settings were not addes to settings"
    assert data.get_nonlinear_status(), "Flag nonlinear_prepared was not set in data"

    # perform JidtGaussianCMI WITH nonlinear data
    nonlin_analysis = MultivariateTE()
    results = nonlin_analysis.analyse_single_target(settings, data,
                                                    target=settings["nonlinear_settings"][
                                                        "nonlinear_target_predictors"],
                                                    sources=settings["nonlinear_settings"][
                                                        "nonlinear_source_predictors"])

    single_target = results.get_single_target(1, fdr=False)

    # Test if all result outputs exist
    assert "performed_nonlinear_analysis" in single_target, \
        "Results do no contain 'performed_nonlinear_analysis'"
    assert "lin_and_nonlin_target_predictors_tested" in single_target, \
        "Results do no contain 'lin_and_nonlin_target_predictors_tested'"
    assert "lin_and_nonlin_sources_tested" in single_target, \
        "Results do no contain 'lin_and_nonlin_sources_tested'"
    assert "nonlinear_process_desc" in single_target, "Results do no contain 'nonlinear_process_desc'"
    assert "selected_vars_sources_type" in single_target, "Results do no contain 'selected_vars_sources_type'"
    assert "selected_vars_targets_type" in single_target, "Results do no contain 'selected_vars_targets_type'"
    assert "selected_vars_target_orig" in single_target, "Results do no contain 'selected_vars_target_orig'"
    assert "selected_vars_sources_orig" in single_target, "Results do no contain 'selected_vars_sources_orig'"

    # check transformation of selected squared sources back to original in selected_vars_...
    assert single_target.selected_vars_sources[0][0] < results.data_properties['n_nodes'],\
        "Selected var sources were not transformed back to number processes"
    assert single_target.selected_vars_target[0][0] < results.data_properties['n_nodes'],\
        "Selected var target were not transformed back to number processes"

    # check if expected squared source was found
    assert single_target.selected_vars_sources_orig[0][0] == 2, \
        "Nonlinear source was not detected in nonlinear data"
    assert single_target.selected_vars_sources_type[0] == "squared",\
        "Nonlinear source was not detected in nonlinear data"


def test_check_target_and_source_set():
    """Test the method _check_source_set.
    This method sets the list of source processes from which candidates are
    taken for multivariate TE estimation.
    """
    data = Data(seed=SEED, normalise=False)
    data.generate_mute_data(100, 5)
    # Add target and list of sources.
    target = 0
    sources = [1, 2, 3]
    settings = {
        "target": target,
        "sources": sources,
    }
    # prepare data object for nonlinear analysis
    settings, data = data.prepare_nonlinear(settings, data)

    # test target and sources after prepare_nonlinear
    nonlin_target = settings["nonlinear_settings"]["nonlinear_target_predictors"]
    assert nonlin_target[0] == target
    assert nonlin_target[1] == target + int(data.n_processes/2)
    nonlin_sources = settings["nonlinear_settings"]["nonlinear_source_predictors"]
    assert nonlin_sources[:len(nonlin_sources) // 2] == sources
    assert nonlin_sources[len(nonlin_sources) // 2:] == [i+int(data.n_processes/2) for i in sources]

    nw_0 = MultivariateTE()
    nw_0.settings = {"verbose": True}

    nw_0._check_lin_and_nonlin_targets(nonlin_target, data.n_processes)
    assert nw_0.target == target
    assert nw_0.targets == nonlin_target
    nw_0._check_source_set(nonlin_sources, data.n_processes)
    assert nw_0.source_set == nonlin_sources, "Sources were not added correctly."

    # Assert that initialisation fails if the target is also in the source list
    data = Data(seed=SEED, normalise=False)
    data.generate_mute_data(100, 5)
    sources = [0, 1, 2, 3]
    settings = {
        "target": target,
        "sources": sources,
    }

    with pytest.raises(RuntimeError):
        settings, data = data.prepare_nonlinear(settings, data)

    # Test if a single source, no list is added correctly.
    data = Data(seed=SEED, normalise=False)
    data.generate_mute_data(100, 5)
    # Add target and list of sources.
    target = 0
    sources = 1
    settings = {
        "target": target,
        "sources": sources,
    }
    # prepare data object for nonlinear analysis
    settings, data = data.prepare_nonlinear(settings, data)

    # check if targets are set correctly in the analysis
    nonlin_target = settings["nonlinear_settings"]["nonlinear_target_predictors"]
    assert nonlin_target[0] == target
    assert nonlin_target[1] == target + int(data.n_processes / 2)
    nw_0._check_lin_and_nonlin_targets(nonlin_target, data.n_processes)
    assert nw_0.target == target
    assert nw_0.targets == nonlin_target

    # check if sources are set correctly in the analysis
    nonlin_sources = settings["nonlinear_settings"]["nonlinear_source_predictors"]
    assert nonlin_sources[:len(nonlin_sources) // 2] == [sources]
    assert nonlin_sources[len(nonlin_sources) // 2:] == [sources + int(data.n_processes / 2)]
    nw_0._check_source_set(nonlin_sources, data.n_processes)
    assert type(nw_0.source_set) is list

    # Test if 'all' is handled correctly
    data = Data(seed=SEED, normalise=False)
    data.generate_mute_data(100, 5)
    # Add target and list of sources.
    target = 0
    sources = "all"
    settings = {
        "target": target,
        "sources": sources,
    }
    # prepare data object for nonlinear analysis
    settings, data = data.prepare_nonlinear(settings, data)

    # check if targets are set correctly in the analysis
    nonlin_target = settings["nonlinear_settings"]["nonlinear_target_predictors"]
    assert nonlin_target[0] == target
    assert nonlin_target[1] == target + int(data.n_processes / 2)
    nw_0._check_lin_and_nonlin_targets(nonlin_target, data.n_processes)
    assert nw_0.target == target
    assert nw_0.targets == nonlin_target

    # check if sources are set correctly in the analysis
    nonlin_sources = settings["nonlinear_settings"]["nonlinear_source_predictors"]
    assert nonlin_sources[:len(nonlin_sources) // 2] == [1, 2, 3, 4]
    assert nonlin_sources[len(nonlin_sources) // 2:] == [i+int(data.n_processes/2) for i in [1, 2, 3, 4]]
    # no need to nw_0._check_source_set("all", data.n_processes) - can not happen in nonlinear analysis


def test_nonlinear_result_functions():
    """Test nonlinear results function for nonlinear granger analysis."""
    data = Data(normalise=False)  # initialise an empty data object
    data.generate_nonlinear_data(n_samples=1000, n_replications=1)

    target = 1
    expected_source = 2
    expected_u = 2
    # prepare settings
    settings = {
        "target": target,  # mandatory for analyse_single_target (nonlinear granger)
        "cmi_estimator": "JidtGaussianCMI",
        "n_perm_max_stat": 21,
        "n_perm_min_stat": 21,
        "n_perm_omnibus": 21,
        "n_perm_max_seq": 21,
        "max_lag_sources": 3,
        "min_lag_sources": 1,
    }

    # prepare data object for nonlinear analysis
    settings, data = data.prepare_nonlinear(settings, data)

    # perform JidtGaussianCMI WITH nonlinear data
    nonlin_analysis = MultivariateTE()
    results = nonlin_analysis.analyse_single_target(settings, data,
                                                    target=settings["nonlinear_settings"][
                                                        "nonlinear_target_predictors"],
                                                    sources=settings["nonlinear_settings"][
                                                        "nonlinear_source_predictors"])

    # test output of nonlinear result functions
    target_sources = results.get_nonlinear_target_sources(target, fdr=False)
    assert target_sources == expected_source
    source_variables = results.get_nonlinear_source_variables(fdr=False)
    assert source_variables[0]["target"] == target
    assert source_variables[0]["selected_vars_sources"][0][0] == expected_source
    assert source_variables[0]["selected_vars_sources"][0][1] == expected_u

    criterion = ["max_te", "max_p"]
    for c in criterion:
        target_delays = results.get_nonlinear_target_delays(target, criterion=c, fdr=False)
        assert target_delays == expected_u

    weights = ["max_te_lag", "max_p_lag", "binary", "vars_count"]
    for w in weights:
        adj_mat = results.get_nonlinear_adjacency_matrix(w, fdr=False)
        assert adj_mat.type_matrix[0][1] > 0
        assert adj_mat.edge_matrix[0][1]


def test_nonlinear_network_analysis():
    """Test method for full network analysis."""

    # Test all to all analysis
    # -----------------------------------------------
    data = Data(seed=SEED, normalise=False)
    data.generate_mute_data(10, 5)
    settings = {
        "cmi_estimator": "JidtGaussianCMI",
        "n_perm_max_stat": 21,
        "n_perm_min_stat": 21,
        "n_perm_max_seq": 21,
        "n_perm_omnibus": 21,
        "max_lag_sources": 5,
        "min_lag_sources": 4,
        "max_lag_target": 5,
    }
    # prepare data object for nonlinear analysis
    settings, data = data.prepare_nonlinear(settings, data)

    nw_0 = MultivariateTE()
    results = nw_0.analyse_network(settings, data, targets="all", sources="all")

    # test analysed targets
    targets_analysed = results.targets_analysed
    assert all(
        np.array(targets_analysed) == np.arange(int(data.n_processes/2))
    ), "Network analysis did not run on all targets."

    # test sources per target
    sources = np.arange(data.n_processes)
    for t in results.targets_analysed:
        s = np.array(list(set(sources) - set([t, t+int(data.n_processes/2)])))
        assert all(
            np.array(results._single_target[t].sources_tested) == s
        ), f"Network analysis did not run on all sources for target {t}"

    # Test analysis for subset of targets
    # -----------------------------------------------
    data = Data(seed=SEED, normalise=False)
    data.generate_mute_data(10, 5)
    target_list = [1, 2, 3]
    settings = {
        "cmi_estimator": "JidtGaussianCMI",
        "n_perm_max_stat": 21,
        "n_perm_min_stat": 21,
        "n_perm_max_seq": 21,
        "n_perm_omnibus": 21,
        "max_lag_sources": 5,
        "min_lag_sources": 4,
        "max_lag_target": 5,
    }
    settings, data = data.prepare_nonlinear(settings, data)
    nw_0 = MultivariateTE()
    results = nw_0.analyse_network(settings, data,
                                    targets=target_list,
                                    sources="all")

    # check if nonlinear network analysis did run on all targets
    targets_analysed = results.targets_analysed
    assert all(
        np.array(targets_analysed) == np.array(target_list)
    ), "Network analysis did not run on correct subset of targets."

    # check if nonlinear network analysis did run on all sources
    for t in results.targets_analysed:
        s = np.array(list(set(sources) - set([t, t+int(data.n_processes/2)])))
        assert all(
            np.array(results._single_target[t].sources_tested) == s
        ), f"Network analysis did not run on all sources for target {t}"

    # Test analysis for subset of sources
    # -----------------------------------------------
    data = Data(seed=SEED, normalise=False)
    data.generate_mute_data(10, 5)
    source_list = [1, 2, 3]
    target_list = [0, 4]
    settings = {
        "cmi_estimator": "JidtGaussianCMI",
        "n_perm_max_stat": 21,
        "n_perm_min_stat": 21,
        "n_perm_max_seq": 21,
        "n_perm_omnibus": 21,
        "max_lag_sources": 5,
        "min_lag_sources": 4,
        "max_lag_target": 5,
    }
    settings, data = data.prepare_nonlinear(settings, data)
    nw_0 = MultivariateTE()
    results = nw_0.analyse_network(
        settings, data, targets=target_list, sources=source_list
    )

    # check if nonlinear network analysis did run on all targets
    targets_analysed = results.targets_analysed
    assert all(
        np.array(targets_analysed) == np.array(target_list)
    ), "Network analysis did not run for all targets."

    # check if nonlinear network analysis did run on all sources
    for t in results.targets_analysed:
        assert all(
            results._single_target[t].sources_tested == np.array(
                source_list + [i+int(data.n_processes/2) for i in source_list])
        ), f"Network analysis did not run on the correct subset of sources for target {t}"


def test_return_local_values():
    """Test estimation of local values."""
    max_lag = 5
    data = Data(seed=SEED, normalise=False)
    data.generate_nonlinear_data(n_samples=1000, n_replications=1)

    target = 1
    sources = 0

    settings = {
        "cmi_estimator": "JidtGaussianCMI",
        "noise_level": 0,
        "local_values": True,  # request calculation of local values
        "n_perm_max_stat": 21,
        "n_perm_min_stat": 21,
        "n_perm_max_seq": 21,
        "n_perm_omnibus": 21,
        "max_lag_sources": max_lag,
        "min_lag_sources": 4,
        "max_lag_target": max_lag,
        "target": target,
        "sources": sources
    }

    # prepare data object for nonlinear analysis
    settings, data = data.prepare_nonlinear(settings, data)

    te = MultivariateTE()
    results = te.analyse_single_target(settings, data,
                                                 target=settings["nonlinear_settings"]["nonlinear_target_predictors"],
                                                 sources=settings["nonlinear_settings"]["nonlinear_source_predictors"])
    settings["local_values"] = False
    results_avg = te.analyse_single_target(settings, data,
                                           target=settings["nonlinear_settings"]["nonlinear_target_predictors"],
                                           sources=settings["nonlinear_settings"]["nonlinear_source_predictors"])

    # Test if any sources were inferred. If not, return (this may happen
    # sometimes due to too few samples, however, a higher no. samples is not
    # feasible for a unit test).
    if results.get_single_target(target, fdr=False)["te"] is None:
        return
    if results_avg.get_single_target(target, fdr=False)["te"] is None:
        return

    lte = results.get_single_target(target, fdr=False)["te"]
    n_sources = len(results.get_nonlinear_target_sources(target, fdr=False))
    assert (
        type(lte) is np.ndarray
    ), "LTE estimation did not return an array of values: {0}".format(lte)
    assert (
        lte.shape[0] == n_sources
    ), "Wrong dim (no. sources) in LTE estimate: {0}".format(lte.shape)
    assert lte.shape[1] == data.n_realisations_samples(
        (0, max_lag)
    ), "Wrong dim (no. samples) in LTE estimate: {0}".format(lte.shape)
    assert (
        lte.shape[2] == data.n_replications
    ), "Wrong dim (no. replications) in LTE estimate: {0}".format(lte.shape)

    # Check if average and mean local values are the same. Test each source
    # separately. Inferred sources and variables may differ between the two
    # calls to analyse_single_target() due to low number of surrogates used in
    # unit testing.
    te_single_link = results_avg.get_single_target(target, fdr=False)["te"]
    sources_local = results.get_target_sources(target, fdr=False)
    sources_avg = results_avg.get_target_sources(target, fdr=False)
    for s in list(set(sources_avg).intersection(sources_local)):
        i1 = sources_avg[0]
        i2 = np.where(sources_local == s)[0][0]

        vars_local = [
            v
            for v in results_avg.get_single_target(
                target, fdr=False
            ).selected_vars_sources
            if v[0] == sources_avg
        ]
        vars_avg = [
            v
            for v in results.get_single_target(target, fdr=False).selected_vars_sources
            if v[0] == sources_local
        ]
        if vars_local != vars_avg:
            continue

        print(
            "Compare average ({0:.4f}) and local values ({1:.4f}).".format(
                    te_single_link[i1], np.mean(lte[i2, :, :])
            )
        )
        assert np.isclose(te_single_link[i1], np.mean(lte[i2, :, :]), rtol=0.00005), (
            "Single link average MI ({0:.6f}) and mean LMI ({1:.6f}) "
            " deviate.".format(te_single_link[i1], np.mean(lte[i2, :, :]))
        )


def test_add_conditional():
    """Enforce the conditioning on additional variables.
    Adding valid conditionals and test if they were added correctly (incl. nonlinear conds) in the network analysis
    """
    # generate data
    data = Data(seed=SEED, normalise=False)
    data.generate_mute_data()

    settings = {"target": 0,
                "sources": [1, 2],
                "cmi_estimator": "JidtGaussianCMI",
                "max_lag_sources": 5,
                "min_lag_sources": 3,
                "max_lag_target": 7,
                "add_conditionals": [(0, 1), (1, 3)]
                }

    # prepare data object for nonlinear analysis
    settings, data = data.prepare_nonlinear(settings, data)

    # initialise network_analysis
    nw = MultivariateTE()
    nw._initialise(settings=settings, data=data,
                   target=settings["nonlinear_settings"]["nonlinear_target_predictors"],
                   sources=settings["nonlinear_settings"]["nonlinear_source_predictors"])

    # Get list of conditionals after initialisation and convert absolute samples
    # back to lags for comparison.
    cond_list = nw._idx_to_lag(nw.selected_vars_full)
    assert (settings["add_conditionals"][0] in cond_list), \
        "First enforced conditional is missing from results."
    assert (settings["add_conditionals"][1] in cond_list), \
        "Second enforced conditional is missing from results."
    assert (settings["add_conditionals"][2] in cond_list), \
        "First nonlinear enforced conditional is missing from results."
    assert (settings["add_conditionals"][3] in cond_list), \
        "Second nonlinear enforced conditional is missing from results."


if __name__ == '__main__':
    #test_gauss_data()
    #test_flags_and_result_output()
    #test_check_target_and_source_set()
    test_nonlinear_result_functions()
    #test_nonlinear_network_analysis()
    #test_return_local_values()
    #test_add_conditional()

