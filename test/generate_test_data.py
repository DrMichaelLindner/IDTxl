"""Providing test data generation functions for unit and system tests.

If running this script it generates test data for IDTxl network comparison 
unit and system tests:
Generate test data for IDTxl network comparison unit and system tests. Simulate
discrete and continous data from three correlated Gaussian data sets. Perform
network inference using bivariate/multivariate mutual information (MI)/transfer
entropy (TE) analysis. Results are saved used for unit and system testing of
network comparison (systemtest_network_comparison.py).
A coupling is simulated as a lagged, linear correlation between three Gaussian
variables and looks like this:
    1 -> 2 -> 3  with a delay of 1 sample for each coupling

"""
import pickle
from pathlib import Path
import numpy as np
import random as rn

from idtxl.multivariate_te import MultivariateTE
from idtxl.bivariate_te import BivariateTE
from idtxl.multivariate_mi import MultivariateMI
from idtxl.bivariate_mi import BivariateMI
from idtxl.estimators_jidt import JidtDiscreteCMI
from idtxl.estimators_python import PythonDiscreteCMI
from idtxl.data import Data
from idtxl.idtxl_utils import calculate_mi

# path = os.path.join(os.path.dirname(__file__) + '/data/')
path = Path("data/")


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




def generate_discrete_idtxl_data(n_samples=10000, n_replications=1):
    """Generate Gaussian test data: 1 -> 2 -> 3, delay 1."""
    d = generate_gauss_data(n_samples=n_samples, n_replications=n_replications, discrete=True)
    data = Data(d, dim_order="psr", normalise=False)
    return data


def generate_continuous_idtxl_data(n_samples=10000, n_replications=1):
    """Generate Gaussian test data: 1 -> 2 -> 3, delay 1."""
    d = generate_gauss_data(n_samples=n_samples, n_replications=n_replications, discrete=False)
    data = Data(d, dim_order="psr", normalise=True)
    return data


def generate_gauss_data(n_samples=10000, n_replications=1, discrete=False):
    settings = {"discretise_method": "equal", "n_discrete_bins": 5}
    est = JidtDiscreteCMI(settings)
    covariance_1 = 0.4
    covariance_2 = 0.3
    delay = 1
    if discrete:
        d = np.zeros((3, n_samples - 2 * delay, n_replications), dtype=int)
    else:
        d = np.zeros((3, n_samples - 2 * delay, n_replications))
    for r in range(n_replications):
        proc_1 = np.random.normal(0, 1, size=n_samples)
        proc_2 = covariance_1 * proc_1 + (1 - covariance_1) * np.random.normal(
            0, 1, size=n_samples
        )
        proc_3 = covariance_2 * proc_2 + (1 - covariance_2) * np.random.normal(
            0, 1, size=n_samples
        )
        proc_1 = proc_1[(2 * delay) :]
        proc_2 = proc_2[delay:-delay]
        proc_3 = proc_3[: -(2 * delay)]

        if discrete:  # discretise data
            proc_1_dis, proc_2_dis = est._discretise_vars(var1=proc_1, var2=proc_2)
            proc_1_dis, proc_3_dis = est._discretise_vars(var1=proc_1, var2=proc_3)
            d[0, :, r] = proc_1_dis
            d[1, :, r] = proc_2_dis
            d[2, :, r] = proc_3_dis
        else:
            d[0, :, r] = proc_1
            d[1, :, r] = proc_2
            d[2, :, r] = proc_3
    return d




def analyse_mute_te_data():
    # Generate example data: the following was ran once to generate example
    # data, which is now in the data sub-folder of the test-folder.
    data = Data()
    data.generate_mute_data(100, 5)
    # analysis settings
    settings = {
        "cmi_estimator": "JidtKraskovCMI",
        "n_perm_max_stat": 50,
        "n_perm_min_stat": 50,
        "n_perm_omnibus": 200,
        "n_perm_max_seq": 50,
        "max_lag_target": 5,
        "max_lag_sources": 5,
        "min_lag_sources": 1,
        "permute_in_time": True,
    }
    # network inference for individual data sets
    nw = MultivariateTE()
    list_of_targets = [[0, 1], [1, 2], [0, 2], [0, 1, 2], [1, 2]]
    for i, targets in enumerate(list_of_targets):
        res = nw.analyse_network(settings, data, targets=targets, sources="all")
        with open(path.joinpath(f"mute_results_{i}.p"), "wb") as output_file:
            pickle.dump(res, output_file)
    res = nw.analyse_network(settings, data)
    with open(path.joinpath("mute_results_full.p"), "wb") as output_file:
        pickle.dump(res, output_file)



def analyse_discrete_data():
    """Run network inference on discrete data."""
    data = generate_discrete_idtxl_data()
    settings = {
        "cmi_estimator": "JidtDiscreteCMI",
        "discretise_method": "none",
        "n_discrete_bins": 5,  # alphabet size of the variables analysed
        "min_lag_sources": 1,
        "max_lag_sources": 3,
        "max_lag_target": 1,
    }

    nw = MultivariateTE()
    res = nw.analyse_network(settings=settings, data=data)
    pickle.dump(
        res,
        open(
            path.joinpath(f"discrete_results_mte_{settings["cmi_estimator"]}.p"), "wb"
            #"{0}discrete_results_mte_{1}.p".format(path, settings["cmi_estimator"]), "wb",
        ),
    )

    nw = BivariateTE()
    res = nw.analyse_network(settings=settings, data=data)
    pickle.dump(
        res,
        open(
            path.joinpath(f"discrete_results_bte_{settings["cmi_estimator"]}.p"), "wb"
            #"{0}discrete_results_bte_{1}.p".format(path, settings["cmi_estimator"]), "wb",
        ),
    )

    nw = MultivariateMI()
    res = nw.analyse_network(settings=settings, data=data)
    pickle.dump(
        res,
        open(
            path.joinpath(f"discrete_results_mmi_{settings["cmi_estimator"]}.p"), "wb"
            #"{0}discrete_results_mmi_{1}.p".format(path, settings["cmi_estimator"]), "wb",
        ),
    )

    nw = BivariateMI()
    res = nw.analyse_network(settings=settings, data=data)
    pickle.dump(
        res,
        open(
            path.joinpath(f"discrete_results_bmi_{settings["cmi_estimator"]}.p"), "wb"
            #"{0}discrete_results_bmi_{1}.p".format(path, settings["cmi_estimator"]), "wb",
        ),
    )


def analyse_continuous_data():
    """Run network inference on continuous data."""
    data = generate_continuous_idtxl_data()
    settings = {"min_lag_sources": 1, "max_lag_sources": 3, "max_lag_target": 1}

    nw = MultivariateTE()
    for estimator in ["JidtGaussianCMI", "JidtKraskovCMI"]:
        settings["cmi_estimator"] = estimator
        res = nw.analyse_network(settings=settings, data=data)
        pickle.dump(
            res, open(
                path.joinpath(f"continuous_results_mte_{estimator}.p"), "wb"
                #"{0}continuous_results_mte_{1}.p".format(path, estimator), "wb"
                )
        )

    nw = BivariateTE()
    for estimator in ["JidtGaussianCMI", "JidtKraskovCMI"]:
        settings["cmi_estimator"] = estimator
        res = nw.analyse_network(settings=settings, data=data)
        pickle.dump(
            res, open(
                path.joinpath(f"continuous_results_bte_{estimator}.p"), "wb"
                #"{0}continuous_results_bte_{1}.p".format(path, estimator), "wb"
                )
        )

    nw = MultivariateMI()
    for estimator in ["JidtGaussianCMI", "JidtKraskovCMI"]:
        settings["cmi_estimator"] = estimator
        res = nw.analyse_network(settings=settings, data=data)
        pickle.dump(
            res, open(
                path.joinpath(f"continuous_results_mni_{estimator}.p"), "wb"
                #"{0}continuous_results_mmi_{1}.p".format(path, estimator), "wb"
                )
        )

    nw = BivariateMI()
    for estimator in ["JidtGaussianCMI", "JidtKraskovCMI"]:
        settings["cmi_estimator"] = estimator
        res = nw.analyse_network(settings=settings, data=data)
        pickle.dump(
            res, open(
                path.joinpath(f"continuous_results_bmi_{estimator}.p"), "wb"
                #"{0}continuous_results_bmi_{1}.p".format(path, estimator), "wb"
                )
        )


def assert_results():
    for algo in ["mmi", "mte", "bmi", "bte"]:
        # Test continuous data:
        for estimator in ["JidtGaussianCMI", "JidtKraskovCMI"]:
            res = pickle.load(
                open("data/continuous_results_{0}_{1}.p".format(algo, estimator), "rb")
            )
            print("\nInference algorithm: {0} (estimator: {1})".format(algo, estimator))
            _print_result(res)

        # Test discrete data:
        estimator = "JidtDiscreteCMI"
        res = pickle.load(
            open("data/discrete_results_{0}_{1}.p".format(algo, estimator), "rb")
        )
        print("\nInference algorithm: {0} (estimator: {1})".format(algo, estimator))
        _print_result(res)


def _print_result(res):
    adjacency_matrix = res.get_adjacency_matrix(weights="max_te_lag")
    #res.adjacency_matrix.print_matrix()
    adjacency_matrix.print_matrix()
    tp = 0
    fp = 0
    if adjacency_matrix.edge_matrix[0, 1] == True:
        tp += 1
    if adjacency_matrix.edge_matrix[1, 2] == True:
        tp += 1
    if adjacency_matrix.edge_matrix[0, 2] == True:
        fp += 1
    fn = 2 - tp
    print("TP: {0}, FP: {1}, FN: {2}".format(tp, fp, fn))


if __name__ == "__main__":
    analyse_discrete_data()
    analyse_mute_te_data()
    analyse_continuous_data()
    assert_results()
