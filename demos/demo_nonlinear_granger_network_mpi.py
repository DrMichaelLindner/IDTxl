""" Demo of nonlinear granger network analysis

start script using (depending on your installed MPI implementation and your number of
    threads (-n <your num_threads>)):
    e.g. with num_threads 16:
        mpirun -n 16 python demo_nonlinear_granger_network_mpi.py 16
        srun -n 16 python demo_nonlinear_granger_network_mpi.py 16
        mpiexec -n 16 python demo_nonlinear_granger_network_mpi.py 16


Usage of nonlinear granger network analysis:

1) For nonlinear granger analysis the data need to be NOT normalised (for data.prepare_nonlinear)
and has to be in order: processes x samples x replications.
Hence, you should use the data function data.set_data(data, dimorder) to prepare your data.
    e.g.
        >>> data = Data(normalise=False)  # initialise an empty data object without normalisation
        >>> data.set_data(<your_data>, <your_dimorder>)

2) Specify the settings
    e.g.
    >>> settings = {
    >>> "cmi_estimator": "JidtGaussianCMI",
    >>> "max_lag_sources": 5,
    >>> "min_lag_sources": 1,
    >>> "MPI": max_workers > 0,
    >>> "max_workers": max_workers,
    }

3) Use data.prepare_nonlinear to on your data set. This function adds squared processes to data and overwrites the data
    >>> settings, data = data.prepare_nonlinear(settings, data)
        e.g.
            original data
                2 processes, n samples, m replications
                    target, n, m
                    source, n, m
            new data:
                4 processes, n samples, m replications
                    target, n, m
                    source, n, m
                    target^2, n, m
                    source^2, n, m
    The new settings and the data will now contain all infos and flags for the nonlinear granger network analysis.

4) Perform nonlinear granger network analysis using the new settings and data
        >>> network_analysis = MultivariateTE()
        >>> results = network_analysis.analyse_network(settings, data)

    The results of nonlinear granger analysis will contain additional informatio then the standard results:

        "lin_and_nonlin_target_predictors_tested": (list of tuples) all combinatios of lin and nonlin target
                                                    predictors and lags that were tested
        "lin_and_nonlin_sources_tested": (list of tuples) all combinatios of lin and nonlin target
                                                    predictors and lags that were tested
        "selected_vars_target_orig":    (list of tuples) all combinatios of lin and nonlin target
                                                    predictors and lags that were found
        "selected_vars_sources_orig":   (list of tuples) all combinatios of lin and nonlin sources and lags that
                                                    were found
        "selected_vars_target":         (list of tuples) all combinatios of target predictors and lags that were found.
                                                    A nonlinear target predictor - lag combination is replaced by the
                                                    original target predictor value
        "selected_vars_sources":        (list of tuples) all combinatios of sources and lags that were found.
                                                    A nonlinear source - lag combination is replaced by the
                                                    original source value
        "selected_vars_sources_type":   (list of strings) "orig" vs "squared" representing the type of the found
                                                    target predictor - lag combination
        "selected_vars_targets_type":   (list of strings) "orig" vs "squared" representing the type of the found
                                                    source - lag combination

        "nonlinear_process_desc": (list) n*2 x 2 infos sources vs targets and orig vs squared
                    e.g. [["target", "source", "target", "source"]
                          ["orig", "orig", "squared", "squared"]]

5) (optional) save the data

6) visualize results
    1) use results function print_nonlinear_edge_list
        e.g.
        >>> results.print_nonlinear_edge_list(weights="max_te_lag", fdr=False)
            0 -> 1, max_te_lag: 2, selected source type: 2
            0 -> 2, max_te_lag: 3, selected source type: 1
            0 -> 3, max_te_lag: 2, selected source type: 2
            3 -> 4, max_te_lag: 1, selected source type: 1
            4 -> 3, max_te_lag: 1, selected source type: 1
        For each detected source -> target you will get:
            - the max lag
            - the type of source that was used for the nonlinear analysis:
                1 = original process
                2 = squared process
    2) use function plot_netwerok (from visualise_graph)
        e.g.
            >>> plot_network(results=results, weights="max_te_lag", fdr=False)
        The numbers ontop of the adjacency matrix (on the right) represent the type of source that was
        used for the nonlinear analysis:
                1 = original process
                2 = squared process

"""

# Import classes
import sys
import time
import pickle
import matplotlib.pyplot as plt
from mpi4py import MPI

from idtxl.data import Data
from idtxl.multivariate_te import MultivariateTE
from idtxl.visualise_graph import plot_network


def main(args):

    assert MPI.COMM_WORLD.Get_rank() == 0
    max_workers = int(args[1])
    print(f"Running nonlinear granger network analysis with {max_workers} MPI workers.")

    start_time = time.time()

    # a) initialise an empty data object without normalisation
    data = Data(normalise=False)  # initialise an empty data object

    # b) add data
    # data.generate_mute_data(n_samples=1000, n_replications=5)
    data.generate_mute_data(n_samples=200, n_replications=5)

    # c) specify settings
    settings = {
        "cmi_estimator": "JidtGaussianCMI",
        "max_lag_sources": 5,
        "min_lag_sources": 1,
        "MPI": max_workers > 0,
        "max_workers": max_workers,
    }

    # d) prepare data object for nonlinear analysis
    settings, data = data.prepare_nonlinear(settings, data)

    # e) perform JidtGaussianCMI WITH nonlinear data
    network_analysis = MultivariateTE()
    results = network_analysis.analyse_network(settings, data)

    runtime = time.time() - start_time
    print("---- {0} minutes".format(runtime / 60))

    # f) (optional) Save results
    # pickle.dump(results, open('demo_nonlinear_granger_network_mpi_results.p', 'wb'))

    # g) plot edge list in terminal
    results.print_nonlinear_edge_list(weights="max_te_lag", fdr=False)

    # d) Plot inferred network to console and via matplotlib
    plot_network(results=results, weights="max_te_lag", fdr=False)
    plt.show()


if __name__ == "__main__":
    main(sys.argv)
