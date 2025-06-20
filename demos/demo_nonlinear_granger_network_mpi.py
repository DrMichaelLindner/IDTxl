""" Demo of nonlinear granger network analysis

start script using (depending on your installed MPI implementation and your number of
    threads (-n <your num_threads>)):
    e.g. with num_threads 16:
        mpirun -n 16 python demo_nonlinear_granger_network_mpi.py 16
        srun -n 16 python demo_nonlinear_granger_network_mpi.py 16
        mpiexec -n 16 python demo_nonlinear_granger_network_mpi.py 16
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
    data.generate_mute_data(n_samples=1000, n_replications=5)

    # c) specify settings
    settings = {
        "cmi_estimator": "JidtGaussianCMI",
        "n_perm_max_stat": 500,
        "n_perm_min_stat": 200,
        "n_perm_omnibus": 500,
        "n_perm_max_seq": 500,
        "max_lag_sources": 5,
        "min_lag_sources": 1,
        "MPI": max_workers > 0,  # mandatory in settings for using MPI
        "max_workers": max_workers,  # mandatory in settings for using MPI
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
