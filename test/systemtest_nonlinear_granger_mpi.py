"""Run test on nonlinear data preparation and nonlinear JidtGaussianCMI estimation using MPI

strat script using (depending on you installed MPI implementation):
    mpirun -n 8 python systemtest_nonlinear_granger_mpi.py
    srun -n 8 python systemtest_nonlinear_granger_mpi.py
    mpiexec -n 8 python systemtest_nonlinear_granger_mpi.py
"""
import os
import time
import pickle
from pathlib import Path
import copy

from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data

start_time = time.time()

data = Data()  # initialise an empty data object
data.normalise = False
data.generate_nonlinear_data(n_samples=1000, n_replications=10)
data_nonlin = copy.deepcopy(data)

settings = {
    "MPI": True,
    "num_threads": 8,
    "target": 1,
    "sources": 0,
    "cmi_estimator": "JidtGaussianCMI",
    "n_perm_max_stat": 500,
    "n_perm_min_stat": 200,
    "n_perm_omnibus": 500,
    "n_perm_max_seq": 500,
    "max_lag_sources": 5,
    "min_lag_sources": 1,
}

# prepare data object for nonlinear analysis
settings_nonlin, data_nonlin = data.prepare_nonlinear(settings, data_nonlin)

# perform JidtGaussianCMI WITH nonlinear data
nonlinear_analysis1 = MultivariateTE()
results1 = nonlinear_analysis1.analyse_single_target(settings_nonlin, data_nonlin,
                                                     target=settings_nonlin["nonlinear_settings"]["nonlinear_all_targets"],
                                                     sources=settings_nonlin["nonlinear_settings"]["nonlinear_all_sources"])

path = Path(os.path.dirname(__file__)).joinpath("data")
with open(path.joinpath("test_nonlinear_granger_mpi.p"), "wb") as output_file:
    pickle.dump(results1, output_file)

runtime = time.time() - start_time
print("---- {0:.2f} minutes".format(runtime / 60))
