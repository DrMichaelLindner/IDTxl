"""Run test on nonlinear data preparation and nonlinear JidtGaussianCMI estimation using MPI

    ATTENTION:  For nonlinear analysis the data has to be in order: processes x samples x replications.
                You should use the data function data.set_data(data, dimorder) to prepare your data.

start script using (depending on your installed MPI implementation):
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
data.generate_nonlinear_data(n_samples=1000, n_replications=10)
data.normalise = False

settings = {
    "MPI": True,        # mandatory in settings for using MPI
    "num_threads": 8,   # mandatory in settings for using MPI
    "target": 1,        # mandatory in settings for nonlinear analysis
    "sources": 0,       # optional in settings for nonlinear analysis
    "cmi_estimator": "JidtGaussianCMI",
    "n_perm_max_stat": 500,
    "n_perm_min_stat": 200,
    "n_perm_omnibus": 500,
    "n_perm_max_seq": 500,
    "max_lag_sources": 5,
    "min_lag_sources": 1,
}

# prepare data object for nonlinear analysis
settings, data = data.prepare_nonlinear(settings, data)

# perform JidtGaussianCMI WITH nonlinear data
nonlin_analysis = MultivariateTE()
results = nonlin_analysis.analyse_single_target(settings, data,
                                                target=settings["nonlinear_settings"]["nonlinear_target_predictors"],
                                                sources=settings["nonlinear_settings"]["nonlinear_source_predictors"])

runtime = time.time() - start_time
print("---- {0:.2f} minutes".format(runtime / 60))

# Save results
# path = Path(os.path.dirname(__file__)).joinpath("data")
# with open(path.joinpath("test_nonlinear_granger_mpi.p"), "wb") as output_file:
#     pickle.dump(results, output_file)


