"""Run test on nonlinear data preparation and nonlinear JidtGaussianCMI and PythonGaussianCMI estimation

    ATTENTION:  For nonlinear granger analysis the data need to be NOT normalised (for data.prepare_nonlinear)
                and has to be in order: processes x samples x replications.
                Hence, you should use the data function data.set_data(data, dimorder) to prepare your data.
                e.g.
                    data = Data(normalise=False)  # initialise an empty data object without normalisation
                    data.set_data(<your_data>, <your_dimorder>)
"""

import time
import pickle
from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data
import copy

start_time = time.time()

samples = 100
reps = 3

data = Data(normalise=False)  # initialise an empty data object
data.generate_nonlinear_data(n_samples=samples, n_replications=reps)

data2 = copy.deepcopy(data)

settings = {
    "target": 1,   # mandatory in settings for nonlinear single target analysis
    "sources": 0,  # optional in settings for nonlinear  single targetanalysis
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



start_time = time.time()

#data = Data(normalise=False)  # initialise an empty data object
#data.generate_nonlinear_data(n_samples=1000, n_replications=10)

settings2 = {
    "target": 1,   # mandatory in settings for nonlinear single target analysis
    "sources": 0,  # optional in settings for nonlinear  single targetanalysis
    "cmi_estimator": "PythonGaussianCMI",
    "n_perm_max_stat": 500,
    "n_perm_min_stat": 200,
    "n_perm_omnibus": 500,
    "n_perm_max_seq": 500,
    "max_lag_sources": 5,
    "min_lag_sources": 1,
}

# prepare data object for nonlinear analysis
settings, data2 = data2.prepare_nonlinear(settings2, data2)

# perform PythonGaussianCMI WITH nonlinear data
nonlin_analysis = MultivariateTE()
results = nonlin_analysis.analyse_single_target(settings2, data2,
                                                 target=settings["nonlinear_settings"]["nonlinear_target_predictors"],
                                                 sources=settings["nonlinear_settings"]["nonlinear_source_predictors"])

runtime = time.time() - start_time
print("---- {0:.2f} minutes".format(runtime / 60))
