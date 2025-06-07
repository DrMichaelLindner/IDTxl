"""Run test on nonlinear data preparation and nonlinear JidtGaussianCMI estimation in network_analysis

    ATTENTION:  For nonlinear granger analysis the data need to be NOT normalised (for data.prepare_nonlinear)
                and has to be in order: processes x samples x replications.
                Hence, you should use the data function data.set_data(data, dimorder) to prepare your data.
                e.g.
                    data = Data(normalise=False)  # initialise an empty data object without normalisation
                    data.set_data(<your_data>, <your_dimorder>)
"""
import pickle
import time
from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data

start_time = time.time()
data = Data(normalise=False)  # initialise an empty data object
data.generate_mute_data(n_samples=1000, n_replications=10)

settings = {
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
network_analysis = MultivariateTE()
results = network_analysis.analyse_network(settings, data)
runtime = time.time() - start_time
print("---- {0} minutes".format(runtime / 60))

# Save results
# pickle.dump(results, open('test_nonlinear_network_results.p', 'wb'))
