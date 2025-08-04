"""
provide structure an functions for getting distribution 
likelihood of the given data
"""

import numpy as np
import scipy
from scipy.special import gamma, digamma
from scipy.stats import entropy, differential_entropy, kstest
import pandas as pd
import joblib
from joblib.parallel import Parallel, delayed
from tqdm import tqdm
import multiprocessing
import threading
import contextlib
import warnings


# A solution to wrap joblib parallel call in tqdm from
# https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution/58936697#58936697
# and https://github.com/louisabraham/tqdm_joblib
@contextlib.contextmanager
def tqdm_joblib(*args, **kwargs):
    """Context manager to patch joblib to report into tqdm progress bar
    given as argument"""

    tqdm_object = tqdm(*args, **kwargs)

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


class get_distribution_likelihood():
    """

    Args:
    data : idtxl data object
    bins : int [optional] - number of bins for fitting (default=100)
    distributions : string or list [optional]
                    "all"
                        tests all available distributions of the fitter toolbox
                    "common" (default)
                        tests common distributions of the fitter toolbox
                    "<your distribution>"
                        tests only this distribution
                        e.g. "norm" for normal distribution
                    list of distributions you want to test
                        tests selected distributions in this list

    For getting all distributions you can test here use e.g.:
        >>> dl = get_distribution_likelihood
        >>> dl.show_distributions

    """

    def __init__(self,
                 data=None,
                 bins=100,
                 distributions=None):

        # get data
        if not data:
            raise RuntimeError("data not specified. ")
        self.data = data.data
        self.n_processes = data.n_processes
        self.n_replications = data.n_replications

        self.all_distributions = self.get_fit_distributions()

        # Check and set dist inputs
        self._check_dist_input(distributions)

        self.bins = bins

        # get min and max
        self._xmin = self.data.min(1)
        self._xmax = self.data.max(1)

        self.n_best = 5

    def _check_dist_input(self, dists):
        """check if all specified dstributions are in the list"""
        if isinstance(dists, list):
            for d in dists:
                if d not in self.all_distributions:
                    raise RuntimeError(f"The distribution {d} from the list is not in the list of possible "
                                       f"distribtutions. \nCall:\n     >>> "
                                       f"get_distribution_likelihood.show_distributions()\nto get "
                                       f"a list of all possible distributions.")
            self.distributions = dists
        elif dists == "common":
            self.distributions = self.get_common_distributions()
        elif dists == "all":
            self.distributions = self.get_fit_distributions()
        elif isinstance(dists, str):
            if dists != "all" and dists != "common":
                if dists not in self.all_distributions:
                    raise RuntimeError(f"The distribution {dists} is not in the list of possible distribtutions. \n"
                                       f"Call:\n     >>> get_distribution_likelihood.show_distributions()\nto"
                                       f" get a list of all possible distributions.")
                else:
                    self.distributions = [dists]
        else:
            raise RuntimeError(f"The distri")

    def _check_processes(self, processes):
        """check if processes are valid"""
        if isinstance(processes, list):
            if max(processes) > self.n_processes-1:
                raise RuntimeError(f"The specified process {max(processes)} is higher then the number "
                                   f"of processes in the data {self.n_processes}.")
            else:
                procs = processes
        elif isinstance(processes, int):
            if processes > self.n_processes-1:
                raise RuntimeError(f"The specified process {processes} is higher then the number "
                                   f"of processes in the data {self.n_processes}.")
            else:
                procs = [processes]
        elif processes == "all":
            procs = range(self.n_processes)
        else:
            raise RuntimeError(f"Incorrect input for processes {processes}. It need to be an int, "
                               f"a list of ints or \"all\".")

        return procs

    def _update_data_pdf(self, process, replication=None):
        # histogram retuns X with N+1 values. So, we rearrange the X output into only N
        if replication is None:
            self.y, self.x = np.histogram(self.data[process, :, :], bins=self.bins, density=True)
        else:
            self.y, self.x = np.histogram(self.data[process, :, replication], bins=self.bins, density=True)
        self.x = [(this + self.x[i + 1]) / 2.0 for i, this in enumerate(self.x[0:-1])]

    def _trim_data(self, process, replication=None):
        if replication is None:
            dat = self.data[process, :, :]
            self._data = dat[np.logical_and(dat >= self._xmin[process, :].min(),
                                            dat <= self._xmax[process, :].min())]
        else:
            dat = self.data[process, :, replication]
            self._data = dat[np.logical_and(dat >= self._xmin[process, replication],
                                            dat <= self._xmax[process, replication])]

    def get_fit_distributions(self):
        distributions = []
        for i in dir(scipy.stats):
            if "fit" in eval("dir(scipy.stats." + i + ")"):
                distributions.append(i)
        distributions = distributions[1:]
        return distributions

    def get_common_distributions(self):
        distributions = self.get_fit_distributions()
        common = [
            "cauchy",
            "chi2",
            "expon",
            "exponpow",
            "gamma",
            "lognorm",
            "norm",
            "powerlaw",
            "rayleigh",
            "uniform",
        ]
        common = [x for x in common if x in distributions]
        return common

    def show_distributions(self):
        print(self.all_distributions)
        print(f"You can specify a list of these distributions or a single one you to test.")

    def fit_single_distribution(self, distribution, data, x, y):

        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            # get scipy distribution
            dist = eval("scipy.stats." + distribution)

            # fit
            param = self._with_timeout(dist.fit, args=(self._data,)) #, timeout=timeout)

            # get pdf
            pdf_fitted = dist.pdf(x, *param)

            # calculate error
            sq_error = np.sum((pdf_fitted - y) ** 2)

            # calculate information criteria
            logLik = np.sum(dist.logpdf(x, *param))
            k = len(param[:])
            n = len(data)
            aic = 2 * k - 2 * logLik

            #if distribution == "gaussian":
            #    bic = n * np.log(sq_error / n) + k * np.log(n)
            #else:
            bic = k * np.log(n) - 2 * logLik

            # calculate kullback leibler divergence
            kl_div = entropy(pdf_fitted, y)

            # calculate goodness-of-fit statistic
            dist_fitted = dist(*param)
            ks_stat, ks_pval = kstest(data, dist_fitted.cdf)

            return distribution, (param, pdf_fitted, sq_error, aic, bic, kl_div, ks_stat, ks_pval)
        except Exception:
            return distribution, None

    def fit(self, mode="over_all_replications", processes="all", max_workers=-1, prefer="processes"):
        """ get the likelihoods of distributions of the given data

            Args:
            mode : string [optional]
                "over_all_replications" (default)
                    fits data over all replications of a process
                "per_replication"
                    fits data for each replication of a process


            Returns an object containing:

                .n_data_proceses
                .d_data_replications
                .results: List
                    Depending on the used mode this functions returns a list of size
                        nr data processes (mode="over_all_replications")
                        nr [data processes][nr data replications] (mode="per_replications")

                    Each element of the list contains a structure contains the following information:
                        "params" - fitted parameters
                        "pdf_fitted" - fitted pdf
                        "sse": sum squared error
                        "aic": aic
                        "bic": bic
                        "kldiv": Kullback-Leibler divergence
                        "ks_stat": ks statistic
                        "pval": ks p-value
                        "summary": summary of stats in panda dataframe


                    In case of specifying only a subset of processes to test, the not tested processes will
                    contain "Not tested" in the output

            The Object provides a set of functions:

                .print_summary()
                    print results of all processes (and replications) in console
                .get_all_dists(<process>)
                    get list of the the 5 best fitted distribution names and a list with their p-values
                    dist, pval = dl.get_all_dists(<process>)
                .get_best_dist(<process>)
                    get the name of the best fitted distribution
                .get_best_pval(<process>)
                    get the pval of the best fitted distribution

        """

        # check processes
        self.processes = self._check_processes(processes)

        # initialise distribution likelihood output
        dl = distribution_likelihoods_results(n_processes=self.n_processes,
                                              n_replications=self.n_replications,
                                              mode=mode)

        # get distribution fits
        for p in self.processes:
            print(f"Process {p}:")
            if mode == "over_all_replications":

                _fitted_param = {}
                _fitted_pdf = {}
                _fitted_errors = {}
                _aic = {}
                _bic = {}
                _kldiv = {}
                _ks_stat = {}
                _ks_pval = {}

                self._trim_data(p)
                self._update_data_pdf(p)

                N = len(self.distributions)
                with tqdm_joblib(desc=f"Fitting {N} distributions", total=N, disable=True) as progress_bar:
                    results = Parallel(n_jobs=max_workers, prefer=prefer)(
                        delayed(self.fit_single_distribution)(dist, self._data, self.x, self.y) for
                        dist in self.distributions
                    )

                for distribution, values in results:
                    if values is not None:
                        param, pdf_fitted, sq_error, aic, bic, kullback_leibler, ks_stat, ks_pval = values

                        _fitted_param[distribution] = param
                        _fitted_pdf[distribution] = pdf_fitted
                        _fitted_errors[distribution] = sq_error
                        _aic[distribution] = aic
                        _bic[distribution] = bic
                        _kldiv[distribution] = kullback_leibler
                        _ks_stat[distribution] = ks_stat
                        _ks_pval[distribution] = ks_pval

                    else:
                        _fitted_param[distribution] = None
                        _fitted_pdf[distribution] = None
                        _fitted_errors[distribution] = np.inf
                        _aic[distribution] = np.inf
                        _bic[distribution] = np.inf
                        _kldiv[distribution] = np.inf
                        _ks_stat[distribution] = None
                        _ks_pval[distribution] = None

                self.df_errors = pd.DataFrame(
                        {
                        "sumsquare_error": _fitted_errors,
                        "aic": _aic,
                        "bic": _bic,
                        "kl_div": _kldiv,
                        "ks_statistic": _ks_stat,
                        "ks_pvalue": _ks_pval})
                self.df_errors.sort_index(inplace=True)

                summary = self.summary()

                # shannon entropy
                sha_ent_dat = entropy(self._data)

                dl.results[p] = {
                    "params": _fitted_param,
                    "pdf_fitted": _fitted_pdf,
                    "sse": _fitted_errors,
                    "aic": _aic,
                    "bic": _bic,
                    "kldiv": _kldiv,
                    "ks_stat": _ks_stat,
                    "pval": _ks_pval,
                    "sha_ent_data": sha_ent_dat,
                    "summary": summary}

            elif mode == "per_replication":
                for r in range(self.n_replications):
                    print(f"Replication {r}: ")

                    _fitted_param = {}
                    _fitted_pdf = {}
                    _fitted_errors = {}
                    _aic = {}
                    _bic = {}
                    _kldiv = {}
                    _ks_stat = {}
                    _ks_pval = {}

                    self._trim_data(p, r)
                    self._update_data_pdf(p, r)

                    N = len(self.distributions)
                    with tqdm_joblib(desc=f"Fitting {N} distributions", total=N, disable=True) as progress_bar:
                        results = Parallel(n_jobs=max_workers, prefer=prefer)(
                            delayed(self.fit_single_distribution)(dist, self._data, self.x, self.y) for dist
                            in self.distributions
                        )

                    for distribution, values in results:
                        if values is not None:
                            param, pdf_fitted, sq_error, aic, bic, kullback_leibler, ks_stat, ks_pval = values

                            _fitted_param[distribution] = param
                            _fitted_pdf[distribution] = pdf_fitted
                            _fitted_errors[distribution] = sq_error
                            _aic[distribution] = aic
                            _bic[distribution] = bic
                            _kldiv[distribution] = kullback_leibler
                            _ks_stat[distribution] = ks_stat
                            _ks_pval[distribution] = ks_pval

                        else:
                            _fitted_param[distribution] = None
                            _fitted_pdf[distribution] = None
                            _fitted_errors[distribution] = np.inf
                            _aic[distribution] = np.inf
                            _bic[distribution] = np.inf
                            _kldiv[distribution] = np.inf
                            _ks_stat[distribution] = None
                            _ks_pval[distribution] = None

                    self.df_errors = pd.DataFrame(
                        {
                        "sumsquare_error": _fitted_errors,
                        "aic": _aic,
                        "bic": _bic,
                        "kl_div": _kldiv,
                        "ks_statistic": _ks_stat,
                        "ks_pvalue": _ks_pval
                        })
                    self.df_errors.sort_index(inplace=True)

                    summary = self.summary()

                    # shannon entropy
                    sha_ent_dat = entropy(self._data)

                    dl.results[p][r] = {
                        "params": _fitted_param,
                        "pdf_fitted": _fitted_pdf,
                        "sse": _fitted_errors,
                        "aic": _aic,
                        "bic": _bic,
                        "kldiv": _kldiv,
                        "ks_stat": _ks_stat,
                        "pval": _ks_pval,
                        "sha_ent_data": sha_ent_dat,
                        "summary": summary}

        # print results
        dl.print_summary()

        return dl

    def summary(self, n_best=5, method="sumsquare_error"):
        n = min(n_best, len(self.distributions))
        try:
            names = self.df_errors.sort_values(by=method).index[0:n]
        except:
            names = self.df_errors.sort(method).index[0:n]
        summary = self.df_errors.loc[names]
        print(summary)
        return summary

    @staticmethod
    def _with_timeout(func, args=(), kwargs={}, timeout=30):
        with multiprocessing.pool.ThreadPool(1) as pool:
            async_result = pool.apply_async(func, args, kwargs)
            return async_result.get(timeout=timeout)


class distribution_likelihoods_results():
    """provides output class for the results of the distribution fits
    This class is only called from the data function get_distribution_likelihood
    """
    def __init__(self, n_processes=None, n_replications=None, mode=None):
        self.n_data_processes = n_processes
        self.n_data_replications = n_replications
        self.mode = mode

        # prepare result list
        if self.mode == "over_all_replications":
            self.results = ["Not tested"] * self.n_data_processes
        elif self.mode == "per_replication":
            self.results = [["Not tested"] * self.n_data_replications for i in range(self.n_data_processes)]
        else:
            raise RuntimeError(f"Wrong input for mode: {mode}\n"
                               f"use \"over_all_replications\" or \"per_replication\"")

        self.de_dists = {"norm": 1, "uniform": 2, "rayleigh": 1, "expon": 1, "beta": 2, "cauchy": 0, "chi": 1,
                         "chi2": 1, "erlang": 2, "f": 2, "gamma": 1, "laplace": 1, "logistic": 1, "lognorm": 2,
                         "maxwell": 1, "gennorm": 2, "pareto": 2, "t": 1, "triang": 2, "weibull_max": 2,
                         "multivariate_normal": 2}

    def print_summary(self):
        """prints results of distribution fit for all tested processes and replications"""
        # print results
        for p in range(self.n_data_processes):
            print(f"\nProcess {p}")
            if self.mode == "over_all_replications":
                print(f"--------------- fitted over all replications\n")
                if self.results[p] == "Not tested":
                    print("not tested")
                else:
                    print(self.results[p]["summary"])
            elif self.mode == "per_replication":
                print(f"--------------- fitted for each replication\n")
                if self.results[p] == "Not tested":
                    print("not tested")
                else:
                    for r in range(self.n_data_replications):
                        print(f"- Replication {r}")
                        print(self.results[p][r]["summary"])

    def get_best_pval(self, process):                                                       # TODO ????????????????????????????????
        """get the best fitted p-val of the given process (for all replications depending on the
        used mode)"""
        if self.mode == "over_all_replications":
            pval = self.results[process]["summary"].T.values[5][0]
        else:
            pval = [None] * self.n_data_replications
            for r in range(self.n_data_replications):
                pval[r] = list(self.results[process][r]["summary"].T.values[5][0])
        return pval

    def get_best_dist(self, process):                                                       # TODO ????????????????????????????????
        """get the best fitted distribution of the given process (for all replications depending on the
        used mode)"""
        if self.mode == "over_all_replications":
            dist = self.results[process]["summary"].T.columns[0]
        else:
            dist = [None] * self.n_data_replications
            for r in range(self.n_data_replications):
                dist[r] = self.results[process][r]["summary"].T.columns[0]
        return dist

    def get_all_dists(self, process):
        """get all fitted distribution of the given process (for all replications depending on the
        used mode)"""
        if self.mode == "over_all_replications":
            dist_names = self.results[process]["params"].keys()
        else:
            dist_names = [None] * self.n_data_replications
            for r in range(self.n_data_replications):
                dist_names[r] = self.results[process][r]["params"].keys()
        return dist_names

    def get_all_pval(self, process):
        """get all fitted distribution of the given process (for all replications depending on the
        used mode)"""
        if self.mode == "over_all_replications":
            pval = self.results[process]["pval"]
        else:
            pval = [None] * self.n_data_replications
            for r in range(self.n_data_replications):
                pval[r] = self.results[process][r]["pval"]
        return pval

    def get_all_aic(self, process):
        """get all fitted distribution of the given process (for all replications depending on the
        used mode)"""
        if self.mode == "over_all_replications":
            aic = self.results[process]["aic"]
        else:
            aic = [None] * self.n_data_replications
            for r in range(self.n_data_replications):
                aic[r] = self.results[process][r]["aic"]
        return aic

    def get_parameters(self, process, dists="all"):
        """get fitted parameters (of all fitted distributions) for a given process"""

        if self.mode == "over_all_replications":
            if dists == "all":
                dist_names = self.results[process]["params"].keys()
                params = []
                for d in dist_names:
                    params.append(self.results[process]["params"][d])

            elif isinstance(dists, str):
                if dists not in self.results[process]["params"].keys():
                    keys = self.results[process]["params"].keys()
                    raise RuntimeError(f"Distribution {dists} is not in the results {keys}")
                else:
                    params = self.results[process]["params"][dists]
                    dist_names = [dists]
            else:
                raise RuntimeError(f"input con only be a string: \"all\" or \"<your distribution>\".")
        else:
            params = [None] * self.n_data_replications
            dist_names = [None] * self.n_data_replications
            for r in range(self.n_data_replications):
                if dists == "all":
                    dist_n = self.results[process]["params"].keys()
                    par = []
                    for d in dist_n:
                        par.append(self.results[process]["params"][d])
                    params[r] = par
                    dist_names[r] = dist_n

                elif isinstance(dists, str):
                    if dists not in self.results[process]["params"].keys():
                        keys = self.results[process]["params"].keys()
                        raise RuntimeError(f"Distribution {dists} is not in the results {keys}")
                    else:
                        params[r] = self.results[process]["params"][dists]
                        dist_names[r] = dists
                else:
                    raise RuntimeError(f"input con only be a string: \"all\" or \"<your distribution>\".")
        return params, dist_names
