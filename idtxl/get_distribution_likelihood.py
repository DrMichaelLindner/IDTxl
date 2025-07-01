"""
provide structure an functions for getting distribution 
likelihood of the given data
"""

import numpy as np
from fitter import Fitter, get_common_distributions

#from . import idtxl_utils as utils


class get_distribution_likelihood():
    """




    For getting all distributions you can test here use e.g.:
        >>> dl = get_distribution_likelihood
        >>> dl.show_distributions

    """

    def __init__(self, data=None):

        if not data:
            raise RuntimeError("data not specified. ")
        

        # check input data  ????????????????????????????????????????????????

        # get data
        self.data = data
        self.n_processes = self.data.n_processes
        self.n_replications = self.data.n_replications


        self.all_distributions = ["_fit", "alpha", "anglit", "arcsine", "argus", "beta", "betaprime", "bradford", "burr",
                             	"burr12", "cauchy", "chi", "chi2", "cosine", "crystalball", "dgamma", "dweibull", "erlang",
                             	"expon", "exponnorm", "exponpow", "exponweib", "f", "fatiguelife", "fisk", "foldcauchy",
                             	"foldnorm", "gamma", "gausshyper", "genexpon", "genextreme", "gengamma", "genhalflogistic",
                             	"genhyperbolic", "geninvgauss", "genlogistic", "gennorm", "genpareto", "gibrat",
                             	"gompertz", "gumbel_l", "gumbel_r", "halfcauchy", "halfgennorm", "halflogistic",
                             	"halfnorm", "hypsecant", "invgamma", "invgauss", "invweibull", "jf_skew_t", "johnsonsb",
                             	"johnsonsu", "kappa3", "kappa4", "ksone", "kstwo", "kstwobign", "laplace",
                             	"laplace_asymmetric", "levy", "levy_l", "levy_stable", "loggamma", "logistic",
                             	"loglaplace", "lognorm", "loguniform", "lomax", "maxwell", "mielke", "moyal",
                             	"multivariate_normal", "nakagami", "ncf", "nct", "ncx2", "norm", "norminvgauss",
                             	"pareto", "pearson3", "powerlaw", "powerlognorm", "powernorm", "rayleigh", "rdist",
                             	"recipinvgauss", "reciprocal", "rel_breitwigner", "rice", "rv_continuous", "rv_histogram",
                             	"semicircular", "skewcauchy", "skewnorm", "studentized_range", "t", "trapezoid", "trapz",
                             	"triang", "truncexpon", "truncnorm", "truncpareto", "truncweibull_min", "tukeylambda",
                             	"uniform", "vonmises", "vonmises_fisher", "vonmises_line", "wald", "weibull_max",
                             	"weibull_min", "wrapcauchy"]

    @staticmethod
    def _check_fitter():
        # test if fitter module is installed
        try:
            from fitter import Fitter, get_common_distributions
        except ImportError as e:
            raise RuntimeError("Module fitter is not available! Please install fitter e.g. >>>pip install fitter")

    def _check_dist_input(self, dists):
        """check if all specified dstributions are in the list"""
        if isinstance(dists, list):
            for d in dists:
                if d not in self.all_distributions:
                    raise RuntimeError(f"The distribution {d} from the list is not in the list of possible "
                                       f"distribtutions. \nCall:\n     >>> "
                                       f"data.get_distribution_likelihood(show_distributions=True)\nto get "
                                       f"a list of all possible distributions.")

        else:
            if dists != "all" and dists != "common":
                if dists not in self.all_distributions:
                    raise RuntimeError(f"The distribution {dists} is not in the list of possible distribtutions. \n"
                                       f"Call:\n     >>> data.get_distribution_likelihood(show_distributions=True)\nto"
                                       f" get a list of all possible distributions.")

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


    def show_distributions(self):
        print(self.all_distributions)
        print(f"You can specify a list of these distributions or a single one you to test.")


    def fit(self, mode="over_all_replications", dists="common", processes="all"):
        """ get the likelihoods of distribution of the data

        The distribuitions are fittet using the Fitter toolbox (https://github.com/cokelaer/fitter) ????????????????????????????????????????????????????????????????? REF MISSING

            Args:
            mode : string [optional]
                "over_all_replications" (default)
                    fits data over all replications of a process
                "per_replication"
                    fits data for each replication of a process
            dists : string or list [optional]
                "all"
                    tests all available distributions of the fitter toolbox
                "common" (default)
                    tests common distributions of the fitter toolbox
                "<your distribution>"
                    tests only this distribution
                    e.g. "norm" for normal distribution
                list of distribution you want to test
                    tests selected distributions in this list

            
            Returns an object containing:

                .n_data_proceses
                .d_data_replications
                .results: List
                    Depending on the used mode this functions returns a list of size
                        nr data processes (mode="over_all_replications")
                        nr [data processes][nr data replications] (mode="per_replications")

                    Each element of the list contains a structure containn the results of the fitter toolbox
                    with the following information:
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
                -get_best_dist(<process>)
                    get the name of the best fitted distribution
                .get_best_pval(<process>)
                    get the pval of the best fitted distribution

        """

        # test if fitter module is installed
        self._check_fitter()

        # Check inputs
        self._check_dist_input(dists)

        # check processes
        procs = self._check_processes(processes)

        # initialise distribution likelihood output
        dl = distribution_likelihoods_results(n_processes=self.n_processes,
                                              n_replications=self.n_replications,
                                              mode=mode)

        # get distribution fits
        for p in procs:
            if mode == "over_all_replications":

                # specify distributions depending on input
                if isinstance(dists, list):
                    f = Fitter(self.data.data[p, :, :],
                               distributions=dists)
                else:
                    if dists == "all":
                        f = Fitter(self.data.data[p, :, :])
                    elif dists == "common":
                        f = Fitter(self.data.data[p, :, :],
                                   distributions=get_common_distributions())
                    else:
                        f = Fitter(self.data.data[p, :, :],
                                   distributions=[dists])

                # fit distributions
                f.fit()

                dl.results[p] = {
                "params": f.fitted_param,
                "pdf_fitted": f.fitted_pdf,
                "sse": f._fitted_errors,
                "aic": f._aic,
                "bic": f._bic,
                "kldiv": f._kldiv,
                "ks_stat": f._ks_stat,
                "pval": f._ks_pval,
                "summary": f.summary()}


            elif mode == "per_replication":
                for r in range(self.n_replications):

                    # specify distributions depending on input
                    if isinstance(dists, list):
                        f = Fitter(self.data[p, :, r],
                                   distributions=dists)
                    else:
                        if dists == "all":
                            f = Fitter(self.data.data[p, :, r])
                        elif dists == "common":
                            f = Fitter(self.data.data[p, :, r],
                                       distributions=get_common_distributions())
                        else:
                            f = Fitter(self.data.data[p, :, r],
                                       distributions=[dists])

                    # fit distributions
                    f.fit()

                    # get summary of fit
                    dl.results[p][r] = {
                            "params": f.fitted_param,
                            "pdf_fitted": f.fitted_pdf,
                            "sse": f._fitted_errors,
                            "aic": f._aic,
                            "bic": f._bic,
                            "kldiv": f._kldiv,
                            "ks_stat": f._ks_stat,
                            "pval": f._ks_pval,
                            "summary": f.summary()}

        # print results
        dl.print_summary()

        return dl



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

    # def get_parameters(self, process):
    #     """get fitted parameters of given process."""
    #     if self.mode == "over_all_replications":
    #         params = self.results[process]["params"]
    #     else:
    #         params = [None] * self.n_data_replications
    #         for r in range(self.n_data_replications):
    #             params = self.results[process][r]["params"]
    #     return params

    def get_best_pval(self, process):
        """get the best fitted p-val of the given process (for all replications depending on the
        used mode)"""
        if self.mode == "over_all_replications":
            pval = self.results[process]["summary"].T.values[5][0]
        else:
            pval = [None] * self.n_data_replications
            for r in range(self.n_data_replications):
                pval[r] = list(self.results[process][r]["summary"].T.values[5][0])
        return pval

    def get_best_dist(self, process):
        """get the best fitted distribution of the given process (for all replications depending on the
        used mode)"""
        if self.mode == "over_all_replications":
            dist = self.results[process]["summary"].T.columns[0]
        else:
            dist = [None] * self.n_data_replications
            for r in range(self.n_data_replications):
                dist[r] = self.results[process][r]["summary"].T.columns[0]
        return dist

    def get_all_dist(self, process):
        """get all fitted distribution of the given process (for all replications depending on the
        used mode)"""
        if self.mode == "over_all_replications":
            dist_names = self.results[process]["params"].keys()
        else:
            dist_names = [None] * self.n_data_replications
            for r in range(self.n_data_replications):
                dist_name[r] = self.results[process][r]["params"].keys()
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
                raise RuntimeError(f"input con only be  a string: \"all\" or \"<your distribution>\".")
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
                    dist_name[r] = dist_n

                elif isinstance(dists, str):
                    if dists not in self.results[process]["params"].keys():
                        keys = self.results[process]["params"].keys()
                        raise RuntimeError(f"Distribution {dists} is not in the results {keys}")
                    else:
                        params[r] = self.results[process]["params"][dists]
                        dist_name[r] = dists
                else:
                    raise RuntimeError(f"input con only be a string: \"all\" or \"<your distribution>\".")
        return params, dist_names

    def __de_uniform(self, a, b):
        """calculate differential entropy for uniform distribution"""
        de = np.log2(b-a)
        return de

    def __de_norm(self, sigma):
        """calculate differential entropy for normal distribution"""
        de = np.log2(sigma * np.sqrt(2 * np.pi * np.e))
        return de

    def __de_expo(self, lam):
        """calculate differential entropy for exponential distribution"""
        de = 1 - np.log2(lam)
        return de

    def __de_raleigh(self, sigma):
        """calculate differential entropy for raleigh distribution"""
        de = 1 + np.log2(sigma/(np.sqrt(2))) + np.e/2
        return de

    def __de_beta(self, a, b):
        """calculate differential entropy for beta distribution"""
        de =  np.random.beta(a, b) - (a - 1) * (np.digamma(a) - np.digamma(a + b)) \
              - (b - 1) * (np.digamma(b) - np.digamma(a + b))
        return de

    def __de_cauchy(self):
        """calculate differential entropy for cauchy distribution"""
        de = np.log2(2 * np.pi * np.e)
        return de

    def __de_chi(self, k):
        """calculate differential entropy for chi distribution"""
        de = np.log2(np.gamma(k/2) / np.sqrt(2)) - (k-1)/2 * np.digamma(k/2) + k/2
        return de

    def __de_chi2(self, k):
        """calculate differential entropy for chi2 distribution"""
        de = np.log2(2*np.gamma(k/2)) - (1 - k/2) * np.digamma(k/2) + k/2
        return de

    def __de_erlang(self, k, l):
        """calculate differential entropy for erlang distribution"""
        de = (1 - k) * np.digamma(k) + np.log2(np.gamma(k) / l) + k
        return de

    def __de_f(self, n1, n2):
        """calculate differential entropy for F distribution"""
        de = np.log2(n1/n2) * np.random.beta(n1/2, n2/2) +\
             (1 - n1/2) * np.digamma(n1/2) -\
             (1 + n2/2) * np.digamma(n2/2) + \
             (n1 + n2)/2 * np.digamma((n1 + n2)/2)
        return de

    def __de_gamma(self, k):
        """calculate differential entropy for gamma distribution"""
        de = np.log2( o * np.gamma(k)) + (1 - k) * np.digamma(k) + k   # ???????????????????????????????????????????????????????????????
        return de

    def __de_laplace(self, b):
        """calculate differential entropy for laplace distribution"""
        de = 1 + np.log2(2 * b)
        return de

    def __de_logistic(self, s):
        """calculate differential entropy for logistic distribution"""
        de = np.log2(s) + 2
        return de

    def __de_lognorm(self, m, s):
        """calculate differential entropy for lognormal distribution"""
        de = m + 0.5 * np.log2(2 * np.pi * np.e * s**2)
        return de

    def __de_maxwell(self, a):
        """calculate differential entropy for Maxwell-Boltzmann distribution"""
        de = np.log2(a * np.sqrt(2 * np.pi)) + np.e - 0.5
        return de

    def __de_gennorm(self, a, b):
        """calculate differential entropy for Generalized normal distribution"""
        de = np.log2(np.gamma(a/2) / 2 * b**0.5) - (a - 1)/2 * np.digamma(a/2) + a/2
        return de

    def __de_pareto(self, a, x):
        """calculate differential entropy for pareto distribution"""
        de = np.log2(x/a) + 1 + 1/a
        return de

    def __de_t(self, v):
        """calculate differential entropy for Student's t distribution"""
        de = (v + 1)/2 * (np.digamma((v + 1)/2 - np.digamma(v/2)) + np.log2(np.sqrt(v) * np.random-beta(0.5, v/2)))
        return de

    def __de_triang(self, a, b):
        """calculate differential entropy for triangular distribution"""
        de = 0.5 + np.log2((b - a)/2)
        return de

    def __de_weibull(self, k, l):
        """calculate differential entropy for weibull distribution"""
        de = (k - 1)/k * np.e + np.log2(l/k) + 1
        return de

    def __de_multivariate_normal(self, k, l):
        """calculate differential entropy for multivariate normal distribution"""
        # de = 0.5 * np.log2(  )  ?????????????????????????????????????????????????????????????????????????????????????????????????????
        return de
