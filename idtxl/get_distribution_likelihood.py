"""
provide structure an functions for getting distribution 
likelihood of the given data
"""

import numpy as np
from fitter import Fitter, get_common_distributions

#from . import idtxl_utils as utils


class distribution_likelihood():
    """




    For getting all distributions you can test here use e.g.:
        >>> dl = distribution_likelihood
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


    def fit_distributions(self, mode="over_all_replications", dists="common", 
    						processes="all"):
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
        dl = distribution_likelihoods_results(n_processes=self.n_processes, n_replications=self.n_replications, mode=mode)

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

    def get_parameters(self, process):
        """get fitted parameters of given process."""
        if self.mode == "over_all_replications":
            params = self.results[process]["params"]
        else:
            params = [None] * self.n_data_replications
            for r in range(self.n_data_replications):
                params = self.results[process][r]["params"]
        return params

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

    def get_all_dists(self, process):
        """get the (max 5) best fitted distribution of the given process (for all replications depending on the
        used mode)"""
        if self.mode == "over_all_replications":
            rep_list = list(self.results[process]["summary"].T.columns)
            pval_list = list(self.results[process]["summary"].T.values[5])
        else:
            rep_list = [None] * self.n_data_replications
            pval_list = [None] * self.n_data_replications
            for r in range(self.n_data_replications):
                rep_list[r] = list(self.results[process][r]["summary"].T.columns)
                pval_list[r] = list(self.results[process][r]["summary"].T.values[5])

        return rep_list, pval_list
