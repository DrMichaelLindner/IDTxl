"""Provide Python estimators."""
import numpy as np

from scipy.special import digamma
from scipy.spatial import cKDTree

from idtxl.estimator import Estimator
from idtxl.knn.knn_finder_factory import get_knn_finder

import idtxl.idtxl_utils as utils

from collections import Counter
import math

class PythonEstimator(Estimator):
    """Abstract class for implementation of Python estimators

    Abstract class for implementation of Python estimators, child classes
    implement estimators for mutual information (MI), conditional mutual
    information (CMI), active information storage (AIS) and
    transfer entropy (TE)
    
    using the Kraskov-Grassberger-Stoegbauer estimator for continuous data,
    plug-in estimators for discrete data, and Gaussian estimators for
    continuous Gaussian data.
    """

    def __init__(self, settings=None):
        """Set default estimator settings.""" ######################################################### TODO

        
        self.settings = settings.copy()

    def _normalise_data(self, data: np.ndarray):
        """Standardise data to zero mean and unit variance."""
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    

    #################################################################################################### TODO

    def embed_past_current(self, process, num_valid, history, tau):
        # Build the embedded past vectors and current values

        past = np.zeros((num_valid, history), dtype=np.float64)
        current = np.zeros(num_valid, dtype=np.float64)
        
        for i in range(num_valid):
        #past = np.zeros((len(num_valid), history), dtype=np.float64)
        #current = np.zeros(len(num_valid), dtype=np.float64)

        #for i in range(len(num_valid)):
            t = i + history * tau + 1
            #current[i] = process[t+start]
            current[i] = process[t]
            for j in range(history):
                #past[i, j] = process[t+start - (j + 1) * tau]
                past[i, j] = process[t - (j + 1) * tau]

        return past, current

    #################################################################################################### TODO

    def takens_embedding(x, dim, delay):
        x = np.asarray(x, dtype=float)
        if x.ndim != 1:
            raise ValueError("x must be a 1D array")  
        n = len(x)
        start = (dim - 1) * delay
        if start >= n:
            raise ValueError("Embedding dimension and delay too large for data length")
        Y = np.zeros((n - start, dim), dtype=float)
        for i in range(dim):
            t = start - i * delay
            Y[:, i] = x[t:t + n - start]
        return Y

    def delay_embedding(ts, dim, delay, step):
        ts = np.asarray(ts)
        n = len(ts)
        m = n - (dim - 1) * delay
        if m <= 0:
            return np.empty((0, dim), dtype=ts.dtype)

        idx = np.arange(m)[:, None] + delay * np.arange(dim)[None, :]
        idx = idx[::step]
        return ts[idx]


    #################################################################################################### TODO


    def is_analytic_null_estimator(self):
        return False

    def is_parallel(self):
        return False


class PythonKraskov(PythonEstimator):
    """Abstract class for implementation of Python Kraskov estimators

    Abstract class for implementation of Python Kraskov estimators, child classes
    implement estimators for mutual information (MI), conditional mutual
    information (CMI), actice information storage (AIS)
    and transfer entropy (TE) 

    Args:
        settings : dict [optional]
            set estimator parameters:

            - kraskov_k : int [optional] - no. nearest neighbours for KNN
              search (default=4)
            - normalise : bool [optional] - z-standardise data (default=False)
            
            ################################################################################################## TODO?
            - theiler_t : int [optional] - no. next temporal neighbours ignored
              in KNN and range searches (default=0)
            


            - noise_level : float [optional] - random noise added to the data
              (default=1e-8)
            - num_threads : int | str [optional] - number of threads used for
              estimation (default='USE_ALL', note that this uses *all*
              available threads on the current machine)
            - local_values : bool [optional] - return local MI/TE instead of
              average MI/TE (default=False)
            
    """

    def __init__(self, settings):
        """Initialise estimator with settings."""
        settings.setdefault('kraskov_k', 4)
        settings.setdefault('normalise', False)
        settings.setdefault('theiler_t', 0)
        settings.setdefault('base', np.e)
        settings.setdefault('noise_level', 1e-8)
        settings.setdefault('num_threads', 'USE_ALL')
        settings.setdefault("knn_finder", "scipy_ckdtree")
        settings.setdefault("lag_mi", 0)
        settings.setdefault('local_values', False)
        super().__init__(settings)

        ################################################################################# TODO

    


    def _compute_epsilon(self, data: np.ndarray, k: int):
        """Compute the distance to the kth nearest neighbor for each point in x."""
        knn_finder = self._knn_finder_class(data, **self._knn_finder_settings)
        return knn_finder.find_all_dists_to_kth_neighbor(k)
    
    def _compute_n(self, data: np.ndarray, r: np.ndarray):
        """Count the number of neighbors strictly within a given radius r for each point in x.
        Returns the number of neighbors plus one, because the point itself is included in the data.
        """
        knn_finder = self._knn_finder_class(data, **self._knn_finder_settings)
        return knn_finder.count_all_neighbors(r) + 1

    #def _compute_n_theiler(self, data: np.ndarray, r: np.ndarray, theiler: int):
    #    knn_finder = self._knn_finder_class(data, **self._knn_finder_settings)
    #    return knn_finder.count_all_neighbors_theiler(r, self.settings['theiler_t']) + 1

    def is_analytic_null_estimator(self):
        return False


class PythonGaussian(PythonEstimator):
    """Abstract class for implementation of Python Gaussian-estimators.

    Abstract class for implementation of Python Gaussian-estimators, child
    classes implement estimators for mutual information (MI), conditional
    mutual information (CMI), actice information storage (AIS) and
    transfer entropy (TE) using python Gaussian estimator for continuous data. 

    Args:
        settings : dict [optional]
            set estimator parameters:

            - normalise : bool [optional] - z-standardise data (default=False)
            - noise_level : float [optional] - random noise added to the data
              (default=0)
    """

    def __init__(self, settings):
        # Check for currently unsupported settings
        if settings.get('local_values', False):
            raise ValueError('This estimator currently does not support local_values.')
        settings.setdefault('local_values', False)

        settings.setdefault('normalise', False)
        settings.setdefault('noise_level', 0)
        super().__init__(settings)
        
    def estimate_surrogates_analytic(self, n_perm=200, **data):
        """Estimate the surrogate distribution analytically.
        This method must be implemented because this class'
        is_analytic_null_estimator() method returns true

        Args:
            n_perms : int
                number of permutations (default=200)
            data : numpy arrays
                realisations of random variables required for the calculation
                (varies between estimators, e.g. 2 variables for MI, 3 for
                CMI). Formatted as per estimate_parallel for this estimator.

        Returns:
            float | numpy array
                n_perm surrogates of the average MI/CMI/TE over all samples
                under the null hypothesis of no relationship between var1 and
                var2 (in the context of conditional)
        """
        return common_estimate_surrogates_analytic(self, n_perm, **data)

    ##################################################################################################### TODO


################################################################################ TODO
class PythonDiscrete(PythonEstimator):
    """Abstract class for implementation of Python Gaussian-estimators.

    Abstract class for implementation of Python Gaussian-estimators, child
    classes implement estimators for mutual information (MI), conditional
    mutual information (CMI), 

    active information storage (AIS), ############################################# ??????????????????????? TODO?

    transfer
    entropy (TE) using python Gaussian estimator for continuous data. 

    Args:
        settings : dict [optional]
            set estimator parameters:

            - normalise : bool [optional] - z-standardise data (default=False)
            - noise_level : float [optional] - random noise added to the data
              (default=0)
    """

    def __init__(self, settings):
        settings.setdefault('discretise_method', 'none')
        super().__init__(settings)

    def _discretise_vars(self, var1, var2, conditional=None):
        # Discretise variables if requested. Otherwise assert data are discrete
        # and provided alphabet sizes are correct.
        if self.settings['discretise_method'] == 'equal':
            var1 = utils.discretise(var1, self.settings['alph1'])
            var2 = utils.discretise(var2, self.settings['alph2'])
            if conditional is not None:
                conditional = utils.discretise(conditional,
                                               self.settings['alphc'])

        elif self.settings['discretise_method'] == 'max_ent':
            var1 = utils.discretise_max_ent(var1, self.settings['alph1'])
            var2 = utils.discretise_max_ent(var2, self.settings['alph2'])
            if not (conditional is None):
                conditional = utils.discretise_max_ent(conditional,
                                                       self.settings['alphc'])

        elif self.settings['discretise_method'] == 'none':
            assert issubclass(var1.dtype.type, np.integer), (
                'Var1 is not an integer numpy array. '
                'Discretise data to use this estimator.')
            assert issubclass(var2.dtype.type, np.integer), (
                'Var2 is not an integer numpy array. '
                'Discretise data to use this estimator.')
            assert np.min(var1) >= 0, 'Minimum of var1 is smaller than 0.'
            assert np.min(var2) >= 0, 'Minimum of var2 is smaller than 0.'
            assert np.max(var1) < self.settings['alph1'], (
                        'Maximum of var1 is larger than the alphabet size.')
            assert np.max(var2) < self.settings['alph2'], (
                        'Maximum of var2 is larger than the alphabet size.')
            if conditional is not None:
                assert np.min(conditional) >= 0, (
                        'Minimum of conditional is smaller than 0.')
                assert issubclass(conditional.dtype.type, np.integer), (
                    'Conditional is not an integer numpy array. '
                    'Discretise data to use this estimator.')
                assert np.max(conditional) < self.settings['alphc'], (
                    'Maximum of conditional is larger than the alphabet size.')
        else:
            raise ValueError('Unkown discretisation method.')

        if conditional is not None:
            return var1, var2, conditional
        else:
            return var1, var2

    def is_analytic_null_estimator(self):
        return True

    def get_analytic_distribution(self, **data):
        """Return a JIDT AnalyticNullDistribution object.

        Required so that our estimate_surrogates_analytic method can use the
        common_estimate_surrogates_analytic() method, where data is formatted
        as per the estimate method for this estimator.

        Args:
            data : numpy arrays
                realisations of random variables required for the calculation
                (varies between estimators, e.g. 2 variables for MI, 3 for
                CMI). Formatted as per the estimate method for this estimator.

        Returns:
            Java object
                JIDT calculator that was used here
        """
        pass

    def estimate_surrogates_analytic(self, n_perm=200, **data):
        """Return estimate of the analytical surrogate distribution.

        This method must be implemented because this class'
        is_analytic_null_estimator() method returns true.

        Args:
            n_perms : int [optional]
                number of permutations (default=200)
            data : numpy arrays
                realisations of random variables required for the calculation
                (varies between estimators, e.g. 2 variables for MI, 3 for
                CMI). Formatted as per the estimate method for this estimator.

        Returns:
            float | numpy array
                n_perm surrogates of the average MI/CMI/TE over all samples
                under the null hypothesis of no relationship between var1 and
                var2 (in the context of conditional)
        """
        return common_estimate_surrogates_analytic(self, n_perm, **data)

        



class PythonKraskovMI(PythonKraskov):
    """Estimate mutual information using Kraskov's estimator.

    Calculate the mutual information between two variables. 

    Results are returned in nats.
    
    Args:
        settings : dict [optional]
            set estimator parameters:

            - kraskov_k : int [optional] - no. nearest neighbours for KNN
              search (default=4)
            - base : float - base of returned values (default=np=e)
            - normalise : bool [optional] - z-standardise data (default=False)
            - noise_level : float [optional] - random noise added to the data
              (default=1e-8)

            - rng_seed : int | None [optional] - random seed if noise level > 0
            - num_threads : int | str [optional] - number of threads used for
              estimation (default='USE_ALL', note that this uses *all*
              available threads on the current machine)
            - lag_mi : int [optional] - time difference in samples to calculate
              the lagged MI between processes (default=0)
            - local_values : bool [optional] - return local MI/TE instead of
              average MI/TE (default=False)

            #   ################################################################################################################## TODO
            - theiler_t : int [optional] - no. next temporal neighbours ignored
              in KNN and range searches (default=0)
            

    """
    def __init__(self, settings):
        """Initialise estimator with settings."""
        settings = self._check_settings(settings)
        super().__init__(settings)

        ################################################################################################## TODO
        # Check for currently unsupported settings
        if settings.get('algorithm_num', 1) != 1:
            raise ValueError('This estimator currently does not support algorithm_num arguments.')
        """
        

        self._kraskov_k = settings.get("kraskov_k", 4)
        self._base = settings.get("base", np.e)
        self._normalise = settings.get("normalise", False)

        # Set number of threads
        num_threads = settings.get("num_threads", -1)
        if num_threads == "USE_ALL":
            num_threads = -1
        self._knn_finder_settings["num_threads"] = num_threads

        # Init rng for added gaussian noise
        self._noise_level = settings.get("noise_level", 1e-8)
        """

        if self.settings['noise_level'] > 0:
            rng_seed = settings.get("rng_seed", None)
            self._rng = np.random.default_rng(rng_seed)
        
        self._knn_finder_settings = settings.get("knn_finder_settings", {})
        
        # Set number of threads
        num_threads = settings.get("num_threads", -1)
        if num_threads == "USE_ALL":
            num_threads = -1
        self._knn_finder_settings["num_threads"] = num_threads

        # Get KNN finder class
        self._knn_finder_name = settings.get("knn_finder", "scipy_kdtree")
        self._knn_finder_class = get_knn_finder(self._knn_finder_name)
        

    def estimate(self, var1: np.ndarray, var2: np.ndarray):
        """Estimate mutual information.

        Args:
            var1 : numpy array
                realisations of first variable, either a 2D numpy array where
                array dimensions represent [realisations x variable dimension]
                or a 1D array representing [realisations]
            var2 : numpy array
                realisations of the second variable (similar to var1)

        Returns:
            float | numpy array
                average MI over all samples 

                ################################################ ???????????????????????????????????????????????????? local values ??????
                or local MI for individual
                samples if 'local_values'=True
        """

        # Check the input data
        var1 = self._ensure_two_dim_input(var1)
        var2 = self._ensure_two_dim_input(var2)
        
        assert (
            var1.shape[0] == var2.shape[0]
        ), f"Unequal number of observations (var1: {var1.shape[0]}, var2: {var2.shape[0]})"

        # Check if number of points is sufficient for estimation.
        if var1.shape[0] - 1 < self.settings['kraskov_k']:
            raise ValueError(
                f"Not enough observations for Kraskov estimator (need at least {self.settings['kraskov_k'] + 1}, got {var1.shape[0]})."
            )

        # Normalise data
        if self.settings['normalise']:
            var1 = self._normalise_data(var1)
            var2 = self._normalise_data(var2)

        # Add noise to avoid duplicate points
        # Do not add noise inplace, because it would change the input data
        if self.settings['noise_level'] > 0:
            var1 = var1 + self._rng.normal(0, self.settings['noise_level'], var1.shape)
            var2 = var2 + self._rng.normal(0, self.settings['noise_level'], var2.shape)

        # Shift variables to calculate a lagged MI.
        if self.settings['lag_mi'] > 0:
            var1 = var1[:-self.settings['lag_mi'], :]
            var2 = var2[self.settings['lag_mi']:, :]


        # Compute distances to kth nearest neighbors in the joint space
        epsilon = self._compute_epsilon(
            np.concatenate((var1, var2), axis=1), self.settings['kraskov_k']
        )

        # Count neighbors ############################################################################################ TODO Theiler correction
        #if self.settings['theiler_t'] > 0:
        #    n_c_var1 = self._compute_n_theiler(var1, epsilon, self.settings['theiler_t'])
        #else:
        n_c_var1 = self._compute_n(var1, epsilon)
        digamma_nc_var1 = digamma(n_c_var1)
        del n_c_var1

        #if self.settings['theiler_t'] > 0: ############################################################################################ TODO Theiler correction
        #    n_c_var2 = self._compute_n_theiler(var2, epsilon, self.settings['theiler_t'])
        #else:
        n_c_var2 = self._compute_n(var2, epsilon)
        digamma_nc_var2 = digamma(n_c_var2)
        del n_c_var2

        # Compute MI
        if self.settings["local_values"]:
            mi = (digamma(self.settings['kraskov_k']) 
                + digamma(len(var1))
                - digamma_nc_var1
                - digamma_nc_var2
            ) / np.log(self.settings['base'])

        else:
            mi = (digamma(self.settings['kraskov_k']) 
                + digamma(len(var1))
                - np.mean(digamma_nc_var1)
                - np.mean(digamma_nc_var2)
            ) / np.log(self.settings['base'])

        return mi


class PythonKraskovCMI(PythonKraskov):
    """Estimate conditional mutual information using Kraskov's first estimator.

    Calculate the conditional mutual information (CMI) between three variables.
    If no conditional is given (is None), the function returns the mutual information 
    between var1 and var2.

    Results are returned in nats.

    Args:
        settings : dict [optional]
            set estimator parameters:

            - kraskov_k : int [optional] - no. nearest neighbours for KNN
              search (default=4)
            - base : float - base of returned values (default=np=e)
            - normalise : bool [optional] - z-standardise data (default=False)
            - noise_level : float [optional] - random noise added to the data
              (default=1e-8)
            - theiler_t : int [optional] - no. next temporal neighbours ignored
              in KNN and range searches (default=0)
            - rng_seed : int | None [optional] - random seed if noise level > 0
            - num_threads : int | str [optional] - number of threads used for
              estimation (default='USE_ALL', note that this uses *all*
              available threads on the current machine)
            - knn_finder : str [optional] - knn algorithm to use, can be
              'scipy_kdtree' (default), 'sklearn_kdtree', or 'sklearn_balltree'
            - local_values : bool [optional] - return local MI/TE instead of
              average MI/TE (default=False)

    """

    def __init__(self, settings):
        """Initialise estimator with settings."""
        settings = self._check_settings(settings)
        super().__init__(settings)
        
        # Check for currently unsupported settings
        if settings.get('algorithm_num', 1) != 1:
            raise ValueError('This estimator currently does not support algorithm_num arguments.')

        if self.settings['noise_level'] > 0:
            rng_seed = settings.get("rng_seed", None)
            self._rng = np.random.default_rng(rng_seed)
        
        self._knn_finder_settings = settings.get("knn_finder_settings", {})
        
        # Set number of threads
        num_threads = settings.get("num_threads", -1)
        if num_threads == "USE_ALL":
            num_threads = -1
        self._knn_finder_settings["num_threads"] = num_threads

        # Get KNN finder class
        self._knn_finder_name = settings.get("knn_finder", "scipy_ckdtree")
        self._knn_finder_class = get_knn_finder(self._knn_finder_name)




    def estimate(self, var1: np.ndarray, var2: np.ndarray, conditional=None):
        """Estimate conditional mutual information between var1 and var2, given
        conditional.
        
        Args:
            var1 : numpy array
                realisations of first variable, either a 2D numpy array where
                array dimensions represent [realisations x variable dimension]
                or a 1D array representing [realisations]
            var2 : numpy array
                realisations of the second variable (similar to var1)
            conditional : numpy array [optional]
                realisations of the conditioning variable (similar to var), if
                no conditional is provided, return MI between var1 and var2

        Returns:
            float | numpy array
                average MI over all samples 

                ################################################ ???????????????????????????????????????????????????? local values ??????

        """

        # Return MI if no conditioning variable was provided.
        if conditional is None:
            if (self.est_mi is None):
                self.est_mi = PythonKraskovMI(self.settings)
            return self.est_mi.estimate(var1, var2)
        else:
            assert(conditional.size != 0), 'Conditional Array is empty.'

        # Check the input data
        var1 = self._ensure_two_dim_input(var1)
        var2 = self._ensure_two_dim_input(var2)
        conditional = self._ensure_two_dim_input(conditional)

        assert (
            var1.shape[0] == var2.shape[0] == conditional.shape[0]
        ), f"Unequal number of observations (var1: {var1.shape[0]}, var2: {var2.shape[0]}, conditional: {conditional.shape[0]})"

        # Check if number of points is sufficient for estimation.
        if var1.shape[0] - 1 < self.settings['kraskov_k']:
            raise ValueError(
                f"Not enough observations for Kraskov estimator (need at least {self.settings['kraskov_k'] + 1}, got {var1.shape[0]})."
            )

        # Normalise data
        if self.settings['normalise']:
            var1 = self._normalise_data(var1)
            var2 = self._normalise_data(var2)
            conditional = self._normalise_data(conditional)

        # Add noise to avoid duplicate points
        # Do not add noise inplace, because it would change the input data
        if self.settings['noise_level'] > 0:
            var1 = var1 + self._rng.normal(0, self.settings['noise_level'], var1.shape)
            var2 = var2 + self._rng.normal(0, self.settings['noise_level'], var2.shape)
            conditional = conditional + self._rng.normal(
                0, self.settings['noise_level'], conditional.shape
            )

        # Compute distances to kth nearest neighbors in the joint space
        epsilon = self._compute_epsilon(
            np.concatenate((var1, var2, conditional), axis=1), self.settings['kraskov_k']
        )

        # Count neighbors in the conditional space
        if conditional.shape[1] > 0:
            #if int(self.settings['theiler_t']) > 0: ############################################################################################ TODO Theiler correction
            #    n_c = self._compute_n_theiler(conditional, epsilon, self.settings['theiler_t'])
            #else:
            n_c = self._compute_n(conditional, epsilon)
            #mean_digamma_nc = np.mean(digamma(n_c))
            digamma_nc = digamma(n_c)
            del n_c

        #if int(self.settings['theiler_t']) > 0: ############################################################################################ TODO Theiler correction
        #    n_c_var1 = self._compute_n_theiler(np.concatenate((var1, conditional), axis=1), epsilon, self.settings['theiler_t'])
        #else: 
        n_c_var1 = self._compute_n(np.concatenate((var1, conditional), axis=1), epsilon)
        #mean_digamma_nc_var1 = np.mean(digamma(n_c_var1))
        digamma_nc_var1 = digamma(n_c_var1)
        del n_c_var1

        #if int(self.settings['theiler_t']) > 0: ############################################################################################ TODO Theiler correction
        #    n_c_var2 = self._compute_n_theiler(np.concatenate((var2, conditional), axis=1), epsilon, self.settings['theiler_t'])
        #else: 
        n_c_var2 = self._compute_n(np.concatenate((var2, conditional), axis=1), epsilon)
        #mean_digamma_nc_var2 = np.mean(digamma(n_c_var2))
        digamma_nc_var2 = digamma(n_c_var2)
        del n_c_var2

        # Compute CMI
        if self.settings["local_values"]:
            return (
                digamma(self.settings['kraskov_k'])
                + digamma_nc
                - digamma_nc_var1
                - digamma_nc_var2
            ) / np.log(self.settings['base'])     
        else:
            return (
                digamma(self.settings['kraskov_k'])
                + np.mean(digamma_nc)
                - np.mean(digamma_nc_var1)
                - np.mean(digamma_nc_var2)
            ) / np.log(self.settings['base'])
            

class PythonKraskovAIS(PythonKraskov):
    """Calculate active information storage with Python Kraskov implementation.

    Calculate active information storage (AIS) for some process using Python
    implementation of the Kraskov type 1 estimator. AIS is defined as the
    mutual information between the processes' past state and current value.

    The past state needs to be defined in the settings dictionary, where a past
    state is defined as a uniform embedding with parameters history and tau.
    The history describes the number of samples taken from a processes' past,
    tau describes the embedding delay, i.e., the spacing between every two
    samples from the processes' past.

    Results are returned in nats.

    Args:
        settings : dict
            sets estimation parameters:

            - history : int - number of samples in the processes' past used as
              embedding
            - tau : int [optional] - the processes' embedding delay (default=1)
            - kraskov_k : int [optional] - no. nearest neighbours for KNN
              search (default=4)
            - normalise : bool [optional] - z-standardise data (default=False)
            - theiler_t : int [optional] - no. next temporal neighbours ignored
              in KNN and range searches (default=0)
            - noise_level : float [optional] - random noise added to the data
              (default=1e-8)
            - num_threads : int | str [optional] - number of threads used for
              estimation (default='USE_ALL', note that this uses *all*
              available threads on the current machine)
            - local_values : bool [optional] - return local TE instead of
              average TE (default=False)
            
            ########################################################################################################## TODO algorithm number
            - algorithm_num : int [optional] - which Kraskov algorithm (1 or 2)
              to use (default=1)
            
    """

    def __init__(self, settings):
        settings = self._check_settings(settings)
        # Check for history for AIS estimation.
        try:
            settings['history']
        except KeyError:
            raise RuntimeError('No history was provided for AIS estimation.')
        settings.setdefault('tau', 1)
        assert type(settings['history']) is int, (
                                            'History has to be an integer.')
        assert type(settings['tau']) is int, ('Tau has to be an integer.')

        super().__init__(settings)


    def estimate(self, process):
        """Estimate active information storage.

        Args:
            process : numpy array
                realisations of first variable, either a 2D numpy array where
                array dimensions represent [realisations x variable dimension]
                or a 1D array representing [realisations]

        Returns:
            float | numpy array
                average AIS over all samples or local AIS for individual
                samples if 'local_values'=True
        """
        process = self._ensure_one_dim_input(process)

        # Check if number of points is sufficient for estimation.
        self._check_number_of_points(process.shape[0])

        # len process
        n = len(process)

        min_length = self.settings['history'] * self.settings['tau'] + 2
        
        if n < min_length:
            raise ValueError(f"Data too short: need at least {min_length} samples, got {n}")
    
        num_valid = n - self.settings['history'] * self.settings['tau'] - 1
    
        if num_valid <= 0:
            raise ValueError(f"Not enough valid embedding vectors")
        
        past, current = self.embed_past_current(process, num_valid, self.settings['history'], self.settings['tau'])

        est_mi=PythonKraskovMI(self.settings)
        mi = est_mi.estimate(var1=current, var2=past)

        return mi    


################################################################################ TODO
class PythonKraskovTE(PythonKraskov):
    """Estimate transfer using Kraskov's estimator.
     
    Calculate transfer entropy between a source and a target variable using
    Python implementation of the Kraskov estimator. Transfer entropy is
    defined as the conditional mutual information between the source's past
    state and the target's current value, conditional on the target's past.
    
    The past state needs to be defined in the settings dictionary, where a past
    state is defined as a uniform embedding with parameters history and tau.
    The history describes the number of samples taken from a processes' past,
    tau describes the embedding delay, i.e., the spacing between every two
    samples from the processes' past.
    
    Results are returned in nats.

    Args:
        settings : dict 
            set estimator parameters:
            - kraskov_k : int [optional] - no. nearest neighbours for KNN
              search (default=4)
            - history_target : int - number of samples in the target's past
              used as embedding
            - history_source  : int [optional] - number of samples in the
              source's past used as embedding (default=same as the target
              history)
            - tau_source : int [optional] - source's embedding delay
              (default=1)
            - tau_target : int [optional] - target's embedding delay
              (default=1)
            - source_target_delay : int [optional] - information transfer delay
              between source and target (default=1)
            

            ################################################ ???????????????????????????????????????????????????? local values ??????
            - local_values : bool [optional] - return local TE instead of
              average TE (default=False)
            

            - algorithm_num : int [optional] - which Kraskov algorithm (1 or 2)
              to use (default=1)


            

    """
    def __init__(self, settings):
        """Initialise estimator with settings."""
        settings = self._check_settings(settings)
        settings = self._set_te_defaults(settings)
        super().__init__(settings)

        # Check for currently unsupported settings
        if settings.get('local_values', False) or settings.get('algorithm_num', 1) != 1:
            raise ValueError('This estimator currently does not support local_values or algorithm_num arguments.')

        if self.settings['noise_level'] > 0:
            rng_seed = settings.get("rng_seed", None)
            self._rng = np.random.default_rng(rng_seed)
        
        self._knn_finder_settings = settings.get("knn_finder_settings", {})
        
        # Set number of threads
        num_threads = settings.get("num_threads", -1)
        if num_threads == "USE_ALL":
            num_threads = -1
        self._knn_finder_settings["num_threads"] = num_threads

        # Get KNN finder class
        self._knn_finder_name = settings.get("knn_finder", "scipy_kdtree")
        self._knn_finder_class = get_knn_finder(self._knn_finder_name)

        
    def check_number_of_points(self, n_points):
        """Sanity check for number of points going into the estimator."""
        if (n_points - 1) <= int(self.settings["kraskov_k"]):
            raise RuntimeError(
                f"Insufficient number of points ({n_points}) for the requested number of nearest neighbours "
                f"(kraskov_k: {self.settings['kraskov_k']})."
            )

    def _set_te_defaults(self, settings):
        """Set defaults for transfer entropy estimation."""
        try:
            history_target = settings['history_target']
        except KeyError:
            raise RuntimeError('No target history was provided for TE '
                               'estimation.')
        settings.setdefault('history_source', history_target)
        settings.setdefault('tau_target', 1)
        settings.setdefault('tau_source', 1)
        settings.setdefault('source_target_delay', 1)

        assert type(settings['tau_target']) is int, (
            'Target tau has to be an integer.')
        assert type(settings['tau_source']) is int, (
            'Source tau has to be an integer.')
        assert type(settings['history_target']) is int, (
            'Target history has to be an integer.')
        assert type(settings['history_source']) is int, (
            'Source history has to be an integer.')
        assert type(settings['source_target_delay']) is int, (
            'Source-target delay has to be an integer.')
        assert settings['tau_target'] >= 1, 'Target tau must be >= 1'
        assert settings['tau_source'] >= 1, 'Source tau must be >= 1'
        assert settings['history_target'] >= 0, 'Target history must be >= 0'
        assert settings['history_source'] >= 1, 'Source history must be >= 1'
        assert settings['source_target_delay'] >= 0, (
            'Source-target delay must be >= 0')
        return settings

    def estimate(self, source: np.ndarray, target: np.ndarray):
        """Estimate transfer entropy from a source to a target variable.

        Args:
            source : numpy array
                realisations of source variable, either a 2D numpy array where
                array dimensions represent [realisations x variable dimension]
                or a 1D array representing [realisations]
            target : numpy array
                realisations of target variable (similar to var1)

        Returns:
            float | numpy array
                average TE over all samples 

                ################################################ ???????????????????????????????????????????????????? local values ??????
                or local TE for individual
                samples if 'local_values'=True

        """
        ##################################################################################################### TODO

        source = self._ensure_one_dim_input(source)
        target = self._ensure_one_dim_input(target)

        # Check if number of points is sufficient for estimation.
        self.check_number_of_points(source.shape[0] -
                                     self.settings['source_target_delay'])

        assert (
            source.shape[0] == target.shape[0]
        ), f"Unequal number of observations (source: {source.shape[0]}, target: {target.shape[0]})"
        
        N = source.shape[0]


        # Normalise data
        if self.settings['normalise']:
            source = self._normalise_data(source)
            target = self._normalise_data(target)
            
        # Add noise to avoid duplicate points
        # Do not add noise inplace, because it would change the input data
        if self.settings['noise_level'] > 0:
            source = source + self._rng.normal(0, self.settings['noise_level'], source.shape)
            target = target + self._rng.normal(0, self.settings['noise_level'], target.shape)
        


        min_length = max(self.settings['history_source'] * self.settings['tau_source'] + 2,
                        self.settings['history_target'] * self.settings['tau_target'] + 2)
        
        if N < min_length:
            raise ValueError(f"Data too short: need at least {min_length} samples, got {N}")
        
        valids= np.array([N - self.settings['history_source'] * self.settings['tau_source'] - 1, N - self.settings['history_target'] * self.settings['tau_target'] - 1])
        
        min_valid = min(valids) 
        
        num_valid = min_valid - self.settings['source_target_delay']
        """
        starts = valids - num_valid
        #starts[0] = starts[0] -self.settings['source_target_delay']

        valid_vec_source = np.arange(num_valid)
        valid_vec_target = valid_vec_source + self.settings['source_target_delay']

        print(valids)
        print(min_valid)
        print(num_valid)
        print(starts)
        print("source: ", valid_vec_source[0]," : ",valid_vec_source[-1])
        print(len(valid_vec_source))
        print("tartget: ", valid_vec_target[0]," : ",valid_vec_target[-1])
        print(len(valid_vec_target))
        print("source: ",N - self.settings['history_source'] * self.settings['tau_source'] - 1)
        print("target: ", N - self.settings['history_target'] * self.settings['tau_target'] - 1)
        """

        if num_valid <= 0:
            raise ValueError(f"Not enough valid embedding vectors")
        

        
        #source_past, source_current = self.embed_past_current(source, valid_vec_source, self.settings['history_source'], self.settings['tau_source'])
        source_past, source_current = self.embed_past_current(source, num_valid, 0, self.settings['history_source'], self.settings['tau_source'])
        
        #target_past, target_current = self.embed_past_current(target, valid_vec_target, self.settings['history_target'], self.settings['tau_target'])
        target_past, target_current = self.embed_past_current(target, num_valid, self.settings['source_target_delay'], self.settings['history_target'], self.settings['tau_target'])

        
        if target_current.ndim == 1:
            target_current = target_current[:, None]
     
        # Compute distances to kth nearest neighbors in the joint space
        epsilon = self._compute_epsilon(
            np.hstack([target_current, target_past, source_past]), self.settings['kraskov_k']
        )
        
        k = self.settings['kraskov_k']

        
        # algorithm 1
        
        #eps_cor=0
        epsi = epsilon#np.nextafter(epsilon, 0)
        n_xy = self._compute_n(np.concatenate((target_past, source_past), axis=1), epsi)
        
        n_yf = self._compute_n(np.concatenate((target_current, target_past), axis=1), epsi)
        
        n_y = self._compute_n(target_past, epsi)


        # compute estimate
        
        te = digamma(k) + np.mean(digamma(n_y) - digamma(n_xy) - digamma(n_yf))

        ##################################################################################### TODO local value
        #print(digamma(k) + digamma(n_y) - digamma(n_xy) - digamma(n_yf))



        ######################################################################################################## TODO algorithm 2
        # algorithm 2
        """
        #eps_cor = 1e-14
        epsi=epsilon+1e-15
        #n_xy2 = self._compute_n(np.concatenate((target_pastX, source_pastX), axis=1), epsilon-eps_cor)
        n_xy2 = self._compute_n(np.concatenate((target_pastX, source_pastX), axis=1), epsi)
        
        #n_yf2 = self._compute_n(np.concatenate((target_currentX, target_pastX), axis=1), epsilon-eps_cor)
        n_yf2 = self._compute_n(np.concatenate((target_currentX, target_pastX), axis=1), epsi)

        #n_y2 = self._compute_n(target_pastX, epsilon-eps_cor)
        n_y2 = self._compute_n(target_pastX, epsi)

        #I2 = digamma(k) + digamma(len(target_pastX)) - np.mean( digamma(n_y2) - digamma(n_xy2) - digamma(n_yf2) )
        #I2 = digamma(k) - np.mean( digamma(n_y2) - digamma(n_xy2) - digamma(n_yf2) )
        I2 = digamma(k) - digamma(len(target_pastX)) - np.mean( digamma(n_yf2) + digamma(n_y2) - digamma(n_xy2) )
        
        print("alg 2: ", I2)

        ## I = digamma(k) + <digamma(n_y + 1) - digamma(n_xy + 1) - digamma(n_yf + 1)>  (some variants)
        ## We'll follow the commonly used expression:
        ## I = digamma(k) - (1/k) * sum( digamma(n_xy + 1) + digamma(n_yf + 1) - digamma(n_y + 1) ) + digamma(N)
        ##print(digamma(k) - (1/k) * sum( digamma(n_xy + 1) + digamma(n_yf + 1) - digamma(n_y + 1) ) + digamma(mn))
        print(digamma(k) - (1/k) * sum( digamma(n_xy2 + 1) + digamma(n_yf2 + 1) - digamma(n_y2 + 1) ) + digamma(N))
        ## However typical KSG form for conditional MI (see Frenzel & Pompe 2007) is:
        ## I = digamma(k) + <digamma(n_y) - digamma(n_xy) - digamma(n_yf)>    (with counts strictly within eps)
        print(digamma(k) + np.mean(digamma(n_y2) - digamma(n_xy2) - digamma(n_yf2)))
        ## To avoid off-by-one differences, we'll implement the widely used form:
        ## I = digamma(k) + (1/L) * sum( digamma(n_y + 1) - digamma(n_xy + 1) - digamma(n_yf + 1) ) 
        ##print(digamma(k) + (1/mn) * sum( digamma(n_y + 1) - digamma(n_xy + 1) - digamma(n_yf + 1) ) )
        print(digamma(k) + (1/N) * sum( digamma(n_y2) - digamma(n_xy2) - digamma(n_yf2) ) )

        print(digamma(k) + digamma(len(target_pastX)) - sum( digamma(n_y2) - digamma(n_xy2) - digamma(n_yf2) ) )
        print(digamma(k) + digamma(len(target_pastX)) - np.mean( digamma(n_y2) - digamma(n_xy2) - digamma(n_yf2) ) )

        """
        


        return te

        

################################################################################ TODO
class PythonDiscreteMI(PythonDiscrete):
    """Calculate MI with Python discrete-variable implementation.

    Calculate the mutual information (MI) between two variables. 

    Results are returned in bits.

    Args:
        settings : dict [optional]
            sets estimation parameters:

            
            - discretise_method : str [optional] - if and how to discretise
              incoming continuous data, can be 'max_ent' for maximum entropy
              binning, 'equal' for equal size bins, and 'none' if no binning is
              required (default='none')
            - n_discrete_bins : int [optional] - number of discrete bins/
              levels or the base of each dimension of the discrete variables
              (default=2). If set, this parameter overwrites/sets alph1 and
              alph2
            - alph1 : int [optional] - number of discrete bins/levels for var1
              (default=2, or the value set for n_discrete_bins)
            - alph2 : int [optional] - number of discrete bins/levels for var2
              (default=2, or the value set for n_discrete_bins)
            - lag_mi : int [optional] - time difference in samples to calculate
              the lagged MI between processes (default=0)
            
            ################################################ ???????????????????????????????????????????????????? local values ??????
            - local_values : bool [optional] - return local TE instead of
              average TE (default=False)
            
    """

    def __init__(self, settings=None):
        settings = self._check_settings(settings)
        # Set default alphabet sizes. Try to overwrite alphabet sizes with
        # number of bins for discretisation if provided, otherwise assume
        # binary variables.
        super().__init__(settings)
        self.settings.setdefault('lag_mi', int(0))
        try:
            n_discrete_bins = int(self.settings['n_discrete_bins'])
            self.settings['alph1'] = n_discrete_bins
            self.settings['alph2'] = n_discrete_bins
        except KeyError:
            pass  # Do nothing and use the default for alph_* set below
        self.settings.setdefault('alph1', int(2))
        self.settings.setdefault('alph2', int(2))

        

    def estimate(self, var1, var2):
        """Estimate mutual information.

        Args:
            var1 : numpy array
                realisations of first variable, either a 2D numpy array where
                array dimensions represent [realisations x variable dimension]
                or a 1D array representing [realisations], array type can be
                float (requires discretisation) or int
            var2 : numpy array
                realisations of the second variable (similar to var1)
            return_calc : boolean
                return the calculator used here as well as the numeric
                calculated value(s)

        Returns:
            float | numpy array
                average MI over all samples 

                ################################################ ???????????????????????????????????????????????????? local values ??????
                or local MI for individual
                samples if 'local_values'=True
        """
        # Check and remember the no. dimensions for each variable before
        # collapsing them into univariate arrays later.
        var1 = self._ensure_two_dim_input(var1)
        var2 = self._ensure_two_dim_input(var2)
        var1_dim = var1.shape[1]
        var2_dim = var2.shape[1]

        # Discretise variables if requested.
        var1, var2 = self._discretise_vars(var1, var2)

        # Then collapse any multivariates into univariate arrays:
        var1 = utils.combine_discrete_dimensions(var1, self.settings['alph1'])
        var2 = utils.combine_discrete_dimensions(var2, self.settings['alph2'])

        # Initialise estimator
        #base_for_var1 = int(np.power(self.settings['alph1'], var1_dim))
        #base_for_var2 = int(np.power(self.settings['alph2'], var2_dim))


        # Count joint and marginal frequencies
        joint_counts = Counter(zip(var1, var2))
        var1_counts = Counter(var1)
        var2_counts = Counter(var2)
        n = len(var1)
        
        # Calculate mutual information: I(X;Y) = Σ p(x,y) * log(p(x,y) / (p(x) * p(y)))
        mi = 0.0
        for (xi, yi), count in joint_counts.items():
            p_xy = count / n
            p_x = var1_counts[xi] / n
            p_y = var1_counts[yi] / n
            
            if p_xy > 0 and p_x > 0 and p_y > 0:
                mi += p_xy * math.log2(p_xy / (p_x * p_y))
        
        return mi


    def compute_joint_probs(data, indices):
        """
        Compute joint probability distribution over the variables at given indices.

        data: list of tuples (samples), each tuple is a multivariate discrete sample.
        indices: tuple of ints, the variable indices to consider.

        Returns: Counter of joint outcomes -> probability.
        """
        # collect joint values for the given indices
        joint_vals = tuple(
            tuple(sample[i] for i in indices) for sample in data
        )
        counts = Counter(joint_vals)
        n = len(data)
        probs = {vals: float(c) / n for vals, c in counts.items()}
        return probs










class PythonGaussianMI(PythonGaussian):
    """Calculate mutual information with python Gaussian implementation.

    Calculate the mutual information between two variables.

    Results are returned in nats.

    Args:
        settings : dict [optional]
            set estimator parameters:
            
            - lag_mi : int [optional] - time difference in samples to calculate
              the lagged MI between processes (default=0)
            
            ################################################ ???????????????????????????????????????????????????? local values ??????
            - local_values : False, 
            

    """
    def __init__(self, settings):
        """Initialise estimator with settings."""
        settings = self._check_settings(settings)
        settings.setdefault('lag_mi', int(0))
        super().__init__(settings)

    def estimate(self, var1: np.ndarray, var2: np.ndarray):
        """Estimate mutual information between var1 and var2

        Args:
            var1 : numpy array
                realisations of first variable, either a 2D numpy array where
                array dimensions represent [realisations x variable dimension]
                or a 1D array representing [realisations]
            var2 : numpy array
                realisations of the second variable (similar to var1)

        Returns:
            float | numpy array
                average MI over all samples 

                ################################################ ???????????????????????????????????????????????????? local values ??????
                or local MI for individual
                samples if 'local_values'=True
        """

        var1 = self._ensure_two_dim_input(var1)
        var2 = self._ensure_two_dim_input(var2)

        assert (
            var1.shape[0] == var2.shape[0]
        ), f"Unequal number of observations (var1: {var1.shape[0]}, var2: {var2.shape[0]})"

        # Normalise data
        if self.settings['normalise']:
            var1 = self._normalise_data(var1)
            var2 = self._normalise_data(var2)

        # Add noise to avoid duplicate points
        # Do not add noise inplace, because it would change the input data
        if self.settings['noise_level'] > 0:
            var1 = var1 + self._rng.normal(0, self._noise_level, var1.shape)
            var2 = var2 + self._rng.normal(0, self._noise_level, var2.shape)

        # Shift variables to calculate a lagged MI.
        if self.settings['lag_mi'] > 0:
            var1 = var1[:-self.settings['lag_mi'], :]
            var2 = var2[self.settings['lag_mi']:, :]


        Z = np.concatenate((var1, var2), axis=1)
        #Z = np.hstack([X, Y])

        cov = np.cov(Z, rowvar=False, bias=False)
        dx = var1.shape[1]
        dy = var2.shape[1]

        cov_x = cov[:dx, :dx]
        cov_y = cov[dx:, dx:]

        sign_z, logdet_z = np.linalg.slogdet(cov)
        sign_x, logdet_x = np.linalg.slogdet(cov_x)
        sign_y, logdet_y = np.linalg.slogdet(cov_y)

        if sign_z <= 0 or sign_x <= 0 or sign_y <= 0:
            raise ValueError("Covariance matrix is not positive definite enough.")

        mi = 0.5 * (logdet_x + logdet_y - logdet_z)
        return mi


class PythonGaussianCMI(PythonGaussian):
    """Calculate conditional mutual information with python Gaussian implementation.

    Computes the differential conditional mutual information of two
    multivariate sets of observations, conditioned on another, assuming that
    the probability distribution function for these observations is a
    multivariate Gaussian distribution.
    If no conditional is given (is None), the function returns the mutual
    information between var1 and var2.

    Results are returned in nats.

    Args:
        settings : dict [optional]
            sets estimation parameters:
            
            ################################################ ???????????????????????????????????????????????????? local values ??????
            - local_values : bool [optional] - return local TE instead of
              average TE (default=False)

    """

    def __init__(self, settings):
        """Initialise estimator with settings."""
        settings = self._check_settings(settings)
        super().__init__(settings)

    def estimate(self, var1: np.ndarray, var2: np.ndarray, conditional=None):
        """Estimate conditional mutual information between var1 and var2, given
        conditional.

        Args:
            var1 : numpy array
                realisations of first variable, either a 2D numpy array where
                array dimensions represent [realisations x variable dimension]
                or a 1D array representing [realisations]
            var2 : numpy array
                realisations of the second variable (similar to var1)
            conditional : numpy array [optional]
                realisations of the conditioning variable (similar to var), if
                no conditional is provided, return MI between var1 and var2

        Returns:
            float | numpy array
                average MI over all samples 

                ################################################ ???????????????????????????????????????????????????? local values ??????
        """

        # Return MI if no conditioning variable was provided.
        if conditional is None:
            if (self.est_mi is None):
                self.est_mi = PythonGaussianMI(self.settings)
            return self.est_mi.estimate(var1, var2)
        else:
            assert(conditional.size != 0), 'Conditional Array is empty.'

        var1 = self._ensure_two_dim_input(var1)
        var2 = self._ensure_two_dim_input(var2)
        conditional = self._ensure_two_dim_input(conditional)

        
        assert (
            var1.shape[0] == var2.shape[0] == conditional.shape[0]
        ), f"Unequal number of observations (var1: {var1.shape[0]}, var2: {var2.shape[0]}, conditional: {conditional.shape[0]})"


        # Normalise data
        if self.settings['normalise']:
            var1 = self._normalise_data(var1)
            var2 = self._normalise_data(var2)
            conditional = self._normalise_data(conditional)

        # Add noise to avoid duplicate points
        # Do not add noise inplace, because it would change the input data
        if self.settings['noise_level'] > 0:
            var1 = var1 + self._rng.normal(0, self.settings['noise_level'], var1.shape)
            var2 = var2 + self._rng.normal(0, self.settings['noise_level'], var2.shape)
            conditional = conditional + self._rng.normal(
                0, self._noise_level, conditional.shape
            )

        xyz = np.hstack([var1, var2, conditional])

        cov = np.cov(xyz, rowvar=False, bias=False)

        nx = 1
        ny = 1
        nz = 1

        ix = slice(0, nx)
        iy = slice(nx, nx + ny)
        iz = slice(nx + ny, nx + ny + nz)

        cov_xz = cov[np.ix_(list(range(nx)) + list(range(nx + ny, nx + ny + nz)),
                            list(range(nx)) + list(range(nx + ny, nx + ny + nz)))]
        cov_yz = cov[np.ix_(list(range(nx, nx + ny)) + list(range(nx + ny, nx + ny + nz)),
                            list(range(nx, nx + ny)) + list(range(nx + ny, nx + ny + nz)))]
        cov_z = cov[np.ix_(list(range(nx + ny, nx + ny + nz)),
                           list(range(nx + ny, nx + ny + nz)))]
        cov_xyz = cov

        # add tiny regularization for numerical stability
        eps = 1e-10
        cov_xz += eps * np.eye(cov_xz.shape[0])
        cov_yz += eps * np.eye(cov_yz.shape[0])
        cov_z += eps * np.eye(cov_z.shape[0])
        cov_xyz += eps * np.eye(cov_xyz.shape[0])

        det_xz = np.linalg.det(cov_xz)
        det_yz = np.linalg.det(cov_yz)
        det_z = np.linalg.det(cov_z)
        det_xyz = np.linalg.det(cov_xyz)

        cmi = 0.5 * np.log((det_xz * det_yz) / (det_z * det_xyz))

        return cmi


class PythonGaussianAIS(PythonGaussian):
    """Calculate active information storage with Python Gaussian implementation.

    Calculate active information storage (AIS) for some process using Python
    implementation of the Gaussian estimator. AIS is defined as the
    mutual information between the processes' past state and current value.

    The past state needs to be defined in the settings dictionary, where a past
    state is defined as a uniform embedding with parameters history and tau.
    The history describes the number of samples taken from a processes' past,
    tau describes the embedding delay, i.e., the spacing between every two
    samples from the processes' past.

    Results are returned in nats.

    Args:
        settings : dict
            sets estimation parameters:

            - history : int - number of samples in the processes' past used as
              embedding
            - tau : int [optional] - the processes' embedding delay (default=1)
            
            # ################################################################################################# TODO local values
            - local_values : bool [optional] - return local TE instead of
              average TE (default=False)
    """
    def __init__(self, settings):
        settings = self._check_settings(settings)
        # Check for history for AIS estimation.
        try:
            settings['history']
        except KeyError:
            raise RuntimeError('No history was provided for AIS estimation.')
        settings.setdefault('tau', 1)
        assert type(settings['history']) is int, (
                                            'History has to be an integer.')
        assert type(settings['tau']) is int, ('Tau has to be an integer.')

        super().__init__(settings)

    def estimate(self, process):
        """Estimate active information storage.

        Args:
            process : numpy array
                realisations of first variable, either a 2D numpy array where
                array dimensions represent [realisations x variable dimension]
                or a 1D array representing [realisations]

        Returns:
            float | numpy array
                average AIS over all samples 

        """

        process = self._ensure_one_dim_input(process)

        # len process
        n = len(process)

        min_length = self.settings['history'] * self.settings['tau'] + 2
        
        if n < min_length:
            raise ValueError(f"Data too short: need at least {min_length} samples, got {n}")
    
        num_valid = n - self.settings['history'] * self.settings['tau'] - 1
    
        if num_valid <= 0:
            raise ValueError(f"Not enough valid embedding vectors")
        
        past, current = self.embed_past_current(process, num_valid, self.settings['history'], self.settings['tau'])
        past = self.delay_embedding(process)

        est_mi=PythonGaussianMI(self.settings)
        mi = est_mi.estimate(var1=current, var2=past)
        
        return mi


################################################################ TODO
class PythonGaussianTE(PythonGaussian):
    """Calculate transfer entropy with Python Gaussian implementation.

    Calculate transfer entropy between a source and a target variable using
    Pathon implementation of the Gaussian estimator. Transfer entropy is
    defined as the conditional mutual information between the source's past
    state and the target's current value, conditional on the target's past.

    Past states need to be defined in the settings dictionary, where a past
    state is defined as a uniform embedding with parameters history and tau.
    The history describes the number of samples taken from a variable's past,
    tau descrices the embedding delay, i.e., the spacing between every two
    samples from the processes' past.

    Results are returned in nats.        

    Args:
        settings : dict
            sets estimation parameters:

            - history_target : int - number of samples in the target's past
              used as embedding
            - history_source  : int [optional] - number of samples in the
              source's past used as embedding (default=same as the target
              history)
            - tau_source : int [optional] - source's embedding delay
              (default=1)
            - tau_target : int [optional] - target's embedding delay
              (default=1)
            - source_target_delay : int [optional] - information transfer delay
              between source and target (default=1)
            
            ################################################ ???????????????????????????????????????????????????? local values ??????
            - local_values : bool [optional] - return local TE instead of
              average TE (default=False)


    """
    def __init__(self, settings):
        """Initialise estimator with settings."""
        settings = self._check_settings(settings)
        settings = self._set_te_defaults(settings)
        super().__init__(settings)

        # Check for currently unsupported settings
        if settings.get('local_values', False) or settings.get('algorithm_num', 1) != 1:
            raise ValueError('This estimator currently does not support local_values or algorithm_num arguments.')


    def _set_te_defaults(self, settings):
        """Set defaults for transfer entropy estimation."""
        try:
            history_target = settings['history_target']
        except KeyError:
            raise RuntimeError('No target history was provided for TE '
                               'estimation.')
        settings.setdefault('history_source', history_target)
        settings.setdefault('tau_target', 1)
        settings.setdefault('tau_source', 1)
        settings.setdefault('source_target_delay', 1)

        assert type(settings['tau_target']) is int, (
            'Target tau has to be an integer.')
        assert type(settings['tau_source']) is int, (
            'Source tau has to be an integer.')
        assert type(settings['history_target']) is int, (
            'Target history has to be an integer.')
        assert type(settings['history_source']) is int, (
            'Source history has to be an integer.')
        assert type(settings['source_target_delay']) is int, (
            'Source-target delay has to be an integer.')
        assert settings['tau_target'] >= 1, 'Target tau must be >= 1'
        assert settings['tau_source'] >= 1, 'Source tau must be >= 1'
        assert settings['history_target'] >= 0, 'Target history must be >= 0'
        assert settings['history_source'] >= 1, 'Source history must be >= 1'
        assert settings['source_target_delay'] >= 0, (
            'Source-target delay must be >= 0')
        return settings
    
    def check_number_of_points(self, n_points):
        """Sanity check for number of points going into the estimator."""
        if (n_points - 1) <= int(self.settings["kraskov_k"]):
            raise RuntimeError(
                f"Insufficient number of points ({n_points}) for the requested number of nearest neighbours "
                f"(kraskov_k: {self.settings['kraskov_k']})."
            )

    def estimate(self, source: np.ndarray, target: np.ndarray):
        """Estimate transfer entropy from a source to a target variable.

        Args:
            source : numpy array
                realisations of source variable, either a 2D numpy array where
                array dimensions represent [realisations x variable dimension]
                or a 1D array representing [realisations]
            target : numpy array
                realisations of target variable (similar to var1)

        Returns:
            float | numpy array
                average TE over all samples 

                ################################################ ???????????????????????????????????????????????????? local values ??????
                or local TE for individual
                samples if 'local_values'=True
        


        TE_{Y->X} = 0.5 * log det Sigma(X_t | X_past) / det Sigma(X_t | X_past, Y_past)
        where X_past = X_{t-lag}, Y_past = Y_{t-lag}
        
        """

        source = self._ensure_one_dim_input(source)
        target = self._ensure_one_dim_input(target)

        ############################################################################################### TODO
        # Check if number of points is sufficient for estimation.
        self.check_number_of_points(source.shape[0] -
                                     self.settings['source_target_delay'])

        assert (
            source.shape[0] == target.shape[0]
        ), f"Unequal number of observations (source: {source.shape[0]}, target: {target.shape[0]})"


        
        N = source.shape[0]

        min_length = max(self.settings['history_source'] * self.settings['tau_source'] + 2,
                        self.settings['history_target'] * self.settings['tau_target'] + 2)
        
        if N < min_length:
            raise ValueError(f"Data too short: need at least {min_length} samples, got {N}")
    
        num_valid = min(N - self.settings['history_source'] * self.settings['tau_source'] - 1,
            N - self.settings['history_target'] * self.settings['tau_target'] - 1)
    
        if num_valid <= 0:
            raise ValueError(f"Not enough valid embedding vectors")
        
        
        source_past, source_current = self.embed_past_current(source, num_valid, self.settings['history_source'], self.settings['tau_source'])
        
        target_past, target_current = self.embed_past_current(target, num_valid, self.settings['history_target'], self.settings['tau_target'])

        
        if target_current.ndim == 1:
            target_current = target_current[:, None]
     



        

        #target_f = target[max_shift:]
        #target_past = self.delay_embed(target[:-1], self.settings['history_target'], self.settings['tau_target'])
        #target_past = target_past[max((self.settings['history_target'] - 1) * self.settings['tau_target'], self.settings['source_target_delay']):]

        #source_past = self.delay_embed(source[:-1-self.settings['source_target_delay']], self.settings['history_source'], self.settings['tau_source'])
        #source_past = source_past[max((self.settings['history_source'] -1) * self.settings['tau_source'], 0):]

        #m = min(len(target_f), len(target_past), len(source_past))
        #target_f = target_f[:m]
        #target_past = target_past[:m]
        #source_past = source_past[:m]

        var_y = self._ols_resid_var(target_current, target_past)
        var_yx = self._ols_resid_var(target_current, np.column_stack([target_past, source_past]))
        te_nats = 0.5 * np.log(var_y / var_yx)
        #print(te_nats)
        te_bits = te_nats / np.log(2.0)
        #print(te_bits)

        
        return te_nats


    def takens_embed(x, m=1, tau=1):
        x = np.asarray(x, dtype=float).ravel()
        n = x.size - (m - 1) * tau
        if n <= 0:
            raise ValueError("Time series too short for the requested embedding.")
        return np.column_stack([x[i*tau:i*tau+n] for i in range(m)])

    def align_te_series(x, y, m_x=1, m_y=1, tau_x=1, tau_y=1, delay=1):
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        max_lag = max((m_x - 1) * tau_x, (m_y - 1) * tau_y + delay)
        n = min(len(x), len(y)) - max_lag - 1
        if n <= 0:
            raise ValueError("Time series too short for the requested delay/embedding.")
        xt = np.column_stack([x[max_lag - i*tau_x : max_lag - i*tau_x + n] for i in range(m_x)])
        yt_past = np.column_stack([y[max_lag - delay - i*tau_y : max_lag - delay - i*tau_y + n] for i in range(m_y)])
        x_next = x[max_lag + 1 : max_lag + 1 + n][:, None]
        return x_next, xt, yt_past

    def gaussian_entropy(cov):
        cov = np.asarray(cov, dtype=float)
        d = cov.shape[0]
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            raise np.linalg.LinAlgError("Covariance matrix must be positive definite.")
        return 0.5 * (d * np.log(2.0 * np.pi * np.e) + logdet)


    def delay_embed(self, x, dim, tau):
        x = np.asarray(x, dtype=float)
        n = len(x)
        m = n - (dim - 1) * tau
        if m <= 0:
            raise ValueError("Time series too short for the chosen embedding.")
        return np.column_stack([x[(dim - 1 - i) * tau:(dim - 1 - i) * tau + m] for i in range(dim)])

    def _ols_resid_var(self, y, X):
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.size == 0:
            resid = y - y.mean()
            return np.mean(resid ** 2)
        X = np.asarray(X, dtype=float)
        X1 = np.column_stack([np.ones(len(y)), X])
        beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
        resid = y - X1 @ beta
        return np.mean(resid ** 2)

    def delay_embedding(self, series, dimension, delay, step):
        series = list(series)
        n = len(series) - (dimension - 1) * delay
        if n <= 0:
            return []

        embedded = []
        for i in range(0, n, step):
            point = [series[i + j * delay] for j in range(dimension)]
            embedded.append(point)

        print("embedded:")
        print(len(embedded))
        print("embedded as array:")
        print(np.array(embedded, dtype=np.float64).shape)

        return np.array(embedded, dtype=np.float64)

        """
        #??????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        # Normalise data
        if self._normalise:
            source = self._normalise_data(source)
            target = self._normalise_data(target)
            
        # Add noise to avoid duplicate points
        # Do not add noise inplace, because it would change the input data
        if self._noise_level > 0:
            source = source + self._rng.normal(0, self._noise_level, source.shape)
            target = target + self._rng.normal(0, self._noise_level, target.shape)
        #??????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        
        
        ##################################################################################################### TODO
        Xt = source[lag:]
        Xp = source[:-lag]
        Yp = target[:-lag]

        num = conditional_logdet(Xt, Xp)
        den = conditional_logdet(Xt, _stack(Xp, Yp))
        te = 0.5 * (num - den)

        return te
        """

    """
    def _as_2d(a):
        a = np.asarray(a, dtype=float)
        if a.ndim == 1:
            a = a[:, None]
        te = a

        return te

    def _logdet_psd(a, eps=1e-12):
        a = np.asarray(a, dtype=float)
        a = 0.5 * (a + a.T)
        s, logdet = np.linalg.slogdet(a + eps * np.eye(a.shape[0]))
        if s <= 0:
            w = np.linalg.eigvalsh(a + eps * np.eye(a.shape[0]))
            return np.sum(np.log(np.maximum(w, eps)))
        return logdet

    def _cov(x):
        x = _as_2d(x)
        x = x - x.mean(axis=0, keepdims=True)
        return np.cov(x, rowvar=False, bias=False)

    def _stack(*arrays):
        arrays = [_as_2d(a) for a in arrays]
        return np.hstack(arrays)

    def _cov_block(*arrays):
        return _cov(_stack(*arrays))

    def conditional_logdet(target, cond):
        target = _as_2d(target)
        cond = _as_2d(cond)
        if cond.shape[1] == 0:
            return _logdet_psd(_cov(target))
        c_tt = _cov(target)
        c_cc = _cov(cond)
        c_tc = np.cov(_stack(target, cond), rowvar=False, bias=False)[:target.shape[1], target.shape[1]:]
        c_ct = c_tc.T
        schur = c_tt - c_tc @ np.linalg.pinv(c_cc) @ c_ct
        return _logdet_psd(schur)

    """




def common_estimate_surrogates_analytic(estimator, n_perm=200, **data):
    """Estimate the surrogate distribution analytically for PythonEstimator.

    Estimate the surrogate distribution analytically for a PythonEstimator
    which is_analytic_null_estimator(), by sampling estimates at random
    p-values in the analytic distribution.

    Args:
        estimator : a JidtEstimator object, which returns True to a call to
            its is_analytic_null_estimator() method
        n_perms : int
            number of permutations (default=200)
        data : numpy arrays
            realisations of random variables required for the calculation
            (varies between estimators, e.g. 2 variables for MI, 3 for CMI)

    Returns:
        float | numpy array
            n_perm surrogates of the average MI/CMI/TE over all samples
            under the null hypothesis of no relationship between var1 and
            var2 (in the context of conditional)
    """
    # Compute the statistical significance of the estimate to get an
    #  AnalyticMeasurementDistribution object:
    analytic_distribution = estimator.get_analytic_distribution(**data)
    # Then compute surrogates at n_perm random p-values
    surrogate_estimates = np.empty(n_perm)
    for perm in range(n_perm):
        surrogate_estimates[perm] = \
            analytic_distribution.computeEstimateForGivenPValue(
                np.random.random())
    return surrogate_estimates
