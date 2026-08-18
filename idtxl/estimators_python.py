"""Provide Python estimators."""
import numpy as np

from scipy.special import digamma
from scipy.spatial import cKDTree

from idtxl.estimator import Estimator
from idtxl.knn.knn_finder_factory import get_knn_finder

import idtxl.idtxl_utils as utils

from collections import Counter, defaultdict
import math
import scipy as sp
from scipy.stats import multivariate_normal, chi2, norm, kstest
from scipy.spatial import distance_matrix
from scipy.linalg import cholesky, solve_triangular, cho_solve

from idtxl.measurement_distributions_python import EmpiricalMeasurementDistribution, AnalyticalMeasurementDistribution, ChiSquareMeasurementDistribution

import multiprocessing
import random

from typing import Tuple, Optional

from idtxl.estimators_jidt import JidtKraskovMI, JidtKraskovCMI, JidtKraskovAIS, JidtKraskovTE, JidtKraskovCTE



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
        """Set default estimator settings."""
        self.settings = settings.copy()

    def _normalise_data(self, data: np.ndarray):
        """Standardise data to zero mean and unit variance."""
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    
    def computeStartTimeForFirstDestEmbedding(self, history_target, tau_target, history_source, tau_source, delay): 
        """get first time point for embedding"""        
        startTimeBasedOnTargetPast = (history_target - 1) * tau_target
        startTimeBasedOnSourcePast = (history_source - 1) * tau_source + delay - 1
        return max(startTimeBasedOnTargetPast, startTimeBasedOnSourcePast);

    def makeDelayEmbeddingVector(self, ts, history, tau, startFirstPoint, numEmbeddingVectors):
        """create past delay embedding vector of given data and settings """
        embedded_vector = np.zeros((numEmbeddingVectors, history))
        
        for t in range(startFirstPoint, numEmbeddingVectors+startFirstPoint):
            for i in range(history):
                embedded_vector[t - startFirstPoint, i] = ts[t - i * tau]

        return embedded_vector

    def makeDelayEmbeddingVectorCurrent(self, ts, history, startFirstPoint, numEmbeddingVectors):
        """create current delay embedding vector of given data and settings """
        embedded_vector = np.zeros((numEmbeddingVectors, history))

        for t in range(startFirstPoint, numEmbeddingVectors+startFirstPoint):
            for i in range(history):
                embedded_vector[t - startFirstPoint, i] = ts[t - i]

        return embedded_vector

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
    

    def _set_cte_defaults(self, settings):
        """Set defaults for conditional transfer entropy estimation."""

        settings.setdefault('history_conditional', settings['history_target'])
        settings.setdefault('tau_conditional', 1)
        settings.setdefault('conditional_target_delay', 1)
        
        assert type(settings['history_conditional']) is int, (
            'Conditional history has to be an integer.')
        assert type(settings['tau_conditional']) is int, (
            'Conditional tau has to be an integer.')
        assert settings['tau_conditional'] >= 1, 'Conditional tau must be >= 1'
        assert settings['history_conditional'] >= 1, 'Conditional history must be >= 1'
        assert settings['conditional_target_delay'] >= 0, (
            'Conditional-target delay must be >= 0')

        return settings
        

    def is_parallel(self):
        return False

            
###############################
# Kraskov estimators
###############################

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
            - theiler_t : int [optional] - no. next temporal neighbours ignored
              in KNN and range searches (default=0)
            - noise_level : float [optional] - random noise added to the data
              (default=1e-8)
            - num_threads : int | str [optional] - number of threads used for
              estimation (default='USE_ALL', note that this uses *all*
              available threads on the current machine)
            - knn_finder : str [optional] - knn algorithm to use, can be
              'scipy_ckdtree' (default), 'scipy_kdtree' , 'sklearn_kdtree', or 
              'sklearn_balltree'
            - local_values : bool [optional] - return local MI/TE instead of
              average MI/TE (default=False)
            
    """

    def __init__(self, settings):
        """Initialise estimator with settings."""
        settings.setdefault('kraskov_k', int(4))
        settings.setdefault('normalise', False)
        settings.setdefault('theiler_t', int(0))
        settings.setdefault('base', np.e)
        settings.setdefault('noise_level', 1e-8)
        settings.setdefault('num_threads', 'USE_ALL')
        settings.setdefault('knn_finder', 'scipy_ckdtree')
        settings.setdefault('local_values', False)
        settings.setdefault('algorithm_num', 1)
        
        super().__init__(settings)

        self.settings['kraskov_k'] = int(self.settings['kraskov_k'])
        self.settings['theiler_t'] = int(self.settings['theiler_t'])

        if self.settings['theiler_t'] > 0:
            if settings['knn_finder'] == 'numba_brute':
                raise ValueError('Theiler_t correction is not supproted for knn_finder numba_brute.')

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

    def _compute_epsilon(self, data: np.ndarray, k: int):
        """Compute the distance to the kth nearest neighbor for each point in x."""
        knn_finder = self._knn_finder_class(data, **self._knn_finder_settings)
        #return knn_finder.find_kth_neighbor(k, self.settings['theiler_t'])
        return knn_finder.find_all_dist_to_kth_neighbor(k)

    def _compute_n(self, data: np.ndarray, r: np.ndarray):
        """Count the number of neighbors within a given radius r for each point in x.
        Returns the number of neighbors plus one, because the point itself is included in the data.
        """
        knn_finder = self._knn_finder_class(data, **self._knn_finder_settings)
        #return knn_finder.count_all_neighbors(r, self.settings['theiler_t'], alg)
        return knn_finder.count_all_neighbors(r)
    
    def _compute_n_within(self, data: np.ndarray, r: np.ndarray):
        """Count the number of neighbors strictly within a given radius <= r for each point in x.
        Returns the number of neighbors plus one, because the point itself is included in the data.
        """
        knn_finder = self._knn_finder_class(data, **self._knn_finder_settings)
        #return knn_finder.count_all_neighbors_within(r, self.settings['theiler_t'], alg)
        return knn_finder.count_all_neighbors_within(r)
            
    def getCountsMI(self, var1, var2):
        """get all Counts for Kraskov MI calculation"""
            
        # Compute distances to kth nearest neighbors in the joint space
        epsilon = self._compute_epsilon(
            np.concatenate((var1, var2), axis=1), self.settings['kraskov_k']
        )   

        # Count neighbors stricly within eps in marginal spaces X, Y 
        n_c_var1 = self._compute_n_within(var1, epsilon)
        n_c_var2 = self._compute_n_within(var2, epsilon)
        
        return n_c_var1, n_c_var2

    def getCountsCMI(self, var1, var2, conditional):
        """get digamma values for CMI estimation"""
        
        # Compute distances to kth nearest neighbors in the joint space
        epsilon = self._compute_epsilon(
            np.concatenate((var1, var2, conditional), axis=1), self.settings['kraskov_k'])

        # Count neighbors strictly within eps in marginal spaces X, Y and Z    
        n_c_var1 = self._compute_n_within(np.concatenate((var1, conditional), axis=1), epsilon)
        n_c_var2 = self._compute_n_within(np.concatenate((var2, conditional), axis=1), epsilon)
        n_c = self._compute_n_within(conditional, epsilon)
        
        return n_c_var1, n_c_var2, n_c

    def is_analytic_null_estimator(self):
        return False


class PythonKraskovMI(PythonKraskov):
    """Estimate mutual information using Kraskov's estimator.

    Calculate the mutual information between two variables. 

    Results are returned in nats.
    
    Args:
        settings : dict [optional]
            set estimator parameters:

            - kraskov_k : int [optional] - no. nearest neighbours for KNN
              search (default=4)
            - base : float - base of returned values (default=np.e)
            - theiler_t : int [optional] - no. next temporal neighbours ignored
              in KNN and range searches (default=0)
            - normalise : bool [optional] - z-standardise data (default=False)
            - noise_level : float [optional] - random noise added to the data
              (default=1e-8)
            - rng_seed : int | None [optional] - random seed if noise level > 0
            - num_threads : int | str [optional] - number of threads used for
              estimation (default='USE_ALL', note that this uses *all*
              available threads on the current machine)
            - lag_mi : int [optional] - time difference in samples to calculate
              the lagged MI between processes (default=0)
            - knn_finder : str [optional] - knn algorithm to use, can be
              'scipy_ckdtree' (default), 'scipy_kdtree' , 'sklearn_kdtree', or 
              'sklearn_balltree'
            - local_values : bool [optional] - return local MI/TE instead of
              average MI/TE (default=False)
            - algorithm_num : int [optional] - which Kraskov algorithm (1 or 2)
              to use (default=1)            
    """

    def __init__(self, settings=None):
        """Initialise estimator with settings."""
        settings = self._check_settings(settings)
        settings.setdefault('lag_mi', 0)
        super().__init__(settings)
        
    def calculateLocalMI(self, var1, var2):
        """calculate lokal Kraskov MI"""

        n_c_var1, n_c_var2 = self.getCountsMI(var1, var2)

        mi = (digamma(self.settings['kraskov_k']) 
                + digamma(len(var1))
                - digamma(n_c_var1 + 1)
                - digamma(n_c_var2 + 1)
            ) / np.log(self.settings['base'])
    
        return mi
    
    def calculateAverageMI(self, var1, var2):
        """calculate Average Kraskov MI"""

        n_c_var1, n_c_var2 = self.getCountsMI(var1, var2)

        mi = (digamma(self.settings['kraskov_k']) 
                + digamma(len(var1))
                - np.mean(digamma(n_c_var1 + 1) + digamma(n_c_var2 + 1))
            ) / np.log(self.settings['base'])

        return mi

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
                average MI over all samples or local MI for individual
                samples if 'local_values'=True
        """

        # Pass to JidtKraskovMI if theiler_t != 0 or KSG2 is used
        if self.settings['algorithm_num'] != 1 or self.settings['theiler_t'] > 0: 
            print("PythonKraskovMI does not support algorithm_num=2 and theiler_t>0.")
            print("The data is passed to JidtKraskovMI.")
            est = JidtKraskovMI(self.settings)
            return est.estimate(var1, var2)
     
        # Check the input data
        var1 = self._ensure_two_dim_input(var1)
        var2 = self._ensure_two_dim_input(var2)
        
        # Check if number of points is sufficient for estimation.
        self._check_number_of_points(var1.shape[0])

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
            var1 = var1 + self._rng.normal(0, self.settings['noise_level'], var1.shape)
            var2 = var2 + self._rng.normal(0, self.settings['noise_level'], var2.shape)

        # Shift variables to calculate a lagged MI.
        if self.settings['lag_mi'] > 0:
            var1 = var1[:-self.settings['lag_mi'], :]
            var2 = var2[self.settings['lag_mi']:, :]
        
        # Compute MI
        if self.settings["local_values"]:
            return self.calculateLocalMI(var1, var2)
        else:

            #return np.mean(lmi)
            return self.calculateAverageMI(var1, var2)


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
            - base : float - base of returned values (default=np.e)
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
              'scipy_ckdtree' (default), 'scipy_kdtree' , 'sklearn_kdtree', or 
              'sklearn_balltree'
            - local_values : bool [optional] - return local MI/TE instead of
              average MI/TE (default=False)
            - algorithm_num : int [optional] - which Kraskov algorithm (1 or 2)
              to use (default=1)            
    """

    def __init__(self, settings=None):
        """Initialise estimator with settings."""
        settings = self._check_settings(settings)
        super().__init__(settings)

        self._knn_finder_settings = settings.get("knn_finder_settings", {})
        
        # Set number of threads
        num_threads = settings.get("num_threads", -1)
        if num_threads == "USE_ALL":
            num_threads = -1
        self._knn_finder_settings["num_threads"] = num_threads

        # Get KNN finder class
        self._knn_finder_name = settings.get("knn_finder", "scipy_ckdtree")
        self._knn_finder_class = get_knn_finder(self._knn_finder_name)

    def calculateLocalCMI(self, var1, var2, conditional):
        """calculate local Kraskov CMI"""
        n_c_var1, n_c_var2, n_c = self.getCountsCMI(var1, var2, conditional)
            
        cmi =(digamma(self.settings['kraskov_k'])
            + digamma(n_c + 1)
            - digamma(n_c_var1 + 1)
            - digamma(n_c_var2 + 1)
            ) / np.log(self.settings['base'])     

        return cmi
    
    def calculateAverageCMI(self, var1, var2, conditional):
        """calculate Average Kraskov CMI"""
        n_c_var1, n_c_var2, n_c = self.getCountsCMI(var1, var2, conditional)
        
        cmi = (
            digamma(self.settings['kraskov_k'])
            + np.mean(digamma(n_c + 1))
            - np.mean(digamma(n_c_var1 + 1))
            - np.mean(digamma(n_c_var2 + 1))
            ) / np.log(self.settings['base'])
    
        return cmi    

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
                average CMI over all samples or local CMI for individual
                samples if 'local_values'=True
                
        """
        # Return MI if no conditioning variable was provided.
        if conditional is None:
            #if (self.est_mi is None):
            self.est_mi = PythonKraskovMI(self.settings)
            return self.est_mi.estimate(var1, var2)
        else:
            assert(conditional.size != 0), 'Conditional Array is empty.'

        # Pass to JidtKraskovCMI if theiler_t != 0 or KSG2 is used
        if self.settings['algorithm_num'] != 1 or self.settings['theiler_t'] > 0: 
            print("PythonKraskovCMI does not support algorithm_num=2 and theiler_t>0.")
            print("The data is passed to JidtKraskovCMI.")
            est = JidtKraskovCMI(self.settings)
            return est.estimate(var1, var2)
     
        # Check the input data
        var1 = self._ensure_two_dim_input(var1)
        var2 = self._ensure_two_dim_input(var2)
        conditional = self._ensure_two_dim_input(conditional)
        
        # Check if number of points is sufficient for estimation.
        self._check_number_of_points(var1.shape[0])
        
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

        # Compute CMI
        if self.settings["local_values"]:
            cmi = self.calculateLocalCMI(var1, var2, conditional)
        else:
            cmi = self.calculateAverageCMI(var1, var2, conditional)
            
        return cmi
            

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
            - base : float - base of returned values (default=np.e)
            - normalise : bool [optional] - z-standardise data (default=False)
            - theiler_t : int [optional] - no. next temporal neighbours ignored
              in KNN and range searches (default=0)
            - noise_level : float [optional] - random noise added to the data
              (default=1e-8)
            - rng_seed : int | None [optional] - random seed if noise level > 0
            - num_threads : int | str [optional] - number of threads used for
              estimation (default='USE_ALL', note that this uses *all*
              available threads on the current machine)
            - local_values : bool [optional] - return local TE instead of
              average TE (default=False)
            - knn_finder : str [optional] - knn algorithm to use, can be
              'scipy_ckdtree' (default), 'scipy_kdtree' , 'sklearn_kdtree', or 
              'sklearn_balltree'
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
        # Pass to JidtKraskovAIS if theiler_t != 0 or KSG2 is used
        if self.settings['algorithm_num'] != 1 or self.settings['theiler_t'] > 0: 
            print("PythonKraskovAIS does not support algorithm_num=2 and theiler_t>0.")
            print("The data is passed to JidtKraskovAIS.")
            est = JidtKraskovAIS(self.settings)
            return est.estimate(process)
     
        # Check the input data
        process = self._ensure_one_dim_input(process)

        # Check if number of points is sufficient for estimation.
        self._check_number_of_points(process.shape[0])

        # Normalise data
        if self.settings['normalise']:
            process = self._normalise_data(process)

        # Add noise to avoid duplicate points
        # Do not add noise inplace, because it would change the input data
        if self.settings['noise_level'] > 0:
            process = process + self._rng.normal(0, self.settings['noise_level'], process.shape)

        startFirstPoint = (self.settings['history']-1) * self.settings['tau'] 

        process_current = self.makeDelayEmbeddingVectorCurrent(process,
            1,
            startFirstPoint + 1,
            process.shape[0] - startFirstPoint - 1)

        process_past = self.makeDelayEmbeddingVector(process, 
            self.settings['history'], 
            self.settings['tau'], 
            startFirstPoint, 
            process.shape[0] - startFirstPoint - 1)
        
        if self.settings['local_values']:
            ais = PythonKraskovMI.calculateLocalMI(self, process_past, process_current)
            # correction to compare with JidtGaussianTE results
            ais = np.hstack([np.zeros(startFirstPoint+1), ais])

        else:
            ais = PythonKraskovMI.calculateAverageMI(self, process_past, process_current)

        return ais


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
            - local_values : bool [optional] - return local TE instead of
              average TE (default=False)
            - theiler_t : int [optional] - no. next temporal neighbours ignored
              in KNN and range searches (default=0)
            - knn_finder : str [optional] - knn algorithm to use, can be
              'scipy_ckdtree' (default), 'scipy_kdtree' , 'sklearn_kdtree', or 
              'sklearn_balltree'
            - algorithm_num : int [optional] - which Kraskov algorithm (1 or 2)
              to use (default=1)
    """
    def __init__(self, settings):
        """Initialise estimator with settings."""
        settings = self._check_settings(settings)
        settings = self._set_te_defaults(settings)
        super().__init__(settings)
        
        self._knn_finder_settings = settings.get("knn_finder_settings", {})
        
        # Set number of threads
        num_threads = settings.get("num_threads", -1)
        if num_threads == "USE_ALL":
            num_threads = -1
        self._knn_finder_settings["num_threads"] = num_threads

        # Get KNN finder class
        self._knn_finder_name = settings.get("knn_finder", "scipy_kdtree")
        self._knn_finder_class = get_knn_finder(self._knn_finder_name)
     
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
                average TE over all samples or local TE for individual
                samples if 'local_values'=True

        """
        # Pass to JidtKraskovTE if theiler_t != 0 or KSG2 is used
        if self.settings['algorithm_num'] != 1 or self.settings['theiler_t'] > 0: 
            print("PythonKraskovTE does not support algorithm_num=2 and theiler_t>0.")
            print("The data is passed to JidtKraskovAIS.")
            est = JidtKraskovTE(self.settings)
            return est.estimate(source, target)
     
        # Check the input data
        source = self._ensure_one_dim_input(source)
        target = self._ensure_one_dim_input(target)

        # Check if number of points is sufficient for estimation.
        self._check_number_of_points(source.shape[0] -
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
        
        # delay embedding
        startFirstPoint = self.computeStartTimeForFirstDestEmbedding(
            self.settings['history_target'],
            self.settings['tau_target'],
            self.settings['history_source'],
            self.settings['tau_source'],
            self.settings['source_target_delay'],
            )
        target_past = self.makeDelayEmbeddingVector(target, 
            self.settings['history_target'], 
            self.settings['tau_target'], 
            startFirstPoint, 
            target.shape[0] - startFirstPoint - 1)
        target_current = self.makeDelayEmbeddingVectorCurrent(target,
            1,
            startFirstPoint + 1,
            target.shape[0] - startFirstPoint - 1)
        
        source_past = self.makeDelayEmbeddingVector(source,
            self.settings['history_source'],
            self.settings['tau_source'],
            startFirstPoint + 1 - self.settings['source_target_delay'],
            source.shape[0] - startFirstPoint - 1)
            
        if self.settings['local_values']:
            te = PythonKraskovCMI.calculateLocalCMI(self, source_past, target_current, target_past)
            # correction to compare with JidtKraskovTE results
            te = np.hstack([np.zeros(startFirstPoint+1), te])

        else:
            te = PythonKraskovCMI.calculateAverageCMI(self, source_past, target_current, target_past)
            
        return te
        

class PythonKraskovCTE(PythonKraskov):
    """Calculate conditional transfer entropy with Python Gaussian 
    implementation.
    
    Calculate transfer entropy between a source and a target variable using
    Pathon implementation of the Gaussian estimator. Transfer entropy is
    defined as the conditional mutual information between the source's past
    state and the target's current value, conditional on the target's and 
    another conditional's past
    .

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
            - history_conditional  : int [optional] - number of samples in the
              conditional's past used as embedding (default=same as the target
              history)
            - tau_source : int [optional] - source's embedding delay
              (default=1)
            - tau_target : int [optional] - target's embedding delay
              (default=1)
            - tau_conditional : int [optional] - conditional's embedding delay
              (default=1)
            - source_target_delay : int [optional] - information transfer delay
              between source and target (default=1)
            - conditional_target_delay : int [optional] - information transfer delay
              between conditional and target (default=1)
            - theiler_t : int [optional] - no. next temporal neighbours ignored
              in KNN and range searches (default=0)
            - knn_finder : str [optional] - knn algorithm to use, can be
              'scipy_ckdtree' (default), 'scipy_kdtree' , 'sklearn_kdtree', or 
              'sklearn_balltree'
            - local_values : bool [optional] - return local TE instead of
              average TE (default=False)
            - algorithm_num : int [optional] - which Kraskov algorithm (1 or 2)
              to use (default=1)
    """
    def __init__(self, settings):
        """Initialise estimator with settings."""
        settings = self._check_settings(settings)
        settings = self._set_te_defaults(settings)
        settings = self._set_cte_defaults(settings)
        super().__init__(settings)

    def estimate(self, source: np.ndarray, target: np.ndarray, conditional=None):
        """Estimate conditional transfer entropy from a source to a target variable
        conditioned on another.

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
        
        """

        # Return TE if no conditioning variable was provided.
        if conditional is None:
            self.est_mi = PythonKraskovTE(self.settings)
            return self.est_mi.estimate(source, target)
        else:
            assert(conditional.size != 0), 'Conditional Array is empty.'

        # Pass to JidtKraskovCTE if theiler_t != 0 or KSG2 is used
        if self.settings['algorithm_num'] != 1 or self.settings['theiler_t'] > 0: 
            print("PythonKraskovCTE does not support algorithm_num=2 and theiler_t>0.")
            print("The data is passed to JidtKraskovCTE.")
            est = JidtKraskovCTE(self.settings)
            return est.estimate(source, target, conditional)
     
        # check the imput data
        source = self._ensure_one_dim_input(source)
        target = self._ensure_one_dim_input(target)
        conditional = self._ensure_one_dim_input(conditional)

        # Check if number of points is sufficient for estimation.
        self._check_number_of_points(source.shape[0] -
                                     self.settings['source_target_delay'])
        assert (
            source.shape[0] == target.shape[0] == conditional.shape[0]
        ), f"Unequal number of observations (source: {source.shape[0]}, target: {conditional.shape[0]}, target: {conditional.shape[0]})"

        # delay embedding
        startTimeBasedOnTargetPast = (self.settings['history_target'] - 1) * self.settings['tau_target']
        startTimeBasedOnSourcePast = (self.settings['history_source'] - 1) * self.settings['tau_source'] + self.settings['source_target_delay'] - 1
        startTimeBasedOnConditionalPast = (self.settings['history_conditional'] - 1) * self.settings['tau_conditional'] + self.settings['conditional_target_delay'] - 1
        
        startFirstPoint = max(startTimeBasedOnTargetPast, startTimeBasedOnSourcePast, startTimeBasedOnConditionalPast)

        target_past = self.makeDelayEmbeddingVector(target, 
            self.settings['history_target'], 
            self.settings['tau_target'], 
            startFirstPoint, 
            target.shape[0] - startFirstPoint - 1)
        target_current = self.makeDelayEmbeddingVectorCurrent(target,
            1,
            startFirstPoint + 1,
            target.shape[0] - startFirstPoint - 1)
        source_past = self.makeDelayEmbeddingVector(source,
            self.settings['history_source'],
            self.settings['tau_source'],
            startFirstPoint + 1 - self.settings['source_target_delay'],
            source.shape[0] - startFirstPoint - 1)

        conditional_past = self.makeDelayEmbeddingVector(conditional,
            self.settings['history_conditional'],
            self.settings['tau_conditional'],
            startFirstPoint + 1 - self.settings['conditional_target_delay'],
            conditional.shape[0] - startFirstPoint - 1)
        
        # combine target_current and conditional_past as conditional for CMI
        condCombine = np.hstack([target_past, conditional_past])
       
        if self.settings['local_values']:
            cte = PythonKraskovCMI.calculateLocalCMI(self, source_past, target_current, condCombine)
            cte = np.hstack([np.zeros(startFirstPoint+1), cte])
        else:
            cte = PythonKraskovCMI.calculateAverageCMI(self, source_past, target_current, condCombine)
            
        return cte



###############################
# Gaussian estimators
###############################

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
            - local_values : bool [optional] - return local MI/TE instead of
              average MI/TE (default=False)
    """

    def __init__(self, settings):
        settings.setdefault('local_values', False)
        settings.setdefault('normalise', False)
        settings.setdefault('noise_level', 0)
        super().__init__(settings)

        self.actualValue = None

        self.surr_est_type = "fast" 
        
    def logdet_cholesky(self, cov):
        """
        Stable log-determinant via Cholesky.
        cov must be symmetric positive definite.
        """
        L = cholesky(cov, lower=True, check_finite=False)
        return 2.0 * np.sum(np.log(np.diag(L)))

    def cov_reg(self, X, eps=1e-10):
        C = np.cov(X, rowvar=False, bias=True)
        if C.ndim != 0:
            return C + eps * np.eye(C.shape[0])
        else:
            return C
    
    def logpdf_gaussian(self, X, mu, cov):
        X = np.atleast_2d(X)
        d = cov.shape[0]
        L = cholesky(cov, lower=True, check_finite=False)
        XC = (X - mu).T
        Y = solve_triangular(L, XC, lower=True, check_finite=False)
        maha = np.sum(Y * Y, axis=0)
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        return -0.5 * (d * np.log(2.0 * np.pi) + logdet + maha)

    def is_analytic_null_estimator(self):
        return True

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


class PythonGaussianMI(PythonGaussian):
    """Calculate mutual information with python Gaussian implementation.

    Calculate the mutual information between two variables.

    Results are returned in nats.

    Args:
        settings : dict [optional]
            set estimator parameters:
            
            - lag_mi : int [optional] - time difference in samples to calculate
              the lagged MI between processes (default=0)
            - local_values : bool [optional] - return local TE instead of
              average TE (default=False) 
            

    """
    def __init__(self, settings=None):
        """Initialise estimator with settings."""
        settings = self._check_settings(settings)
        super().__init__(settings)
        self.settings.setdefault('lag_mi', int(0))
        
    def calculateAverageMI(self, var1, var2):
        """calculate avarage mutual information for gaussian data"""

        xy = np.hstack([var1, var2])
        
        cov_xy = np.cov(xy, rowvar=False, bias=False)
        cov_x = np.cov(var1, rowvar=False, bias=False)
        cov_y = np.cov(var2, rowvar=False, bias=False)
        
        if cov_x.ndim == 0:
            cov_x = np.array([[cov_x]])
        if cov_y.ndim == 0:
            cov_y = np.array([[cov_y]])
        
        ld_xy = self.logdet_cholesky(cov_xy)
        ld_x  = self.logdet_cholesky(cov_x)
        ld_y  = self.logdet_cholesky(cov_y)
        
        mi = 0.5 * (ld_x + ld_y - ld_xy)

        return mi

    def calculateLocalMI(self, var1, var2):
        """calculate avarage mutual information for gaussian data"""

        xy = np.hstack([var1, var2])
        
        eps=1e-10
        cov_xy = self.cov_reg(xy, eps)

        dx = var1.shape[1]

        cov_x = cov_xy[:dx, :dx]
        cov_y = cov_xy[dx:, dx:]

        mu_xy = xy.mean(axis=0)
        mu_x = var1.mean(axis=0)
        mu_y = var2.mean(axis=0)
        
        l_xy = self.logpdf_gaussian(xy, mu_xy, cov_xy)
        l_x  = self.logpdf_gaussian(var1,  mu_x,  cov_x)
        l_y  = self.logpdf_gaussian(var2,  mu_y,  cov_y)

        mi = l_xy -l_x - l_y 
        
        return mi
        
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
                average MI over all samples or local MI for individual
                samples if 'local_values'=True
        """
        # Check the input data
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

        self.n_samples = var1.shape[0]
        self.var1_dim = var1.shape[1]
        self.var2_dim = var2.shape[1]

        if self.settings['local_values']:
            mi = self.calculateLocalMI(var1, var2)
            self.actualValue = np.mean(mi)
        else:
            mi = self.calculateAverageMI(var1, var2)
            self.actualValue = mi

        return mi
    
    def computeSignificance(self):
        C = ChiSquareMeasurementDistribution()
        C.ChiSquareMeasurementDistribution(self.actualValue,
                self.n_samples,
                self.var1_dim * self.var2_dim,
                False,
                self.surr_est_type)
        return C

    def get_analytic_distribution(self, var1, var2):
        """Return a Python AnalyticNullDistribution object.

        Required so that our estimate_surrogates_analytic method can use the
        common_estimate_surrogates_analytic() method, where data is formatted
        as per the estimate method for this estimator.

        Args:
            var1 : numpy array
                realisations of first variable, either a 2D numpy array where
                array dimensions represent [realisations x variable dimension]
                or a 1D array representing [realisations]
            var2 : numpy array
                realisations of the second variable (similar to var1)
            conditional : numpy array [optional]
                realisations of the conditioning variable (similar to var), if
                no conditional is provided, return MI between var1 and var2#

        Returns:
            idtxl calculator that was used here
        """
        self.estimate(var1, var2)
        return self.computeSignificance()


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
            
            - local_values : bool [optional] - return local TE instead of
              average TE (default=False)

    """

    def __init__(self, settings=None):
        """Initialise estimator with settings."""
        settings = self._check_settings(settings)
        super().__init__(settings)
        self.est_mi = None
    
    def calculateAverageCMI(self, var1, var2, conditional):
        """calculate avarage conditional mutual information for gaussian data"""

        if conditional.ndim == 1:
            conditional = conditional[:,None]

        xyz = np.hstack([var1, var2, conditional])
        xz = np.hstack([var1, conditional])
        yz = np.hstack([var2, conditional])
        
        eps=1e-10
        cov_xyz = self.cov_reg(xyz, eps)
        cov_xz  = self.cov_reg(xz, eps)
        cov_yz  = self.cov_reg(yz, eps)
        cov_z   = self.cov_reg(conditional, eps)

        # Handle 1D Z cleanly
        if cov_z.ndim == 0:
            cov_z = np.array([[cov_z]])

        ld_xyz = self.logdet_cholesky(cov_xyz)
        ld_xz  = self.logdet_cholesky(cov_xz)
        ld_yz  = self.logdet_cholesky(cov_yz)
        ld_z   = self.logdet_cholesky(cov_z)

        cmi = 0.5 * (ld_xz + ld_yz - ld_z - ld_xyz)

        return cmi

    def calculateLocalCMI(self, var1, var2, conditional):
        """calculate avarage conditional mutual information for gaussian data"""

        if conditional.ndim == 1:
            conditional = conditional[:,None]
        
        xyz = np.hstack([var1, var2, conditional])
        xz = np.hstack([var1, conditional])
        yz = np.hstack([var2, conditional])

        mu_xyz = xyz.mean(axis=0)
        mu_xz = xz.mean(axis=0)
        mu_yz = yz.mean(axis=0)
        mu_z = conditional.mean(axis=0)

        eps=1e-10
        cov_xyz = self.cov_reg(xyz, eps)
        cov_xz  = self.cov_reg(xz, eps)
        cov_yz  = self.cov_reg(yz, eps)
        cov_z   = self.cov_reg(conditional, eps)
        
        # Handle 1D Z cleanly
        if cov_z.ndim == 0:
            cov_z = np.array([[cov_z]])

        l_xyz = self.logpdf_gaussian(xyz, mu_xyz, cov_xyz)
        l_xz  = self.logpdf_gaussian(xz,  mu_xz,  cov_xz)
        l_yz  = self.logpdf_gaussian(yz,  mu_yz,  cov_yz)
        l_z   = self.logpdf_gaussian(conditional,   mu_z,   cov_z)
        
        lcmi = l_xyz + l_z - l_xz - l_yz

        return lcmi

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
                average CMI over all samples or local CMI for individual
                samples if 'local_values'=True

        """

        # Return MI if no conditioning variable was provided.
        if conditional is None:
            #if (self.est_mi is None):
            self.est_mi = PythonGaussianMI(self.settings)
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

        self.n_samples = var1.shape[0]
        self.var1_dim = var1.shape[1]
        self.var2_dim = var2.shape[1]

        if self.settings['local_values']:
            cmi = self.calculateLocalCMI(var1, var2, conditional)
            self.actualValue = np.mean(cmi)
        else:
            cmi = self.calculateAverageCMI(var1, var2, conditional)
            self.actualValue = cmi

        return cmi

    def computeSignificance(self):
        C = ChiSquareMeasurementDistribution()
        C.ChiSquareMeasurementDistribution(self.actualValue,
                self.n_samples,
                self.var1_dim * self.var2_dim,
                False,
                self.surr_est_type)
        return C

    def get_analytic_distribution(self, var1, var2, conditional=None):
        """Return a Python AnalyticNullDistribution object.

        Required so that our estimate_surrogates_analytic method can use the
        common_estimate_surrogates_analytic() method, where data is formatted
        as per the estimate method for this estimator.

        Args:
            var1 : numpy array
                realisations of first variable, either a 2D numpy array where
                array dimensions represent [realisations x variable dimension]
                or a 1D array representing [realisations]
            var2 : numpy array
                realisations of the second variable (similar to var1)
            conditional : numpy array [optional]
                realisations of the conditioning variable (similar to var), if
                no conditional is provided, return MI between var1 and var2#

        Returns:
            idtxl calculator that was used here
        """
        if (conditional is None):
            mi = PythonGaussianMI(self.settings)
            mi.estimate(var1, var2)
            return mi.computeSignificance()
        else:
            self.estimate(var1, var2, conditional)
            return self.computeSignificance()


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
        # Check the input data
        process = self._ensure_one_dim_input(process)

        startFirstPoint = (self.settings['history']-1) * self.settings['tau'] 

        process_current = self.makeDelayEmbeddingVectorCurrent(process,
            1,
            startFirstPoint + 1,
            process.shape[0] - startFirstPoint - 1)

        process_past = self.makeDelayEmbeddingVector(process, 
            self.settings['history'], 
            self.settings['tau'], 
            startFirstPoint, 
            process.shape[0] - startFirstPoint - 1)
        
        self.n_samples = process_current.shape[0]
        self.process_current_dim = process_current.shape[1]
        self.process_past_dim = process_past.shape[1]

        if self.settings['local_values']:
            ais = PythonGaussianMI.calculateLocalMI(self, process_current, process_past)
            # correction to compare with JidtGaussianTE results
            ais = np.hstack([np.zeros(startFirstPoint+1), ais])
            self.actualValue = np.mean(ais)
        else:
            ais = PythonGaussianMI.calculateAverageMI(self, process_current, process_past)
            self.actualValue = ais
        
        return ais

    def computeSignificance(self):
        C = ChiSquareMeasurementDistribution()
        C.ChiSquareMeasurementDistribution(self.actualValue,
                self.n_samples,
                self.process_current_dim * self.process_past_dim,
                False,
                self.surr_est_type)
        return C

    def get_analytic_distribution(self, process):
        """Return a Python AnalyticNullDistribution object.

        Required so that our estimate_surrogates_analytic method can use the
        common_estimate_surrogates_analytic() method, where data is formatted
        as per the estimate method for this estimator.

        Args:
            var1 : numpy array
                realisations of first variable, either a 2D numpy array where
                array dimensions represent [realisations x variable dimension]
                or a 1D array representing [realisations]
            var2 : numpy array
                realisations of the second variable (similar to var1)
            conditional : numpy array [optional]
                realisations of the conditioning variable (similar to var), if
                no conditional is provided, return MI between var1 and var2#

        Returns:
            idtxl calculator that was used here
        """
        self.estimate(process)
        return self.computeSignificance()


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
            - local_values : bool [optional] - return local TE instead of
              average TE (default=False)


    """
    def __init__(self, settings):
        """Initialise estimator with settings."""
        settings = self._check_settings(settings)
        settings = self._set_te_defaults(settings)
        super().__init__(settings)

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
                average TE over all samples or local TE for individual
                samples if 'local_values'=True
        
        TE_{Y->X} = 0.5 * log det Sigma(X_t | X_past) / det Sigma(X_t | X_past, Y_past)
        where X_past = X_{t-lag}, Y_past = Y_{t-lag}
        """
        # Check the input data
        source = self._ensure_one_dim_input(source)
        target = self._ensure_one_dim_input(target)

        assert (
            source.shape[0] == target.shape[0]
        ), f"Unequal number of observations (source: {source.shape[0]}, target: {target.shape[0]})"

     
        # delay embedding
        startFirstPoint = self.computeStartTimeForFirstDestEmbedding(
            self.settings['history_target'],
            self.settings['tau_target'],
            self.settings['history_source'],
            self.settings['tau_source'],
            self.settings['source_target_delay'],
            )

        target_past = self.makeDelayEmbeddingVector(target, 
            self.settings['history_target'], 
            self.settings['tau_target'], 
            startFirstPoint, 
            target.shape[0] - startFirstPoint - 1)
        target_current = self.makeDelayEmbeddingVectorCurrent(target,
            1,
            startFirstPoint + 1,
            target.shape[0] - startFirstPoint - 1)
        
        source_past = self.makeDelayEmbeddingVector(source,
            self.settings['history_source'],
            self.settings['tau_source'],
            startFirstPoint + 1 - self.settings['source_target_delay'],
            source.shape[0] - startFirstPoint - 1)
        
        self.n_samples = source.shape[0]
        self.source_past_dim = source_past.shape[1]
        self.target_current_dim = target_current.shape[1]
        #self.target_past_dim = target_past.shape[1]

        if self.settings['local_values']:
            te = PythonGaussianCMI.calculateLocalCMI(self, source_past, target_current, target_past)
            # correction to compare with JidtGaussianTE results
            te = np.hstack([np.zeros(startFirstPoint+1), te])
            self.actualValue = np.mean(te)

        else:
            te = PythonGaussianCMI.calculateAverageCMI(self, source_past, target_current, target_past)
            self.actualValue = te

        return te

    def computeSignificance(self):
        C = ChiSquareMeasurementDistribution()
        C.ChiSquareMeasurementDistribution(self.actualValue,
                self.n_samples,
                self.source_past_dim * self.target_current_dim,
                False,
                self.surr_est_type)
        return C

    def get_analytic_distribution(self, source, target):
        """Return a Python AnalyticNullDistribution object.

        Required so that our estimate_surrogates_analytic method can use the
        common_estimate_surrogates_analytic() method, where data is formatted
        as per the estimate method for this estimator.

        Args:
            var1 : numpy array
                realisations of first variable, either a 2D numpy array where
                array dimensions represent [realisations x variable dimension]
                or a 1D array representing [realisations]
            var2 : numpy array
                realisations of the second variable (similar to var1)
            conditional : numpy array [optional]
                realisations of the conditioning variable (similar to var), if
                no conditional is provided, return MI between var1 and var2#

        Returns:
            idtxl calculator that was used here
        """
        self.estimate(source, target)
        return self.computeSignificance()


class PythonGaussianCTE(PythonGaussian):
    """Calculate conditional transfer entropy with Python Gaussian 
    implementation.

    Calculate transfer entropy between a source and a target variable using
    Pathon implementation of the Gaussian estimator. Transfer entropy is
    defined as the conditional mutual information between the source's past
    state and the target's current value, conditional on the target's and 
    another conditional'spast.

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
            - history_conditional  : int [optional] - number of samples in the
              conditional's past used as embedding (default=same as the target
              history)
            - tau_source : int [optional] - source's embedding delay
              (default=1)
            - tau_target : int [optional] - target's embedding delay
              (default=1)
            - tau_conditional : int [optional] - conditional's embedding delay
              (default=1)
            - source_target_delay : int [optional] - information transfer delay
              between source and target (default=1)
            - conditional_target_delay : int [optional] - information transfer delay
              between conditional and target (default=1)
            - local_values : bool [optional] - return local TE instead of
              average TE (default=False)


    """
    def __init__(self, settings):
        """Initialise estimator with settings."""
        settings = self._check_settings(settings)
        settings = self._set_te_defaults(settings)
        settings = self._set_cte_defaults(settings)
        super().__init__(settings)
        
    def estimate(self, source: np.ndarray, target: np.ndarray, conditional=None):
        """Estimate conditional transfer entropy from a source to a target variable
        conditioned on another.

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
        
        """
        # Return TE if no conditioning variable was provided.
        if conditional is None:
            est = PythonGaussianTE(self.settings)
            return est.estimate(source, target)
        else:
            assert(conditional.size != 0), 'Conditional Array is empty.'

        # Check the input data
        source = self._ensure_one_dim_input(source)
        target = self._ensure_one_dim_input(target)
        conditional = self._ensure_one_dim_input(conditional)

        assert (
            source.shape[0] == target.shape[0] == conditional.shape[0]
        ), f"Unequal number of observations (source: {source.shape[0]}, target: {conditional.shape[0]}, target: {conditional.shape[0]})"

        # delay embedding
        startTimeBasedOnTargetPast = (self.settings['history_target'] - 1) * self.settings['tau_target']
        startTimeBasedOnSourcePast = (self.settings['history_source'] - 1) * self.settings['tau_source'] + self.settings['source_target_delay'] - 1
        startTimeBasedOnConditionalPast = (self.settings['history_conditional'] - 1) * self.settings['tau_conditional'] + self.settings['conditional_target_delay'] - 1
        
        startFirstPoint = max(startTimeBasedOnTargetPast, startTimeBasedOnSourcePast, startTimeBasedOnConditionalPast)

        target_past = self.makeDelayEmbeddingVector(target, 
            self.settings['history_target'], 
            self.settings['tau_target'], 
            startFirstPoint, 
            target.shape[0] - startFirstPoint - 1)
        target_current = self.makeDelayEmbeddingVectorCurrent(target,
            1,
            startFirstPoint + 1,
            target.shape[0] - startFirstPoint - 1)
        source_past = self.makeDelayEmbeddingVector(source,
            self.settings['history_source'],
            self.settings['tau_source'],
            startFirstPoint + 1 - self.settings['source_target_delay'],
            source.shape[0] - startFirstPoint - 1)

        conditional_past = self.makeDelayEmbeddingVector(conditional,
            self.settings['history_conditional'],
            self.settings['tau_conditional'],
            startFirstPoint + 1 - self.settings['conditional_target_delay'],
            conditional.shape[0] - startFirstPoint - 1)
        
        # combine target_current and conditional_past as conditional for CMI
        condCombine = np.hstack([target_past, conditional_past])
        
        self.n_samples = source_past.shape[0]
        self.source_past_dim = source_past.shape[1]
        self.target_current_dim = target_current.shape[1]

        if self.settings['local_values']:
            cte = PythonGaussianCMI.calculateLocalCMI(self, source_past, target_current, condCombine)
            cte = np.hstack([np.zeros(startFirstPoint+1), cte])
            self.actualValue = np.mean(cte)
        else:
            cte = PythonGaussianCMI.calculateAverageCMI(self, source_past, target_current, condCombine)
            self.actualValue = cte
            
        return cte

    def computeSignificance(self):
        C = ChiSquareMeasurementDistribution()
        C.ChiSquareMeasurementDistribution(self.actualValue,
                self.n_samples,
                self.source_past_dim * self.target_current_dim,
                False,
                self.surr_est_type)
        return C

    def get_analytic_distribution(self, source, target, conditional=None):
        """Return a Python AnalyticNullDistribution object.

        Required so that our estimate_surrogates_analytic method can use the
        common_estimate_surrogates_analytic() method, where data is formatted
        as per the estimate method for this estimator.

        Args:
            var1 : numpy array
                realisations of first variable, either a 2D numpy array where
                array dimensions represent [realisations x variable dimension]
                or a 1D array representing [realisations]
            var2 : numpy array
                realisations of the second variable (similar to var1)
            conditional : numpy array [optional]
                realisations of the conditioning variable (similar to var), if
                no conditional is provided, return MI between var1 and var2#

        Returns:
            idtxl calculator that was used here
        """
        if (conditional is None):
            te = PythonGaussianTE(self.settings)
            te.estimate(source, target)
            return te.computeSignificance()
        else:
            self.estimate(source, target, conditional)
            return self.computeSignificance()



###############################
# Discrete estimators
###############################

class PythonDiscrete(PythonEstimator):
    """Abstract class for implementation of Python Gaussian-estimators.

    Abstract class for implementation of Python Gaussian-estimators, child
    classes implement estimators for mutual information (MI), conditional
    mutual information (CMI), active information storage (AIS), transfer
    entropy (TE) using python Gaussian estimator for continuous data. 

    Args:
        settings : dict [optional]
            set estimator parameters:

            - normalise : bool [optional] - z-standardise data (default=False)
            - noise_level : float [optional] - random noise added to the data
              (default=0)
            - local_values : bool [optional] - return local MI/TE instead of
              average MI/TE (default=False)
    """

    def __init__(self, settings):
        settings.setdefault('discretise_method', 'none')
        settings.setdefault('local_values', False)
        super().__init__(settings)

        self.actualValue = None
        self.surr_est_type = "fast"

    def _discretise_vars(self, var1, var2=None, conditional=None):
        # Discretise variables if requested. Otherwise assert data are discrete
        # and provided alphabet sizes are correct.
        if self.settings['discretise_method'] == 'equal':
            var1 = utils.discretise(var1, self.settings['alph1'])
            if var2 is not None:
                var2 = utils.discretise(var2, self.settings['alph2'])
            if conditional is not None:
                conditional = utils.discretise(conditional,
                                               self.settings['alphc'])

        elif self.settings['discretise_method'] == 'max_ent':
            var1 = utils.discretise_max_ent(var1, self.settings['alph1'])
            if var2 is not None:
                var2 = utils.discretise_max_ent(var2, self.settings['alph2'])
            if not (conditional is None):
                conditional = utils.discretise_max_ent(conditional,
                                                       self.settings['alphc'])

        elif self.settings['discretise_method'] == 'none':
            assert issubclass(var1.dtype.type, np.integer), (
                'Var1 is not an integer numpy array. '
                'Discretise data to use this estimator.')
            assert np.min(var1) >= 0, 'Minimum of var1 is smaller than 0.'
            assert np.max(var1) < self.settings['alph1'], (
                        'Maximum of var1 is larger than the alphabet size.')
            if var2 is not None:
                assert issubclass(var2.dtype.type, np.integer), (
                'Var2 is not an integer numpy array. '
                'Discretise data to use this estimator.')
                assert np.min(var2) >= 0, 'Minimum of var2 is smaller than 0.'
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
                assert var1.shape[0] == var2.shape[0] == conditional.shape[0], (
                    'var1, var2 and conditional must have same length.')

        else:
            raise ValueError('Unkown discretisation method.')

        if conditional is not None:
            return var1, var2, conditional
        elif var2 is not None:    
            return var1, var2
        else:
            return var1


    def _encode_multidim_states(self, arr):
        """
        Map each row of an integer-valued array to a single integer state.
        Optimized: uses intp dtype, vectorized stride computation, dot product.
        """
        arr = np.asarray(arr, dtype=np.intp)  # Use platform-native int for indexing
        
        if arr.ndim == 1:
            mn = arr.min()
            codes = arr - mn  # Avoid explicit astype
            n_states = codes.max() + 1
            return codes, n_states

        mn = arr.min(axis=0)
        col = arr - mn
        bases = col.max(axis=0) + 1
        
        # Vectorized stride computation (replaces Python loop)
        strides = np.empty(len(bases), dtype=np.intp)
        strides[-1] = 1
        if len(bases) > 1:
            strides[:-1] = np.cumprod(bases[:0:-1])[::-1]
        
        codes = col @ strides  # Dot product instead of sum
        n_states = codes.max() + 1
        return codes, n_states

    
    def is_analytic_null_estimator(self):
        return True

    def get_analytic_distribution(self, **data):
        """Return a Python AnalyticNullDistribution object.

        Required so that our estimate_surrogates_analytic method can use the
        common_estimate_surrogates_analytic() method, where data is formatted
        as per the estimate method for this estimator.

        Args:
            data : numpy arrays
                realisations of random variables required for the calculation
                (varies between estimators, e.g. 2 variables for MI, 3 for
                CMI). Formatted as per the estimate method for this estimator.

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


class PythonDiscreteMI(PythonDiscrete):
    """Calculate MI with Python discrete-variable implementation.

    Calculate the mutual information (MI) between two variables. 

    Results are returned in bits.

    Args:
        settings : dict [optional]
            sets estimation parameters:
            
            - discretise_method : str [optional] - if and how to discretise
              incoming continuous data, can be 'max_ent' for maximum entropy
              binning 'equal' for equal size bins, and or 'none' if no binning is
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
    
    def calculateAverageMI(self, var1, var2):
        """Calculate average mutual information for discrete data."""

        var1 = np.asarray(var1)
        var2 = np.asarray(var2)

        n = var1.shape[0]
        if n == 0:
            return 0.0

        _, x_labels, x_counts = np.unique(
            var1, axis=0, return_inverse=True, return_counts=True
        )
        _, y_labels, y_counts = np.unique(
            var2, axis=0, return_inverse=True, return_counts=True
        )

        nx = x_counts.size

        # Use int64 unless nx * number_of_y_categories can overflow.
        joint_labels = x_labels.astype(np.int64) * y_counts.size + y_labels

        _, _, joint_counts = np.unique(
            joint_labels,
            return_inverse=True,
            return_counts=True,
        )

        # Recover x/y labels for each unique joint label.
        joint_unique = np.unique(joint_labels)
        joint_x = joint_unique // y_counts.size
        joint_y = joint_unique % y_counts.size

        p_xy = joint_counts.astype(np.float64) / n
        p_x = x_counts[joint_x].astype(np.float64) / n
        p_y = y_counts[joint_y].astype(np.float64) / n

        return np.sum(p_xy * np.log2(p_xy / (p_x * p_y)))

    
    def calculateLocalMI(self, X, Y):
        """Calculate average mutual information for discrete data."""
        X = np.asarray(X)
        Y = np.asarray(Y)
        if X.shape != Y.shape:
            raise ValueError(f"Shape mismatch: X.shape={X.shape}, Y.shape={Y.shape}")

        # Flatten once, keep original shape
        orig_shape = X.shape
        x_flat = X.ravel()
        y_flat = Y.ravel()
        n = x_flat.size

        # Relabel to contiguous integer indices
        x_vals, x_idx = np.unique(x_flat, return_inverse=True)
        y_vals, y_idx = np.unique(y_flat, return_inverse=True)
        nx = x_vals.size
        ny = y_vals.size

        # Joint counts via 1D bincount on combined index
        joint_idx = x_idx * ny + y_idx
        joint_counts = np.bincount(joint_idx, minlength=nx * ny).reshape(nx, ny)

        # Marginals
        px_counts = joint_counts.sum(axis=1)
        py_counts = joint_counts.sum(axis=0)

        # Probabilities
        n_float = float(n)
        px = px_counts / n_float
        py = py_counts / n_float
        pxy = joint_counts / n_float

        # Precompute log terms only where needed
        # pxy > 0 mask
        mask = pxy > 0
        if not np.any(mask):
            # All joint probs zero → MI zero everywhere
            return np.zeros(orig_shape, dtype=float)

        # Compute log(pxy / (px * py)) only on support
        px_grid = px[:, None]
        py_grid = py[None, :]

        # Avoid division by zero; but px, py are > 0 wherever pxy > 0 by construction
        denom = px_grid * py_grid
        i_grid = np.zeros_like(pxy, dtype=float)
        i_grid[mask] = np.log(pxy[mask] / denom[mask]) / np.log(2.0)

        # Map back to samples
        i_flat = i_grid[x_idx, y_idx]
        return i_flat.reshape(orig_shape)

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
                average MI over all samples or local MI for individual
                samples if 'local_values'=True
        """
        # Check the input data
        var1 = self._ensure_two_dim_input(var1)
        var2 = self._ensure_two_dim_input(var2)
        
        assert (
            var1.shape[0] == var2.shape[0]
        ), f"Unequal number of observations (var1: {var1.shape[0]}, var2: {var2.shape[0]})"
        
        # Discretise variables if requested.
        var1, var2 = self._discretise_vars(var1, var2)

        # Then collapse any multivariates into univariate arrays:
        var1 = utils.combine_discrete_dimensions(var1, self.settings['alph1'])
        var2 = utils.combine_discrete_dimensions(var2, self.settings['alph2'])
        
        self.n_samples = var1.shape[0]
        
        # Shift variables to calculate a lagged MI.
        if self.settings['lag_mi'] > 0:
            var1 = var1[:-self.settings['lag_mi']]
            var2 = var2[self.settings['lag_mi']:]

        if self.settings['local_values']:
            var1 = self._ensure_one_dim_input(var1)
            var2 = self._ensure_one_dim_input(var2)
            mi = self.calculateLocalMI(var1, var2)
            self.actualValue = np.mean(mi)
        else:
            var1 = self._ensure_two_dim_input(var1)
            var2 = self._ensure_two_dim_input(var2)
            mi = self.calculateAverageMI(var1, var2)
            self.actualValue = mi

        return mi
    
    def computeSignificance(self):
        C = ChiSquareMeasurementDistribution()
        C.ChiSquareMeasurementDistribution(self.actualValue,
                self.n_samples,
                (self.settings['alph1'] - 1) * (self.settings['alph2'] -1),
                False,
                self.surr_est_type)
        return C
    
    def get_analytic_distribution(self, var1, var2):
        """Return a Python AnalyticNullDistribution object.

        Required so that our estimate_surrogates_analytic method can use the
        common_estimate_surrogates_analytic() method, where data is formatted
        as per the estimate method for this estimator.

        Args:
            var1 : numpy array
                realisations of first variable, either a 2D numpy array where
                array dimensions represent [realisations x variable dimension]
                or a 1D array representing [realisations], array type can be
                float (requires discretisation) or int
            var2 : numpy array
                realisations of the second variable (similar to var1)

        Returns:
            idtxl calculator that was used here
        """
        self.estimate(var1, var2)
        return self.computeSignificance()


class PythonDiscreteCMI(PythonDiscrete):
    """Calculate CMI with Python implementation for discrete variables.

    Calculate the conditional mutual information between two variables given
    the third. 

    Results are returned in bits.

    Args:
        settings : dict [optional]
            sets estimation parameters:

            - discretise_method : str [optional] - if and how to discretise
              incoming continuous data, can be 'max_ent' for maximum entropy
              binning, 'equal' for equal size bins, and or 'none' if no binning is
              required (default='none')
            - n_discrete_bins : int [optional] - number of discrete bins/
              levels or the base of each dimension of the discrete variables
              (default=2). If set, this parameter overwrites/sets alph1, alph2
              and alphc
            - alph1 : int [optional] - number of discrete bins/levels for var1
              (default=2, or the value set for n_discrete_bins)
            - alph2 : int [optional] - number of discrete bins/levels for var2
              (default=2, or the value set for n_discrete_bins)
            - alphc : int [optional] - number of discrete bins/levels for
              conditional (default=2, or the value set for n_discrete_bins)
            - local_values : bool [optional] - return local TE instead of
              average TE (default=False)
    """

    def __init__(self, settings=None):
        settings = self._check_settings(settings)
        # Set default alphabet sizes. Try to overwrite alphabet sizes with
        # number of bins for discretisation if provided, otherwise assume
        # binary variables.
        try:
            n_discrete_bins = int(settings['n_discrete_bins'])
            settings['alph1'] = n_discrete_bins
            settings['alph2'] = n_discrete_bins
            settings['alphc'] = n_discrete_bins
        except KeyError:
            pass  # Do nothing and use the default for alph_* set below
        settings.setdefault('alph1', int(2))
        settings.setdefault('alph2', int(2))
        settings.setdefault('alphc', int(2))
        super().__init__(settings)

    
    def calculateLocalCMI(self, var1, var2, conditional):
        """Local conditional mutual information for discrete data.

        Returns local values in bits.
        Assumes _encode_multidim_states returns:
            codes: integer array of shape (n,)
            nstates: number of encoded states
        """
        x, nx = self._encode_multidim_states(var1)
        y, ny = self._encode_multidim_states(var2)
        z, nz = self._encode_multidim_states(conditional)

        n = x.size

        # Use mixed-radix encoding directly.
        # Cast before multiplication to avoid integer overflow.
        x = np.asarray(x, dtype=np.int64)
        y = np.asarray(y, dtype=np.int64)
        z = np.asarray(z, dtype=np.int64)

        xz = x * nz + z
        yz = y * nz + z
        xyz = (x * ny + y) * nz + z

        # These encodings are dense if x, y, and z are dense.
        c_xyz = np.bincount(xyz)
        c_xz = np.bincount(xz)
        c_yz = np.bincount(yz)
        c_z = np.bincount(z)

        num = c_xyz[xyz] * c_z[z]
        den = c_xz[xz] * c_yz[yz]

        local_cmi = np.zeros(n, dtype=np.float64)
        valid = (num > 0) & (den > 0)

        local_cmi[valid] = np.log2(
            num[valid].astype(np.float64) /
            den[valid].astype(np.float64)
        )

        return local_cmi


    def estimate(self, var1, var2, conditional=None):
        """Estimate conditional mutual information.

        Args:
            var1 : numpy array
                realisations of first variable, either a 2D numpy array where
                array dimensions represent [realisations x variable dimension]
                or a 1D array representing [realisations], array type can be
                float (requires discretisation) or int
            var2 : numpy array
                realisations of the second variable (similar to var1)
            conditional : numpy array [optional]
                realisations of the conditioning variable (similar to var), if
                no conditional is provided, return MI between var1 and var2
            
        Returns:
            float | numpy array
                average CMI over all samples or local CMI for individual
                samples if 'local_values'=True
            
        """
        # Return MI if no conditioning variable was provided.
        if conditional is None:
            #if (self.est_mi is None):
            self.est_mi = PythonDiscreteMI(self.settings)
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

        # Discretise if requested.
        var1, var2, conditional = self._discretise_vars(var1, var2,
                                                        conditional)

        # Then collapse any mulitvariates into univariate arrays:
        var1 = utils.combine_discrete_dimensions(var1, self.settings['alph1'])
        var2 = utils.combine_discrete_dimensions(var2, self.settings['alph2'])
        conditional = utils.combine_discrete_dimensions(conditional,
                                                        self.settings['alphc'])

        var1 = self._ensure_one_dim_input(var1)
        var2 = self._ensure_one_dim_input(var2)
        conditional = self._ensure_one_dim_input(conditional)

        self.n_samples = var1.shape[0]

        cmi = self.calculateLocalCMI(var1, var2, conditional)
        self.actualValue = np.mean(cmi)

        if not self.settings['local_values']:
            cmi = np.mean(cmi)
        
        return cmi

    def computeSignificance(self):
        C = ChiSquareMeasurementDistribution()
        C.ChiSquareMeasurementDistribution(self.actualValue,
                self.n_samples,
                (self.settings['alph1'] - 1) * (self.settings['alph2'] - 1) * (self.settings['alphc']),
                False,
                self.surr_est_type)
        return C

    def get_analytic_distribution(self, var1, var2, conditional=None):
        """Return a Python AnalyticNullDistribution object.

        Required so that our estimate_surrogates_analytic method can use the
        common_estimate_surrogates_analytic() method, where data is formatted
        as per the estimate method for this estimator.

        Args:
            var1 : numpy array
                realisations of first variable, either a 2D numpy array where
                array dimensions represent [realisations x variable dimension]
                or a 1D array representing [realisations], array type can be
                float (requires discretisation) or int
            var2 : numpy array
                realisations of the second variable (similar to var1)
            conditional : numpy array [optional]
                realisations of the conditioning variable (similar to var), if
                no conditional is provided, return MI between var1 and var2

        Returns:
            idtxl calculator that was used here
        """
        if (conditional is None):
            mi = PythonDiscreteMI(self.settings)
            mi.estimate(var1, var2)
            return mi.computeSignificance()
        else:
            self.estimate(var1, var2, conditional)
            return self.computeSignificance()


class PythonDiscreteAIS(PythonDiscrete):
    """Calculate AIS with Python discrete-variable implementation.

    Calculate the active information storage (AIS) for one process. 

    Results are returned in bits.

    Args:
        settings : dict
            set estimator parameters:
            
            - history : int - number of samples in the target's past used as
              embedding (>= 0)
            - tau : int [optional] - the processes' embedding delay (default=1)
            - local_values : bool [optional] - return local TE instead of
              average TE (default=False)
            - discretise_method : str [optional] - if and how to discretise
              incoming continuous data, can be 'max_ent' for maximum entropy
              binning, 'equal' for equal size bins, and 'none' if no binning is
              required (default='none')
            - n_discrete_bins : int [optional] - number of discrete bins/
              levels or the base of each dimension of the discrete variables
              (default=2). If set, this parameter overwrites/sets alph. (>= 2)
            - alph1 : int [optional] - number of discrete bins/levels for var1
              (default=2 , or the value set for n_discrete_bins). (>= 2)
    """

    def __init__(self, settings):
        settings = self._check_settings(settings)
        try:
            settings['history']
        except KeyError:
            raise RuntimeError('No history was provided for AIS estimation.')
        assert type(settings['history']) is int, (
                                            'History has to be an integer.')
        assert settings['history'] >= 0, 'History must be >= 0'

        settings.setdefault('tau', 1)
        assert type(settings['tau']) is int, (
                                            'tau has to be an integer.')
        assert settings['tau'] >= 0, 'tau must be >= 0'

        # Get alphabet sizes and check if discretisation is requested
        try:
            n_discrete_bins = int(settings['n_discrete_bins'])
            settings['alph1'] = n_discrete_bins
        except KeyError:
            pass  # Do nothing and use the default for alph set below
        settings.setdefault('alph1', int(2))
        assert settings['alph1'] >= 2, 'Number of bins must be >= 2'
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
                average AIS over all samples or local MI for individual
                samples if 'local_values'=True

        """
        # Check the input data
        process = self._ensure_one_dim_input(process)
        
        # Discretise variables if requested.
        process = self._discretise_vars(process)

        # delay embedding
        startFirstPoint = (self.settings['history'] - 1)

        process_current = self.makeDelayEmbeddingVectorCurrent(process,
            1,
            startFirstPoint + 1,
            process.shape[0] - startFirstPoint - 1)

        process_past = self.makeDelayEmbeddingVector(process, 
            self.settings['history'], 
            1, 
            startFirstPoint, 
            process.shape[0] - startFirstPoint - 1)
        process_past = utils.combine_discrete_dimensions(process_past, self.settings['alph1'])
        process_past = self._ensure_two_dim_input(process_past)
        process_past = process_past.astype(int)
        
        self.n_samples = process.shape[0]

        if self.settings['local_values']:
            ais = PythonDiscreteMI.calculateLocalMI(self, process_past, process_current)
            ais = np.hstack([np.zeros(self.settings['history']), ais[:,0]])
            self.actualValue = np.mean(ais)
        else:
            ais = PythonDiscreteMI.calculateAverageMI(self, process_past, process_current)
            self.actualValue = ais
        
        return ais

    def computeSignificance(self):
        C = ChiSquareMeasurementDistribution()
        C.ChiSquareMeasurementDistribution(self.actualValue,
                self.n_samples,
                (self.settings['alph1'] - 1) * (np.power(self.settings['alph1'], self.settings['history']) - 1),
                False,
                self.surr_est_type)
        return C

    def get_analytic_distribution(self, process):
        """Return a Python AnalyticNullDistribution object.

        Required so that our estimate_surrogates_analytic method can use the
        common_estimate_surrogates_analytic() method, where data is formatted
        as per the estimate method for this estimator.

        Args:
            process : numpy array
                realisations as either a 2D numpy array where array dimensions
                represent [realisations x variable dimension] or a 1D array
                representing [realisations], array type can be float (requires
                discretisation) or int

        Returns:
            idtxl calculator that was used here
        """
        self.estimate(process)
        return self.computeSignificance()


class PythonDiscreteTE(PythonDiscrete):
    """Calculate TE with Python implementation for discrete variables.

    Calculate the transfer entropy between two time series processes.
    
    Results are returned in bits.

    Args:
        settings : dict
            sets estimation parameters:

            - history_target : int - number of samples in the target's past
              used as embedding. (>= 0)
            - history_source  : int [optional] - number of samples in the
              source's past used as embedding (default=same as the target
              history). (>= 1)
            - tau_source : int [optional] - source's embedding delay
              (default=1). (>= 1)
            - tau_target : int [optional] - target's embedding delay
              (default=1). (>= 1)
            - source_target_delay : int [optional] - information transfer delay
              between source and target (default=1) (>= 0)
            - discretise_method : str [optional] - if and how to discretise
              incoming continuous data, can be 'max_ent' for maximum entropy
              binning, 'equal' for equal size bins, and 'none' if no binning is
              required (default='none')
            - n_discrete_bins : int [optional] - number of discrete bins/
              levels or the base of each dimension of the discrete variables
              (default=2). If set, this parameter overwrites/sets alph1 and
              alph2. (>= 2)
            - alph1 : int [optional] - number of discrete bins/levels for
              source (default=2, or the value set for n_discrete_bins). (>= 2)
            - alph2 : int [optional] - number of discrete bins/levels for
              target (default=2, or the value set for n_discrete_bins). (>= 2)
            - local_values : bool [optional] - return local TE instead of
              average TE (default=False)
    """

    def __init__(self, settings):
        settings = self._check_settings(settings)
        # Get embedding and delay parameters.
        settings = self._set_te_defaults(settings)

        # Get alphabet sizes and check if discretisation is requested. Try to
        # overwrite alphabet sizes with number of bins.
        try:
            n_discrete_bins = int(settings['n_discrete_bins'])
            settings['alph1'] = n_discrete_bins
            settings['alph2'] = n_discrete_bins
        except KeyError:
            # do nothing and set alphabet sizes to default below
            pass
        settings.setdefault('alph1', int(2))
        settings.setdefault('alph2', int(2))
        assert type(settings['alph1']) is int, (
            'Num discrete levels for source has to be an integer.')
        assert type(settings['alph2']) is int, (
            'Num discrete levels for target has to be an integer.')
        assert settings['alph1'] >= 2, (
            'Num discrete levels for source must be >= 2')
        assert settings['alph2'] >= 2, (
            'Num discrete levels for target must be >= 2')
        super().__init__(settings)

    def combine_embedding_dimensions(self, var, alph):
        var = utils.combine_discrete_dimensions(var, alph)
        var = self._ensure_one_dim_input(var)
        var = var.astype(int)
        return var

    def estimate(self, source, target):
        """Estimate transfer entropy from a source to a target variable.

        Args:
            source : numpy array
                realisations of source variable, either a 2D numpy array where
                array dimensions represent [realisations x variable dimension]
                or a 1D array representing [realisations], array type can be
                float (requires discretisation) or int
            target : numpy array
                realisations of target variable (similar to var1)
            return_calc : boolean
                return the calculator used here as well as the numeric
                calculated value(s)

        Returns:
            float | numpy array
                average TE over all samples or local TE for individual
                samples if 'local_values'=True
            
        """
        # Check the input data
        source = self._ensure_one_dim_input(source)
        target = self._ensure_one_dim_input(target)

        # Discretise variables if requested.
        source, target = self._discretise_vars(source, target)

        assert (
            source.shape[0] == target.shape[0]
        ), f"Unequal number of observations (source: {source.shape[0]}, target: {target.shape[0]})"
                
        # delay embedding
        startFirstPoint = self.computeStartTimeForFirstDestEmbedding(
            self.settings['history_target'],
            self.settings['tau_target'],
            self.settings['history_source'],
            self.settings['tau_source'],
            self.settings['source_target_delay'],
            )

        target_past = self.makeDelayEmbeddingVector(target, 
            self.settings['history_target'], 
            self.settings['tau_target'], 
            startFirstPoint, 
            target.shape[0] - startFirstPoint - 1)
        target_past = self.combine_embedding_dimensions(target_past, self.settings['alph2'])
        
        target_current = self.makeDelayEmbeddingVectorCurrent(target, 
            1,
            startFirstPoint + 1,
            target.shape[0] - startFirstPoint - 1)
        target_current = self._ensure_one_dim_input(target_current)
        target_current = target_current.astype(int)

        source_past = self.makeDelayEmbeddingVector(source,
            self.settings['history_source'],
            self.settings['tau_source'],
            startFirstPoint + 1 - self.settings['source_target_delay'],
            source.shape[0] - startFirstPoint - 1)
                
        source_past = self.combine_embedding_dimensions(source_past, self.settings['alph1'])
        
        self.n_samples = source_past.shape[0]

        te = PythonDiscreteCMI.calculateLocalCMI(self, source_past, target_current, target_past)
        self.actualValue = np.mean(te)

        if self.settings['local_values']:
            # correction to compare with JidtGaussianTE results
            te = np.hstack([np.zeros(startFirstPoint+1), te])
        else:
            te = np.mean(te)
        
        return te

    def computeSignificance(self):
        C = ChiSquareMeasurementDistribution()
        C.ChiSquareMeasurementDistribution(self.actualValue,
                self.n_samples,
                (np.power(self.settings['alph1'], self.settings['history_source']) - 1) * (self.settings['alph1'] - 1) * np.power(self.settings['alph2'], self.settings['history_target']),
                False,
                self.surr_est_type)
        return C

    def get_analytic_distribution(self, source, target):
        """Return a Python AnalyticNullDistribution object.

        Required so that our estimate_surrogates_analytic method can use the
        common_estimate_surrogates_analytic() method, where data is formatted
        as per the estimate method for this estimator.

        Args:
            source : numpy array
                realisations of source variable, either a 2D numpy array where
                array dimensions represent [realisations x variable dimension]
                or a 1D array representing [realisations], array type can be
                float (requires discretisation) or int
            target : numpy array
                realisations of target variable (similar to var1)

        Returns:
            idtxl calculator that was used here
        """
        # Make one estimate to prepare the calculator:
        self.estimate(source, target)
        return self.computeSignificance()


def common_estimate_surrogates_analytic(estimator, n_perm=200, **data):
    """Estimate the surrogate distribution analytically for PythonEstimator.

    Estimate the surrogate distribution analytically for a PythonEstimator
    which is_analytic_null_estimator(), by sampling estimates at random
    p-values in the analytic distribution.

    Args:
        estimator : a PythonEstimator, which returns True to a call to
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
    # AnalyticMeasurementDistribution object:
    analytic_distribution = estimator.get_analytic_distribution(**data)
    
    # Then compute surrogates at n_perm random p-values
    surrogate_estimates = np.empty(n_perm)
    for perm in range(n_perm):
        surrogate_estimates[perm] = \
            analytic_distribution.computeEstimateForGivenPValue(
                np.random.random())

    return surrogate_estimates
