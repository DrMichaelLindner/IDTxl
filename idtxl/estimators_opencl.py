"""Provide OpenCL estimators."""
import sys
import copy
import logging
from pkg_resources import resource_filename
from scipy.special import digamma
from scipy.linalg import cholesky
import numpy as np
from idtxl.estimator import Estimator
import idtxl.idtxl_utils as utils
from . import idtxl_exceptions as ex
from idtxl.measurement_distributions_python import EmpiricalMeasurementDistribution, AnalyticalMeasurementDistribution, ChiSquareMeasurementDistribution

try:
    import pyopencl as cl
    import pyopencl.array as cla
    from pyopencl.reduction import ReductionKernel
except ImportError as err:
    ex.package_missing(err, 'PyOpenCl is not available on this system.\n'
                            'If you want to use the idtxl OpenCL CMI estimators:\n'
                            '1. install your vendors OpenCl drivers (e.g. for Intel, NVidia)\n'
                            '2. pip install pyopencl\n'
                            'for more information: https://documen.tician.de/pyopencl/misc.html#installing-from-conda-forge')
    sys.exit()

logger = logging.getLogger(__name__)
C = 1024**2


class OpenCLEstimator(Estimator):
    """Abstract class for implementation of OpenCL estimators.

    implemented in idtxl by Michael Lindner, 2026
    
    """
    def __init__(self, settings=None):
        settings = self._check_settings(settings)
        
        self.settings = settings.copy()

    def _get_device(self, gpuid):
        """Return OpenCL devices, context, and queue.
        
        Prefers GPU; falls back to CPU if no GPU is available.
        """
        all_platforms = cl.get_platforms()

        # Try GPU first
        platform = next((p for p in all_platforms if
                         p.get_devices(device_type=cl.device_type.GPU) != []),
                        None)
        if platform is not None:
            my_devices = platform.get_devices(device_type=cl.device_type.GPU)
            if gpuid > len(my_devices) - 1:
                raise RuntimeError(
                    'No device with gpuid {0} (available device IDs: {1}).'.format(
                        gpuid, np.arange(len(my_devices))))

            device_type_str = "GPU"
        else:
            # Fallback to CPU
            platform = next((p for p in all_platforms if
                             p.get_devices(device_type=cl.device_type.CPU) != []),
                            None)
            if platform is not None:
                # get device and type
                my_devices = platform.get_devices(device_type=cl.device_type.CPU)
                device_type_str = "CPU"
            else:
                # if no GPU or CPU available
                raise RuntimeError('No OpenCL GPU or CPU device found.')

        # get context and queue
        context = cl.Context(devices=my_devices)
        queue = cl.CommandQueue(context, my_devices[gpuid])

        logger.debug(
            "Selected %s Device: %s (platform: %s)",
            device_type_str,
            my_devices[gpuid].name,
            my_devices[gpuid].platform.name
        )

        return my_devices, context, queue, device_type_str

    def _get_max_mem(self):
        """Return max. GPU main memory available for computation."""
        if 'max_mem' in self.settings:
            return self.settings['max_mem']
        elif 'max_mem_frac' in self.settings:
            return self.settings['max_mem_frac'] * self.devices[
                                    self.settings['gpuid']].global_mem_size
        else:
            return 0.9 * self.devices[self.settings['gpuid']].global_mem_size

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

    def computeStartTimeForFirstDestEmbedding(self, history_target, tau_target, history_source, tau_source, delay):
        """get first time point for embedding"""
        startTimeBasedOnTargetPast = (history_target - 1) * tau_target
        startTimeBasedOnSourcePast = (history_source - 1) * tau_source + delay - 1
        return max(startTimeBasedOnTargetPast, startTimeBasedOnSourcePast);

    def makeDelayEmbeddingVector(self, ts, history, tau, startFirstPoint, numEmbeddingVectors):
        """create past delay embedding vector of given data and settings """
        embedded_vector = np.zeros((numEmbeddingVectors, history))

        for t in range(startFirstPoint, numEmbeddingVectors + startFirstPoint):
            for i in range(history):
                embedded_vector[t - startFirstPoint, i] = ts[t - i * tau]

        return embedded_vector

    def makeDelayEmbeddingVectorCurrent(self, ts, history, startFirstPoint, numEmbeddingVectors):
        """create current delay embedding vector of given data and settings """
        embedded_vector = np.zeros((numEmbeddingVectors, history))

        for t in range(startFirstPoint, numEmbeddingVectors + startFirstPoint):
            for i in range(history):
                embedded_vector[t - startFirstPoint, i] = ts[t - i]

        return embedded_vector

    def set_data(self, flag, X, Y=None, Z=None):
        """Set data to self for calculation of Local or Average XXX"""

        if flag == "AIS":
            self.flag = "AIS"
            self.var1 = X
            self.var2 = Y
        elif flag == "MI":
            self.flag = "MI"
            self.var1 = X
            self.var2 = Y
        elif flag == "CMI":
            self.flag = "CMI"
            self.var1 = X
            self.var2 = Y
            self.conditional = Z
        elif flag == "TE":
            self.flag = "TE"
            self.var1 = X
            self.var2 = Y
            self.conditional = Z
        elif flag == "CTE":
            self.flag = "CTE"
            self.var1 = X
            self.var2 = Y
            self.conditional = Z

    def get_data(self):
        """get data in calculate functions"""
        if self.flag == "MI":
            return self.var1, self.var2
        if self.flag == "CMI":
            return self.var1, self.var2, self.conditional
        if self.flag == "AIS":
            return self.var1, self.var2
        if self.flag == "TE":
            return self.var1, self.var2, self.conditional
        if self.flag == "CTE":
            return self.var1, self.var2, self.conditional

    def remove_data(self):
        """Remove data from self after calculation"""
        if self.flag == "MI":
            del self.var1
            del self.var2
        if self.flag == "CMI":
            del self.var1
            del self.var2
            del self.conditional
        if self.flag == "AIS":
            del self.var1
            del self.var2
        if self.flag == "TE":
            del self.var1
            del self.var2
            del self.conditional
        if self.flag == "CTE":
            del self.var1
            del self.var2
            del self.conditional

        del self.flag


###############################
# Kraskov estimators
###############################
class OpenCLKraskov(OpenCLEstimator):
    """Abstract class for implementation of OpenCLKraskov estimators.

    Abstract class for implementation of OpenCL estimators, child classes
    implement estimators for mutual information (MI) and conditional mutual
    information (CMI) using the Kraskov-Grassberger-Stoegbauer estimator for
    continuous data.

    References:

    - Kraskov, A., Stoegbauer, H., & Grassberger, P. (2004). Estimating mutual
      information. Phys Rev E, 69(6), 066138.
    - Lizier, Joseph T., Mikhail Prokopenko, and Albert Y. Zomaya. (2012).
      Local measures of information storage in complex distributed computation.
      Inform Sci, 208, 39-54.
    - Schreiber, T. (2000). Measuring information transfer. Phys Rev Lett,
      85(2), 461.

    Estimators can be used to perform multiple, independent searches in
    parallel. Each of these parallel searches is called a 'chunk'. To search
    multiple chunks, provide point sets as 2D arrays, where the first
    dimension represents samples or points, and the second dimension
    represents the points' dimensions. Concatenate chunk data in the first
    dimension and pass the number of chunks to the estimators. Chunks must be
    of equal size.

    Set common estimation parameters for OpenCL estimators. For usage of these
    estimators see documentation for the child classes.

    modified by Michael Lindner, 2026
    
    Args:
        settings : dict [optional]
            set estimator parameters:

            - gpuid : int [optional] - device ID used for estimation (if more
              than one device is available on the current platform) (default=0)
            - kraskov_k : int [optional] - no. nearest neighbours for KNN
              search (default=4)
            - normalise : bool [optional] - z-standardise data (default=False)
            - theiler_t : int [optional] - no. next temporal neighbours ignored
              in KNN and range searches (default=0)
            - noise_level : float [optional] - random noise added to the data
              (default=1e-8)
            - padding : bool [optional] - pad data to a length that is a
              multiple of 1024, workaround for a
            - debug : bool [optional] - calculate intermediate results, i.e.
              neighbour counts from range searches and KNN distances, print
              debug output to console (default=False)
            - return_counts : bool [optional] - return intermediate results,
              i.e. neighbour counts from range searches and KNN distances
              (default=False)
    """

    def __init__(self, settings=None):
        # Get defaults for estimator settings
        super().__init__(settings)
        self.settings.setdefault('gpuid', int(0))
        self.settings.setdefault('kraskov_k', int(4))
        self.settings.setdefault('theiler_t', int(0))
        self.settings.setdefault('noise_level', np.float32(1e-8))
        self.settings.setdefault('local_values', False)
        self.settings.setdefault('padding', True)
        self.settings.setdefault('debug', False)
        self.settings.setdefault('return_counts', False)
        self.settings.setdefault('verbose', True)
        self.sizeof_float = int(np.dtype(np.float32).itemsize)
        self.sizeof_int = int(np.dtype(np.int32).itemsize)

        if self.settings['return_counts'] and not self.settings['debug']:
            raise RuntimeError(
                'Set debug option to True to return neighbor counts.')

        # Get kernel and devices.
        self.devices, self.context, self.queue, self.device_type_str = self._get_device(
                                                        self.settings['gpuid'])
        self.kernel_location = resource_filename(__name__,
                                                 'gpuKnnKernelNoIdx.cl')
        self.kNN_kernel, self.RS_kernel = self._get_kernels()

    def is_parallel(self):
        return True

    def is_analytic_null_estimator(self):
        return False

    def _get_kernels(self):
        """Return KNN and range search OpenCL kernels."""
        kernel_source = open(self.kernel_location).read()
        program = cl.Program(self.context, kernel_source).build()
        kNN_kernel = program.kernelKNNshared
        kNN_kernel.set_scalar_arg_dtypes([None, None, None, np.int32,
                                          np.int32, np.int32, np.int32,
                                          np.int32, np.int32, None])  # MW: added one int32 argument

        RS_kernel = program.kernelBFRSAllshared
        RS_kernel.set_scalar_arg_dtypes([None, None, None, None,
                                         np.int32, np.int32, np.int32,
                                         np.int32, np.int32, None])  # MW: added one int32 argument
        return (kNN_kernel, RS_kernel)


class OpenCLKraskovMI(OpenCLKraskov):
    """Calculate mutual information with OpenCL Kraskov implementation.

    Calculate the mutual information (MI) between two variables using OpenCL
    GPU-code. See parent class for references.

    Results are returned in nats.

    Args:
        settings : dict [optional]
            set estimator parameters:

            - gpuid : int [optional] - device ID used for estimation (if more
              than one device is available on the current platform) (default=0)
            - kraskov_k : int [optional] - no. nearest neighbours for KNN
              search (default=4)
            - normalise : bool [optional] - z-standardise data (default=False)
            - theiler_t : int [optional] - no. next temporal neighbours ignored
              in KNN and range searches (default=0)
            - noise_level : float [optional] - random noise added to the data
              (default=1e-8)
            - debug : bool [optional] - return intermediate results, i.e.
              neighbour counts from range searches and KNN distances
              (default=False)
            - return_counts : bool [optional] - return intermediate results,
              i.e. neighbour counts from range searches and KNN distances
              (default=False)
            - lag_mi : int [optional] - time difference in samples to calculate
              the lagged MI between processes (default=0)
    """

    def __init__(self, settings=None):
        # Set default estimator settings.
        super().__init__(settings)
        self.settings.setdefault('lag_mi', 0)


    def estimate(self, var1, var2, n_chunks=1):
        """Estimate mutual information.

        Args:
            var1 : numpy array
                realisations of first variable, either a 2D numpy array where
                array dimensions represent [(realisations * n_chunks) x
                variable dimension] or a 1D array representing [realisations],
                array type should be int32
            var2 : numpy array
                realisations of the second variable (similar to var1)
            n_chunks : int
                number of data chunks, no. data points has to be the same for
                each chunk

        Returns:
            float | numpy array
                average MI over all samples or local MI for individual
                samples if 'local_values'=True
            numpy arrays
                distances and neighborhood counts for var1 and var2 if
                debug=True and return_counts=True
        """
        # Prepare data: check if variable realisations are passed as 1D or 2D
        # arrays and have equal no. observations.
        var1 = self._ensure_two_dim_input(var1)
        var2 = self._ensure_two_dim_input(var2)
        assert var1.shape[0] == var2.shape[0]
        assert var1.shape[0] % n_chunks == 0
        # Shift variables to calculate a lagged MI.
        if self.settings['lag_mi'] > 0:
            var1 = var1[:-self.settings['lag_mi'], :]
            var2 = var2[self.settings['lag_mi']:, :]
        self._check_number_of_points(var1.shape[0])
        signallength = var1.shape[0]
        chunklength = signallength // n_chunks
        var1dim = var1.shape[1]
        var2dim = var2.shape[1]
        pointdim = var1dim + var2dim
        kraskov_k = self.settings['kraskov_k']

        mem_data = self.sizeof_float * chunklength * pointdim
        mem_dist = self.sizeof_float * chunklength * kraskov_k
        mem_ncnt = 2 * self.sizeof_int * chunklength
        mem_chunk = mem_data + mem_dist + mem_ncnt
        max_mem = self._get_max_mem()

        max_chunks_per_run = np.floor(max_mem/mem_chunk).astype(int)
        chunks_per_run = min(max_chunks_per_run, n_chunks)

        logger.debug(
            'Memory per chunk: {0:.5f} MB, GPU global memory: {1} MB, chunks '
            'per run: {2}.'.format(
                mem_chunk / C, max_mem / C, chunks_per_run))
        if mem_chunk > max_mem:
            raise RuntimeError('Size of single chunk exceeds GPU global '
                               'memory.')

        mi_array = np.array([])
        if self.settings['debug']:
            distances = np.array([])
            count_var1 = np.array([])
            count_var2 = np.array([])

        for r in range(0, n_chunks, chunks_per_run):
            startidx = r*chunklength
            stopidx = min(r+chunks_per_run, n_chunks)*chunklength
            subset1 = var1[startidx:stopidx, :]
            subset2 = var2[startidx:stopidx, :]
            n_chunks_current_run = subset1.shape[0] // chunklength
            results = self._estimate_single_run(subset1, subset2,
                                                n_chunks_current_run)
            if self.settings['debug']:
                mi_array = np.concatenate((mi_array,   results[0]))
                distances = np.concatenate((distances,  results[1]))
                count_var1 = np.concatenate((count_var1, results[2]))
                count_var2 = np.concatenate((count_var2, results[3]))
            else:
                mi_array = np.concatenate((mi_array, results))

        if self.settings['return_counts']:
            return mi_array, distances, count_var1, count_var2
        else:
            return mi_array


    def _estimate_single_run(self, var1, var2, n_chunks=1):
        """Estimate mutual information in a single GPU run.

        This method should not be called directly, only inside estimate()
        after memory bounds have been checked.

        Args:
            var1 : numpy array
                realisations of first variable, either a 2D numpy array where
                array dimensions represent [(realisations * n_chunks) x
                variable dimension] or a 1D array representing [realisations],
                array type should be int32
            var2 : numpy array
                realisations of the second variable (similar to var1)
            n_chunks : int
                number of data chunks, no. data points has to be the same for
                each chunk

        Returns:
            float | numpy array
                average MI over all samples or local MI for individual
                samples if 'local_values'=True
        """
        # Prepare data and add noise: check if variable realisations are passed
        # as 1D or 2D arrays and have equal no. observations.
        var1 = self._ensure_two_dim_input(var1)
        var2 = self._ensure_two_dim_input(var2)
        assert var1.shape[0] == var2.shape[0]
        assert var1.shape[0] % n_chunks == 0
        self._check_number_of_points(var1.shape[0])
        signallength = var1.shape[0]
        chunklength = signallength // n_chunks
        assert signallength % n_chunks == 0
        var1dim = var1.shape[1]
        var2dim = var2.shape[1]
        pointdim = var1dim + var2dim

        # prepare for the padding
        signallength_orig = signallength  # used for clarity at present

        if self.settings['padding']:
            # Pad time series to make GPU memory regions a multiple of 1024
            # This value of 1024 should be replaced by
            #  self.devices[self.settings['gpuid']].CL_DEVICE_MEM_BASE_ADDR_ALIGN
            # or something similar, as professional cards are known to have
            # base adress alignment of 4096 sometimes
            pad_target = 4096
            pad_size = (int(np.ceil(signallength/pad_target)) * pad_target -
                        signallength)
            pad_var1 = np.vstack(
                [var1, 999999 + 0.1 * np.random.rand(pad_size, var1dim)])
            pad_var2 = np.vstack(
                [var2, 999999 + 0.1 * np.random.rand(pad_size, var2dim)])
            pointset = np.hstack((pad_var1, pad_var2)).T.copy()
            signallength_padded = signallength + pad_size
        else:
            pad_size = 0
            pointset = np.hstack((var1, var2)).T.copy()
            signallength_padded = signallength

        if not pointset.dtype == np.float32:
            pointset = pointset.astype(np.float32)
        if self.settings['noise_level'] > 0:
            pointset += np.random.normal(
                scale=self.settings['noise_level'],
                size=pointset.shape).astype(np.float32)

        if self.settings['debug']:
            # Print memory requirements after padding
            mem_data_pad = (self.sizeof_float *
                            pointset.shape[0] * pointset.shape[1])
            mem_dist = (self.sizeof_float * signallength_padded *
                        self.settings['kraskov_k'])
            mem_ncnt = 2 * self.sizeof_int * signallength_padded
            mem_total = mem_data_pad + mem_dist + mem_ncnt
            logger.debug(
                'Memory req. after padding: {0:.2f} MB ({1} elements, shape: '
                '{2}, {3} chunks, chunksize: {4}) -- Padding: {5}'.format(
                    mem_total / C, pointset.size, pointset.shape,
                    n_chunks, chunklength, pad_size))
            assert (pointset.shape[1] - pad_size) % n_chunks == 0

        # Set OpenCL kernel launch parameters
        if chunklength < self.devices[
                                self.settings['gpuid']].max_work_group_size:
            workitems_x = 8
        elif self.devices[self.settings['gpuid']].max_work_group_size < 256:
            workitems_x = self.devices[
                                self.settings['gpuid']].max_work_group_size
        else:
            workitems_x = 256
        NDRange_x = (workitems_x *
                     (int((signallength_padded - 1)/workitems_x) + 1))
        logger.debug('NDRange_x: {}, workitems_x: {}'.format(
            NDRange_x, workitems_x))

        # Allocate and copy memory to device
        kraskov_k = self.settings['kraskov_k']
        d_pointset = cl.Buffer(
                        self.context,
                        cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                        hostbuf=pointset)
        d_var1 = d_pointset.get_sub_region(
                        0,
                        self.sizeof_float * signallength_padded * var1dim,
                        cl.mem_flags.READ_ONLY)
        d_var2 = d_pointset.get_sub_region(
                        self.sizeof_float * signallength_padded * var1dim,
                        self.sizeof_float * signallength_padded * var2dim,
                        cl.mem_flags.READ_ONLY)
        d_distances = cl.Buffer(
                        self.context, cl.mem_flags.READ_WRITE,
                        self.sizeof_float * kraskov_k * signallength_padded)
        d_vecradius = d_distances.get_sub_region(
                    signallength_padded * (kraskov_k - 1) * self.sizeof_float,
                    signallength_padded * self.sizeof_float)
        d_npointsrange_x = cl.Buffer(self.context, cl.mem_flags.READ_WRITE,
                                     self.sizeof_int * signallength_padded)
        d_npointsrange_y = cl.Buffer(self.context, cl.mem_flags.READ_WRITE,
                                     self.sizeof_int * signallength_padded)

        # Neighbour search
        theiler_t = np.int32(self.settings['theiler_t'])
        localmem = cl.LocalMemory(self.sizeof_float * kraskov_k * workitems_x)
        self.kNN_kernel(self.queue, (NDRange_x,), (workitems_x,), d_pointset,
                        d_pointset, d_distances, np.int32(pointdim),
                        np.int32(chunklength), np.int32(signallength_padded),
                        np.int32(signallength_orig),
                        np.int32(kraskov_k), theiler_t, localmem)
        distances = np.zeros(signallength_padded * kraskov_k, dtype=np.float32)
        try:
            cl.enqueue_copy(self.queue, distances, d_distances)
        except cl._cl.RuntimeError as e:
            print(e)
            # Print memory requirements after padding
            mem_data_pad = (self.sizeof_float *
                            pointset.shape[0] * pointset.shape[1])
            mem_dist = (self.sizeof_float * signallength_padded *
                        self.settings['kraskov_k'])
            mem_ncnt = 2 * self.sizeof_int * signallength_padded
            mem_total = mem_data_pad + mem_dist + mem_ncnt
            print(
                'Memory req. after padding: {0:.2f} MB ({1} elements, shape: '
                '{2}, {3} chunks, chunksize: {4}) -- Padding: {5}'.format(
                    mem_total / C, pointset.size, pointset.shape,
                    n_chunks, chunklength, pad_size))
            assert (pointset.shape[1] - pad_size) % n_chunks == 0
            sys.exit(1)
        self.queue.finish()

        # Range search in var1
        localmem = cl.LocalMemory(self.sizeof_int * workitems_x)
        self.RS_kernel(
            self.queue, (NDRange_x,), (workitems_x,), d_var1,
            d_var1, d_vecradius, d_npointsrange_x,
            var1dim, chunklength, signallength_padded, signallength_orig,
            theiler_t, localmem)  # MW: added signallength_orig
        count_var1 = np.zeros(signallength_padded, dtype=np.int32)
        cl.enqueue_copy(self.queue, count_var1, d_npointsrange_x)

        # Range search in var2
        self.RS_kernel(
            self.queue, (NDRange_x,), (workitems_x,), d_var2,
            d_var2, d_vecradius, d_npointsrange_y,
            var2dim, chunklength, signallength_padded, signallength_orig,
            theiler_t, localmem)  # MW: added signallength_orig
        count_var2 = np.zeros(signallength_padded, dtype=np.int32)
        cl.enqueue_copy(self.queue, count_var2, d_npointsrange_y)

        d_pointset.release()
        d_distances.release()
        d_npointsrange_x.release()
        d_npointsrange_y.release()
        d_var1.release()
        d_var2.release()
        d_vecradius.release()

        # Calculate and sum digammas
        if self.settings['local_values']:
            mi_array = -np.inf * np.ones(chunklength * n_chunks,
                                         dtype=np.float64)
            idx = 0
            for c in range(n_chunks):
                mi = (digamma(kraskov_k) + digamma(chunklength) -
                      digamma(count_var1[c*chunklength:(c+1)*chunklength]+1) -
                      digamma(count_var2[c*chunklength:(c+1)*chunklength]+1))
                mi_array[idx:idx+chunklength] = mi
                idx += chunklength

        else:
            mi_array = -np.inf * np.ones(n_chunks, dtype=np.float64)
            for c in range(n_chunks):
                mi = (digamma(kraskov_k) + digamma(chunklength) - np.mean(
                      digamma(count_var1[c*chunklength:(c+1)*chunklength]+1) +
                      digamma(count_var2[c*chunklength:(c+1)*chunklength]+1)))
                mi_array[c] = mi
        assert signallength_orig == (c+1)*chunklength, 'Original signal length does not match no. processed points.'

        if self.settings['debug']:
            return (mi_array,
                    distances[:signallength_orig],
                    count_var1[:signallength_orig],
                    count_var2[:signallength_orig])
        else:
            return mi_array


class OpenCLKraskovCMI(OpenCLKraskov):
    """Calculate conditional mutual inform with OpenCL Kraskov implementation.

    Calculate the conditional mutual information (CMI) between three variables
    using OpenCL GPU-code. If no conditional is given (is None), the function
    returns the mutual information between var1 and var2. See parent class for
    references.

    Results are returned in nats.

    Args:
        settings : dict [optional]
            set estimator parameters:

            - gpuid : int [optional] - device ID used for estimation (if more
              than one device is available on the current platform) (default=0)
            - kraskov_k : int [optional] - no. nearest neighbours for KNN
              search (default=4)
            - normalise : bool [optional] - z-standardise data (default=False)
            - theiler_t : int [optional] - no. next temporal neighbours ignored
              in KNN and range searches (default=0)
            - noise_level : float [optional] - random noise added to the data
              (default=1e-8)
            - debug : bool [optional] - return intermediate results, i.e.
              neighbour counts from range searches and KNN distances
              (default=False)
            - return_counts : bool [optional] - return intermediate results,
              i.e. neighbour counts from range searches and KNN distances
              (default=False)
    """

    def __init__(self, settings=None):
        super().__init__(settings)

    def estimate(self, var1, var2, conditional=None, n_chunks=1):
        """Estimate conditional mutual information.

        If conditional is None, the mutual information between var1 and var2 is
        calculated.

        Args:
            var1 : numpy array
                realisations of first variable, either a 2D numpy array where
                array dimensions represent [(realisations * n_chunks) x
                variable dimension] or a 1D array representing [realisations],
                array type should be int32
            var2 : numpy array
                realisations of the second variable (similar to var1)
            conditional : numpy array
                realisations of conditioning variable (similar to var1)
            n_chunks : int
                number of data chunks, no. data points has to be the same for
                each chunk

        Returns:
            float | numpy array
                average CMI over all samples or local CMI for individual
                samples if 'local_values'=True
            numpy arrays
                distances and neighborhood counts for var1 and var2 if
                debug=True and return_counts=True
        """
        # Return MI if no conditional is provided
        if conditional is None:
            est_mi = OpenCLKraskovMI(self.settings)
            return est_mi.estimate(var1, var2, n_chunks)

        # Prepare data: check if variable realisations are passed as 1D or 2D
        # arrays and have equal no. observations.
        var1 = self._ensure_two_dim_input(var1)
        var2 = self._ensure_two_dim_input(var2)
        conditional = self._ensure_two_dim_input(conditional)
        assert var1.shape[0] == var2.shape[0]
        assert var1.shape[0] == conditional.shape[0]
        assert var1.shape[0] % n_chunks == 0
        self._check_number_of_points(var1.shape[0])
        signallength = var1.shape[0]
        chunklength = signallength // n_chunks
        var1dim = var1.shape[1]
        var2dim = var2.shape[1]
        conddim = conditional.shape[1]
        pointdim = var1dim + var2dim + conddim
        kraskov_k = self.settings['kraskov_k']

        mem_data = self.sizeof_float * chunklength * pointdim
        mem_dist = self.sizeof_float * chunklength * kraskov_k
        mem_ncnt = 2 * self.sizeof_int * chunklength
        mem_chunk = mem_data + mem_dist + mem_ncnt
        max_mem = self._get_max_mem()

        max_chunks_per_run = np.floor(max_mem/mem_chunk).astype(int)
        chunks_per_run = min(max_chunks_per_run, n_chunks)

        logger.debug(
            'Memory per chunk: {0:.5f} MB, GPU global memory: {1} MB, chunks '
            'per run: {2}.'.format(
                mem_chunk / C, max_mem / C, chunks_per_run))
        if mem_chunk > max_mem:
            raise RuntimeError('Size of single chunk exceeds GPU global '
                               'memory.')

        cmi_array = np.array([])
        if self.settings['debug']:
            distances = np.array([])
            count_var1 = np.array([])
            count_var2 = np.array([])
            count_cond = np.array([])

        for r in range(0, n_chunks, chunks_per_run):
            startidx = r*chunklength
            stopidx = min(r+chunks_per_run, n_chunks)*chunklength
            subset1 = var1[startidx:stopidx, :]
            subset2 = var2[startidx:stopidx, :]
            subset3 = conditional[startidx:stopidx, :]
            n_chunks_current_run = subset1.shape[0] // chunklength
            results = self._estimate_single_run(subset1, subset2, subset3,
                                                n_chunks_current_run)
            if self.settings['debug']:
                cmi_array = np.concatenate((cmi_array,  results[0]))
                distances = np.concatenate((distances,  results[1]))
                count_var1 = np.concatenate((count_var1, results[2]))
                count_var2 = np.concatenate((count_var2, results[3]))
                count_cond = np.concatenate((count_cond, results[4]))
            else:
                cmi_array = np.concatenate((cmi_array, results))

        if self.settings['return_counts']:
            return cmi_array, distances, count_var1, count_var2, count_cond
        else:
            return cmi_array

    def _estimate_single_run(self, var1, var2, conditional=None, n_chunks=1):
        """Estimate conditional mutual information in a single GPU run.

        This method should not be called directly, only inside estimate()
        after memory bounds have been checked.

        If conditional is None, the mutual information between var1 and var2 is
        calculated.

        Args:
            var1 : numpy array
                realisations of first variable, either a 2D numpy array where
                array dimensions represent [(realisations * n_chunks) x
                variable dimension] or a 1D array representing [realisations],
                array type should be int32
            var2 : numpy array
                realisations of the second variable (similar to var1)
            conditional : numpy array
                realisations of conditioning variable (similar to var1)
            n_chunks : int
                number of data chunks, no. data points has to be the same for
                each chunk

        Returns:
            float | numpy array
                average CMI over all samples or local CMI for individual
                samples if 'local_values'=True
        """
        # Return MI if no conditional is provided
        if conditional is None:
            est_mi = OpenCLKraskovMI(self.settings)
            return est_mi.estimate(var1, var2, n_chunks)

        # Prepare data and add noise: check if variable realisations are passed
        # as 1D or 2D arrays and have equal no. observations.
        var1 = self._ensure_two_dim_input(var1)
        var2 = self._ensure_two_dim_input(var2)
        conditional = self._ensure_two_dim_input(conditional)
        assert var1.shape[0] == var2.shape[0]
        assert var1.shape[0] == conditional.shape[0]
        assert var1.shape[0] % n_chunks == 0
        self._check_number_of_points(var1.shape[0])
        signallength = var1.shape[0]
        chunklength = signallength // n_chunks
        var1dim = var1.shape[1]
        var2dim = var2.shape[1]
        conddim = conditional.shape[1]
        pointdim = var1dim + var2dim + conddim

        # prepare padding
        signallength_orig = signallength

        if self.settings['padding']:
            # Pad time series to make GPU memory regions a multiple of 4096
            # 4096 is the largestknown value for opencl subbuffer alignment targets
            # but see comment in MI estimator above
            pad_target = 4096
            pad_size = (int(np.ceil(signallength/pad_target)) * pad_target -
                        signallength)
            pad_var1 = np.vstack(
                [var1, 999999 + 0.1 * np.random.rand(pad_size, var1dim)])
            pad_var2 = np.vstack(
                [var2, 999999 + 0.1 * np.random.rand(pad_size, var2dim)])
            pad_conditional = np.vstack(
                [conditional, 999999 + 0.1 * np.random.rand(pad_size, conddim)])
            pointset = np.hstack((pad_var1, pad_conditional, pad_var2)).T.copy()
            signallength_padded = signallength + pad_size
        else:
            pad_size = 0
            pointset = np.hstack((var1, conditional, var2)).T.copy()
            signallength_padded = signallength

        if not pointset.dtype == np.float32:
            pointset = pointset.astype(np.float32)
        if self.settings['noise_level'] > 0:
            pointset += np.random.normal(
                scale=self.settings['noise_level'],
                size=pointset.shape).astype(np.float32)

        if self.settings['debug']:
            # Print memory requirements after padding
            mem_data_pad = (self.sizeof_float *
                            pointset.shape[0] * pointset.shape[1])
            mem_dist = (self.sizeof_float * signallength_padded *
                        self.settings['kraskov_k'])
            mem_ncnt = 2 * self.sizeof_int * signallength_padded
            mem_total = mem_data_pad + mem_dist + mem_ncnt
            logger.debug(
                'Memory req. after padding: {0:.2f} MB ({1} elements) -- Padding: {2}.'.format(
                      mem_total / C, pointset.size, pad_size))

        # Set OpenCL kernel launch parameters
        if chunklength < self.devices[
                                self.settings['gpuid']].max_work_group_size:
            workitems_x = 8
        elif self.devices[self.settings['gpuid']].max_work_group_size < 256:
            workitems_x = self.devices[
                                self.settings['gpuid']].max_work_group_size
        else:
            workitems_x = 256
        NDRange_x = (workitems_x *
                     (int((signallength_padded - 1)/workitems_x) + 1))

        # Allocate and copy memory to device
        kraskov_k = self.settings['kraskov_k']
        d_pointset = cl.Buffer(
                    self.context,
                    cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                    hostbuf=pointset)
        d_src = d_pointset.get_sub_region(
                    0,
                    self.sizeof_float * signallength_padded * var1dim,
                    cl.mem_flags.READ_ONLY)
        d_cnd = d_pointset.get_sub_region(
                    self.sizeof_float * signallength_padded * var1dim,
                    self.sizeof_float * signallength_padded * conddim,
                    cl.mem_flags.READ_ONLY)
        d_distances = cl.Buffer(
                    self.context, cl.mem_flags.READ_WRITE,
                    self.sizeof_float * kraskov_k * signallength_padded)
        d_vecradius = d_distances.get_sub_region(
                    signallength_padded * (kraskov_k - 1) * self.sizeof_float,
                    signallength_padded * self.sizeof_float)
        d_npointsrange_x = cl.Buffer(self.context,
                                     cl.mem_flags.READ_WRITE,
                                     self.sizeof_int * signallength_padded)
        d_npointsrange_y = cl.Buffer(self.context, cl.mem_flags.READ_WRITE,
                                     self.sizeof_int * signallength_padded)
        d_npointsrange_z = cl.Buffer(self.context, cl.mem_flags.READ_WRITE,
                                     self.sizeof_int * signallength_padded)

        # Neighbour search in full space
        theiler_t = np.int32(self.settings['theiler_t'])
        localmem = cl.LocalMemory(self.sizeof_float * kraskov_k * workitems_x)
        self.kNN_kernel(self.queue, (NDRange_x,), (workitems_x,), d_pointset,
                        d_pointset, d_distances, np.int32(pointdim),
                        np.int32(chunklength), np.int32(signallength_padded),
                        np.int32(signallength_orig),
                        np.int32(kraskov_k),
                        theiler_t, localmem)  # MW: added signallength_orig
        distances = np.zeros(signallength_padded * kraskov_k, dtype=np.float32)
        cl.enqueue_copy(self.queue, distances, d_distances)
        self.queue.finish()

        # Range search in source and conditional
        localmem = cl.LocalMemory(self.sizeof_int * workitems_x)
        self.RS_kernel(self.queue, (NDRange_x,), (workitems_x,), d_src, d_src,
                       d_vecradius, d_npointsrange_x, var1dim + conddim,
                       chunklength, signallength_padded, signallength_orig,
                       theiler_t, localmem)  # MW: added signallength_orig
        count_src = np.zeros(signallength_padded, dtype=np.int32)
        cl.enqueue_copy(self.queue, count_src, d_npointsrange_x)

        # Range search in target and conditional
        self.RS_kernel(self.queue, (NDRange_x,), (workitems_x,), d_cnd, d_cnd,
                       d_vecradius, d_npointsrange_y, var2dim + conddim,
                       chunklength, signallength_padded,  signallength_orig,
                       theiler_t, localmem)  # MW: added signallength_orig
        count_tgt = np.zeros(signallength_padded, dtype=np.int32)
        cl.enqueue_copy(self.queue, count_tgt, d_npointsrange_y)

        # Range search in conditional
        self.RS_kernel(self.queue, (NDRange_x,), (workitems_x,), d_cnd, d_cnd,
                       d_vecradius, d_npointsrange_z, conddim, chunklength,
                       signallength_padded, signallength_orig,
                       theiler_t, localmem)  # MW: added signallength_orig
        count_cnd = np.zeros(signallength_padded, dtype=np.int32)
        cl.enqueue_copy(self.queue, count_cnd, d_npointsrange_z)

        d_pointset.release()
        d_distances.release()
        d_npointsrange_x.release()
        d_npointsrange_y.release()
        d_npointsrange_z.release()
        d_src.release()
        d_cnd.release()
        d_vecradius.release()

        # Calculate and sum digammas
        if self.settings['local_values']:
            cmi_array = -np.inf * np.ones(n_chunks * chunklength,
                                          dtype=np.float64)
            idx = 0
            for c in range(n_chunks):
                cmi = (digamma(kraskov_k) +
                       digamma(count_cnd[c*chunklength:(c+1)*chunklength]+1) -
                       digamma(count_src[c*chunklength:(c+1)*chunklength]+1) -
                       digamma(count_tgt[c*chunklength:(c+1)*chunklength]+1))
                cmi_array[idx:idx+chunklength] = cmi
                idx += chunklength

        else:
            cmi_array = -np.inf * np.ones(n_chunks, dtype=np.float64)
            for c in range(n_chunks):
                cmi = (digamma(kraskov_k) + np.mean(
                        digamma(count_cnd[c*chunklength:(c+1)*chunklength]+1) -
                        digamma(count_src[c*chunklength:(c+1)*chunklength]+1) -
                        digamma(count_tgt[c*chunklength:(c+1)*chunklength]+1)))
                cmi_array[c] = cmi
        assert signallength_orig == (c+1)*chunklength, 'Original signal length does not match no. processed points.'

        if self.settings['debug']:
            return (cmi_array,
                    distances[:signallength_orig],
                    count_src[:signallength_orig],
                    count_tgt[:signallength_orig],
                    count_cnd[:signallength_orig])
        else:
            return cmi_array


###############################
# Gaussian estimators
###############################
class OpenCLGaussian(OpenCLEstimator):
    """Abstract class for implementation of OpenCL Gaussian estimators.

    Abstract class for implementation of OpenCL Gaussian-estimators, 
    child classes implement estimators for mutual information (MI), 
    conditional mutual information (CMI), actice information storage (AIS)
    and transfer entropy (TE) using OpenCL Gaussian estimator for continuous data.

    Set common estimation parameters for OpenCL estimators.

    implemented in idtxl by Michael Lindner, 2026
    
    Args:
        settings : dict [optional]
            set estimator parameters:

            - gpuid : int [optional] - device ID used for estimation 
              (if more than one device is available on the current 
              platform) (default=0)
            - normalise : bool [optional] - z-standardise data
              (default=False)
            - noise_level : float [optional] - random noise added to the
              data (default=0)
            - local_values : bool [optional] - return local MI/TE 
              instead of average MI/TE (default=False)
    """
    def __init__(self, settings=None):
        # Get defaults for estimator settings
        super().__init__(settings)
        #self.settings = settings.copy()
        self.settings.setdefault('gpuid', int(0))
        self.settings.setdefault('normalise', False)
        self.settings.setdefault('noise_level', np.float32(1e-8))
        self.settings.setdefault('local_values', False)
        self.settings.setdefault('verbose', True)
        self.sizeof_float = int(np.dtype(np.float32).itemsize)
        self.sizeof_int = int(np.dtype(np.int32).itemsize)

        # get devices.
        self.devices, self.context, self.queue, self.device_type_str = self._get_device(
                                                self.settings['gpuid'])

        # get max_work_item_sizes of device
        self.work_group_size = self.queue.device.max_work_item_sizes[0]
        self.work_group_size2 = self.queue.device.max_work_item_sizes[1]
        self.work_group_size3 = self.queue.device.max_work_item_sizes[2]

        if self.work_group_size >= 1024:
            self.tile_2d = 32
        elif self.work_group_size >= 256:
            self.tile_2d = 16
        elif self.work_group_size >= 64:
            self.tile_2d = 8
        else:
            self.tile_2d = 4

        if self.work_group_size2 >= 1024:
            self.observation_tile = 32
        elif self.work_group_size2 >= 256:
            self.observation_tile = 16
        elif self.work_group_size2 >= 64:
            self.observation_tile = 8
        else:
            self.observation_tile = 4

        # get kernel
        self.kernel_location = resource_filename(__name__,
                                            'gpuGaussianKernel.cl')
        self._get_kernels()

        # get rng seed
        if self.settings['noise_level'] > 0:
            rng_seed = self.settings.get("rng_seed", None)
            self._rng = np.random.default_rng(rng_seed)

        self.actualValue = None
        self.surr_est_type = "fast"

    def is_parallel(self):
        return False

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

    def _get_kernels(self):
        """Return KNN and range search OpenCL kernel."""
        kernel_source = open(self.kernel_location).read()
        program = cl.Program(self.context, kernel_source).build()
        # MI
        self.means_xy = program.means_xy
        self.center_x = program.center_x
        self.center_y = program.center_y
        self.center_xy = program.center_xy
        self.covariance_one = program.covariance_one
        self.quadratic_forms_three = program.quadratic_forms_three
        # CMI
        self.means_xyz = program.means_xyz
        self.quadratic_form = program.quadratic_form
        self.concat_two = program.concat_two
        self.concat_three = program.concat_three
        self.center_one = program.center_one
        self.quadratic_forms_four = program.quadratic_forms_four

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

    # common functions
    def round_up(self, value, multiple):
        """Round value upward to a multiple of multiple."""
        return ((value + multiple - 1) // multiple) * multiple

    def _pad_features(self, d):
        return self.round_up(d, self.tile_2d)

    def _pad_observations(self, n):
        return self.round_up(n, self.observation_tile)

    def _covariance(self, centered_dev, n, d, dp, wait_for=None):
        
        covariance_dev = cla.empty(self.queue, (dp, dp), dtype=np.float64)

        event = self.covariance_one(
            self.queue,
            (
                self.round_up(dp, 16),
                self.round_up(dp, 16),
            ),
            (
                16,
                16,
            ),
            centered_dev.data,
            covariance_dev.data,
            np.int32(n),
            np.int32(d),
            np.int32(dp),
            wait_for=wait_for,
        )

        return covariance_dev, event

    @staticmethod
    def _factorize(covariance_padded, d, dp, eps):
        
        covariance = np.array(
            covariance_padded[:d, :d],
            dtype=np.float64,
            order="C",
            copy=True,
        )

        covariance += eps * np.eye(d, dtype=np.float64,)

        L = cholesky(covariance, lower=True, check_finite=False)

        inv_L = np.linalg.solve(L, np.eye(d, dtype=np.float64))

        inv_L_padded = np.zeros((dp, dp), dtype=np.float64, order="C")

        inv_L_padded[:d, :d] = inv_L

        logdet = 2.0 * np.log(np.diag(L)).sum()

        return inv_L_padded, logdet

    # for MI
    def _allocate_buffers(self, var1, var2):
        var1 = np.asarray(var1, dtype=np.float64, order="C")
        var2 = np.asarray(var2, dtype=np.float64, order="C")

        if var1.shape[0] == 0:
            raise ValueError(
                "Inputs must not be empty"
            )

        if not np.isfinite(var1).all():
            raise ValueError(
                "var1 contains NaN or infinite values"
            )

        if not np.isfinite(var2).all():
            raise ValueError(
                "var2 contains NaN or infinite values"
            )

        n, dx = var1.shape
        _, dy = var2.shape
        dxy = dx+dy

        px = self._pad_features(dx)
        py = self._pad_features(dy)
        pxy = self._pad_features(dx + dy)
        n_pad = self._pad_observations(n)

        # Only the original input arrays are transferred.
        x_dev = cla.to_device(self.queue, var1)
        y_dev = cla.to_device(self.queue, var2)

        # means
        mean_x_dev = cla.empty(self.queue, (dx,), dtype=np.float64)
        mean_y_dev = cla.empty(self.queue, (dy,), dtype=np.float64)

        # Centered arrays
        cx_dev = cla.empty(self.queue, (n_pad, px), dtype=np.float64)
        cy_dev = cla.empty(self.queue, (n_pad, py), dtype=np.float64)
        cxy_dev = cla.empty(self.queue, (n_pad, pxy), dtype=np.float64)

        return {
            "x_dev": x_dev,
            "y_dev": y_dev,
            "mean_x_dev": mean_x_dev,
            "mean_y_dev": mean_y_dev,
            "cx_dev": cx_dev,
            "cy_dev": cy_dev,
            "cxy_dev": cxy_dev,
            "n": n,
            "n_pad": n_pad,
            "dx": dx,
            "dy": dy,
            "dxy": dxy,
            "px": px,
            "py": py,
            "pxy": pxy,
        }

    def _center_inputs(self, b):
        n = b["n"]
        n_pad = b["n_pad"]
        dx = b["dx"]
        dy = b["dy"]
        px = b["px"]
        py = b["py"]
        pxy = b["pxy"]

        mean_event = self.means_xy(
            self.queue,
            (
                self.round_up(
                    max(dx, dy),
                    self.work_group_size,
                ),
            ),
            (
                self.work_group_size,
            ),
            b["x_dev"].data,
            b["y_dev"].data,
            b["mean_x_dev"].data,
            b["mean_y_dev"].data,
            np.int32(n),
            np.int32(dx),
            np.int32(dy),
        )

        center_x_event = self.center_x(
            self.queue,
            (
                self.round_up(n, self.tile_2d),
                self.round_up(px, self.tile_2d),
            ),
            (
                self.tile_2d,
                self.tile_2d,
            ),
            b["x_dev"].data,
            b["mean_x_dev"].data,
            b["cx_dev"].data,
            np.int32(n),
            np.int32(n_pad),
            np.int32(dx),
            np.int32(px),
            wait_for=[mean_event],
        )

        center_y_event = self.center_y(
            self.queue,
            (
                self.round_up(n, self.tile_2d),
                self.round_up(py, self.tile_2d),
            ),
            (
                self.tile_2d,
                self.tile_2d,
            ),
            b["y_dev"].data,
            b["mean_y_dev"].data,
            b["cy_dev"].data,
            np.int32(n),
            np.int32(n_pad),
            np.int32(dy),
            np.int32(py),
            wait_for=[mean_event],
        )

        center_xy_event = self.center_xy(
            self.queue,
            (
                self.round_up(n_pad, self.tile_2d),
                self.round_up(pxy, self.tile_2d),
            ),
            (
                self.tile_2d,
                self.tile_2d
            ),
            b["cx_dev"].data,
            b["cy_dev"].data,
            b["cxy_dev"].data,
            np.int32(n),
            np.int32(dx),
            np.int32(dy),
            np.int32(px),
            np.int32(py),
            np.int32(pxy),
            np.int32(n_pad),
            wait_for=[center_x_event, center_y_event],
        )

        return center_x_event, center_y_event, center_xy_event

    # for CMI
    def _allocate_buffers_cmi(self, x, y, z):
        n, dx = x.shape
        _, dy = y.shape
        _, dz = z.shape

        dxz = dx + dz
        dyz = dy + dz
        dxyz = dx + dy + dz

        px = self._pad_features(dx)
        py = self._pad_features(dy)
        pz = self._pad_features(dz)

        pxz = self._pad_features(dxz)
        pyz = self._pad_features(dyz)
        pxyz = self._pad_features(dxyz)

        n_pad = self._pad_observations(n)

        # Large host-to-device transfers.
        x_dev = cla.to_device(self.queue, x)
        y_dev = cla.to_device(self.queue, y)
        z_dev = cla.to_device(self.queue, z)

        # Means
        mean_x_dev = cla.empty(self.queue, (dx,), dtype=np.float64)
        mean_y_dev = cla.empty(self.queue, (dy,), dtype=np.float64)
        mean_z_dev = cla.empty(self.queue, (dz,), dtype=np.float64)

        # Centered individual arrays
        cx_dev = cla.empty(self.queue, (n_pad, px), dtype=np.float64)
        cy_dev = cla.empty(self.queue, (n_pad, py), dtype=np.float64)
        cz_dev = cla.empty(self.queue, (n_pad, pz), dtype=np.float64)

        # Centered concatenated arrays
        cxz_dev = cla.empty(self.queue, (n_pad, pxz), dtype=np.float64)
        cyz_dev = cla.empty(self.queue, (n_pad, pyz), dtype=np.float64)
        cxyz_dev = cla.empty(self.queue, (n_pad, pxyz), dtype=np.float64)

        return {
            "x_dev": x_dev,
            "y_dev": y_dev,
            "z_dev": z_dev,
            "mean_x_dev": mean_x_dev,
            "mean_y_dev": mean_y_dev,
            "mean_z_dev": mean_z_dev,
            "cx_dev": cx_dev,
            "cy_dev": cy_dev,
            "cz_dev": cz_dev,
            "cxz_dev": cxz_dev,
            "cyz_dev": cyz_dev,
            "cxyz_dev": cxyz_dev,
            "n": n,
            "n_pad": n_pad,
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "dxz": dxz,
            "dyz": dyz,
            "dxyz": dxyz,
            "px": px,
            "py": py,
            "pz": pz,
            "pxz": pxz,
            "pyz": pyz,
            "pxyz": pxyz,
        }

    def _center_inputs_cmi(self, b):
        n = b["n"]
        dx = b["dx"]
        dy = b["dy"]
        dz = b["dz"]

        mean_event = self.means_xyz(
            self.queue,
            (
                self.round_up(
                    max(dx, dy, dz),
                    self.work_group_size,
                ),
            ),
            (
                self.work_group_size,
            ),
            b["x_dev"].data,
            b["y_dev"].data,
            b["z_dev"].data,
            b["mean_x_dev"].data,
            b["mean_y_dev"].data,
            b["mean_z_dev"].data,
            np.int32(n),
            np.int32(dx),
            np.int32(dy),
            np.int32(dz),
        )

        center_x_event = self.center_one(
            self.queue,
            (
                self.round_up(n, self.tile_2d),
                self.round_up(b["px"], self.tile_2d),
            ),
            (
                self.tile_2d,
                self.tile_2d,
            ),
            b["x_dev"].data,
            b["mean_x_dev"].data,
            b["cx_dev"].data,
            np.int32(n),
            np.int32(dx),
            np.int32(b["px"]),
            wait_for=[mean_event],
        )

        center_y_event = self.center_one(
            self.queue,
            (
                self.round_up(n, self.tile_2d),
                self.round_up(b["py"], self.tile_2d),
            ),
            (
                self.tile_2d,
                self.tile_2d,
            ),
            b["y_dev"].data,
            b["mean_y_dev"].data,
            b["cy_dev"].data,
            np.int32(n),
            np.int32(dy),
            np.int32(b["py"]),
            wait_for=[mean_event],
        )

        center_z_event = self.center_one(
            self.queue,
            (
                self.round_up(n, self.tile_2d),
                self.round_up(b["pz"], self.tile_2d),
            ),
            (
                self.tile_2d,
                self.tile_2d,
            ),
            b["z_dev"].data,
            b["mean_z_dev"].data,
            b["cz_dev"].data,
            np.int32(n),
            np.int32(dz),
            np.int32(b["pz"]),
            wait_for=[mean_event],
        )

        return [
            center_x_event,
            center_y_event,
            center_z_event,
        ]


class OpenCLGaussianMI(OpenCLGaussian):
    """Calculate mutual information with OpenCL Kraskov implementation.

    Calculate the mutual information (MI) between two variables using OpenCL
    GPU-code. See parent class for references.

    Results are returned in nats.

    implemented in idtxl by Michael Lindner, 2026
    
    Args:
        settings : dict [optional]
            set estimator parameters:

            - gpuid : int [optional] - device ID used for estimation (if more
              than one device is available on the current platform) (default=0)
            - normalise : bool [optional] - z-standardise data (default=False)
            - noise_level : float [optional] - random noise added to the data
              (default=1e-8)
            - lag_mi : int [optional] - time difference in samples to calculate
              the lagged MI between processes (default=0)
            - local_values : bool [optional] - return local MI/TE instead of
              average MI/TE (default=False)
    """

    def __init__(self, settings=None):
        # Set default estimator settings.
        super().__init__(settings)
        self.settings.setdefault('lag_mi', 0)

        
    def calculateLocalMI(self):
        """calculate local mutual information for gaussian data
        This function can not be called directly! You need to call .estimate(X,Y)
        with estimator setting local_values = True."""

        if not hasattr(self, 'flag'):
            raise RuntimeError('calculateLocalMI can not be called directly! You need to call .estimate(X,Y) '
                'with estimator setting local_values = True!')

        var1, var2 = self.get_data()

        b = self._allocate_buffers(var1, var2)

        # center inputs on GPU
        _, _, center_xy_event = self._center_inputs(b)

        # calculate covariances on GPU
        cov_x_dev, cov_x_event = self._covariance(
            b["cx_dev"],
            b["n"],
            b["dx"],
            b["px"],
            wait_for=[center_xy_event])

        cov_y_dev, cov_y_event = self._covariance(
            b["cy_dev"],
            b["n"],
            b["dy"],
            b["py"],
            wait_for=[center_xy_event])

        cov_xy_dev, cov_xy_event = self._covariance(
            b["cxy_dev"],
            b["n"],
            b["dx"] + b["dy"],
            b["pxy"],
            wait_for=[center_xy_event])

        # Only small covariance matrices are copied to the CPU.
        cov_x_event.wait()
        cov_y_event.wait()
        cov_xy_event.wait()

        cov_x = cov_x_dev.get(self.queue)
        cov_y = cov_y_dev.get(self.queue)
        cov_xy = cov_xy_dev.get(self.queue)

        # cholesky logdet on CPU
        eps = 1e-10
        inv_lx, logdet_x = self._factorize(cov_x,
                                            b["dx"],
                                            b["px"],
                                            eps)

        inv_ly, logdet_y = self._factorize(cov_y,
                                            b["dy"],
                                            b["py"],
                                            eps)

        inv_lxy, logdet_xy = self._factorize(cov_xy,
                                            b["dx"] + b["dy"],
                                            b["pxy"],
                                            eps)

        # copy to GPU and calculate quadratic forms on GPU
        inv_lx_dev = cla.to_device(self.queue, inv_lx)
        inv_ly_dev = cla.to_device(self.queue, inv_ly)
        inv_lxy_dev = cla.to_device(self.queue, inv_lxy)

        qx_dev = cla.empty(self.queue, (b["n"],), dtype=np.float64)
        qy_dev = cla.empty(self.queue, (b["n"],), dtype=np.float64)
        qxy_dev = cla.empty(self.queue, (b["n"],), dtype=np.float64)

        q_event = self.quadratic_forms_three(
            self.queue,
            (
                self.round_up(
                    b["n"],
                    self.work_group_size,
                ),
            ),
            (
                self.work_group_size,
            ),
            b["cx_dev"].data,
            b["cy_dev"].data,
            b["cxy_dev"].data,
            inv_lx_dev.data,
            inv_ly_dev.data,
            inv_lxy_dev.data,
            qx_dev.data,
            qy_dev.data,
            qxy_dev.data,
            np.int32(b["n"]),
            np.int32(b["dx"]),
            np.int32(b["dy"]),
            np.int32(b["dx"] + b["dy"]),
            np.int32(b["px"]),
            np.int32(b["py"]),
            np.int32(b["pxy"]),
        )

        q_event.wait()

        # get data from GPU
        qx = qx_dev.get(self.queue)
        qy = qy_dev.get(self.queue)
        qxy = qxy_dev.get(self.queue)

        log_two_pi = np.log(2.0 * np.pi)
        logpdf_x = -0.5 * (b["dx"] * log_two_pi + logdet_x + qx)
        logpdf_y = -0.5 * (b["dy"] * log_two_pi + logdet_y + qy)
        logpdf_xy = -0.5 * ((b["dx"] + b["dy"]) * log_two_pi 
            + logdet_xy + qxy)

        return (logpdf_xy - logpdf_x - logpdf_y)

    """
    def calculateAverageMI(self):
        calculate local mutual information for gaussian data
        This function can not be called directly! You need to call .estimate(X,Y)
        with estimator setting local_values = False.

        if not hasattr(self, 'flag'):
            raise RuntimeError('calculateAverageMI can not be called directly! You need to call .estimate(X,Y) '
                'with estimator setting local_values = False!')

        lcmi = self.calculateLocalMI()

        return np.mean(lcmi)
    """

    def estimate(self, var1, var2):
        """Estimate mutual information.

        Args:
            var1 : numpy array
                realisations of first variable, either a 2D numpy array 
                where array dimensions represent 
                [(realisations * n_chunks) x variable dimension] 
                or a 1D array representing [realisations], array type 
                should be int32
            var2 : numpy array
                realisations of the second variable (similar to var1)

        Returns:
            float | numpy array
                average MI over all samples or local MI for individual
                samples if 'local_values'=True
            numpy arrays
                distances and neighborhood counts for var1 and var2 if
                debug=True and return_counts=True
        """
        # Check the input data
        var1 = self._ensure_two_dim_input(var1)
        var2 = self._ensure_two_dim_input(var2)

        assert var1.shape[0] == var2.shape[0]
        
        # Shift variables to calculate a lagged MI.
        if self.settings['lag_mi'] > 0:
            var1 = var1[:-self.settings['lag_mi'], :]
            var2 = var2[self.settings['lag_mi']:, :]

        # for analystic distribution measurement
        self.n_samples = var1.shape[0]
        self.var1_dim = var1.shape[1]
        self.var2_dim = var2.shape[1]

        self.set_data("MI", var1, var2)

        if self.settings['local_values']:
            mi = self.calculateLocalMI()
            self.actualValue = np.mean(mi)
        else:
            mi = np.mean(self.calculateLocalMI())
            self.actualValue = mi

        self.remove_data()

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


class OpenCLGaussianCMI(OpenCLGaussian):
    """Calculate conditional mutual inform with OpenCL Gaussian implementation.

    Calculate the conditional mutual information (CMI) between three variables
    using OpenCL GPU-code. If no conditional is given (is None), the function
    returns the mutual information between var1 and var2. See parent class for
    references.

    Results are returned in nats.

    implemented in idtxl by Michael Lindner, 2026
    
    Args:
        settings : dict [optional]
            sets estimation parameters:
            
            - gpuid : int [optional] - device ID used for estimation 
              (if more than one device is available on the current 
              platform) (default=0)
            - normalise : bool [optional] - z-standardise data
              (default=False)
            - noise_level : float [optional] - random noise added to the
              data (default=0)
            - local_values : bool [optional] - return local TE instead of
              average TE (default=False)

    """

    def __init__(self, settings=None):
        # Set default estimator settings.
        super().__init__(settings)


    def calculateAverageCMI(self):
        """calculate conditional mutual information for gaussian data
        This function can not be called directly! You need to call .estimate(X,Y,Z)
        with estimator setting local_values = True."""

        if not hasattr(self, 'flag'):
            raise RuntimeError('calculateAverageCMI can not be called directly! You need to call .estimate(X,Y,Z) '
                'with estimator setting local_values = False!')

        var1, var2, conditional = self.get_data()

        b = self._allocate_buffers_cmi(var1, var2, conditional)

        # center inputs on GPU
        center_events = self._center_inputs_cmi(b)

        # concatenate data on GPU
        cxz_event = self.concat_two(
            self.queue,
            (
                self.round_up(b["n"], self.tile_2d),
                self.round_up(b["pxz"], self.tile_2d),
            ),
            (self.tile_2d, self.tile_2d),
            b["cx_dev"].data,
            b["cz_dev"].data,
            b["cxz_dev"].data,
            np.int32(b["n"]),
            np.int32(b["dx"]),
            np.int32(b["dz"]),
            np.int32(b["dxz"]),
            np.int32(b["px"]),
            np.int32(b["pz"]),
            np.int32(b["pxz"]),
            wait_for=center_events,
        )

        cyz_event = self.concat_two(
            self.queue,
            (
                self.round_up(b["n"], self.tile_2d),
                self.round_up(b["pyz"], self.tile_2d),
            ),
            (self.tile_2d, self.tile_2d),
            b["cy_dev"].data,
            b["cz_dev"].data,
            b["cyz_dev"].data,
            np.int32(b["n"]),
            np.int32(b["dy"]),
            np.int32(b["dz"]),
            np.int32(b["dyz"]),
            np.int32(b["py"]),
            np.int32(b["pz"]),
            np.int32(b["pyz"]),
            wait_for=center_events,
        )

        cxyz_event = self.concat_three(
            self.queue,
            (
                self.round_up(b["n"], self.tile_2d),
                self.round_up(b["pxyz"], self.tile_2d),
            ),
            (self.tile_2d, self.tile_2d),
            b["cx_dev"].data,
            b["cy_dev"].data,
            b["cz_dev"].data,
            b["cxyz_dev"].data,
            np.int32(b["n"]),
            np.int32(b["dx"]),
            np.int32(b["dy"]),
            np.int32(b["dz"]),
            np.int32(b["px"]),
            np.int32(b["py"]),
            np.int32(b["pz"]),
            np.int32(b["pxyz"]),
            wait_for=center_events,
        )

        # calculate covariances on GPU
        cov_xz_dev, cov_xz_event = self._covariance(
            b["cxz_dev"],
            b["n"],
            b["dxz"],
            b["pxz"],
            wait_for=[cxz_event])

        cov_yz_dev, cov_yz_event = self._covariance(
            b["cyz_dev"],
            b["n"],
            b["dyz"],
            b["pyz"],
            wait_for=[cyz_event])

        cov_z_dev, cov_z_event = self._covariance(
            b["cz_dev"],
            b["n"],
            b["dz"],
            b["pz"],
            wait_for=center_events)

        cov_xyz_dev, cov_xyz_event = self._covariance(
            b["cxyz_dev"],
            b["n"],
            b["dxyz"],
            b["pxyz"],
            wait_for=[cxyz_event])

        del cxyz_event

        # Only small covariance matrices are copied to the CPU.
        cov_xz_event.wait()
        cov_yz_event.wait()
        cov_xyz_event.wait()
        cov_z_event.wait()

        cov_xz = cov_xz_dev.get(self.queue)
        cov_yz = cov_yz_dev.get(self.queue)
        cov_xyz = cov_xyz_dev.get(self.queue)
        cov_z = cov_z_dev.get(self.queue)
        
        # cholesky logdet on CPU
        eps = 1e-10
        inv_lxz, logdet_xz = self._factorize(cov_xz, b["dxz"], b["pxz"], eps)
        inv_lyz, logdet_yz = self._factorize(cov_yz, b["dyz"], b["pyz"], eps)
        inv_lxyz, logdet_xyz = self._factorize(cov_xyz, 
            b["dx"] + b["dy"] + b["dz"], 
            b["pxyz"],
            eps)
        inv_lz, logdet_z = self._factorize(cov_z,
            b["dz"],
            b["pz"],
            eps)

        mi = 0.5 * (
            logdet_xz
            + logdet_yz
            - logdet_z
            - logdet_xyz)

        return mi
        
    def calculateLocalCMI(self):
        """calculate local conditional mutual information for gaussian data
        This function can not be called directly! You need to call .estimate(X,Y,Z)
        with estimator setting local_values = True."""

        if not hasattr(self, 'flag'):
            raise RuntimeError('calculateLocalCMI can not be called directly! You need to call .estimate(X,Y,Z) '
                'with estimator setting local_values = True!')

        var1, var2, conditional = self.get_data()

        b = self._allocate_buffers_cmi(var1, var2, conditional)

        center_events = self._center_inputs_cmi(b)

        # Build XZ.
        cxz_event = self.concat_two(
            self.queue,
            (
                self.round_up(b["n"], self.tile_2d),
                self.round_up(b["pxz"], self.tile_2d),
            ),
            (self.tile_2d, self.tile_2d),
            b["cx_dev"].data,
            b["cz_dev"].data,
            b["cxz_dev"].data,
            np.int32(b["n"]),
            np.int32(b["dx"]),
            np.int32(b["dz"]),
            np.int32(b["dxz"]),
            np.int32(b["px"]),
            np.int32(b["pz"]),
            np.int32(b["pxz"]),
            wait_for=center_events,
        )

        # Build YZ.
        cyz_event = self.concat_two(
            self.queue,
            (
                self.round_up(b["n"], self.tile_2d),
                self.round_up(b["pyz"], self.tile_2d),
            ),
            (self.tile_2d, self.tile_2d),
            b["cy_dev"].data,
            b["cz_dev"].data,
            b["cyz_dev"].data,
            np.int32(b["n"]),
            np.int32(b["dy"]),
            np.int32(b["dz"]),
            np.int32(b["dyz"]),
            np.int32(b["py"]),
            np.int32(b["pz"]),
            np.int32(b["pyz"]),
            wait_for=center_events,
        )

        # Build XYZ directly from X, Y, Z.
        cxyz_event = self.concat_three(
            self.queue,
            (
                self.round_up(b["n"], self.tile_2d),
                self.round_up(b["pxyz"], self.tile_2d),
            ),
            (self.tile_2d, self.tile_2d),
            b["cx_dev"].data,
            b["cy_dev"].data,
            b["cz_dev"].data,
            b["cxyz_dev"].data,
            np.int32(b["n"]),
            np.int32(b["dx"]),
            np.int32(b["dy"]),
            np.int32(b["dz"]),
            np.int32(b["px"]),
            np.int32(b["py"]),
            np.int32(b["pz"]),
            np.int32(b["pxyz"]),
            wait_for=center_events,
        )

        # Compute the four covariance matrices.
        cov_z_dev, cov_z_event = self._covariance(
            b["cz_dev"],
            b["n"],
            b["dz"],
            b["pz"],
            wait_for=center_events,
        )

        cov_xz_dev, cov_xz_event = self._covariance(
            b["cxz_dev"],
            b["n"],
            b["dxz"],
            b["pxz"],
            wait_for=[cxz_event],
        )

        cov_yz_dev, cov_yz_event = self._covariance(
            b["cyz_dev"],
            b["n"],
            b["dyz"],
            b["pyz"],
            wait_for=[cyz_event],
        )

        cov_xyz_dev, cov_xyz_event = self._covariance(
            b["cxyz_dev"],
            b["n"],
            b["dxyz"],
            b["pxyz"],
            wait_for=[cxyz_event],
        )

        # Wait for all covariance kernels before device-to-host copies.
        cov_z_event.wait()
        cov_xz_event.wait()
        cov_yz_event.wait()
        cov_xyz_event.wait()

        cov_z = cov_z_dev.get(self.queue)
        cov_xz = cov_xz_dev.get(self.queue)
        cov_yz = cov_yz_dev.get(self.queue)
        cov_xyz = cov_xyz_dev.get(self.queue)

        # Factorize on the CPU.
        eps = 1e-10
        inv_lz, logdet_z = self._factorize(cov_z, b["dz"], b["pz"], eps)
        inv_lxz, logdet_xz = self._factorize(cov_xz, b["dxz"], b["pxz"], eps)
        inv_lyz, logdet_yz = self._factorize(cov_yz, b["dyz"], b["pyz"], eps)
        inv_lxyz, logdet_xyz = self._factorize(cov_xyz, b["dxyz"], b["pxyz"], 
            eps)

        # Transfer the small inverse Cholesky matrices to the GPU.
        inv_lxz_dev = cla.to_device(self.queue, inv_lxz)
        inv_lyz_dev = cla.to_device(self.queue, inv_lyz)
        inv_lxyz_dev = cla.to_device(self.queue, inv_lxyz)
        inv_lz_dev = cla.to_device(self.queue, inv_lz)

        qxz_dev = cla.empty(self.queue, (b["n"],), dtype=np.float64)
        qyz_dev = cla.empty(self.queue, (b["n"],), dtype=np.float64)
        qxyz_dev = cla.empty(self.queue, (b["n"],), dtype=np.float64)
        qz_dev = cla.empty(self.queue, (b["n"],), dtype=np.float64)

        # Compute all four quadratic forms on GPU.
        q_event = self.quadratic_forms_four(
            self.queue,
            (
                self.round_up(
                    b["n"],
                    self.work_group_size,
                ),
            ),
            (
                self.work_group_size,
            ),
            b["cz_dev"].data,
            b["cxz_dev"].data,
            b["cyz_dev"].data,
            b["cxyz_dev"].data,
            inv_lz_dev.data,
            inv_lxz_dev.data,
            inv_lyz_dev.data,
            inv_lxyz_dev.data,
            qz_dev.data,
            qxz_dev.data,
            qyz_dev.data,
            qxyz_dev.data,
            np.int32(b["n"]),
            np.int32(b["dz"]),
            np.int32(b["dxz"]),
            np.int32(b["dyz"]),
            np.int32(b["dxyz"]),
            np.int32(b["pz"]),
            np.int32(b["pxz"]),
            np.int32(b["pyz"]),
            np.int32(b["pxyz"]),
        )

        q_event.wait()

        # Four device-to-host transfers.
        qz = qz_dev.get(self.queue)
        qxz = qxz_dev.get(self.queue)
        qyz = qyz_dev.get(self.queue)
        qxyz = qxyz_dev.get(self.queue)

        # calculate mi
        local_mi = 0.5 * (
            (logdet_xz + qxz)
            + (logdet_yz + qyz)
            - (logdet_z + qz)
            - (logdet_xyz + qxyz))

        return local_mi
        
    def estimate(self, var1, var2, conditional=None):
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
            self.est_mi = OpenCLGaussianMI(self.settings)
            return self.est_mi.estimate(var1, var2)
        else:
            assert(conditional.size != 0), 'Conditional Array is empty.'

        # Check the input data
        var1 = self._ensure_two_dim_input(var1)
        var2 = self._ensure_two_dim_input(var2)
        conditional = self._ensure_two_dim_input(conditional)

        assert (
            var1.shape[0] == var2.shape[0] == conditional.shape[0]
        ), f"Unequal number of observations (var1: {var1.shape[0]}, \
            var2: {var2.shape[0]}, conditional: {conditional.shape[0]})"

        # Normalise data
        if self.settings['normalise']:
            var1 = self._normalise_data(var1)
            var2 = self._normalise_data(var2)
            conditional = self._normalise_data(conditional)

        # Add noise to avoid duplicate points
        # Do not add noise inplace, because it would change the input data
        if self.settings['noise_level'] > 0:
            var1 = var1 + self._rng.normal(0,
                self.settings['noise_level'],
                var1.shape)
            var2 = var2 + self._rng.normal(0,
                self.settings['noise_level'],
                var2.shape)
            conditional = conditional + self._rng.normal(0,
                self.settings['noise_level'],
                conditional.shape)

        # for analystic distribution measurement
        self.n_samples = var1.shape[0]
        self.var1_dim = var1.shape[1]
        self.var2_dim = var2.shape[1]

        #print(var1.shape)
        #print(var2.shape)
        #print(conditional.shape)

        var1 = np.ascontiguousarray(var1, dtype=np.float64)
        var2 = np.ascontiguousarray(var2, dtype=np.float64)
        conditional = np.ascontiguousarray(conditional, dtype=np.float64)
        self.set_data("CMI", var1, var2, conditional)

        if self.settings['local_values']:
            cmi = self.calculateLocalCMI()
            self.actualValue = np.mean(cmi)
        else:
            cmi = np.mean(self.calculateLocalCMI())
            self.actualValue = cmi

        self.remove_data()

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
            mi = OpenCLGaussianMI(self.settings)
            mi.estimate(var1, var2)
            return mi.computeSignificance()
        else:
            self.estimate(var1, var2, conditional)
            return self.computeSignificance()


class OpenCLGaussianAIS(OpenCLGaussian):
    """Calculate active information storage with OpenCL Gaussian 
    implementation.

    Calculate active information storage (AIS) for some process using OpenCL
    implementation of the Gaussian estimator. AIS is defined as the
    mutual information between the processes' past state and current value.

    The past state needs to be defined in the settings dictionary, where a past
    state is defined as a uniform embedding with parameters history and tau.
    The history describes the number of samples taken from a processes' past,
    tau describes the embedding delay, i.e., the spacing between every two
    samples from the processes' past.

    Results are returned in nats.

    implemented in idtxl by Michael Lindner, 2026
    
    Args:
        settings : dict
            sets estimation parameters:
            - gpuid : int [optional] - device ID used for estimation (if more
              than one device is available on the current platform) (default=0)
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

        self.set_data("AIS", process_past, process_current)

        if self.settings['local_values']:
            ais = OpenCLGaussianMI.calculateLocalMI(self)
            # correction to compare with JidtGaussianTE results
            ais = np.hstack([np.zeros(startFirstPoint+1), ais])
            self.actualValue = np.mean(ais)
        else:
            ais = np.mean(OpenCLGaussianMI.calculateLocalMI(self))
            self.actualValue = ais

        self.remove_data()

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


class OpenCLGaussianTE(OpenCLGaussian):
    """Calculate transfer entropy with OpenCL Gaussian implementation.

    Calculate transfer entropy between a source and a target variable using
    OpenCL implementation of the Gaussian estimator. Transfer entropy is
    defined as the conditional mutual information between the source's past
    state and the target's current value, conditional on the target's past.

    Past states need to be defined in the settings dictionary, where a past
    state is defined as a uniform embedding with parameters history and tau.
    The history describes the number of samples taken from a variable's past,
    tau descrices the embedding delay, i.e., the spacing between every two
    samples from the processes' past.

    Results are returned in nats.        

    implemented in idtxl by Michael Lindner, 2026
    
    Args:
        settings : dict
            sets estimation parameters:
            
            - gpuid : int [optional] - device ID used for estimation (if more
              than one device is available on the current platform) (default=0)
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
    def __init__(self, settings=None):
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

        self.set_data("TE", source_past, target_current, target_past)

        if self.settings['local_values']:
            te = OpenCLGaussianCMI.calculateLocalCMI(self)
            ## correction to compare with JidtGaussianTE results
            te = np.hstack([np.zeros(startFirstPoint+1), te])
            self.actualValue = np.mean(te)
        else:
            te = OpenCLGaussianCMI.calculateAverageCMI(self)
            self.actualValue = te

        self.remove_data()

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


class OpenCLGaussianCTE(OpenCLGaussian):
    """Calculate conditional transfer entropy with OpenCL Gaussian 
    implementation.

    Calculate transfer entropy between a source and a target variable using
    OpenCL implementation of the Gaussian estimator. Transfer entropy is
    defined as the conditional mutual information between the source's past
    state and the target's current value, conditional on the target's and 
    another conditional's past.

    Past states need to be defined in the settings dictionary, where a past
    state is defined as a uniform embedding with parameters history and tau.
    The history describes the number of samples taken from a variable's past,
    tau descrices the embedding delay, i.e., the spacing between every two
    samples from the processes' past.

    Results are returned in nats.        

    implemented in idtxl by Michael Lindner, 2026
    
    Args:
        settings : dict
            sets estimation parameters:
            
            - gpuid : int [optional] - device ID used for estimation 
              (if more than one device is available on the current 
              platform) (default=0)
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
            est = OpenCLGaussianTE(self.settings)
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

        self.set_data("CTE", source_past, target_current, condCombine)

        if self.settings['local_values']:
            cte = OpenCLGaussianCMI.calculateLocalCMI(self)
            cte = np.hstack([np.zeros(startFirstPoint+1), cte])
            self.actualValue = np.mean(cte)
        else:
            cte = OpenCLGaussianCMI.calculateAverageCMI(self)
            self.actualValue = cte

        self.remove_data()

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
            te = OpenCLGaussianTE(self.settings)
            te.estimate(source, target)
            return te.computeSignificance()
        else:
            self.estimate(source, target, conditional)
            return self.computeSignificance()


###############################
# Discrete estimators
###############################
class OpenCLDiscrete(OpenCLEstimator):
    """Abstract class for implementation of OpenCL Discrete-estimators.

    Abstract class for implementation of OpenCL Discrete-estimators, child
    classes implement estimators for mutual information (MI), conditional
    mutual information (CMI), active information storage (AIS), transfer
    entropy (TE) using OpenCL estimator for discrete data.

    implemented in idtxl by Michael Lindner, 2026

    Args:
        settings : dict [optional]
            set estimator parameters:

            - gpuid : int [optional] - device ID used for estimation
              (if more than one device is available on the current
              platform) (default=0)
            - normalise : bool [optional] - z-standardise data (default=False)
            - noise_level : float [optional] - random noise added to the data
              (default=0)
            - local_values : bool [optional] - return local MI/TE instead of
              average MI/TE (default=False)
    """

    def __init__(self, settings):
        settings.setdefault('gpuid', int(0))
        settings.setdefault('discretise_method', 'none')
        settings.setdefault('normalise', False)
        settings.setdefault('noise_level', np.float32(1e-8))
        settings.setdefault('local_values', False)
        super().__init__(settings)

        # get devices.
        self.devices, self.context, self.queue, self.device_type_str = self._get_device(
            self.settings['gpuid'])

        # get max_work_item_sizes of device
        self.work_group_size = self.queue.device.max_work_item_sizes[0]
        self.work_group_size2 = self.queue.device.max_work_item_sizes[1]
        self.work_group_size3 = self.queue.device.max_work_item_sizes[2]

        if self.work_group_size >= 1024:
            self.tile_2d = 32
        elif self.work_group_size >= 256:
            self.tile_2d = 16
        elif self.work_group_size >= 64:
            self.tile_2d = 8
        else:
            self.tile_2d = 4

        if self.work_group_size2 >= 1024:
            self.observation_tile = 32
        elif self.work_group_size2 >= 256:
            self.observation_tile = 16
        elif self.work_group_size2 >= 64:
            self.observation_tile = 8
        else:
            self.observation_tile = 4

        # get kernels
        self.kernel_location = resource_filename(__name__,
                                                 'gpuDiscreteKernel.cl')
        self._get_kernels()

        # PyOpenCL generates a parallel reduction kernel.
        self.reduce = ReductionKernel(
            self.context,
            dtype_out=np.float32,
            neutral="0.0f",
            reduce_expr="a+b",
            map_expr="x[i]",
            arguments="__global const float *x",
        )

        # get mem flag
        self.mf = cl.mem_flags

        # get rng seed
        if self.settings['noise_level'] > 0:
            rng_seed = settings.get("rng_seed", None)
            self._rng = np.random.default_rng(rng_seed)

        self.actualValue = None
        self.surr_est_type = "fast"

    def is_parallel(self):
        return False

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

    def _get_kernels(self):
        """Return KNN and range search OpenCL kernel."""
        kernel_source = open(self.kernel_location).read()
        program = cl.Program(self.context, kernel_source).build()
        # MI
        self.histogram_joint = program.histogram_joint
        self.local_mi = program.local_mi
        self.mi_terms = program.mi_terms
        # CMI
        self.count_cmi = program.count_cmi
        self.local_cmi = program.local_cmi

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

    def encode_multidim_states(self, arr):
        """
        Map each row of an integer-valued array to a single integer state.
        Optimized: uses intp dtype, vectorized stride computation, dot product.
        """
        arr = np.asarray(arr, dtype=np.intp)  # Use platform-native int for indexing

        if arr.ndim == 1:
            mn = arr.min()
            codes = arr - mn  # Avoid explicit astype
            n_states = codes.max() + 1
            return np.ascontiguousarray(codes), n_states

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
        return np.ascontiguousarray(codes), n_states

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


class OpenCLDiscreteMI(OpenCLDiscrete):
    """Calculate MI with OpenCL discrete-variable implementation.

    Calculate the mutual information (MI) between two variables.

    Results are returned in bits.

    implemented in idtxl by Michael Lindner, 2026

    Args:
        settings : dict [optional]
            sets estimation parameters:

            - gpuid : int [optional] - device ID used for estimation
              (if more than one device is available on the current
              platform) (default=0)
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

    def calculateLocalMI(self,):
        """Calculate average mutual information for discrete data.
        This function can not be called directly! You need to call .estimate(X,Y)
        with estimator setting local_values = True."""

        if not hasattr(self, 'flag'):
            raise RuntimeError('calculateLocalMI can not be called directly! You need to call .estimate(X,Y) '
                               'with estimator setting local_values = True!')

        var1, var2 = self.get_data()
        X = np.asarray(var1)
        Y = np.asarray(var2)

        if X.shape != Y.shape:
            raise ValueError(
                f"Shape mismatch: X.shape={X.shape}, Y.shape={Y.shape}")

        orig_shape = X.shape
        x_flat = np.ascontiguousarray(X.ravel())
        y_flat = np.ascontiguousarray(Y.ravel())
        n = x_flat.size

        if n == 0:
            return np.empty(orig_shape, dtype=np.float64)

        # This part remains on the CPU.
        # It is generally preferable to avoid transferring the original
        # possibly-large categorical arrays to the device.
        _, x_idx = np.unique(x_flat, return_inverse=True)
        _, y_idx = np.unique(y_flat, return_inverse=True)

        # GPU kernels use 32-bit indices and counts.
        x_idx = np.ascontiguousarray(x_idx, dtype=np.int32)
        y_idx = np.ascontiguousarray(y_idx, dtype=np.int32)

        nx = int(x_idx.max()) + 1
        ny = int(y_idx.max()) + 1

        if n > np.iinfo(np.uint32).max:
            raise ValueError("uint32 histogram counts would overflow")

        mf = self.mf

        x_dev = cl.Buffer(
            self.context,
            mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=x_idx,
        )
        y_dev = cl.Buffer(
            self.context,
            mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=y_idx,
        )

        joint_dev = cl.Buffer(
            self.context,
            mf.READ_WRITE,
            size=nx * ny * np.dtype(np.uint32).itemsize,
        )

        # Zero the joint histogram.
        cl.enqueue_fill_buffer(
            self.queue,
            joint_dev,
            np.uint32(0),
            0,
            nx * ny * np.dtype(np.uint32).itemsize,
        )

        # Build the joint histogram.
        evt_hist = self.histogram_joint(
            self.queue,
            (n,),
            None,
            x_dev,
            y_dev,
            joint_dev,
            np.int32(ny),
            np.int32(n),
        )

        # Compute marginal counts on the GPU.
        # Each marginal can be obtained efficiently by copying the compact
        # joint table back, but this version computes them on the CPU.
        #
        # For large nx*ny, replace this with separate GPU reduction kernels.
        joint_counts = np.empty((nx, ny), dtype=np.uint32)
        cl.enqueue_copy(self.queue, joint_counts, joint_dev, wait_for=[evt_hist])

        x_counts = np.ascontiguousarray(joint_counts.sum(axis=1), dtype=np.uint32)
        y_counts = np.ascontiguousarray(joint_counts.sum(axis=0), dtype=np.uint32)

        x_counts_dev = cl.Buffer(
            self.context,
            mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=x_counts,
        )
        y_counts_dev = cl.Buffer(
            self.context,
            mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=y_counts,
        )

        result_dev = cl.Buffer(
            self.context,
            mf.WRITE_ONLY,
            size=n * np.dtype(np.float32).itemsize,
        )

        evt_mi = self.local_mi(
            self.queue,
            (n,),
            None,
            x_dev,
            y_dev,
            joint_dev,
            x_counts_dev,
            y_counts_dev,
            result_dev,
            np.int32(ny),
            np.float32(n),
            np.int32(n),
            wait_for=[evt_hist],
        )

        result = np.empty(n, dtype=np.float32)
        cl.enqueue_copy(self.queue, result, result_dev, wait_for=[evt_mi]).wait()

        # Match the original function's float64 output.
        return result.astype(np.float64).reshape(orig_shape)

    def calculateAverageMI(self):
        """Calculate average mutual information for discrete data.
                This function can not be called directly! You need to call .estimate(X,Y)
                with estimator setting local_values = False."""

        if not hasattr(self, 'flag'):
            raise RuntimeError('calculateAverageMI can not be called directly! You need to call .estimate(X,Y) '
                               'with estimator setting local_values = False!')

        var1, var2 = self.get_data()
        X = np.asarray(var1)
        Y = np.asarray(var2)
        n = X.size

        if n == 0:
            return 0.0

        x = X.ravel()
        y = Y.ravel()

        # CPU relabeling. Prefer caching these labels if this method
        # is called repeatedly for the same categorical data.
        _, x_idx = np.unique(x, return_inverse=True)
        _, y_idx = np.unique(y, return_inverse=True)

        x_idx = np.ascontiguousarray(x_idx, dtype=np.int32)
        y_idx = np.ascontiguousarray(y_idx, dtype=np.int32)

        nx = int(x_idx.max()) + 1
        ny = int(y_idx.max()) + 1

        if n > np.iinfo(np.uint32).max:
            raise ValueError("uint32 histogram counts would overflow")

        mf = cl.mem_flags
        queue = self.queue

        x_dev = cl.Buffer(
            self.context,
            mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=x_idx,
        )

        y_dev = cl.Buffer(
            self.context,
            mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=y_idx,
        )

        joint_dev = cl.Buffer(
            self.context,
            mf.READ_WRITE,
            size=nx * ny * np.dtype(np.uint32).itemsize,
        )

        cl.enqueue_fill_buffer(
            queue,
            joint_dev,
            np.uint32(0),
            0,
            nx * ny * np.dtype(np.uint32).itemsize,
        )

        evt = self.histogram_joint(
            queue,
            (n,),
            None,
            x_dev,
            y_dev,
            joint_dev,
            np.int32(ny),
            np.int32(n),
        )

        # The table is only nx*ny elements, usually much smaller than n.
        joint = np.empty((nx, ny), dtype=np.uint32)
        cl.enqueue_copy(queue, joint, joint_dev, wait_for=[evt]).wait()

        px = np.ascontiguousarray(joint.sum(axis=1), dtype=np.uint32)
        py = np.ascontiguousarray(joint.sum(axis=0), dtype=np.uint32)

        px_dev = cl.Buffer(
            self.context,
            mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=px,
        )

        py_dev = cl.Buffer(
            self.context,
            mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=py,
        )

        terms_dev = cla.empty(
            self.queue,
            (nx * ny,),
            dtype=np.float32,
        )

        evt = self.mi_terms(
            queue,
            (nx * ny,),
            None,
            joint_dev,
            px_dev,
            py_dev,
            terms_dev.data,
            np.int32(nx),
            np.int32(ny),
            np.float32(n),
            wait_for=[evt],
        )

        mi_device = self.reduce(
            terms_dev,
            queue=self.queue,
        )
        mi = float(mi_device.get())

        return mi

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
            self.set_data("MI", var1, var2)
            mi = self.calculateLocalMI()
            self.actualValue = np.mean(mi)
        else:
            var1 = self._ensure_two_dim_input(var1)
            var2 = self._ensure_two_dim_input(var2)
            self.set_data("MI", var1, var2)
            mi = self.calculateAverageMI()
            self.actualValue = mi

        self.remove_data()

        return mi

    def computeSignificance(self):
        C = ChiSquareMeasurementDistribution()
        C.ChiSquareMeasurementDistribution(self.actualValue,
                                           self.n_samples,
                                           (self.settings['alph1'] - 1) * (self.settings['alph2'] - 1),
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


class OpenCLDiscreteCMI(OpenCLDiscrete):
    """Calculate CMI with OpenCL implementation for discrete variables.

    Calculate the conditional mutual information between two variables given
    the third.

    Results are returned in bits.

    implemented in idtxl by Michael Lindner, 2026

    Args:
        settings : dict [optional]
            sets estimation parameters:

            - gpuid : int [optional] - device ID used for estimation
              (if more than one device is available on the current
              platform) (default=0)
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


    def calculateLocalCMI(self):
        """Local conditional mutual information for discrete data.

                Assumes _encode_multidim_states returns:
                    codes: integer array of shape (n,)
                    nstates: number of encoded states

                This function can not be called directly! You need to call .estimate(X,Y,Z)
                with estimator setting local_values = True."""

        if not hasattr(self, 'flag'):
            raise RuntimeError('calculateLocalCMI can not be called directly! You need to call .estimate(X,Y,Z) '
                               'with estimator setting local_values = True!')

        var1, var2, conditional = self.get_data()

        x, nx = self.encode_multidim_states(var1)
        y, ny = self.encode_multidim_states(var2)
        z, nz = self.encode_multidim_states(conditional)

        if not (x.size == y.size == z.size):
            raise ValueError("All variables must have the same number of samples")

        n = x.size

        if n == 0:
            return np.empty(0, dtype=np.float64)

        if nx * ny * nz >= 2 ** 32:
            raise ValueError(
                "The mixed-radix state space exceeds uint32 addressing capacity"
            )

        if n >= 2 ** 31:
            raise ValueError(
                "This implementation uses int32 histogram counters"
            )

        # The kernels use long for state values and ulong for key arithmetic.
        x = np.ascontiguousarray(x, dtype=np.int64)
        y = np.ascontiguousarray(y, dtype=np.int64)
        z = np.ascontiguousarray(z, dtype=np.int64)

        mf = self.mf

        # allocate memory on GPU and copy data
        bx = cl.Buffer(
            self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x
        )
        by = cl.Buffer(
            self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y
        )
        bz = cl.Buffer(
            self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z
        )

        # These are exactly the maximum possible mixed-radix keys.
        nxyz = nx * ny * nz
        nxz = nx * nz
        nyz = ny * nz

        c_xyz = cl.Buffer(self.context, mf.READ_WRITE, nxyz * np.int32().nbytes)
        c_xz = cl.Buffer(self.context, mf.READ_WRITE, nxz * np.int32().nbytes)
        c_yz = cl.Buffer(self.context, mf.READ_WRITE, nyz * np.int32().nbytes)
        c_z = cl.Buffer(self.context, mf.READ_WRITE, nz * np.int32().nbytes)

        zero = np.int32(0)

        for buf, size in (
                (c_xyz, nxyz),
                (c_xz, nxz),
                (c_yz, nyz),
                (c_z, nz),
        ):
            cl.enqueue_fill_buffer(
                self.queue,
                buf,
                zero,
                0,
                size * np.int32().nbytes
            )

        local = cl.Buffer(self.context, mf.WRITE_ONLY, n * np.float64().nbytes)

        # A multiple of the device's preferred work-group size is usually useful.
        device = self.queue.device
        preferred = device.max_work_group_size
        local_size = min(256, preferred)
        global_size = ((n + local_size - 1) // local_size) * local_size

        self.count_cmi(
            self.queue,
            (global_size,),
            (local_size,),
            bx, by, bz,
            c_xyz, c_xz, c_yz, c_z,
            np.int64(ny),
            np.int64(nz),
            np.uint64(n)
        )

        self.local_cmi(
            self.queue,
            (global_size,),
            (local_size,),
            bx, by, bz,
            c_xyz, c_xz, c_yz, c_z,
            local,
            np.int64(ny),
            np.int64(nz),
            np.uint64(n)
        )

        result = np.empty(n, dtype=np.float64)
        cl.enqueue_copy(self.queue, result, local).wait()

        return result

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
            # if (self.est_mi is None):
            self.est_mi = OpenCLDiscreteMI(self.settings)
            return self.est_mi.estimate(var1, var2)
        else:
            assert (conditional.size != 0), 'Conditional Array is empty.'

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

        self.set_data("CMI", var1, var2, conditional)

        cmi = self.calculateLocalCMI()
        self.actualValue = np.mean(cmi)

        if not self.settings['local_values']:
            cmi = np.mean(cmi)

        self.remove_data()

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
            mi = OpenCLDiscreteMI(self.settings)
            mi.estimate(var1, var2)
            return mi.computeSignificance()
        else:
            self.estimate(var1, var2, conditional)
            return self.computeSignificance()


class OpenCLDiscreteAIS(OpenCLDiscrete):
    """Calculate AIS with OpenCL discrete-variable implementation.

    Calculate the active information storage (AIS) for one process.

    Results are returned in bits.

    implemented in idtxl by Michael Lindner, 2026

    Args:
        settings : dict
            set estimator parameters:

            - gpuid : int [optional] - device ID used for estimation
              (if more than one device is available on the current
              platform) (default=0)
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

        self.set_data("AIS", process_past, process_current)

        if self.settings['local_values']:
            ais = OpenCLDiscreteMI.calculateLocalMI(self)
            ais = np.hstack([np.zeros(self.settings['history']), ais[:, 0]])
            self.actualValue = np.mean(ais)
        else:
            ais = OpenCLDiscreteMI.calculateAverageMI(self)
            self.actualValue = ais

        self.remove_data()

        return ais

    def computeSignificance(self):
        C = ChiSquareMeasurementDistribution()
        C.ChiSquareMeasurementDistribution(self.actualValue,
                                           self.n_samples,
                                           (self.settings['alph1'] - 1) * (
                                                       np.power(self.settings['alph1'], self.settings['history']) - 1),
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


class OpenCLDiscreteTE(OpenCLDiscrete):
    """Calculate TE with OpenCL implementation for discrete variables.

    Calculate the transfer entropy between two time series processes.

    Results are returned in bits.

    implemented in idtxl by Michael Lindner, 2026

    Args:
        settings : dict
            sets estimation parameters:

             - gpuid : int [optional] - device ID used for estimation
              (if more than one device is available on the current
              platform) (default=0)
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

        self.set_data("TE", source_past, target_current, target_past)

        te = OpenCLDiscreteCMI.calculateLocalCMI(self)
        self.actualValue = np.mean(te)

        if self.settings['local_values']:
            # correction to compare with JidtGaussianTE results
            te = np.hstack([np.zeros(startFirstPoint + 1), te])
        else:
            te = np.mean(te)

        self.remove_data()

        return te

    def computeSignificance(self):
        C = ChiSquareMeasurementDistribution()
        C.ChiSquareMeasurementDistribution(self.actualValue,
                                           self.n_samples,
                                           (np.power(self.settings['alph1'], self.settings['history_source']) - 1) * (
                                                       self.settings['alph1'] - 1) * np.power(self.settings['alph2'],
                                                                                              self.settings[
                                                                                                  'history_target']),
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
