"""Test Python estimators.

This module provides unit tests for Python estimators.

"""

import numpy as np
import random as rn
import time
import pytest
from scipy.special import digamma
from idtxl.estimators_python import (PythonKraskovMI, PythonKraskovCMI, 
                                PythonKraskovAIS, PythonKraskovTE, PythonKraskovCTE,
                                PythonGaussianMI, PythonGaussianCMI,
                                PythonGaussianAIS, PythonGaussianTE, PythonGaussianCTE,
                                PythonDiscreteMI, PythonDiscreteCMI,
                                PythonDiscreteAIS, PythonDiscreteTE,)

from generate_test_data import (_get_gauss_data,
                                _get_ar_data,
                                _get_mem_binary_data,
                                )

from idtxl.idtxl_utils import calculate_mi
import idtxl.idtxl_exceptions as ex

SEED = 0


def _assert_result(results, expected_res, estimator, measure, tol=0.05):
    # Compare estimates with analytic results and print output.
    print(f"{estimator} - {measure} - result: {results:.4f} nats expected to be close to {expected_res:.4f} nats.  - {np.isclose(results, expected_res, atol=tol)}")
    assert np.isclose(results, expected_res, atol=tol), (
        '{0} calculation failed (error larger than {1}).'.format(measure, tol))

def _compare_result(res1, res2, estimator1, estimator2, measure, tol=0.05):
    # Compare estimates with each other and print output.
    
    print(f"{estimator1} vs. {estimator2} - {measure} result: {res1:.4f} nats vs. {res2:.4f} nats.  - {np.isclose(res1, res2, atol=tol)}")
    assert np.isclose(res1, res2, atol=tol), (
                        '{0} calculation failed (error larger than '
                        '{1}).'.format(measure, tol))


def test_mi_gauss_data():
    """Test MI estimators on correlated Gauss data.

    Note that the calculation is based on a random variable (because the
    generated data is a set of random variables) - the result will be of the
    order of what we expect, but not exactly equal to it in fact, there will
    be a large variance around it.
    """
    expected_mi, source1, source2, target = _get_gauss_data(seed=SEED)

    # Test Kraskov
    mi_estimator = PythonKraskovMI(settings={"noise_level": 0})
    mi_cor = mi_estimator.estimate(source1, target)
    mi_uncor = mi_estimator.estimate(source2, target)
    _assert_result(mi_cor, expected_mi, 'PythonKraskovMI', 'MI (correlated)  ')
    _assert_result(mi_uncor, 0, 'PythonKraskovMI', 'MI (uncorrelated)')

    # Test Gaussian
    mi_estimator = PythonGaussianMI(settings={"noise_level": 0})
    mi_cor = mi_estimator.estimate(source1, target)
    mi_uncor = mi_estimator.estimate(source2, target)
    _assert_result(mi_cor, expected_mi, 'PythonGaussianMI', 'MI (correlated)  ')
    _assert_result(mi_uncor, 0, 'PythonGaussianMI', 'MI (uncorrelated)')

    # Test Discrete
    settings = {'discretise_method': 'equal', 'n_discrete_bins': 5}
    mi_estimator = PythonDiscreteMI(settings=settings)
    mi_cor = mi_estimator.estimate(source1, target)
    mi_uncor = mi_estimator.estimate(source2, target)
    _assert_result(mi_cor, expected_mi, 'PythonDiscreteMI', 'MI (correlated)  ', 0.08)  # More variability here
    _assert_result(mi_uncor, 0, 'PythonDiscreteMI', 'MI (uncorrelated)', 0.08)  # More variability here

def test_cmi_gauss_data_no_cond():
    """Test estimators on correlated Gauss data without a conditional.

    The estimators should return the MI if no conditional variable is
    provided.

    Note that the calculation is based on a random variable (because the
    generated data is a set of random variables) - the result will be of the
    order of what we expect, but not exactly equal to it in fact, there will
    be a large variance around it.
    """
    expected_mi, source1, source2, target = _get_gauss_data(seed=SEED)

    # Test Kraskov
    mi_estimator = PythonKraskovCMI(settings={"noise_level": 0})
    mi_cor = mi_estimator.estimate(source1, target)
    mi_uncor = mi_estimator.estimate(source2, target)
    _assert_result(mi_cor, expected_mi, 'PythonKraskovCMI', 'CMI (no cond.)        ')
    _assert_result(mi_uncor, 0, 'PythonKraskovCMI', 'CMI (uncorr., no cond.)')

    # Test Gaussian
    mi_estimator = PythonGaussianCMI(settings={"noise_level": 0})
    mi_cor = mi_estimator.estimate(source1, target)
    mi_uncor = mi_estimator.estimate(source2, target)
    _assert_result(mi_cor, expected_mi, 'PythonGaussianCMI', 'CMI (no cond.)        ')
    _assert_result(mi_uncor, 0, 'PythonGaussianCMI', 'CMI (uncorr., no cond.)')

    # Test Discrete
    settings = {'discretise_method': 'equal', 'n_discrete_bins': 5}
    mi_estimator = PythonDiscreteCMI(settings=settings)
    mi_cor = mi_estimator.estimate(source1, target)
    mi_uncor = mi_estimator.estimate(source2, target)
    _assert_result(mi_cor, expected_mi, 'PythonDiscreteCMI', 'CMI (no cond.)        ', 0.08) # More variability here
    _assert_result(mi_uncor, 0, 'PythonDiscreteCMI', 'CMI (uncorr., no cond.)', 0.08) # More variability here

def test_cmi_gauss_data():
    """Test CMI estimation on two sets of Gaussian random data.

    The first test is on uncorrelated conditional, the second on uncorrelated
    source.

    Note that the calculation is based on a random variable (because the
    generated data is a set of random variables) - the result will be of the
    order of what we expect, but not exactly equal to it in fact, there will
    be a large variance around it.
    """
    expected_mi, source1, source2, target = _get_gauss_data(seed=SEED)

    # Test Kraskov
    mi_estimator = PythonKraskovCMI(settings={"noise_level": 0})
    mi_cor = mi_estimator.estimate(source1, target, source2)
    mi_uncor = mi_estimator.estimate(source2, target, source1)
    _assert_result(mi_cor, expected_mi, 'PythonKraskovCMI', 'CMI (uncorr. cond)  ')
    _assert_result(mi_uncor, 0, 'PythonKraskovCMI', 'CMI (uncorr. source)')

    # Test Gaussian
    mi_estimator = PythonGaussianCMI(settings={"noise_level": 0})
    mi_cor = mi_estimator.estimate(source1, target, source2)
    mi_uncor = mi_estimator.estimate(source2, target, source1)
    _assert_result(mi_cor, expected_mi, 'PythonGaussianCMI', 'CMI (uncorr. cond)  ')
    _assert_result(mi_uncor, 0, 'PythonGaussianCMI', 'CMI (uncorr. source)')

    # Test Discrete
    settings = {'discretise_method': 'equal', 'n_discrete_bins': 5}
    mi_estimator = PythonDiscreteCMI(settings=settings)
    mi_cor = mi_estimator.estimate(source1, target, source2)
    mi_uncor = mi_estimator.estimate(source2, target, source1)
    _assert_result(mi_cor, expected_mi, 'PythonDiscreteCMI', 'CMI (uncorr. cond)  ', 0.08) # More variability here
    _assert_result(mi_uncor, 0, 'PythonDiscreteCMI', 'CMI (uncorr. source)', 0.08) # More variability here

def test_ais_gauss_data():
    """Test AIS estimation on an autoregressive process.

    The first test is on correlated variables, the second on uncorrelated
    variables.

    Note that the calculation is based on a random variable (because the
    generated data is a set of random variables) - the result will be of the
    order of what we expect, but not exactly equal to it in fact, there will
    be a large variance around it.
    """
    source1, source2 = _get_ar_data(seed=SEED)

    settings = {'discretise_method': 'equal',
                'n_discrete_bins': 4,
                'history': 2,
                "noise_level": 0,}

    # Test Kraskov
    mi_estimator = PythonKraskovAIS(settings=settings)
    mi_cor_k = mi_estimator.estimate(source1)
    mi_uncor = mi_estimator.estimate(source2)
    _assert_result(mi_uncor, 0, 'PythonKraskovAIS', 'AIS (uncorr.)')

    # Test Gaussian
    mi_estimator = PythonGaussianAIS(settings=settings)
    mi_cor_g = mi_estimator.estimate(source1)
    mi_uncor = mi_estimator.estimate(source2)
    _assert_result(mi_uncor, 0, 'PythonGaussianAIS', 'AIS (uncorr.)')

    # TODO is this a meaningful test?
    # # Test Discrete
    # mi_estimator = PythonDiscreteAIS(settings=settings)
    # mi_cor_d = mi_estimator.estimate(source1)
    # mi_uncor = mi_estimator.estimate(source2)
    # _assert_result(mi_uncor, 0, 'PythonDiscreteAIS', 'AIS (uncorr.)', tol=0.5)

    # Compare results for AR process.
    _compare_result(mi_cor_k, mi_cor_g, 'PythonKraskovAIS', 'PythonGaussianAIS',
                    'AIS (AR process)')
    # _compare_result(mi_cor_k, mi_cor_d, 'PythonKraskovAIS', 'PythonDiscreteAIS',
    #                 'AIS (AR process)')

def test_te_gauss_data():
    """Test TE estimation on two sets of Gaussian random data.

    The first test is on correlated variables, the second on uncorrelated
    variables.

    Note that the calculation is based on a random variable (because the
    generated data is a set of random variables) - the result will be of the
    order of what we expect, but not exactly equal to it in fact, there will
    be a large variance around it.
    """
    expected_mi, source1, source2, target = _get_gauss_data(expand=False, seed=SEED)
    # add delay of one sample
    source1 = source1[1:]
    source2 = source2[1:]
    target = target[:-1]
    settings = {'discretise_method': 'equal',
                'n_discrete_bins': 4,
                'history_target': 1,
                'noise_level': 0,}
    # Test Kraskov
    mi_estimator = PythonKraskovTE(settings=settings)
    mi_cor = mi_estimator.estimate(source1, target)
    mi_uncor = mi_estimator.estimate(source2, target)
    _assert_result(mi_cor, expected_mi, 'PythonKraskovTE', 'TE (corr.)  ')
    _assert_result(mi_uncor, 0, 'PythonKraskovTE', 'TE (uncorr.)')

    # Test Gaussian
    mi_estimator = PythonGaussianTE(settings=settings)
    mi_cor = mi_estimator.estimate(source1, target)
    mi_uncor = mi_estimator.estimate(source2, target)
    _assert_result(mi_cor, expected_mi, 'PythonGaussianTE', 'TE (corr.)  ')
    _assert_result(mi_uncor, 0, 'PythonGaussianTE', 'TE (uncorr.)')

    # Test Discrete
    mi_estimator = PythonDiscreteTE(settings=settings)
    mi_cor = mi_estimator.estimate(source1, target)
    mi_uncor = mi_estimator.estimate(source2, target)
    _assert_result(mi_cor, expected_mi, 'PythonDiscreteTE', 'TE (corr.)  ', 0.08) # More variability here
    _assert_result(mi_uncor, 0, 'PythonDiscreteTE', 'TE (uncorr.)', 0.08) # More variability here

def test_cte_gauss_data_no_cond():
    """Test estimators on correlated Gauss data without a conditional.

    The estimators should return the TE if no conditional variable is
    provided.

    Note that the calculation is based on a random variable (because the
    generated data is a set of random variables) - the result will be of the
    order of what we expect, but not exactly equal to it in fact, there will
    be a large variance around it.
    """
    expected_mi, source1, source2, target = _get_gauss_data(seed=SEED)
    source1 = source1[1:]
    source2 = source2[1:]
    target = target[:-1]
    
    # Test Kraskov
    mi_estimator = PythonKraskovCTE(settings={'history_target': 1, "noise_level": 0})
    mi_cor = mi_estimator.estimate(source1, target)
    mi_uncor = mi_estimator.estimate(source2, target)
    _assert_result(mi_cor, expected_mi, 'PythonKraskovCTE', 'CTE (no cond.)        ')
    _assert_result(mi_uncor, 0, 'PythonKraskovCTE', 'CTE (uncorr., no cond.)')

    # Test Gaussian
    mi_estimator = PythonGaussianCTE(settings={'history_target': 1, "noise_level": 0})
    mi_cor = mi_estimator.estimate(source1, target)
    mi_uncor = mi_estimator.estimate(source2, target)
    _assert_result(mi_cor, expected_mi, 'PythonGaussianCTE', 'CTE (no cond.)        ')
    _assert_result(mi_uncor, 0, 'PythonGaussianCTE', 'CTE (uncorr., no cond.)')

def test_cte_gauss_data():
    """Test CMI estimation on two sets of Gaussian random data.

    The first test is on uncorrelated conditional, the second on uncorrelated
    source.

    Note that the calculation is based on a random variable (because the
    generated data is a set of random variables) - the result will be of the
    order of what we expect, but not exactly equal to it in fact, there will
    be a large variance around it.
    """
    expected_mi, source1, source2, target = _get_gauss_data(seed=SEED)
    source1 = source1[1:]
    source2 = source2[1:]
    target = target[:-1]
    
    # Test Kraskov
    mi_estimator = PythonKraskovCTE(settings={'history_target': 1, "noise_level": 0})
    mi_cor = mi_estimator.estimate(source1, target, source2)
    mi_uncor = mi_estimator.estimate(source2, target, source1)
    _assert_result(mi_cor, expected_mi, 'PythonKraskovCTE', 'CTE (uncorr. cond)  ')
    _assert_result(mi_uncor, 0, 'PythonKraskovCTE', 'CTE (uncorr. source)')

    # Test Gaussian
    mi_estimator = PythonGaussianCTE(settings={'history_target': 1, "noise_level": 0})
    mi_cor = mi_estimator.estimate(source1, target, source2)
    mi_uncor = mi_estimator.estimate(source2, target, source1)
    _assert_result(mi_cor, expected_mi, 'PythonGaussianCTE', 'CTE (uncorr. cond)  ')
    _assert_result(mi_uncor, 0, 'PythonGaussianCTE', 'CTE (uncorr. source)')

def test_one_two_dim_input_kraskov():
    """Test one- and two-dimensional input for Kraskov estimators."""
    expected_mi, src_one, s, target_one = _get_gauss_data(
        expand=False, seed=SEED)
    src_two = np.expand_dims(src_one, axis=1)
    target_two = np.expand_dims(target_one, axis=1)
    ar_src_one, s = _get_ar_data(expand=False, seed=SEED)
    ar_src_two = np.expand_dims(ar_src_one, axis=1)

    # MI
    mi_estimator = PythonKraskovMI(settings={"noise_level": 0})
    mi_cor_one = mi_estimator.estimate(src_one, target_one)
    _assert_result(mi_cor_one, expected_mi, 'PythonKraskovMI', 'MI')
    mi_cor_two = mi_estimator.estimate(src_two, target_two)
    _assert_result(mi_cor_two, expected_mi, 'PythonKraskovMI', 'MI')
    _compare_result(mi_cor_one, mi_cor_two,
                    'PythonKraskovMI one dim', 'PythonKraskovMI two dim', 'MI')
    # CMI
    cmi_estimator = PythonKraskovCMI(settings={"noise_level": 0})
    mi_cor_one = cmi_estimator.estimate(src_one, target_one)
    _assert_result(mi_cor_one, expected_mi, 'PythonKraskovCMI', 'CMI')
    mi_cor_two = cmi_estimator.estimate(src_two, target_two)
    _assert_result(mi_cor_two, expected_mi, 'PythonKraskovCMI', 'CMI')
    _compare_result(mi_cor_one, mi_cor_two,
                    'PythonKraskovCMI one dim', 'PythonKraskovCMI two dim', 'CMI')
    # TE
    te_estimator = PythonKraskovTE(settings={'history_target': 1, "noise_level": 0})
    mi_cor_one = te_estimator.estimate(src_one[1:], target_one[:-1])
    _assert_result(mi_cor_one, expected_mi, 'PythonKraskovTE', 'TE')
    mi_cor_two = te_estimator.estimate(src_two[1:], target_two[:-1])
    _assert_result(mi_cor_two, expected_mi, 'PythonKraskovTE', 'TE')
    _compare_result(mi_cor_one, mi_cor_two,
                    'PythonKraskovTE one dim', 'PythonKraskovTE two dim', 'TE')
    # AIS
    ais_estimator = PythonKraskovAIS(settings={'history': 2, "noise_level": 0})
    mi_cor_one = ais_estimator.estimate(ar_src_one)
    mi_cor_two = ais_estimator.estimate(ar_src_two)
    _compare_result(mi_cor_one, mi_cor_two,
                    'PythonKraskovAIS one dim', 'PythonKraskovAIS two dim',
                    'AIS (AR process)')

    # CTE
    cmi_estimator = PythonKraskovCTE(settings={'history_target': 1, "noise_level": 0})
    mi_cor_one = cmi_estimator.estimate(src_one[1:], target_one[:-1])
    _assert_result(mi_cor_one, expected_mi, 'PythonKraskovCTE', 'CTE')
    mi_cor_two = cmi_estimator.estimate(src_two[1:], target_two[:-1])
    _assert_result(mi_cor_two, expected_mi, 'PythonKraskovCTE', 'CTE')
    _compare_result(mi_cor_one, mi_cor_two,
                    'PythonKraskovCTE one dim', 'PythonKraskovCTE two dim', 'CTE')

def test_one_two_dim_input_gaussian():
    """Test one- and two-dimensional input for Gaussian estimators."""
    expected_mi, src_one, s, target_one = _get_gauss_data(
        expand=False, seed=SEED)
    src_two = np.expand_dims(src_one, axis=1)
    target_two = np.expand_dims(target_one, axis=1)
    ar_src_one, s = _get_ar_data(expand=False, seed=SEED)
    ar_src_two = np.expand_dims(ar_src_one, axis=1)

    # MI
    mi_estimator = PythonGaussianMI(settings={})
    mi_cor_one = mi_estimator.estimate(src_one, target_one)
    _assert_result(mi_cor_one, expected_mi, 'PythonGaussianMI', 'MI')
    mi_cor_two = mi_estimator.estimate(src_two, target_two)
    _assert_result(mi_cor_two, expected_mi, 'PythonGaussianMI', 'MI')
    _compare_result(mi_cor_one, mi_cor_two,
                    'PythonGaussianMI one dim', 'PythonGaussianMI two dim', 'MI')
    # CMI
    cmi_estimator = PythonGaussianCMI(settings={})
    mi_cor_one = cmi_estimator.estimate(src_one, target_one)
    _assert_result(mi_cor_one, expected_mi, 'PythonGaussianCMI', 'CMI')
    mi_cor_two = cmi_estimator.estimate(src_two, target_two)
    _assert_result(mi_cor_two, expected_mi, 'PythonGaussianCMI', 'CMI')
    _compare_result(mi_cor_one, mi_cor_two,
                    'PythonGaussianCMI one dim', 'PythonGaussianCMI two dim', 'CMI')
    # TE
    te_estimator = PythonGaussianTE(settings={'history_target': 1})
    mi_cor_one = te_estimator.estimate(src_one[1:], target_one[:-1])
    _assert_result(mi_cor_one, expected_mi, 'PythonGaussianTE', 'TE')
    mi_cor_two = te_estimator.estimate(src_two[1:], target_two[:-1])
    _assert_result(mi_cor_two, expected_mi, 'PythonGaussianTE', 'TE')
    _compare_result(mi_cor_one, mi_cor_two,
                    'PythonGaussianTE one dim', 'PythonGaussianTE two dim', 'TE')
    # AIS
    ais_estimator = PythonGaussianAIS(settings={'history': 2})
    mi_cor_one = ais_estimator.estimate(ar_src_one)
    mi_cor_two = ais_estimator.estimate(ar_src_two)
    _compare_result(mi_cor_one, mi_cor_two,
                    'PythonGaussianAIS one dim', 'PythonGaussianAIS two dim',
                    'AIS (AR process)')

    # CTE
    cmi_estimator = PythonGaussianCTE(settings={'history_target': 1})
    mi_cor_one = cmi_estimator.estimate(src_one[1:], target_one[:-1])
    _assert_result(mi_cor_one, expected_mi, 'PythonGaussianCTE', 'CTE')
    mi_cor_two = cmi_estimator.estimate(src_two[1:], target_two[:-1])
    _assert_result(mi_cor_two, expected_mi, 'PythonGaussianCTE', 'CTE')
    _compare_result(mi_cor_one, mi_cor_two,
                    'PythonGaussianCTE one dim', 'PythonGaussianCTE two dim', 'CTE')

def test_one_two_dim_input_discrete():
    """Test one- and two-dimensional input for discrete estimators."""
    expected_mi, src_one, s, target_one = _get_gauss_data(
        expand=False, seed=SEED)
    src_two = np.expand_dims(src_one, axis=1)
    target_two = np.expand_dims(target_one, axis=1)
    ar_src_one, s = _get_ar_data(expand=False, seed=SEED)
    ar_src_two = np.expand_dims(ar_src_one, axis=1)

    settings = {'discretise_method': 'equal',
                'n_discrete_bins': 4,
                'history_target': 1,
                'history': 2}
    # MI
    mi_estimator = PythonDiscreteMI(settings=settings)
    mi_cor_one = mi_estimator.estimate(src_one, target_one)
    _assert_result(mi_cor_one, expected_mi, 'PythonDiscreteMI', 'MI', 0.08) # More variability here
    mi_cor_two = mi_estimator.estimate(src_two, target_two)
    _assert_result(mi_cor_two, expected_mi, 'PythonDiscreteMI', 'MI', 0.08) # More variability here
    _compare_result(mi_cor_one, mi_cor_two,
                    'PythonDiscreteMI one dim', 'PythonDiscreteMI two dim', 'MI')
    # CMI
    cmi_estimator = PythonDiscreteCMI(settings=settings)
    mi_cor_one = cmi_estimator.estimate(src_one, target_one)
    _assert_result(mi_cor_one, expected_mi, 'PythonDiscreteCMI', 'CMI', 0.08) # More variability here
    mi_cor_two = cmi_estimator.estimate(src_two, target_two)
    _assert_result(mi_cor_two, expected_mi, 'PythonDiscreteCMI', 'CMI', 0.08) # More variability here
    _compare_result(mi_cor_one, mi_cor_two,
                    'PythonDiscreteMI one dim', 'PythonDiscreteMI two dim', 'CMI')
    # TE
    te_estimator = PythonDiscreteTE(settings=settings)
    mi_cor_one = te_estimator.estimate(src_one[1:], target_one[:-1])
    _assert_result(mi_cor_one, expected_mi, 'PythonDiscreteTE', 'TE', 0.08) # More variability here
    mi_cor_two = te_estimator.estimate(src_two[1:], target_two[:-1])
    _assert_result(mi_cor_two, expected_mi, 'PythonDiscreteTE', 'TE', 0.08) # More variability here
    _compare_result(mi_cor_one, mi_cor_two,
                    'PythonDiscreteMI one dim', 'PythonDiscreteMI two dim', 'TE')
    # AIS
    ais_estimator = PythonDiscreteAIS(settings=settings)
    mi_cor_one = ais_estimator.estimate(ar_src_one)
    mi_cor_two = ais_estimator.estimate(ar_src_two)
    _compare_result(mi_cor_one, mi_cor_two,
                    'PythonDiscreteAIS one dim', 'PythonDiscreteAIS two dim',
                    'AIS (AR process)')

def test_local_values():
    """Test estimation of local values and their return type."""
    expected_mi, source, s, target = _get_gauss_data(expand=False, seed=SEED)
    ar_proc, s = _get_ar_data(expand=False, seed=SEED)

    settings = {'discretise_method': 'equal',
                'n_discrete_bins': 4,
                'history_target': 1,
                'history': 2,
                'noise_level': 0,
                'local_values': True}

    # MI - Discrete
    mi_estimator = PythonDiscreteMI(settings=settings)
    mi = mi_estimator.estimate(source, target)
    _assert_result(np.mean(mi), expected_mi, 'PythonDiscreteMI', 'MI local', 0.08) # More variability here
    assert type(mi) is np.ndarray, 'Local values are not a numpy array.'

    # MI - Gaussian
    mi_estimator = PythonGaussianMI(settings=settings)
    mi = mi_estimator.estimate(source, target)
    _assert_result(np.mean(mi), expected_mi, 'PythonGaussianMI', 'MI local')
    assert type(mi) is np.ndarray, 'Local values are not a numpy array.'

    # MI - Kraskov
    mi_estimator = PythonKraskovMI(settings=settings)
    mi = mi_estimator.estimate(source, target)
    _assert_result(np.mean(mi), expected_mi, 'PythonKraskovMI', 'MI local')
    assert type(mi) is np.ndarray, 'Local values are not a numpy array.'

    # CMI - Discrete
    cmi_estimator = PythonDiscreteCMI(settings=settings)
    mi = cmi_estimator.estimate(source, target)
    _assert_result(np.mean(mi), expected_mi, 'PythonDiscreteCMI', 'CMI local', 0.08) # More variability here
    assert type(mi) is np.ndarray, 'Local values are not a numpy array.'

    # MI - Gaussian
    mi_estimator = PythonGaussianCMI(settings=settings)
    mi = mi_estimator.estimate(source, target)
    _assert_result(np.mean(mi), expected_mi, 'PythonGaussianCMI', 'CMI local')
    assert type(mi) is np.ndarray, 'Local values are not a numpy array.'

    # MI - Kraskov
    mi_estimator = PythonKraskovCMI(settings=settings)
    mi = mi_estimator.estimate(source, target)
    _assert_result(np.mean(mi), expected_mi, 'PythonKraskovCMI', 'CMI local')
    assert type(mi) is np.ndarray, 'Local values are not a numpy array.'

    # TE - Discrete
    te_estimator = PythonDiscreteTE(settings=settings)
    mi = te_estimator.estimate(source[1:], target[:-1])
    _assert_result(np.mean(mi), expected_mi, 'PythonDiscreteTE', 'TE local', 0.08) # More variability here
    assert type(mi) is np.ndarray, 'Local values are not a numpy array.'

    # TE - Gaussian
    mi_estimator = PythonGaussianTE(settings=settings)
    mi = mi_estimator.estimate(source[1:], target[:-1])
    _assert_result(np.mean(mi), expected_mi, 'PythonGaussianTE', 'TE local')
    assert type(mi) is np.ndarray, 'Local values are not a numpy array.'

    # TE - Kraskov
    mi_estimator = PythonKraskovTE(settings=settings)
    mi = mi_estimator.estimate(source[1:], target[:-1])
    _assert_result(np.mean(mi), expected_mi, 'PythonKraskovTE', 'TE local')
    assert type(mi) is np.ndarray, 'Local values are not a numpy array.'

    # AIS - Kraskov
    ais_estimator = PythonKraskovAIS(settings=settings)
    mi_k = ais_estimator.estimate(ar_proc)
    assert type(mi_k) is np.ndarray, 'Local values are not a numpy array.'

    # AIS - Discrete
    ais_estimator = PythonDiscreteAIS(settings=settings)
    mi_d = ais_estimator.estimate(ar_proc)
    assert type(mi_d) is np.ndarray, 'Local values are not a numpy array.'
    
    # AIS - Gaussian
    ais_estimator = PythonGaussianAIS(settings=settings)
    mi_g = ais_estimator.estimate(ar_proc)
    assert type(mi_g) is np.ndarray, 'Local values are not a numpy array.'
    _compare_result(np.mean(mi_k), np.mean(mi_g),
                    'PythonKraskovAIS', 'PythonGaussianAIS', 'AIS (AR process)')

def test_lagged_mi():
    """Test estimation of lagged MI."""
    n = 10000
    cov = 0.4
    source = [rn.normalvariate(0, 1) for r in range(n)]
    target = [0] + [sum(pair) for pair in zip(
                        [cov * y for y in source[0:n - 1]],
                        [(1 - cov) * y for y in
                            [rn.normalvariate(0, 1) for r in range(n - 1)]])]
    source = np.array(source)
    target = np.array(target)
    settings = {
        'discretise_method': 'equal',
        'n_discrete_bins': 4,
        'history': 1,
        'history_target': 1,
        'lag_mi': 1,
        'source_target_delay': 1,
        'noise_level': 0}

    est_te_k = PythonKraskovTE(settings)
    te_k = est_te_k.estimate(source, target)
    est_te_d = PythonDiscreteTE(settings)
    te_d = est_te_d.estimate(source, target)
    est_te_g = PythonGaussianTE(settings)
    te_g = est_te_g.estimate(source, target)
    
    est_d = PythonDiscreteMI(settings)
    mi_d = est_d.estimate(source, target)
    est_k = PythonKraskovMI(settings)
    mi_k = est_k.estimate(source, target)
    est_g = PythonGaussianMI(settings)
    mi_g = est_g.estimate(source, target)
    _compare_result(mi_d, te_d, 'PythonDiscreteMI', 'PythonDiscreteTE',
                    'lagged MI', tol=0.05)
    _compare_result(mi_k, te_k, 'PythonKraskovMI', 'PythonKraskovTE',
                    'lagged MI', tol=0.05)
    _compare_result(mi_g, te_k, 'PythonGaussianMI', 'PythonGaussianTE',
                    'lagged MI', tol=0.05)

def test_invalid_settings_input():
    """Test handling of wrong inputs for settings dictionary."""

    # Wrong input type for settings dict.
    with pytest.raises(TypeError): PythonDiscreteMI(settings=1)
    with pytest.raises(TypeError): PythonDiscreteCMI(settings=1)
    with pytest.raises(TypeError): PythonDiscreteAIS(settings=1)
    with pytest.raises(TypeError): PythonDiscreteTE(settings=1)
    with pytest.raises(TypeError): PythonGaussianMI(settings=1)
    with pytest.raises(TypeError): PythonGaussianCMI(settings=1)
    with pytest.raises(TypeError): PythonGaussianAIS(settings=1)
    with pytest.raises(TypeError): PythonGaussianTE(settings=1)
    with pytest.raises(TypeError): PythonGaussianCTE(settings=1)
    with pytest.raises(TypeError): PythonKraskovMI(settings=1)
    with pytest.raises(TypeError): PythonKraskovCMI(settings=1)
    with pytest.raises(TypeError): PythonKraskovAIS(settings=1)
    with pytest.raises(TypeError): PythonKraskovTE(settings=1)
    with pytest.raises(TypeError): PythonKraskovCTE(settings=1)

    # Test if settings dict is initialised correctly.
    e = PythonDiscreteMI()
    assert type(e.settings) is dict, 'Did not initialise settings as dictionary.'
    e = PythonDiscreteCMI()
    assert type(e.settings) is dict, 'Did not initialise settings as dictionary.'
    e = PythonGaussianMI()
    assert type(e.settings) is dict, 'Did not initialise settings as dictionary.'
    #e = PythonGaussianCMI()
    #assert type(e.settings) is dict, 'Did not initialise settings as dictionary.'
    e = PythonKraskovMI()
    assert type(e.settings) is dict, 'Did not initialise settings as dictionary.'
    e = PythonKraskovCMI()
    assert type(e.settings) is dict, 'Did not initialise settings as dictionary.'

    # History parameter missing for AIS and TE estimation.
    with pytest.raises(RuntimeError): PythonDiscreteAIS(settings={})
    with pytest.raises(RuntimeError): PythonDiscreteTE(settings={})
    with pytest.raises(RuntimeError): PythonGaussianAIS(settings={})
    with pytest.raises(RuntimeError): PythonGaussianTE(settings={})
    with pytest.raises(RuntimeError): PythonGaussianCTE(settings={})
    with pytest.raises(RuntimeError): PythonKraskovAIS(settings={})
    with pytest.raises(RuntimeError): PythonKraskovTE(settings={})
    with pytest.raises(RuntimeError): PythonKraskovCTE(settings={})

def test_invalid_history_parameters():
    """Ensure invalid history parameters raise a RuntimeError."""

    # TE: Parameters are not integers
    settings = {'history_target': 4, 'history_source': 4,
                'tau_source': 2, 'tau_target': 2.5}
    with pytest.raises(AssertionError): PythonDiscreteTE(settings=settings)
    with pytest.raises(AssertionError): PythonGaussianTE(settings=settings)
    with pytest.raises(AssertionError): PythonKraskovTE(settings=settings)
    with pytest.raises(AssertionError): PythonGaussianCTE(settings=settings)
    with pytest.raises(AssertionError): PythonKraskovCTE(settings=settings)

    settings['tau_source'] = 2.5
    settings['tau_target'] = 2
    with pytest.raises(AssertionError): PythonDiscreteTE(settings=settings)
    with pytest.raises(AssertionError): PythonGaussianTE(settings=settings)
    with pytest.raises(AssertionError): PythonKraskovTE(settings=settings)
    with pytest.raises(AssertionError): PythonGaussianCTE(settings=settings)
    with pytest.raises(AssertionError): PythonKraskovCTE(settings=settings)
    settings['history_source'] = 2.5
    settings['tau_source'] = 2
    with pytest.raises(AssertionError): PythonDiscreteTE(settings=settings)
    with pytest.raises(AssertionError): PythonGaussianTE(settings=settings)
    with pytest.raises(AssertionError): PythonKraskovTE(settings=settings)
    with pytest.raises(AssertionError): PythonGaussianCTE(settings=settings)
    with pytest.raises(AssertionError): PythonKraskovCTE(settings=settings)
    settings['history_target'] = 2.5
    settings['history_source'] = 4
    with pytest.raises(AssertionError): PythonDiscreteTE(settings=settings)
    with pytest.raises(AssertionError): PythonGaussianTE(settings=settings)
    with pytest.raises(AssertionError): PythonKraskovTE(settings=settings)
    with pytest.raises(AssertionError): PythonGaussianCTE(settings=settings)
    with pytest.raises(AssertionError): PythonKraskovCTE(settings=settings)

    # AIS: Parameters are not integers.
    settings = {'history': 4, 'tau': 2.5}
    with pytest.raises(AssertionError): PythonGaussianAIS(settings=settings)
    with pytest.raises(AssertionError): PythonKraskovAIS(settings=settings)
    settings = {'history': 4.5, 'tau': 2}
    with pytest.raises(AssertionError): PythonDiscreteAIS(settings=settings)
    with pytest.raises(AssertionError): PythonGaussianAIS(settings=settings)
    with pytest.raises(AssertionError): PythonKraskovAIS(settings=settings)

def test_insufficient_no_points():
    """Test if estimation aborts for too few data points."""
    expected_mi, source1, source2, target = _get_gauss_data(n=4)

    settings = {
        'kraskov_k': 4,
        'theiler_t': 0,
        'history': 1,
        'history_target': 1,
        'lag_mi': 1,
        'source_target_delay': 1,
        'noise_level': 0}

    # Test first settings combination with k==N
    est = PythonKraskovTE(settings)
    with pytest.raises(RuntimeError): est.estimate(source1, target)
    est = PythonKraskovMI(settings)
    with pytest.raises(RuntimeError): est.estimate(source1, target)
    est = PythonKraskovCMI(settings)
    with pytest.raises(RuntimeError): est.estimate(source1, target, target)
    est = PythonKraskovAIS(settings)
    with pytest.raises(RuntimeError): est.estimate(source1)
    est = PythonKraskovCTE(settings)
    with pytest.raises(RuntimeError): est.estimate(source1, target, target)
    
    # Test a second combination with a Theiler-correction != 0
    settings['theiler_t'] = 1
    settings['kraskov_k'] = 2

    est = PythonKraskovTE(settings)
    with pytest.raises(RuntimeError): est.estimate(source1, target)
    est = PythonKraskovMI(settings)
    with pytest.raises(RuntimeError): est.estimate(source1, target)
    est = PythonKraskovCMI(settings)
    with pytest.raises(RuntimeError): est.estimate(source1, target, target)
    est = PythonKraskovAIS(settings)
    with pytest.raises(RuntimeError): est.estimate(source1)
    est = PythonKraskovCTE(settings)
    with pytest.raises(RuntimeError): est.estimate(source1, target, target)

def test_discrete_ais():
    """Test results for discrete AIS estimation against other estimators."""

    settings = {'discretise_method': 'none',
                'alph': 2,
                'history': 2,
                'local_values': False}

    proc1, proc2 = _get_mem_binary_data()

    # Compare discrete to Gaussian estimator
    ais_estimator = PythonDiscreteAIS(settings=settings)
    mi_d = ais_estimator.estimate(proc1)

    ais_estimator = PythonGaussianAIS(settings=settings)
    mi_g = ais_estimator.estimate(proc1.astype(float))
    _compare_result(np.mean(mi_d), np.mean(mi_g), 'PythonDiscreteAIS',
                    'PythonGaussianAIS', 'AIS (AR process)', tol=0.07)

    # Compare discrete to Gaussian estimator on memoryless data
    ais_estimator = PythonDiscreteAIS(settings=settings)
    mi_d = ais_estimator.estimate(proc2)

    ais_estimator = PythonGaussianAIS(settings=settings)
    mi_g = ais_estimator.estimate(proc2.astype(float))
    _compare_result(np.mean(mi_d), np.mean(mi_g), 'PythonDiscreteAIS',
                    'PythonGaussianAIS', 'AIS (AR process, no mem.)', tol=0.05)
    _assert_result(mi_d, 0, 'PythonDiscreteAIS', 'MI (no memory)')
    _assert_result(mi_g, 0, 'PythonGaussianAIS', 'MI (no memory)')

def test_kraskov_alg1And2():
    """Test that Python Kraskov estimate changes properly when we change KSG algorithm"""
    n = 100
    source = [sum(pair) for pair in zip(
                        [y for y in range(n)],
                        [rn.normalvariate(0, 0.000001) for r in range(n)])]
    source = np.array(source)
    target = np.array(source)  # Target copies source on purpose
    # We've generated simple data 0:99, plus a little noise to ensure
    #  we only even get K nearest neighbours in each space.
    # So result should be:
    settings = {
        'lag': 0,
        'kraskov_k': 4,
        'noise_level': 0,
        'algorithm_num': 1}
    for k in range(4, 16):
        settings['kraskov_k'] = k
        settings['algorithm_num'] = 1
        est1 = PythonKraskovMI(settings)
        mi_alg1 = est1.estimate(source, target)
        # Neighbour counts n_x and n_y will be k-1 because they are
        #  *strictly* within the boundary
        expected_alg1 = digamma(k) - 2*digamma((k-1)+1) + digamma(n)
        _compare_result(mi_alg1, expected_alg1, 'PythonKraskovMI_alg1',
                        'Analytic', 'MI', tol=0.00001)
        settings['algorithm_num'] = 2
        est2 = PythonKraskovMI(settings)
        mi_alg2 = est2.estimate(source, target)
        expected_alg2 = digamma(k) - 1/k - 2*digamma(k) + digamma(n)
        _compare_result(mi_alg2, expected_alg2, 'PythonKraskovMI_alg2',
                        'Analytic', 'MI', tol=0.00001)
        # And now check that it doesn't work for algorithm "3"
        settings['algorithm_num'] = 3
        caughtAssertionError = False
        try:
            PythonKraskovMI(settings)
        except AssertionError:
            caughtAssertionError = True
        assert caughtAssertionError, 'Assertion error not raised for KSG algorithm 3 request'

"""
def test_discrete_mi_memerror():
    #Test exception handling for memory exhausted exceptions.
    var1, var2 = _get_mem_binary_data()

    # Check that we catch instantiation error for an enormous history:
    caughtException = False
    settings = {'n_discrete_bins': 1000000000}
    result = -1
    try:
        mi_estimator = JidtDiscreteMI(settings=settings)
        result = mi_estimator.estimate(var1, var2)
        print('Result of MI calc (which should not have completed) was ', result)
    except ex.JidtOutOfMemoryError:
        caughtException = True
        print('ex.JidtOutOfMemoryError caught as required')
    assert caughtException, 'No exception instantiating MI calculator with 10^18 bins'
    # Check that we instantiate correctly for a small history, even after
    #  the error encountered above:
    caughtException = False
    settings = {'n_discrete_bins': 2}
    try:
        mi_estimator = JidtDiscreteMI(settings=settings)
        mi_estimator.estimate(var1, var2)
        print('Subsequent JIDT calculation worked OK')
    except ex.JidtOutOfMemoryError:
        caughtException = True
    assert not(caughtException), 'Unable to instantiate MI calculator with 2 bins'
"""


if __name__ == '__main__':
    
    test_mi_gauss_data()
    test_cmi_gauss_data_no_cond()
    test_cmi_gauss_data()
    test_ais_gauss_data()
    test_te_gauss_data()
    test_cte_gauss_data_no_cond()
    test_cte_gauss_data()
    test_lagged_mi()
    test_discrete_ais()
    test_local_values()
    test_one_two_dim_input_kraskov()
    test_one_two_dim_input_gaussian()
    test_one_two_dim_input_discrete()
    test_invalid_settings_input()
    test_invalid_history_parameters()
    test_insufficient_no_points()    
    
    
    #test_discrete_mi_memerror()
    
    #test_kraskov_alg1And2()
    
    
    


    
    