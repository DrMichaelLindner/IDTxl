"""Test OpenCL estimators.

This module provides unit tests for OpenCL estimators. Estimators are tested
against Python and partially also with Jidt estimators.
"""
import math
import pytest
import numpy as np
from idtxl.estimators_opencl import (OpenCLKraskovMI,
									OpenCLKraskovCMI,
									OpenCLGaussianMI,
									OpenCLGaussianCMI,
									OpenCLGaussianAIS,
									OpenCLGaussianTE,
									OpenCLGaussianCTE,
									OpenCLDiscreteMI,
									OpenCLDiscreteCMI,
									OpenCLDiscreteAIS,
									OpenCLDiscreteTE)
from idtxl.estimators_python import (PythonKraskovMI,
									PythonKraskovCMI,
									PythonGaussianMI,
									PythonGaussianCMI,
									PythonGaussianAIS,
									PythonGaussianTE,
									PythonGaussianCTE,
									PythonDiscreteMI,
									PythonDiscreteCMI,
									PythonDiscreteAIS,
									PythonDiscreteTE)
from idtxl.estimators_jidt import (JidtKraskovMI,
                                   JidtKraskovCMI,
                                   JidtGaussianMI,
                                   JidtGaussianAIS,
                                   JidtGaussianCMI,
                                   JidtGaussianTE,
                                   JidtDiscreteMI,
                                   JidtDiscreteAIS,
                                   JidtDiscreteCMI,
                                   JidtDiscreteTE)
from generate_test_data import _get_gauss_data, _get_ar_data
from testutils import opencl_missing, jpype_missing

SEED = 0

@opencl_missing
def test_debug_setting():
    """Test setting of debugging options."""
    settings = {'debug': False, 'return_counts': True}
    # Estimators should raise an error if returning of neighborhood counts is
    # requested without the debugging option being set.
    with pytest.raises(RuntimeError): OpenCLKraskovMI(settings=settings)
    with pytest.raises(RuntimeError): OpenCLKraskovCMI(settings=settings)

    settings['debug'] = True
    est = OpenCLKraskovMI(settings=settings)
    res = est.estimate(np.arange(10), np.arange(10))
    assert len(res) == 4, (
        'Requesting debugging output from MI estimator did not return the '
        'correct no. values.')
    est = OpenCLKraskovCMI(settings=settings)
    res = est.estimate(np.arange(10), np.arange(10), np.arange(10))
    assert len(res) == 5, (
        'Requesting debugging output from CMI estimator did not return the '
        'correct no. values.')

@opencl_missing
def test_amd_data_padding():
    """Test padding necessary for AMD devices."""
    expected_mi, source, source_uncorr, target = _get_gauss_data(seed=SEED)

    settings = {'debug': True, 'return_counts': True}
    est_mi = OpenCLKraskovMI(settings=settings)
    est_cmi = OpenCLKraskovCMI(settings=settings)

    # Run OpenCL estimator for various data sizes.
    for n in [11, 13, 25, 64, 100, 128, 999, 10000, 3781, 50000]:
        for n_chunks in [1, 3, 10, 50, 99]:
            data_run_source = np.tile(source[:n], (n_chunks, 1))
            data_run_target = np.tile(target[:n], (n_chunks, 1))
            mi, dist, n_range_var1, n_range_var2 = est_mi.estimate(
                data_run_source, data_run_target, n_chunks=n_chunks)
            cmi, dist, n_range_var1, n_range_var2 = est_cmi.estimate(
                data_run_source, data_run_target, n_chunks=n_chunks)
    # Run OpenCL esitmator for various no. points and check result for
    # correctness. Note that for smaller sample sizes the error becomes too
    # large.
    n_chunks = 1
    for n in [832, 999, 10000, 3781, 50000]:
        data_run_source = np.tile(source[:n], (n_chunks, 1))
        data_run_target = np.tile(target[:n], (n_chunks, 1))
        mi, dist, n_range_var1, n_range_var2 = est_mi.estimate(
            data_run_source, data_run_target, n_chunks=n_chunks)
        cmi, dist, n_range_var1, n_range_var2 = est_cmi.estimate(
            data_run_source, data_run_target, n_chunks=n_chunks)
        print('{0} points, {1} chunks: OpenCL MI result: {2:.4f} nats; '
              'expected to be close to {3:.4f} nats for correlated '
              'Gaussians.'.format(n, n_chunks, mi[0], expected_mi))
        assert np.isclose(mi[0], expected_mi, atol=0.05), (
            'MI estimation for uncorrelated Gaussians using the OpenCL '
            'estimator failed (error larger 0.05).')
        print('OpenCL CMI result: {0:.4f} nats; expected to be close to '
              '{1:.4f} nats for correlated Gaussians.'.format(
                    cmi[0], expected_mi))
        assert np.isclose(cmi[0], expected_mi, atol=0.05), (
            'CMI estimation for uncorrelated Gaussians using the OpenCL '
            'estimator failed (error larger 0.05).')

    # Test debugging switched off
    settings = {'debug': False, 'return_counts': False}
    est_mi = OpenCLKraskovMI(settings=settings)
    est_cmi = OpenCLKraskovCMI(settings=settings)
    mi = est_mi.estimate(source, target)
    cmi = est_cmi.estimate(source, target)

    settings['local_values'] = True
    est_mi = OpenCLKraskovMI(settings=settings)
    est_cmi = OpenCLKraskovCMI(settings=settings)
    mi = est_mi.estimate(source, target)
    cmi = est_cmi.estimate(source, target)

@opencl_missing
def test_user_input():
    print("### Test user input")

    est_mi_kraskov = OpenCLKraskovMI()
    est_cmi_kraskov = OpenCLKraskovCMI()

    est_mi_gaussian = OpenCLGaussianMI()
    est_cmi_gaussian = OpenCLGaussianCMI()
    est_te_gaussian = OpenCLGaussianTE({"history_target": 1})
    est_cte_gaussian = OpenCLGaussianCTE({"history_target": 1})

    est_mi_discrete = OpenCLDiscreteMI()
    est_cmi_discrete = OpenCLDiscreteCMI()
    est_te_discrete = OpenCLDiscreteTE({"history_target": 1})

    N = 1000

    # Unequal variable dimensions.
    # Kraskov
    with pytest.raises(AssertionError):
        est_mi_kraskov.estimate(var1=np.random.randn(N, 1),
                        var2=np.random.randn(N + 1, 1))
    with pytest.raises(AssertionError):
        est_cmi_kraskov.estimate(var1=np.random.randn(N, 1),
                         var2=np.random.randn(N + 1, 1),
                         conditional=np.random.randn(N, 1))
    with pytest.raises(AssertionError):
        est_cmi_kraskov.estimate(var1=np.random.randn(N, 1),
                         var2=np.random.randn(N, 1),
                         conditional=np.random.randn(N + 1, 1))

    # No. chunks doesn't fit the signal length.
    with pytest.raises(AssertionError):
        est_mi_kraskov.estimate(var1=np.random.randn(N, 1),
                        var2=np.random.randn(N, 1),
                        n_chunks=7)
    with pytest.raises(AssertionError):
        est_cmi_kraskov.estimate(var1=np.random.randn(N, 1),
                         var2=np.random.randn(N, 1),
                         conditional=np.random.randn(N, 1),
                         n_chunks=7)

    # Gaussian
    with pytest.raises(AssertionError):
        est_mi_gaussian.estimate(var1=np.random.randn(N, 1),
                        var2=np.random.randn(N + 1, 1))
    with pytest.raises(AssertionError):
        est_cmi_gaussian.estimate(var1=np.random.randn(N, 1),
                         var2=np.random.randn(N + 1, 1),
                         conditional=np.random.randn(N, 1))
    with pytest.raises(AssertionError):
        est_cmi_gaussian.estimate(var1=np.random.randn(N, 1),
                         var2=np.random.randn(N, 1),
                         conditional=np.random.randn(N + 1, 1))
    with pytest.raises(AssertionError):
        est_te_gaussian.estimate(source=np.random.randn(N, 1),
                                 target=np.random.randn(N + 1, 1))
    with pytest.raises(AssertionError):
        est_cte_gaussian.estimate(source=np.random.randn(N, 1),
                                  target=np.random.randn(N + 1, 1),
                                  conditional=np.random.randn(N, 1))
    with pytest.raises(AssertionError):
        est_cte_gaussian.estimate(source=np.random.randn(N, 1),
                                  target=np.random.randn(N, 1),
                                  conditional=np.random.randn(N + 1, 1))

    # Discrete
    with pytest.raises(AssertionError):
        est_mi_discrete.estimate(var1=np.random.randn(N, 1),
                        var2=np.random.randn(N + 1, 1))
    with pytest.raises(AssertionError):
        est_cmi_discrete.estimate(var1=np.random.randn(N, 1),
                         var2=np.random.randn(N + 1, 1),
                         conditional=np.random.randn(N, 1))
    with pytest.raises(AssertionError):
        est_cmi_discrete.estimate(var1=np.random.randn(N, 1),
                         var2=np.random.randn(N, 1),
                         conditional=np.random.randn(N + 1, 1))
    with pytest.raises(AssertionError):
        est_te_discrete.estimate(source=np.random.randn(N, 1),
                                 target=np.random.randn(N + 1, 1))

    print("- DONE")

# MI
@opencl_missing
@jpype_missing
def test_mi_correlated_gaussians_kraskov():
    """Test estimators on correlated Gaussian data."""
    print("### Test OpenCLKraskovMI 1D corr:")
    expected_mi, source, source_uncorr, target = _get_gauss_data(seed=SEED)

    # Run OpenCL estimator.
    settings = {'debug': True, 'return_counts': True}
    ocl_est = OpenCLKraskovMI(settings=settings)
    mi_ocl, dist, n_range_var1, n_range_var2 = ocl_est.estimate(source, target)

    mi_ocl = mi_ocl[0]
    # Run JIDT estimator.
    jidt_est = JidtKraskovMI(settings={})
    mi_jidt = jidt_est.estimate(source, target)

    print('JIDT MI result: {0:.4f} nats; OpenCL MI result: {1:.4f} nats; '
          'expected to be close to {2:.4f} nats for correlated '
          'Gaussians.'.format(mi_jidt, mi_ocl, expected_mi))
    assert np.isclose(mi_jidt, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'JIDT estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_jidt, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')

@opencl_missing
def test_mi_correlated_gaussians_gaussian():
    """Test estimators on correlated Gaussian data."""
    print("### Test OpenCLGaussianMI 1D corr:")
    expected_mi, source, source_uncorr, target = _get_gauss_data(seed=SEED)

    # Run OpenCL estimator.
    settings = {'normalise': False, 'noise_level': 0}
    ocl_est = OpenCLGaussianMI(settings=settings)
    mi_ocl = ocl_est.estimate(source, target)

    # Run Python estimator.
    python_est = PythonGaussianMI(settings=settings)
    mi_python = python_est.estimate(source, target)

    print('Python MI result: {0:.4f} nats; OpenCL MI result: {1:.4f} nats; '
          'expected to be close to {2:.4f} nats for correlated '
          'Gaussians.'.format(mi_python, mi_ocl, expected_mi))
    assert np.isclose(mi_python, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'Python estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_python, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')

@opencl_missing
@jpype_missing
def test_mi_correlated_gaussians_discrete():
    """Test estimators on correlated Gaussian data."""
    print("### Test OpenCLDiscreteMI 1D corr:")
    expected_mi, source, source_uncorr, target = _get_gauss_data(seed=SEED)

    # Run OpenCL estimator.
    settings = {'discretise_method': 'equal', 'noise_level': 0}
    ocl_est = OpenCLDiscreteMI(settings=settings)
    mi_ocl = ocl_est.estimate(source, target)

    # Run Python estimator.
    python_est = PythonDiscreteMI(settings=settings)
    mi_python = python_est.estimate(source, target)

    # Run Jidt estimator.
    jidt_est = JidtDiscreteMI(settings=settings)
    mi_jidt = jidt_est.estimate(source, target)

    print('Python MI result: {0:.4f} nats; OpenCL MI result: {1:.4f} nats; Jidt MI result: {2:.4f} nats; '
          'expected to be close to {3:.4f} nats for correlated '
          'Gaussians.'.format(mi_python, mi_ocl, mi_jidt, expected_mi))
    assert np.isclose(mi_python, mi_jidt, atol=0.001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'Python estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_jidt, atol=0.001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_python, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')

@opencl_missing
@jpype_missing
def test_mi_uncorrelated_gaussians_kraskov():
    """Test MI estimator on uncorrelated Gaussian data."""
    print("### Test OpenCLKraskovMI 1D uncorr:")
    n_obs = 10000
    np.random.seed(SEED)
    var1 = np.random.randn(n_obs, 1)
    var2 = np.random.randn(n_obs, 1)

    # Run OpenCL estimator.
    settings = {'debug': True, 'return_counts': True}
    ocl_est = OpenCLKraskovMI(settings=settings)
    mi_ocl, dist, n_range_var1, n_range_var2 = ocl_est.estimate(var1, var2)
    mi_ocl = mi_ocl[0]

    # Run JIDT estimator.
    jidt_est = JidtKraskovMI(settings={'noise_level':0})
    mi_jidt = jidt_est.estimate(var1, var2)

    print('JIDT MI result: {0:.4f} nats; OpenCL MI result: {1:.4f} nats; '
          'expected to be close to 0 nats for uncorrelated '
          'Gaussians.'.format(mi_jidt, mi_ocl))
    assert np.isclose(mi_jidt, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'JIDT estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_jidt, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')

@opencl_missing
def test_mi_uncorrelated_gaussians_gaussian():
    """Test MI estimator on uncorrelated Gaussian data."""
    print("### Test OpenCLGaussianMI 1D uncorr:")
    n_obs = 10000
    np.random.seed(SEED)
    var1 = np.random.randn(n_obs, 1)
    var2 = np.random.randn(n_obs, 1)

    # Run OpenCL estimator.
    settings = {'normalise': False, 'noise_level': 0}
    ocl_est = OpenCLGaussianMI(settings=settings)
    mi_ocl = ocl_est.estimate(var1, var2)

    # Run Python estimator.
    python_est = PythonGaussianMI(settings=settings)
    mi_python = python_est.estimate(var1, var2)

    print('Python MI result: {0:.4f} nats; OpenCL MI result: {1:.4f} nats; '
          'expected to be close to 0 nats for uncorrelated '
          'Gaussians.'.format(mi_python, mi_ocl))
    assert np.isclose(mi_python, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'Python estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_python, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')

@opencl_missing
def test_mi_uncorrelated_gaussians_discrete():
    """Test MI estimator on uncorrelated Gaussian data."""
    print("### Test OpenCLDiscreteMI 1D uncorr:")
    n_obs = 10000
    np.random.seed(SEED)
    var1 = np.random.randn(n_obs, 1)
    var2 = np.random.randn(n_obs, 1)

    # Run OpenCL estimator.
    settings = {'discretise_method': 'max_ent', 'normalise': False, 'noise_level': 0}
    ocl_est = OpenCLDiscreteMI(settings=settings)
    mi_ocl = ocl_est.estimate(var1, var2)

    # Run Python estimator.
    python_est = PythonDiscreteMI(settings=settings)
    mi_python = python_est.estimate(var1, var2)

    print('Python MI result: {0:.4f} nats; OpenCL MI result: {1:.4f} nats; '
          'expected to be close to 0 nats for uncorrelated '
          'Gaussians.'.format(mi_python, mi_ocl))
    assert np.isclose(mi_python, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'Python estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_python, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')

@opencl_missing
@jpype_missing
def test_mi_uncorrelated_gaussians_three_dims_kraskov():
    """Test MI estimator on uncorrelated 3D Gaussian data."""
    print("### Test OpenCLKrakovMI 2D uncorr:")
    n_obs = 10000
    dim = 3
    np.random.seed(SEED)
    var1 = np.random.randn(n_obs, dim)
    var2 = np.random.randn(n_obs, dim)

    # Run OpenCL estimator.
    settings = {'debug': True, 'return_counts': True}
    ocl_est = OpenCLKraskovMI(settings=settings)
    mi_ocl, dist, n_range_var1, n_range_var2 = ocl_est.estimate(var1, var2)
    mi_ocl = mi_ocl[0]

    # Run JIDT estimator.
    jidt_est = JidtKraskovMI(settings={'noise_level':0})
    mi_jidt = jidt_est.estimate(var1, var2)

    print('JIDT MI result: {0:.4f} nats; OpenCL MI result: {1:.4f} nats; '
          'expected to be close to 0 nats for uncorrelated '
          'Gaussians.'.format(mi_jidt, mi_ocl))
    assert np.isclose(mi_jidt, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'JIDT estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_jidt, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')

@opencl_missing
def test_mi_uncorrelated_gaussians_three_dims_gaussian():
    """Test MI estimator on uncorrelated 3D Gaussian data."""
    print("### Test OpenCLGaussianMI 2D uncorr:")
    n_obs = 10000
    dim = 3
    np.random.seed(SEED)
    var1 = np.random.randn(n_obs, dim)
    var2 = np.random.randn(n_obs, dim)

    # Run OpenCL estimator.
    settings = {'noise_level':0}
    ocl_est = OpenCLGaussianMI(settings=settings)
    mi_ocl = ocl_est.estimate(var1, var2)

    # Run Python estimator.
    python_est = PythonGaussianMI(settings=settings)
    mi_python = python_est.estimate(var1, var2)

    print('Python MI result: {0:.4f} nats; OpenCL MI result: {1:.4f} nats; '
          'expected to be close to 0 nats for uncorrelated '
          'Gaussians.'.format(mi_python, mi_ocl))
    assert np.isclose(mi_python, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'Python estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_python, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')

@opencl_missing
@jpype_missing
def test_mi_uncorrelated_gaussians_three_dims_discrete():
    """Test MI estimator on uncorrelated 3D Gaussian data."""
    print("### Test OpenCLDiscreteMI 2D uncorr:")
    n_obs = 10000
    dim = 3
    np.random.seed(SEED)
    var1 = np.random.randn(n_obs, dim)
    var2 = np.random.randn(n_obs, dim)

    # Run OpenCL estimator.
    settings = {'discretise_method': 'equal', 'noise_level':0}
    ocl_est = OpenCLDiscreteMI(settings=settings)
    mi_ocl = ocl_est.estimate(var1, var2)

    # Run Python estimator.
    python_est = PythonDiscreteMI(settings=settings)
    mi_python = python_est.estimate(var1, var2)

    print('Python MI result: {0:.4f} nats; OpenCL MI result: {1:.4f} nats; '
          'expected to be close to 0 nats for uncorrelated '
          'Gaussians.'.format(mi_python, mi_ocl))
    assert np.isclose(mi_python, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'Python estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_python, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')

@opencl_missing
@jpype_missing
def test_mi_correlated_gaussians_two_chunks():
    """Test estimators on two chunks of correlated Gaussian data."""
    expected_mi, source, source_uncorr, target = _get_gauss_data(
        n=20000, seed=SEED)
    n_points = source.shape[0]

    # Run OpenCL estimator.
    n_chunks = 2
    settings = {'debug': True, 'return_counts': True}
    ocl_est = OpenCLKraskovMI(settings=settings)
    mi_ocl, dist, n_range_var1, n_range_var2 = ocl_est.estimate(
                                                            source, target,
                                                            n_chunks=n_chunks)

    # Run JIDT estimator.
    jidt_est = JidtKraskovMI(settings={'noise_level':0})
    mi_jidt = jidt_est.estimate(source[0:int(n_points/2), :],
                                target[0:int(n_points/2), :])

    print('JIDT MI result: {0:.4f} nats; OpenCL MI result: [{1:.4f}, {2:.4f}] '
          'nats; expected to be close to {3:.4f} nats for correlated '
          'Gaussians.'.format(mi_jidt, mi_ocl[0], mi_ocl[1], expected_mi))
    assert np.isclose(mi_jidt, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'JIDT estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl[0], expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl[0], mi_jidt, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl[1], mi_jidt, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl[0], mi_ocl[1], atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')

# CMI
@opencl_missing
@jpype_missing
def test_cmi_correlated_gaussians_kraskov():
    """Test estimators on correlated Gaussian data with conditional."""
    print("### Test OpenCLKraskovCMI 1D corr:")
    expected_mi, source, source_uncorr, target = _get_gauss_data(seed=SEED)

    # Run OpenCL estimator.
    settings = {'debug': True, 'return_counts': True}
    ocl_est = OpenCLKraskovCMI(settings=settings)
    (mi_ocl, dist, n_range_var1,
     n_range_var2, n_range_cond) = ocl_est.estimate(source, target,
                                                    source_uncorr)

    mi_ocl = mi_ocl[0]
    # Run Python estimator.
    python_est = PythonKraskovCMI(settings={'noise_level':0})
    mi_python = python_est.estimate(source, target, source_uncorr)

    print('Python MI result: {0:.4f} nats; OpenCL MI result: {1:.4f} nats; '
          'expected to be close to {2:.4f} nats for correlated '
          'Gaussians.'.format(mi_python, mi_ocl, expected_mi))
    assert np.isclose(mi_python, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'Python estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_python, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')

@opencl_missing
def test_cmi_correlated_gaussians_gaussian():
    """Test estimators on correlated Gaussian data with conditional."""
    print("### Test OpenCLGaussianCMI 1D corr:")
    expected_mi, source, source_uncorr, target = _get_gauss_data(seed=SEED)

    # Run OpenCL estimator.
    settings = {'noise_level': 0, 'normalize': False}
    ocl_est = OpenCLGaussianCMI(settings=settings)
    mi_ocl = ocl_est.estimate(source, target, source_uncorr)

    # Run Python estimator.
    python_est = PythonGaussianCMI(settings=settings)
    mi_python = python_est.estimate(source, target, source_uncorr)

    print('Python MI result: {0:.4f} nats; OpenCL MI result: {1:.4f} nats; '
          'expected to be close to {2:.4f} nats for correlated '
          'Gaussians.'.format(mi_python, mi_ocl, expected_mi))
    assert np.isclose(mi_python, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'Python estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_python, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')

@opencl_missing
def test_cmi_correlated_gaussians_discrete():
    """Test estimators on correlated Gaussian data with conditional."""
    print("### Test OpenCLGaussianCMI 1D corr:")
    expected_mi, source, source_uncorr, target = _get_gauss_data(seed=SEED)

    # Run OpenCL estimator.
    settings = {'discretise_method': 'equal', 'noise_level': 0}
    ocl_est = OpenCLDiscreteCMI(settings=settings)
    mi_ocl = ocl_est.estimate(source, target, source_uncorr)

    # Run Python estimator.
    python_est = PythonDiscreteCMI(settings=settings)
    mi_python = python_est.estimate(source, target, source_uncorr)

    # Run Jidt estimator.
    jidt_est = JidtDiscreteCMI(settings=settings)
    mi_jidt = jidt_est.estimate(source, target, source_uncorr)

    print('Python MI result: {0:.4f} nats; OpenCL MI result: {1:.4f} nats; Jidt MI result: {2:.4f} nats; '
          'expected to be close to {3:.4f} nats for correlated '
          'Gaussians.'.format(mi_python, mi_ocl, mi_jidt, expected_mi))
    assert np.isclose(mi_python, mi_jidt, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'Python estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_jidt, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_python, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')

@opencl_missing
@jpype_missing
def test_cmi_uncorrelated_gaussians_kraskov():
    """Test estimators on correlated Gaussian data with conditional."""
    print("### Test OpenCLKraskovCMI 1D uncorr:")
    n_obs = 10000
    np.random.seed(SEED)
    var1 = np.random.randn(n_obs, 1)
    var2 = np.random.randn(n_obs, 1)
    var3 = np.random.randn(n_obs, 1)

    # Run OpenCL estimator.
    settings = {'debug': True, 'return_counts': True}
    ocl_est = OpenCLKraskovCMI(settings=settings)
    (mi_ocl, dist, n_range_var1,
     n_range_var2, n_range_cond) = ocl_est.estimate(var1, var2, var3)

    mi_ocl = mi_ocl[0]
    # Run Python estimator.
    python_est = PythonKraskovCMI(settings={'noise_level':0})
    mi_python = python_est.estimate(var1, var2, var3)

    print('Python MI result: {0:.4f} nats; OpenCL MI result: {1:.4f} nats; '
          'expected to be close to 0 nats for correlated '
          'Gaussians.'.format(mi_python, mi_ocl))
    assert np.isclose(mi_python, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'Python estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_python, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')

@opencl_missing
def test_cmi_uncorrelated_gaussians_gaussian():
    """Test estimators on correlated Gaussian data with conditional."""
    print("### Test OpenCLGaussianCMI 1D uncorr:")
    n_obs = 10000
    np.random.seed(SEED)
    var1 = np.random.randn(n_obs, 1)
    var2 = np.random.randn(n_obs, 1)
    var3 = np.random.randn(n_obs, 1)

    # Run OpenCL estimator.
    settings = {'noise_level': 0, 'normalize': False}
    ocl_est = OpenCLGaussianCMI(settings=settings)
    mi_ocl = ocl_est.estimate(var1, var2, var3)

    # Run Python estimator.
    python_est = PythonGaussianCMI(settings=settings)
    mi_python = python_est.estimate(var1, var2, var3)

    print('Python MI result: {0:.4f} nats; OpenCL MI result: {1:.4f} nats; '
          'expected to be close to 0 nats for correlated '
          'Gaussians.'.format(mi_python, mi_ocl))
    assert np.isclose(mi_python, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'Python estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_python, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')

@opencl_missing
def test_cmi_uncorrelated_gaussians_discrete():
    """Test estimators on correlated Gaussian data with conditional."""
    print("### Test OpenCLGaussianCMI 1D uncorr:")
    n_obs = 10000
    np.random.seed(SEED)
    var1 = np.random.randn(n_obs, 1)
    var2 = np.random.randn(n_obs, 1)
    var3 = np.random.randn(n_obs, 1)

    # Run OpenCL estimator.
    settings = {'discretise_method': 'equal', 'noise_level': 0}
    ocl_est = OpenCLDiscreteCMI(settings=settings)
    mi_ocl = ocl_est.estimate(var1, var2, var3)

    # Run Python estimator.
    python_est = PythonDiscreteCMI(settings=settings)
    mi_python = python_est.estimate(var1, var2, var3)

    # Run Jidt estimator.
    jidt_est = JidtDiscreteCMI(settings=settings)
    mi_jidt = jidt_est.estimate(var1, var2, var3)

    print('Python MI result: {0:.4f} nats; OpenCL MI result: {1:.4f} nats; Jidt MI result: {2:.4f} nats; '
          'expected to be close to 0 nats for correlated '
          'Gaussians.'.format(mi_python, mi_ocl, mi_jidt))
    assert np.isclose(mi_python, mi_jidt, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'Python estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_jidt, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_python, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')

@opencl_missing
@jpype_missing
def test_cmi_uncorrelated_gaussians_three_dims_kraskov():
    """Test CMI estimator on uncorrelated 3D Gaussian data."""
    print("### Test OpenCLKraskovCMI 2D:")
    n_obs = 10000
    dim = 3
    np.random.seed(SEED)
    var1 = np.random.randn(n_obs, dim)
    var2 = np.random.randn(n_obs, dim)
    var3 = np.random.randn(n_obs, dim)

    # Run OpenCL estimator.
    settings = {'debug': True, 'return_counts': True}
    ocl_est = OpenCLKraskovCMI(settings=settings)
    mi_ocl, dist, n_range_var1, n_range_var2 = ocl_est.estimate(var1, var2)
    mi_ocl = mi_ocl[0]

    # Run JIDT estimator.
    jidt_est = JidtKraskovCMI(settings={'noise_level':0})
    mi_jidt = jidt_est.estimate(var1, var2)

    print('JIDT MI result: {0:.4f} nats; OpenCL MI result: {1:.4f} nats; '
          'expected to be close to 0 nats for uncorrelated '
          'Gaussians.'.format(mi_jidt, mi_ocl))
    assert np.isclose(mi_jidt, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'JIDT estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_jidt, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')

    # Run with conditional
    (mi_ocl, dist, n_range_var1,
     n_range_var2, n_range_var3) = ocl_est.estimate(var1, var2, var3)
    mi_ocl = mi_ocl[0]
    mi_jidt = jidt_est.estimate(var1, var2, var3)

    print('JIDT MI result: {0:.4f} nats; OpenCL MI result: {1:.4f} nats; '
          'expected to be close to 0 nats for uncorrelated '
          'Gaussians.'.format(mi_jidt, mi_ocl))
    assert np.isclose(mi_jidt, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'JIDT estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_jidt, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')

@opencl_missing
def test_cmi_uncorrelated_gaussians_three_dims_gaussian():
    """Test CMI estimator on uncorrelated 3D Gaussian data."""
    print("### Test OpenCLGaussianCMI 2D:")
    n_obs = 10000
    dim = 3
    np.random.seed(SEED)
    var1 = np.random.randn(n_obs, dim)
    var2 = np.random.randn(n_obs, dim)
    var3 = np.random.randn(n_obs, dim)

    # Run OpenCL estimator.
    settings = {'noise_level':0}
    ocl_est = OpenCLGaussianCMI(settings=settings)
    mi_ocl = ocl_est.estimate(var1, var2)

    # Run Python estimator.
    python_est = PythonGaussianCMI(settings=settings)
    mi_python = python_est.estimate(var1, var2)

    print('Python MI result: {0:.4f} nats; OpenCL MI result: {1:.4f} nats; '
          'expected to be close to 0 nats for uncorrelated '
          'Gaussians.'.format(mi_python, mi_ocl))
    assert np.isclose(mi_python, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'Python estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_python, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')

    # Run with conditional
    mi_ocl = ocl_est.estimate(var1, var2, var3)
    mi_python = python_est.estimate(var1, var2, var3)

    print('Python CMI result: {0:.4f} nats; OpenCL CMI result: {1:.4f} nats; '
          'expected to be close to 0 nats for uncorrelated '
          'Gaussians.'.format(mi_python, mi_ocl))
    assert np.isclose(mi_python, 0, atol=0.05), (
                        'CMI estimation for uncorrelated Gaussians using the '
                        'Python estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, 0, atol=0.05), (
                        'CMI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_python, atol=0.0001), (
                        'CMI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')

@opencl_missing
def test_cmi_uncorrelated_gaussians_three_dims_discrete():
    """Test CMI estimator on uncorrelated 3D Gaussian data."""
    print("### Test OpenCLGaussianCMI 2D:")
    n_obs = 10000
    dim = 3
    np.random.seed(SEED)
    var1 = np.random.randn(n_obs, dim)
    var2 = np.random.randn(n_obs, dim)
    var3 = np.random.randn(n_obs, dim)

    # Run OpenCL estimator.
    settings = {'discretise_method': 'equal', 'noise_level':0}
    ocl_est = OpenCLDiscreteCMI(settings=settings)
    mi_ocl = ocl_est.estimate(var1, var2)

    # Run Python estimator.
    python_est = PythonDiscreteCMI(settings=settings)
    mi_python = python_est.estimate(var1, var2)

    print('Python MI result: {0:.4f} nats; OpenCL MI result: {1:.4f} nats; '
          'expected to be close to 0 nats for uncorrelated '
          'Gaussians.'.format(mi_python, mi_ocl))
    assert np.isclose(mi_python, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'Python estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_python, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')

    # Run with conditional
    mi_ocl = ocl_est.estimate(var1, var2, var3)
    mi_python = python_est.estimate(var1, var2, var3)

    print('Python CMI result: {0:.4f} nats; OpenCL CMI result: {1:.4f} nats; '
          'expected to be close to 0 nats for uncorrelated '
          'Gaussians.'.format(mi_python, mi_ocl))
    assert np.isclose(mi_python, 0, atol=0.05), (
                        'CMI estimation for uncorrelated Gaussians using the '
                        'Python estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, 0, atol=0.05), (
                        'CMI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_python, atol=0.0001), (
                        'CMI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')

@opencl_missing
@jpype_missing
def test_cmi_uncorrelated_gaussians_unequal_dims_kraskov():
    """Test CMI estimator on uncorrelated Gaussian data with unequal dims."""
    print("### Test OpenCLKraskovCMI unequal dims:")
    n_obs = 10000
    np.random.seed(SEED)
    var1 = np.random.randn(n_obs, 3)
    var2 = np.random.randn(n_obs, 5)
    var3 = np.random.randn(n_obs, 7)

    # Run OpenCL estimator.
    settings = {'debug': True, 'return_counts': True}
    ocl_est = OpenCLKraskovCMI(settings=settings)
    mi_ocl, dist, n_range_var1, n_range_var2 = ocl_est.estimate(var1, var2)
    mi_ocl = mi_ocl[0]

    # Run JIDT estimator.
    jidt_est = JidtKraskovCMI(settings={'noise_level':0})
    mi_jidt = jidt_est.estimate(var1, var2)

    print('JIDT MI result: {0:.4f} nats; OpenCL MI result: {1:.4f} nats; '
          'expected to be close to 0 nats for uncorrelated '
          'Gaussians.'.format(mi_jidt, mi_ocl))
    assert np.isclose(mi_jidt, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'JIDT estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_jidt, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')

    # Run estimation with conditionals.
    (mi_ocl, dist, n_range_var1,
     n_range_var2, n_range_var3) = ocl_est.estimate(var1, var2, var3)
    mi_ocl = mi_ocl[0]
    mi_jidt = jidt_est.estimate(var1, var2, var3)

    print('JIDT CMI result: {0:.4f} nats; OpenCL CMI result: {1:.4f} nats; '
          'expected to be close to 0 nats for uncorrelated '
          'Gaussians.'.format(mi_jidt, mi_ocl))
    assert np.isclose(mi_jidt, 0, atol=0.05), (
                        'CMI estimation for uncorrelated Gaussians using the '
                        'JIDT estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, 0, atol=0.05), (
                        'CMI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_jidt, atol=0.0001), (
                        'CMI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')

@opencl_missing
@jpype_missing
def test_cmi_uncorrelated_gaussians_unequal_dims_gaussian():
    """Test CMI estimator on uncorrelated Gaussian data with unequal dims."""
    print("### Test OpenCLGaussianCMI unequal dims:")
    n_obs = 10000
    np.random.seed(SEED)
    var1 = np.random.randn(n_obs, 3)
    var2 = np.random.randn(n_obs, 5)
    var3 = np.random.randn(n_obs, 7)

    # Run OpenCL estimator.
    settings = {'noise_level':0}
    ocl_est = OpenCLGaussianCMI(settings=settings)
    mi_ocl = ocl_est.estimate(var1, var2)

    # Run Python estimator.
    python_est = PythonGaussianCMI(settings=settings)
    mi_python = python_est.estimate(var1, var2)

    print('Python MI result: {0:.4f} nats; OpenCL MI result: {1:.4f} nats; '
          'expected to be close to 0 nats for uncorrelated '
          'Gaussians.'.format(mi_python, mi_ocl))
    assert np.isclose(mi_python, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'JIDT estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_python, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')

    # Run estimation with conditionals.
    mi_ocl = ocl_est.estimate(var1, var2, var3)
    mi_python = python_est.estimate(var1, var2, var3)

    print('Python CMI result: {0:.4f} nats; OpenCL CMI result: {1:.4f} nats; '
          'expected to be close to 0 nats for uncorrelated '
          'Gaussians.'.format(mi_python, mi_ocl))
    assert np.isclose(mi_python, 0, atol=0.05), (
                        'CMI estimation for uncorrelated Gaussians using the '
                        'Python estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, 0, atol=0.05), (
                        'CMI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_python, atol=0.0001), (
                        'CMI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')

@opencl_missing
@jpype_missing
def test_cmi_uncorrelated_gaussians_unequal_dims_discrete():
    """Test CMI estimator on uncorrelated Gaussian data with unequal dims."""
    print("### Test OpenCLDiscreteCMI unequal dims:")
    n_obs = 10000
    np.random.seed(SEED)
    var1 = np.random.randn(n_obs, 3)
    var2 = np.random.randn(n_obs, 5)
    var3 = np.random.randn(n_obs, 7)

    # Run OpenCL estimator.
    settings = {'discretise_method': 'equal', 'noise_level': 0}
    ocl_est = OpenCLDiscreteCMI(settings=settings)
    mi_ocl = ocl_est.estimate(var1, var2)

    # Run JIDT estimator.
    jidt_est = JidtDiscreteCMI(settings=settings)
    mi_jidt = jidt_est.estimate(var1, var2)

    # Run Python estimator.
    python_est = PythonDiscreteCMI(settings=settings)
    mi_python = python_est.estimate(var1, var2)

    print('JIDT MI result: {0:.4f} nats; OpenCL MI result: {1:.4f} nats; Python MI result: {2:.4f} nats; '
          'expected to be close to 0 nats for uncorrelated '
          'Gaussians.'.format(mi_jidt, mi_ocl, mi_python))
    assert np.isclose(mi_jidt, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'JIDT estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_jidt, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')

    # Run estimation with conditionals.
    mi_ocl = ocl_est.estimate(var1, var2, var3)
    mi_jidt = jidt_est.estimate(var1, var2, var3)
    mi_python = python_est.estimate(var1, var2, var3)

    print('JIDT CMI result: {0:.4f} nats; OpenCL CMI result: {1:.4f} nats; OpenCL CMI result: {2:.4f} nats; '
          'expected to be close to 0 nats for uncorrelated '
          'Gaussians.'.format(mi_jidt, mi_ocl, mi_python))
    assert np.isclose(mi_python, mi_jidt, atol=0.05), (
                        'CMI estimation for uncorrelated Gaussians using the '
                        'JIDT estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_python, atol=0.05), (
                        'CMI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_jidt, atol=0.0001), (
                        'CMI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')

@opencl_missing
@jpype_missing
def test_cmi_no_cond_correlated_gaussians_kraskov():
    """Test estimators on correlated Gaussian data without conditional."""
    print("### Test OpenCLKraskovCMI no cond corr")
    expected_mi, source, source_uncorr, target = _get_gauss_data(seed=SEED)

    # Run OpenCL estimator.
    settings = {'debug': True, 'return_counts': True}
    ocl_est = OpenCLKraskovCMI(settings=settings)
    mi_ocl, dist, n_range_var1, n_range_var2 = ocl_est.estimate(source, target)
    mi_ocl = mi_ocl[0]

    # Run JIDT estimator.
    jidt_est = JidtKraskovCMI(settings={'noise_level':0})
    mi_jidt = jidt_est.estimate(source, target)

    print('JIDT MI result: {0:.4f} nats; OpenCL MI result: {1:.4f} nats; '
          'expected to be close to {2:.4f} nats for correlated '
          'Gaussians.'.format(mi_jidt, mi_ocl, expected_mi))
    assert np.isclose(mi_jidt, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'JIDT estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_jidt, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')

@opencl_missing
def test_cmi_no_cond_correlated_gaussians_gaussian():
    """Test estimators on correlated Gaussian data without conditional."""
    print("### Test OpenCLGaussianCMI no cond corr")
    expected_mi, source, source_uncorr, target = _get_gauss_data(seed=SEED)

    # Run OpenCL estimator.
    settings = {'noise_level': 0}
    ocl_est = OpenCLGaussianCMI(settings=settings)
    mi_ocl = ocl_est.estimate(source, target)

    # Run Python estimator.
    python_est = PythonGaussianCMI(settings=settings)
    mi_python = python_est.estimate(source, target)

    print('Python MI result: {0:.4f} nats; OpenCL MI result: {1:.4f} nats; '
          'expected to be close to {2:.4f} nats for correlated '
          'Gaussians.'.format(mi_python, mi_ocl, expected_mi))
    assert np.isclose(mi_python, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'Python estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_python, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')

@opencl_missing
def test_cmi_no_cond_correlated_gaussians_discrete():
    """Test estimators on correlated Gaussian data without conditional."""
    print("### Test OpenCLDiscreteCMI no cond corr")
    expected_mi, source, source_uncorr, target = _get_gauss_data(seed=SEED)

    # Run OpenCL estimator.
    settings = {'discretise_method': 'equal', 'noise_level': 0}
    ocl_est = OpenCLDiscreteCMI(settings=settings)
    mi_ocl = ocl_est.estimate(source, target)

    # Run Python estimator.
    python_est = PythonDiscreteCMI(settings=settings)
    mi_python = python_est.estimate(source, target)

    # Run Jidt estimator.
    jidt_est = JidtDiscreteCMI(settings=settings)
    mi_jidt = jidt_est.estimate(source, target)

    print('Python MI result: {0:.4f} nats; OpenCL MI result: {1:.4f} nats; Jidt MI result: {2:.4f} nats; '
          'expected to be close to {3:.4f} nats for correlated '
          'Gaussians.'.format(mi_python, mi_ocl, mi_jidt, expected_mi))
    assert np.isclose(mi_python, mi_jidt, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'Python estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_jidt, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_python, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')

# AIS
@opencl_missing
def test_ais_gaussian():
    """Test estimators on AR data."""
    print("### Test OpenCLGaussianAIS 1D with history:")
    source1, source2 = _get_ar_data(seed=SEED)

    settings = {'history': 2, 'tau': 1}
    opencl_estimator = OpenCLGaussianAIS(settings=settings)
    python_estimator = PythonGaussianAIS(settings=settings)
    jidt_estimator = JidtGaussianAIS(settings=settings)

    ais_ocl = opencl_estimator.estimate(source1)
    ais_python = python_estimator.estimate(source1)
    ais_jidt = jidt_estimator.estimate(source1)

    print('Python AIS result: {0:.4f} nats; OpenCL AIS result: {1:.4f} nats '
          'Jidt AIS result: {2:.4f} nats;.'.format(ais_python, ais_ocl, ais_jidt))
    assert np.isclose(ais_ocl, ais_python, atol=0.001), (
        'AIS estimation for uncorrelated Gaussians using the '
        'OpenCL estimator failed (error larger 0.001).')
    assert np.isclose(ais_ocl, ais_jidt, atol=0.001), (
        'AIS estimation for uncorrelated Gaussians using the '
        'OpenCL estimator failed (error larger 0.001).')

    print("### Test OpenCLGaussianAIS 1D without history:")
    ais_ocl = opencl_estimator.estimate(source2)
    ais_python = python_estimator.estimate(source2)
    ais_jidt = jidt_estimator.estimate(source2)

    print('Python AIS result: {0:.4f} nats; OpenCL AIS result: {1:.4f} nats '
          'Jidt AIS result: {2:.4f} nats should be close to 0.'.format(ais_python, ais_ocl, ais_jidt))
    assert np.isclose(ais_python, 0, atol=0.05), (
        'AIS estimation for uncorrelated Gaussians using the '
        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(ais_ocl, 0, atol=0.05), (
        'AIS estimation for uncorrelated Gaussians using the '
        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(ais_ocl, ais_python, atol=0.0001), (
        'AIS estimation for uncorrelated Gaussians using the '
        'OpenCL estimator failed (error larger 0.0001).')
    assert np.isclose(ais_ocl, ais_jidt, atol=0.0001), (
        'AIS estimation for uncorrelated Gaussians using the '
        'OpenCL estimator failed (error larger 0.0001).')

    print("### Test OpenCLGaussianAIS 1D with history - local values:")
    settings = {'history': 2, 'tau': 1, 'local_valuies': True}
    opencl_estimator = OpenCLGaussianAIS(settings=settings)
    python_estimator = PythonGaussianAIS(settings=settings)
    jidt_estimator = JidtGaussianAIS(settings=settings)

    ais_ocl3 = opencl_estimator.estimate(source1)
    ais_python3 = python_estimator.estimate(source1)
    ais_jidt3 = jidt_estimator.estimate(source1)

    print('Python AIS result: {0:.4f} nats; OpenCL AIS result: {1:.4f} nats '
          'Jidt AIS result: {2:.4f} nats;.'.format(np.mean(ais_python3), np.mean(ais_ocl3), np.mean(ais_jidt3)))
    assert np.allclose(ais_ocl, ais_python, atol=0.001), (
        'AIS estimation for uncorrelated Gaussians using the '
        'OpenCL estimator failed (error larger 0.001).')
    assert np.isclose(ais_ocl, ais_jidt, atol=0.001), (
        'AIS estimation for uncorrelated Gaussians using the '
        'OpenCL estimator failed (error larger 0.001).')

    print("### Test OpenCLGaussianAIS 1D without history - local values:")
    ais_ocl = opencl_estimator.estimate(source2)
    ais_python = python_estimator.estimate(source2)
    ais_jidt = jidt_estimator.estimate(source2)

    print('Python AIS result: {0:.4f} nats; OpenCL AIS result: {1:.4f} nats '
          'Jidt AIS result: {2:.4f} nats;.'.format(np.mean(ais_python), np.mean(ais_ocl), np.mean(ais_jidt)))
    assert np.allclose(ais_ocl, ais_python, atol=0.001), (
        'AIS estimation for uncorrelated Gaussians using the '
        'OpenCL estimator failed (error larger 0.001).')
    assert np.isclose(ais_ocl, ais_jidt, atol=0.001), (
        'AIS estimation for uncorrelated Gaussians using the '
        'OpenCL estimator failed (error larger 0.001).')

@opencl_missing
def test_ais_discrete():
    """Test estimators on AR data."""
    print("### Test OpenCLDiscreteAIS 1D with history:")
    source1, source2 = _get_ar_data(seed=SEED)

    settings = {'discretise_method': 'max_ent', 'n_discrete_bins': 2, 'history': 2, 'tau': 1, 'noise_level': 0, 'normalise': False }
    opencl_estimator = OpenCLDiscreteAIS(settings=settings)
    python_estimator = PythonDiscreteAIS(settings=settings)
    jidt_estimator = JidtDiscreteAIS(settings=settings)

    ais_ocl1 = opencl_estimator.estimate(source1)
    ais_python1 = python_estimator.estimate(source1)
    ais_jidt1 = jidt_estimator.estimate(source1)

    print('Python AIS result: {0:.4f} nats; OpenCL AIS result: {1:.4f} nats '
          'Jidt AIS result: {2:.4f} nats;.'.format(ais_python1, ais_ocl1, ais_jidt1))
    assert np.isclose(ais_ocl1, ais_python1, atol=0.0001), (
        'AIS estimation for uncorrelated Gaussians using the '
        'OpenCL estimator failed (error larger 0.0001).')


    print("### Test OpenCLDiscreteAIS 1D without history:")
    ais_ocl2 = opencl_estimator.estimate(source2)
    ais_python2 = python_estimator.estimate(source2)
    ais_jidt2 = jidt_estimator.estimate(source2)

    print('Python AIS result: {0:.4f} nats; OpenCL AIS result: {1:.4f} nats '
          'Jidt AIS result: {2:.4f} nats;.'.format(ais_python2, ais_ocl2, ais_jidt2))
    assert np.isclose(ais_python2, 0, atol=0.05), (
        'AIS estimation for uncorrelated Gaussians using the '
        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(ais_ocl2, 0, atol=0.05), (
        'AIS estimation for uncorrelated Gaussians using the '
        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(ais_ocl2, ais_python2, atol=0.0001), (
        'AIS estimation for uncorrelated Gaussians using the '
        'OpenCL estimator failed (error larger 0.0001).')

    print("### Test OpenCLDiscreteAIS 1D with history - local values:")
    settings = {'discretise_method': 'max_ent', 'n_discrete_bins': 2, 'history': 2, 'tau': 1, 'noise_level': 0,
                'normalise': False, 'local_valuies': True}
    opencl_estimator = OpenCLDiscreteAIS(settings=settings)
    python_estimator = PythonDiscreteAIS(settings=settings)
    jidt_estimator = JidtDiscreteAIS(settings=settings)

    ais_ocl3 = opencl_estimator.estimate(source1)
    ais_python3 = python_estimator.estimate(source1)
    ais_jidt3 = jidt_estimator.estimate(source1)

    print('Mean of local values: Python AIS result: {0:.4f} nats; OpenCL AIS result: {1:.4f} nats '
          'Jidt AIS result: {2:.4f} nats;.'.format(np.mean(ais_python3), np.mean(ais_ocl3), np.mean(ais_jidt3)))
    assert np.allclose(ais_ocl3, ais_python3, atol=0.0001), (
        'AIS estimation for uncorrelated Gaussians using the '
        'OpenCL estimator failed (error larger 0.0001).')
    assert np.allclose(ais_ocl3, ais_jidt3, atol=0.0001), (
        'AIS estimation for uncorrelated Gaussians using the '
        'OpenCL estimator failed (error larger 0.0001).')

    print("### Test OpenCLDiscreteAIS 1D without history - local values:")
    ais_ocl4 = opencl_estimator.estimate(source2)
    ais_python4 = python_estimator.estimate(source2)
    ais_jidt4 = jidt_estimator.estimate(source2)

    print('Mean of local values: Python AIS result: {0:.4f} nats; OpenCL AIS result: {1:.4f} nats'
          'Jidt AIS result: {2:.4f} nats;.'.format(np.mean(ais_python4), np.mean(ais_ocl4), np.mean(ais_jidt4)))
    assert np.allclose(ais_ocl4, ais_python4, atol=0.0001), (
        'AIS estimation for uncorrelated Gaussians using the '
        'OpenCL estimator failed (error larger 0.0001).')
    assert np.allclose(ais_ocl4, ais_jidt4, atol=0.0001), (
        'AIS estimation for uncorrelated Gaussians using the '
        'OpenCL estimator failed (error larger 0.0001).')

# TE
@opencl_missing
def test_te_correlated_gaussians_gaussian():
    """Test estimators on correlated Gaussian data."""
    print("### Test OpenCLGaussianTE 1D corr:")
    expected_mi, source, source_uncorr, target = _get_gauss_data(seed=SEED)
    source=source[1:]
    target=target[:-1]
    source_uncorr=source_uncorr[1:]

    # Run OpenCL estimator.
    settings = {'normalise': False, 'history_target': 2,  'noise_level': 0}
    ocl_est = OpenCLGaussianTE(settings=settings)
    mi_ocl = ocl_est.estimate(source, target)

    # Run Python estimator.
    python_est = PythonGaussianTE(settings=settings)
    mi_python = python_est.estimate(source, target)

    print('Python MI result: {0:.4f} nats; OpenCL MI result: {1:.4f} nats; '
          'expected to be close to {2:.4f} nats for correlated '
          'Gaussians.'.format(mi_python, mi_ocl, expected_mi))
    assert np.isclose(mi_python, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'Python estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_python, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')

@opencl_missing
@jpype_missing
def test_te_correlated_gaussians_discrete():
    """Test estimators on correlated Gaussian data."""
    print("### Test OpenCLDiscreteTE 1D corr:")
    expected_mi, source, source_uncorr, target = _get_gauss_data(seed=SEED)
    source = source[1:]
    target = target[:-1]
    source_uncorr = source_uncorr[1:]

    # Run OpenCL estimator.
    settings = {'discretise_method': 'equal', 'history_target': 2, 'noise_level': 0}
    ocl_est = OpenCLDiscreteTE(settings=settings)
    mi_ocl = ocl_est.estimate(source, target)

    # Run Python estimator.
    python_est = PythonDiscreteTE(settings=settings)
    mi_python = python_est.estimate(source, target)

    # Run Jidt estimator.
    jidt_est = JidtDiscreteTE(settings=settings)
    mi_jidt = jidt_est.estimate(source, target)

    print('Python TE result: {0:.4f} nats; OpenCL TE result: {1:.4f} nats; Jidt TE result: {2:.4f} nats; '
          'expected to be close to {3:.4f} nats for correlated '
          'Gaussians.'.format(mi_python, mi_ocl, mi_jidt, expected_mi))
    assert np.isclose(mi_python, mi_jidt, atol=0.001), (
                        'TE estimation for uncorrelated Gaussians using the '
                        'Python estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_jidt, atol=0.001), (
                        'tE estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_python, atol=0.0001), (
                        'TE estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')

# CTE
@opencl_missing
def test_cte_correlated_gaussians_gaussian():
    """Test estimators on correlated Gaussian data with conditional."""
    print("### Test OpenCLGaussianCTE 1D corr:")
    expected_mi, source, source_uncorr, target = _get_gauss_data(seed=SEED)
    source=source[1:]
    target=target[:-1]
    source_uncorr=source_uncorr[1:]

    # Run OpenCL estimator.
    settings = {'history_target': 2,'noise_level': 0, 'normalize': False}
    ocl_est = OpenCLGaussianCTE(settings=settings)
    mi_ocl = ocl_est.estimate(source, target, source_uncorr)

    # Run Python estimator.
    python_est = PythonGaussianCTE(settings=settings)
    mi_python = python_est.estimate(source, target, source_uncorr)

    print('Python MI result: {0:.4f} nats; OpenCL MI result: {1:.4f} nats; '
          'expected to be close to {2:.4f} nats for correlated '
          'Gaussians.'.format(mi_python, mi_ocl, expected_mi))
    assert np.isclose(mi_python, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'Python estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_python, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')

@opencl_missing
def test_cte_uncorrelated_gaussians_gaussian():
    """Test estimators on correlated Gaussian data with conditional."""
    print("### Test OpenCLGaussianCMI 1D uncorr:")
    n_obs = 10000
    np.random.seed(SEED)
    var1 = np.random.randn(n_obs, 1)
    var2 = np.random.randn(n_obs, 1)
    var3 = np.random.randn(n_obs, 1)

    # Run OpenCL estimator.
    settings = {'history_target': 2, 'noise_level': 0, 'normalize': False}
    ocl_est = OpenCLGaussianCTE(settings=settings)
    mi_ocl = ocl_est.estimate(var1, var2, var3)

    # Run Python estimator.
    python_est = PythonGaussianCTE(settings=settings)
    mi_python = python_est.estimate(var1, var2, var3)

    print('Python MI result: {0:.4f} nats; OpenCL MI result: {1:.4f} nats; '
          'expected to be close to 0 nats for correlated '
          'Gaussians.'.format(mi_python, mi_ocl))
    assert np.isclose(mi_python, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'Python estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, 0, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_python, atol=0.0001), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')

@opencl_missing
def test_cte_no_cond_correlated_gaussians_gaussian():
    """Test estimators on correlated Gaussian data without conditional."""
    print("### Test OpenCLGaussianCTE no cond corr")
    expected_mi, source, source_uncorr, target = _get_gauss_data(seed=SEED)
    source = source[1:]
    target = target[:-1]
    source_uncorr = source_uncorr[1:]

    # Run OpenCL estimator.
    settings = {'history_target': 2, 'noise_level': 0}
    ocl_est = OpenCLGaussianCTE(settings=settings)
    mi_ocl = ocl_est.estimate(source, target)

    # Run Python estimator.
    python_est = PythonGaussianCTE(settings=settings)
    mi_python = python_est.estimate(source, target)

    print('Python TE result: {0:.4f} nats; OpenCL TE result: {1:.4f} nats; '
          'expected to be close to {2:.4f} nats for correlated '
          'Gaussians.'.format(mi_python, mi_ocl, expected_mi))
    assert np.isclose(mi_python, expected_mi, atol=0.05), (
                        'TE estimation for uncorrelated Gaussians using the '
                        'Python estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, expected_mi, atol=0.05), (
                        'TE estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')
    assert np.isclose(mi_ocl, mi_python, atol=0.0001), (
                        'TE estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')


@opencl_missing
def test_local_values_kraskov():
    """Test estimation of local MI and CMI using OpenCL estimators."""
    print("### Test OpenCLKraskovXX local values:")
    # Get data
    n_chunks = 2
    expec_mi, source, source_uncorr, target = _get_gauss_data(
        n=20000, seed=SEED)
    chunklength = int(source.shape[0] / n_chunks)

    # Estimate local values
    settings = {'local_values': True}
    est_cmi = OpenCLKraskovCMI(settings=settings)
    cmi = est_cmi.estimate(source, target, source_uncorr, n_chunks=n_chunks)

    est_mi = OpenCLKraskovMI(settings=settings)
    mi = est_mi.estimate(source, target, n_chunks=n_chunks)

    mi_ch1 = np.mean(mi[0:chunklength])
    mi_ch2 = np.mean(mi[chunklength:])
    cmi_ch1 = np.mean(cmi[0:chunklength])
    cmi_ch2 = np.mean(cmi[chunklength:])

    # Estimate non-local values for comparison
    settings = {'local_values': False}
    est_cmi = OpenCLKraskovCMI(settings=settings)
    mi = est_cmi.estimate(source, target, source_uncorr, n_chunks=n_chunks)

    est_mi = OpenCLKraskovMI(settings=settings)
    cmi = est_mi.estimate(source, target, n_chunks=n_chunks)

    # Report results
    print('OpenCL MI result: {0:.4f} nats (chunk 1); {1:.4f} nats (chunk 2) '
          'expected to be close to {2:.4f} nats for uncorrelated '
          'Gaussians.'.format(mi_ch1, mi_ch2, expec_mi))
    print('OpenCL CMI result: {0:.4f} nats (chunk 1); {1:.4f} nats (chunk 2) '
          'expected to be close to {2:.4f} nats for uncorrelated '
          'Gaussians.'.format(cmi_ch1, cmi_ch2, expec_mi))

    assert np.isclose(mi_ch1, expec_mi, atol=0.05)
    assert np.isclose(mi_ch2, expec_mi, atol=0.05)
    assert np.isclose(cmi_ch1, expec_mi, atol=0.05)
    assert np.isclose(cmi_ch2, expec_mi, atol=0.05)
    assert np.isclose(mi_ch1, mi_ch2, atol=0.05)
    assert np.isclose(mi_ch1, mi[0], atol=0.05)
    assert np.isclose(mi_ch2, mi[1], atol=0.05)
    assert np.isclose(cmi_ch1, cmi_ch2, atol=0.05)
    assert np.isclose(cmi_ch1, cmi[0], atol=0.05)
    assert np.isclose(cmi_ch2, cmi[1], atol=0.05)

@opencl_missing
def test_local_values_gaussian():
    """Test estimation of local MI and CMI using OpenCL estimators."""
    print("### Test OpenCLGaussianXX local values:")
    expec_mi, source, source_uncorr, target = _get_gauss_data(
        n=20000, seed=SEED)
    source2=source[1:]
    target2=target[:-1]
    source_uncorr2=source_uncorr[1:]

    # Estimate local values
    settings = {'local_values': True}

    est_mi = OpenCLGaussianMI(settings=settings)
    lmi = est_mi.estimate(source, target)
    est_cmi = OpenCLGaussianCMI(settings=settings)
    lcmi = est_cmi.estimate(source, target, source_uncorr)

    settings = {'local_values': True, 'history_target': 2}
    est_te = OpenCLGaussianTE(settings=settings)
    lte = est_te.estimate(source2, target2)
    est_cte = OpenCLGaussianCTE(settings=settings)
    lcte = est_cte.estimate(source2, target2, source_uncorr2)

    lmi_mean = np.mean(lmi)
    lcmi_mean = np.mean(lcmi)
    lte_mean = np.mean(lte)
    lcte_mean = np.mean(lcte)

    # Estimate non-local values for comparison
    settings = {'local_values': False}
    est_mi = OpenCLGaussianMI(settings=settings)
    cmi = est_mi.estimate(source, target)
    est_cmi = OpenCLGaussianCMI(settings=settings)
    mi = est_cmi.estimate(source, target, source_uncorr)

    settings = {'local_values': False, 'history_target': 2}
    est_te = OpenCLGaussianTE(settings=settings)
    te = est_te.estimate(source2, target2)
    est_cte = OpenCLGaussianCTE(settings=settings)
    cte = est_cte.estimate(source2, target2)

    # Report results
    print('OpenCL MI result: {0:.4f} nats  '
          'expected to be close to {1:.4f} nats for correlated '
          'Gaussians.'.format(lmi_mean, expec_mi))
    print('OpenCL CMI result: {0:.4f} '
          'expected to be close to {1:.4f} nats for correlated '
          'Gaussians.'.format(lcmi_mean, expec_mi))
    print('OpenCL TE result: {0:.4f} nats  '
          'expected to be close to {1:.4f} nats for correlated '
          'Gaussians.'.format(lte_mean, expec_mi))
    print('OpenCL CTE result: {0:.4f} nats  '
          'expected to be close to {1:.4f} nats for correlated '
          'Gaussians.'.format(lcte_mean, expec_mi))

    assert np.isclose(lmi_mean, expec_mi, atol=0.05)
    assert np.isclose(lmi_mean, mi, atol=0.05)

    assert np.isclose(lcmi_mean, expec_mi, atol=0.05)
    assert np.isclose(lcmi_mean, cmi, atol=0.05)

    assert np.isclose(lte_mean, expec_mi, atol=0.05)
    assert np.isclose(lte_mean, te, atol=0.05)

    assert np.isclose(lcte_mean, expec_mi, atol=0.05)
    assert np.isclose(lcte_mean, cte, atol=0.05)

@opencl_missing
def test_local_values_discrete():
    """Test estimation of local MI and CMI using OpenCL estimators."""
    print("### Test OpenCLDiscreteXX local values:")
    expec_mi, source, source_uncorr, target = _get_gauss_data(
        n=20000, seed=SEED)
    source2=source[1:]
    target2=target[:-1]
    source_uncorr2=source_uncorr[1:]

    # Estimate local values
    settings = {'discretise_method': 'equal', 'local_values': True}

    est_mi = OpenCLDiscreteMI(settings=settings)
    lmi = est_mi.estimate(source, target)
    est_cmi = OpenCLDiscreteCMI(settings=settings)
    lcmi = est_cmi.estimate(source, target, source_uncorr)

    settings = {'discretise_method': 'equal','local_values': True, 'history_target': 2}
    est_te = OpenCLDiscreteTE(settings=settings)
    lte = est_te.estimate(source2, target2)

    lmi_mean = np.mean(lmi)
    lcmi_mean = np.mean(lcmi)
    lte_mean = np.mean(lte)

    # Estimate non-local values for comparison
    settings = {'discretise_method': 'equal', 'local_values': False}
    est_mi = OpenCLDiscreteMI(settings=settings)
    cmi = est_mi.estimate(source, target)
    est_cmi = OpenCLDiscreteCMI(settings=settings)
    mi = est_cmi.estimate(source, target, source_uncorr)

    settings = {'discretise_method': 'equal', 'local_values': False, 'history_target': 2}
    est_te = OpenCLDiscreteTE(settings=settings)
    te = est_te.estimate(source2, target2)

    # Report results
    print('OpenCL MI result: {0:.4f} nats  '
          'expected to be close to {1:.4f} nats for correlated '
          'Gaussians.'.format(lmi_mean, expec_mi))
    print('OpenCL CMI result: {0:.4f} '
          'expected to be close to {1:.4f} nats for correlated '
          'Gaussians.'.format(lcmi_mean, expec_mi))
    print('OpenCL TE result: {0:.4f} nats  '
          'expected to be close to {1:.4f} nats for correlated '
          'Gaussians.'.format(lte_mean, expec_mi))

    assert np.isclose(lmi_mean, mi, atol=0.05)

    assert np.isclose(lcmi_mean, cmi, atol=0.05)

    assert np.isclose(lte_mean, te, atol=0.05)


@opencl_missing
def test_insufficient_no_points():
    """Test if estimation aborts for too few data points."""
    expected_mi, source1, source2, target = _get_gauss_data(n=4, seed=SEED)

    settings = {
        'kraskov_k': 4,
        'theiler_t': 0,
        'history': 1,
        'history_target': 1,
        'lag_mi': 1,
        'source_target_delay': 1}

    # Test first settings combination with k==N
    est = OpenCLKraskovMI(settings)
    with pytest.raises(RuntimeError): est.estimate(source1, target)
    est = OpenCLKraskovCMI(settings)
    with pytest.raises(RuntimeError): est.estimate(source1, target, target)

    # Test a second combination with a Theiler-correction != 0
    settings['theiler_t'] = 1
    settings['kraskov_k'] = 2
    est = OpenCLKraskovMI(settings)
    with pytest.raises(RuntimeError): est.estimate(source1, target)
    est = OpenCLKraskovCMI(settings)
    with pytest.raises(RuntimeError): est.estimate(source1, target, target)

@opencl_missing
@jpype_missing
def test_multi_gpu():
    """Test use of multiple GPUs."""
    expected_mi, source, source_uncorr, target = _get_gauss_data(seed=SEED)
    settings = {'debug': True, 'return_counts': True}

    # Get no. available devices on current platform.
    device_list, _, _, _ = OpenCLKraskovCMI()._get_device(gpuid=0)
    print(device_list)
    n_devices = len(device_list)

    # Try initialising estimator with unavailable GPU ID
    with pytest.raises(RuntimeError):
        settings['gpuid'] = n_devices + 1
        OpenCLKraskovCMI(settings=settings)

    # Run OpenCL estimator on available device with highest available ID.
    settings['gpuid'] = n_devices - 1
    ocl_est = OpenCLKraskovCMI(settings=settings)

    (mi_ocl, dist, n_range_var1,
     n_range_var2, n_range_cond) = ocl_est.estimate(source, target,
                                                    source_uncorr)

    mi_ocl = mi_ocl[0]
    print('Expected MI: {0:.4f} nats; OpenCL MI result: {1:.4f} nats; '
          'expected to be close to 0 nats for uncorrelated '
          'Gaussians.'.format(expected_mi, mi_ocl))
    assert np.isclose(mi_ocl, expected_mi, atol=0.05), (
                        'MI estimation for uncorrelated Gaussians using the '
                        'OpenCL estimator failed (error larger 0.05).')


@opencl_missing
def test_invalid_calculation_call():
    """Test OpenCL Gaussian and Discrete MI and CMI estimators for invalid call
     of calculate functions.
    Testing direct call of local and average calculate functions before AND
    after an estimate to check inf the flags are removed correctly"""

    print("Test invalid calls of calculate defs in all Python MI and CMI estimators")

    expected_mi, source1, source2, target = _get_gauss_data(n=100,seed=SEED)

    # Gaussian MI
    estimator = OpenCLGaussianMI(settings={})
    #with pytest.raises(RuntimeError): res = estimator.calculateAverageMI()
    with pytest.raises(RuntimeError): res = estimator.calculateLocalMI()
    res = estimator.estimate(source1, target)
    #with pytest.raises(RuntimeError): res = estimator.calculateAverageMI()
    with pytest.raises(RuntimeError): res = estimator.calculateLocalMI()

    # Gaussian CMI
    estimator = OpenCLGaussianCMI(settings={})
    with pytest.raises(RuntimeError): res = estimator.calculateAverageCMI()
    with pytest.raises(RuntimeError): res = estimator.calculateLocalCMI()
    res = estimator.estimate(source1, target, source2)
    with pytest.raises(RuntimeError): res = estimator.calculateAverageCMI()
    with pytest.raises(RuntimeError): res = estimator.calculateLocalCMI()

    # Discrete MI
    estimator = OpenCLDiscreteMI(settings={'discretise_method': 'max_ent'})
    with pytest.raises(RuntimeError): res = estimator.calculateAverageMI()
    with pytest.raises(RuntimeError): res = estimator.calculateLocalMI()
    res = estimator.estimate(source1, target)
    with pytest.raises(RuntimeError): res = estimator.calculateAverageMI()
    with pytest.raises(RuntimeError): res = estimator.calculateLocalMI()

    # Discrete CMI
    estimator = OpenCLDiscreteCMI(settings={'discretise_method': 'max_ent'})
    with pytest.raises(RuntimeError): res = estimator.calculateLocalCMI()
    res = estimator.estimate(source1, target, source2)
    with pytest.raises(RuntimeError): res = estimator.calculateLocalCMI()


if __name__ == '__main__':

    test_invalid_calculation_call()

    # all estimators
    test_user_input()
    # mi
    test_mi_correlated_gaussians_kraskov()
    test_mi_correlated_gaussians_gaussian()
    test_mi_correlated_gaussians_discrete()
    test_mi_uncorrelated_gaussians_kraskov()
    test_mi_uncorrelated_gaussians_gaussian()
    test_mi_uncorrelated_gaussians_discrete()
    test_mi_uncorrelated_gaussians_three_dims_kraskov()
    test_mi_uncorrelated_gaussians_three_dims_gaussian()
    test_mi_uncorrelated_gaussians_three_dims_discrete()
    # cmi
    test_cmi_correlated_gaussians_kraskov()
    test_cmi_correlated_gaussians_gaussian()
    test_cmi_correlated_gaussians_discrete()
    test_cmi_uncorrelated_gaussians_kraskov()
    test_cmi_uncorrelated_gaussians_gaussian()
    test_cmi_uncorrelated_gaussians_discrete()
    test_cmi_uncorrelated_gaussians_unequal_dims_kraskov()
    test_cmi_uncorrelated_gaussians_unequal_dims_gaussian()
    test_cmi_uncorrelated_gaussians_unequal_dims_discrete()
    test_cmi_uncorrelated_gaussians_three_dims_kraskov()
    test_cmi_uncorrelated_gaussians_three_dims_gaussian()
    test_cmi_uncorrelated_gaussians_three_dims_discrete()
    test_cmi_no_cond_correlated_gaussians_kraskov()
    test_cmi_no_cond_correlated_gaussians_gaussian()
    test_cmi_no_cond_correlated_gaussians_discrete()
    # ais
    test_ais_gaussian()
    test_ais_discrete()
    # te
    test_te_correlated_gaussians_gaussian()
    test_te_correlated_gaussians_discrete()
    # cte
    test_cte_correlated_gaussians_gaussian()
    test_cte_uncorrelated_gaussians_gaussian()
    test_cte_no_cond_correlated_gaussians_gaussian()

    test_local_values_kraskov()
    test_local_values_gaussian()
    test_local_values_discrete()

    # only kraskov
    test_amd_data_padding()
    test_mi_correlated_gaussians_two_chunks()
    test_debug_setting()
    test_insufficient_no_points

    test_multi_gpu()


