import numpy as np
import time

import pytest

from idtxl.estimators_jidt import (JidtKraskovMI, 
                                JidtKraskovCMI, 
                                JidtKraskovTE, 
                                JidtKraskovAIS,
                                JidtGaussianMI, 
                                JidtGaussianCMI, 
                                JidtGaussianTE, 
                                JidtGaussianAIS,
                                JidtDiscreteMI)

from idtxl.estimators_python import (PythonKraskovMI, 
                                PythonKraskovCMI, 
                                PythonKraskovTE,  
                                PythonKraskovAIS,
                                PythonGaussianMI, 
                                PythonGaussianCMI, 
                                PythonGaussianTE, 
                                PythonGaussianAIS,
                                PythonDiscreteMI)

from idtxl.idtxl_utils import calculate_mi
import random as rn

SEED = 42


def _compute_gaussian_mi(Sigma, s_dim, t_dim):
    SigmaS = Sigma[:s_dim, :s_dim]
    SigmaT = Sigma[s_dim : s_dim + t_dim, s_dim : s_dim + t_dim]

    I = 0.5 * np.log(
        np.linalg.det(SigmaS) * np.linalg.det(SigmaT) / np.linalg.det(Sigma)
    )

    return I


def _compute_gaussian_cmi(Sigma, s_dim, t_dim, c_dim):
    I_S_TC = _compute_gaussian_mi(Sigma, s_dim, t_dim + c_dim)

    Sigma_T_C = Sigma[s_dim:, s_dim:]
    I_T_C = _compute_gaussian_mi(Sigma_T_C, t_dim, c_dim)

    return I_S_TC - I_T_C

_Sigmas_2var = np.array(
    [
        # Test one: No corr. between S and T
        [[1, 0], [0, 1]],
        # Test two: Some corr. between S and T
        [[1, 0.5], [0.5, 1]],
        # Test three: Strong corr. between S and T
        [[1, 0.99], [0.99, 1]],
    ]
)

_Sigmas_3var = np.array(
    [
        # Test one: Strong corr. between S and T,
        # no corr. between S and C,
        # no corr. between T and C
        [[1, 0.99, 0], [0.99, 1, 0], [0, 0, 1]],
        # Test two: Strong corr. between S and T,
        # some corr. between S and C,
        # some corr. between T and C
        [[1, 0.99, 0.5], [0.99, 1, 0.5], [0.5, 0.5, 1]],
        # Test three: Strong corr. between S and T,
        # strong corr. between S and C,
        # strong corr. between T and C
        [[1, 0.99, 0.99], [0.99, 1, 0.99], [0.99, 0.99, 1]],
    ]
)


def _get_gauss_data(n=10000, covariance=0.4, expand=True, seed=None):
    """Generate correlated and uncorrelated Gaussian variables.

    Generate two sets of random normal data, where one set has a given
    covariance and the second is uncorrelated.
    """
    np.random.seed(seed)
    corr_expected = covariance / (1 * np.sqrt(covariance**2 + (1-covariance)**2))
    expected_mi = calculate_mi(corr_expected)
    src_corr = [rn.normalvariate(0, 1) for r in range(n)]  # correlated src
    src_uncorr = [rn.normalvariate(0, 1) for r in range(n)]  # uncorrelated src
    target = [sum(pair) for pair in zip(
                    [covariance * y for y in src_corr[0:n]],
                    [(1-covariance) * y for y in [
                        rn.normalvariate(0, 1) for r in range(n)]])]
    # Make everything numpy arrays so jpype understands it. Add an additional
    # axis if requested (MI/CMI estimators accept 2D arrays, TE/AIS only 1D).
    if expand:
        src_corr = np.expand_dims(np.array(src_corr), axis=1)
        src_uncorr = np.expand_dims(np.array(src_uncorr), axis=1)
        target = np.expand_dims(np.array(target), axis=1)
    else:
        src_corr = np.array(src_corr)
        src_uncorr = np.array(src_uncorr)
        target = np.array(target)
    return expected_mi, src_corr, src_uncorr, target


def _get_ar_data(n=10000, expand=False, seed=None):
    """Simulate a process with memory using an AR process of order 2.

    Return data with memory and random data without memory.
    """
    order = 2
    source1 = np.zeros(n + order)
    source1[0:order] = np.random.normal(size=(order))
    term_1 = 0.95 * np.sqrt(2)
    for n in range(order, n + order):
        source1[n] = (term_1 * source1[n - 1] - 0.9025 * source1[n - 2] +
                      np.random.normal())
    source2 = np.random.randn(n + order)
    if expand:
        return np.expand_dims(source1, axis=1), np.expand_dims(source2, axis=1)
    else:
        return source1, source2


def _get_mem_binary_data(n=10000, expand=False):
    """Simulate simple binary process with memory.

    Return data with memory and random data without memory.
    """
    source1 = np.zeros(n + 2)
    source1[0:2] = np.random.randint(2, size=(2))
    for n in range(2, n + 2):
        source1[n] = np.logical_xor(source1[n - 1], np.random.rand() > 0.15)
    source1 = source1.astype(int)
    source2 = np.random.randint(2, size=(n + 2))
    if expand:
        return np.expand_dims(source1, axis=1), np.expand_dims(source2, axis=1)
    else:
        return source1, source2


@pytest.mark.parametrize("Sigma", _Sigmas_3var)
def test_kraskov_cmi_gaussian(Sigma):
    rng = np.random.default_rng(SEED)
    S, T, C = rng.multivariate_normal(np.zeros(3), Sigma, 10_000).T
    S, T, C = S[:, np.newaxis], T[:, np.newaxis], C[:, np.newaxis]

    cmi_gaussian = _compute_gaussian_cmi(Sigma, 1, 1, 1)

    print(f"\nAnalytical CMI: {cmi_gaussian}")

    # Run JIDT estimator as a reference

    jidt_estimator = JidtKraskovCMI(
        {"kraskov_k": 4, "noise_level": 0, "num_threads": 1}
    )

    itic = time.perf_counter()
    cmi_jidt = jidt_estimator.estimate(var1=S, var2=T, conditional=C)
    itoc = time.perf_counter()

    print(f"JidtKraskovCMI: {cmi_jidt} (took {itoc - itic} seconds)")
    assert np.isclose(cmi_gaussian, cmi_jidt, rtol=0.08)

    # Run Python estimators with different knn_finders

    python_estimator = PythonKraskovCMI(
        {"kraskov_k": 4, "noise_level": 0, "knn_finder": "scipy_kdtree"}
    )

    itic = time.perf_counter()
    cmi_python = python_estimator.estimate(var1=S, var2=T, conditional=C)
    itoc = time.perf_counter()

    print(f"PythonKraskovCMI (scipy_kdtree): {cmi_python} (took {itoc - itic} seconds)")
    assert np.isclose(cmi_jidt, cmi_python, rtol=1e-4)
    #assert np.isclose(cmi_gaussian, cmi_python, rtol=0.08)


    python_estimator = PythonKraskovCMI(
        {"kraskov_k": 4, "noise_level": 0, "knn_finder": "scipy_ckdtree"}
    )

    itic = time.perf_counter()
    cmi_python = python_estimator.estimate(var1=S, var2=T, conditional=C)
    itoc = time.perf_counter()

    print(f"PythonKraskovCMI (scipy_ckdtree): {cmi_python} (took {itoc - itic} seconds)")
    assert np.isclose(cmi_jidt, cmi_python, rtol=1e-4)
    #assert np.isclose(cmi_gaussian, cmi_python, rtol=0.08)


    python_estimator = PythonKraskovCMI(
        {"kraskov_k": 4, "noise_level": 0, "knn_finder": "sklearn_kdtree"}
    )

    itic = time.perf_counter()
    cmi_python = python_estimator.estimate(var1=S, var2=T, conditional=C)
    itoc = time.perf_counter()

    print(f"PythonKraskovCMI (sklearn_kdtree): {cmi_python} (took {itoc - itic} seconds)")
    assert np.isclose(cmi_jidt, cmi_python, rtol=1e-4)
    #assert np.isclose(cmi_gaussian, cmi_python, rtol=0.08)

    python_estimator = PythonKraskovCMI(
        {"kraskov_k": 4, "noise_level": 0, "knn_finder": "sklearn_balltree"}
    )

    itic = time.perf_counter()
    cmi_python = python_estimator.estimate(var1=S, var2=T, conditional=C)
    itoc = time.perf_counter()

    print(f"PythonKraskovCMI (sklearn_balltree): {cmi_python} (took {itoc - itic} seconds)")
    assert np.isclose(cmi_jidt, cmi_python, rtol=1e-4)
    #assert np.isclose(cmi_gaussian, cmi_python, rtol=0.08)


    python_estimator = PythonKraskovCMI({'kraskov_k':4, 'noise_level':0, 'knn_finder':'numba_brute'})

    itic = time.perf_counter()
    cmi_python = python_estimator.estimate(var1=S, var2=T, conditional=C)
    itoc = time.perf_counter()

    print(f'PythonKraskovCMI (numba_brute): {cmi_python} (took {itoc - itic} seconds)')
    assert np.isclose(cmi_jidt, cmi_python, rtol=1e-4)
    #assert np.isclose(cmi_gaussian, cmi_python, rtol=0.08)


@pytest.mark.parametrize("Sigma", _Sigmas_2var)
def test_kraskov_mi_gaussian(Sigma):
    rng = np.random.default_rng(SEED)
    S, T = rng.multivariate_normal(np.zeros(2), Sigma, 10_000).T
    S, T = S[:, np.newaxis], T[:, np.newaxis]

    mi_gaussian = _compute_gaussian_mi(Sigma, 1, 1)
    print(f"\nAnalytical MI: {mi_gaussian}")

    # Run JIDT estimator as a reference

    jidt_estimator = JidtKraskovMI(
        {"kraskov_k": 4, "noise_level": 0, "num_threads": 1}
    )

    itic = time.perf_counter()
    mi_jidt = jidt_estimator.estimate(var1=S, var2=T)
    itoc = time.perf_counter()

    print(f"JidtKraskovMI: {mi_jidt} (took {itoc - itic} seconds)")
    assert np.isclose(mi_gaussian, mi_jidt, rtol=0.08, atol=0.01)

    # Run Python estimators with different knn_finders

    python_estimator = PythonKraskovMI(
        {"kraskov_k": 4, "noise_level": 0, "num_threads": 1, "knn_finder": "scipy_kdtree"}
    )

    itic = time.perf_counter()
    mi_python = python_estimator.estimate(var1=S, var2=T)
    itoc = time.perf_counter()

    print(f"PythonKraskovMI (scipy_kdtree): {mi_python} (took {itoc - itic} seconds)")
    assert np.isclose(mi_jidt, mi_python, rtol=1e-4, atol=1e-4)

    python_estimator = PythonKraskovMI(
        {"kraskov_k": 4, "noise_level": 0, "num_threads": 1, "knn_finder": "scipy_ckdtree"}
    )

    itic = time.perf_counter()
    mi_python = python_estimator.estimate(var1=S, var2=T)
    itoc = time.perf_counter()

    print(f"PythonKraskovMI (scipy_ckdtree): {mi_python} (took {itoc - itic} seconds)")
    assert np.isclose(mi_jidt, mi_python, rtol=1e-4, atol=1e-4)


    python_estimator = PythonKraskovMI(
        {"kraskov_k": 4, "noise_level": 0, "num_threads": 1, "knn_finder": "sklearn_kdtree"}
    )

    itic = time.perf_counter()
    mi_python = python_estimator.estimate(var1=S, var2=T)
    itoc = time.perf_counter()

    print(f"PythonKraskovMI (sklearn_kdtree): {mi_python} (took {itoc - itic} seconds)")
    assert np.isclose(mi_jidt, mi_python, rtol=1e-4, atol=1e-4)

    python_estimator = PythonKraskovMI(
        {"kraskov_k": 4, "noise_level": 0, "num_threads": 1, "knn_finder": "sklearn_balltree"}
    )

    itic = time.perf_counter()
    mi_python = python_estimator.estimate(var1=S, var2=T)
    itoc = time.perf_counter()

    print(f"PythonKraskovMI (sklearn_balltree): {mi_python} (took {itoc - itic} seconds)")
    assert np.isclose(mi_jidt, mi_python, rtol=1e-4, atol=1e-4)

    python_estimator = PythonKraskovMI(
        {'kraskov_k':4, 'noise_level':0, "num_threads": 1, 'knn_finder':'numba_brute'}
    )

    itic = time.perf_counter()
    mi_python = python_estimator.estimate(var1=S, var2=T)
    itoc = time.perf_counter()

    print(f'PythonKraskovMI (numba_brute): {mi_python} (took {itoc - itic} seconds)')
    assert np.isclose(mi_jidt, mi_python, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("Sigma", _Sigmas_2var)
def test_gaussian_mi_gaussian(Sigma):
    rng = np.random.default_rng(SEED)
    S, T = rng.multivariate_normal(np.zeros(2), Sigma, 10_000).T
    S, T = S[:, np.newaxis], T[:, np.newaxis]

    mi_gaussian = _compute_gaussian_mi(Sigma, 1, 1)
    print(f"\nAnalytical MI: {mi_gaussian}")

    # Run JIDT estimator as a reference

    jidt_estimator = JidtGaussianMI(
        {"kraskov_k": 4, "noise_level": 0, "num_threads": 1}
    )

    itic = time.perf_counter()
    mi_jidt = jidt_estimator.estimate(var1=S, var2=T)
    itoc = time.perf_counter()

    print(f"JidtGaussianMI: {mi_jidt} (took {itoc - itic} seconds)")
    assert np.isclose(mi_gaussian, mi_jidt, rtol=0.08, atol=0.01)

    # Run Python estimators with different knn_finders

    python_estimator = PythonGaussianMI( 
        {"kraskov_k": 4, "noise_level": 0, "num_threads": 1}
    )

    itic = time.perf_counter()
    mi_python = python_estimator.estimate(var1=S, var2=T)
    itoc = time.perf_counter()

    print(f"PythonGaussianMI: {mi_python} (took {itoc - itic} seconds)")
    assert np.isclose(mi_jidt, mi_python, rtol=1e-4, atol=1e-4)
 

@pytest.mark.parametrize("Sigma", _Sigmas_3var)
def test_gaussian_cmi_gaussian(Sigma):
    rng = np.random.default_rng(SEED)
    S, T, C = rng.multivariate_normal(np.zeros(3), Sigma, 10_000).T
    S, T, C = S[:, np.newaxis], T[:, np.newaxis], C[:, np.newaxis]

    cmi_gaussian = _compute_gaussian_cmi(Sigma, 1, 1, 1)

    print(f"\nAnalytical CMI: {cmi_gaussian}")

    # Run JIDT estimator as a reference

    jidt_estimator = JidtGaussianCMI(
        {"kraskov_k": 4, "noise_level": 0, "num_threads": 1}
    )

    itic = time.perf_counter()
    cmi_jidt = jidt_estimator.estimate(var1=S, var2=T, conditional=C)
    itoc = time.perf_counter()

    print(f"JidtGaussianCMI: {cmi_jidt} (took {itoc - itic} seconds)")
    assert np.isclose(cmi_gaussian, cmi_jidt, rtol=0.08)

    python_estimator = PythonGaussianCMI(
        {"noise_level": 0, "num_threads": 1}
    )

    itic = time.perf_counter()
    cmi_python = python_estimator.estimate(var1=S, var2=T, conditional=C)
    itoc = time.perf_counter()

    print(f"PythonGaussianCMI: {cmi_python} (took {itoc - itic} seconds)")
    assert np.isclose(cmi_jidt, cmi_python, rtol=1e-4)


############################################################################################### TODO
def test_gaussian_ais():
    """Test AIS estimation on an autoregressive process. """

    source1, source2 = _get_ar_data(seed=SEED)

    settings = {'history': 2}

    # Run Jidt estimator as reference
    jidt_estimator = JidtGaussianAIS(settings=settings)
    itic = time.perf_counter()
    jidt_mi_cor = jidt_estimator.estimate(source1)
    itoc = time.perf_counter()
    print(f"JidtGaussianAIS (cor) : {jidt_mi_cor} (took {itoc - itic} seconds)")

    itic = time.perf_counter()
    jidt_mi_uncor = jidt_estimator.estimate(source2)
    itoc = time.perf_counter()
    print(f"JidtGaussianAIS (uncor) : {jidt_mi_uncor} (took {itoc - itic} seconds)")

    # Run Python estimator
    python_estimator = PythonGaussianAIS(settings=settings)
    itic = time.perf_counter()
    python_mi_cor = python_estimator.estimate(source1)
    itoc = time.perf_counter()
    print(f"PythonGaussianAIS (cor) : {python_mi_cor} (took {itoc - itic} seconds)")

    itic = time.perf_counter()
    python_mi_uncor = python_estimator.estimate(source2)
    itoc = time.perf_counter()
    print(f"PythonGaussianAIS (uncor) : {python_mi_uncor} (took {itoc - itic} seconds)")

    assert np.isclose(jidt_mi_cor, python_mi_cor, rtol=1e-4)
    #assert np.isclose(jidt_mi_uncor, python_mi_uncor, rtol=1e-3)


############################################################################################### TODO
def test_kraskov_ais():
    """Test AIS estimation on an autoregressive process. """

    source1, source2 = _get_ar_data(seed=SEED)

    settings_j = {'history': 2}
    settings_p = {'history': 2}


    # Run Jidt estimator as reference
    jidt_estimator = JidtKraskovAIS(settings=settings_j)
    itic = time.perf_counter()
    jidt_mi_cor = jidt_estimator.estimate(source1)
    itoc = time.perf_counter()
    print(f"JidtKraskovAIS (cor) : {jidt_mi_cor} (took {itoc - itic} seconds)")

    itic = time.perf_counter()
    jidt_mi_uncor = jidt_estimator.estimate(source2)
    itoc = time.perf_counter()
    print(f"JidtKraskovAIS (uncor) : {jidt_mi_uncor} (took {itoc - itic} seconds)")


    # Run Python estimator
    python_estimator = PythonKraskovAIS(settings=settings_p)
    itic = time.perf_counter()
    python_mi_cor = python_estimator.estimate(source1)
    itoc = time.perf_counter()
    print(f"PythonKraskovAIS (cor) : {python_mi_cor} (took {itoc - itic} seconds)")

    itic = time.perf_counter()
    python_mi_uncor = python_estimator.estimate(source2)
    itoc = time.perf_counter()
    print(f"PythonKraskovAIS (uncor) : {python_mi_uncor} (took {itoc - itic} seconds)")

    assert np.isclose(jidt_mi_cor, python_mi_cor, rtol=1e-4)
    #assert np.isclose(jidt_mi_uncor, python_mi_uncor, rtol=1e-3)

    
############################################################################################### TODO
def test_kraskov_te_gaussian():
    
    expected_mi, source1, source2, target = _get_gauss_data(expand=False, seed=SEED)
    # add delay of one sample
    source1 = source1[1:]
    source2 = source2[1:]
    target = target[:-1]
    
    ht=2
    tt=2
    hs=2
    ts=2
    hst=1

    print(f"\nAnalytical MI: {expected_mi}")

    # Run JIDT estimator as a reference
    settings = {"kraskov_k": 4, "history_target": ht, "history_source": hs, "tau_target": tt,
                            "tau_source": ts, "source_target_delay": hst ,"noise_level": 0, "num_threads": 1}
    jidt_estimator = JidtKraskovTE(settings)

    itic = time.perf_counter()
    te_jidt = jidt_estimator.estimate(source=source1, target=target)
    itoc = time.perf_counter()

    print(f"\nJidtKraskovTE: {te_jidt} (took {itoc - itic} seconds)")

    """
    settings = {"kraskov_k": 4, "history_target": 1, "noise_level": 0, "num_threads": 1, "algorithm_num": 2}
    jidt_estimator = JidtKraskovTE(settings)

    itic = time.perf_counter()
    te_jidt2 = jidt_estimator.estimate(source=source1, target=target)
    itoc = time.perf_counter()

    print(f"\nJidtKraskovTE (alg2): {te_jidt2} (took {itoc - itic} seconds)")
    """

    settings = {"kraskov_k": 4, "history_target": ht, "history_source": hs, "tau_target": tt,
                            "tau_source": ts, "source_target_delay": hst ,"noise_level": 0, "num_threads": 1, "knn_finder": "scipy_kdtree"}
    #print(settings)
    python_estimator = PythonKraskovTE(settings)

    itic = time.perf_counter()
    te_python = python_estimator.estimate(source=source1, target=target)
    itoc = time.perf_counter()

    print(f"PythonKraskovTE (scipy_kdtree): {te_python} (took {itoc - itic} seconds)")
    assert np.isclose(te_jidt, te_python, rtol=1e-4, atol=1e-4)
    """
    settings = {"kraskov_k": 4, "history_target": ht, "noise_level": 0, "num_threads": 1, "knn_finder": "scipy_ckdtree"}
    #print(settings)
    python_estimator = PythonKraskovTE(settings)

    itic = time.perf_counter()
    te_python = python_estimator.estimate(source=source1, target=target)
    itoc = time.perf_counter()

    print(f"PythonKraskovTE (scipy_ckdtree): {te_python} (took {itoc - itic} seconds)")
    assert np.isclose(te_jidt, te_python, rtol=1e-4, atol=1e-4)

    settings = {"kraskov_k": 4, "history_target": ht, "noise_level": 0, "num_threads": 1, "knn_finder": "sklearn_kdtree"}
    python_estimator = PythonKraskovTE(settings)

    itic = time.perf_counter()
    te_python = python_estimator.estimate(source=source1, target=target)
    itoc = time.perf_counter()

    print(f"PythonKraskovTE (sklearn_kdtree): {te_python} (took {itoc - itic} seconds)")
    assert np.isclose(te_jidt, te_python, rtol=1e-4, atol=1e-4)
    
    settings = {"kraskov_k": 4, "history_target": ht, "noise_level": 0, "num_threads": 1, "knn_finder": "sklearn_balltree"}
    python_estimator = PythonKraskovTE(settings)

    itic = time.perf_counter()
    te_python = python_estimator.estimate(source=source1, target=target)
    itoc = time.perf_counter()

    print(f"PythonKraskovTE (sklearn_balltree): {te_python} (took {itoc - itic} seconds)")
    assert np.isclose(te_jidt, te_python, rtol=1e-4, atol=1e-4)

    settings = {"kraskov_k": 4, "history_target": ht, "noise_level": 0, "num_threads": 1, "knn_finder": "numba_brute"}
    python_estimator = PythonKraskovTE(settings)

    itic = time.perf_counter()
    te_python = python_estimator.estimate(source=source1, target=target)
    itoc = time.perf_counter()

    print(f"PythonKraskovTE (numba_brute): {te_python} (took {itoc - itic} seconds)")
    assert np.isclose(te_jidt, te_python, rtol=1e-4, atol=1e-4)
    """

############################################################################################### TODO
def test_gaussian_te_gaussian():

    expected_mi, source1, source2, target = _get_gauss_data(expand=False, seed=SEED)
    # add delay of one sample
    source1 = source1[1:]
    source2 = source2[1:]
    target = target[:-1]
    settings = {"kraskov_k": 4, "history_target": 1, "noise_level": 0, "num_threads": 1}

    print(f"\nAnalytical MI: {expected_mi}")

    # Run JIDT estimator as a reference
    jidt_estimator = JidtGaussianTE(settings)

    itic = time.perf_counter()
    te_jidt = jidt_estimator.estimate(source=source1, target=target)
    itoc = time.perf_counter()

    print(f"\nJidtGaussianTE: {te_jidt} (took {itoc - itic} seconds)")

    settings = {"kraskov_k": 4, "history_target": 1, "noise_level": 0, "num_threads": 1, "knn_finder": "scipy_kdtree"}
    python_estimator = PythonGaussianTE(settings)

    itic = time.perf_counter()
    te_python = python_estimator.estimate(source=source1, target=target)
    itoc = time.perf_counter()

    print(f"PythonGaussianTE (scipy_kdtree): {te_python} (took {itoc - itic} seconds)")
    assert np.isclose(te_jidt, te_python, rtol=1e-4, atol=1e-4)
    
    settings = {"kraskov_k": 4, "history_target": 1, "noise_level": 0, "num_threads": 1, "knn_finder": "scipy_ckdtree"}
    python_estimator = PythonGaussianTE(settings)

    itic = time.perf_counter()
    te_python = python_estimator.estimate(source=source1, target=target)
    itoc = time.perf_counter()

    print(f"PythonGaussianTE (scipy_ckdtree): {te_python} (took {itoc - itic} seconds)")
    assert np.isclose(te_jidt, te_python, rtol=1e-4, atol=1e-4)
    
    settings = {"kraskov_k": 4, "history_target": 1, "noise_level": 0, "num_threads": 1, "knn_finder": "sklearn_kdtree"}
    python_estimator = PythonGaussianTE(settings)

    itic = time.perf_counter()
    te_python = python_estimator.estimate(source=source1, target=target)
    itoc = time.perf_counter()

    print(f"PythonGaussianTE (sklearn_kdtree): {te_python} (took {itoc - itic} seconds)")
    assert np.isclose(te_jidt, te_python, rtol=1e-4, atol=1e-4)
    
    settings = {"kraskov_k": 4, "history_target": 1, "noise_level": 0, "num_threads": 1, "knn_finder": "sklearn_balltree"}
    python_estimator = PythonGaussianTE(settings)

    itic = time.perf_counter()
    te_python = python_estimator.estimate(source=source1, target=target)
    itoc = time.perf_counter()

    print(f"PythonGaussianTE (sklearn_balltree): {te_python} (took {itoc - itic} seconds)")
    assert np.isclose(te_jidt, te_python, rtol=1e-4, atol=1e-4)

    settings = {"kraskov_k": 4, "history_target": 1, "noise_level": 0, "num_threads": 1, "knn_finder": "numba_brute"}
    python_estimator = PythonGaussianTE(settings)

    itic = time.perf_counter()
    te_python = python_estimator.estimate(source=source1, target=target)
    itoc = time.perf_counter()

    print(f"PythonGaussianTE (numba_brute): {te_python} (took {itoc - itic} seconds)")
    assert np.isclose(te_jidt, te_python, rtol=1e-4, atol=1e-4)
    








############################################################################################################ TODO
def test_discrete_mi_gaussian():
    expected_mi, source1, source2, target = _get_gauss_data(expand=False, seed=SEED)
    # add delay of one sample
    source1 = source1[1:]
    source2 = source2[1:]
    target = target[:-1]
    settings = {'discretise_method': 'equal',
                'n_discrete_bins': 4,
                'history_target': 1,
                'noise_level': 0,}

    print(f"\nAnalytical MI: {expected_mi}")

    # Run JIDT estimator as a reference

    jidt_estimator = JidtDiscreteMI(settings=settings)

    itic = time.perf_counter()
    mi_jidt = jidt_estimator.estimate(source1, target)
    itoc = time.perf_counter()

    print(f"JidtDiscreteMI: {mi_jidt} (took {itoc - itic} seconds)")
    #assert np.isclose(expected_mi, mi_jidt, rtol=0.08, atol=0.01)

    # Run Python estimator

    python_estimator = PythonDiscreteMI(settings=settings)

    itic = time.perf_counter()
    mi_python = python_estimator.estimate(source1, target)
    itoc = time.perf_counter()

    print(f"PythonDisreteMI: {mi_python} (took {itoc - itic} seconds)")
    #assert np.isclose(mi_jidt, mi_python, rtol=1e-4, atol=1e-4)




############################################################################################################ TODO
def test_kraskov_theiler_t():
    expected_mi, source1, source2, target = _get_gauss_data(expand=False, seed=SEED)
    # add delay of one sample
    source1_l = source1[1:]
    source2_l = source2[1:]
    target_l = target[:-1]
    settings = {"kraskov_k": 4, "history_target": 1, "theiler_t": 1, "noise_level": 0, "num_threads": 1}


    # Run JIDT Kraskov MI estimator as a reference

    jidt_estimator = JidtKraskovMI(settings)

    itic = time.perf_counter()
    mi_jidt = jidt_estimator.estimate(var1=source1, var2=target)
    itoc = time.perf_counter()

    print(f"JidtKraskovMI (theiler_t = 1): {mi_jidt} (took {itoc - itic} seconds)")
    assert np.isclose(expected_mi, mi_jidt, rtol=0.08, atol=0.01)
    
    # Run Python Kraskov MI estimator

    python_estimator = PythonKraskovMI(settings)

    itic = time.perf_counter()
    mi_python = python_estimator.estimate(var1=source1, var2=target)
    itoc = time.perf_counter()

    print(f"PythonKraskovMI (theiler_t = 1): {mi_python} (took {itoc - itic} seconds)")
    assert np.isclose(mi_python, mi_jidt, rtol=0.08, atol=0.01)
    

    # Run JIDT Kraskov CMI estimator as a reference

    jidt_estimator = JidtKraskovCMI(settings)

    itic = time.perf_counter()
    mi_jidt = jidt_estimator.estimate(var1=source1, var2=target, conditional=source2)
    itoc = time.perf_counter()

    print(f"JidtKraskovCMI (theiler_t = 1): {mi_jidt} (took {itoc - itic} seconds)")
    assert np.isclose(expected_mi, mi_jidt, rtol=0.08, atol=0.01)
    
    # Run Python Kraskov CMI estimator

    python_estimator = PythonKraskovCMI(settings)

    itic = time.perf_counter()
    mi_python = python_estimator.estimate(var1=source1, var2=target, conditional=source2)
    itoc = time.perf_counter()

    print(f"PythonKraskovCMI (theiler_t = 1): {mi_python} (took {itoc - itic} seconds)")
    assert np.isclose(mi_python, mi_jidt, rtol=0.08, atol=0.01)


    """
    # Run JIDT Kraskov TE estimator as a reference

    jidt_estimator = JidtKraskovTE(settings)

    itic = time.perf_counter()
    mi_jidt = jidt_estimator.estimate(source1, target)
    itoc = time.perf_counter()

    print(f"JidtKraskovTE (theiler_t = 1): {mi_jidt} (took {itoc - itic} seconds)")
    assert np.isclose(expected_mi, mi_jidt, rtol=0.08, atol=0.01)
    
    # Run Python Kraskov TE estimator

    python_estimator = PythonKraskovTE(settings)

    itic = time.perf_counter()
    mi_python = python_estimator.estimate(source1, target)
    itoc = time.perf_counter()

    print(f"PythonKraskovTE (theiler_t = 1): {mi_python} (took {itoc - itic} seconds)")
    assert np.isclose(mi_python, mi_jidt, rtol=0.08, atol=0.01)
    """




if __name__ == '__main__':
    
    
    #print("\n\nTest Kraskov MI:\n")
    #for sigma in _Sigmas_2var:
    #    test_kraskov_mi_gaussian(sigma)

    #print("\n\nTest Kraskov CMI:\n")
    #for sigma in _Sigmas_3var:
    #    test_kraskov_cmi_gaussian(sigma)
        
    #print("\n\nTest Kraskov TE:\n")
    #test_kraskov_te_gaussian()

    #print("\n\nTest Kraskov Theiler_T correction:\n") ################################################## TODO
    #test_kraskov_theiler_t()

    #print("\n\nTest Kraskov AIS:\n")
    #test_kraskov_ais()
    
    #print("\n\nTest Gaussian MI:\n")
    #for sigma in _Sigmas_2var:
    #    test_gaussian_mi_gaussian(sigma)
    
    print("\n\nTest Gaussian CMI:\n")
    for sigma in _Sigmas_3var:
        test_gaussian_cmi_gaussian(sigma)

    #print("\n\nTest Gaussian TE:\n")
    #test_gaussian_te_gaussian()
    
    #print("\n\nTest Gaussian AIS:\n")
    #test_gaussian_ais()
    


    
    
    #print("\n\nTest Discrete MI:\n") ################################################## TODO
    #test_discrete_mi_gaussian()


    
    print("All tests passed.")
