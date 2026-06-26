



import numpy as np

import time
import sys

from idtxl.estimators_jidt import JidtKraskovMI, JidtKraskovCMI, JidtKraskovTE, JidtKraskovAIS, JidtGaussianMI, JidtGaussianCMI, JidtGaussianTE, JidtGaussianCTE, JidtGaussianAIS, JidtDiscreteMI, JidtDiscreteCMI , JidtDiscreteAIS, JidtDiscreteTE
from idtxl.estimators_python import PythonKraskovMI, PythonKraskovCMI, PythonKraskovTE, PythonKraskovAIS, PythonGaussianMI, PythonGaussianCMI, PythonGaussianTE, PythonGaussianCTE, PythonGaussianAIS, PythonDiscreteMI, PythonDiscreteCMI, PythonDiscreteAIS, PythonDiscreteTE, PythonSpectralMI

from idtxl.idtxl_utils import calculate_mi
import random as rn


SEED = 42

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


def _get_freq_data(sample_rate=10000, duration=1.0, seed=0):
	"""Generate correlated and uncorrelated Frequency variables."""
	
	np.random.seed(seed)

	n_samples = int(sample_rate * duration)
	t = np.linspace(0, duration, n_samples, endpoint=False)
	
	# Signal 1: 50 Hz sine + noise
	signal1 = np.sin(2 * np.pi * 40 * t) + 0.2 * np.random.randn(n_samples)
	
	# Signal 2: 50 Hz sine (correlated) + noise
	signal2 = np.sin(2 * np.pi * 40 * t) + 0.3 * np.random.randn(n_samples)
	
	# Signal 2: 50 Hz sine (correlated) + noise
	signal3 = np.sin(2 * np.pi * 23 * t) + 0.9 * np.random.randn(n_samples)
	
	return signal1, signal2, signal3


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
        return np.expand_dims(source1[:n], axis=1), np.expand_dims(source2[:n], axis=1)
    else:
        return source1[:n], source2[:n]


def _generate_mute_data(n_samples=10000, n_replications=1):
        """Generate example data for a 6-process network.
		
		

        0 -> 1, u = 2 (non-linear)
        0 -> 2, u = 3
        0 -> 3, u = 2 (non-linear)
        3 -> 4, u = 1
        4 -> 3, u = 1
		
		5 is pure noise
        """
        n_processes = 6

        x = np.zeros((n_processes, n_samples + 3, n_replications))
        x[:, 0:3, :] = np.random.normal(size=(n_processes, 3, n_replications))
        term_1 = 0.95 * np.sqrt(2)
        term_2 = 0.25 * np.sqrt(2)
        term_3 = -0.25 * np.sqrt(2)
        for r in range(n_replications):
            for n in range(3, n_samples + 3):
                x[0, n, r] = (
                    term_1 * x[0, n - 1, r]
                    - 0.9025 * x[0, n - 2, r]
                    + np.random.normal()
                )
                x[1, n, r] = 0.5 * x[0, n - 2, r] ** 2 + np.random.normal()
                x[2, n, r] = -0.4 * x[0, n - 3, r] + np.random.normal()
                x[3, n, r] = (
                    -0.5 * x[0, n - 2, r] ** 2
                    + term_2 * x[3, n - 1, r]
                    + term_2 * x[4, n - 1, r]
                    + np.random.normal()
                )
                x[4, n, r] = (
                    term_3 * x[3, n - 1, r]
                    + term_2 * x[4, n - 1, r]
                    + np.random.normal()
                )
                x[5, n, r] = np.random.normal()

        return x


def verbose(res_jidt, res_python, values, est, local=False):

	if local:
		addstring=" local"
	else:
		addstring=""

	if np.allclose(res_jidt, res_python, rtol=1e-04, atol=1e-04):

		print(f"{values} - all{addstring} {est} results within tolerance (atol and rtol=1e-04)")
	else:
		if np.allclose(res_jidt, res_python, rtol=1e-03, atol=1e-03):
			print(f"{values} - all{addstring} {est} results within tolerance (atol and rtol=1e-03)")
		else:
			diff = abs(res_jidt - res_python)
			num = (diff>1e-03).sum()
			print(f"{values} - {num}/{res_jidt.shape[0]} of{addstring} {est} results are not within tolerance (1e-03) !!!!!!")




# Test Gaussian estimators
def test_gaussian_mi():

	expected_mi, source1, source2, target = _get_gauss_data(expand=True, seed=SEED)
	

	mi_jidt_cor = np.zeros(4)
	mi_jidt_uncor = np.zeros(4)
	mi_python_cor = np.zeros(4)
	mi_python_uncor = np.zeros(4)
	time_jidt_cor = np.zeros(4)
	time_jidt_uncor = np.zeros(4)
	time_python_cor = np.zeros(4)
	time_python_uncor = np.zeros(4)

	vals = [0,1,2,3]

	for lags in vals:
		settings = {}
		settings = {"lag_mi": lags,
					"noise_level": 0}

		# cor
		jidt_estimator = JidtGaussianMI(settings)
		itic = time.perf_counter()
		mi_jidt_cor[lags] = jidt_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_jidt_cor[lags] = itoc - itic
		
		python_estimator = PythonGaussianMI(settings)
		itic = time.perf_counter()
		mi_python_cor[lags] = python_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_python_cor[lags] = itoc - itic
		
		# uncor
		jidt_estimator = JidtGaussianMI(settings)
		itic = time.perf_counter()
		mi_jidt_uncor[lags] = jidt_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		time_jidt_uncor[lags] = itoc - itic
		
		python_estimator = PythonGaussianMI(settings)
		itic = time.perf_counter()
		mi_python_uncor[lags] = python_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		time_python_uncor[lags] = itoc - itic

	print(f"Summary Jidt vs Python GaussianMI lags ({vals}):")

	print("MI values:")
	print("correlated data:")
	print("lag\tJidtGaussianMI\t\tPythonGaussianMI")
	for i in vals:
		print(f"{i}\t{mi_jidt_cor[i]}\t{mi_python_cor[i]}")
	
	verbose(mi_jidt_cor, mi_python_cor, "", "MI (cor)", False)
	

	print("uncorrelated data:")
	print("lag\tJidtGaussianMI\t\tPythonGaussianMI")
	for i in vals:
		print(f"{i}\t{mi_jidt_uncor[i]}\t{mi_python_uncor[i]}")

	verbose(mi_jidt_uncor, mi_python_uncor, "", "MI (uncor)", False)
	
	print("\nmean calculation times:")
	print(" JidtGaussianMI (cor): ", np.mean(time_jidt_cor) )
	print(" PythonGaussianMI (cor): ", np.mean(time_python_cor) )
	print(" JidtGaussianMI (uncor): ", np.mean(time_jidt_uncor) )
	print(" PythonGaussianMI (uncor): ", np.mean(time_python_uncor) )


def test_gaussian_mi_local_values():

	expected_mi, source1, source2, target = _get_gauss_data(expand=True, seed=SEED)

	vals = [0,1,2,3]

	for lags in vals:
		settings = {}
		settings = {"lag_mi": lags,
					"local_values": True,
					"noise_level": 0}
		
		jidt_estimator = JidtGaussianMI(settings)
		itic = time.perf_counter()
		mi_jidt = jidt_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		print(f"\nJidtGaussianMI local_values (cor) took {itoc - itic} seconds)")

		python_estimator = PythonGaussianMI(settings)
		itic = time.perf_counter()
		mi_python = python_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		print(f"PythonGaussianMI local_values (cor) took {itoc - itic} seconds)")

		verbose(mi_jidt, mi_python, "", "MI (cor)", True)

		jidt_estimator = JidtGaussianMI(settings)
		itic = time.perf_counter()
		mi_jidt = jidt_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		print(f"\nJidtGaussianMI local_values (uncor) took {itoc - itic} seconds)")
		
		python_estimator = PythonGaussianMI(settings)
		itic = time.perf_counter()
		mi_python = python_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		print(f"PythonGaussianMI local_values (uncor) took {itoc - itic} seconds)")
		
		verbose(mi_jidt, mi_python, "", "MI (uncor)", True)

	
def test_gaussian_cmi():

	for i in [0.2, 0.4, 0.6, 0.8]:

		print(f"\n\nGaussian CMI Test data with covariance: {i}")

		expected_mi, source1, source2, target = _get_gauss_data(expand=True, covariance=i, seed=SEED)

		settings={}
		time_python=0
		time_jidt=0

		res_jidt = np.zeros(2)
		res_python = np.zeros(2)

		jidt_estimator = JidtGaussianCMI(settings)
		python_estimator = PythonGaussianCMI(settings)
		
		
		itic = time.perf_counter()
		mi_jidt = jidt_estimator.estimate(source1, target, source2)
		itoc = time.perf_counter()
		res_jidt[0] = mi_jidt
		time_jidt += itoc - itic
		print(f"\nJidtGaussianCMI (uncor conditional): {mi_jidt} (took {itoc - itic} seconds)")

		itic = time.perf_counter()
		mi_python = python_estimator.estimate(source1, target, source2)
		itoc = time.perf_counter()
		res_python[0] = mi_python
		time_python += itoc - itic
		print(f"PythonGaussianCMI (uncor conditional): {mi_python} (took {itoc - itic} seconds)")

		itic = time.perf_counter()
		mi_jidt = jidt_estimator.estimate(source2, target, source1)
		itoc = time.perf_counter()
		res_jidt[1] = mi_jidt
		time_jidt += itoc - itic

		print(f"JidtGaussianCMI (uncor source): {mi_jidt} (took {itoc - itic} seconds)")
		itic = time.perf_counter()
		mi_python = python_estimator.estimate(source2, target, source1)
		itoc = time.perf_counter()
		res_python[1] = mi_python
		time_python += itoc - itic
		print(f"PythonGaussianCMI (uncor source): {mi_python} (took {itoc - itic} seconds)")

		verbose(res_jidt, res_python, "", "MI", False)


def test_gaussian_cmi_local_values():

	expected_mi, source1, source2, target = _get_gauss_data(expand=True, seed=SEED)

	settings={'local_values': True}
	time_python=0
	time_jidt=0

	jidt_estimator = JidtGaussianCMI(settings)
	python_estimator = PythonGaussianCMI(settings)
	
	
	itic = time.perf_counter()
	mi_jidt = jidt_estimator.estimate(source1, target, source2)
	itoc = time.perf_counter()
	print(f"\nJidtGaussianCMI local_values (uncor conditional) took {itoc - itic} seconds")

	itic = time.perf_counter()
	mi_python = python_estimator.estimate(source1, target, source2)
	itoc = time.perf_counter()
	print(f"PythonGaussianCMI local_values (uncor conditional) took {itoc - itic} seconds")

	verbose(mi_jidt, mi_python, "", "MI (uncor conditional)", True)
		

	itic = time.perf_counter()
	mi_jidt = jidt_estimator.estimate(source2, target, source1)
	itoc = time.perf_counter()
	print(f"\nJidtGaussianCMI local_values (uncor source) took {itoc - itic} seconds")

	itic = time.perf_counter()
	mi_python = python_estimator.estimate(source2, target, source1)
	itoc = time.perf_counter()
	print(f"PythonGaussianCMI local_values (uncor source) took {itoc - itic} seconds")

	verbose(mi_jidt, mi_python, "", "MI (uncor source)", True)


def test_gaussian_ais():

	source1, source2 = _get_ar_data(seed=SEED)

	vals =  [1,2,3]
	time_jidt_cor = np.zeros(np.power(len(vals),2))
	res_jidt_cor = np.zeros(np.power(len(vals),2))
	time_python_cor = np.zeros(np.power(len(vals),2))
	res_python_cor = np.zeros(np.power(len(vals),2))
	time_jidt_uncor = np.zeros(np.power(len(vals),2))
	res_jidt_uncor = np.zeros(np.power(len(vals),2))
	time_python_uncor = np.zeros(np.power(len(vals),2))
	res_python_uncor = np.zeros(np.power(len(vals),2))

	count = 0

	for h in vals:
		for t in vals:

			settings_j = {'history': h, 'tau': t}

			settings_p = {'history': h, 'tau': t}
	
			jidt_estimator = JidtGaussianAIS(settings=settings_j)
			python_estimator = PythonGaussianAIS(settings=settings_p)

			itic = time.perf_counter()
			res_jidt_cor[count] = jidt_estimator.estimate(source1)
			itoc = time.perf_counter()
			time_jidt_cor[count] = itoc - itic
	
			itic = time.perf_counter()
			res_jidt_uncor[count] = jidt_estimator.estimate(source2)
			itoc = time.perf_counter()
			time_jidt_uncor[count] = itoc - itic
			
			itic = time.perf_counter()
			res_python_cor[count] = python_estimator.estimate(source1)
			itoc = time.perf_counter()
			time_python_cor[count] = itoc - itic
			
			itic = time.perf_counter()
			res_python_uncor[count] = python_estimator.estimate(source2)
			itoc = time.perf_counter()
			time_python_uncor[count] = itoc - itic

			count += 1

	print("JidtGaussianAIS\t\tPythonGaussianAIS")
	print("AR with history")
	for i in range(len(res_jidt_cor)):
		print(f"{res_jidt_cor[i]}\t{res_python_cor[i]}")

	verbose(res_jidt_cor, res_python_cor, "", "AIS (with hist)", True)
	
	print("no history")
	for i in range(len(res_jidt_uncor)):
		print(f"{res_jidt_uncor[i]}\t{res_python_uncor[i]}")

	verbose(res_jidt_uncor, res_python_uncor, "", "AIS (no hist)", True)
	
	print("\nmean calculation times:")
	print(" JidtGaussianAIS (cor): ", np.mean(time_jidt_cor) )
	print(" PythonGaussianAIS (cor): ", np.mean(time_python_cor) )
	print(" JidtGaussianAIS (uncor): ", np.mean(time_jidt_uncor) )
	print(" PythonGaussianAIS (uncor): ", np.mean(time_python_uncor) )


def test_gaussian_ais_local_values():

	source1, source2 = _get_ar_data(seed=SEED)

	vals = [1,2,3]
	
	time_jidt = np.zeros(np.power(len(vals),2))
	time_python = np.zeros(np.power(len(vals),2))
	
	print("hist,tau\t\tJidtGaussianAIS\t\tPythonGaussianAIS\tclose")

	count = 0
	for h in vals:
		for t in vals:

			settings = {}
			settings_j = {'history': h, 
						'tau': t,
						'local_values': True}
			settings_p = {'history': h, 
						'tau': t,
						'knn_finder': "scipy_ckdtree",
						'local_values': True}
				
			jidt_estimator = JidtGaussianAIS(settings_j)
			python_estimator = PythonGaussianAIS(settings_p)

			itic = time.perf_counter()
			ais_jidt = jidt_estimator.estimate(source1)
			itoc = time.perf_counter()
			time_jidt[count] = itoc - itic
			
			itic = time.perf_counter()
			ais_python = python_estimator.estimate(source1)
			itoc = time.perf_counter()
			time_python[count] = itoc - itic
			
			verbose(ais_jidt, ais_python, [h, t], "AIS (with hist)", True)
					
			count += 1

	print("\nmean calculation times:")
	print(" JidtGaussianAIS: ", np.mean(time_jidt) )
	print(" PythonGaussianAIS: ", np.mean(time_python) )
	

def test_gaussian_te():

	expected_mi, source1, source2, target = _get_gauss_data(expand=True, seed=SEED)

	source1 = source1[1:]
	source2 = source2[1:]
	target = target[:-1]

	vals = [1,2,3]

	time_jidt = np.empty(np.power(len(vals),5))
	res_jidt = np.empty(np.power(len(vals),5))
	time_python = np.empty(np.power(len(vals),5))
	res_python = np.empty(np.power(len(vals),5))

	conds = np.empty((np.power(len(vals),5),5))

	print("hst,ht,tt,hs,ts\t\tJidtGaussianTE\t\tPythonGaussianTE\tclose")

	count = 0
	for hst in vals:

		for ht in vals:
			for tt in vals:
				for hs in vals:
					for ts in vals:

						conds[count,:] = [hst, ht, tt, hs, ts]
						settings_j = {"history_target": ht,
									"history_source": hs,
									"tau_target": tt,
									"tau_source": ts,
									"source_target_delay": hst}

						jidt_estimator = JidtGaussianTE(settings_j)
						
						itic = time.perf_counter()
						te_jidt = jidt_estimator.estimate(source=source1, target=target)
						itoc = time.perf_counter()

						time_jidt[count] = itoc-itic
						res_jidt[count] = te_jidt

						settings_p = {"history_target": ht,
									"history_source": hs,
									"tau_target": tt,
									"tau_source": ts,
									"source_target_delay": hst}

						python_estimator = PythonGaussianTE(settings_p)
						
						itic = time.perf_counter()
						te_python = python_estimator.estimate(source=source1, target=target)
						itoc = time.perf_counter()

						time_python[count] = itoc-itic
						res_python[count] = te_python
			
						print(f"{conds[count,:]}\t{te_jidt}\t{te_python}\t{np.isclose(te_jidt, te_python, rtol=1e-03, atol=1e-03)}")

						count += 1

						
	verbose(res_jidt, res_python, "", "TE")

	print("\nmean calculation times:")
	print(" JidtGaussianTE (cor): ", np.mean(time_jidt) )
	print(" PythonGaussianTE (cor): ", np.mean(time_python) )
	

def test_gaussian_cte():

	#i = 0.4
	#expected_mi, source1, source2, target = _get_gauss_data(expand=True, covariance=i, seed=SEED)

	data = _generate_mute_data(n_replications=1)
	source1 = data[0,:]
	target = data[4,:]
	cond = data[3,:]
	nocond = data[5,:]

	vals = [2,3]
	#vals = [1]

	time_jidt_cond = np.empty(np.power(len(vals),8))
	res_jidt_cond = np.empty(np.power(len(vals),8))
	time_jidt_nocond = np.empty(np.power(len(vals),8))
	res_jidt_nocond = np.empty(np.power(len(vals),8))
	
	time_python_cond = np.empty(np.power(len(vals),8))
	res_python_cond = np.empty(np.power(len(vals),8))
	time_python_nocond = np.empty(np.power(len(vals),8))
	res_python_nocond = np.empty(np.power(len(vals),8))
	

	conds = np.empty((np.power(len(vals),5),8))

	close = 1e-03

	print(f"\t\t\t\tJidtGaussianCTE\t\tPythonGaussianCTE\tclose {close}")
	print("hst,cst,ht,tt,hs,ts,hc,tc\tcte cond \t\tcte cond\t\tcond\tnocond")

	count = 0
	for hst in vals:
		for cst in vals:
			for ht in vals:
				for tt in vals:
					for hs in vals:
						for ts in vals:
							for hc in vals:
								for tc in vals:

									settings = {"history_target": ht,
										"history_source": hs,
										"history_conditional": hc,
										"tau_target": tt,
										"tau_source": ts,
										"tau_conditional": tc,
										"source_target_delay": hst,
										"conditional_target_delay": cst}
									
									
									jidt_estimator = JidtGaussianCTE(settings)
									
									itic = time.perf_counter()
									cte_jidt_cond = jidt_estimator.estimate(source1, target, cond)
									itoc = time.perf_counter()
									res_jidt_cond[0] = cte_jidt_cond
									time_jidt_cond += itoc - itic
									
									itic = time.perf_counter()
									cte_jidt_nocond = jidt_estimator.estimate(source1, target, nocond)
									itoc = time.perf_counter()
									res_jidt_nocond[0] = cte_jidt_nocond
									time_jidt_nocond += itoc - itic
									
									python_estimator = PythonGaussianCTE(settings)
									
									itic = time.perf_counter()
									cte_python_cond = python_estimator.estimate(source1, target, cond)
									itoc = time.perf_counter()
									res_python_cond[0] = cte_python_cond
									time_python_cond += itoc - itic
									
									itic = time.perf_counter()
									cte_python_nocond = python_estimator.estimate(source1, target, nocond)
									itoc = time.perf_counter()
									res_python_nocond[0] = cte_python_nocond
									time_python_nocond += itoc - itic
									
									print(f"{hst,cst,ht,tt,hs,ts,hc,tc}\t{cte_jidt_cond}\t{cte_python_cond}\t{np.isclose(cte_jidt_cond, cte_python_cond, rtol=close, atol=close)}\t{np.isclose(cte_jidt_nocond, cte_python_nocond, rtol=close, atol=close)}")

	verbose(res_jidt_cond, res_python_cond, "", "CTE cond")
	verbose(res_jidt_nocond, res_python_nocond, "", "CTE nocond")

	print("\nmean calculation times:")
	print(" JidtGaussianCTE (cond): ", np.mean(time_jidt_cond) )
	print(" PythonGaussianCTE (cond): ", np.mean(time_python_cond) )
	print(" JidtGaussianCTE (nocond): ", np.mean(time_jidt_nocond) )
	print(" PythonGaussianCTE (nocond): ", np.mean(time_python_nocond) )




def test_gaussian_te_local_values():

	expected_mi, source1, source2, target = _get_gauss_data(expand=True, seed=SEED)

	source1 = source1[1:]
	source2 = source2[1:]
	target = target[:-1]

	vals = [1,2,3]
	
	time_jidt = np.empty(np.power(len(vals),5))
	res_jidt = np.empty(np.power(len(vals),5))
	time_python = np.empty(np.power(len(vals),5))
	res_python = np.empty(np.power(len(vals),5))
	
	conds = np.empty((np.power(len(vals),5),5))
	
	print("hst,ht,tt,hs,ts\t\tJidtGaussianTE\t\tPythonGaussianTE\tclose")

	count = 0
	for hst in vals:

		for ht in vals:
			for tt in vals:
				for hs in vals:
					for ts in vals:

						conds[count,:] = [hst, ht, tt, hs, ts]
						settings_j = {"history_target": ht,
									"history_source": hs,
									"tau_target": tt,
									"tau_source": ts,
									"source_target_delay": hst,
									"local_values": True}

						#print("\n\n")
						#print(settings)
						#print("\n")
						
						jidt_estimator = JidtGaussianTE(settings_j)
						
						itic = time.perf_counter()
						te_jidt = jidt_estimator.estimate(source=source1, target=target)
						itoc = time.perf_counter()

						time_jidt[count] = itoc-itic
						

						settings_p = {"history_target": ht,
									"history_source": hs,
									"tau_target": tt,
									"tau_source": ts,
									"source_target_delay": hst,
									"local_values": True}

						python_estimator = PythonGaussianTE(settings_p)
						
						itic = time.perf_counter()
						te_python = python_estimator.estimate(source=source1, target=target)
						itoc = time.perf_counter()

						time_python[count] = itoc-itic
						
						count += 1
						

						if np.allclose(te_jidt, te_python, rtol=1e-04, atol=1e-04):
							print([hst, ht, tt, hs, ts], " - all local te results within tolerance (atol and rtol=1e-04)")
						else:
							if np.allclose(te_jidt, te_python, rtol=1e-03, atol=1e-03):
								print([hst, ht, tt, hs, ts], " - all local te results within tolerance (atol and rtol=1e-03)")
							else:
								diff = abs(te_jidt - te_python)
								num = (diff>1e-03).sum()

								print(f"{[hst, ht, tt, hs, ts]} - {num}/{te_jidt.shape[0]} of local te results are not within tolerance (1e-03) !!!!!")
	
	print("\nmean calculation times:")
	print(" JidtKraskovTE: ", np.mean(time_jidt) )
	print(" PythonKraskovTE: ", np.mean(time_python) )


	
# Test Kraskov estimators
def test_kraskov_mi():

	expected_mi, source1, source2, target = _get_gauss_data(expand=True, seed=SEED)

	mi_jidt = np.zeros(8)
	mi_python = np.zeros(8)
	time_jidt = np.zeros(8)
	time_python = np.zeros(8)

	vals = [1,2,3,4]

	count = 0

	for k in vals:
		settings = {}
		settings_j = {"kraskov_k": k,
					"noise_level": 0,
					"num_threads": "USE_ALL"}
		settings_p = {"kraskov_k": k,
					"noise_level": 0,
					"knn_finder": "scipy_ckdtree",
					"num_threads": "USE_ALL"}

		jidt_estimator = JidtKraskovMI(settings_j)
		python_estimator = PythonKraskovMI(settings_p)

		itic = time.perf_counter()
		mi_jidt[count] = jidt_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_jidt[count] = itoc - itic


		itic = time.perf_counter()
		mi_python[count] = python_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_python[count] = itoc - itic

		itic = time.perf_counter()
		mi_jidt[count+4] = jidt_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		time_jidt[count+4] = itoc - itic


		itic = time.perf_counter()
		mi_python[count+4] = python_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		time_python[count+4] = itoc - itic

		count += 1

	print("k\tJidtKraskovMI\t\tPythonKraskovMI")
	print("correlated")
	for i in range(len(vals)):
		print(f"{vals[i]}\t{mi_jidt[i]}\t{mi_python[i]}")
	print("uncorrelated")
	for i in range(len(vals)):
		print(f"{vals[i]}\t{mi_jidt[i+4]}\t{mi_python[i+4]}")

	verbose(mi_jidt, mi_python, "", "MI", False)

	
	print("\nmean calculation times:")
	print(" JidtKraskovMI: ", np.mean(time_jidt) )
	print(" PythonKraskovMI: ", np.mean(time_python) )


def test_kraskov_mi_local_values():

	expected_mi, source1, source2, target = _get_gauss_data(expand=True, seed=SEED)

	vals = [1,2,3,4]

	count = 0

	for k in vals:
		print("kraskov k = ", k)
		settings = {}
		settings_j = {"kraskov_k": k,
					"noise_level": 0,
					"local_values": True,
					"num_threads": "USE_ALL"}
		settings_p = {"kraskov_k": k,
					"noise_level": 0,
					"knn_finder": "scipy_ckdtree",
					"local_values": True,
					"num_threads": "USE_ALL"}

		jidt_estimator = JidtKraskovMI(settings_j)
		python_estimator = PythonKraskovMI(settings_p)

		itic = time.perf_counter()
		mi_jidt = jidt_estimator.estimate(source1, target)
		itoc = time.perf_counter()


		itic = time.perf_counter()
		mi_python = python_estimator.estimate(source1, target)
		itoc = time.perf_counter()

		verbose(mi_jidt, mi_python, k, "MI", True)

def test_kraskov_cmi():

	for i in [0.2, 0.4, 0.6, 0.8]:

		print(f"\n\nGaussian CMI Test data with covariance: {i}")

		expected_mi, source1, source2, target = _get_gauss_data(expand=True, covariance=i, seed=SEED)

		mi_jidt = np.zeros(8)
		mi_python = np.zeros(8)
		time_jidt = np.zeros(8)
		time_python = np.zeros(8)

		vals = [1,2,3,4]

		count = 0

		for k in vals:
			settings = {}
			settings_j = {"kraskov_k": k,
						"noise_level": 0,
						"num_threads": "USE_ALL"}
			settings_p = {"kraskov_k": k,
						"noise_level": 0,
						"knn_finder": "scipy_ckdtree",
						"num_threads": "USE_ALL"}

			jidt_estimator = JidtKraskovCMI(settings_j)
			python_estimator = PythonKraskovCMI(settings_p)

			itic = time.perf_counter()
			mi_jidt[count] = jidt_estimator.estimate(source1, target, source2)
			itoc = time.perf_counter()
			time_jidt[count] = itoc - itic


			itic = time.perf_counter()
			mi_python[count] = python_estimator.estimate(source1, target, source2)
			itoc = time.perf_counter()
			time_python[count] = itoc - itic

			itic = time.perf_counter()
			mi_jidt[count+4] = jidt_estimator.estimate(source2, target, source1)
			itoc = time.perf_counter()
			time_jidt[count+4] = itoc - itic


			itic = time.perf_counter()
			mi_python[count+4] = python_estimator.estimate(source2, target, source1)
			itoc = time.perf_counter()
			time_python[count+4] = itoc - itic

			count += 1

		print("k\tJidtKraskovMI\t\tPythonKraskovMI")
		print("uncorr conditional")
		for i in range(len(vals)):
			print(f"{vals[i]}\t{mi_jidt[i]}\t{mi_python[i]}")
		print("uncorr source")
		for i in range(len(vals)):
			print(f"{vals[i]}\t{mi_jidt[i+4]}\t{mi_python[i+4]}")
		
		verbose(mi_jidt, mi_python, "", "CMI", False)

		print("\nmean calculation times:")
		print(" JidtKraskovCMI: ", np.mean(time_jidt) )
		print(" PythonKraskovCMI: ", np.mean(time_python) )


def test_kraskov_cmi_local_values():

	expected_mi, source1, source2, target = _get_gauss_data(expand=True, seed=SEED)

	vals = [1,2,3,4]

	count = 0

	for k in vals:
		print("kraskov k = ", k)
		settings = {}
		settings_j = {"kraskov_k": k,
					"noise_level": 0,
					"local_values": True,
					"num_threads": "USE_ALL"}
		settings_p = {"kraskov_k": k,
					"noise_level": 0,
					"knn_finder": "scipy_ckdtree",
					"local_values": True,
					"num_threads": "USE_ALL"}

		jidt_estimator = JidtKraskovCMI(settings_j)
		python_estimator = PythonKraskovCMI(settings_p)

		itic = time.perf_counter()
		mi_jidt = jidt_estimator.estimate(source1, target, source2)
		itoc = time.perf_counter()


		itic = time.perf_counter()
		mi_python = python_estimator.estimate(source1, target, source2)
		itoc = time.perf_counter()

		verbose(mi_jidt, mi_python, k, "CMI", True)
								

def test_kraskov_ais():

	source1, source2 = _get_ar_data(seed=SEED)

	vals =  [1,2,3]
	time_jidt_cor = np.zeros(np.power(len(vals),2))
	res_jidt_cor = np.zeros(np.power(len(vals),2))
	time_python_cor = np.zeros(np.power(len(vals),2))
	res_python_cor = np.zeros(np.power(len(vals),2))
	time_jidt_uncor = np.zeros(np.power(len(vals),2))
	res_jidt_uncor = np.zeros(np.power(len(vals),2))
	time_python_uncor = np.zeros(np.power(len(vals),2))
	res_python_uncor = np.zeros(np.power(len(vals),2))

	count = 0

	for h in vals:
		for t in vals:

			settings_j = {'history': h, 'tau': t}

			settings_p = {'history': h, 'tau': t}
	
			jidt_estimator = JidtKraskovAIS(settings=settings_j)
			python_estimator = PythonKraskovAIS(settings=settings_p)

			itic = time.perf_counter()
			res_jidt_cor[count] = jidt_estimator.estimate(source1)
			itoc = time.perf_counter()
			time_jidt_cor[count] = itoc - itic
	
			itic = time.perf_counter()
			res_jidt_uncor[count] = jidt_estimator.estimate(source2)
			itoc = time.perf_counter()
			time_jidt_uncor[count] = itoc - itic
			
			itic = time.perf_counter()
			res_python_cor[count] = python_estimator.estimate(source1)
			itoc = time.perf_counter()
			time_python_cor[count] = itoc - itic
			
			itic = time.perf_counter()
			res_python_uncor[count] = python_estimator.estimate(source2)
			itoc = time.perf_counter()
			time_python_uncor[count] = itoc - itic

			count += 1

	print("JidtKraskovAIS\t\tPythonKraskovAIS")
	print("correlated")
	for i in range(len(res_jidt_cor)):
		print(f"{res_jidt_cor[i]}\t{res_python_cor[i]}")
	
	verbose(res_jidt_cor, res_python_cor, "", "AIS (corr)", False)

	
	print("uncorrelated")
	for i in range(len(res_jidt_uncor)):
		print(f"{res_jidt_uncor[i]}\t{res_python_uncor[i]}")

	verbose(res_jidt_uncor, res_python_uncor, "", "AIS (uncorr)", False)

	
	print("\nmean calculation times:")
	print(" JidtKraskovAIS (cor): ", np.mean(time_jidt_cor) )
	print(" PythonKraskovAIS (cor): ", np.mean(time_python_cor) )
	print(" JidtKraskovAIS (uncor): ", np.mean(time_jidt_uncor) )
	print(" PythonKraskovAIS (uncor): ", np.mean(time_python_uncor) )


def test_kraskov_ais_local_values():

	source1, source2 = _get_ar_data(seed=SEED)

	vals = [1,2,3,4]

	count = 0

	for k in vals:
		print("kraskov k = ", k)
		settings = {}
		settings_j = {'history': 2, 
					'tau': 2,
					'kraskov_k': k,
					'local_values': True}
		settings_p = {'history': 2, 
					'tau': 2,
					'kraskov_k': k,
					'knn_finder': "scipy_ckdtree",
					'local_values': True}
		
		jidt_estimator = JidtKraskovAIS(settings_j)
		python_estimator = PythonKraskovAIS(settings_p)

		itic = time.perf_counter()
		ais_jidt = jidt_estimator.estimate(source1)
		itoc = time.perf_counter()


		itic = time.perf_counter()
		ais_python = python_estimator.estimate(source1)
		itoc = time.perf_counter()

		verbose(ais_jidt, ais_python, "", "AIS", True)

		
def test_kraskov_te():

	expected_mi, source1, source2, target = _get_gauss_data(expand=False, seed=SEED)
	# add delay of one sample
	source1 = source1[1:]
	source2 = source2[1:]
	target = target[:-1]

	#vals = [1,2,3]
	vals = [1,2]
	

	time_jidt = np.empty(np.power(len(vals),5))
	res_jidt = np.empty(np.power(len(vals),5))
	time_python = np.empty(np.power(len(vals),5))
	res_python = np.empty(np.power(len(vals),5))

	conds = np.empty((np.power(len(vals),5),5))
	
	print("hst,ht,tt,hs,ts\t\tJidtKraskovTE\t\tPythonKraskovTE\tclose")

	count = 0
	for hst in vals:

		for ht in vals:
			for hs in vals:
				for tt in vals:
					for ts in vals:
					
						conds[count,:] = [hst, ht, tt, hs, ts]

						settings_j = {"kraskov_k": 4, 
							"history_target": ht,
							"history_source": hs,
							"tau_target": tt,
							"tau_source": ts,
							"source_target_delay": hst,
							"noise_level": 0, 
							"num_threads": 1}

						jidt_estimator = JidtKraskovTE(settings_j)
						
						itic = time.perf_counter()
						te_jidt = jidt_estimator.estimate(source=source1, target=target)
						itoc = time.perf_counter()

						time_jidt[count] = itoc-itic
						res_jidt[count] = te_jidt

						settings_p = {"kraskov_k": 4, 
							"history_target": ht,
							"history_source": hs,
							"tau_target": tt,
							"tau_source": ts,
							"source_target_delay": hst,
							"noise_level": 0, 
							"num_threads": 1}

						
						python_estimator = PythonKraskovTE(settings_p)
						
						itic = time.perf_counter()
						te_python = python_estimator.estimate(source=source1, target=target)
						itoc = time.perf_counter()

						time_python[count] = itoc-itic
						res_python[count] = te_python

						count += 1

						print(f"{[hst, ht, tt, hs, ts]}\t{te_jidt}\t{te_python}\t{np.isclose(te_jidt, te_python, rtol=1e-03, atol=1e-03)}")

	
	verbose(res_jidt, res_python, "", "TE", False)

	print("\nmean calculation times:")
	print(" JidtKraskovTE: ", np.mean(time_jidt) )
	print(" PythonKraskovTE: ", np.mean(time_python) )
	

def test_kraskov_te_local_values():

	expected_mi, source1, source2, target = _get_gauss_data(expand=True, seed=SEED)

	source1 = source1[1:]
	source2 = source2[1:]
	target = target[:-1]

	#vals = [1,2,3]
	vals = [1,2]
	
	time_jidt = np.empty(np.power(len(vals),5))
	res_jidt = np.empty(np.power(len(vals),5))
	time_python = np.empty(np.power(len(vals),5))
	res_python = np.empty(np.power(len(vals),5))
	
	conds = np.empty((np.power(len(vals),5),5))

	print("hst,ht,tt,hs,ts\t\tJidtKraskovTE\t\tPythonKraskovTE\tclose")

	count = 0
	for hst in vals:

		for ht in vals:
			for tt in vals:
				for hs in vals:
					for ts in vals:

						conds[count,:] = [hst, ht, tt, hs, ts]
						settings_j = {"history_target": ht,
									"history_source": hs,
									"tau_target": tt,
									"tau_source": ts,
									"source_target_delay": hst,
									"local_values": True}

						jidt_estimator = JidtKraskovTE(settings_j)
						
						itic = time.perf_counter()
						te_jidt = jidt_estimator.estimate(source=source1, target=target)
						itoc = time.perf_counter()

						time_jidt[count] = itoc-itic
						

						settings_p = {"history_target": ht,
									"history_source": hs,
									"tau_target": tt,
									"tau_source": ts,
									"source_target_delay": hst,
									"local_values": True}

						python_estimator = PythonKraskovTE(settings_p)
						
						itic = time.perf_counter()
						te_python = python_estimator.estimate(source=source1, target=target)
						itoc = time.perf_counter()

						time_python[count] = itoc-itic
						
						
						count += 1
						
						verbose(te_jidt, te_python, [hst, ht, tt, hs, ts], "TE", True)

	print("\nmean calculation times:")
	print(" JidtKraskovTE: ", np.mean(time_jidt) )
	print(" PythonKraskovTE: ", np.mean(time_python) )




# Test Discrete estimators
def test_discrete_mi():

	expected_mi, source1, source2, target = _get_gauss_data(expand=True, seed=SEED)
	
	mi_jidt_cor = np.zeros(4)
	mi_python_cor = np.zeros(4)
	time_jidt_cor = np.zeros(4)
	time_python_cor = np.zeros(4)
	
	mi_jidt_uncor = np.zeros(4)
	mi_python_uncor = np.zeros(4)
	time_jidt_uncor = np.zeros(4)
	time_python_uncor = np.zeros(4)

	vals = [2,5,8]

	count = 0
	for i in vals:
		settings = {'discretise_method': 'max_ent',
					'n_discrete_bins': i}
		
		jidt_estimator = JidtDiscreteMI(settings=settings)
		itic = time.perf_counter()
		mi_jidt_cor[count] = jidt_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_jidt_cor[count] = itoc - itic
		
		python_estimator = PythonDiscreteMI(settings=settings)
		itic = time.perf_counter()
		mi_python_cor[count] = python_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_python_cor[count] = itoc - itic

		jidt_estimator = JidtDiscreteMI(settings=settings)
		itic = time.perf_counter()
		mi_jidt_uncor[count] = jidt_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		time_jidt_uncor[count] = itoc - itic
		
		python_estimator = PythonDiscreteMI(settings=settings)
		itic = time.perf_counter()
		mi_python_uncor[count] = python_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		time_python_uncor[count] = itoc - itic

		count += 1
		
	print(f"Summary Jidt vs Python DiscreteMI 2D discretised Gaussian data:")

	print("MI values correlated data:")
	print("nbins\tJidtDiscreteMI\t\tPythonDiscreteMI")
	for i in range(len(vals)):
		print(f"{vals[i]}\t{mi_jidt_cor[i]}\t{mi_python_cor[i]}")

	verbose(mi_jidt_cor, mi_python_cor, "", "MI (cor)", False)


	print("\nmean calculation times:")
	print(" JidtDiscreteMI: ", np.mean(time_jidt_cor) )
	print(" PythonDiscreteMI: ", np.mean(time_python_cor) )
	
	print("\nMI values uncorrelated data:")
	print("nbins\tJidtDiscreteMI\t\tPythonDiscreteMI")
	for i in range(len(vals)):
		print(f"{vals[i]}\t{mi_jidt_uncor[i]}\t{mi_python_uncor[i]}")
	
	verbose(mi_jidt_uncor, mi_python_uncor, "", "MI (uncor)", False)

	print("\nmean calculation times:")
	print(" JidtDiscreteMI: ", np.mean(time_jidt_cor) )
	print(" PythonDiscreteMI: ", np.mean(time_python_cor) )
	
	# test bin data
	print("\nTest bin mem data:")
	varx, vary = _get_mem_binary_data(expand=True)
	settings = {'discretise_method': 'none'}
	est = JidtDiscreteMI(settings)
	itic = time.perf_counter()
	mi_jidt = est.estimate(varx, vary)
	itoc = time.perf_counter()
	print(f"JidtDiscreteMI: Estimated MI: {mi_jidt} - took: {itoc - itic}")
	est = PythonDiscreteMI(settings)
	itic = time.perf_counter()
	mi_python = est.estimate(varx, vary)
	itoc = time.perf_counter()
	print(f"PythonDiscreteMI: Estimated MI: {mi_python} - took: {itoc - itic}")
	assert np.isclose(mi_python, mi_jidt, rtol=0.08, atol=0.01)

	# test lags
	mi_jidt_cor = np.zeros(4)
	mi_jidt_uncor = np.zeros(4)
	mi_python_cor = np.zeros(4)
	mi_python_uncor = np.zeros(4)
	time_jidt_cor = np.zeros(4)
	time_jidt_uncor = np.zeros(4)
	time_python_cor = np.zeros(4)
	time_python_uncor = np.zeros(4)

	vals = [0,1,2,3]

	for lags in vals:
		settings = {}
		settings = {"lag_mi": lags,
					'discretise_method': 'max_ent'}

		# cor
		jidt_estimator = JidtDiscreteMI(settings)
		itic = time.perf_counter()
		mi_jidt_cor[lags] = jidt_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_jidt_cor[lags] = itoc - itic
		
		python_estimator = PythonDiscreteMI(settings)
		itic = time.perf_counter()
		mi_python_cor[lags] = python_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_python_cor[lags] = itoc - itic

		# uncor
		jidt_estimator = JidtDiscreteMI(settings)
		itic = time.perf_counter()
		mi_jidt_uncor[lags] = jidt_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		time_jidt_uncor[lags] = itoc - itic

		python_estimator = PythonDiscreteMI(settings)
		itic = time.perf_counter()
		mi_python_uncor[lags] = python_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		time_python_uncor[lags] = itoc - itic

	print(f"Summary Jidt vs Python DiscreteMI lags ({vals}):")

	print("MI values:")
	print("correlated data:")
	print("lag\tJidtDiscreteMI\t\tPythonDiscreteMI")
	for i in vals:
		print(f"{i}\t{mi_jidt_cor[i]}\t{mi_python_cor[i]}")
	
	verbose(mi_jidt_cor, mi_python_cor, "", "MI (cor)", False)


	print("uncorrelated data:")
	print("lag\tJidtDiscreteMI\t\tPythonDiscreteMI")
	for i in vals:
		print(f"{i}\t{mi_jidt_uncor[i]}\t{mi_python_uncor[i]}")

	verbose(mi_jidt_uncor, mi_python_uncor, "", "MI (uncor)", False)

	
	print("\nmean calculation times:")
	print(" JidtDiscrete (cor): ", np.mean(time_jidt_cor) )
	print(" PythonDiscreteMI (cor): ", np.mean(time_python_cor) )
	print(" JidtDiscreteMI (uncor): ", np.mean(time_jidt_uncor) )
	print(" PythonDiscreteMI (uncor): ", np.mean(time_python_uncor) )


def test_discrete_mi_local_values():

	expected_mi, source1, source2, target = _get_gauss_data(expand=True, seed=SEED)

	vals = [0,1,2,3]

	for lags in vals:
		settings = {}
		settings = {'lag_mi': lags,
					'local_values': True,
					'discretise_method': 'max_ent',
					'n_discrete_bins': 2}
		
		jidt_estimator = JidtDiscreteMI(settings)
		itic = time.perf_counter()
		mi_jidt = jidt_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		print(f"\nJidtDiscreteMI local_values (cor) took {itoc - itic} seconds)")

		python_estimator = PythonDiscreteMI(settings)
		itic = time.perf_counter()
		mi_python = python_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		print(f"PythonDiscreteMI local_values (cor) took {itoc - itic} seconds)")


		verbose(mi_jidt, mi_python, "", "MI (cor)", True)

		jidt_estimator = JidtDiscreteMI(settings)
		itic = time.perf_counter()
		mi_jidt = jidt_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		print(f"\nJidtDiscreteMI local_values (uncor) took {itoc - itic} seconds)")
		
		python_estimator = PythonDiscreteMI(settings)
		itic = time.perf_counter()
		mi_python = python_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		print(f"PythonDiscreteMI local_values (uncor) took {itoc - itic} seconds)")

		verbose(mi_jidt, mi_python, "", "MI (uncor)", True)
		
			

def test_discrete_cmi():

	expected_mi, source1, source2, target = _get_gauss_data(expand=True, seed=SEED)
	
	mi_jidt_cor = np.zeros(4)
	mi_python_cor = np.zeros(4)
	time_jidt_cor = np.zeros(4)
	time_python_cor = np.zeros(4)
	
	mi_jidt_uncor = np.zeros(4)
	mi_python_uncor = np.zeros(4)
	time_jidt_uncor = np.zeros(4)
	time_python_uncor = np.zeros(4)

	vals = [2,5,8]
	count = 0
	for i in vals:
		settings = {'discretise_method': 'max_ent',
					'n_discrete_bins': i}
		
		jidt_estimator = JidtDiscreteCMI(settings=settings)
		itic = time.perf_counter()
		mi_jidt_cor[count] = jidt_estimator.estimate(source1, target, source2)
		itoc = time.perf_counter()
		time_jidt_cor[count] = itoc - itic
		
		python_estimator = PythonDiscreteCMI(settings=settings)
		itic = time.perf_counter()
		mi_python_cor[count] = python_estimator.estimate(source1, target, source2)
		itoc = time.perf_counter()
		time_python_cor[count] = itoc - itic

		jidt_estimator = JidtDiscreteCMI(settings=settings)
		itic = time.perf_counter()
		mi_jidt_uncor[count] = jidt_estimator.estimate(source2, target, source1)
		itoc = time.perf_counter()
		time_jidt_uncor[count] = itoc - itic
		
		python_estimator = PythonDiscreteCMI(settings=settings)
		itic = time.perf_counter()
		mi_python_uncor[count] = python_estimator.estimate(source2, target, source1)
		itoc = time.perf_counter()
		time_python_uncor[count] = itoc - itic

		count += 1
		
	print(f"Summary Jidt vs Python DiscreteCMI 2D discretised Gaussian data:")

	print("CMI values uncorrelated conditional:")
	print("nbins\tJidtDiscreteCMI\t\tPythonDiscreteCMI")
	for i in range(len(vals)):
		print(f"{vals[i]}\t{mi_jidt_cor[i]}\t{mi_python_cor[i]}")
	
	verbose(mi_jidt_cor, mi_python_cor, "", "CMI (uncorrelated conditional)", False)

	
	print("\nmean calculation times:")
	print(" JidtDiscreteCMI: ", np.mean(time_jidt_cor) )
	print(" PythonDiscreteCMI: ", np.mean(time_python_cor) )
	
	print("\nMI values uncorrelated source:")
	print("nbins\tJidtDiscreteCMI\t\tPythonDiscreteCMI")
	for i in range(len(vals)):
		print(f"{vals[i]}\t{mi_jidt_uncor[i]}\t{mi_python_uncor[i]}")

	verbose(mi_jidt_uncor, mi_python_uncor, "", "CMI (uncorrelated source)", False)


	print("\nmean calculation times:")
	print(" JidtDiscreteCMI: ", np.mean(time_jidt_cor) )
	print(" PythonDiscreteCMI: ", np.mean(time_python_cor) )
	

	# test bin data
	print("\nTest bin mem data:")
	varx, vary = _get_mem_binary_data(expand=True)
	varz, _ = _get_mem_binary_data(expand=True)
	varx = varx[:10000]
	vary = vary[:10000]
	varz = varz[:10000]
	settings = {'discretise_method': 'none'}
	est = JidtDiscreteCMI(settings)
	itic = time.perf_counter()
	mi_jidt = est.estimate(varx, vary, varz)
	itoc = time.perf_counter()
	print(f"JidtDiscreteMI: Estimated MI: {mi_jidt} - took: {itoc - itic}")
	est = PythonDiscreteCMI(settings)
	itic = time.perf_counter()
	mi_python = est.estimate(varx, vary, varz)
	itoc = time.perf_counter()
	print(f"PythonDiscreteMI: Estimated MI: {mi_python} - took: {itoc - itic}")
	assert np.isclose(mi_python, mi_jidt, rtol=0.08, atol=0.01)


def test_discrete_cmi_local_values():

	expected_mi, source1, source2, target = _get_gauss_data(expand=True, seed=SEED)

	settings = {}
	settings = {'local_values': True,
				'discretise_method': 'max_ent',
				'n_discrete_bins': 2}
		
	jidt_estimator = JidtDiscreteCMI(settings)
	itic = time.perf_counter()
	mi_jidt = jidt_estimator.estimate(source1, target, source2)
	itoc = time.perf_counter()
	print(f"\nJidtDiscreteCMI local_values (uncorrelated conditional) took {itoc - itic} seconds)")

	python_estimator = PythonDiscreteCMI(settings)
	itic = time.perf_counter()
	mi_python = python_estimator.estimate(source1, target, source2)
	itoc = time.perf_counter()
	print(f"PythonDiscreteCMI local_values (uncorrelated conditional) took {itoc - itic} seconds)")

	verbose(mi_jidt, mi_python, "", "CMI (uncorrelated conditional)", True)

	jidt_estimator = JidtDiscreteCMI(settings)
	itic = time.perf_counter()
	mi_jidt = jidt_estimator.estimate(source2, target, source1)
	itoc = time.perf_counter()
	print(f"\nJidtDiscreteCMI local_values (uncorrelated source) took {itoc - itic} seconds)")
		
	python_estimator = PythonDiscreteCMI(settings)
	itic = time.perf_counter()
	mi_python = python_estimator.estimate(source2, target, source1)
	itoc = time.perf_counter()
	print(f"PythonDiscreteCMI local_values (uncorrelated source) took {itoc - itic} seconds)")
	
	verbose(mi_jidt, mi_python, "", "CMI (uncorrelated source)", True)
	


def test_discrete_ais():

	source1, source2 = _get_ar_data(seed=SEED)
	#expected_mi, source1, source2, target = _get_gauss_data(expand=True, seed=SEED)


	vals = [2,3]

	time_jidt_cor = np.zeros(np.power(len(vals),2))
	res_jidt_cor = np.zeros(np.power(len(vals),2))
	time_python_cor = np.zeros(np.power(len(vals),2))
	res_python_cor = np.zeros(np.power(len(vals),2))
	time_jidt_uncor = np.zeros(np.power(len(vals),2))
	res_jidt_uncor = np.zeros(np.power(len(vals),2))
	time_python_uncor = np.zeros(np.power(len(vals),2))
	res_python_uncor = np.zeros(np.power(len(vals),2))

	count = 0

	vals = [2,3]

	for h in vals:
		for t in vals:
			
			settings_j = {'history': h, 
						'tau': t,
						'discretise_method': 'max_ent',
						'n_discrete_bins': 2}

			settings_p = {'history': h, 
						'tau': t,
						'discretise_method': 'max_ent',
						'n_discrete_bins': 2}
	
			jidt_estimator = JidtDiscreteAIS(settings=settings_j)
			python_estimator = PythonDiscreteAIS(settings=settings_p)

			itic = time.perf_counter()
			res_jidt_cor[count] = jidt_estimator.estimate(source1)
			itoc = time.perf_counter()
			time_jidt_cor[count] = itoc - itic
	
			itic = time.perf_counter()
			res_jidt_uncor[count] = jidt_estimator.estimate(source2)
			itoc = time.perf_counter()
			time_jidt_uncor[count] = itoc - itic
			
			itic = time.perf_counter()
			res_python_cor[count] = python_estimator.estimate(source1)
			itoc = time.perf_counter()
			time_python_cor[count] = itoc - itic
			
			itic = time.perf_counter()
			res_python_uncor[count] = python_estimator.estimate(source2)
			itoc = time.perf_counter()
			time_python_uncor[count] = itoc - itic

			count += 1

	print("JidtDiscreteAIS\t\tPythonDiscreteAIS")
	print("correlated")
	for i in range(len(res_jidt_cor)):
		print(f"{res_jidt_cor[i]}\t{res_python_cor[i]}")
	if np.allclose(res_jidt_cor, res_python_cor, rtol=1e-04, atol=1e-04):
		print("All mi results (corr) within tolerance (atol and rtol=1e-04)")
	else:
		print("!!!!!!!!!!!!!!!!!!!!!! some results are not within tolerance (atol and rtol=1e-04)")
	
	print("uncorrelated")
	for i in range(len(res_jidt_uncor)):
		print(f"{res_jidt_uncor[i]}\t{res_python_uncor[i]}")

	if np.allclose(res_jidt_uncor, res_python_uncor, rtol=1e-04, atol=1e-04):
		print("All mi results (uncorr) within tolerance (atol and rtol=1e-04)")
	else:
		if np.allclose(res_jidt_uncor, res_python_uncor, rtol=1e-03, atol=1e-03):
			print("All mi results (uncorr) within tolerance (atol and rtol=1e-03)")
		else:
			print("!!!!!!!!!!!!!!!!!!!!!! some results (uncor) are not within tolerance (atol and rtol=1e-03)")

	print("\nmean calculation times:")
	print(" JidtDiscreteAIS (cor): ", np.mean(time_jidt_cor) )
	print(" PythonDiscreteAIS (cor): ", np.mean(time_python_cor) )
	print(" JidtDiscreteAIS (uncor): ", np.mean(time_jidt_uncor) )
	print(" PythonDiscreteAIS (uncor): ", np.mean(time_python_uncor) )########################## TODO########################## TODO

	s=1


def test_discrete_ais_local_values():

	source1, source2 = _get_ar_data(seed=SEED)

	
	settings = {}
	settings_j = {'history': 2, 
				'tau': 2,
				'discretise_method': 'max_ent',
				'n_discrete_bins': 2,
				'local_values': True}
	settings_p = {'history': 2, 
				'tau': 2,
				'discretise_method': 'max_ent',
				'n_discrete_bins': 2,
				'local_values': True}
		
	jidt_estimator = JidtDiscreteAIS(settings_j)
	python_estimator = PythonDiscreteAIS(settings_p)

	itic = time.perf_counter()
	ais_jidt = jidt_estimator.estimate(source1)
	itoc = time.perf_counter()
	print(f"\nJidtDiscreteAIS local_values {ais_jidt} took {itoc - itic} seconds)")

	itic = time.perf_counter()
	ais_python = python_estimator.estimate(source1)
	itoc = time.perf_counter()
	print(f"\nPythonDiscreteAIS local_values {ais_python} took {itoc - itic} seconds)")

	print(ais_jidt[:20])
	print(ais_python[:20])

	min_len=min(len(ais_jidt), len(ais_python))-100
	shift = 3
	ais_jidt=ais_jidt[shift:min_len+shift]
	
	if np.allclose(ais_jidt, ais_python[:min_len], rtol=1e-04, atol=1e-04):
		print(f"local mi results within tolerance (atol and rtol=1e-04)")
	else:
		if np.allclose(ais_jidt, ais_python[:min_len], rtol=1e-03, atol=1e-03):
			print(f"local mi results within tolerance (atol and rtol=1e-03)")
		else:
			if np.allclose(ais_jidt, ais_python[:min_len], rtol=1e-02, atol=1e-02):
				print(f"local mi results within tolerance (atol and rtol=1e-02)")
			else:
				print("!!!!!!!!!!!!!!!!!!!!!! some results are not within tolerance (atol and rtol=1e-04)")




def test_discrete_te():

	expected_mi, source1, source2, target = _get_gauss_data(expand=False, seed=SEED)
	# add delay of one sample
	source1 = source1[1:]
	source2 = source2[1:]
	target = target[:-1]

	vals = [1]

	time_jidt = np.empty(np.power(len(vals),5))
	res_jidt = np.empty(np.power(len(vals),5))
	time_python = np.empty(np.power(len(vals),5))
	res_python = np.empty(np.power(len(vals),5))

	count = 0
	for hst in vals:

		for ht in vals:
			for hs in vals:
				for tt in vals:
					for ts in vals:
					

						settings_j = {"history_target": ht,
									"history_source": hs,
									"tau_target": tt,
									"tau_source": ts,
									"source_target_delay": hst,
									'discretise_method': 'max_ent',
									'n_discrete_bins': 2}

						#print("\n\n")
						#print(settings)
						#print("\n")
						
						jidt_estimator = JidtDiscreteTE(settings_j)
						
						itic = time.perf_counter()
						te_jidt = jidt_estimator.estimate(source=source1, target=target)
						itoc = time.perf_counter()

						time_jidt[count] = itoc
						res_jidt[count] = te_jidt

						print(f"\nJidtDiscreteTE: {te_jidt} (took {itoc - itic} seconds)")
						

						settings_p = {"history_target": ht,
									"history_source": hs,
									"tau_target": tt,
									"tau_source": ts,
									"source_target_delay": hst,
									'discretise_method': 'max_ent',
									'n_discrete_bins': 2}

						
						python_estimator = PythonDiscreteTE(settings_p)
						
						itic = time.perf_counter()
						te_python = python_estimator.estimate(source=source1, target=target)
						itoc = time.perf_counter()

						time_python[count] = itoc
						res_python[count] = te_python

						print(f"PythonDiscreteTE: {te_python} (took {itoc - itic} seconds)")

						count += 1

	if np.allclose(res_jidt, res_python, rtol=1e-04, atol=1e-04):
		print("All mi results within tolerance (atol and rtol=1e-04)")
	else:
		print("!!!!!!!!!!!!!!!!!!!!!! some results are not within tolerance (atol and rtol=1e-04)")########################## TODO


# Test Spectral estimator
def test_spectral_mi():

	#source1, target, source2 = _get_freq_data(sample_rate=1000, duration=1.0, seed=SEED)
	source1, target, source2 = _get_freq_data(sample_rate=10000, duration=1.0)
	
	source1=source1[10:]
	source2=source2[10:]
	target=target[:-10]

	vals = [0,5,10,15,20]

	mi_python_cor = np.zeros(len(vals))
	mi_python_uncor = np.zeros(len(vals))
	time_python_cor = np.zeros(len(vals))
	time_python_uncor = np.zeros(len(vals))

	
	count=0
	for lags in vals:
		settings = {}
		settings = {"bins": 100,
					"lag_mi": lags,
					"noise_level": 0}

		python_estimator = PythonSpectralMI(settings)
		itic = time.perf_counter()
		mi_python_cor[count] = python_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_python_cor[count] = itoc - itic
		
		python_estimator = PythonSpectralMI(settings)
		itic = time.perf_counter()
		mi_python_uncor[count] = python_estimator.estimate(source1, source2)
		itoc = time.perf_counter()
		time_python_uncor[count] = itoc - itic

		count+=1

		

	print(f"Summary PythonSpectralMI lags ({vals}):")

	print("MI values:")
	print("correlated data:")
	print("lag\tPythonSpectralMI")
	for i in range(len(vals)):
		print(f"{i}\t{mi_python_cor[i]}")
	#if np.allclose(mi_jidt_cor, mi_python_cor, rtol=1e-04, atol=1e-04):
	#	print("all mi results within tolerance (atol and rtol=1e-04)")
	#else:
	#	print("some results are not within tolerance (atol and rtol=1e-04)")


	print("uncorrelated data:")
	print("lag\tPythonGaussianMI")
	for i in range(len(vals)):
		print(f"{i}\t{mi_python_uncor[i]}")
	#if np.allclose(mi_jidt_uncor, mi_python_uncor, rtol=1e-04, atol=1e-04):
	#	print("All mi results within tolerance (atol and rtol=1e-04)")
	#else:
	#	print("!!!!!!!!!!!!!!!!!!!!!! some results are not within tolerance (atol and rtol=1e-04)")

	print("\nmean calculation times:")
	print(" PythonSpectralMI (cor): ", np.mean(time_python_cor) )
	print(" PythonSpectralMI (uncor): ", np.mean(time_python_uncor) )########################## TODO
	


if __name__ == '__main__':
    
    # Gaussian
	"""
	print("\n\nCompare GaussianMI:\n")
	test_gaussian_mi()
	
	print("\n\nCompare GaussianMI local values:\n")
	test_gaussian_mi_local_values()
	
	print("\n\nCompare GaussianCMI:\n")
	test_gaussian_cmi()

	# test 2D input  ################################################## TODO ???

	print("\n\nCompare GaussianCMI local values:\n")
	test_gaussian_cmi_local_values()
	
	print("\n\nCompare GaussianAIS:\n")
	test_gaussian_ais()

	
	print("\n\nCompare GaussianAIS local values:\n")
	test_gaussian_ais_local_values()
	
	print("\n\nCompare GaussianTE:\n")
	test_gaussian_te()
	
	# test opt_source  ################################################## TODO
	
	print("\n\nCompare GaussianTE local values:\n")
	test_gaussian_te_local_values()
	"""
	
	print("\n\nCompare GaussianCTE:\n")
	test_gaussian_cte()
	
	
	


	# Kraskov
	"""
	print("\n\nCompare KraskovMI:\n")
	test_kraskov_mi()
	
	print("\n\nCompare KraskovMI local values:\n")
	test_kraskov_mi_local_values()

	print("\n\nCompare KraskovCMI:\n")
	test_kraskov_cmi()

	# test 2D input  ################################################## TODO ???
	
	print("\n\nCompare KraskovCMI local values:\n")
	test_kraskov_cmi_local_values()

	print("\n\nCompare KraskovAIS:\n")
	test_kraskov_ais()
	
	print("\n\nCompare KraskovAIS local values:\n")
	test_kraskov_ais_local_values()
	
	print("\n\nCompare KraskovTE:\n")
	test_kraskov_te()

	"""
	
	# test opt_source  ################################################## TODO

	#print("\n\nCompare KraskovTE local values:\n") ################################################## TODO
	#test_kraskov_te_local_values()
	

	#print("\n\nCompare KraskovTE theiler T correction:\n") ####################################### TODO
	#test_kraskov_te_theilert()


	#print("\n\nTest DiscreteMI:\n") ############################################# TODO
	#test_discrete_mi()

	#print("\n\nTest DiscreteMI local values:\n") ############################################# TODO
	#test_discrete_mi_local_values()

	#print("\n\nTest DiscreteCMI:\n") 
	#test_discrete_cmi()

	# test 2D input  ################################################## TODO ???

	#print("\n\nTest DiscreteCMI local values:\n") 
	#test_discrete_cmi_local_values()

	#print("\n\nTest DiscreteAIS:\n") ############################################# TODO
	#test_discrete_ais()

	#print("\n\nTest DiscreteAIS local values:\n") ############################################# TODO
	#test_discrete_ais_local_values()

	#print("\n\nTest DiscreteTE:\n") ############################################## TODO
	#test_discrete_te()

	# test opt source ############################################## TODO

	#print("\n\nTest DiscreteTE local values:\n") ############################################## TODO
	#test_discrete_te_local_values()


	

	#print("\n\nTest SpectralMI:\n") ################################################## TODO
	#test_spectral_mi()


