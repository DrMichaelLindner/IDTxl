



import numpy as np

import time
import sys

from idtxl.estimators_jidt import JidtKraskovMI, JidtKraskovCMI, JidtKraskovTE, JidtGaussianMI, JidtGaussianCMI, JidtGaussianTE, JidtGaussianAIS, JidtDiscreteMI
from idtxl.estimators_python import PythonKraskovMI, PythonKraskovCMI, PythonKraskovTE, PythonGaussianMI, PythonGaussianCMI, PythonGaussianTE, PythonGaussianAIS, PythonDiscreteMI

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




def test_gaussian_mi():

	expected_mi, source1, source2, target = _get_gauss_data(expand=False, seed=SEED)
	

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

		#print("\n")
		#print(settings)
		
		jidt_estimator = JidtGaussianMI(settings)
		itic = time.perf_counter()
		mi_jidt_cor[lags] = jidt_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_jidt_cor[lags] = itoc - itic
		#print(f"JidtGaussianMI (cor): {mi_jidt} (took {itoc - itic} seconds)")

		python_estimator = PythonGaussianMI(settings)
		itic = time.perf_counter()
		mi_python_cor[lags] = python_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_python_cor[lags] = itoc - itic
		#print(f"PythonGaussianMI (cor): {mi_python} (took {itoc - itic} seconds)")


		jidt_estimator = JidtGaussianMI(settings)
		itic = time.perf_counter()
		mi_jidt_uncor[lags] = jidt_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		time_jidt_uncor[lags] = itoc - itic
		#print(f"JidtGaussianMI (uncor): {mi_jidt} (took {itoc - itic} seconds)")

		python_estimator = PythonGaussianMI(settings)
		itic = time.perf_counter()
		mi_python_uncor[lags] = python_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		time_python_uncor[lags] = itoc - itic

		#print(f"PythonGaussianMI (uncor): {mi_python} (took {itoc - itic} seconds)")


	print(f"Summary Jidt vs Python GaussianMI lags ({vals}):")

	print("MI values:")
	print("correlated data:")
	print("lag\tJidtGaussianMI\t\tPythonGaussianMI")
	for i in vals:
		print(f"{i}\t{mi_jidt_cor[i]}\t{mi_python_cor[i]}")
	if np.allclose(mi_jidt_cor, mi_python_cor, rtol=1e-04, atol=1e-04):
		print("all mi results within tolerance (atol and rtol=1e-04)")
	else:
		print("some results are not within tolerance (atol and rtol=1e-04)")


	print("uncorrelated data:")
	print("lag\tJidtGaussianMI\t\tPythonGaussianMI")
	for i in vals:
		print(f"{i}\t{mi_jidt_uncor[i]}\t{mi_python_uncor[i]}")
	if np.allclose(mi_jidt_uncor, mi_python_uncor, rtol=1e-04, atol=1e-04):
		print("All mi results within tolerance (atol and rtol=1e-04)")
	else:
		print("!!!!!!!!!!!!!!!!!!!!!! some results are not within tolerance (atol and rtol=1e-04)")

	print("\nmean calculation times:")
	print(" JidtGaussianMI (cor): ", np.mean(time_jidt_cor) )
	print(" PythonGaussianMI (cor): ", np.mean(time_python_cor) )
	print(" JidtGaussianMI (uncor): ", np.mean(time_jidt_uncor) )
	print(" PythonGaussianMI (uncor): ", np.mean(time_python_uncor) )
	
	#xfast = np.mean([time_jidt_cor,time_jidt_uncor])/np.mean([time_python_cor,time_python_uncor])
	
	#print(f"\nPythonGaussianMI is approx {xfast} times faster")

	
def test_gaussian_cmi():

	expected_mi, source1, source2, target = _get_gauss_data(expand=False, seed=SEED)

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

	print(f"\nJidtGaussianCMI (uncor source): {mi_jidt} (took {itoc - itic} seconds)")
	itic = time.perf_counter()
	mi_python = python_estimator.estimate(source2, target, source1)
	itoc = time.perf_counter()
	res_python[1] = mi_python
	time_python += itoc - itic
	print(f"PythonGaussianCMI (uncor source): {mi_python} (took {itoc - itic} seconds)")

	#print(f"\nPythonGaussianCMI is approx {time_jidt/time_python} times faster")
	
	if np.allclose(res_jidt, res_python, rtol=1e-04, atol=1e-04):
		print("All mi results within tolerance (atol and rtol=1e-04)")
	else:
		print("!!!!!!!!!!!!!!!!!!!!!! some results are not within tolerance (atol and rtol=1e-04)")


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

	print("JidtKraskovMI\t\tPythonKraskovMI")
	print("correlated")
	for i in range(len(res_jidt_cor)):
		print(f"{res_jidt_cor[i]}\t{res_python_cor[i]}")
	print("uncorrelated")
	for i in range(len(res_jidt_uncor)):
		print(f"{res_jidt_uncor[i]}\t{res_python_uncor[i]}")

	if np.allclose(res_jidt_cor, res_python_cor, rtol=1e-04, atol=1e-04):
		print("All mi results (corr) within tolerance (atol and rtol=1e-04)")
	else:
		print("!!!!!!!!!!!!!!!!!!!!!! some results are not within tolerance (atol and rtol=1e-04)")
	if np.allclose(res_jidt_uncor, res_python_uncor, rtol=1e-04, atol=1e-04):
		print("All mi results (uncorr) within tolerance (atol and rtol=1e-04)")
	else:
		print("!!!!!!!!!!!!!!!!!!!!!! some results are not within tolerance (atol and rtol=1e-04)")

	print("\nmean calculation times:")
	print(" JidtGaussianAIS (cor): ", np.mean(time_jidt_cor) )
	print(" PythonGaussianAIS (cor): ", np.mean(time_python_cor) )
	print(" JidtGaussianAIS (uncor): ", np.mean(time_jidt_uncor) )
	print(" PythonGaussianAIS (uncor): ", np.mean(time_python_uncor) )




def test_kraskov_mi():

	expected_mi, source1, source2, target = _get_gauss_data(expand=False, seed=SEED)

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
	if np.allclose(mi_jidt, mi_python, rtol=1e-04, atol=1e-04):
		print("All mi results within tolerance (atol and rtol=1e-04)")
	else:
		print("!!!!!!!!!!!!!!!!!!!!!! some results are not within tolerance (atol and rtol=1e-04)")

	print("\nmean calculation times:")
	print(" JidtKraskovMI: ", np.mean(time_jidt) )
	print(" PythonKraskovMI: ", np.mean(time_python) )


def test_kraskov_mi_local_values():

	expected_mi, source1, source2, target = _get_gauss_data(expand=False, seed=SEED)

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

		if np.allclose(mi_jidt, mi_python, rtol=1e-04, atol=1e-04):
			print(f"local mi results within tolerance (atol and rtol=1e-04)")
		else:
			print("!!!!!!!!!!!!!!!!!!!!!! some results are not within tolerance (atol and rtol=1e-04)")


def test_kraskov_cmi():

	expected_mi, source1, source2, target = _get_gauss_data(expand=False, seed=SEED)

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
	if np.allclose(mi_jidt, mi_python, rtol=1e-04, atol=1e-04):
		print("All mi results within tolerance (atol and rtol=1e-04)")
	else:
		print("!!!!!!!!!!!!!!!!!!!!!! some results are not within tolerance (atol and rtol=1e-04)")

	print("\nmean calculation times:")
	print(" JidtKraskovCMI: ", np.mean(time_jidt) )
	print(" PythonKraskovCMI: ", np.mean(time_python) )


def test_kraskov_cmi_local_values():

	expected_mi, source1, source2, target = _get_gauss_data(expand=False, seed=SEED)

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

		if np.allclose(mi_jidt, mi_python, rtol=1e-04, atol=1e-04):
			print(f"local mi results within tolerance (atol and rtol=1e-04)")
		else:
			print("!!!!!!!!!!!!!!!!!!!!!! some results are not within tolerance (atol and rtol=1e-04)")


def test_kraskov_te():

	expected_mi, source1, source2, target = _get_gauss_data(expand=False, seed=SEED)
	# add delay of one sample
	source1 = source1[1:]
	source2 = source2[1:]
	target = target[:-1]

	vals = [1,2,3]

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
					

						settings_j = {"kraskov_k": 4, 
							"history_target": ht,
							"history_source": hs,
							"tau_target": tt,
							"tau_source": ts,
							"source_target_delay": hst,
							"noise_level": 0, 
							"num_threads": 1}

						#print("\n\n")
						#print(settings)
						#print("\n")
						
						jidt_estimator = JidtKraskovTE(settings_j)
						
						itic = time.perf_counter()
						te_jidt = jidt_estimator.estimate(source=source1, target=target)
						itoc = time.perf_counter()

						time_jidt[count] = itoc
						res_jidt[count] = te_jidt

						print(f"\nJidtKraskovTE: {te_jidt} (took {itoc - itic} seconds)")
						

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

						time_python[count] = itoc
						res_python[count] = te_python

						print(f"PythonKraskovTE: {te_python} (took {itoc - itic} seconds)")

						count += 1






if __name__ == '__main__':
    
	""" 
	print("\n\nCompare GaussianMI:\n")
	test_gaussian_mi()
	
	print("\n\nCompare GaussianCMI:\n")
	test_gaussian_cmi()
	"""

	#print("\n\nCompare GaussianAIS:\n")
	#test_gaussian_ais()

	#print("\n\nCompare GaussianTE:\n") ################################################## TODO
	#test_gaussian_te()


	"""
	print("\n\nCompare KraskovMI:\n")
	test_kraskov_mi()

	print("\n\nCompare KraskovMI local values:\n")
	test_kraskov_mi_local_values()

	print("\n\nCompare KraskovCMI:\n")
	test_kraskov_cmi()

	print("\n\nCompare KraskovCMI local values:\n")
	test_kraskov_cmi_local_values()
	"""

	#print("\n\nCompare KraskovAIS:\n") ################################################## TODO
	#test_kraskov_ais()

	print("\n\nCompare KraskovTE:\n") ################################################## TODO
	test_kraskov_te()

	#print("\n\nCompare KraskovTE theiler T correction:\n") ################################################## TODO
	#test_kraskov_te_theilert()


	## TODO
	# discete 