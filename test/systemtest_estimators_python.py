"""
Provide tests to compare Jidt and Python estimators

THIS TEST DOES NOT RUN WITHOUT PRESELECTING TESTS!

Hences, you should run appropriate parts of it separately 
(by uncommenting them in the main section at the end) and 
pype the output to a text file:

e.g.
python systemtest_estimators_python.py > your_output_file.txt

BE AWARE:
Running all tests in one go will take several hours and will 
produce a very long output! 

"""

import numpy as np

import time
import sys
import copy

from idtxl.estimators_jidt import (JidtKraskovMI, JidtKraskovCMI, JidtKraskovAIS, JidtKraskovTE, JidtKraskovCTE, 
									JidtGaussianMI, JidtGaussianCMI, JidtGaussianTE, JidtGaussianCTE, JidtGaussianAIS, 
									JidtDiscreteMI, JidtDiscreteCMI , JidtDiscreteAIS, JidtDiscreteTE)
from idtxl.estimators_python import (PythonKraskovMI, PythonKraskovCMI, PythonKraskovAIS, PythonKraskovTE, PythonKraskovCTE, 
									PythonGaussianMI, PythonGaussianCMI, PythonGaussianTE, PythonGaussianCTE, PythonGaussianAIS, 
									PythonDiscreteMI, PythonDiscreteCMI, PythonDiscreteAIS, PythonDiscreteTE)

from idtxl.multivariate_te import MultivariateTE
from idtxl.bivariate_te import BivariateTE
from idtxl.multivariate_mi import MultivariateMI
from idtxl.bivariate_mi import BivariateMI

from idtxl.idtxl_utils import calculate_mi
from idtxl.data import Data

import random as rn
import itertools
from generate_test_data import (_get_gauss_data, _get_ar_data, _generate_mute_data,
                                _get_mem_binary_data, _get_freq_data, generate_continuous_idtxl_data)


SEED = 42

def verbose(res_jidt, res_python, values, est, rtol=1e-04, atol=1e-04, local=False):

	if local:
		addstring = " local"
	else:
		addstring = ""

	if isinstance(res_jidt, float):
		addall = ""
		addres = "result"
	else:
		addall = "all"
		addres = "results"

	if values == "":
		values = "------------------------"
	
	if atol < 1e-03:

		if np.allclose(res_jidt, res_python, rtol=rtol, atol=atol):

			print(f"{values} - {addall}{addstring} {est} {addres} within tolerance (atol = {atol:.0e}) +++")
		else:
		
			rtol=rtol*10
			atol=atol*10
			if np.allclose(res_jidt, res_python, rtol=1e-03, atol=1e-03):
				print(f"{values} - {addall}{addstring} {est} {addres} within tolerance (atol = {atol:.0e}) ---")
			else:
				diff = abs(res_jidt - res_python)
				num = (diff>1e-03).sum()
				try:
					print(f"{values} - {num}/{res_jidt.shape[0]} of{addstring} {est} {addres} are not within tolerance (atol = {atol:.0e}) !!!!!!")
				except:
					print(f"{values} - {res_jidt} - {res_python} {est} result is not within tolerance (atol = {atol:.0e}) !!!!!!")

	else:
		rtol = atol
		if np.allclose(res_jidt, res_python, rtol=rtol, atol=atol):

			print(f"{values} - {addall}{addstring} {est} {addres} within tolerance (atol = {str(atol)}) +++")
		else:
			diff = abs(res_jidt - res_python)
			num = (diff>1e-03).sum()
			try: 	
				print(f"{values} - {num}/{res_jidt.shape[0]} of{addstring} {est} {addres} are not within tolerance (atol = {str(atol)}) !!!!!!")
			except:
				print(f"{values} - {res_jidt} - {res_python} {est} result is not within tolerance (atol = {str(atol)}) !!!!!!")

def testhead(est):
	print("\n\n#######################################################################")
	print(f"\n            Compare {est}:\n")
	print("#######################################################################")



#### Test Kraskov estimators
def test_kraskov_mi():
	
	# test 1D data
	expected_mi, source1, source2, target = _get_gauss_data(expand=True, seed=SEED)
	source1 = source1[1:]
	source2 = source2[1:]
	target = target[:-1]

	lvals = [0,1,2,3]
	kvals = [2,4,6,8]
	
	# Test 1D data
	print(f"\n\nTesting average MI using 1D gaussian data with covariance 0.4 and lag 1 \ntesting settings kraskov k {kvals} and lags_mi {lvals}\n")
	
	time_jidt_cor = np.empty(np.power(len(kvals),2))
	mi_jidt_cor = np.empty(np.power(len(kvals),2))
	time_python_cor = np.empty(np.power(len(kvals),2))
	mi_python_cor = np.empty(np.power(len(kvals),2))
	
	time_jidt_uncor = np.empty(np.power(len(kvals),2))
	mi_jidt_uncor = np.empty(np.power(len(kvals),2))
	time_python_uncor = np.empty(np.power(len(kvals),2))
	mi_python_uncor = np.empty(np.power(len(kvals),2))
	
	conds = np.empty((np.power(len(kvals),2),2))

	count = 0

	for k in kvals:
		for l in lvals:
			conds[count,:] = [k, l]
			
			settings_j = {"kraskov_k": k,
						"noise_level": 0,
						"num_threads": "USE_ALL",
						"lag_mi": l,
						}
			settings_p = {"kraskov_k": k,
						"noise_level": 0,
						"knn_finder": "scipy_ckdtree",
						"num_threads": "USE_ALL",
						"lag_mi": l,
						}
			
			jidt_estimator = JidtKraskovMI(settings_j)
			python_estimator = PythonKraskovMI(settings_p)
	
			itic = time.perf_counter()
			mi_jidt_cor[count] = jidt_estimator.estimate(source1, target)
			itoc = time.perf_counter()
			time_jidt_cor[count] = itoc - itic

			itic = time.perf_counter()
			mi_python_cor[count] = python_estimator.estimate(source1, target)
			itoc = time.perf_counter()
			time_python_cor[count] = itoc - itic
		
			itic = time.perf_counter()
			mi_jidt_uncor[count] = jidt_estimator.estimate(source2, target)
			itoc = time.perf_counter()
			time_jidt_uncor[count] = itoc - itic

			itic = time.perf_counter()
			mi_python_uncor[count] = python_estimator.estimate(source2, target)
			itoc = time.perf_counter()
			time_python_uncor[count] = itoc - itic

			count += 1

	atol = 1e-03
	print(f"k,lag\tJidtKraskovMI\t\tPythonKraskovMI\t\tclose {atol}")
	print("correlated")
	for i in range(len(mi_python_cor)):
		print(f"{conds[i,:]}\t{mi_jidt_cor[i]}\t{mi_python_cor[i]}\t{np.isclose(mi_jidt_cor[i], mi_python_cor[i], atol=atol)}")
	print("uncorrelated")
	for i in range(len(mi_python_uncor)):
		print(f"{conds[i,:]}\t{mi_jidt_uncor[i]}\t{mi_python_uncor[i]}\t{np.isclose(mi_jidt_uncor[i], mi_python_uncor[i], atol=atol)}")

	verbose(mi_jidt_cor, mi_python_cor, f"correlated data", "MI", local=False)
	verbose(mi_jidt_uncor, mi_python_uncor, f"uncorrelated data", "MI", local=False)
	
	print("\nmean calculation times:")
	print(" JidtKraskovMI (cor): ", np.mean(time_jidt_cor) )
	print(" PythonKraskovMI (cor): ", np.mean(time_python_cor) )
	print(" JidtKraskovMI (uncor): ", np.mean(time_jidt_uncor) )
	print(" PythonKraskovMI (uncor): ", np.mean(time_python_uncor) )

	print("\n=========================================================================")

	# Test passing to Jidt
	tvals = [0,2]
	avals = [1,2]
	print(f"\n\nTest passing algorithm_num= 2 and theiler_t>2 to JidtKraskovMI")
	print(f"average MI using 1D gaussian data with covariance 0.4 and lag 1")
	print(f"testing settings kraskov k = 4, and lags_mi = 1, algorithm_num {avals} and theiler_t {tvals}\n")
	
	print("algorithm_num, theiler_t")
	for a in avals:
	
		for t in tvals:

			print(f"{a, t}:")
			
			settings_j = {"kraskov_k": 4,
						"theiler_t": t,
						"algorithm_num": a,
						"noise_level": 0,
						"num_threads": "USE_ALL",
						"lag_mi": 1,
						}
			settings_p = {"kraskov_k": 4,
						"theiler_t": t,
						"algorithm_num": a,
						"noise_level": 0,
						"knn_finder": "scipy_ckdtree",
						"num_threads": "USE_ALL",
						"lag_mi": 1,
						}
			
			jidt_estimator = JidtKraskovMI(settings_j)
			python_estimator = PythonKraskovMI(settings_p)

			itic = time.perf_counter()
			mi_jidt_cor = jidt_estimator.estimate(source1, target)
			itoc = time.perf_counter()
			time_jidt_cor = itoc - itic

			itic = time.perf_counter()
			mi_python_cor = python_estimator.estimate(source1, target)
			itoc = time.perf_counter()
			time_python_cor = itoc - itic

	print("\n=========================================================================")
			
	# test different knn
	knns = ['scipy_ckdtree', 'scipy_kdtree', 'sklearn_kdtree', 'sklearn_balltree', 'numba_brute']

	print(f"\n\nTesting average MI using 1D gaussian data with covariance 0.4 and lag 1 ")
	print(f"testing knn finder: {knns}\n")
	
	settings_j = {"kraskov_k": 4,
				"noise_level": 0,
				"num_threads": "USE_ALL",
				"lag_mi": 1,
					}

	jidt_estimator = JidtKraskovMI(settings_j)
	itic = time.perf_counter()
	mi_jidt = jidt_estimator.estimate(source1, target)
	itoc = time.perf_counter()
	time_jidt = itoc - itic

	print(f"JidtKraskovMI - mi: {mi_jidt} - took: {time_jidt} ")
	
	count = 0
	for knn in knns:
		settings_p = {"kraskov_k": 4,
					"noise_level": 0,
					"knn_finder": knn,
					"num_threads": "USE_ALL",
					"lag_mi": 1,
					}
		python_estimator = PythonKraskovMI(settings_p)
		itic = time.perf_counter()
		mi_python = python_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_python = itoc - itic

		print(f"PythonKraskovMI ({knn}) - mi: {mi_python} - took: {time_python}- close to Jidt: {np.isclose(mi_jidt, mi_python)}")

		count += 1

	print("\n=========================================================================")
	
	# test 2D data
	data = _generate_mute_data(n_samples=2000, n_replications=4)

	source1 = data[0,:,:]
	target = data[2,:,:]
	source2 = data[4,:,:]

	time_jidt_cor = np.empty(np.power(len(kvals),2))
	mi_jidt_cor = np.empty(np.power(len(kvals),2))
	time_python_cor = np.empty(np.power(len(kvals),2))
	mi_python_cor = np.empty(np.power(len(kvals),2))
	time_jidt_uncor = np.empty(np.power(len(kvals),2))
	mi_jidt_uncor = np.empty(np.power(len(kvals),2))
	time_python_uncor = np.empty(np.power(len(kvals),2))
	mi_python_uncor = np.empty(np.power(len(kvals),2))

	conds = np.empty((np.power(len(kvals),2),2))

	print(f"\n\nTesting average MI using 2D mute data testing settings kraskov k {kvals} and lag_mi {lvals}\n")
	
	count = 0

	for k in kvals:
		for l in lvals:
			conds[count,:] = [k, l]
		
			settings_j = {"kraskov_k": k,
						"noise_level": 0,
						"num_threads": "USE_ALL",
						"lag_mi": l}
			settings_p = {"kraskov_k": k,
						"noise_level": 0,
						"knn_finder": "scipy_ckdtree",
						"num_threads": "USE_ALL",
						"lag_mi": l}

			jidt_estimator = JidtKraskovMI(settings_j)
			python_estimator = PythonKraskovMI(settings_p)

			itic = time.perf_counter()
			mi_jidt_cor[count] = jidt_estimator.estimate(source1, target)
			itoc = time.perf_counter()
			time_jidt_cor[count] = itoc - itic


			itic = time.perf_counter()
			mi_python_cor[count] = python_estimator.estimate(source1, target)
			itoc = time.perf_counter()
			time_python_cor[count] = itoc - itic

			itic = time.perf_counter()
			mi_jidt_uncor[count] = jidt_estimator.estimate(source2, target)
			itoc = time.perf_counter()
			time_jidt_uncor[count] = itoc - itic

			itic = time.perf_counter()
			mi_python_uncor[count] = python_estimator.estimate(source2, target)
			itoc = time.perf_counter()
			time_python_uncor[count] = itoc - itic

			count += 1

	print("k,lag\tJidtKraskovMI\t\tPythonKraskovMI")
	print("correlated")
	for i in range(len(mi_python_cor)):
		print(f"{conds[i,:]}\t{mi_jidt_cor[i]}\t{mi_python_cor[i]}")
	print("uncorrelated")
	for i in range(len(mi_python_uncor)):
		print(f"{conds[i,:]}\t{mi_jidt_uncor[i]}\t{mi_python_uncor[i]}")

	verbose(mi_jidt_cor, mi_python_cor, "correlated data", "MI", local=False)
	verbose(mi_jidt_uncor, mi_python_uncor, "uncorrelated data", "MI", local=False)

	print("\nmean calculation times:")
	print(" JidtKraskovMI (cor): ", np.mean(time_jidt_cor) )
	print(" PythonKraskovMI (cor): ", np.mean(time_python_cor) )
	print(" JidtKraskovMI (uncor): ", np.mean(time_jidt_uncor) )
	print(" PythonKraskovMI (uncor): ", np.mean(time_python_uncor) )

	print("\n=========================================================================")
	
	# test mixed dimension input
	d = [1, 2, 3, 5]

	print(f"\n\nTesting average MI using mixed dimensions\ntesting dimensions {d} for var1 and var2 each\n")
	print("Shapes:")
	data = _generate_mute_data(n_replications=5)
	source_o = data[0,:,:]
	target_o = data[2,:,:]
	
	settings2 = {"kraskov_k": 4,
				"noise_level": 0,
				"normalise": False,
				"num_threads": "USE_ALL",
				"local_values": True,
				"lag_mi": 2}
	
	for s in d:
		for t in d:
			source1 = source_o[:,:s]
			target = target_o[:,:t]
			
			cond = f"var1: {source1.shape}\tvar2: {target.shape}"

			jidt_estimator = JidtKraskovMI(settings2)
			python_estimator = PythonKraskovMI(settings2)
		
			itic = time.perf_counter()
			mi_jidt_cor = jidt_estimator.estimate(source1, target)
			itoc = time.perf_counter()
			time_jidt_cor = itoc - itic
			
			itic = time.perf_counter()
			mi_python_cor = python_estimator.estimate(source1, target)
			itoc = time.perf_counter()
			time_python_cor = itoc - itic

			verbose(mi_jidt_cor, mi_python_cor, cond, "MI")

def test_kraskov_mi_local_values():

	expected_mi, source1, source2, target = _get_gauss_data(expand=True, seed=SEED)
	
	lvals = [0,1,2,3]
	kvals = [2,4,6,8]
	print(f"\n\nTesting local MI using 1D gaussian data with covariance 0.4 and lag 1 \ntesting settings kraskov k {kvals} and lag_mi {lvals}\n")
	
	time_jidt_cor = np.empty(np.power(len(kvals),2))
	time_python_cor = np.empty(np.power(len(kvals),2))
	time_jidt_uncor = np.empty(np.power(len(kvals),2))
	time_python_uncor = np.empty(np.power(len(kvals),2))

	conds = np.empty((np.power(len(kvals),2),2))

	print(f"k, lag\t\tJidtKraskovMI vs PythonKraskovMI")
	count = 0
	for k in kvals:
		for l in lvals:
			conds[count,:] = [k, l]
			settings = {}
			settings_j = {"kraskov_k": k,
						"noise_level": 0,
						"normalise": False,
						"local_values": True,
						"num_threads": "USE_ALL",
						"lag_mi": l,
						}
			settings_p = {"kraskov_k": k,
						"noise_level": 0,
						"normalise": False,
						"knn_finder": "scipy_ckdtree",
						"local_values": True,
						"num_threads": "USE_ALL",
						"lag_mi": l,
						}

			jidt_estimator = JidtKraskovMI(settings_j)
			python_estimator = PythonKraskovMI(settings_p)

			itic = time.perf_counter()
			mi_jidt_cor = jidt_estimator.estimate(source1, target)
			itoc = time.perf_counter()
			time_jidt_cor[count] = itoc - itic

			itic = time.perf_counter()
			mi_python_cor = python_estimator.estimate(source1, target)
			itoc = time.perf_counter()
			time_python_cor[count] = itoc - itic

			verbose(mi_jidt_cor, mi_python_cor, f"{conds[count,:]} correlated", "MI", local=True)

			itic = time.perf_counter()
			mi_jidt_uncor = jidt_estimator.estimate(source2, target)
			itoc = time.perf_counter()
			time_jidt_uncor[count] = itoc - itic

			itic = time.perf_counter()
			mi_python_uncor = python_estimator.estimate(source2, target)
			itoc = time.perf_counter()
			time_python_uncor[count] = itoc - itic

			verbose(mi_jidt_uncor, mi_python_uncor, f"{conds[count,:]} uncorrelated", "MI", local=True)
			
			count += 1

	print("\nmean calculation times:")
	print(" JidtKraskovMI (cor): ", np.mean(time_jidt_cor) )
	print(" PythonKraskovMI (cor): ", np.mean(time_python_cor) )
	print(" JidtKraskovMI (uncor): ", np.mean(time_jidt_uncor) )
	print(" PythonKraskovMI (uncor): ", np.mean(time_python_uncor) )

	print("\n=========================================================================")

	# test 2D
	data = _generate_mute_data(n_samples=2000, n_replications=4)

	source1 = data[0,:,:]
	target = data[2,:,:]
	source2 = data[4,:,:]

	time_jidt_cor = np.empty(np.power(len(kvals),2))
	time_python_cor = np.empty(np.power(len(kvals),2))
	time_jidt_uncor = np.empty(np.power(len(kvals),2))
	time_python_uncor = np.empty(np.power(len(kvals),2))
	
	conds = np.empty((np.power(len(kvals),2),2))

	print(f"\n\nTesting local MI using 2D mute data with and without coupling testing settings kraskov k {kvals} and lag_mi {lvals}\n")
	
	print(f"k, lag\t\tJidtKraskovMI vs PythonKraskovMI")
	count = 0
	for k in kvals:
		for l in lvals:
			conds[count,:] = [k, l]
			settings = {}
			settings_j = {"kraskov_k": k,
						"noise_level": 0,
						"normalise": False,
						"local_values": True,
						"num_threads": "USE_ALL",
						"lag_mi": l}
			settings_p = {"kraskov_k": k,
						"noise_level": 0,
						"normalise": False,
						"knn_finder": "scipy_ckdtree",
						"local_values": True,
						"num_threads": "USE_ALL",
						"lag_mi": l}

			jidt_estimator = JidtKraskovMI(settings_j)
			python_estimator = PythonKraskovMI(settings_p)

			itic = time.perf_counter()
			mi_jidt_cor = jidt_estimator.estimate(source1, target)
			itoc = time.perf_counter()
			time_jidt_cor[count] = itoc - itic

			itic = time.perf_counter()
			mi_python_cor = python_estimator.estimate(source1, target)
			itoc = time.perf_counter()
			time_python_cor[count] = itoc - itic

			verbose(mi_jidt_cor, mi_python_cor, f"{conds[count,:]} correlated", "MI", local=True)

			itic = time.perf_counter()
			mi_jidt_uncor = jidt_estimator.estimate(source2, target)
			itoc = time.perf_counter()
			time_jidt_uncor[count] = itoc - itic

			itic = time.perf_counter()
			mi_python_uncor = python_estimator.estimate(source2, target)
			itoc = time.perf_counter()
			time_python_uncor[count] = itoc - itic

			verbose(mi_jidt_uncor, mi_python_uncor, f"{conds[count,:]} uncorrelated", "MI", local=True)

			count += 1
	
	print("\nmean calculation times:")
	print(" JidtKraskovMI (cor): ", np.mean(time_jidt_cor) )
	print(" PythonKraskovMI (cor): ", np.mean(time_python_cor) )
	print(" JidtKraskovMI (uncor): ", np.mean(time_jidt_uncor) )
	print(" PythonKraskovMI (uncor): ", np.mean(time_python_uncor) )

	print("\n=========================================================================")

	# test mixed dimension input
	d = [1, 2, 3, 5]

	print(f"\n\nTesting average MI using mixed dimensions\ntesting dimensions {d} for var1 and var2 each\n")
	print("Shapes:")
	data = _generate_mute_data(n_replications=5)
	source_o = data[0,:,:]
	target_o = data[2,:,:]
	
	settings2 = {"kraskov_k": k,
				"noise_level": 0,
				"normalise": False,
				"local_values": True,
				"num_threads": "USE_ALL",
				"lag_mi": 2}
	
	for s in d:
		for t in d:
			
			source1 = source_o[:,:s]
			target = target_o[:,:t]
			
			cond = f"var1: {source1.shape}\tvar2: {target.shape}"

			jidt_estimator = JidtKraskovMI(settings2)
			python_estimator = PythonKraskovMI(settings2)
		
			itic = time.perf_counter()
			mi_jidt_cor = jidt_estimator.estimate(source1, target)
			itoc = time.perf_counter()
			time_jidt_cor = itoc - itic
			
			itic = time.perf_counter()
			mi_python_cor = python_estimator.estimate(source1, target)
			itoc = time.perf_counter()
			time_python_cor = itoc - itic
		
			verbose(mi_jidt_cor, mi_python_cor, cond, "MI", local=True, atol=1e-03)

def test_kraskov_cmi():
	
	cvals = [0.2, 0.4, 0.6, 0.8]
	kvals = [2,4,6,8]
	
	# test 1D data
	print(f"\n\nTesting average CMI using 1D gaussian data with covariances {cvals} \ntesting settings kraskov k {kvals} and uncorrelated conditional and uncorrelated source\n")
		
	time_jidt_cor = np.empty(np.power(len(kvals),2))
	mi_jidt_cor = np.empty(np.power(len(kvals),2))
	time_python_cor = np.empty(np.power(len(kvals),2))
	mi_python_cor = np.empty(np.power(len(kvals),2))
	time_jidt_uncor = np.empty(np.power(len(kvals),2))
	mi_jidt_uncor = np.empty(np.power(len(kvals),2))
	time_python_uncor = np.empty(np.power(len(kvals),2))
	mi_python_uncor = np.empty(np.power(len(kvals),2))

	conds = np.empty((np.power(len(kvals),2),2))
	
	count = 0
	for k in kvals:
		for i in cvals:
			conds[count,:] = [k,i]

			expected_mi, source1, source2, target = _get_gauss_data(expand=True, covariance=i, seed=SEED)
			
			settings_j = {"kraskov_k": k,
						"noise_level": 0,
						"normalise": False,
						"num_threads": "USE_ALL",
						}
			settings_p = {"kraskov_k": k,
						"noise_level": 0,
						"normalise": False,
						"knn_finder": "scipy_ckdtree",
						"num_threads": "USE_ALL",
						}

			jidt_estimator = JidtKraskovCMI(settings_j)
			python_estimator = PythonKraskovCMI(settings_p)

			itic = time.perf_counter()
			mi_jidt_cor[count] = jidt_estimator.estimate(source1, target, source2)
			itoc = time.perf_counter()
			time_jidt_cor[count] = itoc - itic

			itic = time.perf_counter()
			mi_python_cor[count] = python_estimator.estimate(source1, target, source2)
			itoc = time.perf_counter()
			time_python_cor[count] = itoc - itic

			itic = time.perf_counter()
			mi_jidt_uncor[count] = jidt_estimator.estimate(source2, target, source1)
			itoc = time.perf_counter()
			time_jidt_uncor[count] = itoc - itic

			itic = time.perf_counter()
			mi_python_uncor[count] = python_estimator.estimate(source2, target, source1)
			itoc = time.perf_counter()
			time_python_uncor[count] = itoc - itic

			count += 1

	atol = 1e-03
	print(f"k, cov\t\tJidtKraskovCMI\t\tPythonKraskovCMI\tclose {atol}")
	print("uncorrelated conditional")
	for i in range(len(mi_jidt_cor)):
		print(f"{conds[i,:]}\t{mi_jidt_cor[i]}\t{mi_python_cor[i]}\t{np.isclose(mi_jidt_cor[i], mi_python_cor[i], atol=atol)}")
	print("uncorrelated source")
	for i in range(len(mi_jidt_uncor)):
		print(f"{conds[i,:]}\t{mi_jidt_uncor[i]}\t{mi_python_uncor[i]}\t{np.isclose(mi_jidt_cor[i], mi_python_cor[i], atol=atol)}")
	
	verbose(mi_jidt_cor, mi_python_cor, "uncorrelated conditional", "CMI", local=False)
	verbose(mi_jidt_uncor, mi_python_uncor, "uncorrelated source", "CMI", local=False)

	print(f"\nmean calculation times:")
	print(" JidtKraskovCMI: (uncorrelated conditional)", np.mean(time_jidt_cor) )
	print(" PythonKraskovCMI: (uncorrelated conditional)", np.mean(time_python_cor) )
	print(" JidtKraskovCMI: (uncorrelated source)", np.mean(time_jidt_uncor) )
	print(" PythonKraskovCMI: (uncorrelated source)", np.mean(time_python_uncor) )

	print("\n=========================================================================")

	# Test passing to Jidt
	tvals = [0,2]
	avals = [1,2]
	print(f"\n\nTest passing algorithm_num= 2 and theiler_t>2 to JidtKraskovCMI")
	print(f"average MI using 1D gaussian data with covariance 0.4 and lag 1")
	print(f"testing settings kraskov k = 4, algorithm_num {avals} and theiler_t {tvals}\n")
	
	print("algorithm_num, theiler_t")
	for a in avals:
	
		for t in tvals:
			print(f"{a,t}:")
			settings_j = {"kraskov_k": 4,
						"theiler_t": t,
						"algorithm_num": a,
						"noise_level": 0,
						"num_threads": "USE_ALL",
						}
			settings_p = {"kraskov_k": 4,
						"theiler_t": t,
						"algorithm_num": a,
						"noise_level": 0,
						"knn_finder": "scipy_ckdtree",
						"num_threads": "USE_ALL",
						}
			
			jidt_estimator = JidtKraskovCMI(settings_j)
			python_estimator = PythonKraskovCMI(settings_p)

			itic = time.perf_counter()
			mi_jidt_cor = jidt_estimator.estimate(source1, target, source2)
			itoc = time.perf_counter()
			time_jidt_cor = itoc - itic

			itic = time.perf_counter()
			mi_python_cor = python_estimator.estimate(source1, target, source2)
			itoc = time.perf_counter()
			time_python_cor = itoc - itic

	print("\n=========================================================================")

	# test 2D data
	time_jidt_cor = np.empty(4)
	mi_jidt_cor = np.empty(4)
	time_python_cor = np.empty(4)
	mi_python_cor = np.empty(4)
	time_jidt_uncor = np.empty(4)
	mi_jidt_uncor = np.empty(4)
	time_python_uncor = np.empty(4)
	mi_python_uncor = np.empty(4)

	print(f"\n\nTesting average CMI using 2D mute data \ntesting settings kraskov {kvals} and uncorrelated conditional and uncorrelated source\n")

	count = 0
	for k in kvals:
		data = _generate_mute_data(n_samples=2000, n_replications=4)

		source1 = data[0,:,:]
		target = data[2,:,:]
		source2 = data[4,:,:]

		settings_j = {"kraskov_k": k,
					"noise_level": 0,
					"normalise": False,
					"num_threads": "USE_ALL"}
		settings_p = {"kraskov_k": k,
					"noise_level": 0,
					"normalise": False,
					"knn_finder": "scipy_ckdtree",
					"num_threads": "USE_ALL"}

		jidt_estimator = JidtKraskovCMI(settings_j)
		python_estimator = PythonKraskovCMI(settings_p)

		itic = time.perf_counter()
		mi_jidt_cor[count] = jidt_estimator.estimate(source1, target, source2)
		itoc = time.perf_counter()
		time_jidt_cor[count] = itoc - itic

		itic = time.perf_counter()
		mi_python_cor[count] = python_estimator.estimate(source1, target, source2)
		itoc = time.perf_counter()
		time_python_cor[count] = itoc - itic

		itic = time.perf_counter()
		mi_jidt_uncor[count] = jidt_estimator.estimate(source2, target, source1)
		itoc = time.perf_counter()
		time_jidt_uncor[count] = itoc - itic

		itic = time.perf_counter()
		mi_python_uncor[count] = python_estimator.estimate(source2, target, source1)
		itoc = time.perf_counter()
		time_python_uncor[count] = itoc - itic

		count += 1

	print("k,cov\tJidtKraskovCMI\t\tPythonKraskovCMI")
	print("uncorrelated conditional")
	for i in range(len(mi_jidt_cor)):
		print(f"{kvals[i]}\t{mi_jidt_cor[i]}\t{mi_python_cor[i]}")
	print("uncorrelated source")
	for i in range(len(mi_jidt_uncor)):
		print(f"{kvals[i]}\t{mi_jidt_uncor[i]}\t{mi_python_uncor[i]}")
	
	verbose(mi_jidt_cor, mi_python_cor, "uncorrelated conditional", "CMI", local=False)
	verbose(mi_jidt_uncor, mi_python_uncor, "uncorrelated source", "CMI", local=False)

	print("\nmean calculation times:")
	print(" JidtKraskovCMI: (uncorrelated conditional)", np.mean(time_jidt_cor) )
	print(" PythonKraskovCMI: (uncorrelated conditional)", np.mean(time_python_cor) )
	print(" JidtKraskovCMI: (uncorrelated source)", np.mean(time_jidt_uncor) )
	print(" PythonKraskovCMI: (uncorrelated source)", np.mean(time_python_uncor) )

	print("\n=========================================================================")

	# test mixed dimension input
	d = [1, 2, 3]

	print(f"\n\nTesting average MI using mixed dimensions\ntesting dimensions {d} for var1, var2 and cond each\n")
	print("Shapes:")
	
	data = _generate_mute_data()

	source_o = data[0,:,:]
	target_o = data[2,:,:]
	cond_o = data[4,:,:]
	
	settings = {"kraskov_k": 4,
				"noise_level": 0,
				"normalise": False}
	
	for s in d:
		for t in d:
			for c in d:
			
				source1 = source_o[:,:s]
				target = target_o[:,:t]
				conditional = cond_o[:,:c]
				
				conds = f"var1: {source1.shape[1]} var2: {target.shape[1]} cond: {conditional.shape[1]}"

				jidt_estimator = JidtKraskovCMI(settings)
				python_estimator = PythonKraskovCMI(settings)
			
				itic = time.perf_counter()
				mi_jidt_cor = jidt_estimator.estimate(source1, target, conditional)
				itoc = time.perf_counter()
				time_jidt_cor = itoc - itic
				
				itic = time.perf_counter()
				mi_python_cor = python_estimator.estimate(source1, target, conditional)
				itoc = time.perf_counter()
				time_python_cor = itoc - itic

				verbose(mi_jidt_cor, mi_python_cor, conds, "CMI", local=False, atol=1e-03)
		
def test_kraskov_cmi_local_values():

	# test 1D	
	expected_mi, source1, source2, target = _get_gauss_data(expand=True, seed=SEED)
	source1_1D = source1[1:]
	source2_1D = source2[1:]
	target_1D = target[:-1]

	data = _generate_mute_data(n_samples=2000, n_replications=4)

	source1_2D = data[0,:,:]
	target_2D = data[2,:,:]
	source2_2D = data[4,:,:]

	vals = [2,4,6,8]
	
	knn_list = ['scipy_kdtree', 'scipy_ckdtree', 'sklearn_kdtree', 'sklearn_balltree']
	
	print(f"\n\nTesting local CMI using 1D gaussian data with covariances 0.4 and lag 1\ntesting settings kraskov {vals} and uncorrelated conditional and uncorrelated source\n")
	
	kcount = 0
	for knn in knn_list:
		print(f"\nKNN finder: {knn}\n")

		time_jidt_cor = np.zeros(4)
		time_python_cor = np.zeros(4)
		time_jidt_uncor = np.zeros(4)
		time_python_uncor = np.zeros(4)
		
		print("kraskov k")
		count = 0
		for k in vals:

			settings = {}
			settings_j = {"kraskov_k": k,
						"noise_level": 0,
						"local_values": True,
						"num_threads": "USE_ALL"}
			settings_p = {"kraskov_k": k,
						"noise_level": 0,
						"knn_finder": "scipy_ckdtree",
						"local_values": True,
						"num_threads": "USE_ALL",
						"knn_finder": knn}

			jidt_estimator = JidtKraskovCMI(settings_j)
			python_estimator = PythonKraskovCMI(settings_p)

			itic = time.perf_counter()
			cmi_jidt_cor = jidt_estimator.estimate(source1_1D, target_1D, source2_1D)
			itoc = time.perf_counter()
			time_jidt_cor[count] = itoc - itic
			
			itic = time.perf_counter()
			cmi_python_cor = python_estimator.estimate(source1_1D, target_1D, source2_1D)
			itoc = time.perf_counter()
			time_python_cor[count] = itoc - itic
			
			verbose(cmi_jidt_cor, cmi_python_cor, f"{k} uncorrelated conditional", "CMI", local=True, atol=1e-03)

			itic = time.perf_counter()
			cmi_jidt_uncor = jidt_estimator.estimate(source1_1D, target_1D, source2_1D)
			itoc = time.perf_counter()
			time_jidt_uncor[count] = itoc - itic

			itic = time.perf_counter()
			cmi_python_uncor = python_estimator.estimate(source1_1D, target_1D, source2_1D)
			itoc = time.perf_counter()
			time_python_uncor[count] = itoc - itic

			verbose(cmi_jidt_uncor, cmi_python_uncor, f"{k} uncorrelated source     ", "CMI", local=True, atol=1e-03)

		print("\nmean calculation times:")
		print(" JidtKraskovCMI: (uncorrelated conditional)", np.mean(time_jidt_cor) )
		print(" PythonKraskovCMI: (uncorrelated conditional)", np.mean(time_python_cor) )
		print(" JidtKraskovCMI: (uncorrelated source)", np.mean(time_jidt_uncor) )
		print(" PythonKraskovCMI: (uncorrelated source)", np.mean(time_python_uncor) )

	print("\n=========================================================================")

	# test 2D
	print(f"\n\nTesting local CMI using 2D mute data\ntesting settings kraskov {vals} and uncorrelated conditional and uncorrelated source\n")
		
	kcount = 0
	for knn in knn_list:
		print(f"\nKNN finder: {knn}\n")
		
		time_jidt_cor = np.zeros(4)
		time_python_cor = np.zeros(4)
		time_jidt_uncor = np.zeros(4)
		time_python_uncor = np.zeros(4)

		count = 0
		for k in vals:

			settings = {}
			settings_j = {"kraskov_k": k,
						"noise_level": 0,
						"normalise": False,
						"local_values": True,
						"num_threads": "USE_ALL"}
			settings_p = {"kraskov_k": k,
						"noise_level": 0,
						"normalise": False,
						"knn_finder": "scipy_ckdtree",
						"local_values": True,
						"num_threads": "USE_ALL",
						"knn_finder": knn}

			jidt_estimator = JidtKraskovCMI(settings_j)
			python_estimator = PythonKraskovCMI(settings_p)

			itic = time.perf_counter()
			cmi_jidt_cor = jidt_estimator.estimate(source1_2D, target_2D, source2_2D)
			itoc = time.perf_counter()
			time_jidt_cor[count] = itoc - itic

			itic = time.perf_counter()
			cmi_python_cor = python_estimator.estimate(source1_2D, target_2D, source2_2D)
			itoc = time.perf_counter()
			time_python_cor[count] = itoc - itic

			verbose(cmi_jidt_cor, cmi_python_cor, f"{k} uncorrelated conditional", "CMI", local=True, atol=1e-03)

			itic = time.perf_counter()
			cmi_jidt_uncor = jidt_estimator.estimate(source1_2D, target_2D, source2_2D)
			itoc = time.perf_counter()
			time_jidt_uncor[count] = itoc - itic

			itic = time.perf_counter()
			cmi_python_uncor = python_estimator.estimate(source1_2D, target_2D, source2_2D)
			itoc = time.perf_counter()
			time_python_uncor[count] = itoc - itic

			verbose(cmi_jidt_uncor, cmi_python_uncor, f"{k} uncorrelated source", "CMI", local=True, atol=1e-03)

		print("\nmean calculation times:")
		print(" JidtKraskovCMI: (uncorrelated conditional)", np.mean(time_jidt_cor) )
		print(" PythonKraskovCMI: (uncorrelated conditional)", np.mean(time_python_cor) )
		print(" JidtKraskovCMI: (uncorrelated source)", np.mean(time_jidt_uncor) )
		print(" PythonKraskovCMI: (uncorrelated source)", np.mean(time_python_uncor) )

	print("\n=========================================================================")

	# test mixed dimension input
	d = [1, 2, 3, 5]
	print(f"\n\nTesting local MI using mixed dimensions\ntesting dimensions {d} for var1, var2 and conditional each - (default k=4)\n")
	print("Shapes:")
	data = _generate_mute_data(n_replications=5)
	source_o = data[0,:,:]
	target_o = data[2,:,:]
	cond_o = data[4,:,:]
	
	settings2 = {"kraskov_k": 4,
				"normalise": False,
				"noise_level": 0,
				"num_threads": "USE_ALL"}
	
	for s in d:
		for t in d:
			for c in d:
				source1 = source_o[:,:s]
				target = target_o[:,:t]
				conditional = cond_o[:,:t]
				
				cond = f"var1: {source1.shape[1]} - var2: {target.shape[1]} - cond: {conditional.shape[1]}"

				jidt_estimator = JidtKraskovCMI(settings2)
				python_estimator = PythonKraskovCMI(settings2)
			
				itic = time.perf_counter()
				mi_jidt_cor = jidt_estimator.estimate(source1, target, conditional)
				itoc = time.perf_counter()
				time_jidt_cor = itoc - itic
				
				itic = time.perf_counter()
				mi_python_cor = python_estimator.estimate(source1, target, conditional)
				itoc = time.perf_counter()
				time_python_cor = itoc - itic
				
				verbose(mi_jidt_cor, mi_python_cor, cond, "CMI")
	
def test_kraskov_ais():

	vals = [1,2,3]
	kvals = [2,4,8]
	print(f"\n\nTesting average AIS using 1D AR data with history and pure noise\ntesting settings, kraskov k {kvals}, history {vals} and tau {vals}")
	
	source1, source2 = _get_ar_data(seed=SEED)

	time_jidt_cor = np.zeros(np.power(len(vals),3))
	res_jidt_cor = np.zeros(np.power(len(vals),3))
	time_python_cor = np.zeros(np.power(len(vals),3))
	res_python_cor = np.zeros(np.power(len(vals),3))
	time_jidt_uncor = np.zeros(np.power(len(vals),3))
	res_jidt_uncor = np.zeros(np.power(len(vals),3))
	time_python_uncor = np.zeros(np.power(len(vals),3))
	res_python_uncor = np.zeros(np.power(len(vals),3))
	conds = np.zeros([np.power(len(vals),3),3])
	
	count = 0
	for k in kvals:
		for h in vals:
			for t in vals:
				conds[count,:] = [k, h, t]
				settings_j = {'kraskov_k': k,'history': h, 'tau': t}

				settings_p = {'kraskov_k': k,'history': h, 'tau': t}
		
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

	print("k, hist, tau\tJidtKraskovAIS\t\tPythonKraskovAIS")
	print("with history")
	for i in range(len(res_jidt_cor)):
		print(f"{conds[i]}\t{res_jidt_cor[i]}\t{res_python_cor[i]}")
	
	verbose(res_jidt_cor, res_python_cor, "with history", "AIS", local=False)
	
	print("noise")
	for i in range(len(res_jidt_uncor)):
		print(f"{conds[i]}\t{res_jidt_uncor[i]}\t{res_python_uncor[i]}")

	verbose(res_jidt_uncor, res_python_uncor, "noise", "AIS", local=False)
	
	print("\nmean calculation times:")
	print(" JidtKraskovAIS (with history): ", np.mean(time_jidt_cor) )
	print(" PythonKraskovAIS (with history): ", np.mean(time_python_cor) )
	print(" JidtKraskovAIS (noise): ", np.mean(time_jidt_uncor) )
	print(" PythonKraskovAIS (noise): ", np.mean(time_python_uncor) )

	print("\n=========================================================================")
	
	# Test passing to Jidt
	tvals = [0,2]
	avals = [1,2]
	print(f"\n\nTest passing algorithm_num= 2 and theiler_t>2 to JidtKraskovAIS")
	print(f"average MI using 1D gaussian data with covariance 0.4 and lag 1")
	print(f"testing settings kraskov k = 4, history = 2, and lags_mi = 1, algorithm_num {avals} and theiler_t {tvals}\n")
	
	print("algorithm_num, theiler_t")
	for a in avals:
	
		for t in tvals:
			print(f"{a,t}:")
			settings_j = {'kraskov_k': 4,
						'history': 2, 
						"theiler_t": t,
						"algorithm_num": a,
						"noise_level": 0,
						"num_threads": "USE_ALL",
						}
			settings_p = {"kraskov_k": 4,
						'history': 2, 
						"theiler_t": t,
						"algorithm_num": a,
						"noise_level": 0,
						"knn_finder": "scipy_ckdtree",
						"num_threads": "USE_ALL",
						}
			
			jidt_estimator = JidtKraskovAIS(settings_j)
			python_estimator = PythonKraskovAIS(settings_p)

			itic = time.perf_counter()
			mi_jidt_cor = jidt_estimator.estimate(source1)
			itoc = time.perf_counter()
			time_jidt_cor = itoc - itic

			itic = time.perf_counter()
			mi_python_cor = python_estimator.estimate(source1)
			itoc = time.perf_counter()
			time_python_cor = itoc - itic

def test_kraskov_ais_local_values():

	vals = [1,2,3]
	kvals = [2,4,8]
	print(f"\n\nTesting local AIS using 1D AR data with history and pure noise\ntesting settings, kraskov k {kvals}, history {vals} and tau {vals}\t")
	
	source1, source2 = _get_ar_data(seed=SEED)

	time_jidt_cor = np.zeros(np.power(len(vals),3))
	time_python_cor = np.zeros(np.power(len(vals),3))
	time_jidt_uncor = np.zeros(np.power(len(vals),3))
	time_python_uncor = np.zeros(np.power(len(vals),3))
	conds = np.zeros([np.power(len(vals),3),3])
	
	print("k, hist, tau\tJidtKraskovAIS vs PythonKraskovAIS")
	count = 0
	for k in kvals:
		for h in vals:
			for t in vals:
				conds[count,:] = [k, h, t]
				settings_j = {'kraskov_k': k,'history': h, 'tau': t, 'noise_level': 0, 'local_values': True}

				settings_p = {'kraskov_k': k,'history': h, 'tau': t, 'noise_level': 0, 'local_values': True}
		
				jidt_estimator = JidtKraskovAIS(settings=settings_j)
				python_estimator = PythonKraskovAIS(settings=settings_p)

				itic = time.perf_counter()
				res_jidt_cor = jidt_estimator.estimate(source1)
				itoc = time.perf_counter()
				time_jidt_cor[count] = itoc - itic
		
				itic = time.perf_counter()
				res_jidt_uncor = jidt_estimator.estimate(source2)
				itoc = time.perf_counter()
				time_jidt_uncor[count] = itoc - itic
				
				
				itic = time.perf_counter()
				res_python_cor = python_estimator.estimate(source1)
				itoc = time.perf_counter()
				time_python_cor[count] = itoc - itic
				
				itic = time.perf_counter()
				res_python_uncor = python_estimator.estimate(source2)
				itoc = time.perf_counter()
				time_python_uncor[count] = itoc - itic

				verbose(res_jidt_cor, res_python_cor, f"{conds[count,:]} - with hist ", "AIS", local=True, atol=1e-03)
				verbose(res_jidt_uncor, res_python_uncor, f"{conds[count,:]} - noise\t", "AIS", local=True, atol=1e-03)

				count += 1
	
	print("\nmean calculation times:")
	print(" JidtKraskovAIS (with history): ", np.mean(time_jidt_cor) )
	print(" PythonKraskovAIS (with history): ", np.mean(time_python_cor) )
	print(" JidtKraskovAIS (noise): ", np.mean(time_jidt_uncor) )
	print(" PythonKraskovAIS (noise): ", np.mean(time_python_uncor) )
		
def test_kraskov_te():

	expected_mi, source1, source2, target = _get_gauss_data(expand=False, seed=SEED)
	# add delay of one sample
	source1 = source1[1:]
	source2 = source2[1:]
	target = target[:-1]

	vals = [1,3]

	print(f"\n\nTesting average TE using 1D gaussian data with covariance 0.4 and lag 1\n")
	print(f"testing settings history_source (hs), tau_source (ts), history_target (ht), \ntau_target (tt), source_target_delay (std) with {vals} each.\n")


	time_jidt = np.empty(np.power(len(vals),5))
	res_jidt = np.empty(np.power(len(vals),5))
	time_python = np.empty(np.power(len(vals),5))
	res_python = np.empty(np.power(len(vals),5))

	conds = np.empty((np.power(len(vals),5),5))
	
	print("std,ht,tt,hs,ts\t\tJidtKraskovTE\t\tPythonKraskovTE\t\tclose 1e-04")

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
						settings_p = {"kraskov_k": 4, 
							"history_target": ht,
							"history_source": hs,
							"tau_target": tt,
							"tau_source": ts,
							"source_target_delay": hst,
							"noise_level": 0, 
							"num_threads": 1}

						jidt_estimator = JidtKraskovTE(settings_j)
						python_estimator = PythonKraskovTE(settings_p)
						
						itic = time.perf_counter()
						te_jidt = jidt_estimator.estimate(source=source1, target=target)
						itoc = time.perf_counter()
						time_jidt[count] = itoc-itic
						res_jidt[count] = te_jidt

						itic = time.perf_counter()
						te_python = python_estimator.estimate(source=source1, target=target)
						itoc = time.perf_counter()
						time_python[count] = itoc-itic
						res_python[count] = te_python

						count += 1

						print(f"{[hst, ht, tt, hs, ts]}\t\t{te_jidt}\t{te_python}\t{np.isclose(te_jidt, te_python, rtol=1e-04, atol=1e-04)}")

	verbose(res_jidt, res_python, "", "TE", local=False)

	print("\nmean calculation times:")
	print(" JidtKraskovTE: ", np.mean(time_jidt) )
	print(" PythonKraskovTE: ", np.mean(time_python) )

	print("\n=========================================================================")

	# Test passing to Jidt
	tvals = [0,2]
	avals = [1,2]
	print(f"\n\nTest passing algorithm_num= 2 and theiler_t>2 to JidtKraskovTE")
	print(f"average MI using 1D gaussian data with covariance 0.4 and lag 1")
	print(f"testing settings kraskov k = 4, history_target = 2, algorithm_num {avals} and theiler_t {tvals}\n")
	
	print("algorithm_num, theiler_t")
	for a in avals:
	
		for t in tvals:
			print(f"{a,t}:")
			settings_j = {"kraskov_k": 4,
						"history_target": 2,
						"theiler_t": t,
						"algorithm_num": a,
						"noise_level": 0,
						"num_threads": "USE_ALL",
						}
			settings_p = {"kraskov_k": 4,
						"history_target": 2,
						"theiler_t": t,
						"algorithm_num": a,
						"noise_level": 0,
						"knn_finder": "scipy_ckdtree",
						"num_threads": "USE_ALL",
						}
			
			jidt_estimator = JidtKraskovTE(settings_j)
			python_estimator = PythonKraskovTE(settings_p)

			itic = time.perf_counter()
			mi_jidt_cor = jidt_estimator.estimate(source1, target)
			itoc = time.perf_counter()
			time_jidt_cor = itoc - itic

			itic = time.perf_counter()
			mi_python_cor = python_estimator.estimate(source1, target)
			itoc = time.perf_counter()
			time_python_cor = itoc - itic

def test_kraskov_te_local_values():

	vals = [2,4]
	kraskov_k = 4
	knn_list = ['scipy_kdtree', 'scipy_ckdtree', 'sklearn_kdtree', 'sklearn_balltree']
	
	expected_mi, source1, source2, target = _get_gauss_data(expand=True, seed=SEED)

	source1 = source1[1:]
	source2 = source2[1:]
	target = target[:-1]
	
	print(f"\n\nTesting average TE using 1D gaussian data with covariance 0.4 and lag 1\n")
	print(f"testing settings history_source (hs), tau_source (ts), history_target (ht), \ntau_target (tt), source_target_delay (std) with {vals} each \nand kraskov k = {kraskov_k}.")
	print(f"for the implemented knn finder: {knn_list}\n")

	kcount = 0
	for knn in knn_list:
		print(f"\nKNN finder: {knn}\n")
		time_jidt = np.empty(np.power(len(vals),5))
		res_jidt = np.empty((np.power(len(vals),5), 4, len(source1)))
		time_python = np.empty(np.power(len(vals),5))
		res_python = np.empty((np.power(len(vals),5), 4, len(source1)))
		
		conds = np.empty((np.power(len(vals),5),5))

		print("hst,ht,tt,hs,ts\t- avg close 1e-04 - \t local values JidtKraskovTE vs PythonKraskovTE")

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
										"local_values": True,
										"noise_level": 0,
										"kraskov_k": kraskov_k}

							settings_p = {"history_target": ht,
										"history_source": hs,
										"tau_target": tt,
										"tau_source": ts,
										"source_target_delay": hst,
										"local_values": True,
										"knn_finder": knn,
										"noise_level": 0,
										"kraskov_k": kraskov_k}
							
							jidt_estimator = JidtKraskovTE(settings_j)
							python_estimator = PythonKraskovTE(settings_p)
							
							itic = time.perf_counter()
							te_jidt = jidt_estimator.estimate(source=source1, target=target)
							itoc = time.perf_counter()
							time_jidt[count] = itoc-itic
							
							res_jidt[count,kcount,:] = te_jidt

							itic = time.perf_counter()
							te_python = python_estimator.estimate(source=source1, target=target)
							itoc = time.perf_counter()
							time_python[count] = itoc-itic
							
							res_python[count,kcount,:] = te_python

							count += 1

							verbose(te_jidt, te_python, f"{[hst, ht, tt, hs, ts]}  -\t {np.allclose(te_jidt.mean(), te_python.mean() ,atol=1e-04)}\t", "TE", atol=1e-03, local=True)

		kcount += 1

		print("\nmean calculation times:")
		print(" JidtKraskovTE: ", np.mean(time_jidt) )
		print(" PythonKraskovTE: ", np.mean(time_python) )
	
def test_Kraskov_cte():

	
	vals = [1,3]
	
	print(f"\n\nTesting average CTE using 1D mute data - with coupling and no coupling (default k=4)")
	print(f"testing settings history_source, tau_source, history_target, tau_target, history_conditional")
	print(f"tau_conditional, source_target_delay and conditional_target_delay with {vals}\n")
	
	data = _generate_mute_data(n_replications=1)
	source1 = data[0,:]
	target = data[4,:]
	cond = data[3,:]
	nocond = data[5,:]

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

	atol = 1e-03

	print(f"\t\t\t\tJidtKraskovCTE\t\tPythonKraskovCTE\tclose {atol}")
	print("hst,cst,ht,tt,hs,ts,hc,tc\tcte cond \t\tcte cond\t\tuncor cond\tuncor source")

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
										"conditional_target_delay": cst,
										'noise_level': 0,
										"normalise": False,
									}
									
									
									jidt_estimator = JidtKraskovCTE(settings)
									
									itic = time.perf_counter()
									cte_jidt_cond = jidt_estimator.estimate(source1, target, cond)
									itoc = time.perf_counter()
									res_jidt_cond[count] = cte_jidt_cond
									time_jidt_cond[count] = itoc - itic
									
									itic = time.perf_counter()
									cte_jidt_nocond = jidt_estimator.estimate(source1, target, nocond)
									itoc = time.perf_counter()
									res_jidt_nocond[count] = cte_jidt_nocond
									time_jidt_nocond[count] = itoc - itic
									
									python_estimator = PythonKraskovCTE(settings)
									
									itic = time.perf_counter()
									cte_python_cond = python_estimator.estimate(source1, target, cond)
									itoc = time.perf_counter()
									res_python_cond[count] = cte_python_cond
									time_python_cond[count] = itoc - itic
									
									itic = time.perf_counter()
									cte_python_nocond = python_estimator.estimate(source1, target, nocond)
									itoc = time.perf_counter()
									res_python_nocond[count] = cte_python_nocond
									time_python_nocond[count] = itoc - itic
									
									print(f"{hst,cst,ht,tt,hs,ts,hc,tc}\t{cte_jidt_cond}\t{cte_python_cond}\t\t{np.isclose(cte_jidt_cond, cte_python_cond, rtol=atol, atol=atol)}\t{np.isclose(cte_jidt_nocond, cte_python_nocond, rtol=atol, atol=atol)}")

									count += 1

	verbose(res_jidt_cond, res_python_cond, "", "CTE cond", atol=atol)
	verbose(res_jidt_nocond, res_python_nocond, "", "CTE nocond", atol=atol)

	print("\nmean calculation times:")
	print(" JidtGaussianCTE (cond): ", np.mean(time_jidt_cond) )
	print(" PythonGaussianCTE (cond): ", np.mean(time_python_cond) )
	print(" JidtGaussianCTE (nocond): ", np.mean(time_jidt_nocond) )
	print(" PythonGaussianCTE (nocond): ", np.mean(time_python_nocond) )

	print("\n=========================================================================")

	# Test passing to Jidt
	tvals = [0,2]
	avals = [1,2]
	print(f"\n\nTest passing algorithm_num= 2 and theiler_t>2 to JidtKraskovCTE")
	print(f"average MI using 1D gaussian data with covariance 0.4 and lag 1")
	print(f"testing settings kraskov k = 4, algorithm_num {avals} and theiler_t {tvals}\n")
	
	print("algorithm_num, theiler_t")
	for a in avals:
	
		for t in tvals:
			print(f"{a,t}:")
			settings_j = {"kraskov_k": 4,
						"history_target": 2,
						"theiler_t": t,
						"algorithm_num": a,
						"noise_level": 0,
						"num_threads": "USE_ALL",
						}
			settings_p = {"kraskov_k": 4,
						"history_target": 2,
						"theiler_t": t,
						"algorithm_num": a,
						"noise_level": 0,
						"knn_finder": "scipy_ckdtree",
						"num_threads": "USE_ALL",
						}
			
			jidt_estimator = JidtKraskovCTE(settings_j)
			python_estimator = PythonKraskovCTE(settings_p)

			itic = time.perf_counter()
			mi_jidt_cor = jidt_estimator.estimate(source1, target, cond)
			itoc = time.perf_counter()
			time_jidt_cor = itoc - itic

			itic = time.perf_counter()
			mi_python_cor = python_estimator.estimate(source1, target, cond)
			itoc = time.perf_counter()
			time_python_cor = itoc - itic

def test_kraskov_cte_local_values():
	
	vals = [1,2]

	print(f"\n\nTesting local CTE using 1D mute data - correlated and uncorrelated conditional (default k=4)")
	print(f"testing settings history_source, tau_source, history_target, tau_target, history_conditional")
	print(f"tau_conditional, source_target_delay and conditional_target_delay with {vals}\n")
	
	data = _generate_mute_data(n_replications=1)
	source1 = data[0,:]
	target = data[4,:]
	cond = data[3,:]
	nocond = data[5,:]

	time_jidt_cond = np.empty(np.power(len(vals),8))
	time_jidt_nocond = np.empty(np.power(len(vals),8))
	
	time_python_cond = np.empty(np.power(len(vals),8))
	time_python_nocond = np.empty(np.power(len(vals),8))
	
	atol = 1e-03

	print(f"std,ctd,ht,tt,hs,ts,hc,tc\tavg close {atol}\t\tJidtKraskovCTE vs PythonKraskovCTE")

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
										"conditional_target_delay": cst,
										'noise_level': 0,
										'normalise': False,
										"local_values": True}
																		
									jidt_estimator = JidtKraskovCTE(settings)
									
									itic = time.perf_counter()
									cte_jidt_cond = jidt_estimator.estimate(source1, target, cond)
									itoc = time.perf_counter()
									time_jidt_cond[count] = itoc - itic
									
									itic = time.perf_counter()
									cte_jidt_nocond = jidt_estimator.estimate(source1, target, nocond)
									itoc = time.perf_counter()
									time_jidt_nocond[count] = itoc - itic
									
									python_estimator = PythonKraskovCTE(settings)
									
									itic = time.perf_counter()
									cte_python_cond = python_estimator.estimate(source1, target, cond)
									itoc = time.perf_counter()
									time_python_cond[count] = itoc - itic
									
									itic = time.perf_counter()
									cte_python_nocond = python_estimator.estimate(source1, target, nocond)
									itoc = time.perf_counter()
									time_python_nocond[count] = itoc - itic
					
									verbose(cte_jidt_cond, cte_python_cond, f"{hst,cst,ht,tt,hs,ts,hc,tc} corr cond   - {np.isclose(np.mean(cte_jidt_cond), np.mean(cte_python_cond), atol=atol)}", "CTE", local=True, atol=atol) 
									verbose(cte_jidt_nocond, cte_python_nocond, f"{hst,cst,ht,tt,hs,ts,hc,tc} uncorr cond - {np.isclose(np.mean(cte_jidt_cond), np.mean(cte_python_cond), atol=atol)}", "CTE", local=True, atol=atol) 

									count += 1
	print("\nmean calculation times:")
	print(" JidtKraskovCTE (correlated conditional): ", np.mean(time_jidt_cond) )
	print(" PythonKraskovCTE (correlated conditional): ", np.mean(time_python_cond) )
	print(" JidtKraskovCTE (uncorrelated conditional): ", np.mean(time_jidt_nocond) )
	print(" PythonKraskovCTE (uncorrelated conditional): ", np.mean(time_python_nocond) )


#### Test Gaussian estimators
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

	print(f"\n\nTesting average MI using 1D gaussian data with covariance 0.4 - correlated and uncorrelated:")
	print(f"testing settings lag_mi {vals}\n")

	for lags in vals:
		settings = {}
		settings = {"lag_mi": lags,
					"noise_level": 0}

		jidt_estimator = JidtGaussianMI(settings)
		python_estimator = PythonGaussianMI(settings)
		
		itic = time.perf_counter()
		mi_jidt_cor[lags] = jidt_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_jidt_cor[lags] = itoc - itic
		
		itic = time.perf_counter()
		mi_python_cor[lags] = python_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_python_cor[lags] = itoc - itic
		
		itic = time.perf_counter()
		mi_jidt_uncor[lags] = jidt_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		time_jidt_uncor[lags] = itoc - itic
		
		itic = time.perf_counter()
		mi_python_uncor[lags] = python_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		time_python_uncor[lags] = itoc - itic

	print(f"Summary Jidt vs Python GaussianMI testing lags ({vals}):")

	print("MI values:")
	print("correlated data:")
	print("lag\tJidtGaussianMI\t\tPythonGaussianMI")
	for i in vals:
		print(f"{i}\t{mi_jidt_cor[i]}\t{mi_python_cor[i]}")
	
	verbose(mi_jidt_cor, mi_python_cor, "correlated", "MI", local=False)
	

	print("\nuncorrelated data:")
	print("lag\tJidtGaussianMI\t\tPythonGaussianMI")
	for i in vals:
		print(f"{i}\t{mi_jidt_uncor[i]}\t{mi_python_uncor[i]}")

	verbose(mi_jidt_uncor, mi_python_uncor, "uncorrelated", "MI", local=False)
	
	print("\nmean calculation times:")
	print(" JidtGaussianMI (cor): ", np.mean(time_jidt_cor) )
	print(" PythonGaussianMI (cor): ", np.mean(time_python_cor) )
	print(" JidtGaussianMI (uncor): ", np.mean(time_jidt_uncor) )
	print(" PythonGaussianMI (uncor): ", np.mean(time_python_uncor) )

	print("\n=========================================================================")

	# test 2D
	vals = [0,1,2,3]
	print(f"\n\nTesting average MI using 2D mute data with and without coupling")
	print(f"testing settings lag_mi {vals}\n")

	data = _generate_mute_data()

	source1 = data[0,:,:]
	target = data[2,:,:]
	source2 = data[4,:,:]

	mi_jidt_cor = np.zeros(4)
	mi_jidt_uncor = np.zeros(4)
	mi_python_cor = np.zeros(4)
	mi_python_uncor = np.zeros(4)
	time_jidt_cor = np.zeros(4)
	time_jidt_uncor = np.zeros(4)
	time_python_cor = np.zeros(4)
	time_python_uncor = np.zeros(4)


	for lags in vals:
		settings = {"lag_mi": lags}

		jidt_estimator = JidtGaussianMI(settings)
		python_estimator = PythonGaussianMI(settings)
		
		itic = time.perf_counter()
		mi_jidt_cor[lags] = jidt_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_jidt_cor[lags] = itoc - itic
		
		itic = time.perf_counter()
		mi_python_cor[lags] = python_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_python_cor[lags] = itoc - itic

		itic = time.perf_counter()
		mi_jidt_uncor[lags] = jidt_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		time_jidt_uncor[lags] = itoc - itic

		itic = time.perf_counter()
		mi_python_uncor[lags] = python_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		time_python_uncor[lags] = itoc - itic

	print(f"Summary Jidt vs Python GaussianMI 2D input testing lags ({vals}):")

	print("MI values:")
	print("coupled data:")
	print("lag\tJidtGaussianMI\t\tPythonGaussianMI")
	for i in vals:
		print(f"{i}\t{mi_jidt_cor[i]}\t{mi_python_cor[i]}")
	
	verbose(mi_jidt_cor, mi_python_cor, "with coupling", "MI", local=False)


	print("not coupled data:")
	print("lag\tJidtGaussianMI\t\tPythonGaussianMI")
	for i in vals:
		print(f"{i}\t{mi_jidt_uncor[i]}\t{mi_python_uncor[i]}")

	verbose(mi_jidt_uncor, mi_python_uncor, "without coupling", "MI", local=False)

	print("\nmean calculation times:")
	print(" JidtGaussianMI (coupled): ", np.mean(time_jidt_cor) )
	print(" PythonGaussianMI (coupled): ", np.mean(time_python_cor) )
	print(" JidtGaussianMI (not coupled): ", np.mean(time_jidt_uncor) )
	print(" PythonGaussianMI (not coupled): ", np.mean(time_python_uncor) )

	print("\n=========================================================================")
	
	# test mixed dimension input
	d = [1, 2, 3, 5]
	print(f"\n\nTesting average MI using mixed dimensions\ntesting dimensions {d} for var1 and var2 each\n")
	print("Shapes:")
	data = _generate_mute_data(n_replications=5)
	source_o = data[0,:,:]
	target_o = data[2,:,:]
	
	settings = {}
	
	for s in d:
		for t in d:
			
			source1 = source_o[:,:s]
			target = target_o[:,:t]
			
			cond = f"var1: {source1.shape}\tvar2: {target.shape}"

			jidt_estimator = JidtGaussianMI(settings)
			python_estimator = PythonGaussianMI(settings)
		
			itic = time.perf_counter()
			mi_jidt_cor = jidt_estimator.estimate(source1, target)
			itoc = time.perf_counter()
			time_jidt_cor = itoc - itic
			
			itic = time.perf_counter()
			mi_python_cor = python_estimator.estimate(source1, target)
			itoc = time.perf_counter()
			time_python_cor = itoc - itic

			
			verbose(mi_jidt_cor, mi_python_cor, cond, "MI")

def test_gaussian_mi_local_values():
	
	vals = [0,1,2,3]
	
	print(f"\n\nTesting local MI using 1D gaussian data with covariance 0.4 - uncorrelated and uncorrelated")
	print(f"testing settings lag_mi {vals}\n")

	expected_mi, source1, source2, target = _get_gauss_data(expand=True, seed=SEED)
	
	time_jidt_cor = np.zeros(4)
	time_jidt_uncor = np.zeros(4)
	time_python_cor = np.zeros(4)
	time_python_uncor = np.zeros(4)

	print("\nTesting lags:")
	for lags in vals:
		settings = {}
		settings = {"lag_mi": lags,
					"local_values": True,
					"noise_level": 0}
		
		jidt_estimator = JidtGaussianMI(settings)
		python_estimator = PythonGaussianMI(settings)
		
		itic = time.perf_counter()
		mi_jidt_cor = jidt_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_jidt_cor[lags] = itoc - itic

		itic = time.perf_counter()
		mi_python_cor = python_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_python_cor[lags] = itoc - itic

		itic = time.perf_counter()
		mi_jidt_uncor = jidt_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		time_jidt_uncor[lags] = itoc - itic
		
		itic = time.perf_counter()
		mi_python_uncor = python_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		time_python_uncor[lags] = itoc - itic
		
		verbose(mi_jidt_cor, mi_python_cor, lags, "MI (corr)", local=True)
		verbose(mi_jidt_uncor, mi_python_uncor, lags, "MI (uncorr)", local=True)
	
	print("\nmean calculation times:")
	print(" JidtGaussianMI (corr): ", np.mean(time_jidt_cor) )
	print(" PythonGaussianMI (corr): ", np.mean(time_python_cor) )
	print(" JidtGaussianMI (uncorr): ", np.mean(time_jidt_uncor) )
	print(" PythonGaussianMI (uncorr): ", np.mean(time_python_uncor) )

	print("\n=========================================================================")

	# test 2D
	print(f"\n\nTesting local MI using 2D mute data with and without coupling")
	print(f"testing settings lag_mi {vals}\n")

	data = _generate_mute_data()

	source1 = data[0,:,:]
	target = data[2,:,:]
	source2 = data[4,:,:]

	vals = [0,1,2,3]
	
	time_jidt_cor = np.zeros(len(vals))
	time_jidt_uncor = np.zeros(len(vals))
	time_python_cor = np.zeros(len(vals))
	time_python_uncor = np.zeros(len(vals))

	print("Testing lags:")
	for lags in vals:
		settings = {}
		settings = {"lag_mi": lags,
					'local_values': True,
					'discretise_method': 'max_ent'}

		# cor
		jidt_estimator = JidtGaussianMI(settings)
		itic = time.perf_counter()
		mi_jidt_cor = jidt_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_jidt_cor[lags] = itoc - itic
		
		python_estimator = PythonGaussianMI(settings)
		itic = time.perf_counter()
		mi_python_cor = python_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_python_cor[lags] = itoc - itic

		# uncor
		jidt_estimator = JidtGaussianMI(settings)
		itic = time.perf_counter()
		mi_jidt_uncor = jidt_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		time_jidt_uncor[lags] = itoc - itic

		python_estimator = PythonGaussianMI(settings)
		itic = time.perf_counter()
		mi_python_uncor = python_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		time_python_uncor[lags] = itoc - itic
		
		verbose(mi_jidt_cor, mi_python_cor, lags, "MI (coupled) 2D input", local=True, atol=1e-03)
		verbose(mi_jidt_uncor, mi_python_uncor, lags, "MI (not couled) 2D input", local=True, atol=1e-03)

	print("\nmean calculation times:")
	print(" JidtGaussianMI (coupled): ", np.mean(time_jidt_cor) )
	print(" PythonGaussianMI (coupled): ", np.mean(time_python_cor) )
	print(" JidtGaussianMI (not coupled): ", np.mean(time_jidt_uncor) )
	print(" PythonGaussianMI (not coupled): ", np.mean(time_python_uncor) )

def test_gaussian_cmi():

	cmi_jidt_cor = np.zeros(8)
	cmi_python_cor = np.zeros(8)
	cmi_jidt_uncor = np.zeros(8)
	cmi_python_uncor = np.zeros(8)
	time_jidt_cor = np.zeros(8)
	time_python_cor = np.zeros(8)
	time_jidt_uncor = np.zeros(8)
	time_python_uncor = np.zeros(8)
	
	vals = [0.2, 0.4, 0.6, 0.8]

	print(f"\n\nTesting average CMI using 1D gaussian data with different \ncovariances: {vals} - uncorrelated conditional vs uncorrelated source\n")

	count = 0
	for i in vals:

		expected_mi, source1, source2, target = _get_gauss_data(expand=True, covariance=i, seed=SEED)

		settings={}
		
		jidt_estimator = JidtGaussianCMI(settings)
		python_estimator = PythonGaussianCMI(settings)
		
		itic = time.perf_counter()
		cmi_jidt_cor[count] = jidt_estimator.estimate(source1, target, source2)
		itoc = time.perf_counter()
		time_jidt_cor[count] = itoc - itic

		itic = time.perf_counter()
		cmi_python_cor[count] = python_estimator.estimate(source1, target, source2)
		itoc = time.perf_counter()
		time_python_cor[count] += itoc - itic

		itic = time.perf_counter()
		cmi_jidt_uncor[count] = jidt_estimator.estimate(source2, target, source1)
		itoc = time.perf_counter()
		time_jidt_uncor[count] += itoc - itic

		itic = time.perf_counter()
		cmi_python_uncor[count] = python_estimator.estimate(source2, target, source1)
		itoc = time.perf_counter()
		time_python_uncor[count] += itoc - itic

		count += 1 

	print("cov\tJidtGaussianCMI\t\tPythonGaussianCMI")
	print("uncorr conditional")
	for i in range(len(vals)):
		print(f"{vals[i]}\t{cmi_jidt_cor[i]}\t{cmi_python_cor[i]}")
	verbose(cmi_jidt_cor, cmi_python_cor, "", "CMI", local=False)
	print("uncorr source")
	for i in range(len(vals)):
		print(f"{vals[i]}\t{cmi_jidt_uncor[i]}\t{cmi_python_uncor[i]}")
	verbose(cmi_jidt_uncor, cmi_python_uncor, "", "CMI", local=False)

	print("\nmean calculation times:")
	print(" JidtGaussianCMI (uncorrelated conditional): ", np.mean(time_jidt_cor) )
	print(" PythonGaussianCMI (uncorrelated conditional): ", np.mean(time_python_cor) )
	print(" JidtGaussianCMI (uncorrelated source): ", np.mean(time_jidt_uncor) )
	print(" PythonGaussianCMI (uncorrelated source): ", np.mean(time_python_uncor) )

	print("\n=========================================================================")

	# test mixed dimension input
	print(f"\n\nTesting average CMI using 2D mute data - uncorrelated conditional vs uncorrelated source\n")

	data = _generate_mute_data()

	source1 = data[0,:,:]
	target = data[2,:,:]
	source2 = data[4,:,:]

	settings={}
	
	time_jidt = 0
	time_python = 0

	jidt_estimator = JidtGaussianCMI(settings)
	python_estimator = PythonGaussianCMI(settings)
	
	itic = time.perf_counter()
	cmi_jidt = jidt_estimator.estimate(source1, target, source2)
	itoc = time.perf_counter()
	time_jidt += itoc - itic

	itic = time.perf_counter()
	cmi_python = python_estimator.estimate(source1, target, source2)
	itoc = time.perf_counter()
	time_python += itoc - itic

	verbose(cmi_jidt, cmi_python, f"uncorrelated conditional", "CMI", local=False)

	itic = time.perf_counter()
	cmi_jidt = jidt_estimator.estimate(source2, target, source1)
	itoc = time.perf_counter()
	time_jidt += itoc - itic

	itic = time.perf_counter()
	cmi_python = python_estimator.estimate(source2, target, source1)
	itoc = time.perf_counter()
	time_python += itoc - itic

	verbose(cmi_jidt, cmi_python, "uncorrelated source", "CMI", local=False)

	print("\nmean calculation times:")
	print(" JidtGaussianCMI: ", np.mean(time_jidt) )
	print(" PythonGaussianCMI: ", np.mean(time_python) )

	print("\n=========================================================================")

	# test mixed dimension input
	d = [1, 2, 3]

	print(f"\n\nTesting average MI using mixed dimensions\ntesting dimensions {d} for var1, var2 and cond each\n")
	print("Shapes:")
	data = _generate_mute_data()

	source_o = data[0,:,:]
	target_o = data[2,:,:]
	cond_o = data[4,:,:]
	
	settings = {}
	
	for s in d:
		for t in d:
			for c in d:
			
				source1 = source_o[:,:s]
				target = target_o[:,:t]
				conditional = cond_o[:,:c]
				
				cond = f"var1: {source1.shape} var2: {target.shape} cond: {conditional.shape}"

				jidt_estimator = JidtGaussianCMI(settings)
				python_estimator = PythonGaussianCMI(settings)
			
				itic = time.perf_counter()
				mi_jidt_cor = jidt_estimator.estimate(source1, target, conditional)
				itoc = time.perf_counter()
				time_jidt_cor = itoc - itic
				
				itic = time.perf_counter()
				mi_python_cor = python_estimator.estimate(source1, target, conditional)
				itoc = time.perf_counter()
				time_python_cor = itoc - itic

				
				verbose(mi_jidt_cor, mi_python_cor, cond, "MI")

def test_gaussian_cmi_local_values():


	vals = [0.2, 0.4, 0.6, 0.8]

	print(f"\n\nTesting local CMI using 1D gaussian data with different \ncovariances: {vals} - uncorrelated conditional vs uncorrelated source\n")

	cmi_jidt_cor = np.zeros(len(vals))
	cmi_python_cor = np.zeros(len(vals))
	cmi_jidt_uncor = np.zeros(len(vals))
	cmi_python_uncor = np.zeros(len(vals))
	time_jidt_cor = np.zeros(len(vals))
	time_python_cor = np.zeros(len(vals))
	time_jidt_uncor = np.zeros(len(vals))
	time_python_uncor = np.zeros(len(vals))
	
	print("Tested cov\t\tJidtGaussianCMI vs PythonGaussianCMI")
	count = 0
	for i in vals:

		expected_mi, source1, source2, target = _get_gauss_data(expand=True, covariance=i, seed=SEED)

		settings={"local_values": True}
		
		jidt_estimator = JidtGaussianCMI(settings)
		python_estimator = PythonGaussianCMI(settings)
		
		itic = time.perf_counter()
		cmi_jidt = jidt_estimator.estimate(source1, target, source2)
		cmi_jidt_cor[count] = np.mean(cmi_jidt)
		itoc = time.perf_counter()
		time_jidt_cor[count] = itoc - itic
		
		itic = time.perf_counter()
		cmi_python = python_estimator.estimate(source1, target, source2)
		cmi_python_cor[count] = np.mean(cmi_python)
		itoc = time.perf_counter()
		time_python_cor[count] += itoc - itic

		verbose(cmi_jidt, cmi_python, i, "CMI (corr)", local=True)

		itic = time.perf_counter()
		cmi_jidt = jidt_estimator.estimate(source1, target, source2)
		cmi_jidt_uncor[count] = np.mean(cmi_jidt)
		itoc = time.perf_counter()
		time_jidt_uncor[count] += itoc - itic

		itic = time.perf_counter()
		cmi_python = python_estimator.estimate(source1, target, source2)
		cmi_python_uncor[count] = np.mean(cmi_python)
		itoc = time.perf_counter()
		time_python_uncor[count] += itoc - itic

		verbose(cmi_jidt, cmi_python, i, "CMI (uncorr)", local=True)

		count += 1 

	print("\nAverages of local cmi:")
	print("cov\tJidtGaussianCMI\t\tPythonGaussianCMI")
	print("uncorr conditional")
	for i in range(len(vals)):
		print(f"{vals[i]}\t{cmi_jidt_cor[i]}\t{cmi_python_cor[i]}")
	verbose(cmi_jidt_cor, cmi_python_cor, "", "CMI", local=False)
	print("uncorr source")
	for i in range(len(vals)):
		print(f"{vals[i]}\t{cmi_jidt_uncor[i]}\t{cmi_python_uncor[i]}")
	verbose(cmi_jidt_uncor, cmi_python_uncor, "", "CMI", local=False)

	print("\nmean calculation times:")
	print(" JidtGaussianCMI (uncorrelated conditional): ", np.mean(time_jidt_cor) )
	print(" PythonGaussianCMI (uncorrelated conditional): ", np.mean(time_python_cor) )
	print(" JidtGaussianCMI (uncorrelated source): ", np.mean(time_jidt_uncor) )
	print(" PythonGaussianCMI (uncorrelated source): ", np.mean(time_python_uncor) )

	print("\n=========================================================================")

	print(f"\n\nTesting average CMI using 2D mute data - uncorrelated conditional vs uncorrelated source\n")

	data = _generate_mute_data()

	source1 = data[0,:,:]
	target = data[2,:,:]
	source2 = data[4,:,:]

	settings={'local_values': True}
	time_python=0
	time_jidt=0

	jidt_estimator = JidtGaussianCMI(settings)
	python_estimator = PythonGaussianCMI(settings)
	
	itic = time.perf_counter()
	mi_jidt = jidt_estimator.estimate(source1, target, source2)
	itoc = time.perf_counter()
	time_jidt += itoc - itic
	
	itic = time.perf_counter()
	mi_python = python_estimator.estimate(source1, target, source2)
	itoc = time.perf_counter()
	time_python += itoc - itic

	verbose(mi_jidt, mi_python, f"uncorrelated conditional", "CMI", local=True)

	itic = time.perf_counter()
	mi_jidt = jidt_estimator.estimate(source2, target, source1)
	itoc = time.perf_counter()
	time_jidt += itoc - itic

	itic = time.perf_counter()
	mi_python = python_estimator.estimate(source2, target, source1)
	itoc = time.perf_counter()
	time_python += itoc - itic
	
	verbose(mi_jidt, mi_python, f"uncorrelated source", "CMI", local=True)

	print("\nmean calculation times:")
	print(" JidtGaussianCMI: ", np.mean(time_jidt) )
	print(" PythonGaussianCMI: ", np.mean(time_python) )
	
def test_gaussian_ais():

	print(f"\n\nTesting average AIS using 1D AR data with history and pure noise\n")
	
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
	
	conds = np.zeros([np.power(len(vals),2),2])

	count = 0

	for h in vals:
		for t in vals:

			conds[count,:] = [h,t]
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
	
	print(f"Summary Jidt vs Python GaussianAIS 1D data testing history ({vals}) and tau ({vals}):")

	print("hist,tau\tJidtGaussianAIS\t\tPythonGaussianAIS")
	print("AR with history")
	for i in range(len(res_jidt_cor)):
		print(f"{conds[i]}\t{res_jidt_cor[i]}\t{res_python_cor[i]}")

	verbose(res_jidt_cor, res_python_cor, "", "AIS (with hist)", local=True)
	
	print("noise")
	for i in range(len(res_jidt_uncor)):
		print(f"{conds[i]}\t{res_jidt_uncor[i]}\t{res_python_uncor[i]}")

	verbose(res_jidt_uncor, res_python_uncor, "", "AIS (no hist)", local=True)
	
	print("\nmean calculation times:")
	print(" JidtGaussianAIS (with history): ", np.mean(time_jidt_cor) )
	print(" PythonGaussianAIS (with history): ", np.mean(time_python_cor) )
	print(" JidtGaussianAIS (noise): ", np.mean(time_jidt_uncor) )
	print(" PythonGaussianAIS (noise): ", np.mean(time_python_uncor) )

def test_gaussian_ais_local_values():

	print(f"\n\nTesting local AIS using 1D AR data with history and pure noise\n")
	
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
			
			verbose(ais_jidt, ais_python, [h, t], "AIS (with hist)", local=True)
					
			count += 1

	print("\nmean calculation times:")
	print(" JidtGaussianAIS: ", np.mean(time_jidt) )
	print(" PythonGaussianAIS: ", np.mean(time_python) )

def test_gaussian_te():

	print(f"\n\nTesting average TE using 1D gaussian data with covariance 0.4 and lag 1\n")

	expected_mi, source1, source2, target = _get_gauss_data(expand=True, seed=SEED)

	source1 = source1[1:]
	source2 = source2[1:]
	target = target[:-1]

	vals = [1,3]

	time_jidt = np.empty(np.power(len(vals),5))
	res_jidt = np.empty(np.power(len(vals),5))
	time_python = np.empty(np.power(len(vals),5))
	res_python = np.empty(np.power(len(vals),5))

	conds = np.empty((np.power(len(vals),5),5))

	print("hst,ht,tt,hs,ts\t\tJidtGaussianTE\t\tPythonGaussianTE\tclose 1e-03")

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

def test_gaussian_te_local_values():

	print(f"\n\nTesting average TE using 1D gaussian data with covariance 0.4 and lag 1\n")

	expected_mi, source1, source2, target = _get_gauss_data(expand=True, seed=SEED)

	source1 = source1[1:]
	source2 = source2[1:]
	target = target[:-1]

	vals = [2,4]
	
	time_jidt = np.empty(np.power(len(vals),5))
	res_jidt = np.empty(np.power(len(vals),5))
	time_python = np.empty(np.power(len(vals),5))
	res_python = np.empty(np.power(len(vals),5))
	
	conds = np.empty((np.power(len(vals),5),5))
	
	print("hst,ht,tt,hs,ts\t\tJidtGaussianTE vs PythonGaussianTE")

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
						
						verbose(te_jidt, te_python, [hst, ht, tt, hs, ts], "TE", local=True)
	
	print("\nmean calculation times:")
	print(" JidtGaussianTE: ", np.mean(time_jidt) )
	print(" PythonGaussianTE: ", np.mean(time_python) )

def test_gaussian_cte():

	vals = [1,3]

	print(f"\n\nTesting average CTE using 1D mute data - correlated and uncorrelated conditional\n")
	print(f"testing settings history_source, tau_source, history_target, tau_target, history_conditional")
	print(f"tau_conditional, source_target_delay and conditional_target_delay with {vals}\n")
	
	data = _generate_mute_data(n_replications=1)
	source1 = data[0,:]
	target = data[4,:]
	cond = data[3,:]
	nocond = data[5,:]

	time_jidt_cond = np.empty(np.power(len(vals),8))
	res_jidt_cond = np.empty(np.power(len(vals),8))
	time_jidt_nocond = np.empty(np.power(len(vals),8))
	res_jidt_nocond = np.empty(np.power(len(vals),8))
	
	time_python_cond = np.empty(np.power(len(vals),8))
	res_python_cond = np.empty(np.power(len(vals),8))
	time_python_nocond = np.empty(np.power(len(vals),8))
	res_python_nocond = np.empty(np.power(len(vals),8))
	
	conds = np.empty((np.power(len(vals),5),8))

	atol = 1e-03

	print(f"\t\t\t\tJidtGaussianCTE\t\tPythonGaussianCTE\t\tclose {atol}")
	print("hst,cst,ht,tt,hs,ts,hc,tc\tcte cor cond \t\tcte cor cond\t\t\tcor cond\tuncor cond")

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
									res_jidt_cond[count] = cte_jidt_cond
									time_jidt_cond[count] = itoc - itic
									
									itic = time.perf_counter()
									cte_jidt_nocond = jidt_estimator.estimate(source1, target, nocond)
									itoc = time.perf_counter()
									res_jidt_nocond[count] = cte_jidt_nocond
									time_jidt_nocond[count] = itoc - itic
									
									python_estimator = PythonGaussianCTE(settings)
									
									itic = time.perf_counter()
									cte_python_cond = python_estimator.estimate(source1, target, cond)
									itoc = time.perf_counter()
									res_python_cond[count] = cte_python_cond
									time_python_cond[count] = itoc - itic
									
									itic = time.perf_counter()
									cte_python_nocond = python_estimator.estimate(source1, target, nocond)
									itoc = time.perf_counter()
									res_python_nocond[count] = cte_python_nocond
									time_python_nocond[count] = itoc - itic
									
									print(f"{hst,cst,ht,tt,hs,ts,hc,tc}\t{cte_jidt_cond}\t{cte_python_cond}\t\t{np.isclose(cte_jidt_cond, cte_python_cond, rtol=atol, atol=atol)}\t\t{np.isclose(cte_jidt_nocond, cte_python_nocond, rtol=atol, atol=atol)}")

									count += 1 

	verbose(res_jidt_cond, res_python_cond, "correlated conditional", "CTE", atol=1e-04)
	verbose(res_jidt_nocond, res_python_nocond, "uncorrelated conditional" , "CTE", atol=1e-04)

	print("\nmean calculation times:")
	print(" JidtGaussianCTE (correlated conditional): ", np.mean(time_jidt_cond) )
	print(" PythonGaussianCTE (correlated conditional): ", np.mean(time_python_cond) )
	print(" JidtGaussianCTE (uncorrelated conditional): ", np.mean(time_jidt_nocond) )
	print(" PythonGaussianCTE (uncorrelated conditional): ", np.mean(time_python_nocond) )

def test_gaussian_cte_local_values():
	
	vals = [2,4]

	print(f"\n\nTesting local CTE using 1D mute data - correlated and uncorrelated conditional")
	print(f"testing settings history_source, tau_source, history_target, tau_target, history_conditional")
	print(f"tau_conditional, source_target_delay and conditional_target_delay with {vals}\n")
	
	data = _generate_mute_data(n_replications=1)
	source1 = data[0,:]
	target = data[4,:]
	cond = data[3,:]
	nocond = data[5,:]

	time_jidt_cond = np.empty(np.power(len(vals),8))
	time_jidt_nocond = np.empty(np.power(len(vals),8))
	
	time_python_cond = np.empty(np.power(len(vals),8))
	time_python_nocond = np.empty(np.power(len(vals),8))
	
	atol = 1e-03

	print("std,ctd,ht,tt,hs,ts,hc,tc\t\t\tJidtGaussianCTE vs PythonGaussianCTE")

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
										"conditional_target_delay": cst,
										'noise_level': 0,
										'normalise': True,
										"local_values": True}
																		
									jidt_estimator = JidtGaussianCTE(settings)
									
									itic = time.perf_counter()
									cte_jidt_cond = jidt_estimator.estimate(source1, target, cond)
									itoc = time.perf_counter()
									time_jidt_cond[count] = itoc - itic
									
									itic = time.perf_counter()
									cte_jidt_nocond = jidt_estimator.estimate(source1, target, nocond)
									itoc = time.perf_counter()
									time_jidt_nocond[count] = itoc - itic
									
									python_estimator = PythonGaussianCTE(settings)
									
									itic = time.perf_counter()
									cte_python_cond = python_estimator.estimate(source1, target, cond)
									itoc = time.perf_counter()
									time_python_cond[count] = itoc - itic
									
									itic = time.perf_counter()
									cte_python_nocond = python_estimator.estimate(source1, target, nocond)
									itoc = time.perf_counter()
									time_python_nocond[count] = itoc - itic
									
									verbose(cte_jidt_cond, cte_python_cond, f"{hst,cst,ht,tt,hs,ts,hc,tc} corr conditional", "CTE", local=True, atol=atol) 
									verbose(cte_jidt_nocond, cte_python_nocond, f"{hst,cst,ht,tt,hs,ts,hc,tc} uncorr conditional", "CTE", local=True, atol=atol) 

									count += 1
	print("\nmean calculation times:")
	print(" JidtGaussianCTE (correlated conditional): ", np.mean(time_jidt_cond) )
	print(" PythonGaussianCTE (correlated conditional): ", np.mean(time_python_cond) )
	print(" JidtGaussianCTE (uncorrelated conditional): ", np.mean(time_jidt_nocond) )
	print(" PythonGaussianCTE (uncorrelated conditional): ", np.mean(time_python_nocond) )


#### Test Discrete estimators
def test_discrete_mi():

	vals = [2,5,8,32]
	lvals = [0,1,2,3]

	# test 1D gaussian
	print(f"\n\nTesting average MI using 1D gaussian data with covariance 0.4 - correlated and uncorrelated")
	print(f"testing settings lag_mi {lvals}, n_discrete_bins {vals} and discrete_method max_ent and equal\n")
	
	expected_mi, source1, source2, target = _get_gauss_data(expand=True, seed=SEED)
	
	for m in ['max_ent','equal']:
		print(f"\n--- discretise_method: {m}\n")
		mi_jidt_cor = np.zeros(np.power(len(vals),2))
		mi_python_cor = np.zeros(np.power(len(vals),2))
		time_jidt_cor = np.zeros(np.power(len(vals),2))
		time_python_cor = np.zeros(np.power(len(vals),2))
		
		mi_jidt_uncor = np.zeros(np.power(len(vals),2))
		mi_python_uncor = np.zeros(np.power(len(vals),2))
		time_jidt_uncor = np.zeros(np.power(len(vals),2))
		time_python_uncor = np.zeros(np.power(len(vals),2))

		conds = np.empty((np.power(len(vals),2),2))
		
		count = 0
		for l in lvals:
			for i in vals:
				conds[count,:] = [l,i]
				settings = {'discretise_method': m,
							'n_discrete_bins': i,
							'lag_mi': l}
				
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
				
		atol = 1e-03
		print(f"Summary Jidt vs Python DiscreteMI discretised 1D gaussian data using {m}:")
		print(f"lags, nbins\tJidtDiscreteMI\t\tPythonDiscreteMI\tclose {atol}")
		print("correlated data:")
		for i in range(count):
			print(f"{conds[i]}   \t{mi_jidt_cor[i]}\t{mi_python_cor[i]}\t{np.isclose(mi_jidt_cor[i], mi_python_cor[i], atol=atol)}")

		print("\nuncorrelated data:")
		for i in range(count):
			print(f"{conds[i]}   \t{mi_jidt_uncor[i]}\t{mi_python_uncor[i]}\t{np.isclose(mi_jidt_uncor[i], mi_python_uncor[i], atol=atol)}")
		
		verbose(mi_jidt_cor, mi_python_cor, "correlated", "MI", local=False, atol=1e-03)
		verbose(mi_jidt_uncor, mi_python_uncor, "uncorrelated", "MI", local=False, atol=1e-03)

		print("\nmean calculation times:")
		print(" JidtDiscreteMI (correlated): ", np.mean(time_jidt_cor) )
		print(" PythonDiscreteMI (correlated): ", np.mean(time_python_cor) )
		print(" JidtDiscreteMI (uncorrelated): ", np.mean(time_jidt_uncor) )
		print(" PythonDiscreteMI (uncorrelated): ", np.mean(time_python_uncor) )
		
	print("\n=========================================================================")

	# test 1D bin data
	print(f"\n\n\nTesting average MI using 1D binary data with memory and discrete_method none\n")
	
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
	verbose(mi_jidt, mi_python, "", "MI", atol=1e-03)

	print("\n=========================================================================")

	# test 2D
	lvals = [0,1,2,3]

	print(f"\n\nTesting average MI using 2D mute data - correlated and uncorrelated")
	print(f"testing settings lag_mi {lvals}, n_discrete_bins 2 and discrete_method max_ent and equal\n")
	
	data = _generate_mute_data(n_replications=2)
	source1 = data[0,:,:]
	target = data[2,:,:]
	source2 = data[4,:,:]
	
	for m in ['max_ent','equal']:

		print(f"\n--- discrete_method: {m}\n")
		
		mi_jidt_cor = np.zeros(len(vals))
		mi_python_cor = np.zeros(len(vals))
		time_jidt_cor = np.zeros(len(vals))
		time_python_cor = np.zeros(len(vals))
		mi_jidt_uncor = np.zeros(len(vals))
		mi_python_uncor = np.zeros(len(vals))
		time_jidt_uncor = np.zeros(len(vals))
		time_python_uncor = np.zeros(len(vals))
		
		count = 0
		for l in lvals:
			settings = {'discretise_method': m,
						'n_discrete_bins': 2,
						'lag_mi': l}
			
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
			
		print(f"Summary Jidt vs Python DiscreteMI discretised 2D mute data using {m}:")

		print("lags\tJidtDiscreteMI\t\tPythonDiscreteMI")
		print("correlated data:")
		for i in range(len(vals)):
			print(f"{lvals[i]}   \t{mi_jidt_cor[i]}\t{mi_python_cor[i]}")

		print("\nuncorrelated data:")
		for i in range(len(vals)):
			print(f"{lvals[i]}   \t{mi_jidt_uncor[i]}\t{mi_python_uncor[i]}")
		
		verbose(mi_jidt_cor, mi_python_cor, "correlated", "MI", local=False, atol=1e-03)
		verbose(mi_jidt_uncor, mi_python_uncor, "uncorrelated", "MI", local=False, atol=1e-03)

		print("\nmean calculation times:")
		print(" JidtDiscreteMI (correlated): ", np.mean(time_jidt_cor) )
		print(" PythonDiscreteMI (correlated): ", np.mean(time_python_cor) )
		print(" JidtDiscreteMI (uncorrelated): ", np.mean(time_jidt_uncor) )
		print(" PythonDiscreteMI (uncorrelated): ", np.mean(time_python_uncor) )

	print("\n=========================================================================")

	# test mixed dimension input
	d = [1, 2, 3, 5]

	print(f"\n\nTesting average MI using mixed dimensions\ntesting dimensions {d} for var1 and var2 each\n")
	print("Shapes:")
	
	data = _generate_mute_data(n_replications=5)
	source_o = data[0,:,:]
	target_o = data[2,:,:]
	
	settings = {'discretise_method': 'equal',
						'n_discrete_bins': 2,
						'lag_mi': 2}
	
	d = [1, 2, 3, 5]

	for s in d:
		for t in d:
			
			source1 = source_o[:,:s]
			target = target_o[:,:t]
			
			cond = f"var1: {source1.shape}\tvar2: {target.shape}"

			jidt_estimator = JidtDiscreteMI(settings=settings)
			python_estimator = PythonDiscreteMI(settings=settings)
			
			itic = time.perf_counter()
			mi_jidt_cor = jidt_estimator.estimate(source1, target)
			itoc = time.perf_counter()
			time_jidt_cor = itoc - itic
			
			itic = time.perf_counter()
			mi_python_cor = python_estimator.estimate(source1, target)
			itoc = time.perf_counter()
			time_python_cor = itoc - itic

			
			verbose(mi_jidt_cor, mi_python_cor, cond, "MI")

def test_discrete_mi_local_values():

	atol = 1e-03

	vals = [0,1,2,3]
	print(f"\n\nTesting local MI using 1D gaussian data with covariance 0.4 - correlated and uncorrelated")
	print(f"testing settings lag_mi {vals}, n_discrete_bins 2 and discrete_method max_ent\n")
	
	expected_mi, source1, source2, target = _get_gauss_data(expand=True, seed=SEED)

	vals = [0,1,2,3]

	jidt_time = 0.0
	python_time = 0.0

	mi_jidt_cor = np.zeros(4)
	mi_jidt_uncor = np.zeros(4)
	mi_python_cor = np.zeros(4)
	mi_python_uncor = np.zeros(4)

	print("lags")
	count=0
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
		jidt_time += itoc - itic
		
		python_estimator = PythonDiscreteMI(settings)
		itic = time.perf_counter()
		mi_python = python_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		python_time += itoc - itic
		
		verbose(mi_jidt, mi_python, f"{lags}", "MI (correlated)  ", local=True, atol=atol)

		jidt_estimator = JidtDiscreteMI(settings)
		itic = time.perf_counter()
		mi_jidt = jidt_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		jidt_time += itoc - itic
		
		python_estimator = PythonDiscreteMI(settings)
		itic = time.perf_counter()
		mi_python = python_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		python_time += itoc - itic
		
		verbose(mi_jidt, mi_python, f"{lags}", "MI (uncorrelated)", local=True, atol=atol)
	
	print("\nmean calculation times:")
	print(" JidtDiscreteMI: ", np.mean(jidt_time) )
	print(" PythonDiscreteMI: ", np.mean(python_time) )

	print("\n=========================================================================")

	# test 2D 
	vals = [0,1,2,3]

	print(f"\n\nTesting local MI using 2D mute data - correlated and uncorrelated")
	print(f"testing settings lag_mi {vals}, n_discrete_bins 2 and discrete_method max_ent\n")
	
	data = _generate_mute_data()

	source1 = data[0,:,:]
	target = data[2,:,:]
	source2 = data[4,:,:]

	time_jidt_cor = np.zeros(len(vals))
	time_jidt_uncor = np.zeros(len(vals))
	time_python_cor = np.zeros(len(vals))
	time_python_uncor = np.zeros(len(vals))
	
	print("lags")
	for lags in vals:
		settings = {}
		settings = {"lag_mi": lags,
					'local_values': True,
					'discretise_method': 'max_ent'}

		# cor
		jidt_estimator = JidtDiscreteMI(settings)
		itic = time.perf_counter()
		mi_jidt_cor = jidt_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_jidt_cor[lags] = itoc - itic
		
		python_estimator = PythonDiscreteMI(settings)
		itic = time.perf_counter()
		mi_python_cor = python_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_python_cor[lags] = itoc - itic

		verbose(mi_jidt_cor, mi_python_cor, lags, "MI (correlated)   2D input", local=True, atol=atol)

		# uncor
		jidt_estimator = JidtDiscreteMI(settings)
		itic = time.perf_counter()
		mi_jidt_uncor = jidt_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		time_jidt_uncor[lags] = itoc - itic

		python_estimator = PythonDiscreteMI(settings)
		itic = time.perf_counter()
		mi_python_uncor = python_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		time_python_uncor[lags] = itoc - itic
		
		verbose(mi_jidt_uncor, mi_python_uncor, lags, "MI (uncorrelated) 2D input", local=True, atol=atol)
	
	print("\nmean calculation times:")
	print(" JidtDiscrete (cor): ", np.mean(time_jidt_cor) )
	print(" PythonDiscreteMI (cor): ", np.mean(time_python_cor) )
	print(" JidtDiscreteMI (uncor): ", np.mean(time_jidt_uncor) )
	print(" PythonDiscreteMI (uncor): ", np.mean(time_python_uncor) )

def test_discrete_cmi():

	vals = [2,5,8]

	print(f"\n\nTesting average CMI using 1D gaussian data with covariance 0.4 - uncorrelated \nconditional and uncorrelated source")
	print(f"testing settings n_discrete_bins {vals} and discrete_method max_ent and equal")
	
	expected_mi, source1, source2, target = _get_gauss_data(expand=True, seed=SEED)
	
	for m in ['max_ent','equal']:
		print(f"\n--- discrete_method: {m}\n")

		mi_jidt_cor = np.zeros(len(vals))
		mi_python_cor = np.zeros(len(vals))
		time_jidt_cor = np.zeros(len(vals))
		time_python_cor = np.zeros(len(vals))
		
		mi_jidt_uncor = np.zeros(len(vals))
		mi_python_uncor = np.zeros(len(vals))
		time_jidt_uncor = np.zeros(len(vals))
		time_python_uncor = np.zeros(len(vals))

		count = 0
		for i in vals:
			settings = {'discretise_method': m,
						'n_discrete_bins': i,
						'noise_level': 0,
						'normalise': False,}
			
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
			
		print(f"Summary Jidt vs Python DiscreteCMI discretised 1D gaussian data using {m}:")

		print("nbins\tJidtDiscreteCMI\t\tPythonDiscreteCMI")
		print("uncorrelated conditional:")
		for i in range(len(vals)):
			print(f"{vals[i]}\t{mi_jidt_cor[i]}\t{mi_python_cor[i]}")

		print("\nuncorrelated source:")
		for i in range(len(vals)):
			print(f"{vals[i]}\t{mi_jidt_uncor[i]}\t{mi_python_uncor[i]}")

		verbose(mi_jidt_cor, mi_python_cor, "", "CMI (uncorrelated conditional)", local=False)
		verbose(mi_jidt_uncor, mi_python_uncor, "", "CMI (uncorrelated source)", local=False)

		print("\nmean calculation times:")
		print(" JidtDiscreteCMI(uncorrelated conditional): ", np.mean(time_jidt_cor) )
		print(" PythonDiscreteCMI(uncorrelated conditional): ", np.mean(time_python_cor) )
		print(" JidtDiscreteCMI (uncorrelated source): ", np.mean(time_jidt_uncor) )
		print(" PythonDiscreteCMI (uncorrelated source): ", np.mean(time_python_uncor) )
		
	print("\n=========================================================================")

	# test bin data
	print(f"\n\n\nTesting average CMI using 1D binary data with memory and discrete_method none\n")
	
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
	print(f"JidtDiscreteCMI: Estimated MI: {mi_jidt} - took: {itoc - itic}")
	est = PythonDiscreteCMI(settings)
	itic = time.perf_counter()
	mi_python = est.estimate(varx, vary, varz)
	itoc = time.perf_counter()
	print(f"PythonDiscreteCMI: Estimated MI: {mi_python} - took: {itoc - itic}")
	
	verbose(mi_jidt, mi_python, "", "CMI")
	
	print("\n=========================================================================")

	# test 2D
	print(f"\n\nTesting average CMI using 2D mute data - uncorrelated conditional and uncorrelated source")
	print(f"testing settings n_discrete_bins {vals} and discrete_method max_ent and equal")
	
	data = _generate_mute_data(n_replications=2)
	source1 = data[0,:,:]
	target = data[2,:,:]
	source2 = data[4,:,:]
	
	for m in ['max_ent','equal']:
		print(f"\n--- discrete_method: {m}\n")

		mi_jidt_cor = np.zeros(len(vals))
		mi_python_cor = np.zeros(len(vals))
		time_jidt_cor = np.zeros(len(vals))
		time_python_cor = np.zeros(len(vals))
		
		mi_jidt_uncor = np.zeros(len(vals))
		mi_python_uncor = np.zeros(len(vals))
		time_jidt_uncor = np.zeros(len(vals))
		time_python_uncor = np.zeros(len(vals))

		count = 0
		for i in vals:
			settings = {'discretise_method': m,
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
			
		print(f"Summary Jidt vs Python DiscreteCMI discretised 1D gaussian data using {m}:")
		
		print("nbins\tJidtDiscreteCMI\t\tPythonDiscreteCMI")
		print("CMI values uncorrelated conditional:")
		for i in range(len(vals)):
			print(f"{vals[i]}\t{mi_jidt_cor[i]}\t{mi_python_cor[i]}")
		
		print("\nMI values uncorrelated source:")
		#print("nbins\tJidtDiscreteCMI\t\tPythonDiscreteCMI")
		for i in range(len(vals)):
			print(f"{vals[i]}\t{mi_jidt_uncor[i]}\t{mi_python_uncor[i]}")

		verbose(mi_jidt_cor, mi_python_cor, "", "CMI (uncorrelated conditional)", local=False)
		verbose(mi_jidt_uncor, mi_python_uncor, "", "CMI (uncorrelated source)", local=False)

		print("\nmean calculation times:")
		print(" JidtDiscreteCMI(uncorrelated conditional): ", np.mean(time_jidt_cor) )
		print(" PythonDiscreteCMI(uncorrelated conditional): ", np.mean(time_python_cor) )
		print(" JidtDiscreteCMI (uncorrelated source): ", np.mean(time_jidt_uncor) )
		print(" PythonDiscreteCMI (uncorrelated source): ", np.mean(time_python_uncor) )

	print("\n=========================================================================")

	# test mixed dimension input
	d = [1, 2, 3]

	print(f"\n\nTesting average MI using mixed dimensions\ntesting dimensions {d} for var1, var2 and cond each\n")
	print("Shapes:")
	
	data = _generate_mute_data()

	source_o = data[0,:,:]
	target_o = data[2,:,:]
	cond_o = data[4,:,:]
	
	settings = {'discretise_method': 'max_ent',
				'n_discrete_bins': 2}
	
	for s in d:
		for t in d:
			for c in d:
			
				source1 = source_o[:,:s]
				target = target_o[:,:t]
				conditional = cond_o[:,:c]
				
				cond = f"var1: {source1.shape} var2: {target.shape} cond: {conditional.shape}"

				jidt_estimator = JidtDiscreteCMI(settings)
				python_estimator = PythonDiscreteCMI(settings)
			
				itic = time.perf_counter()
				mi_jidt_cor = jidt_estimator.estimate(source1, target, conditional)
				itoc = time.perf_counter()
				time_jidt_cor = itoc - itic
				
				itic = time.perf_counter()
				mi_python_cor = python_estimator.estimate(source1, target, conditional)
				itoc = time.perf_counter()
				time_python_cor = itoc - itic

				
				verbose(mi_jidt_cor, mi_python_cor, cond, "MI")

def test_discrete_cmi_local_values():

	vals = [2,4,10]

	print(f"\n\nTesting local CMI using 1D gaussian data with covariance 0.4 - uncorrelated \nconditional and uncorrelated source")
	print(f"testing settings n_discrete_bins {vals} and discrete_method max_ent and equal\n")
	
	expected_mi, source1, source2, target = _get_gauss_data(expand=True, seed=SEED)
	
	time_jidt_cor = 0.0
	time_jidt_uncor = 0.0
	time_python_cor = 0.0
	time_python_uncor = 0.0
	 
	print("bins")
	for i in vals:
		settings = {}
		settings = {'local_values': True,
					'discretise_method': 'max_ent',
					'n_discrete_bins': 2}
			
		jidt_estimator = JidtDiscreteCMI(settings)
		itic = time.perf_counter()
		mi_jidt = jidt_estimator.estimate(source1, target, source2)
		itoc = time.perf_counter()
		time_jidt_cor += itoc - itic
		
		python_estimator = PythonDiscreteCMI(settings)
		itic = time.perf_counter()
		mi_python = python_estimator.estimate(source1, target, source2)
		itoc = time.perf_counter()
		time_python_cor += itoc - itic
		
		verbose(mi_jidt, mi_python, i, "CMI (uncorrelated conditional)", local=True, atol=1e-03)

		jidt_estimator = JidtDiscreteCMI(settings)
		itic = time.perf_counter()
		mi_jidt = jidt_estimator.estimate(source2, target, source1)
		itoc = time.perf_counter()
		time_jidt_uncor += itoc - itic
			
		python_estimator = PythonDiscreteCMI(settings)
		itic = time.perf_counter()
		mi_python = python_estimator.estimate(source2, target, source1)
		itoc = time.perf_counter()
		time_python_uncor += itoc - itic
		
		verbose(mi_jidt, mi_python, i, "CMI (uncorrelated source)     ", local=True, atol=1e-03)
		
	print("\nmean calculation times:")
	print(" JidtDiscreteCMI: ", np.mean(time_jidt_cor) )
	print(" PythonDiscreteCMI: ", np.mean(time_python_cor) )

	print("\n=========================================================================")

	# test 2D data
	print(f"\n\nTesting local CMI using 2D mute data  - uncorrelated \nconditional and uncorrelated source")
	print(f"testing settings n_discrete_bins {vals} and discrete_method max_ent and equal\n")
	
	print("\nTest n_discrete_bins using 2D data input:")
	data = _generate_mute_data(n_replications=2)
	source1 = data[0,:,:]
	target = data[1,:,:]
	source2 = data[4,:,:]

	time_jidt_cor = 0.0
	time_jidt_uncor = 0.0
	time_python_cor = 0.0
	time_python_uncor = 0.0
	
	print("bins")
	for i in vals:
		settings = {}
		settings = {'local_values': True,
					'discretise_method': 'max_ent',
					'n_discrete_bins': 2,}
			
		jidt_estimator = JidtDiscreteCMI(settings)
		itic = time.perf_counter()
		mi_jidt = jidt_estimator.estimate(source1, target, source2)
		itoc = time.perf_counter()
		time_jidt_cor += itoc - itic
		
		python_estimator = PythonDiscreteCMI(settings)
		itic = time.perf_counter()
		mi_python = python_estimator.estimate(source1, target, source2)
		itoc = time.perf_counter()
		time_python_cor += itoc - itic
		
		verbose(mi_jidt, mi_python, i, "CMI (uncorrelated conditional)", local=True)

		jidt_estimator = JidtDiscreteCMI(settings)
		itic = time.perf_counter()
		mi_jidt = jidt_estimator.estimate(source2, target, source1)
		itoc = time.perf_counter()
		time_jidt_uncor += itoc - itic
			
		python_estimator = PythonDiscreteCMI(settings)
		itic = time.perf_counter()
		mi_python = python_estimator.estimate(source2, target, source1)
		itoc = time.perf_counter()
		time_python_uncor += itoc - itic
		
		verbose(mi_jidt, mi_python, i, "CMI (uncorrelated source)     ", local=True)
		
	print("\nmean calculation times:")
	print(" JidtDiscreteCMI: ", np.mean(time_jidt_cor) )
	print(" PythonDiscreteCMI: ", np.mean(time_python_cor) )

def test_discrete_ais():
	
	atol = 1e-03
	hvals = [1,2,3]
	nvals = [2,4,8]
	
	print(f"\n\nTesting average AIS using 1D AR with history and noise")
	print(f"testing settings history {hvals} and n_discrete_bins {nvals} and discrete_method max_ent\n")
	
	source1, source2 = _get_ar_data(seed=SEED)

	time_jidt_cor = np.zeros(np.power(len(nvals),2))
	res_jidt_cor = np.zeros(np.power(len(nvals),2))
	time_python_cor = np.zeros(np.power(len(nvals),2))
	res_python_cor = np.zeros(np.power(len(nvals),2))
	time_jidt_uncor = np.zeros(np.power(len(nvals),2))
	res_jidt_uncor = np.zeros(np.power(len(nvals),2))
	time_python_uncor = np.zeros(np.power(len(nvals),2))
	res_python_uncor = np.zeros(np.power(len(nvals),2))
	conds = np.empty((np.power(len(nvals),3),2))

	count = 0
	for h in hvals:
		for i in nvals:
		
			conds[count,:] = [h, i]

			settings_j = {'history': h,
						'discretise_method': 'max_ent',
						'n_discrete_bins': i}

			settings_p = {'history': h, 
						'discretise_method': 'max_ent',
						'n_discrete_bins': i}
	
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
	
	print(f"Summary Jidt vs Python DiscreteAIS discretised 1D gaussian data using max_ent:")

	print(f"hist, bins\tJidtDiscreteAIS\t\tPythonDiscreteAIS\tclose {atol}")
	print("with history")
	count = 0
	for i in range(len(res_jidt_cor)):
		print(f"{conds[i,:]}\t\t{res_jidt_cor[i]}\t{res_python_cor[i]}\t{np.isclose(res_jidt_cor[i], res_python_cor[i] ,atol=atol)}")
		count += 1
	
	print("noise")
	count = 0
	for i in range(len(res_jidt_uncor)):
		print(f"{conds[i,:]}\t\t{res_jidt_uncor[i]}\t{res_python_uncor[i]}\t{np.isclose(res_jidt_uncor[i], res_python_uncor[i] ,atol=atol)}")
		count += 1

	verbose(res_jidt_cor, res_python_cor, "with history", "AIS", atol=atol)
	verbose(res_jidt_uncor, res_python_uncor, "noise", "AIS", atol=atol)

	print("\nmean calculation times:")
	print(" JidtDiscreteAIS (with history): ", np.mean(time_jidt_cor) )
	print(" PythonDiscreteAIS (with history): ", np.mean(time_python_cor) )
	print(" JidtDiscreteAIS (noise): ", np.mean(time_jidt_uncor) )
	print(" PythonDiscreteAIS (noise): ", np.mean(time_python_uncor) )

def test_discrete_ais_local_values():
	
	atol = 1e-03

	hvals = [1,2,3]
	nvals = [2,4,6]
	
	print(f"\n\nTesting local AIS using 1D AR with history and noise")
	print(f"testing settings history {hvals} and n_discrete_bins {nvals} and discrete_method max_ent\n")
	
	source1, source2 = _get_ar_data(seed=SEED+1)
	
	min_len = min(len(source1),len(source2))
	source1 = source1[:min_len]
	source2 = source2[:min_len]

	time_jidt_cor = np.zeros(np.power(len(nvals),2))
	res_jidt_cor = np.zeros(np.power(len(nvals),2))
	time_python_cor = np.zeros(np.power(len(nvals),2))
	res_python_cor = np.zeros(np.power(len(nvals),2))
	time_jidt_uncor = np.zeros(np.power(len(nvals),2))
	res_jidt_uncor = np.zeros(np.power(len(nvals),2))
	time_python_uncor = np.zeros(np.power(len(nvals),2))
	res_python_uncor = np.zeros(np.power(len(nvals),2))
	conds = np.empty((np.power(len(nvals),3),2))

	print("hist, bins\tJidtDiscreteAIS vs PythonDiscreteAIS")
	count = 0
	for h in hvals:
		for i in nvals:
		
			conds[count,:] = [h, i]
			settings = {}
			settings_j = {'history': h,
						'discretise_method': 'max_ent',
						'n_discrete_bins': i,
						'local_values': True}
			settings_p = {'history': h, 
						'discretise_method': 'max_ent',
						'n_discrete_bins': i,
						'local_values': True}

			jidt_estimator = JidtDiscreteAIS(settings=settings_j)
			python_estimator = PythonDiscreteAIS(settings=settings_p)
	
			itic = time.perf_counter()
			res_jidt_cor = jidt_estimator.estimate(source1)
			itoc = time.perf_counter()
			time_jidt_cor[count] = itoc - itic
	
			itic = time.perf_counter()
			res_jidt_uncor = jidt_estimator.estimate(source2)
			itoc = time.perf_counter()
			time_jidt_uncor[count] = itoc - itic
			
			
			itic = time.perf_counter()
			res_python_cor = python_estimator.estimate(source1)
			itoc = time.perf_counter()
			time_python_cor[count] = itoc - itic
			
			itic = time.perf_counter()
			res_python_uncor = python_estimator.estimate(source2)
			itoc = time.perf_counter()
			time_python_uncor[count] = itoc - itic
			
			#print(res_jidt_cor[:20])
			#print(res_python_cor[:20])

			min_len=min(len(res_jidt_cor), len(res_python_cor))
			
			verbose(res_jidt_cor, res_python_cor, f"{conds[count,:]} - with hist", "AIS", local=True, atol=atol)
			verbose(res_jidt_uncor, res_python_uncor, f"{conds[count,:]} - noise    ", "AIS", local=True, atol=atol)

			count += 1
		
	print("\nmean calculation times:")
	print(" JidtKraskovAIS (with history): ", np.mean(time_jidt_cor) )
	print(" PythonKraskovAIS (with history): ", np.mean(time_python_cor) )
	print(" JidtKraskovAIS (noise): ", np.mean(time_jidt_uncor) )
	print(" PythonKraskovAIS (noise): ", np.mean(time_python_uncor) )

def test_discrete_te():

	vals = [1,3]
	nvals = [2,6]

	print(f"\n\nTesting average TE using 1D gaussian data with covariance 0.4 and lag 1\n")
	print(f"testing settings history_source (hs), tau_source (ts), history_target (ht), \ntau_target (tt), source_target_delay (std) with {vals} each.\nand n_discrete_bins{nvals}")

	expected_mi, source1, source2, target = _get_gauss_data(expand=False, seed=SEED)
	# add delay of one sample
	source1 = source1[1:]
	source2 = source2[1:]
	target = target[:-1]
	
	time_jidt_cor = np.empty(np.power(len(vals),6))
	res_jidt_cor = np.empty(np.power(len(vals),6))
	time_python_cor = np.empty(np.power(len(vals),6))
	res_python_cor = np.empty(np.power(len(vals),6))
	time_jidt_uncor = np.empty(np.power(len(vals),6))
	res_jidt_uncor = np.empty(np.power(len(vals),6))
	time_python_uncor = np.empty(np.power(len(vals),6))
	res_python_uncor = np.empty(np.power(len(vals),6))

	conds = np.empty([np.power(len(vals),6),6])
	
	print("hst,ht,tt,hs,ts\t\tJidtDiscreteTE\tPythonDiscreteTE\tclose 1e-03")
	
	count = 0
	for hst in vals:
		for ht in vals:
			for hs in vals:
				for tt in vals:
					for ts in vals:
						for n in nvals:
							conds[count,:] = [hst, ht, tt, hs, ts, n]
							settings_j = {"history_target": ht,
										"history_source": hs,
										"tau_target": tt,
										"tau_source": ts,
										"source_target_delay": hst,
										'discretise_method': 'equal',
										'n_discrete_bins': n}

							settings_p = {"history_target": ht,
										"history_source": hs,
										"tau_target": tt,
										"tau_source": ts,
										"source_target_delay": hst,
										'discretise_method': 'equal',
										'n_discrete_bins': n}

							jidt_estimator = JidtDiscreteTE(settings_j)
							python_estimator = PythonDiscreteTE(settings_p)

							itic = time.perf_counter()
							te_jidt_cor = jidt_estimator.estimate(source=source1, target=target)
							itoc = time.perf_counter()
							#time_jidt += itoc-itic
							time_jidt_cor[count] = itoc-itic
							res_jidt_cor[count] = te_jidt_cor

							
							itic = time.perf_counter()
							te_python_cor = python_estimator.estimate(source=source1, target=target)
							itoc = time.perf_counter()
							#time_python += itoc-itic
							time_python_cor[count] = itoc-itic
							res_python_cor[count] = te_python_cor


							#verbose(te_jidt, te_python, f"{[hst, ht, tt, hs, ts, n]}", "TE")
							print(f"{[hst, ht, tt, hs, ts]}\t\t{te_jidt_cor}\t{te_python_cor}\t{np.isclose(te_jidt_cor, te_python_cor, rtol=1e-03, atol=1e-03)}")

							count += 1

	verbose(res_jidt_cor, res_python_cor, "", "TE", local=False)

	print("\nmean calculation times:")
	print(" JidtDiscreteTE: ", np.mean(time_jidt_cor) )
	print(" PythonDiscreteTE: ", np.mean(time_python_cor) )

def test_discrete_te_local_values():

	vals = [2,4]

	print(f"\n\nTesting average TE using 1D binary data with memory\n")
	print(f"testing settings history_source (hs), tau_source (ts), history_target (ht), \ntau_target (tt), source_target_delay (std) with {vals} each,\nand n_discrete_bins 2\n")

	source1, target = _get_mem_binary_data(expand=True)
	
	time_jidt = np.empty(np.power(len(vals),5))
	res_jidt = np.empty(np.power(len(vals),5))
	time_python = np.empty(np.power(len(vals),5))
	res_python = np.empty(np.power(len(vals),5))
	conds = np.empty((np.power(len(vals),5),5))

	print("hst,ht,tt,hs,ts\t\tJidtDiscreteTE vs PythonDiscreteTE")

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
									"local_values": True,
									'noise_level': 0,
									'n_discrete_bins': 2}

						jidt_estimator = JidtDiscreteTE(settings_j)
						
						itic = time.perf_counter()
						te_jidt = jidt_estimator.estimate(source=source1, target=target)
						itoc = time.perf_counter()

						time_jidt[count] = itoc-itic
						res_jidt[count] = np.mean(te_jidt)

						settings_p = {"history_target": ht,
									"history_source": hs,
									"tau_target": tt,
									"tau_source": ts,
									"source_target_delay": hst,
									"local_values": True,
									'noise_level': 0,
									'n_discrete_bins': 2}

						python_estimator = PythonDiscreteTE(settings_p)
						
						itic = time.perf_counter()
						te_python = python_estimator.estimate(source=source1, target=target)
						itoc = time.perf_counter()

						time_python[count] = itoc-itic
						res_python[count] = np.mean(te_python)

						count += 1
						
						verbose(te_jidt, te_python, f"{[hst, ht, tt, hs, ts]}\t", "local TE", local=True, atol=1e-03)

	print("\nmean calculation times:")
	print(" JidtDiscreteTE: ", np.mean(time_jidt) )
	print(" PythonDiscreteTE: ", np.mean(time_python) )


#### Test analytic distribution
def test_analytic_distribution_mi_gaussian():

    pvals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    m = 'equal'
    bins = 5

    print(f"\n\nTesting Gaussian MI on gaussian data with cov=0.4")
    print(f"testing computeEstimateForGivenPValue for:\n\t{pvals}")

    expected_mi, source, source_uncorr, target = _get_gauss_data(seed=SEED)
    source = source[1:]
    source_uncorr = source_uncorr[1:]
    target = target[:-1]

    EoP_jidt = np.zeros(len(pvals))
    EoP_python = np.zeros(len(pvals))
    
    settings_jidt = {'noise_level': 0, 
        'normalise': False,}

    settings_python = {'noise_level': 0, 
        'normalise': False,}

    est_jidt = JidtGaussianMI(settings_jidt)
    est_python = PythonGaussianMI(settings_python)

    mi = est_jidt.estimate(source, target)
    C_jidt = est_jidt.calc.computeSignificance()
	#C_jidt = est_jidt.get_analytic_distribution(source, target)

    mi2 = est_python.estimate(source, target)
    C_python = est_python.get_analytic_distribution(source, target)

    mean_jidt = C_jidt.getMeanOfDistribution()
    mean_python = C_python.getMeanOfDistribution()
    
    mean_uncorr_jidt = C_jidt.getMeanOfUncorrectedDistribution()
    mean_uncorr_python = C_python.getMeanOfUncorrectedDistribution()
    
    std_jidt = C_jidt.getStdOfDistribution()
    std_python = C_python.getStdOfDistribution()
    
    count = 0
    for p in pvals:
        EoP_jidt[count] = C_jidt.computeEstimateForGivenPValue(p)
        EoP_python[count] = C_python.computeEstimateForGivenPValue(p)
        count += 1

    print(f"Jidt computeSignificance object:\ntype: {type(C_jidt)}")
    print(f"Python computeSignificance object:\ntype: {type(C_python)}")
    
    atol = 1e-04
    print(f"\nSummary Jidt vs Python GaussianMI 1D gaussian data using {m}:\n")

    print(f"\t\t\tJidt\t\t\tPython\t\t\tclose {atol}")
    print(f"actualValue:\n\t\t\t{C_jidt.actualValue}\t{C_python.actualValue}\t{np.isclose(C_jidt.actualValue, C_python.actualValue, atol=atol)}")
    print(f"pValue:\n\t\t\t{C_jidt.pValue}\t{C_python.pValue}\t{np.isclose(C_jidt.pValue, C_python.pValue, atol=atol)}")
    print(f"getMeanOfDistribution:\n\t\t\t{mean_jidt}\t{mean_python}\t{np.isclose(mean_jidt, mean_python, atol=atol)}")
    print(f"getMeanOfUncorrectedDistribution:\n\t\t\t{mean_uncorr_jidt}\t{mean_uncorr_python}\t{np.isclose(mean_uncorr_jidt, mean_uncorr_python, atol=atol)}")
    print(f"StdOfDistribution:\n\t\t\t{std_jidt}\t{std_python}\t{np.isclose(std_jidt, std_python, atol=atol)}")

    print("\nEstimateForGivenPValue:")
    print("p\t\tJidtGaussianMI\t\tPythonGaussianMI")
    for i in range(len(pvals)):
        print(f"{pvals[i]}   \t{EoP_jidt[i]}\t{EoP_python[i]}")
    verbose(EoP_jidt, EoP_python, "", "Estimate for given PValue")

    print("\n=========================================================================")

def test_analytic_distribution_cmi_gaussian():

    pvals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    m = 'equal'
    bins = 5

    print(f"\n\nTesting Gaussian CMI on gaussian data with cov=0.4")
    print(f"testing computeEstimateForGivenPValue for:\n\t{pvals}")

    expected_mi, source, source_uncorr, target = _get_gauss_data(seed=SEED)
    source = source[1:]
    source_uncorr = source_uncorr[1:]
    target = target[:-1]

    EoP_jidt = np.zeros(len(pvals))
    EoP_python = np.zeros(len(pvals))
    
    settings_jidt = {'noise_level': 0, 
        'normalise': False,}

    settings_python = {'noise_level': 0, 
        'normalise': False,}

    est_jidt = JidtGaussianCMI(settings_jidt)
    est_python = PythonGaussianCMI(settings_python)

    mi = est_jidt.estimate(source, target, source_uncorr)
    C_jidt = est_jidt.get_analytic_distribution(source, target, source_uncorr)

    mi2 = est_python.estimate(source, target, source_uncorr)
    C_python = est_python.get_analytic_distribution(source, target, source_uncorr)

    mean_jidt = C_jidt.getMeanOfDistribution()
    mean_python = C_python.getMeanOfDistribution()
    
    mean_uncorr_jidt = C_jidt.getMeanOfUncorrectedDistribution()
    mean_uncorr_python = C_python.getMeanOfUncorrectedDistribution()
    
    std_jidt = C_jidt.getStdOfDistribution()
    std_python = C_python.getStdOfDistribution()
    
    count = 0
    for p in pvals:
        EoP_jidt[count] = C_jidt.computeEstimateForGivenPValue(p)
        EoP_python[count] = C_python.computeEstimateForGivenPValue(p)
        count += 1

    print(f"Jidt computeSignificance object:\ntype: {type(C_jidt)}")
    print(f"Python computeSignificance object:\ntype: {type(C_python)}")
    
    atol = 1e-04
    print(f"\nSummary Jidt vs Python GaussianCMI 1D gaussian data using {m}:\n")

    print(f"\t\t\tJidt\t\t\tPython\t\t\tclose {atol}")
    print(f"actualValue:\n\t\t\t{C_jidt.actualValue}\t{C_python.actualValue}\t{np.isclose(C_jidt.actualValue, C_python.actualValue, atol=atol)}")
    print(f"pValue:\n\t\t\t{C_jidt.pValue}\t{C_python.pValue}\t{np.isclose(C_jidt.pValue, C_python.pValue, atol=atol)}")
    print(f"getMeanOfDistribution:\n\t\t\t{mean_jidt}\t{mean_python}\t{np.isclose(mean_jidt, mean_python, atol=atol)}")
    print(f"getMeanOfUncorrectedDistribution:\n\t\t\t{mean_uncorr_jidt}\t{mean_uncorr_python}\t{np.isclose(mean_uncorr_jidt, mean_uncorr_python, atol=atol)}")
    print(f"StdOfDistribution:\n\t\t\t{std_jidt}\t{std_python}\t{np.isclose(std_jidt, std_python, atol=atol)}")

    print("\nEstimateForGivenPValue:")
    print("p\tJidtGaussianCMI\t\tPythonGaussianCMI")
    for i in range(len(pvals)):
        print(f"{pvals[i]}   \t{EoP_jidt[i]}\t{EoP_python[i]}")
    verbose(EoP_jidt, EoP_python, "", "Estimate for given PValue")

    print("\n=========================================================================")

def test_analytic_distribution_cmi_nocond_gaussian():

    pvals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    m = 'equal'
    bins = 5

    print(f"\n\nTesting Gaussian CMI (conditional=None) on gaussian data with cov=0.4")
    print(f"testing computeEstimateForGivenPValue for:\n\t{pvals}")

    expected_mi, source, source_uncorr, target = _get_gauss_data(seed=SEED)
    source = source[1:]
    source_uncorr = source_uncorr[1:]
    target = target[:-1]

    EoP_jidt = np.zeros(len(pvals))
    EoP_python = np.zeros(len(pvals))
    
    settings_jidt = {'noise_level': 0, 
        'normalise': False,}

    settings_python = {'noise_level': 0, 
        'normalise': False,}

    est_jidt = JidtGaussianCMI(settings_jidt)
    est_python = PythonGaussianCMI(settings_python)

    mi = est_jidt.estimate(source, target)
    C_jidt = est_jidt.get_analytic_distribution(source, target)

    mi2 = est_python.estimate(source, target)
    C_python = est_python.get_analytic_distribution(source, target)

    mean_jidt = C_jidt.getMeanOfDistribution()
    mean_python = C_python.getMeanOfDistribution()
    
    mean_uncorr_jidt = C_jidt.getMeanOfUncorrectedDistribution()
    mean_uncorr_python = C_python.getMeanOfUncorrectedDistribution()
    
    std_jidt = C_jidt.getStdOfDistribution()
    std_python = C_python.getStdOfDistribution()
    
    count = 0
    for p in pvals:
        EoP_jidt[count] = C_jidt.computeEstimateForGivenPValue(p)
        EoP_python[count] = C_python.computeEstimateForGivenPValue(p)
        count += 1

    print(f"Jidt computeSignificance object:\ntype: {type(C_jidt)}")
    print(f"Python computeSignificance object:\ntype: {type(C_python)}")
    
    atol = 1e-04
    print(f"\nSummary Jidt vs Python GaussianCMI (no conditional) 1D gaussian data using {m}:\n")

    print(f"\t\t\tJidt\t\t\tPython\t\t\tclose {atol}")
    print(f"actualValue:\n\t\t\t{C_jidt.actualValue}\t{C_python.actualValue}\t{np.isclose(C_jidt.actualValue, C_python.actualValue, atol=atol)}")
    print(f"pValue:\n\t\t\t{C_jidt.pValue}\t{C_python.pValue}\t{np.isclose(C_jidt.pValue, C_python.pValue, atol=atol)}")
    print(f"getMeanOfDistribution:\n\t\t\t{mean_jidt}\t{mean_python}\t{np.isclose(mean_jidt, mean_python, atol=atol)}")
    print(f"getMeanOfUncorrectedDistribution:\n\t\t\t{mean_uncorr_jidt}\t{mean_uncorr_python}\t{np.isclose(mean_uncorr_jidt, mean_uncorr_python, atol=atol)}")
    print(f"StdOfDistribution:\n\t\t\t{std_jidt}\t{std_python}\t{np.isclose(std_jidt, std_python, atol=atol)}")

    print("\nEstimateForGivenPValue:")
    print("p\tJidtGaussianCMI\t\tPythonGaussianCMI")
    for i in range(len(pvals)):
        print(f"{pvals[i]}   \t{EoP_jidt[i]}\t{EoP_python[i]}")
    verbose(EoP_jidt, EoP_python, "", "Estimate for given PValue")

    print("\n=========================================================================")

def test_analytic_distribution_ais_gaussian():

    pvals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    m = 'equal'
    bins = 5

    print(f"\n\nTesting Gaussian AIS using 1D AR with history \n using discretise_method {m} - {bins} bins\n")
    print(f"testing computeEstimateForGivenPValue for:\n\t{pvals}")

    source1, source2 = _get_ar_data(seed=SEED)

    EoP_jidt = np.zeros(len(pvals))
    EoP_python = np.zeros(len(pvals))
    
    settings_jidt = {'history': 2,
    	"discretise_method": m,
        "n_discrete_bins": bins, 
        'noise_level': 0, 
        'normalise': False,}

    settings_python = {'history': 2,
    	"discretise_method": m,
        "n_discrete_bins": bins, 
        'noise_level': 0, 
        'normalise': False,}

    est_jidt = JidtGaussianAIS(settings_jidt)
    est_python = PythonGaussianAIS(settings_python)

    mi = est_jidt.estimate(source1)
    C_jidt = est_jidt.calc.computeSignificance()
    #C_jidt = est_jidt.get_analytic_distribution() ######### ATTENTION get_analytic.. not working properly

    mi2 = est_python.estimate(source1)
    #C_python = est_python.computeSignificance()
    C_python = est_python.get_analytic_distribution(source1)

    mean_jidt = C_jidt.getMeanOfDistribution()
    mean_python = C_python.getMeanOfDistribution()
    
    mean_uncorr_jidt = C_jidt.getMeanOfUncorrectedDistribution()
    mean_uncorr_python = C_python.getMeanOfUncorrectedDistribution()
    
    std_jidt = C_jidt.getStdOfDistribution()
    std_python = C_python.getStdOfDistribution()
    
    count = 0
    for p in pvals:
        EoP_jidt[count] = C_jidt.computeEstimateForGivenPValue(p)
        EoP_python[count] = C_python.computeEstimateForGivenPValue(p)
        count += 1

    print(f"Jidt computeSignificance object:\ntype: {type(C_jidt)}")
    print(f"Python computeSignificance object:\ntype: {type(C_python)}")

    atol = 1e-04
    print(f"\nSummary Jidt vs Python GaussianAIS on AR data with history using {m}:\n")
    print(f"\t\t\tJidt\t\t\tPython\t\t\tclose {atol}")
    print(f"actualValue:\n\t\t\t{C_jidt.actualValue}\t{C_python.actualValue}\t{np.isclose(C_jidt.actualValue, C_python.actualValue, atol=atol)}")
    print(f"pValue:\n\t\t\t{C_jidt.pValue}\t{C_python.pValue}\t{np.isclose(C_jidt.pValue, C_python.pValue, atol=atol)}")
    print(f"getMeanOfDistribution:\n\t\t\t{mean_jidt}\t{mean_python}\t{np.isclose(mean_jidt, mean_python, atol=atol)}")
    print(f"getMeanOfUncorrectedDistribution:\n\t\t\t{mean_uncorr_jidt}\t{mean_uncorr_python}\t{np.isclose(mean_uncorr_jidt, mean_uncorr_python, atol=atol)}")
    print(f"StdOfDistribution:\n\t\t\t{std_jidt}\t{std_python}\t{np.isclose(std_jidt, std_python, atol=atol)}")

    print("\nEstimateForGivenPValue:")
    print("p\tJidtGaussianAIS\t\tPythonGaussianAIS")
    for i in range(len(pvals)):
        print(f"{pvals[i]}   \t{EoP_jidt[i]}\t{EoP_python[i]}")
    verbose(EoP_jidt, EoP_python, "", "Estimate for given PValue")

    print("\n=========================================================================")

def test_analytic_distribution_te_gaussian():

    pvals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    m = 'equal'
    bins = 5

    print(f"\n\nTesting Gaussian TE on gaussian data with cov=0.4")
    print(f"testing computeEstimateForGivenPValue for:\n\t{pvals}")

    expected_mi, source, source_uncorr, target = _get_gauss_data(seed=SEED)
    source = source[1:]
    source_uncorr = source_uncorr[1:]
    target = target[:-1]

    EoP_jidt = np.zeros(len(pvals))
    EoP_python = np.zeros(len(pvals))

    settings_jidt = {'history_target': 1,
    	'noise_level': 0, 
        'normalise': False,}

    settings_python = {'history_target': 1,
    	'noise_level': 0, 
        'normalise': False,}

    est_jidt = JidtGaussianTE(settings_jidt)
    est_python = PythonGaussianTE(settings_python)

    mi = est_jidt.estimate(source, target)
    #C_jidt = est_jidt.get_analytic_distribution(source, target)
    C_jidt = est_jidt.calc.computeSignificance()

    mi2 = est_python.estimate(source, target)
    C_python = est_python.get_analytic_distribution(source, target)

    mean_jidt = C_jidt.getMeanOfDistribution()
    mean_python = C_python.getMeanOfDistribution()
    
    mean_uncorr_jidt = C_jidt.getMeanOfUncorrectedDistribution()
    mean_uncorr_python = C_python.getMeanOfUncorrectedDistribution()
    
    std_jidt = C_jidt.getStdOfDistribution()
    std_python = C_python.getStdOfDistribution()
    
    count = 0
    for p in pvals:
        EoP_jidt[count] = C_jidt.computeEstimateForGivenPValue(p)
        EoP_python[count] = C_python.computeEstimateForGivenPValue(p)
        count += 1

    print(f"Jidt computeSignificance object:\ntype: {type(C_jidt)}")
    print(f"Python computeSignificance object:\ntype: {type(C_python)}")
    
    atol = 1e-04
    print(f"\nSummary Jidt vs Python GaussianTE 1D gaussian data using {m}:\n")

    print(f"\t\t\tJidt\t\t\tPython\t\t\tclose {atol}")
    print(f"actualValue:\n\t\t\t{C_jidt.actualValue}\t{C_python.actualValue}\t{np.isclose(C_jidt.actualValue, C_python.actualValue, atol=atol)}")
    print(f"pValue:\n\t\t\t{C_jidt.pValue}\t{C_python.pValue}\t{np.isclose(C_jidt.pValue, C_python.pValue, atol=atol)}")
    print(f"getMeanOfDistribution:\n\t\t\t{mean_jidt}\t{mean_python}\t{np.isclose(mean_jidt, mean_python, atol=atol)}")
    print(f"getMeanOfUncorrectedDistribution:\n\t\t\t{mean_uncorr_jidt}\t{mean_uncorr_python}\t{np.isclose(mean_uncorr_jidt, mean_uncorr_python, atol=atol)}")
    print(f"StdOfDistribution:\n\t\t\t{std_jidt}\t{std_python}\t{np.isclose(std_jidt, std_python, atol=atol)}")

    print("\nEstimateForGivenPValue:")
    print("p\tJidtGaussianTE\t\tPythonGaussianTE")
    for i in range(len(pvals)):
        print(f"{pvals[i]}   \t{EoP_jidt[i]}\t{EoP_python[i]}")
    verbose(EoP_jidt, EoP_python, "", "Estimate for given PValue")

    print("\n=========================================================================")

def test_analytic_distribution_cte_gaussian():

    pvals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    m = 'equal'
    bins = 5

    print(f"\n\nTesting Gaussian CTE on gaussian data with cov=0.4")
    print(f"testing computeEstimateForGivenPValue for:\n\t{pvals}")

    expected_mi, source, source_uncorr, target = _get_gauss_data(seed=SEED)
    source = source[1:]
    source_uncorr = source_uncorr[1:]
    target = target[:-1]

    EoP_jidt = np.zeros(len(pvals))
    EoP_python = np.zeros(len(pvals))
    
    settings_jidt = {'history_target': 1,
    	'noise_level': 0, 
        'normalise': False,}

    settings_python = {'history_target': 1,
    	'noise_level': 0, 
        'normalise': False,}

    est_jidt = JidtGaussianCTE(settings_jidt)
    est_python = PythonGaussianCTE(settings_python)

    mi = est_jidt.estimate(source, target, source_uncorr)
    C_jidt = est_jidt.get_analytic_distribution(source, target, source_uncorr)

    mi2 = est_python.estimate(source, target, source_uncorr)
    C_python = est_python.get_analytic_distribution(source, target, source_uncorr)

    mean_jidt = C_jidt.getMeanOfDistribution()
    mean_python = C_python.getMeanOfDistribution()
    
    mean_uncorr_jidt = C_jidt.getMeanOfUncorrectedDistribution()
    mean_uncorr_python = C_python.getMeanOfUncorrectedDistribution()
    
    std_jidt = C_jidt.getStdOfDistribution()
    std_python = C_python.getStdOfDistribution()
    
    count = 0
    for p in pvals:
        EoP_jidt[count] = C_jidt.computeEstimateForGivenPValue(p)
        EoP_python[count] = C_python.computeEstimateForGivenPValue(p)
        count += 1

    print(f"Jidt computeSignificance object:\ntype: {type(C_jidt)}")
    print(f"Python computeSignificance object:\ntype: {type(C_python)}")
    
    atol = 1e-04
    print(f"\nSummary Jidt vs Python GaussianCTE 1D gaussian data using {m}:\n")

    print(f"\t\t\tJidt\t\t\tPython\t\t\tclose {atol}")
    print(f"actualValue:\n\t\t\t{C_jidt.actualValue}\t{C_python.actualValue}\t{np.isclose(C_jidt.actualValue, C_python.actualValue, atol=atol)}")
    print(f"pValue:\n\t\t\t{C_jidt.pValue}\t{C_python.pValue}\t{np.isclose(C_jidt.pValue, C_python.pValue, atol=atol)}")
    print(f"getMeanOfDistribution:\n\t\t\t{mean_jidt}\t{mean_python}\t{np.isclose(mean_jidt, mean_python, atol=atol)}")
    print(f"getMeanOfUncorrectedDistribution:\n\t\t\t{mean_uncorr_jidt}\t{mean_uncorr_python}\t{np.isclose(mean_uncorr_jidt, mean_uncorr_python, atol=atol)}")
    print(f"StdOfDistribution:\n\t\t\t{std_jidt}\t{std_python}\t{np.isclose(std_jidt, std_python, atol=atol)}")

    print("\nEstimateForGivenPValue:")
    print("p\tJidtGaussianCTE\t\tPythonGaussianCTE")
    for i in range(len(pvals)):
        print(f"{pvals[i]}   \t{EoP_jidt[i]}\t{EoP_python[i]}")
    verbose(EoP_jidt, EoP_python, "", "Estimate for given PValue")

    print("\n=========================================================================")

def test_analytic_distribution_cte_nocond_gaussian():

    pvals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    m = 'equal'
    bins = 5

    print(f"\n\nTesting Gaussian CTE (conditional=None) on gaussian data with cov=0.4")
    print(f"testing computeEstimateForGivenPValue for:\n\t{pvals}")

    expected_mi, source, source_uncorr, target = _get_gauss_data(seed=SEED)
    source = source[1:]
    source_uncorr = source_uncorr[1:]
    target = target[:-1]

    EoP_jidt = np.zeros(len(pvals))
    EoP_python = np.zeros(len(pvals))
    
    settings_jidt = {'history_target': 1,
    	'noise_level': 0, 
        'normalise': False,}

    settings_python = {'history_target': 1,
    	'noise_level': 0, 
        'normalise': False,}

    est_jidt = JidtGaussianCTE(settings_jidt)
    est_python = PythonGaussianCTE(settings_python)

    mi = est_jidt.estimate(source, target)
    C_jidt = est_jidt.get_analytic_distribution(source, target)

    mi2 = est_python.estimate(source, target)
    C_python = est_python.get_analytic_distribution(source, target)

    mean_jidt = C_jidt.getMeanOfDistribution()
    mean_python = C_python.getMeanOfDistribution()
    
    mean_uncorr_jidt = C_jidt.getMeanOfUncorrectedDistribution()
    mean_uncorr_python = C_python.getMeanOfUncorrectedDistribution()
    
    std_jidt = C_jidt.getStdOfDistribution()
    std_python = C_python.getStdOfDistribution()
    
    count = 0
    for p in pvals:
        EoP_jidt[count] = C_jidt.computeEstimateForGivenPValue(p)
        EoP_python[count] = C_python.computeEstimateForGivenPValue(p)
        count += 1

    print(f"Jidt computeSignificance object:\ntype: {type(C_jidt)}")
    print(f"Python computeSignificance object:\ntype: {type(C_python)}")
    
    atol = 1e-04
    print(f"\nSummary Jidt vs Python GaussianCTE (no cond) 1D gaussian data using {m}:\n")

    print(f"\t\t\tJidt\t\t\tPython\t\t\tclose {atol}")
    print(f"actualValue:\n\t\t\t{C_jidt.actualValue}\t{C_python.actualValue}\t{np.isclose(C_jidt.actualValue, C_python.actualValue, atol=atol)}")
    print(f"pValue:\n\t\t\t{C_jidt.pValue}\t{C_python.pValue}\t{np.isclose(C_jidt.pValue, C_python.pValue, atol=atol)}")
    print(f"getMeanOfDistribution:\n\t\t\t{mean_jidt}\t{mean_python}\t{np.isclose(mean_jidt, mean_python, atol=atol)}")
    print(f"getMeanOfUncorrectedDistribution:\n\t\t\t{mean_uncorr_jidt}\t{mean_uncorr_python}\t{np.isclose(mean_uncorr_jidt, mean_uncorr_python, atol=atol)}")
    print(f"StdOfDistribution:\n\t\t\t{std_jidt}\t{std_python}\t{np.isclose(std_jidt, std_python, atol=atol)}")

    print("\nEstimateForGivenPValue:")
    print("p\tJidtGaussianCTE\t\tPythonGaussianCTE")
    for i in range(len(pvals)):
        print(f"{pvals[i]}   \t{EoP_jidt[i]}\t{EoP_python[i]}")
    verbose(EoP_jidt, EoP_python, "", "Estimate for given PValue")

    print("\n=========================================================================")

def test_analytic_distribution_mi_discrete():

    pvals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    m = 'equal'
    bins = 5

    print(f"\n\nTesting Discrete MI on discretized gaussian data with cov=0.4\n using discretise_method {m} - {bins} bins\n")
    print(f"testing computeEstimateForGivenPValue for:\n\t{pvals}")

    expected_mi, source, source_uncorr, target = _get_gauss_data(seed=SEED)
    source = source[1:]
    source_uncorr = source_uncorr[1:]
    target = target[:-1]

    EoP_jidt = np.zeros(len(pvals))
    EoP_python = np.zeros(len(pvals))
    
    settings_jidt = {"discretise_method": m,
        "n_discrete_bins": bins, 
        'noise_level': 0, 
        'normalise': False,}

    settings_python = {"discretise_method": m,
        "n_discrete_bins": bins, 
        'noise_level': 0, 
        'normalise': False,}

    est_jidt = JidtDiscreteMI(settings_jidt)
    est_python = PythonDiscreteMI(settings_python)

    mi, calc = est_jidt.estimate(source, target, True)
    C_jidt = calc.computeSignificance()

    mi2 = est_python.estimate(source, target)
    C_python = est_python.computeSignificance()
    
    mean_jidt = C_jidt.getMeanOfDistribution()
    mean_python = C_python.getMeanOfDistribution()
    
    mean_uncorr_jidt = C_jidt.getMeanOfUncorrectedDistribution()
    mean_uncorr_python = C_python.getMeanOfUncorrectedDistribution()
    
    std_jidt = C_jidt.getStdOfDistribution()
    std_python = C_python.getStdOfDistribution()
    
    count = 0
    for p in pvals:
        EoP_jidt[count] = C_jidt.computeEstimateForGivenPValue(p)
        EoP_python[count] = C_python.computeEstimateForGivenPValue(p)
        count += 1

    print(f"Jidt computeSignificance object:\ntype: {type(C_jidt)}")
    print(f"Python computeSignificance object:\ntype: {type(C_python)}")
    
    atol = 1e-06
    print(f"\nSummary Jidt vs Python DiscreteMI discretised 1D gaussian data using {m}:\n")

    print(f"\t\t\tJidt\t\t\tPython\t\t\tclose {atol}")
    print(f"actualValue:\n\t\t\t{C_jidt.actualValue}\t{C_python.actualValue}\t{np.isclose(C_jidt.actualValue, C_python.actualValue, atol=atol)}")
    print(f"pValue:\n\t\t\t{C_jidt.pValue}\t{C_python.pValue}\t{np.isclose(C_jidt.pValue, C_python.pValue, atol=atol)}")
    print(f"getMeanOfDistribution:\n\t\t\t{mean_jidt}\t{mean_python}\t{np.isclose(mean_jidt, mean_python, atol=atol)}")
    print(f"getMeanOfUncorrectedDistribution:\n\t\t\t{mean_uncorr_jidt}\t{mean_uncorr_python}\t{np.isclose(mean_uncorr_jidt, mean_uncorr_python, atol=atol)}")
    print(f"StdOfDistribution:\n\t\t\t{std_jidt}\t{std_python}\t{np.isclose(std_jidt, std_python, atol=atol)}")
    
    print("\nEstimateForGivenPValue:")
    print("p\tJidtDiscreteMI\t\tPythonDiscreteMI")
    for i in range(len(pvals)):
        print(f"{pvals[i]}   \t{EoP_jidt[i]}\t{EoP_python[i]}")
    verbose(EoP_jidt, EoP_python, "", "Estimate for given PValue")

    print("\n=========================================================================")

def test_analytic_distribution_cmi_discrete():

    pvals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    m = 'equal'
    bins = 5

    print(f"\n\nTesting Discrete CMI on discretized gaussian data with cov=0.4\n using discretise_method {m} - {bins} bins\n")
    print(f"testing computeEstimateForGivenPValue for:\n\t{pvals}")

    expected_mi, source, source_uncorr, target = _get_gauss_data(seed=SEED)
    source = source[1:]
    source_uncorr = source_uncorr[1:]
    target = target[:-1]

    EoP_jidt = np.zeros(len(pvals))
    EoP_python = np.zeros(len(pvals))
    
    settings_jidt = {"discretise_method": m,
        "n_discrete_bins": bins, 
        'noise_level': 0, 
        'normalise': False,}

    settings_python = {"discretise_method": m,
        "n_discrete_bins": bins, 
        'noise_level': 0, 
        'normalise': False,}

    est_jidt = JidtDiscreteCMI(settings_jidt)
    est_python = PythonDiscreteCMI(settings_python)

    mi, calc = est_jidt.estimate(source, target, source_uncorr, True)
    C_jidt = calc.computeSignificance()

    mi2 = est_python.estimate(source, target, source_uncorr)
    C_python = est_python.computeSignificance()
    
    mean_jidt = C_jidt.getMeanOfDistribution()
    mean_python = C_python.getMeanOfDistribution()
    
    mean_uncorr_jidt = C_jidt.getMeanOfUncorrectedDistribution()
    mean_uncorr_python = C_python.getMeanOfUncorrectedDistribution()
    
    std_jidt = C_jidt.getStdOfDistribution()
    std_python = C_python.getStdOfDistribution()
    
    count = 0
    for p in pvals:
        EoP_jidt[count] = C_jidt.computeEstimateForGivenPValue(p)
        EoP_python[count] = C_python.computeEstimateForGivenPValue(p)
        count += 1

    print(f"Jidt computeSignificance object:\ntype: {type(C_jidt)}")
    print(f"Python computeSignificance object:\ntype: {type(C_python)}")
    
    atol = 1e-06
    print(f"\nSummary Jidt vs Python DiscreteCMI discretised 1D gaussian data using {m}:\n")

    print(f"\t\t\tJidt\t\t\tPython\t\t\tclose {atol}")
    print(f"actualValue:\n\t\t\t{C_jidt.actualValue}\t{C_python.actualValue}\t{np.isclose(C_jidt.actualValue, C_python.actualValue, atol=atol)}")
    print(f"pValue:\n\t\t\t{C_jidt.pValue}\t{C_python.pValue}\t{np.isclose(C_jidt.pValue, C_python.pValue, atol=atol)}")
    print(f"getMeanOfDistribution:\n\t\t\t{mean_jidt}\t{mean_python}\t{np.isclose(mean_jidt, mean_python, atol=atol)}")
    print(f"getMeanOfUncorrectedDistribution:\n\t\t\t{mean_uncorr_jidt}\t{mean_uncorr_python}\t{np.isclose(mean_uncorr_jidt, mean_uncorr_python, atol=atol)}")
    print(f"StdOfDistribution:\n\t\t\t{std_jidt}\t{std_python}\t{np.isclose(std_jidt, std_python, atol=atol)}")
    
    print("\nEstimateForGivenPValue:")
    print("p\tJidtDiscreteCMI\t\tPythonDiscreteCMI")
    for i in range(len(pvals)):
        print(f"{pvals[i]}   \t{EoP_jidt[i]}\t{EoP_python[i]}")
    verbose(EoP_jidt, EoP_python, "", "Estimate for given PValue")

    print("\n=========================================================================")

def test_analytic_distribution_cmi_nocond_discrete():

    pvals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    m = 'equal'
    bins = 5

    print(f"\n\nTesting Discrete CMI on discretized gaussian data (conditional=None) with cov=0.4\n using discretise_method {m} - {bins} bins\n")
    print(f"testing computeEstimateForGivenPValue for:\n\t{pvals}")

    expected_mi, source, source_uncorr, target = _get_gauss_data(seed=SEED)
    source = source[1:]
    source_uncorr = source_uncorr[1:]
    target = target[:-1]

    EoP_jidt = np.zeros(len(pvals))
    EoP_python = np.zeros(len(pvals))
    
    settings_jidt = {"discretise_method": m,
        "n_discrete_bins": bins, 
        'noise_level': 0, 
        'normalise': False,}

    settings_python = {"discretise_method": m,
        "n_discrete_bins": bins, 
        'noise_level': 0, 
        'normalise': False,}

    est_jidt = JidtDiscreteCMI(settings_jidt)
    est_python = PythonDiscreteCMI(settings_python)

    mi = est_jidt.estimate(source, target)
    C_jidt = est_jidt.get_analytic_distribution(source, target)

    mi2 = est_python.estimate(source, target)
    C_python = est_python.get_analytic_distribution(source, target)
    
    mean_jidt = C_jidt.getMeanOfDistribution()
    mean_python = C_python.getMeanOfDistribution()
    
    mean_uncorr_jidt = C_jidt.getMeanOfUncorrectedDistribution()
    mean_uncorr_python = C_python.getMeanOfUncorrectedDistribution()
    
    std_jidt = C_jidt.getStdOfDistribution()
    std_python = C_python.getStdOfDistribution()
    
    count = 0
    for p in pvals:
        EoP_jidt[count] = C_jidt.computeEstimateForGivenPValue(p)
        EoP_python[count] = C_python.computeEstimateForGivenPValue(p)
        count += 1

    print(f"Jidt computeSignificance object:\ntype: {type(C_jidt)}")
    print(f"Python computeSignificance object:\ntype: {type(C_python)}")
    
    atol = 1e-06
    print(f"\nSummary Jidt vs Python DiscreteCMI (no cond) discretised 1D gaussian data using {m}:\n")

    print(f"\t\t\tJidt\t\t\tPython\t\t\tclose {atol}")
    print(f"actualValue:\n\t\t\t{C_jidt.actualValue}\t{C_python.actualValue}\t{np.isclose(C_jidt.actualValue, C_python.actualValue, atol=atol)}")
    print(f"pValue:\n\t\t\t{C_jidt.pValue}\t{C_python.pValue}\t{np.isclose(C_jidt.pValue, C_python.pValue, atol=atol)}")
    print(f"getMeanOfDistribution:\n\t\t\t{mean_jidt}\t{mean_python}\t{np.isclose(mean_jidt, mean_python, atol=atol)}")
    print(f"getMeanOfUncorrectedDistribution:\n\t\t\t{mean_uncorr_jidt}\t{mean_uncorr_python}\t{np.isclose(mean_uncorr_jidt, mean_uncorr_python, atol=atol)}")
    print(f"StdOfDistribution:\n\t\t\t{std_jidt}\t{std_python}\t{np.isclose(std_jidt, std_python, atol=atol)}")
    
    print("\nEstimateForGivenPValue:")
    print("p\tJidtDiscreteCMI\t\tPythonDiscreteCMI")
    for i in range(len(pvals)):
        print(f"{pvals[i]}   \t{EoP_jidt[i]}\t{EoP_python[i]}")
    verbose(EoP_jidt, EoP_python, "", "Estimate for given PValue")

    print("\n=========================================================================")

def test_analytic_distribution_ais_discrete():

    pvals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    m = 'equal'
    bins = 5

    print(f"\n\nTesting Discrete AIS using 1D AR with history \n using discretise_method {m} - {bins} bins\n")
    print(f"testing computeEstimateForGivenPValue for:\n\t{pvals}")

    source1, source2 = _get_ar_data(seed=SEED)

    EoP_jidt = np.zeros(len(pvals))
    EoP_python = np.zeros(len(pvals))
    
    settings_jidt = {'history': 2,
    	"discretise_method": m,
        "n_discrete_bins": bins, 
        'noise_level': 0, 
        'normalise': False,}

    settings_python = {'history': 2,
    	"discretise_method": m,
        "n_discrete_bins": bins, 
        'noise_level': 0, 
        'normalise': False,}

    est_jidt = JidtDiscreteAIS(settings_jidt)
    est_python = PythonDiscreteAIS(settings_python)

    mi, calc = est_jidt.estimate(source1, True)
    C_jidt = calc.computeSignificance()

    mi2 = est_python.estimate(source1)
    C_python = est_python.computeSignificance()
    
    mean_jidt = C_jidt.getMeanOfDistribution()
    mean_python = C_python.getMeanOfDistribution()
    
    mean_uncorr_jidt = C_jidt.getMeanOfUncorrectedDistribution()
    mean_uncorr_python = C_python.getMeanOfUncorrectedDistribution()
    
    std_jidt = C_jidt.getStdOfDistribution()
    std_python = C_python.getStdOfDistribution()
    
    count = 0
    for p in pvals:
        EoP_jidt[count] = C_jidt.computeEstimateForGivenPValue(p)
        EoP_python[count] = C_python.computeEstimateForGivenPValue(p)
        count += 1

    print(f"Jidt computeSignificance object:\ntype: {type(C_jidt)}")
    print(f"Python computeSignificance object:\ntype: {type(C_python)}")

    atol = 1e-06
    print(f"\nSummary Jidt vs Python DiscreteAIS on AR data with history using {m}:\n")
    print(f"\t\t\tJidt\t\t\tPython\t\t\tclose {atol}")
    print(f"actualValue:\n\t\t\t{C_jidt.actualValue}\t{C_python.actualValue}\t{np.isclose(C_jidt.actualValue, C_python.actualValue, atol=atol)}")
    print(f"pValue:\n\t\t\t{C_jidt.pValue}\t{C_python.pValue}\t{np.isclose(C_jidt.pValue, C_python.pValue, atol=atol)}")
    print(f"getMeanOfDistribution:\n\t\t\t{mean_jidt}\t{mean_python}\t{np.isclose(mean_jidt, mean_python, atol=atol)}")
    print(f"getMeanOfUncorrectedDistribution:\n\t\t\t{mean_uncorr_jidt}\t{mean_uncorr_python}\t{np.isclose(mean_uncorr_jidt, mean_uncorr_python, atol=atol)}")
    print(f"StdOfDistribution:\n\t\t\t{std_jidt}\t{std_python}\t{np.isclose(std_jidt, std_python, atol=atol)}")

    print("\nEstimateForGivenPValue:")
    print("p\tJidtDiscreteAIS\t\tPythonDiscreteAIS")
    for i in range(len(pvals)):
        print(f"{pvals[i]}   \t{EoP_jidt[i]}\t{EoP_python[i]}")
    verbose(EoP_jidt, EoP_python, "", "Estimate for given PValue")

    print("\n=========================================================================")

def test_analytic_distribution_te_discrete():

    pvals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    m = 'equal'
    bins = 5

    print(f"\n\nTesting Discrete TE on discretized gaussian data with cov=0.4\n using discretise_method {m} - {bins} bins\n")
    print(f"testing computeEstimateForGivenPValue for:\n\t{pvals}")

    expected_mi, source, source_uncorr, target = _get_gauss_data(seed=SEED)
    source = source[1:]
    source_uncorr = source_uncorr[1:]
    target = target[:-1]

    EoP_jidt = np.zeros(len(pvals))
    EoP_python = np.zeros(len(pvals))
    
    settings_jidt = {'history_target': 1,
    	"discretise_method": m,
        "n_discrete_bins": bins, 
        'noise_level': 0, 
        'normalise': False,}

    settings_python = {'history_target': 1,
    	"discretise_method": m,
        "n_discrete_bins": bins, 
        'noise_level': 0, 
        'normalise': False,}

    est_jidt = JidtDiscreteTE(settings_jidt)
    est_python = PythonDiscreteTE(settings_python)

    mi, calc = est_jidt.estimate(source, target, True)
    C_jidt = calc.computeSignificance()

    mi2 = est_python.estimate(source, target)
    C_python = est_python.computeSignificance()
    
    mean_jidt = C_jidt.getMeanOfDistribution()
    mean_python = C_python.getMeanOfDistribution()
    
    mean_uncorr_jidt = C_jidt.getMeanOfUncorrectedDistribution()
    mean_uncorr_python = C_python.getMeanOfUncorrectedDistribution()
    
    std_jidt = C_jidt.getStdOfDistribution()
    std_python = C_python.getStdOfDistribution()
    
    count = 0
    for p in pvals:
        EoP_jidt[count] = C_jidt.computeEstimateForGivenPValue(p)
        EoP_python[count] = C_python.computeEstimateForGivenPValue(p)
        count += 1

    print(f"Jidt computeSignificance object:\ntype: {type(C_jidt)}")
    print(f"Python computeSignificance object:\ntype: {type(C_python)}")
    
    atol = 1e-06
    print(f"\nSummary Jidt vs Python DiscreteTE discretised 1D gaussian data using {m}:\n")

    print(f"\t\t\tJidt\t\t\tPython\t\t\tclose {atol}")
    print(f"actualValue:\n\t\t\t{C_jidt.actualValue}\t{C_python.actualValue}\t{np.isclose(C_jidt.actualValue, C_python.actualValue, atol=atol)}")
    print(f"pValue:\n\t\t\t{C_jidt.pValue}\t{C_python.pValue}\t{np.isclose(C_jidt.pValue, C_python.pValue, atol=atol)}")
    print(f"getMeanOfDistribution:\n\t\t\t{mean_jidt}\t{mean_python}\t{np.isclose(mean_jidt, mean_python, atol=atol)}")
    print(f"getMeanOfUncorrectedDistribution:\n\t\t\t{mean_uncorr_jidt}\t{mean_uncorr_python}\t{np.isclose(mean_uncorr_jidt, mean_uncorr_python, atol=atol)}")
    print(f"StdOfDistribution:\n\t\t\t{std_jidt}\t{std_python}\t{np.isclose(std_jidt, std_python, atol=atol)}")
    
    print("\nEstimateForGivenPValue:")
    print("p\tJidtDiscreteTE\t\tPythonDiscreteTE")
    for i in range(len(pvals)):
        print(f"{pvals[i]}   \t{EoP_jidt[i]}\t{EoP_python[i]}")
    verbose(EoP_jidt, EoP_python, "", "Estimate for given PValue")

    print("\n=========================================================================")


#### Test bi- and multivariate analysis (single target)
def test_single_target_analysis(analysis, est_type, numperm=500, samples=10000):
    """Test multivariate TE estimation from correlated Gaussians."""
    
    measure = analysis[-2:].lower()
    jidt_estimator = f"Jidt{est_type}CMI"
    python_estimator = f"Python{est_type}CMI"

    print(f"\n\nTesting average {analysis} (nperms: {numperm}) using discretized 1D gaussian data")
    print(f"with covariance 0.4, lag 1 and {samples} samples\n")

    # Generate data and add a delay one one sample.
    expected_mi, source, source_uncorr, target = _get_gauss_data(n=samples, seed=SEED)
    source = source[1:]
    source_uncorr = source_uncorr[1:]
    target = target[:-1]
    if est_type == "Discrete":
    	est = PythonDiscreteCMI({"discretise_method": "equal", "n_discrete_bins": 2})
    	source, target, source_uncorr = est._discretise_vars(var1=source, var2=target, conditional=source_uncorr)

    data = Data(np.hstack((source, source_uncorr, target)),
       	        dim_order='sp', normalise=False)
    data2 = copy.deepcopy(data)

    settings_jidt = {
        'cmi_estimator': jidt_estimator,
        'n_perm_max_stat': numperm,
        'n_perm_min_stat': numperm,
        'n_perm_max_seq': numperm,
        'n_perm_omnibus': numperm,
        'max_lag_sources': 3,
        'min_lag_sources': 1,
        'noise_level': 0, 
        'normalise': False,
        "discretise_method": "equal",
        "n_discrete_bins": 5, 
        }

    settings_python = {
        'cmi_estimator': python_estimator,
        'n_perm_max_stat': numperm,
        'n_perm_min_stat': numperm,
        'n_perm_max_seq': numperm,
        'n_perm_omnibus': numperm,
        'max_lag_sources': 3,
        'min_lag_sources': 1,
        'noise_level': 0, 
        'normalise': False,
        "discretise_method": "equal",
        "n_discrete_bins": 5, 
        }
    
    nw = eval(f"{analysis}()")
    
    print("\n#### Analyse single target JIDT\n")

    itic = time.perf_counter()
    results_jidt = nw.analyse_single_target(
        settings_jidt, data, target=2, sources=[0, 1])
    mi_jidt = results_jidt.get_single_target(2, fdr=False)[measure][0]
    sources_jidt = results_jidt.get_target_sources(2, fdr=False)
    itoc = time.perf_counter()
    time_jidt = itoc-itic

    # Assert that only the correlated source was detected.
    assert len(sources_jidt) == 1, 'Wrong no. inferred sources: {0}.'.format(
        len(sources_jidt))
    assert sources_jidt[0] == 0, 'Wrong inferred source: {0}.'.format(sources_jidt[0])


    print("\n#### Analyse single target Python\n")

    itic = time.perf_counter()
    results_python = nw.analyse_single_target(
        settings_python, data2, target=2, sources=[0, 1])
    mi_python = results_python.get_single_target(2, fdr=False)[measure][0]
    sources_python = results_python.get_target_sources(2, fdr=False)
    itoc = time.perf_counter()
    time_python = itoc-itic

    assert len(sources_python) == 1, 'Wrong no. inferred sources: {0}.'.format(
        len(sources_python))
    assert sources_python[0] == 0, 'Wrong inferred source: {0}.'.format(sources_python[0])
    
    # Compare MultivariateTE() estimate to JIDT and Python estimate. Mimick realisations used
    # internally by the algorithm.
    settings = {'lag_mi': 0, 'normalise': False, 'noise_level': 0}
    est_jidt = eval(f"{jidt_estimator}(settings)")
    est_python = eval(f"{python_estimator}(settings)")
    
    jidt_mi = est_jidt.estimate(var1=source[1:-1], var2=target[2:])
    python_mi = est_python.estimate(var1=source[1:-1], var2=target[2:])
    
    print(f"Summary of comparing {analysis} using {jidt_estimator} vs {python_estimator}:\n")
    if sources_jidt==sources_python:
        print(f"Jidt {sources_jidt} and Python {sources_python} found identical target_sources. +++")
    else:
        print(f"Jidt {sources_jidt} and Python {sources_python} DID NOT find identical target_sources. !!!!!!!")
    verbose(mi_jidt, jidt_mi, f"Jidt {analysis} vs core", measure.upper(), atol=1e-03)
    verbose(mi_python, python_mi, f"Python {analysis} vs core", measure.upper(), atol=1e-03)
    verbose(mi_jidt, mi_python, f"Jidt {analysis} vs Python {analysis}", measure.upper(), atol=1e-03)
    verbose(jidt_mi, python_mi, "Jidt core vs Python core", measure.upper(), atol=1e-03)

    print("\n calculation times:")
    print(f"single target analysis {analysis} {jidt_estimator} nperms {numperm}: ", np.mean(time_jidt) )
    print(f"single target analysis {analysis} {python_estimator} nperms {numperm}: ", np.mean(time_python) )


#### Test network analysis
def test_network_analysis(analysis, est_type, numperm=300, samples=1000, reps=3):
	
	measure = analysis[-2:].lower()
	jidt_estimator = f"Jidt{est_type}CMI"
	python_estimator = f"Python{est_type}CMI"
	
	print(f"\n\nTesting network analysis via {analysis} (nperms: {numperm})")
	print(f"using mute data ({samples} samples, {reps} replications)\n")
	
	data = Data(normalise=False)  # initialise an empty data object
	data.generate_mute_data(n_samples=samples, n_replications=reps)
	if est_type == "Discrete":
		est = PythonDiscreteCMI({"discretise_method": "equal", "n_discrete_bins": 2})
		d = data.data
		for i in range(5):
			d[i,:,:] = est._discretise_vars(var1=d[i,:,:])
		d = d.astype(int)
		data.set_data(d, "psr")

	data2 = copy.deepcopy(data)

	network_analysis = eval(f"{analysis}()")
	#print(network_analysis)
	
	print("\n#### Analyse network Jidt\n")

	settings = {
	    "cmi_estimator": jidt_estimator,
	    "n_perm_max_stat": numperm,
	    "n_perm_min_stat": numperm,
	    "n_perm_omnibus": numperm,
	    "n_perm_max_seq": numperm,
	    "max_lag_sources": 5,
	    "min_lag_sources": 1,
	}

	itic = time.perf_counter()
	results_jidt = network_analysis.analyse_network(settings, data)
	itoc = time.perf_counter()
	time_jidt = itoc - itic
	
	print("\n#### Analyse network Python\n")

	settings2 = {
	    "cmi_estimator": python_estimator,
	    "n_perm_max_stat": numperm,
	    "n_perm_min_stat": numperm,
	    "n_perm_omnibus": numperm,
	    "n_perm_max_seq": numperm,
	    "max_lag_sources": 5,
	    "min_lag_sources": 1,
	}

	itic = time.perf_counter()
	results_python = network_analysis.analyse_network(settings2, data2)
	itoc = time.perf_counter()
	time_python = itoc - itic

	# get results
	target_delays_jidt = [None]*5
	selected_sources_jidt = [None]*5
	selected_sources_te_jidt = [None]*5
	
	target_delays_python = [None]*5
	selected_sources_python = [None]*5
	selected_sources_te_python = [None]*5

	for t in range(5):
		target_delays_jidt[t] = results_jidt.get_target_delays(t, fdr=False)
		target_delays_python[t] = results_python.get_target_delays(t, fdr=False)

		target_jidt = results_jidt.get_single_target(t, fdr=False)
		selected_sources_jidt[t] = target_jidt['selected_vars_sources']
		selected_sources_te_jidt[t] = target_jidt[f'selected_sources_{measure}']
		
		target_python = results_python.get_single_target(t, fdr=False)
		selected_sources_python[t] = target_python['selected_vars_sources']
		selected_sources_te_python[t] = target_python[f'selected_sources_{measure}']

	
	print(f"\nSummary network analysis {analysis} - {jidt_estimator} vs {python_estimator}\n")
	
	print("\nselected sources:\n")
	print("target\tequal")
	for t in range(5):
		print(f"{t}\t\t{selected_sources_jidt[t]==selected_sources_python[t]}\t{jidt_estimator}  : {selected_sources_jidt[t]}\n\t\t\t\t{python_estimator}: {selected_sources_python[t]}")

	atol = 1e-03
	print("\ntarget delays:\n")
	print("target\t\t\t\t\t\tequal")
	for t in range(5):
		if len(target_delays_jidt[t])==len(target_delays_python[t]):
			equal = np.allclose(target_delays_jidt[t], target_delays_python[t], atol=atol)
		else:
			equal = False
		
		print(f"{t}\t{jidt_estimator}  :\t{target_delays_jidt[t]}{"\t" if len(target_delays_jidt[t])>1 else "\t\t"}{equal}\n\t{python_estimator}:\t{target_delays_python[t]}")
	
	print(f"\nselected sources {measure.upper()}:\n")
	print(f"target\tclose {atol}")

	for t in range(5):
		try:
			if len(selected_sources_te_jidt[t])==len(selected_sources_te_python[t]):
				equal = np.allclose(selected_sources_te_jidt[t], selected_sources_te_python[t], atol=atol)	
			else: 
				equal = False
		except:
			equal = False
		print(f"{t}\t\t{equal}\t{jidt_estimator}  : {selected_sources_te_jidt[t]}\n\t\t\t\t{python_estimator}: {selected_sources_te_python[t]}")
	
	print("\nEdge lists:")
	print("Jidt:")
	results_jidt.print_edge_list("max_te_lag", fdr=False)
	print("Python:")
	results_python.print_edge_list("max_te_lag", fdr=False)

	print("\n calculation times:")
	print(f" network_analysis {analysis} {jidt_estimator} nperms {numperm}: ", np.mean(time_jidt) )
	print(f" network_analysis {analysis} {python_estimator} nperms {numperm}: ", np.mean(time_python) )


#### test nonlinear granger
def test_nonlinear_granger(analysis, est_type, numperm=300, samples=1000, reps=6):
	
	jidt_estimator = f"Jidt{est_type}"
	python_estimator = f"Python{est_type}"

	print(f"\n\nTesting nonlinear granger analysis via {analysis}")
	print(f"using mute data ({samples} samples, {reps} replications)\n")

	
	data = Data(normalise=False)  # initialise an empty data object
	data.generate_mute_data(n_samples=samples, n_replications=reps)
	data2 = copy.deepcopy(data)

	print("\n#### Analyse network Jidt\n")

	settings = {
	    "target": 1,   # mandatory in settings for nonlinear single target analysis
	    "sources": 0,  # optional in settings for nonlinear  single targetanalysis
	    "cmi_estimator": jidt_estimator,
	    "n_perm_max_stat": numperm,
	    "n_perm_min_stat": numperm,
	    "n_perm_omnibus": numperm,
	    "n_perm_max_seq": numperm,
	    "max_lag_sources": 5,
	    "min_lag_sources": 1,
	}

	# prepare data object for nonlinear analysis
	settings, data = data.prepare_nonlinear(settings, data)

	nonlin_analysis = MultivariateTE()
	
	# perform JidtGaussianCMI WITH nonlinear data
	itic = time.perf_counter()
	results_jidt = nonlin_analysis.analyse_network(settings, data)
	itoc = time.perf_counter()
	time_jidt = itoc - itic
	
	print("\n#### Analyse network Python\n")

	settings2 = {
	    "target": 1,   # mandatory in settings for nonlinear single target analysis
	    "sources": 0,  # optional in settings for nonlinear  single targetanalysis
	    "cmi_estimator": python_estimator,
	    "n_perm_max_stat": numperm,
	    "n_perm_min_stat": numperm,
	    "n_perm_omnibus": numperm,
	    "n_perm_max_seq": numperm,
	    "max_lag_sources": 5,
	    "min_lag_sources": 1,
	}

	# prepare data object for nonlinear analysis
	settings2, data2 = data2.prepare_nonlinear(settings2, data2)

	# perform PythonGaussianCMI WITH nonlinear data
	itic = time.perf_counter()
	results_python = nonlin_analysis.analyse_network(settings2, data2)
	itoc = time.perf_counter()
	time_python = itoc - itic

	print(f"\nSummary nonlinear granger network analysis {jidt_estimator} vs {python_estimator}\n")
	
	print("ts = target_sources")
	print("type = type of target sources: 1=lin, 2=nonlin")
	print(f"\t{jidt_estimator}\t\t{python_estimator}")
	print("target\tts\t\ttype\tts\ttype\t\tequal ts\tequal type")
	for t in range(5):
		ts_jidt = results_jidt.get_nonlinear_target_sources(t, fdr=False)
		ts_python = results_python.get_nonlinear_target_sources(t, fdr=False)
		tt_jidt = results_jidt.get_target_source_types(t, fdr=False)
		tt_python = results_python.get_target_source_types(t, fdr=False)

		try:
			equal_ts = ts_jidt==ts_python
		except:
			equal_ts = False
		try:
			equal_tt = tt_jidt==tt_python
		except:
			equal_tt = False

		print(f"{t}\t\t{ts_jidt}{"\t\t" if len(ts_jidt)<=1 else "\t"}{tt_jidt}{"\t\t" if len(ts_jidt)<=1 else "\t"}{ts_python}{"\t\t" if len(ts_python)<=1 else "\t"}{tt_python}\t\t{equal_ts}{"\t\t" if len(tt_python)<=1 else "\t"}{equal_tt}")

	print("\n calculation times:")
	print(f" nonlinear Granger via {analysis} {jidt_estimator}: ", np.mean(time_jidt) )
	print(f" nonlinear Granger via {analysis} {python_estimator}: ", np.mean(time_python) )
	


if __name__ == '__main__':

	#### Test Kraskov estimators
	"""
	testhead("KraskovMI")
	test_kraskov_mi()
	
	testhead("KraskovMI local values")
	test_kraskov_mi_local_values()
	
	testhead("KraskovCMI")
	test_kraskov_cmi()
	
	testhead("KraskovCMI local values")
	test_kraskov_cmi_local_values()
	
	testhead("KraskovAIS")
	test_kraskov_ais()
	
	testhead("KraskovAIS local values")
	test_kraskov_ais_local_values()
	
	testhead("KraskovTE")
	test_kraskov_te()
	
	testhead("KraskovTE local values")
	test_kraskov_te_local_values()
	
	testhead("KraskovCTE")
	test_Kraskov_cte()
	
	testhead("KraskovCTE local values")
	test_kraskov_cte_local_values()
	"""

    #### Test Gaussian estimators
	"""
	testhead("GaussianMI")
	test_gaussian_mi()
	
	testhead("GaussianMI local values") 
	test_gaussian_mi_local_values()
	
	testhead("GaussianCMI")
	test_gaussian_cmi()
	
	testhead("GaussianCMI local values")
	test_gaussian_cmi_local_values()
	
	testhead("GaussianAIS")
	test_gaussian_ais()
	
	testhead("GaussianAIS local values")
	test_gaussian_ais_local_values()
	
	testhead("GaussianTE")
	test_gaussian_te()
	
	testhead("GaussianTE local values")
	test_gaussian_te_local_values()
	
	testhead("GaussianCTE")
	test_gaussian_cte()
	
	testhead("GaussianCTE local values")
	test_gaussian_cte_local_values()
	"""
	
	#### Test Discrete estimators
	"""
	testhead("DiscreteMI")
	test_discrete_mi()

	testhead("DiscreteMI local values")
	test_discrete_mi_local_values()

	testhead("DiscreteCMI")
	test_discrete_cmi()
	
	testhead("DiscreteCMI local values")
	test_discrete_cmi_local_values()
	
	testhead("DiscreteAIS")
	test_discrete_ais()

	testhead("DiscreteAIS local values")
	test_discrete_ais_local_values()

	testhead("DiscreteTE")
	test_discrete_te()

	testhead("DiscreteTE local values")
	test_discrete_te_local_values()
	"""

	#### Test analytic distributions
	"""
	testhead("analytic distribution Gaussian")
	test_analytic_distribution_mi_gaussian()
	test_analytic_distribution_cmi_gaussian()
	test_analytic_distribution_cmi_nocond_gaussian()
	test_analytic_distribution_ais_gaussian()
	test_analytic_distribution_te_gaussian()
	test_analytic_distribution_cte_gaussian()
	test_analytic_distribution_cte_nocond_gaussian()
	"""
	"""
	testhead("analytic distribution Discrete")
	test_analytic_distribution_mi_discrete()
	test_analytic_distribution_cmi_discrete()
	test_analytic_distribution_cmi_nocond_discrete()
	test_analytic_distribution_ais_discrete()
	test_analytic_distribution_te_discrete()
	"""

	#### Test bi- and multivariate analysis (single target) ################## TODO file
	# Kraskov CMI
	"""
	testhead("BivariateMI KraskovCMI (analyse_single_target)")
	test_single_target_analysis("BivariateMI","Kraskov", samples=500)

	testhead("BivariateTE KraskovCMI (analyse_single_target)")
	test_single_target_analysis("BivariateTE","Kraskov", samples=500)
	"""
	testhead("MultivariateMI KraskovCMI (analyse_single_target)")
	test_single_target_analysis("MultivariateMI","Kraskov", samples=100)

	testhead("MultivariateTE KraskovCMI (analyse_single_target)")
	test_single_target_analysis("MultivariateTE","Kraskov", samples=100)
	

	# Gaussian CMI
	"""
	testhead("BivariateMI GaussianCMI (analyse_single_target)")
	test_single_target_analysis("BivariateMI","Gaussian")

	testhead("BivariateTE GaussianCMI (analyse_single_target)")
	test_single_target_analysis("BivariateTE","Gaussian")

	testhead("MultivariateMI GaussianCMI (analyse_single_target)")
	test_single_target_analysis("MultivariateMI","Gaussian")

	testhead("MultivariateTE GaussianCMI (analyse_single_target)")
	test_single_target_analysis("MultivariateTE","Gaussian")
	"""
	
	#Discrete CMI
	"""
	testhead("BivariateMI DiscreteCMI (analyse_single_target)")
	test_single_target_analysis("BivariateMI","Discrete")
	
	testhead("BivariateTE DiscreteCMI (analyse_single_target)")
	test_single_target_analysis("BivariateTE","Discrete")

	testhead("MultivariateMI DiscreteCMI (analyse_single_target)")
	test_single_target_analysis("MultivariateMI","Discrete")

	testhead("MultivariateTE DiscreteCMI (analyse_single_target)")
	test_single_target_analysis("MultivariateTE","Discrete")
	"""

	#### Test network analysis CMI
	# Kraskov
	"""
	testhead("network analysis BivariateMI KraskovCMI")
	test_network_analysis("BivariateMI","Kraskov", numperm=21, samples=100, reps=3)

	testhead("network analysis BivariateTE KraskovCMI")
	test_network_analysis("BivariateTE","Kraskov", numperm=21, samples=100, reps=3)
	
	testhead("network analysis MultivariateMI KraskovCMI")
	test_network_analysis("MultivariateMI","Kraskov", numperm=21, samples=100, reps=3)

	testhead("network analysis MultivariateTE KraskovCMI")
	test_network_analysis("MultivariateTE","Kraskov", numperm=21, samples=100, reps=3)
	"""

	# Gaussian
	"""
	testhead("network analysis BivariateMI GaussianCMI")
	test_network_analysis("BivariateMI","Gaussian", numperm=300, samples=500, reps=6)

	testhead("network analysis BivariateTE GaussianCMI")
	test_network_analysis("BivariateTE","Gaussian", numperm=300, samples=500, reps=6)

	testhead("network analysis MultivariateMI GaussianCMI")
	test_network_analysis("MultivariateMI","Gaussian", numperm=300, samples=500, reps=6)

	testhead("network analysis MultivariateTE GaussianCMI")
	test_network_analysis("MultivariateTE","Gaussian", numperm=300, samples=500, reps=6)
	"""

	# Discrete
	"""
	testhead("network analysis BivariateMI DiscreteCMI")
	test_network_analysis("BivariateMI","Discrete", numperm=300, samples=600, reps=6)

	testhead("network analysis BivariateTE DiscreteCMI")
	test_network_analysis("BivariateTE","Discrete", numperm=300, samples=600, reps=6)
	
	testhead("network analysis MultivariateMI DiscreteCMI")
	test_network_analysis("MultivariateMI","Discrete", numperm=300, samples=600, reps=6)

	testhead("network analysis MultivariateTE DiscreteCMI")
	test_network_analysis("MultivariateTE","Discrete", numperm=300, samples=600, reps=6)
	"""

	# Test nonlinear Granger analysis
	"""
	testhead("nonlinear granger network analysis BivariateTE GaussianCMI") 
	test_nonlinear_granger("BivariateTE", "GaussianCMI", numperm=300, samples=500, reps=3)
	
	testhead("nonlinear granger network analysis MultivariateTE GaussianCMI")
	test_nonlinear_granger("MultivariateTE", "GaussianCMI", numperm=500, samples=500, reps=3)
	"""
