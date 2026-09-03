"""
Provide tests to compare OpenCL and Python estimators

THIS TEST DOES NOT RUN WITHOUT PRESELECTING TESTS!

Hences, you should run appropriate parts of it separately
(by uncommenting them in the main section at the end) and
pype the output to a text file:

e.g.
python systemtest_estimators_opencl.py > your_output_file.txt

BE AWARE:
Running all tests in one go will take several hours and will
produce a very long output!

"""



import numpy as np

import time
import sys
import copy


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

from idtxl.estimators_jidt import (JidtKraskovMI,
									JidtKraskovCMI,
									JidtGaussianMI,
									JidtGaussianCMI)



from idtxl.idtxl_utils import calculate_mi
from idtxl.data import Data

import random as rn
import itertools
from generate_test_data import (_get_gauss_data,
								_generate_mute_data,
								_get_ar_data,
								_get_mem_binary_data)

from idtxl.multivariate_te import MultivariateTE
from idtxl.bivariate_te import BivariateTE
from idtxl.multivariate_mi import MultivariateMI
from idtxl.bivariate_mi import BivariateMI

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


n_samples = 10000

#### Test Gaussian estimators
def test_gaussian_mi():

	expected_mi, source1, source2, target = _get_gauss_data(n=n_samples, expand=True, seed=SEED)
	
	mi_jidt_cor = np.zeros(4)
	mi_jidt_uncor = np.zeros(4)
	mi_python_cor = np.zeros(4)
	mi_python_uncor = np.zeros(4)
	mi_opencl_cor = np.zeros(4)
	mi_opencl_uncor = np.zeros(4)
	time_jidt_cor = np.zeros(4)
	time_jidt_uncor = np.zeros(4)
	time_python_cor = np.zeros(4)
	time_python_uncor = np.zeros(4)
	time_opencl_cor = np.zeros(4)
	time_opencl_uncor = np.zeros(4)
	
	vals = [0,1,2,3]

	print(f"\n\nTesting average MI using 1D gaussian data with covariance 0.4 - correlated and uncorrelated:")
	print(f"testing settings lag_mi {vals}")
	print(f"n_samples = {n_samples}\n")
	
	for lags in vals:
		settings = {}
		settings = {"lag_mi": lags,
					"noise_level": 0}

		jidt_estimator = JidtGaussianMI(settings)
		python_estimator = PythonGaussianMI(settings)
		opencl_estimator = OpenCLGaussianMI(settings)

		# opencl
		itic = time.perf_counter()
		mi_opencl_cor[lags] = opencl_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_opencl_cor[lags] = itoc - itic
		
		itic = time.perf_counter()
		mi_opencl_uncor[lags] = opencl_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		time_opencl_uncor[lags] = itoc - itic
		
		# python
		itic = time.perf_counter()
		mi_python_cor[lags] = python_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_python_cor[lags] = itoc - itic
		
		itic = time.perf_counter()
		mi_python_uncor[lags] = python_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		time_python_uncor[lags] = itoc - itic

		# kraskov
		itic = time.perf_counter()
		mi_jidt_cor[lags] = jidt_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_jidt_cor[lags] = itoc - itic
		
		itic = time.perf_counter()
		mi_jidt_uncor[lags] = jidt_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		time_jidt_uncor[lags] = itoc - itic
		

	print(f"Summary Jidt vs Python vs OpenCL GaussianMI testing lags ({vals}):")

	print("MI values:")
	print("correlated data:")
	print("lag\tJidtGaussianMI\t\tPythonGaussianMI\tOpenCLGaussianMI")
	for i in vals:
		print(f"{i}\t{mi_jidt_cor[i]}\t{mi_python_cor[i]}\t{mi_opencl_cor[i]}")
	
	verbose(mi_opencl_cor, mi_python_cor, "OpenCL vs Python correlated", "MI", local=False)
	

	print("\nuncorrelated data:")
	print("lag\tJidtGaussianMI\t\tPythonGaussianMI\tOpenCLGaussianMI")
	for i in vals:
		print(f"{i}\t{mi_jidt_uncor[i]}\t{mi_python_uncor[i]}\t{mi_opencl_uncor[i]}")

	verbose(mi_jidt_uncor, mi_python_uncor, "OpenCL vs Python uncorrelated", "MI", local=False)
	
	print("\nmean calculation times:")
	print(" JidtGaussianMI (cor): ", np.mean(time_jidt_cor) )
	print(" PythonGaussianMI (cor): ", np.mean(time_python_cor) )
	print(" OpenCLGaussianMI (cor): ", np.mean(time_opencl_cor) )
	print(" JidtGaussianMI (uncor): ", np.mean(time_jidt_uncor) )
	print(" PythonGaussianMI (uncor): ", np.mean(time_python_uncor) )
	print(" OpenCLGaussianMI (uncor): ", np.mean(time_opencl_uncor) )

	print("\n=========================================================================")

	# test 2D
	vals = [0,1,2,3]
	print(f"\n\nTesting average MI using 2D mute data with and without coupling")
	print(f"testing settings lag_mi {vals}")
	print(f"n_samples = {n_samples}\n")
		
	data = _generate_mute_data(n_samples=n_samples)

	source1 = data[0,:,:]
	target = data[2,:,:]
	source2 = data[4,:,:]

	mi_jidt_cor = np.zeros(4)
	mi_jidt_uncor = np.zeros(4)
	mi_python_cor = np.zeros(4)
	mi_python_uncor = np.zeros(4)
	mi_opencl_cor = np.zeros(4)
	mi_opencl_uncor = np.zeros(4)
	time_jidt_cor = np.zeros(4)
	time_jidt_uncor = np.zeros(4)
	time_python_cor = np.zeros(4)
	time_python_uncor = np.zeros(4)
	time_opencl_cor = np.zeros(4)
	time_opencl_uncor = np.zeros(4)


	for lags in vals:
		settings = {"lag_mi": lags}

		jidt_estimator = JidtGaussianMI(settings)
		python_estimator = PythonGaussianMI(settings)
		opencl_estimator = OpenCLGaussianMI(settings)
		
		# opencl
		itic = time.perf_counter()
		mi_opencl_cor[lags] = opencl_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_opencl_cor[lags] = itoc - itic

		itic = time.perf_counter()
		mi_opencl_uncor[lags] = opencl_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		time_opencl_uncor[lags] = itoc - itic

		# python
		itic = time.perf_counter()
		mi_python_cor[lags] = python_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_python_cor[lags] = itoc - itic

		itic = time.perf_counter()
		mi_python_uncor[lags] = python_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		time_python_uncor[lags] = itoc - itic

		# jidt
		itic = time.perf_counter()
		mi_jidt_cor[lags] = jidt_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_jidt_cor[lags] = itoc - itic

		itic = time.perf_counter()
		mi_jidt_uncor[lags] = jidt_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		time_jidt_uncor[lags] = itoc - itic

		
	print(f"Summary Jidt vs Python vs OpenCL GaussianMI 2D input testing lags ({vals}):")

	print("MI values:")
	print("coupled data:")
	print("lag\tJidtGaussianMI\t\tPythonGaussianMI\tOpenCLGaussianMI")
	for i in vals:
		print(f"{i}\t{mi_jidt_cor[i]}\t{mi_python_cor[i]}\t{mi_opencl_cor[i]}")
	
	verbose(mi_opencl_cor, mi_python_cor, "OpenCL vs Python with coupling", "MI", local=False)


	print("not coupled data:")
	print("lag\tJidtGaussianMI\t\tPythonGaussianMI\tOpenCLGaussianMI")
	for i in vals:
		print(f"{i}\t{mi_jidt_uncor[i]}\t{mi_python_uncor[i]}\t{mi_opencl_uncor[i]}")

	verbose(mi_opencl_uncor, mi_python_uncor, "OpenCL vs Python without coupling", "MI", local=False)

	print("\nmean calculation times:")
	print(" JidtGaussianMI (coupled): ", np.mean(time_jidt_cor) )
	print(" PythonGaussianMI (coupled): ", np.mean(time_python_cor) )
	print(" OpenCLGaussianMI (coupled): ", np.mean(time_opencl_cor) )
	print(" JidtGaussianMI (not coupled): ", np.mean(time_jidt_uncor) )
	print(" PythonGaussianMI (not coupled): ", np.mean(time_python_uncor) )
	print(" OpenCLGaussianMI (not coupled): ", np.mean(time_opencl_uncor) )

	print("\n=========================================================================")
	
	# test mixed dimension input
	d = [1, 2, 3, 5]
	print(f"\n\nTesting average MI using mixed dimensions\ntesting dimensions {d} for var1 and var2 each")
	print(f"n_samples = {n_samples}\n")
	
	print("Shapes:")
	data = _generate_mute_data(n_samples=n_samples, n_replications=5)
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
			opencl_estimator = OpenCLGaussianMI(settings)
		
			itic = time.perf_counter()
			mi_opencl_cor = opencl_estimator.estimate(source1, target)
			itoc = time.perf_counter()
			time_opencl_cor = itoc - itic

			itic = time.perf_counter()
			mi_python_cor = python_estimator.estimate(source1, target)
			itoc = time.perf_counter()
			time_python_cor = itoc - itic

			itic = time.perf_counter()
			mi_jidt_cor = jidt_estimator.estimate(source1, target)
			itoc = time.perf_counter()
			time_jidt_cor = itoc - itic
			
			
			verbose(mi_opencl_cor, mi_python_cor, f"{cond} OpenCL vs Python", "MI")

def test_gaussian_mi_local_values():
	
	vals = [0,1,2,3]
	print(f"\n\nTesting local MI using 1D gaussian data with covariance 0.4 - uncorrelated and uncorrelated")
	print(f"testing settings lag_mi {vals}")
	print(f"n_samples = {n_samples}\n")
	
	expected_mi, source1, source2, target = _get_gauss_data(n=n_samples, expand=True, seed=SEED)
	
	time_jidt_cor = np.zeros(4)
	time_jidt_uncor = np.zeros(4)
	time_python_cor = np.zeros(4)
	time_python_uncor = np.zeros(4)
	time_opencl_cor = np.zeros(4)
	time_opencl_uncor = np.zeros(4)

	print("\nTesting lags:")
	for lags in vals:
		settings = {}
		settings = {"lag_mi": lags,
					"local_values": True,
					"noise_level": 0}
		
		jidt_estimator = JidtGaussianMI(settings)
		python_estimator = PythonGaussianMI(settings)
		opencl_estimator = OpenCLGaussianMI(settings)

		itic = time.perf_counter()
		mi_opencl_cor = opencl_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_opencl_cor[lags] = itoc - itic
		
		itic = time.perf_counter()
		mi_opencl_uncor = opencl_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		time_opencl_uncor[lags] = itoc - itic
		
		itic = time.perf_counter()
		mi_python_cor = python_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_python_cor[lags] = itoc - itic
		
		itic = time.perf_counter()
		mi_python_uncor = python_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		time_python_uncor[lags] = itoc - itic
		
		itic = time.perf_counter()
		mi_jidt_cor = jidt_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_jidt_cor[lags] = itoc - itic

		itic = time.perf_counter()
		mi_jidt_uncor = jidt_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		time_jidt_uncor[lags] = itoc - itic

		verbose(mi_opencl_cor, mi_python_cor, f"{lags} OpenCL vs Python ", "MI (corr)", local=True)
		verbose(mi_opencl_uncor, mi_python_uncor, f"{lags} OpenCL vs Python ", "MI (uncorr)", local=True)
	
	print("\nmean calculation times:")
	print(" JidtGaussianMI (corr): ", np.mean(time_jidt_cor) )
	print(" PythonGaussianMI (corr): ", np.mean(time_python_cor) )
	print(" OepnCLGaussianMI (corr): ", np.mean(time_opencl_cor) )
	print(" JidtGaussianMI (uncorr): ", np.mean(time_jidt_uncor) )
	print(" PythonGaussianMI (uncorr): ", np.mean(time_python_uncor) )
	print(" OpenCLGaussianMI (uncorr): ", np.mean(time_opencl_uncor) )

	print("\n=========================================================================")

	# test 2D
	print(f"\n\nTesting local MI using 2D mute data with and without coupling")
	print(f"testing settings lag_mi {vals}\n")
	print(f"n_samples = {n_samples}\n")
	
	data = _generate_mute_data(n_samples=n_samples)

	source1 = data[0,:,:]
	target = data[2,:,:]
	source2 = data[4,:,:]

	vals = [0,1,2,3]
	
	time_jidt_cor = np.zeros(len(vals))
	time_jidt_uncor = np.zeros(len(vals))
	time_python_cor = np.zeros(len(vals))
	time_python_uncor = np.zeros(len(vals))
	time_opencl_cor = np.zeros(len(vals))
	time_opencl_uncor = np.zeros(len(vals))

	print("Testing lags:")
	for lags in vals:
		settings = {}
		settings = {"lag_mi": lags,
					'local_values': True,
					'discretise_method': 'max_ent'}

		jidt_estimator = JidtGaussianMI(settings)
		python_estimator = PythonGaussianMI(settings)
		opencl_estimator = OpenCLGaussianMI(settings)

		itic = time.perf_counter()
		mi_opencl_cor = opencl_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_opencl_cor[lags] = itoc - itic
		
		itic = time.perf_counter()
		mi_opencl_uncor = opencl_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		time_opencl_uncor[lags] = itoc - itic
				
		itic = time.perf_counter()
		mi_python_cor = python_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_python_cor[lags] = itoc - itic
		
		itic = time.perf_counter()
		mi_python_uncor = python_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		time_python_uncor[lags] = itoc - itic
				
		itic = time.perf_counter()
		mi_jidt_cor = jidt_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_jidt_cor[lags] = itoc - itic
		
		itic = time.perf_counter()
		mi_jidt_uncor = jidt_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		time_jidt_uncor[lags] = itoc - itic

		verbose(mi_opencl_cor[:900], mi_python_cor[:900], f"{lags} OpenCL vs Python ", "MI (coupled) 2D input", local=True, atol=1e-03)
		verbose(mi_opencl_uncor[:900], mi_python_uncor[:900], f"{lags} OpenCL vs Python ", "MI (not couled) 2D input", local=True, atol=1e-03)

	print("\nmean calculation times:")
	print(" JidtGaussianMI (coupled): ", np.mean(time_jidt_cor) )
	print(" PythonGaussianMI (coupled): ", np.mean(time_python_cor) )
	print(" OpenCLGaussianMI (coupled): ", np.mean(time_opencl_cor) )
	print(" JidtGaussianMI (not coupled): ", np.mean(time_jidt_uncor) )
	print(" PythonGaussianMI (not coupled): ", np.mean(time_python_uncor) )
	print(" OpenCLGaussianMI (not coupled): ", np.mean(time_opencl_uncor) )

def test_gaussian_cmi():

	cmi_jidt_cor = np.zeros(8)
	cmi_python_cor = np.zeros(8)
	cmi_opencl_cor = np.zeros(8)
	cmi_jidt_uncor = np.zeros(8)
	cmi_python_uncor = np.zeros(8)
	cmi_opencl_uncor = np.zeros(8)
	time_jidt_cor = np.zeros(8)
	time_python_cor = np.zeros(8)
	time_opencl_cor = np.zeros(8)
	time_jidt_uncor = np.zeros(8)
	time_python_uncor = np.zeros(8)
	time_opencl_uncor = np.zeros(8)
	
	vals = [0.2, 0.4, 0.6, 0.8]

	print(f"\n\nTesting average CMI using 1D gaussian data with different \ncovariances: {vals} - uncorrelated conditional vs uncorrelated source")
	print(f"n_samples = {n_samples}\n")
	
	count = 0
	for i in vals:

		expected_mi, source1, source2, target = _get_gauss_data(n=n_samples, expand=True, covariance=i, seed=SEED)

		settings={}
		
		jidt_estimator = JidtGaussianCMI(settings)
		python_estimator = PythonGaussianCMI(settings)
		opencl_estimator = OpenCLGaussianCMI(settings)

		itic = time.perf_counter()
		cmi_opencl_cor[count] = opencl_estimator.estimate(source1, target, source2)
		itoc = time.perf_counter()
		time_opencl_cor[count] += itoc - itic

		itic = time.perf_counter()
		cmi_opencl_uncor[count] = opencl_estimator.estimate(source2, target, source1)
		itoc = time.perf_counter()
		time_opencl_uncor[count] += itoc - itic
		
		itic = time.perf_counter()
		cmi_python_cor[count] = python_estimator.estimate(source1, target, source2)
		itoc = time.perf_counter()
		time_python_cor[count] += itoc - itic

		itic = time.perf_counter()
		cmi_python_uncor[count] = python_estimator.estimate(source2, target, source1)
		itoc = time.perf_counter()
		time_python_uncor[count] += itoc - itic

		itic = time.perf_counter()
		cmi_jidt_cor[count] = jidt_estimator.estimate(source1, target, source2)
		itoc = time.perf_counter()
		time_jidt_cor[count] = itoc - itic

		itic = time.perf_counter()
		cmi_jidt_uncor[count] = jidt_estimator.estimate(source2, target, source1)
		itoc = time.perf_counter()
		time_jidt_uncor[count] += itoc - itic

		count += 1 

	print("cov\tJidtGaussianCMI\t\tPythonGaussianCMI\tOpenCLGaussianCMI")
	print("uncorr conditional")
	for i in range(len(vals)):
		print(f"{vals[i]}\t{cmi_jidt_cor[i]}\t{cmi_python_cor[i]}\t{cmi_opencl_cor[i]}")
	verbose(cmi_jidt_cor, cmi_python_cor, "OpenCL vs Python", "CMI", local=False)
	print("uncorr source")
	for i in range(len(vals)):
		print(f"{vals[i]}\t{cmi_jidt_uncor[i]}\t{cmi_python_uncor[i]}\t{cmi_opencl_uncor[i]}")
	verbose(cmi_jidt_uncor, cmi_python_uncor, "OpenCL vs Python", "CMI", local=False)

	print("\nmean calculation times:")
	print(" JidtGaussianCMI (uncorrelated conditional): ", np.mean(time_jidt_cor) )
	print(" PythonGaussianCMI (uncorrelated conditional): ", np.mean(time_python_cor) )
	print(" OpenCLGaussianCMI (uncorrelated conditional): ", np.mean(time_opencl_cor) )
	print(" JidtGaussianCMI (uncorrelated source): ", np.mean(time_jidt_uncor) )
	print(" PythonGaussianCMI (uncorrelated source): ", np.mean(time_python_uncor) )
	print(" OpenCLGaussianCMI (uncorrelated source): ", np.mean(time_opencl_uncor) )

	print("\n=========================================================================")

	# test 2D input
	print(f"\n\nTesting average CMI using 2D mute data - uncorrelated conditional vs uncorrelated source\n")
	print(f"n_samples = {n_samples}\n")
	
	data = _generate_mute_data(n_samples=n_samples)

	source1 = data[0,:,:]
	target = data[2,:,:]
	source2 = data[4,:,:]

	settings={}
	
	time_jidt = 0
	time_python = 0
	time_opencl = 0

	python_estimator = PythonGaussianCMI(settings)
	opencl_estimator = OpenCLGaussianCMI(settings)
	
	itic = time.perf_counter()
	cmi_opencl = opencl_estimator.estimate(source1, target, source2)
	itoc = time.perf_counter()
	time_opencl += itoc - itic

	itic = time.perf_counter()
	cmi_python = python_estimator.estimate(source1, target, source2)
	itoc = time.perf_counter()
	time_python += itoc - itic

	verbose(cmi_opencl, cmi_python, f"OpenCL vs Python - uncorrelated conditional", "CMI", local=False)

	itic = time.perf_counter()
	cmi_opencl = opencl_estimator.estimate(source2, target, source1)
	itoc = time.perf_counter()
	time_opencl += itoc - itic

	itic = time.perf_counter()
	cmi_python = python_estimator.estimate(source2, target, source1)
	itoc = time.perf_counter()
	time_python += itoc - itic

	verbose(cmi_opencl, cmi_python, f"OpenCL vs Python - uncorrelated source", "CMI", local=False)

	print("\nmean calculation times:")
	print(" OpenCLGaussianCMI: ", np.mean(time_opencl) )
	print(" PythonGaussianCMI: ", np.mean(time_python) )

	print("\n=========================================================================")

	# test mixed dimension input
	d = [1, 2, 3]

	print(f"\n\nTesting average MI using mixed dimensions\ntesting dimensions {d} for var1, var2 and cond each")
	print(f"n_samples = {n_samples}\n")
	
	print("Shapes:")
	data = _generate_mute_data(n_samples=n_samples)

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

				python_estimator = PythonGaussianCMI(settings)
				opencl_estimator = OpenCLGaussianCMI(settings)
			
				itic = time.perf_counter()
				mi_opencl_cor = opencl_estimator.estimate(source1, target, conditional)
				itoc = time.perf_counter()
				time_opencl_cor = itoc - itic

				itic = time.perf_counter()
				mi_python_cor = python_estimator.estimate(source1, target, conditional)
				itoc = time.perf_counter()
				time_python_cor = itoc - itic

				verbose(mi_opencl_cor, mi_python_cor, f"{cond} OpenCL vs Python", "MI")

def test_gaussian_cmi_local_values():

	vals = [0.2, 0.4, 0.6, 0.8]

	print(f"\n\nTesting local CMI using 1D gaussian data with different \ncovariances: {vals} - uncorrelated conditional vs uncorrelated source")
	print(f"n_samples = {n_samples}\n")
	
	cmi_jidt_cor = np.zeros(len(vals))
	cmi_python_cor = np.zeros(len(vals))
	cmi_opencl_cor = np.zeros(len(vals))
	cmi_jidt_uncor = np.zeros(len(vals))
	cmi_python_uncor = np.zeros(len(vals))
	cmi_opencl_uncor = np.zeros(len(vals))
	time_jidt_cor = np.zeros(len(vals))
	time_python_cor = np.zeros(len(vals))
	time_opencl_cor = np.zeros(len(vals))
	time_jidt_uncor = np.zeros(len(vals))
	time_python_uncor = np.zeros(len(vals))
	time_opencl_uncor = np.zeros(len(vals))
	
	print("Tested cov\t\tOpenCLGaussianCMI vs PythonGaussianCMI")
	count = 0
	for i in vals:

		expected_mi, source1, source2, target = _get_gauss_data(n=n_samples, expand=True, covariance=i, seed=SEED)

		settings={'local_values': True,
			'normalise': False,
			'noise_level': 0}
		
		jidt_estimator = JidtGaussianCMI(settings)
		python_estimator = PythonGaussianCMI(settings)
		opencl_estimator = OpenCLGaussianCMI(settings)
		
		itic = time.perf_counter()
		cmi_opencl = opencl_estimator.estimate(source1, target, source2)
		cmi_opencl_cor[count] = np.mean(cmi_opencl)
		itoc = time.perf_counter()
		time_opencl_cor[count] = itoc - itic
		
		itic = time.perf_counter()
		cmi_python = python_estimator.estimate(source1, target, source2)
		cmi_python_cor[count] = np.mean(cmi_python)
		itoc = time.perf_counter()
		time_python_cor[count] += itoc - itic

		itic = time.perf_counter()
		cmi_jidt = jidt_estimator.estimate(source1, target, source2)
		cmi_jidt_cor[count] = np.mean(cmi_jidt)
		itoc = time.perf_counter()
		time_jidt_cor[count] = itoc - itic
		
		verbose(cmi_opencl, cmi_python, f"{i} OpenCL vs Python", "CMI (corr)", local=True)

		itic = time.perf_counter()
		cmi_opencl = opencl_estimator.estimate(source1, target, source2)
		cmi_opencl_uncor[count] = np.mean(cmi_opencl)
		itoc = time.perf_counter()
		time_opencl_uncor[count] += itoc - itic

		itic = time.perf_counter()
		cmi_python = python_estimator.estimate(source1, target, source2)
		cmi_python_uncor[count] = np.mean(cmi_python)
		itoc = time.perf_counter()
		time_python_uncor[count] += itoc - itic

		itic = time.perf_counter()
		cmi_jidt = jidt_estimator.estimate(source1, target, source2)
		cmi_jidt_uncor[count] = np.mean(cmi_jidt)
		itoc = time.perf_counter()
		time_jidt_uncor[count] += itoc - itic

		verbose(cmi_opencl, cmi_python, f"{i} OpenCL vs Python ", "CMI (uncorr)", local=True)

		count += 1 

	print("\nAverages of local cmi:")
	print("cov\tJidtGaussianCMI\t\tPythonGaussianCMI\tOpenCLGaussianCMI")
	print("uncorr conditional")
	for i in range(len(vals)):
		print(f"{vals[i]}\t{cmi_jidt_cor[i]}\t{cmi_python_cor[i]}\t{cmi_opencl_cor[i]}")
	verbose(cmi_opencl_cor, cmi_python_cor, "OpenCL vs Python ", "CMI", local=False)
	print("uncorr source")
	for i in range(len(vals)):
		print(f"{vals[i]}\t{cmi_jidt_uncor[i]}\t{cmi_python_uncor[i]}\t{cmi_opencl_uncor[i]}")
	verbose(cmi_opencl_uncor, cmi_python_uncor, "OpenCL vs Python ", "CMI", local=False)

	print("\nmean calculation times:")
	print(" JidtGaussianCMI (uncorrelated conditional): ", np.mean(time_jidt_cor) )
	print(" PythonGaussianCMI (uncorrelated conditional): ", np.mean(time_python_cor) )
	print(" OpenCLGaussianCMI (uncorrelated conditional): ", np.mean(time_opencl_cor) )
	print(" JidtGaussianCMI (uncorrelated source): ", np.mean(time_jidt_uncor) )
	print(" PythonGaussianCMI (uncorrelated source): ", np.mean(time_python_uncor) )
	print(" OpenCLGaussianCMI (uncorrelated source): ", np.mean(time_opencl_uncor) )

	print("\n=========================================================================")

	print(f"\n\nTesting average CMI using 2D mute data - uncorrelated conditional vs uncorrelated source")
	print(f"n_samples = {n_samples}\n")
	
	data = _generate_mute_data(n_samples=n_samples)

	source1 = data[0,:,:]
	target = data[2,:,:]
	source2 = data[4,:,:]

	settings={'local_values': True,
			'normalise': False,
			'noise_level': 0}
	
	time_python=0
	time_opencl=0

	opencl_estimator = OpenCLGaussianCMI(settings)
	python_estimator = PythonGaussianCMI(settings)
	
	itic = time.perf_counter()
	mi_opencl = opencl_estimator.estimate(source1, target, source2)
	itoc = time.perf_counter()
	time_opencl += itoc - itic
	
	itic = time.perf_counter()
	mi_python = python_estimator.estimate(source1, target, source2)
	itoc = time.perf_counter()
	time_python += itoc - itic

	verbose(mi_opencl, mi_python, f"OpenCL vs Python - uncorrelated conditional", "CMI", local=True)

	itic = time.perf_counter()
	mi_opencl = opencl_estimator.estimate(source2, target, source1)
	itoc = time.perf_counter()
	time_opencl += itoc - itic

	itic = time.perf_counter()
	mi_python = python_estimator.estimate(source2, target, source1)
	itoc = time.perf_counter()
	time_python += itoc - itic
	
	verbose(mi_opencl, mi_python, f"OpenCL vs Python - uncorrelated source", "CMI", local=True)

	print("\nmean calculation times:")
	print(" OpenCLGaussianCMI: ", np.mean(time_opencl) )
	print(" PythonGaussianCMI: ", np.mean(time_python) )
	
def test_gaussian_ais():

	print(f"\n\nTesting average AIS using 1D AR data with history and pure noise")
	print(f"n_samples = {n_samples}\n")
	
	source1, source2 = _get_ar_data(n=n_samples, seed=SEED)

	vals =  [1,2,3]
	time_opencl_cor = np.zeros(np.power(len(vals),2))
	res_opencl_cor = np.zeros(np.power(len(vals),2))
	time_python_cor = np.zeros(np.power(len(vals),2))
	res_python_cor = np.zeros(np.power(len(vals),2))
	time_opencl_uncor = np.zeros(np.power(len(vals),2))
	res_opencl_uncor = np.zeros(np.power(len(vals),2))
	time_python_uncor = np.zeros(np.power(len(vals),2))
	res_python_uncor = np.zeros(np.power(len(vals),2))
	
	conds = np.zeros([np.power(len(vals),2),2])

	count = 0

	for h in vals:
		for t in vals:

			conds[count,:] = [h,t]
			settings_j = {'history': h, 'tau': t}

			settings_p = {'history': h, 'tau': t}
	
			opencl_estimator = OpenCLGaussianAIS(settings=settings_j)
			python_estimator = PythonGaussianAIS(settings=settings_p)

			itic = time.perf_counter()
			res_opencl_cor[count] = opencl_estimator.estimate(source1)
			itoc = time.perf_counter()
			time_opencl_cor[count] = itoc - itic
	
			itic = time.perf_counter()
			res_opencl_uncor[count] = opencl_estimator.estimate(source2)
			itoc = time.perf_counter()
			time_opencl_uncor[count] = itoc - itic
			
			itic = time.perf_counter()
			res_python_cor[count] = python_estimator.estimate(source1)
			itoc = time.perf_counter()
			time_python_cor[count] = itoc - itic
			
			itic = time.perf_counter()
			res_python_uncor[count] = python_estimator.estimate(source2)
			itoc = time.perf_counter()
			time_python_uncor[count] = itoc - itic

			count += 1
	
	print(f"Summary OpenCL vs Python GaussianAIS 1D data testing history ({vals}) and tau ({vals}):")

	print("hist,tau\tOpenCLGaussianAIS\t\tPythonGaussianAIS")
	print("AR with history")
	for i in range(len(res_opencl_cor)):
		print(f"{conds[i]}\t{res_opencl_cor[i]}\t{res_python_cor[i]}")

	verbose(res_opencl_cor, res_python_cor, "", "AIS (with hist)", local=True)
	
	print("noise")
	for i in range(len(res_opencl_uncor)):
		print(f"{conds[i]}\t{res_opencl_uncor[i]}\t{res_python_uncor[i]}")

	verbose(res_opencl_uncor, res_python_uncor, "", "AIS (no hist)", local=True)
	
	print("\nmean calculation times:")
	print(" OpenCLGaussianAIS (with history): ", np.mean(time_opencl_cor) )
	print(" PythonGaussianAIS (with history): ", np.mean(time_python_cor) )
	print(" OpenCLGaussianAIS (noise): ", np.mean(time_opencl_uncor) )
	print(" PythonGaussianAIS (noise): ", np.mean(time_python_uncor) )

def test_gaussian_ais_local_values():

	print(f"\n\nTesting local AIS using 1D AR data with history and pure noise")
	print(f"n_samples = {n_samples}\n")
	
	source1, source2 = _get_ar_data(n=n_samples, seed=SEED)

	vals = [1,2,3]
	
	time_opencl = np.zeros(np.power(len(vals),2))
	time_python = np.zeros(np.power(len(vals),2))
	
	print("hist,tau\t\tOpenCLGaussianAIS\t\tPythonGaussianAIS\tclose")

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
				
			opencl_estimator = OpenCLGaussianAIS(settings_j)
			python_estimator = PythonGaussianAIS(settings_p)

			itic = time.perf_counter()
			ais_opencl = opencl_estimator.estimate(source1)
			itoc = time.perf_counter()
			time_opencl[count] = itoc - itic
			
			itic = time.perf_counter()
			ais_python = python_estimator.estimate(source1)
			itoc = time.perf_counter()
			time_python[count] = itoc - itic
			
			verbose(ais_opencl, ais_python, [h, t], "AIS (with hist)", local=True)
					
			count += 1

	print("\nmean calculation times:")
	print(" OpenCLGaussianAIS: ", np.mean(time_opencl) )
	print(" PythonGaussianAIS: ", np.mean(time_python) )

def test_gaussian_te():

	print(f"\n\nTesting average TE using 1D gaussian data with covariance 0.4 and lag 1")
	print(f"n_samples = {n_samples}\n")
	
	expected_mi, source1, source2, target = _get_gauss_data(n=n_samples, expand=True, seed=SEED)

	source1 = source1[1:]
	source2 = source2[1:]
	target = target[:-1]

	vals = [1,3]

	time_opencl = np.empty(np.power(len(vals),5))
	res_opencl = np.empty(np.power(len(vals),5))
	time_python = np.empty(np.power(len(vals),5))
	res_python = np.empty(np.power(len(vals),5))

	conds = np.empty((np.power(len(vals),5),5))

	print("hst,ht,tt,hs,ts\t\tOpenCLGaussianTE\t\tPythonGaussianTE\tclose 1e-03")

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

						opencl_estimator = OpenCLGaussianTE(settings_j)
						
						itic = time.perf_counter()
						te_opencl = opencl_estimator.estimate(source=source1, target=target)
						itoc = time.perf_counter()
						time_opencl[count] = itoc-itic
						res_opencl[count] = te_opencl

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
			
						print(f"{conds[count,:]}\t{te_opencl}\t{te_python}\t{np.isclose(te_opencl, te_python, rtol=1e-03, atol=1e-03)}")

						count += 1

	verbose(res_opencl, res_python, "", "TE")

	print("\nmean calculation times:")
	print(" OpenCLGaussianTE (cor): ", np.mean(time_opencl) )
	print(" PythonGaussianTE (cor): ", np.mean(time_python) )

def test_gaussian_te_local_values():

	print(f"\n\nTesting average TE using 1D gaussian data with covariance 0.4 and lag 1")
	print(f"n_samples = {n_samples}\n")
	
	expected_mi, source1, source2, target = _get_gauss_data(n=n_samples, expand=True, seed=SEED)

	source1 = source1[1:]
	source2 = source2[1:]
	target = target[:-1]

	vals = [2,4]
	
	time_opencl = np.empty(np.power(len(vals),5))
	res_opencl = np.empty(np.power(len(vals),5))
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

						opencl_estimator = OpenCLGaussianTE(settings_j)
						
						itic = time.perf_counter()
						te_opencl = opencl_estimator.estimate(source=source1, target=target)
						itoc = time.perf_counter()
						time_opencl[count] = itoc-itic
					
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
						
						verbose(te_opencl, te_python, [hst, ht, tt, hs, ts], "TE", local=True)
	
	print("\nmean calculation times:")
	print(" OpenCLGaussianTE: ", np.mean(time_opencl) )
	print(" PythonGaussianTE: ", np.mean(time_python) )

def test_gaussian_cte():

	vals = [1,3]
	print(f"\n\nTesting average CTE using 1D mute data - correlated and uncorrelated conditional\n")
	print(f"testing settings history_source, tau_source, history_target, tau_target, history_conditional")
	print(f"tau_conditional, source_target_delay and conditional_target_delay with {vals}")
	print(f"n_samples = {n_samples}\n")
		
	data = _generate_mute_data(n_samples=n_samples, n_replications=1)
	source1 = data[0,:]
	target = data[4,:]
	cond = data[3,:]
	nocond = data[5,:]

	time_opencl_cond = np.empty(np.power(len(vals),8))
	res_opencl_cond = np.empty(np.power(len(vals),8))
	time_opencl_nocond = np.empty(np.power(len(vals),8))
	res_opencl_nocond = np.empty(np.power(len(vals),8))
	
	time_python_cond = np.empty(np.power(len(vals),8))
	res_python_cond = np.empty(np.power(len(vals),8))
	time_python_nocond = np.empty(np.power(len(vals),8))
	res_python_nocond = np.empty(np.power(len(vals),8))
	
	conds = np.empty((np.power(len(vals),5),8))

	atol = 1e-03

	print(f"\t\t\t\tOpenCLGaussianCTE\t\tPythonGaussianCTE\t\tclose {atol}")
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
									
									opencl_estimator = OpenCLGaussianCTE(settings)
									
									itic = time.perf_counter()
									cte_opencl_cond = opencl_estimator.estimate(source1, target, cond)
									itoc = time.perf_counter()
									res_opencl_cond[count] = cte_opencl_cond
									time_opencl_cond[count] = itoc - itic
									
									itic = time.perf_counter()
									cte_opencl_nocond = opencl_estimator.estimate(source1, target, nocond)
									itoc = time.perf_counter()
									res_opencl_nocond[count] = cte_opencl_nocond
									time_opencl_nocond[count] = itoc - itic
									
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
									
									print(f"{hst,cst,ht,tt,hs,ts,hc,tc}\t{cte_opencl_cond}\t{cte_python_cond}\t\t{np.isclose(cte_opencl_cond, cte_python_cond, rtol=atol, atol=atol)}\t\t{np.isclose(cte_opencl_nocond, cte_python_nocond, rtol=atol, atol=atol)}")

									count += 1 

	verbose(res_opencl_cond, res_python_cond, "correlated conditional", "CTE", atol=1e-04)
	verbose(res_opencl_nocond, res_python_nocond, "uncorrelated conditional" , "CTE", atol=1e-04)

	print("\nmean calculation times:")
	print(" OpenCLGaussianCTE (correlated conditional): ", np.mean(time_opencl_cond) )
	print(" PythonGaussianCTE (correlated conditional): ", np.mean(time_python_cond) )
	print(" OpenCLGaussianCTE (uncorrelated conditional): ", np.mean(time_opencl_nocond) )
	print(" PythonGaussianCTE (uncorrelated conditional): ", np.mean(time_python_nocond) )

def test_gaussian_cte_local_values():
	
	vals = [2,4]
	print(f"\n\nTesting local CTE using 1D mute data - correlated and uncorrelated conditional")
	print(f"testing settings history_source, tau_source, history_target, tau_target, history_conditional")
	print(f"tau_conditional, source_target_delay and conditional_target_delay with {vals}\n")
	print(f"n_samples = {n_samples}\n")
	
	data = _generate_mute_data(n_samples=n_samples, n_replications=1)
	source1 = data[0,:]
	target = data[4,:]
	cond = data[3,:]
	nocond = data[5,:]

	time_opencl_cond = np.empty(np.power(len(vals),8))
	time_opencl_nocond = np.empty(np.power(len(vals),8))
	
	time_python_cond = np.empty(np.power(len(vals),8))
	time_python_nocond = np.empty(np.power(len(vals),8))
	
	atol = 1e-03

	print("std,ctd,ht,tt,hs,ts,hc,tc\t\t\tOpenCLGaussianCTE vs PythonGaussianCTE")

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
																		
									opencl_estimator = OpenCLGaussianCTE(settings)
									
									itic = time.perf_counter()
									cte_opencl_cond = opencl_estimator.estimate(source1, target, cond)
									itoc = time.perf_counter()
									time_opencl_cond[count] = itoc - itic
									
									itic = time.perf_counter()
									cte_opencl_nocond = opencl_estimator.estimate(source1, target, nocond)
									itoc = time.perf_counter()
									time_opencl_nocond[count] = itoc - itic
									
									python_estimator = PythonGaussianCTE(settings)
									
									itic = time.perf_counter()
									cte_python_cond = python_estimator.estimate(source1, target, cond)
									itoc = time.perf_counter()
									time_python_cond[count] = itoc - itic
									
									itic = time.perf_counter()
									cte_python_nocond = python_estimator.estimate(source1, target, nocond)
									itoc = time.perf_counter()
									time_python_nocond[count] = itoc - itic
									
									verbose(cte_opencl_cond, cte_python_cond, f"{hst,cst,ht,tt,hs,ts,hc,tc} corr conditional", "CTE", local=True, atol=atol) 
									verbose(cte_opencl_nocond, cte_python_nocond, f"{hst,cst,ht,tt,hs,ts,hc,tc} uncorr conditional", "CTE", local=True, atol=atol) 

									count += 1
	print("\nmean calculation times:")
	print(" OpenCLGaussianCTE (correlated conditional): ", np.mean(time_opencl_cond) )
	print(" PythonGaussianCTE (correlated conditional): ", np.mean(time_python_cond) )
	print(" OpenCLGaussianCTE (uncorrelated conditional): ", np.mean(time_opencl_nocond) )
	print(" PythonGaussianCTE (uncorrelated conditional): ", np.mean(time_python_nocond) )


#### Test Discrete estimators
def test_discrete_mi():
	vals = [2, 5, 8, 32]
	lvals = [0, 1, 2, 3]

	# test 1D gaussian
	print(f"\n\nTesting average MI using 1D gaussian data with covariance 0.4 - correlated and uncorrelated")
	print(f"testing settings lag_mi {lvals}, n_discrete_bins {vals} and discrete_method max_ent and equal")
	print(f"n_samples = {n_samples}\n")

	expected_mi, source1, source2, target = _get_gauss_data(n=n_samples, expand=True, seed=SEED)

	for m in ['max_ent', 'equal']:
		print(f"\n--- discretise_method: {m}\n")
		mi_opencl_cor = np.zeros(np.power(len(vals), 2))
		mi_python_cor = np.zeros(np.power(len(vals), 2))
		time_opencl_cor = np.zeros(np.power(len(vals), 2))
		time_python_cor = np.zeros(np.power(len(vals), 2))

		mi_opencl_uncor = np.zeros(np.power(len(vals), 2))
		mi_python_uncor = np.zeros(np.power(len(vals), 2))
		time_opencl_uncor = np.zeros(np.power(len(vals), 2))
		time_python_uncor = np.zeros(np.power(len(vals), 2))

		conds = np.empty((np.power(len(vals), 2), 2))

		count = 0
		for l in lvals:
			for i in vals:
				conds[count, :] = [l, i]
				settings = {'discretise_method': m,
							'n_discrete_bins': i,
							'lag_mi': l}

				opencl_estimator = OpenCLDiscreteMI(settings=settings)
				itic = time.perf_counter()
				mi_opencl_cor[count] = opencl_estimator.estimate(source1, target)
				itoc = time.perf_counter()
				time_opencl_cor[count] = itoc - itic

				python_estimator = PythonDiscreteMI(settings=settings)
				itic = time.perf_counter()
				mi_python_cor[count] = python_estimator.estimate(source1, target)
				itoc = time.perf_counter()
				time_python_cor[count] = itoc - itic

				opencl_estimator = OpenCLDiscreteMI(settings=settings)
				itic = time.perf_counter()
				mi_opencl_uncor[count] = opencl_estimator.estimate(source2, target)
				itoc = time.perf_counter()
				time_opencl_uncor[count] = itoc - itic

				python_estimator = PythonDiscreteMI(settings=settings)
				itic = time.perf_counter()
				mi_python_uncor[count] = python_estimator.estimate(source2, target)
				itoc = time.perf_counter()
				time_python_uncor[count] = itoc - itic

				count += 1

		atol = 1e-03
		print(f"Summary OpenCL vs Python DiscreteMI discretised 1D gaussian data using {m}:")
		print(f"lags, nbins\tOpenCLDiscreteMI\t\tPythonDiscreteMI\tclose {atol}")
		print("correlated data:")
		for i in range(count):
			print(
				f"{conds[i]}   \t{mi_opencl_cor[i]}\t{mi_python_cor[i]}\t{np.isclose(mi_opencl_cor[i], mi_python_cor[i], atol=atol)}")

		print("\nuncorrelated data:")
		for i in range(count):
			print(
				f"{conds[i]}   \t{mi_opencl_uncor[i]}\t{mi_python_uncor[i]}\t{np.isclose(mi_opencl_uncor[i], mi_python_uncor[i], atol=atol)}")

		verbose(mi_opencl_cor, mi_python_cor, "correlated", "MI", local=False, atol=1e-03)
		verbose(mi_opencl_uncor, mi_python_uncor, "uncorrelated", "MI", local=False, atol=1e-03)

		print("\nmean calculation times:")
		print(" OpenCLDiscreteMI (correlated): ", np.mean(time_opencl_cor))
		print(" PythonDiscreteMI (correlated): ", np.mean(time_python_cor))
		print(" OpenCLDiscreteMI (uncorrelated): ", np.mean(time_opencl_uncor))
		print(" PythonDiscreteMI (uncorrelated): ", np.mean(time_python_uncor))

	print("\n=========================================================================")

	# test 1D bin data
	print(f"\n\n\nTesting average MI using 1D binary data with memory and discrete_method none")
	print(f"n_samples = {n_samples}\n")

	varx, vary = _get_mem_binary_data(n=n_samples, expand=True)
	settings = {'discretise_method': 'none'}
	est = OpenCLDiscreteMI(settings)
	itic = time.perf_counter()
	mi_opencl = est.estimate(varx, vary)
	itoc = time.perf_counter()
	print(f"OpenCLDiscreteMI: Estimated MI: {mi_opencl} - took: {itoc - itic}")
	est = PythonDiscreteMI(settings)
	itic = time.perf_counter()
	mi_python = est.estimate(varx, vary)
	itoc = time.perf_counter()
	print(f"PythonDiscreteMI: Estimated MI: {mi_python} - took: {itoc - itic}")
	verbose(mi_opencl, mi_python, "", "MI", atol=1e-03)

	print("\n=========================================================================")

	# test 2D
	lvals = [0, 1, 2, 3]

	print(f"\n\nTesting average MI using 2D mute data - correlated and uncorrelated")
	print(f"testing settings lag_mi {lvals}, n_discrete_bins 2 and discrete_method max_ent and equal")
	print(f"n_samples = {n_samples}\n")

	data = _generate_mute_data(n_samples=n_samples, n_replications=2)
	source1 = data[0, :, :]
	target = data[2, :, :]
	source2 = data[4, :, :]

	for m in ['max_ent', 'equal']:

		print(f"\n--- discrete_method: {m}\n")

		mi_opencl_cor = np.zeros(len(vals))
		mi_python_cor = np.zeros(len(vals))
		time_opencl_cor = np.zeros(len(vals))
		time_python_cor = np.zeros(len(vals))
		mi_opencl_uncor = np.zeros(len(vals))
		mi_python_uncor = np.zeros(len(vals))
		time_opencl_uncor = np.zeros(len(vals))
		time_python_uncor = np.zeros(len(vals))

		count = 0
		for l in lvals:
			settings = {'discretise_method': m,
						'n_discrete_bins': 2,
						'lag_mi': l}

			opencl_estimator = OpenCLDiscreteMI(settings=settings)
			itic = time.perf_counter()
			mi_opencl_cor[count] = opencl_estimator.estimate(source1, target)
			itoc = time.perf_counter()
			time_opencl_cor[count] = itoc - itic

			python_estimator = PythonDiscreteMI(settings=settings)
			itic = time.perf_counter()
			mi_python_cor[count] = python_estimator.estimate(source1, target)
			itoc = time.perf_counter()
			time_python_cor[count] = itoc - itic

			opencl_estimator = OpenCLDiscreteMI(settings=settings)
			itic = time.perf_counter()
			mi_opencl_uncor[count] = opencl_estimator.estimate(source2, target)
			itoc = time.perf_counter()
			time_opencl_uncor[count] = itoc - itic

			python_estimator = PythonDiscreteMI(settings=settings)
			itic = time.perf_counter()
			mi_python_uncor[count] = python_estimator.estimate(source2, target)
			itoc = time.perf_counter()
			time_python_uncor[count] = itoc - itic

			count += 1

		print(f"Summary OpenCL vs Python DiscreteMI discretised 2D mute data using {m}:")

		print("lags\tOpenCLDiscreteMI\t\tPythonDiscreteMI")
		print("correlated data:")
		for i in range(len(vals)):
			print(f"{lvals[i]}   \t{mi_opencl_cor[i]}\t{mi_python_cor[i]}")

		print("\nuncorrelated data:")
		for i in range(len(vals)):
			print(f"{lvals[i]}   \t{mi_opencl_uncor[i]}\t{mi_python_uncor[i]}")

		verbose(mi_opencl_cor, mi_python_cor, "correlated", "MI", local=False, atol=1e-03)
		verbose(mi_opencl_uncor, mi_python_uncor, "uncorrelated", "MI", local=False, atol=1e-03)

		print("\nmean calculation times:")
		print(" OpenCLDiscreteMI (correlated): ", np.mean(time_opencl_cor))
		print(" PythonDiscreteMI (correlated): ", np.mean(time_python_cor))
		print(" OpenCLDiscreteMI (uncorrelated): ", np.mean(time_opencl_uncor))
		print(" PythonDiscreteMI (uncorrelated): ", np.mean(time_python_uncor))

	print("\n=========================================================================")

	# test mixed dimension input
	d = [1, 2, 3, 5]

	print(f"\n\nTesting average MI using mixed dimensions\ntesting dimensions {d} for var1 and var2 each")
	print(f"n_samples = {n_samples}\n")

	print("Shapes:")

	data = _generate_mute_data(n_samples=n_samples, n_replications=5)
	source_o = data[0, :, :]
	target_o = data[2, :, :]

	settings = {'discretise_method': 'equal',
				'n_discrete_bins': 2,
				'lag_mi': 2}

	d = [1, 2, 3, 5]

	for s in d:
		for t in d:
			source1 = source_o[:, :s]
			target = target_o[:, :t]

			cond = f"var1: {source1.shape}\tvar2: {target.shape}"

			opencl_estimator = OpenCLDiscreteMI(settings=settings)
			python_estimator = PythonDiscreteMI(settings=settings)

			itic = time.perf_counter()
			mi_opencl_cor = opencl_estimator.estimate(source1, target)
			itoc = time.perf_counter()
			time_opencl_cor = itoc - itic

			itic = time.perf_counter()
			mi_python_cor = python_estimator.estimate(source1, target)
			itoc = time.perf_counter()
			time_python_cor = itoc - itic

			verbose(mi_opencl_cor, mi_python_cor, cond, "MI")

def test_discrete_mi_local_values():
	atol = 1e-03

	vals = [0, 1, 2, 3]
	print(f"\n\nTesting local MI using 1D gaussian data with covariance 0.4 - correlated and uncorrelated")
	print(f"testing settings lag_mi {vals}, n_discrete_bins 2 and discrete_method max_ent")
	print(f"n_samples = {n_samples}\n")

	expected_mi, source1, source2, target = _get_gauss_data(n=n_samples, expand=True, seed=SEED)

	vals = [0, 1, 2, 3]

	opencl_time = 0.0
	python_time = 0.0

	mi_opencl_cor = np.zeros(4)
	mi_opencl_uncor = np.zeros(4)
	mi_python_cor = np.zeros(4)
	mi_python_uncor = np.zeros(4)

	print("lags")
	count = 0
	for lags in vals:
		settings = {}
		settings = {'lag_mi': lags,
					'local_values': True,
					'discretise_method': 'max_ent',
					'n_discrete_bins': 2}

		opencl_estimator = OpenCLDiscreteMI(settings)
		itic = time.perf_counter()
		mi_opencl = opencl_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		opencl_time += itoc - itic

		python_estimator = PythonDiscreteMI(settings)
		itic = time.perf_counter()
		mi_python = python_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		python_time += itoc - itic

		verbose(mi_opencl, mi_python, f"{lags}", "MI (correlated)  ", local=True, atol=atol)

		opencl_estimator = OpenCLDiscreteMI(settings)
		itic = time.perf_counter()
		mi_opencl = opencl_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		opencl_time += itoc - itic

		python_estimator = PythonDiscreteMI(settings)
		itic = time.perf_counter()
		mi_python = python_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		python_time += itoc - itic

		verbose(mi_opencl, mi_python, f"{lags}", "MI (uncorrelated)", local=True, atol=atol)

	print("\nmean calculation times:")
	print(" OpenCLDiscreteMI: ", np.mean(opencl_time))
	print(" PythonDiscreteMI: ", np.mean(python_time))

	print("\n=========================================================================")

	# test 2D
	vals = [0, 1, 2, 3]

	print(f"\n\nTesting local MI using 2D mute data - correlated and uncorrelated")
	print(f"testing settings lag_mi {vals}, n_discrete_bins 2 and discrete_method max_ent")
	print(f"n_samples = {n_samples}\n")

	data = _generate_mute_data(n_samples=n_samples)

	source1 = data[0, :, :]
	target = data[2, :, :]
	source2 = data[4, :, :]

	time_opencl_cor = np.zeros(len(vals))
	time_opencl_uncor = np.zeros(len(vals))
	time_python_cor = np.zeros(len(vals))
	time_python_uncor = np.zeros(len(vals))

	print("lags")
	for lags in vals:
		settings = {}
		settings = {"lag_mi": lags,
					'local_values': True,
					'discretise_method': 'max_ent'}

		# cor
		opencl_estimator = OpenCLDiscreteMI(settings)
		itic = time.perf_counter()
		mi_opencl_cor = opencl_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_opencl_cor[lags] = itoc - itic

		python_estimator = PythonDiscreteMI(settings)
		itic = time.perf_counter()
		mi_python_cor = python_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_python_cor[lags] = itoc - itic

		verbose(mi_opencl_cor, mi_python_cor, lags, "MI (correlated)   2D input", local=True, atol=atol)

		# uncor
		opencl_estimator = OpenCLDiscreteMI(settings)
		itic = time.perf_counter()
		mi_opencl_uncor = opencl_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		time_opencl_uncor[lags] = itoc - itic

		python_estimator = PythonDiscreteMI(settings)
		itic = time.perf_counter()
		mi_python_uncor = python_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		time_python_uncor[lags] = itoc - itic

		verbose(mi_opencl_uncor, mi_python_uncor, lags, "MI (uncorrelated) 2D input", local=True, atol=atol)

	print("\nmean calculation times:")
	print(" OpenCLDiscrete (cor): ", np.mean(time_opencl_cor))
	print(" PythonDiscreteMI (cor): ", np.mean(time_python_cor))
	print(" OpenCLDiscreteMI (uncor): ", np.mean(time_opencl_uncor))
	print(" PythonDiscreteMI (uncor): ", np.mean(time_python_uncor))

def test_discrete_cmi():
	vals = [2, 5, 8]

	print(
		f"\n\nTesting average CMI using 1D gaussian data with covariance 0.4 - uncorrelated \nconditional and uncorrelated source")
	print(f"testing settings n_discrete_bins {vals} and discrete_method max_ent and equal")
	print(f"n_samples = {n_samples}\n")

	expected_mi, source1, source2, target = _get_gauss_data(n=n_samples, expand=True, seed=SEED)

	for m in ['max_ent', 'equal']:
		print(f"\n--- discrete_method: {m}\n")

		mi_opencl_cor = np.zeros(len(vals))
		mi_python_cor = np.zeros(len(vals))
		time_opencl_cor = np.zeros(len(vals))
		time_python_cor = np.zeros(len(vals))

		mi_opencl_uncor = np.zeros(len(vals))
		mi_python_uncor = np.zeros(len(vals))
		time_opencl_uncor = np.zeros(len(vals))
		time_python_uncor = np.zeros(len(vals))

		count = 0
		for i in vals:
			settings = {'discretise_method': m,
						'n_discrete_bins': i,
						'noise_level': 0,
						'normalise': False, }

			opencl_estimator = OpenCLDiscreteCMI(settings=settings)
			itic = time.perf_counter()
			mi_opencl_cor[count] = opencl_estimator.estimate(source1, target, source2)
			itoc = time.perf_counter()
			time_opencl_cor[count] = itoc - itic

			python_estimator = PythonDiscreteCMI(settings=settings)
			itic = time.perf_counter()
			mi_python_cor[count] = python_estimator.estimate(source1, target, source2)
			itoc = time.perf_counter()
			time_python_cor[count] = itoc - itic

			opencl_estimator = OpenCLDiscreteCMI(settings=settings)
			itic = time.perf_counter()
			mi_opencl_uncor[count] = opencl_estimator.estimate(source2, target, source1)
			itoc = time.perf_counter()
			time_opencl_uncor[count] = itoc - itic

			python_estimator = PythonDiscreteCMI(settings=settings)
			itic = time.perf_counter()
			mi_python_uncor[count] = python_estimator.estimate(source2, target, source1)
			itoc = time.perf_counter()
			time_python_uncor[count] = itoc - itic

			count += 1

		print(f"Summary OpenCL vs Python DiscreteCMI discretised 1D gaussian data using {m}:")

		print("nbins\tOpenCLDiscreteCMI\t\tPythonDiscreteCMI")
		print("uncorrelated conditional:")
		for i in range(len(vals)):
			print(f"{vals[i]}\t{mi_opencl_cor[i]}\t{mi_python_cor[i]}")

		print("\nuncorrelated source:")
		for i in range(len(vals)):
			print(f"{vals[i]}\t{mi_opencl_uncor[i]}\t{mi_python_uncor[i]}")

		verbose(mi_opencl_cor, mi_python_cor, "", "CMI (uncorrelated conditional)", local=False)
		verbose(mi_opencl_uncor, mi_python_uncor, "", "CMI (uncorrelated source)", local=False)

		print("\nmean calculation times:")
		print(" OpenCLDiscreteCMI(uncorrelated conditional): ", np.mean(time_opencl_cor))
		print(" PythonDiscreteCMI(uncorrelated conditional): ", np.mean(time_python_cor))
		print(" OpenCLDiscreteCMI (uncorrelated source): ", np.mean(time_opencl_uncor))
		print(" PythonDiscreteCMI (uncorrelated source): ", np.mean(time_python_uncor))

	print("\n=========================================================================")

	# test bin data
	print(f"\n\n\nTesting average CMI using 1D binary data with memory and discrete_method none")
	print(f"n_samples = {n_samples}\n")

	varx, vary = _get_mem_binary_data(n=n_samples, expand=True)
	varz, _ = _get_mem_binary_data(n=n_samples, expand=True)
	varx = varx[:10000]
	vary = vary[:10000]
	varz = varz[:10000]
	settings = {'discretise_method': 'none'}
	est = OpenCLDiscreteCMI(settings)
	itic = time.perf_counter()
	mi_opencl = est.estimate(varx, vary, varz)
	itoc = time.perf_counter()
	print(f"OpenCLDiscreteCMI: Estimated MI: {mi_opencl} - took: {itoc - itic}")
	est = PythonDiscreteCMI(settings)
	itic = time.perf_counter()
	mi_python = est.estimate(varx, vary, varz)
	itoc = time.perf_counter()
	print(f"PythonDiscreteCMI: Estimated MI: {mi_python} - took: {itoc - itic}")

	verbose(mi_opencl, mi_python, "", "CMI")

	print("\n=========================================================================")

	# test 2D
	print(f"\n\nTesting average CMI using 2D mute data - uncorrelated conditional and uncorrelated source")
	print(f"testing settings n_discrete_bins {vals} and discrete_method max_ent and equal")
	print(f"n_samples = {n_samples}\n")

	data = _generate_mute_data(n_samples=n_samples, n_replications=2)
	source1 = data[0, :, :]
	target = data[2, :, :]
	source2 = data[4, :, :]

	for m in ['max_ent', 'equal']:
		print(f"\n--- discrete_method: {m}\n")

		mi_opencl_cor = np.zeros(len(vals))
		mi_python_cor = np.zeros(len(vals))
		time_opencl_cor = np.zeros(len(vals))
		time_python_cor = np.zeros(len(vals))

		mi_opencl_uncor = np.zeros(len(vals))
		mi_python_uncor = np.zeros(len(vals))
		time_opencl_uncor = np.zeros(len(vals))
		time_python_uncor = np.zeros(len(vals))

		count = 0
		for i in vals:
			settings = {'discretise_method': m,
						'n_discrete_bins': i}

			opencl_estimator = OpenCLDiscreteCMI(settings=settings)
			itic = time.perf_counter()
			mi_opencl_cor[count] = opencl_estimator.estimate(source1, target, source2)
			itoc = time.perf_counter()
			time_opencl_cor[count] = itoc - itic

			python_estimator = PythonDiscreteCMI(settings=settings)
			itic = time.perf_counter()
			mi_python_cor[count] = python_estimator.estimate(source1, target, source2)
			itoc = time.perf_counter()
			time_python_cor[count] = itoc - itic

			opencl_estimator = OpenCLDiscreteCMI(settings=settings)
			itic = time.perf_counter()
			mi_opencl_uncor[count] = opencl_estimator.estimate(source2, target, source1)
			itoc = time.perf_counter()
			time_opencl_uncor[count] = itoc - itic

			python_estimator = PythonDiscreteCMI(settings=settings)
			itic = time.perf_counter()
			mi_python_uncor[count] = python_estimator.estimate(source2, target, source1)
			itoc = time.perf_counter()
			time_python_uncor[count] = itoc - itic

			count += 1

		print(f"Summary OpenCL vs Python DiscreteCMI discretised 1D gaussian data using {m}:")

		print("nbins\tOpenCLDiscreteCMI\t\tPythonDiscreteCMI")
		print("CMI values uncorrelated conditional:")
		for i in range(len(vals)):
			print(f"{vals[i]}\t{mi_opencl_cor[i]}\t{mi_python_cor[i]}")

		print("\nMI values uncorrelated source:")
		# print("nbins\tOpenCLDiscreteCMI\t\tPythonDiscreteCMI")
		for i in range(len(vals)):
			print(f"{vals[i]}\t{mi_opencl_uncor[i]}\t{mi_python_uncor[i]}")

		verbose(mi_opencl_cor, mi_python_cor, "", "CMI (uncorrelated conditional)", local=False)
		verbose(mi_opencl_uncor, mi_python_uncor, "", "CMI (uncorrelated source)", local=False)

		print("\nmean calculation times:")
		print(" OpenCLDiscreteCMI(uncorrelated conditional): ", np.mean(time_opencl_cor))
		print(" PythonDiscreteCMI(uncorrelated conditional): ", np.mean(time_python_cor))
		print(" OpenCLDiscreteCMI (uncorrelated source): ", np.mean(time_opencl_uncor))
		print(" PythonDiscreteCMI (uncorrelated source): ", np.mean(time_python_uncor))

	print("\n=========================================================================")

	# test mixed dimension input
	d = [1, 2, 3]

	print(f"\n\nTesting average MI using mixed dimensions\ntesting dimensions {d} for var1, var2 and cond each")
	print(f"n_samples = {n_samples}\n")

	print("Shapes:")

	data = _generate_mute_data(n_samples=n_samples, )

	source_o = data[0, :, :]
	target_o = data[2, :, :]
	cond_o = data[4, :, :]

	settings = {'discretise_method': 'max_ent',
				'n_discrete_bins': 2}

	for s in d:
		for t in d:
			for c in d:
				source1 = source_o[:, :s]
				target = target_o[:, :t]
				conditional = cond_o[:, :c]

				cond = f"var1: {source1.shape} var2: {target.shape} cond: {conditional.shape}"

				opencl_estimator = OpenCLDiscreteCMI(settings)
				python_estimator = PythonDiscreteCMI(settings)

				itic = time.perf_counter()
				mi_opencl_cor = opencl_estimator.estimate(source1, target, conditional)
				itoc = time.perf_counter()
				time_opencl_cor = itoc - itic

				itic = time.perf_counter()
				mi_python_cor = python_estimator.estimate(source1, target, conditional)
				itoc = time.perf_counter()
				time_python_cor = itoc - itic

				verbose(mi_opencl_cor, mi_python_cor, cond, "MI")

def test_discrete_cmi_local_values():
	vals = [2, 4, 10]

	print(
		f"\n\nTesting local CMI using 1D gaussian data with covariance 0.4 - uncorrelated \nconditional and uncorrelated source")
	print(f"testing settings n_discrete_bins {vals} and discrete_method max_ent and equal")
	print(f"n_samples = {n_samples}\n")

	expected_mi, source1, source2, target = _get_gauss_data(n=n_samples, expand=True, seed=SEED)

	time_opencl_cor = 0.0
	time_opencl_uncor = 0.0
	time_python_cor = 0.0
	time_python_uncor = 0.0

	print("bins")
	for i in vals:
		settings = {}
		settings = {'local_values': True,
					'discretise_method': 'max_ent',
					'n_discrete_bins': 2}

		opencl_estimator = OpenCLDiscreteCMI(settings)
		itic = time.perf_counter()
		mi_opencl = opencl_estimator.estimate(source1, target, source2)
		itoc = time.perf_counter()
		time_opencl_cor += itoc - itic

		python_estimator = PythonDiscreteCMI(settings)
		itic = time.perf_counter()
		mi_python = python_estimator.estimate(source1, target, source2)
		itoc = time.perf_counter()
		time_python_cor += itoc - itic

		verbose(mi_opencl, mi_python, i, "CMI (uncorrelated conditional)", local=True, atol=1e-03)

		opencl_estimator = OpenCLDiscreteCMI(settings)
		itic = time.perf_counter()
		mi_opencl = opencl_estimator.estimate(source2, target, source1)
		itoc = time.perf_counter()
		time_opencl_uncor += itoc - itic

		python_estimator = PythonDiscreteCMI(settings)
		itic = time.perf_counter()
		mi_python = python_estimator.estimate(source2, target, source1)
		itoc = time.perf_counter()
		time_python_uncor += itoc - itic

		verbose(mi_opencl, mi_python, i, "CMI (uncorrelated source)     ", local=True, atol=1e-03)

	print("\nmean calculation times:")
	print(" OpenCLDiscreteCMI: ", np.mean(time_opencl_cor))
	print(" PythonDiscreteCMI: ", np.mean(time_python_cor))

	print("\n=========================================================================")

	# test 2D data
	print(f"\n\nTesting local CMI using 2D mute data  - uncorrelated \nconditional and uncorrelated source")
	print(f"testing settings n_discrete_bins {vals} and discrete_method max_ent and equal")
	print(f"n_samples = {n_samples}\n")

	print("\nTest n_discrete_bins using 2D data input:")
	data = _generate_mute_data(n_samples=n_samples, n_replications=2)
	source1 = data[0, :, :]
	target = data[1, :, :]
	source2 = data[4, :, :]

	time_opencl_cor = 0.0
	time_opencl_uncor = 0.0
	time_python_cor = 0.0
	time_python_uncor = 0.0

	print("bins")
	for i in vals:
		settings = {}
		settings = {'local_values': True,
					'discretise_method': 'max_ent',
					'n_discrete_bins': 2, }

		opencl_estimator = OpenCLDiscreteCMI(settings)
		itic = time.perf_counter()
		mi_opencl = opencl_estimator.estimate(source1, target, source2)
		itoc = time.perf_counter()
		time_opencl_cor += itoc - itic

		python_estimator = PythonDiscreteCMI(settings)
		itic = time.perf_counter()
		mi_python = python_estimator.estimate(source1, target, source2)
		itoc = time.perf_counter()
		time_python_cor += itoc - itic

		verbose(mi_opencl, mi_python, i, "CMI (uncorrelated conditional)", local=True)

		opencl_estimator = OpenCLDiscreteCMI(settings)
		itic = time.perf_counter()
		mi_opencl = opencl_estimator.estimate(source2, target, source1)
		itoc = time.perf_counter()
		time_opencl_uncor += itoc - itic

		python_estimator = PythonDiscreteCMI(settings)
		itic = time.perf_counter()
		mi_python = python_estimator.estimate(source2, target, source1)
		itoc = time.perf_counter()
		time_python_uncor += itoc - itic

		verbose(mi_opencl, mi_python, i, "CMI (uncorrelated source)     ", local=True)

	print("\nmean calculation times:")
	print(" OpenCLDiscreteCMI: ", np.mean(time_opencl_cor))
	print(" PythonDiscreteCMI: ", np.mean(time_python_cor))

def test_discrete_ais():
	atol = 1e-03
	hvals = [1, 2, 3]
	nvals = [2, 4, 8]

	print(f"\n\nTesting average AIS using 1D AR with history and noise")
	print(f"testing settings history {hvals} and n_discrete_bins {nvals} and discrete_method max_ent")
	print(f"n_samples = {n_samples}\n")

	source1, source2 = _get_ar_data(n=n_samples, seed=SEED)

	time_opencl_cor = np.zeros(np.power(len(nvals), 2))
	res_opencl_cor = np.zeros(np.power(len(nvals), 2))
	time_python_cor = np.zeros(np.power(len(nvals), 2))
	res_python_cor = np.zeros(np.power(len(nvals), 2))
	time_opencl_uncor = np.zeros(np.power(len(nvals), 2))
	res_opencl_uncor = np.zeros(np.power(len(nvals), 2))
	time_python_uncor = np.zeros(np.power(len(nvals), 2))
	res_python_uncor = np.zeros(np.power(len(nvals), 2))
	conds = np.empty((np.power(len(nvals), 3), 2))

	count = 0
	for h in hvals:
		for i in nvals:
			conds[count, :] = [h, i]

			settings_j = {'history': h,
						  'discretise_method': 'max_ent',
						  'n_discrete_bins': i}

			settings_p = {'history': h,
						  'discretise_method': 'max_ent',
						  'n_discrete_bins': i}

			opencl_estimator = OpenCLDiscreteAIS(settings=settings_j)
			python_estimator = PythonDiscreteAIS(settings=settings_p)

			itic = time.perf_counter()
			res_opencl_cor[count] = opencl_estimator.estimate(source1)
			itoc = time.perf_counter()
			time_opencl_cor[count] = itoc - itic

			itic = time.perf_counter()
			res_opencl_uncor[count] = opencl_estimator.estimate(source2)
			itoc = time.perf_counter()
			time_opencl_uncor[count] = itoc - itic

			itic = time.perf_counter()
			res_python_cor[count] = python_estimator.estimate(source1)
			itoc = time.perf_counter()
			time_python_cor[count] = itoc - itic

			itic = time.perf_counter()
			res_python_uncor[count] = python_estimator.estimate(source2)
			itoc = time.perf_counter()
			time_python_uncor[count] = itoc - itic

			count += 1

	print(f"Summary OpenCL vs Python DiscreteAIS discretised 1D gaussian data using max_ent:")

	print(f"hist, bins\tOpenCLDiscreteAIS\t\tPythonDiscreteAIS\tclose {atol}")
	print("with history")
	count = 0
	for i in range(len(res_opencl_cor)):
		print(
			f"{conds[i, :]}\t\t{res_opencl_cor[i]}\t{res_python_cor[i]}\t{np.isclose(res_opencl_cor[i], res_python_cor[i], atol=atol)}")
		count += 1

	print("noise")
	count = 0
	for i in range(len(res_opencl_uncor)):
		print(
			f"{conds[i, :]}\t\t{res_opencl_uncor[i]}\t{res_python_uncor[i]}\t{np.isclose(res_opencl_uncor[i], res_python_uncor[i], atol=atol)}")
		count += 1

	verbose(res_opencl_cor, res_python_cor, "with history", "AIS", atol=atol)
	verbose(res_opencl_uncor, res_python_uncor, "noise", "AIS", atol=atol)

	print("\nmean calculation times:")
	print(" OpenCLDiscreteAIS (with history): ", np.mean(time_opencl_cor))
	print(" PythonDiscreteAIS (with history): ", np.mean(time_python_cor))
	print(" OpenCLDiscreteAIS (noise): ", np.mean(time_opencl_uncor))
	print(" PythonDiscreteAIS (noise): ", np.mean(time_python_uncor))

def test_discrete_ais_local_values():
	atol = 1e-03

	hvals = [1, 2, 3]
	nvals = [2, 4, 6]

	print(f"\n\nTesting local AIS using 1D AR with history and noise")
	print(f"testing settings history {hvals} and n_discrete_bins {nvals} and discrete_method max_ent")
	print(f"n_samples = {n_samples}\n")

	source1, source2 = _get_ar_data(n=n_samples, seed=SEED + 1)

	min_len = min(len(source1), len(source2))
	source1 = source1[:min_len]
	source2 = source2[:min_len]

	time_opencl_cor = np.zeros(np.power(len(nvals), 2))
	res_opencl_cor = np.zeros(np.power(len(nvals), 2))
	time_python_cor = np.zeros(np.power(len(nvals), 2))
	res_python_cor = np.zeros(np.power(len(nvals), 2))
	time_opencl_uncor = np.zeros(np.power(len(nvals), 2))
	res_opencl_uncor = np.zeros(np.power(len(nvals), 2))
	time_python_uncor = np.zeros(np.power(len(nvals), 2))
	res_python_uncor = np.zeros(np.power(len(nvals), 2))
	conds = np.empty((np.power(len(nvals), 3), 2))

	print("hist, bins\tOpenCLDiscreteAIS vs PythonDiscreteAIS")
	count = 0
	for h in hvals:
		for i in nvals:
			conds[count, :] = [h, i]
			settings = {}
			settings_j = {'history': h,
						  'discretise_method': 'max_ent',
						  'n_discrete_bins': i,
						  'local_values': True}
			settings_p = {'history': h,
						  'discretise_method': 'max_ent',
						  'n_discrete_bins': i,
						  'local_values': True}

			opencl_estimator = OpenCLDiscreteAIS(settings=settings_j)
			python_estimator = PythonDiscreteAIS(settings=settings_p)

			itic = time.perf_counter()
			res_opencl_cor = opencl_estimator.estimate(source1)
			itoc = time.perf_counter()
			time_opencl_cor[count] = itoc - itic

			itic = time.perf_counter()
			res_opencl_uncor = opencl_estimator.estimate(source2)
			itoc = time.perf_counter()
			time_opencl_uncor[count] = itoc - itic

			itic = time.perf_counter()
			res_python_cor = python_estimator.estimate(source1)
			itoc = time.perf_counter()
			time_python_cor[count] = itoc - itic

			itic = time.perf_counter()
			res_python_uncor = python_estimator.estimate(source2)
			itoc = time.perf_counter()
			time_python_uncor[count] = itoc - itic

			# print(res_opencl_cor[:20])
			# print(res_python_cor[:20])

			min_len = min(len(res_opencl_cor), len(res_python_cor))

			verbose(res_opencl_cor, res_python_cor, f"{conds[count, :]} - with hist", "AIS", local=True, atol=atol)
			verbose(res_opencl_uncor, res_python_uncor, f"{conds[count, :]} - noise    ", "AIS", local=True, atol=atol)

			count += 1

	print("\nmean calculation times:")
	print(" OpenCLDiscreteAIS (with history): ", np.mean(time_opencl_cor))
	print(" PythonDiscreteAIS (with history): ", np.mean(time_python_cor))
	print(" OpenCLDiscreteAIS (noise): ", np.mean(time_opencl_uncor))
	print(" PythonDiscreteAIS (noise): ", np.mean(time_python_uncor))

def test_discrete_te():
	vals = [1, 3]
	nvals = [2, 6]

	print(f"\n\nTesting average TE using 1D gaussian data with covariance 0.4 and lag 1\n")
	print(f"testing settings history_source (hs), tau_source (ts), history_target (ht), \ntau_target (tt), source_target_delay (std) with {vals} each.\nand n_discrete_bins{nvals}")
	print(f"n_samples = {n_samples}\n")

	expected_mi, source1, source2, target = _get_gauss_data(n=n_samples, expand=False, seed=SEED)
	# add delay of one sample
	source1 = source1[1:]
	source2 = source2[1:]
	target = target[:-1]

	time_opencl_cor = np.empty(np.power(len(vals), 6))
	res_opencl_cor = np.empty(np.power(len(vals), 6))
	time_python_cor = np.empty(np.power(len(vals), 6))
	res_python_cor = np.empty(np.power(len(vals), 6))
	time_opencl_uncor = np.empty(np.power(len(vals), 6))
	res_opencl_uncor = np.empty(np.power(len(vals), 6))
	time_python_uncor = np.empty(np.power(len(vals), 6))
	res_python_uncor = np.empty(np.power(len(vals), 6))

	conds = np.empty([np.power(len(vals), 6), 6])

	print("hst,ht,tt,hs,ts\t\tOpenCLDiscreteTE\tPythonDiscreteTE\tclose 1e-03")

	count = 0
	for hst in vals:
		for ht in vals:
			for hs in vals:
				for tt in vals:
					for ts in vals:
						for n in nvals:
							conds[count, :] = [hst, ht, tt, hs, ts, n]
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

							opencl_estimator = OpenCLDiscreteTE(settings_j)
							python_estimator = PythonDiscreteTE(settings_p)

							itic = time.perf_counter()
							te_opencl_cor = opencl_estimator.estimate(source=source1, target=target)
							itoc = time.perf_counter()
							# time_opencl += itoc-itic
							time_opencl_cor[count] = itoc - itic
							res_opencl_cor[count] = te_opencl_cor

							itic = time.perf_counter()
							te_python_cor = python_estimator.estimate(source=source1, target=target)
							itoc = time.perf_counter()
							# time_python += itoc-itic
							time_python_cor[count] = itoc - itic
							res_python_cor[count] = te_python_cor

							# verbose(te_opencl, te_python, f"{[hst, ht, tt, hs, ts, n]}", "TE")
							print(
								f"{[hst, ht, tt, hs, ts]}\t\t{te_opencl_cor}\t{te_python_cor}\t{np.isclose(te_opencl_cor, te_python_cor, rtol=1e-03, atol=1e-03)}")

							count += 1

	verbose(res_opencl_cor, res_python_cor, "", "TE", local=False)

	print("\nmean calculation times:")
	print(" OpenCLDiscreteTE: ", np.mean(time_opencl_cor))
	print(" PythonDiscreteTE: ", np.mean(time_python_cor))

def test_discrete_te_local_values():
	vals = [2, 4]

	print(f"\n\nTesting average TE using 1D binary data with memory\n")
	print(f"testing settings history_source (hs), tau_source (ts), history_target (ht), \ntau_target (tt), source_target_delay (std) with {vals} each,\nand n_discrete_bins 2")
	print(f"n_samples = {n_samples}\n")

	source1, target = _get_mem_binary_data(n=n_samples, expand=True)

	time_opencl = np.empty(np.power(len(vals), 5))
	res_opencl = np.empty(np.power(len(vals), 5))
	time_python = np.empty(np.power(len(vals), 5))
	res_python = np.empty(np.power(len(vals), 5))
	conds = np.empty((np.power(len(vals), 5), 5))

	print("hst,ht,tt,hs,ts\t\tOpenCLDiscreteTE vs PythonDiscreteTE")

	count = 0
	for hst in vals:
		for ht in vals:
			for tt in vals:
				for hs in vals:
					for ts in vals:
						conds[count, :] = [hst, ht, tt, hs, ts]
						settings_j = {"history_target": ht,
									  "history_source": hs,
									  "tau_target": tt,
									  "tau_source": ts,
									  "source_target_delay": hst,
									  "local_values": True,
									  'noise_level': 0,
									  'n_discrete_bins': 2}

						opencl_estimator = OpenCLDiscreteTE(settings_j)

						itic = time.perf_counter()
						te_opencl = opencl_estimator.estimate(source=source1, target=target)
						itoc = time.perf_counter()

						time_opencl[count] = itoc - itic
						res_opencl[count] = np.mean(te_opencl)

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

						time_python[count] = itoc - itic
						res_python[count] = np.mean(te_python)

						count += 1

						verbose(te_opencl, te_python, f"{[hst, ht, tt, hs, ts]}\t", "local TE", local=True, atol=1e-03)

	print("\nmean calculation times:")
	print(" OpenCLDiscreteTE: ", np.mean(time_opencl))
	print(" PythonDiscreteTE: ", np.mean(time_python))


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

    EoP_opencl = np.zeros(len(pvals))
    EoP_python = np.zeros(len(pvals))
    
    settings_opencl = {'noise_level': 0, 
        'normalise': False,}

    settings_python = {'noise_level': 0, 
        'normalise': False,}

    est_opencl = OpenCLGaussianMI(settings_opencl)
    est_python = PythonGaussianMI(settings_python)

    mi = est_opencl.estimate(source, target)
    #C_opencl = est_opencl.calc.computeSignificance()
    C_opencl = est_opencl.get_analytic_distribution(source, target)

    mi2 = est_python.estimate(source, target)
    C_python = est_python.get_analytic_distribution(source, target)

    mean_opencl = C_opencl.getMeanOfDistribution()
    mean_python = C_python.getMeanOfDistribution()
    
    mean_uncorr_opencl = C_opencl.getMeanOfUncorrectedDistribution()
    mean_uncorr_python = C_python.getMeanOfUncorrectedDistribution()
    
    std_opencl = C_opencl.getStdOfDistribution()
    std_python = C_python.getStdOfDistribution()
    
    count = 0
    for p in pvals:
        EoP_opencl[count] = C_opencl.computeEstimateForGivenPValue(p)
        EoP_python[count] = C_python.computeEstimateForGivenPValue(p)
        count += 1

    print(f"OpenCL computeSignificance object:\ntype: {type(C_opencl)}")
    print(f"Python computeSignificance object:\ntype: {type(C_python)}")
    
    atol = 1e-04
    print(f"\nSummary OpenCL vs Python GaussianMI 1D gaussian data using {m}:\n")

    print(f"\t\t\tOpenCL\t\t\tPython\t\t\tclose {atol}")
    print(f"actualValue:\n\t\t\t{C_opencl.actualValue}\t{C_python.actualValue}\t{np.isclose(C_opencl.actualValue, C_python.actualValue, atol=atol)}")
    print(f"pValue:\n\t\t\t{C_opencl.pValue}\t{C_python.pValue}\t{np.isclose(C_opencl.pValue, C_python.pValue, atol=atol)}")
    print(f"getMeanOfDistribution:\n\t\t\t{mean_opencl}\t{mean_python}\t{np.isclose(mean_opencl, mean_python, atol=atol)}")
    print(f"getMeanOfUncorrectedDistribution:\n\t\t\t{mean_uncorr_opencl}\t{mean_uncorr_python}\t{np.isclose(mean_uncorr_opencl, mean_uncorr_python, atol=atol)}")
    print(f"StdOfDistribution:\n\t\t\t{std_opencl}\t{std_python}\t{np.isclose(std_opencl, std_python, atol=atol)}")

    print("\nEstimateForGivenPValue:")
    print("p\t\tOpenCLGaussianMI\t\tPythonGaussianMI")
    for i in range(len(pvals)):
        print(f"{pvals[i]}   \t{EoP_opencl[i]}\t{EoP_python[i]}")
    verbose(EoP_opencl, EoP_python, "", "Estimate for given PValue")

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

    EoP_opencl = np.zeros(len(pvals))
    EoP_python = np.zeros(len(pvals))
    
    settings_opencl = {'noise_level': 0, 
        'normalise': False,}

    settings_python = {'noise_level': 0, 
        'normalise': False,}

    est_opencl = OpenCLGaussianCMI(settings_opencl)
    est_python = PythonGaussianCMI(settings_python)

    mi = est_opencl.estimate(source, target, source_uncorr)
    C_opencl = est_opencl.get_analytic_distribution(source, target, source_uncorr)

    mi2 = est_python.estimate(source, target, source_uncorr)
    C_python = est_python.get_analytic_distribution(source, target, source_uncorr)

    mean_opencl = C_opencl.getMeanOfDistribution()
    mean_python = C_python.getMeanOfDistribution()
    
    mean_uncorr_opencl = C_opencl.getMeanOfUncorrectedDistribution()
    mean_uncorr_python = C_python.getMeanOfUncorrectedDistribution()
    
    std_opencl = C_opencl.getStdOfDistribution()
    std_python = C_python.getStdOfDistribution()
    
    count = 0
    for p in pvals:
        EoP_opencl[count] = C_opencl.computeEstimateForGivenPValue(p)
        EoP_python[count] = C_python.computeEstimateForGivenPValue(p)
        count += 1

    print(f"OpenCL computeSignificance object:\ntype: {type(C_opencl)}")
    print(f"Python computeSignificance object:\ntype: {type(C_python)}")
    
    atol = 1e-04
    print(f"\nSummary OpenCL vs Python GaussianCMI 1D gaussian data using {m}:\n")

    print(f"\t\t\tOpenCL\t\t\tPython\t\t\tclose {atol}")
    print(f"actualValue:\n\t\t\t{C_opencl.actualValue}\t{C_python.actualValue}\t{np.isclose(C_opencl.actualValue, C_python.actualValue, atol=atol)}")
    print(f"pValue:\n\t\t\t{C_opencl.pValue}\t{C_python.pValue}\t{np.isclose(C_opencl.pValue, C_python.pValue, atol=atol)}")
    print(f"getMeanOfDistribution:\n\t\t\t{mean_opencl}\t{mean_python}\t{np.isclose(mean_opencl, mean_python, atol=atol)}")
    print(f"getMeanOfUncorrectedDistribution:\n\t\t\t{mean_uncorr_opencl}\t{mean_uncorr_python}\t{np.isclose(mean_uncorr_opencl, mean_uncorr_python, atol=atol)}")
    print(f"StdOfDistribution:\n\t\t\t{std_opencl}\t{std_python}\t{np.isclose(std_opencl, std_python, atol=atol)}")

    print("\nEstimateForGivenPValue:")
    print("p\tOpenCLGaussianCMI\t\tPythonGaussianCMI")
    for i in range(len(pvals)):
        print(f"{pvals[i]}   \t{EoP_opencl[i]}\t{EoP_python[i]}")
    verbose(EoP_opencl, EoP_python, "", "Estimate for given PValue")

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

    EoP_opencl = np.zeros(len(pvals))
    EoP_python = np.zeros(len(pvals))
    
    settings_opencl = {'noise_level': 0, 
        'normalise': False,}

    settings_python = {'noise_level': 0, 
        'normalise': False,}

    est_opencl = OpenCLGaussianCMI(settings_opencl)
    est_python = PythonGaussianCMI(settings_python)

    mi = est_opencl.estimate(source, target)
    C_opencl = est_opencl.get_analytic_distribution(source, target, source_uncorr)

    mi2 = est_python.estimate(source, target)
    C_python = est_python.get_analytic_distribution(source, target, source_uncorr)

    mean_opencl = C_opencl.getMeanOfDistribution()
    mean_python = C_python.getMeanOfDistribution()
    
    mean_uncorr_opencl = C_opencl.getMeanOfUncorrectedDistribution()
    mean_uncorr_python = C_python.getMeanOfUncorrectedDistribution()
    
    std_opencl = C_opencl.getStdOfDistribution()
    std_python = C_python.getStdOfDistribution()
    
    count = 0
    for p in pvals:
        EoP_opencl[count] = C_opencl.computeEstimateForGivenPValue(p)
        EoP_python[count] = C_python.computeEstimateForGivenPValue(p)
        count += 1

    print(f"OpenCL computeSignificance object:\ntype: {type(C_opencl)}")
    print(f"Python computeSignificance object:\ntype: {type(C_python)}")
    
    atol = 1e-04
    print(f"\nSummary OpenCL vs Python GaussianCMI (no conditional) 1D gaussian data using {m}:\n")

    print(f"\t\t\tOpenCL\t\t\tPython\t\t\tclose {atol}")
    print(f"actualValue:\n\t\t\t{C_opencl.actualValue}\t{C_python.actualValue}\t{np.isclose(C_opencl.actualValue, C_python.actualValue, atol=atol)}")
    print(f"pValue:\n\t\t\t{C_opencl.pValue}\t{C_python.pValue}\t{np.isclose(C_opencl.pValue, C_python.pValue, atol=atol)}")
    print(f"getMeanOfDistribution:\n\t\t\t{mean_opencl}\t{mean_python}\t{np.isclose(mean_opencl, mean_python, atol=atol)}")
    print(f"getMeanOfUncorrectedDistribution:\n\t\t\t{mean_uncorr_opencl}\t{mean_uncorr_python}\t{np.isclose(mean_uncorr_opencl, mean_uncorr_python, atol=atol)}")
    print(f"StdOfDistribution:\n\t\t\t{std_opencl}\t{std_python}\t{np.isclose(std_opencl, std_python, atol=atol)}")

    print("\nEstimateForGivenPValue:")
    print("p\tOpenCLGaussianCMI\t\tPythonGaussianCMI")
    for i in range(len(pvals)):
        print(f"{pvals[i]}   \t{EoP_opencl[i]}\t{EoP_python[i]}")
    verbose(EoP_opencl, EoP_python, "", "Estimate for given PValue")

    print("\n=========================================================================")

def test_analytic_distribution_ais_gaussian():

    pvals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    m = 'equal'
    bins = 5

    print(f"\n\nTesting Gaussian AIS using 1D AR with history \n using discretise_method {m} - {bins} bins\n")
    print(f"testing computeEstimateForGivenPValue for:\n\t{pvals}")

    source1, source2 = _get_ar_data(seed=SEED)

    EoP_opencl = np.zeros(len(pvals))
    EoP_python = np.zeros(len(pvals))
    
    settings_opencl = {'history': 2,
    	"discretise_method": m,
        "n_discrete_bins": bins, 
        'noise_level': 0, 
        'normalise': False,}

    settings_python = {'history': 2,
    	"discretise_method": m,
        "n_discrete_bins": bins, 
        'noise_level': 0, 
        'normalise': False,}

    est_opencl = OpenCLGaussianAIS(settings_opencl)
    est_python = PythonGaussianAIS(settings_python)

    mi = est_opencl.estimate(source1)
    #C_opencl = est_opencl.calc.computeSignificance()
    C_opencl = est_opencl.get_analytic_distribution(source1) ######### ATTENTION get_analytic.. not working properly

    mi2 = est_python.estimate(source1)
    #C_python = est_python.computeSignificance()
    C_python = est_python.get_analytic_distribution(source1)

    mean_opencl = C_opencl.getMeanOfDistribution()
    mean_python = C_python.getMeanOfDistribution()
    
    mean_uncorr_opencl = C_opencl.getMeanOfUncorrectedDistribution()
    mean_uncorr_python = C_python.getMeanOfUncorrectedDistribution()
    
    std_opencl = C_opencl.getStdOfDistribution()
    std_python = C_python.getStdOfDistribution()
    
    count = 0
    for p in pvals:
        EoP_opencl[count] = C_opencl.computeEstimateForGivenPValue(p)
        EoP_python[count] = C_python.computeEstimateForGivenPValue(p)
        count += 1

    print(f"OpenCL computeSignificance object:\ntype: {type(C_opencl)}")
    print(f"Python computeSignificance object:\ntype: {type(C_python)}")

    atol = 1e-04
    print(f"\nSummary OpenCL vs Python GaussianAIS on AR data with history using {m}:\n")
    print(f"\t\t\tOpenCL\t\t\tPython\t\t\tclose {atol}")
    print(f"actualValue:\n\t\t\t{C_opencl.actualValue}\t{C_python.actualValue}\t{np.isclose(C_opencl.actualValue, C_python.actualValue, atol=atol)}")
    print(f"pValue:\n\t\t\t{C_opencl.pValue}\t{C_python.pValue}\t{np.isclose(C_opencl.pValue, C_python.pValue, atol=atol)}")
    print(f"getMeanOfDistribution:\n\t\t\t{mean_opencl}\t{mean_python}\t{np.isclose(mean_opencl, mean_python, atol=atol)}")
    print(f"getMeanOfUncorrectedDistribution:\n\t\t\t{mean_uncorr_opencl}\t{mean_uncorr_python}\t{np.isclose(mean_uncorr_opencl, mean_uncorr_python, atol=atol)}")
    print(f"StdOfDistribution:\n\t\t\t{std_opencl}\t{std_python}\t{np.isclose(std_opencl, std_python, atol=atol)}")

    print("\nEstimateForGivenPValue:")
    print("p\tOpenCLGaussianAIS\t\tPythonGaussianAIS")
    for i in range(len(pvals)):
        print(f"{pvals[i]}   \t{EoP_opencl[i]}\t{EoP_python[i]}")
    verbose(EoP_opencl, EoP_python, "", "Estimate for given PValue")

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

    EoP_opencl = np.zeros(len(pvals))
    EoP_python = np.zeros(len(pvals))

    settings_opencl = {'history_target': 1,
    	'noise_level': 0, 
        'normalise': False,}

    settings_python = {'history_target': 1,
    	'noise_level': 0, 
        'normalise': False,}

    est_opencl = OpenCLGaussianTE(settings_opencl)
    est_python = PythonGaussianTE(settings_python)

    mi = est_opencl.estimate(source, target)
    C_opencl = est_opencl.get_analytic_distribution(source, target)
    #C_opencl = est_opencl.calc.computeSignificance()

    mi2 = est_python.estimate(source, target)
    C_python = est_python.get_analytic_distribution(source, target)

    mean_opencl = C_opencl.getMeanOfDistribution()
    mean_python = C_python.getMeanOfDistribution()
    
    mean_uncorr_opencl = C_opencl.getMeanOfUncorrectedDistribution()
    mean_uncorr_python = C_python.getMeanOfUncorrectedDistribution()
    
    std_opencl = C_opencl.getStdOfDistribution()
    std_python = C_python.getStdOfDistribution()
    
    count = 0
    for p in pvals:
        EoP_opencl[count] = C_opencl.computeEstimateForGivenPValue(p)
        EoP_python[count] = C_python.computeEstimateForGivenPValue(p)
        count += 1

    print(f"OpenCL computeSignificance object:\ntype: {type(C_opencl)}")
    print(f"Python computeSignificance object:\ntype: {type(C_python)}")
    
    atol = 1e-04
    print(f"\nSummary OpenCL vs Python GaussianTE 1D gaussian data using {m}:\n")

    print(f"\t\t\tOpenCL\t\t\tPython\t\t\tclose {atol}")
    print(f"actualValue:\n\t\t\t{C_opencl.actualValue}\t{C_python.actualValue}\t{np.isclose(C_opencl.actualValue, C_python.actualValue, atol=atol)}")
    print(f"pValue:\n\t\t\t{C_opencl.pValue}\t{C_python.pValue}\t{np.isclose(C_opencl.pValue, C_python.pValue, atol=atol)}")
    print(f"getMeanOfDistribution:\n\t\t\t{mean_opencl}\t{mean_python}\t{np.isclose(mean_opencl, mean_python, atol=atol)}")
    print(f"getMeanOfUncorrectedDistribution:\n\t\t\t{mean_uncorr_opencl}\t{mean_uncorr_python}\t{np.isclose(mean_uncorr_opencl, mean_uncorr_python, atol=atol)}")
    print(f"StdOfDistribution:\n\t\t\t{std_opencl}\t{std_python}\t{np.isclose(std_opencl, std_python, atol=atol)}")

    print("\nEstimateForGivenPValue:")
    print("p\tOpenCLGaussianTE\t\tPythonGaussianTE")
    for i in range(len(pvals)):
        print(f"{pvals[i]}   \t{EoP_opencl[i]}\t{EoP_python[i]}")
    verbose(EoP_opencl, EoP_python, "", "Estimate for given PValue")

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

    EoP_opencl = np.zeros(len(pvals))
    EoP_python = np.zeros(len(pvals))
    
    settings_opencl = {'history_target': 1,
    	'noise_level': 0, 
        'normalise': False,}

    settings_python = {'history_target': 1,
    	'noise_level': 0, 
        'normalise': False,}

    est_opencl = OpenCLGaussianCTE(settings_opencl)
    est_python = PythonGaussianCTE(settings_python)

    mi = est_opencl.estimate(source, target, source_uncorr)
    C_opencl = est_opencl.get_analytic_distribution(source, target, source_uncorr)

    mi2 = est_python.estimate(source, target, source_uncorr)
    C_python = est_python.get_analytic_distribution(source, target, source_uncorr)

    mean_opencl = C_opencl.getMeanOfDistribution()
    mean_python = C_python.getMeanOfDistribution()
    
    mean_uncorr_opencl = C_opencl.getMeanOfUncorrectedDistribution()
    mean_uncorr_python = C_python.getMeanOfUncorrectedDistribution()
    
    std_opencl = C_opencl.getStdOfDistribution()
    std_python = C_python.getStdOfDistribution()
    
    count = 0
    for p in pvals:
        EoP_opencl[count] = C_opencl.computeEstimateForGivenPValue(p)
        EoP_python[count] = C_python.computeEstimateForGivenPValue(p)
        count += 1

    print(f"OpenCL computeSignificance object:\ntype: {type(C_opencl)}")
    print(f"Python computeSignificance object:\ntype: {type(C_python)}")
    
    atol = 1e-04
    print(f"\nSummary OpenCL vs Python GaussianCTE 1D gaussian data using {m}:\n")

    print(f"\t\t\tOpenCL\t\t\tPython\t\t\tclose {atol}")
    print(f"actualValue:\n\t\t\t{C_opencl.actualValue}\t{C_python.actualValue}\t{np.isclose(C_opencl.actualValue, C_python.actualValue, atol=atol)}")
    print(f"pValue:\n\t\t\t{C_opencl.pValue}\t{C_python.pValue}\t{np.isclose(C_opencl.pValue, C_python.pValue, atol=atol)}")
    print(f"getMeanOfDistribution:\n\t\t\t{mean_opencl}\t{mean_python}\t{np.isclose(mean_opencl, mean_python, atol=atol)}")
    print(f"getMeanOfUncorrectedDistribution:\n\t\t\t{mean_uncorr_opencl}\t{mean_uncorr_python}\t{np.isclose(mean_uncorr_opencl, mean_uncorr_python, atol=atol)}")
    print(f"StdOfDistribution:\n\t\t\t{std_opencl}\t{std_python}\t{np.isclose(std_opencl, std_python, atol=atol)}")

    print("\nEstimateForGivenPValue:")
    print("p\tOpenCLGaussianCTE\t\tPythonGaussianCTE")
    for i in range(len(pvals)):
        print(f"{pvals[i]}   \t{EoP_opencl[i]}\t{EoP_python[i]}")
    verbose(EoP_opencl, EoP_python, "", "Estimate for given PValue")

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

    EoP_opencl = np.zeros(len(pvals))
    EoP_python = np.zeros(len(pvals))
    
    settings_opencl = {'history_target': 1,
    	'noise_level': 0, 
        'normalise': False,}

    settings_python = {'history_target': 1,
    	'noise_level': 0, 
        'normalise': False,}

    est_opencl = OpenCLGaussianCTE(settings_opencl)
    est_python = PythonGaussianCTE(settings_python)

    mi = est_opencl.estimate(source, target)
    C_opencl = est_opencl.get_analytic_distribution(source, target)

    mi2 = est_python.estimate(source, target)
    C_python = est_python.get_analytic_distribution(source, target)

    mean_opencl = C_opencl.getMeanOfDistribution()
    mean_python = C_python.getMeanOfDistribution()
    
    mean_uncorr_opencl = C_opencl.getMeanOfUncorrectedDistribution()
    mean_uncorr_python = C_python.getMeanOfUncorrectedDistribution()
    
    std_opencl = C_opencl.getStdOfDistribution()
    std_python = C_python.getStdOfDistribution()
    
    count = 0
    for p in pvals:
        EoP_opencl[count] = C_opencl.computeEstimateForGivenPValue(p)
        EoP_python[count] = C_python.computeEstimateForGivenPValue(p)
        count += 1

    print(f"OpenCL computeSignificance object:\ntype: {type(C_opencl)}")
    print(f"Python computeSignificance object:\ntype: {type(C_python)}")
    
    atol = 1e-04
    print(f"\nSummary OpenCL vs Python GaussianCTE (no cond) 1D gaussian data using {m}:\n")

    print(f"\t\t\tOpenCL\t\t\tPython\t\t\tclose {atol}")
    print(f"actualValue:\n\t\t\t{C_opencl.actualValue}\t{C_python.actualValue}\t{np.isclose(C_opencl.actualValue, C_python.actualValue, atol=atol)}")
    print(f"pValue:\n\t\t\t{C_opencl.pValue}\t{C_python.pValue}\t{np.isclose(C_opencl.pValue, C_python.pValue, atol=atol)}")
    print(f"getMeanOfDistribution:\n\t\t\t{mean_opencl}\t{mean_python}\t{np.isclose(mean_opencl, mean_python, atol=atol)}")
    print(f"getMeanOfUncorrectedDistribution:\n\t\t\t{mean_uncorr_opencl}\t{mean_uncorr_python}\t{np.isclose(mean_uncorr_opencl, mean_uncorr_python, atol=atol)}")
    print(f"StdOfDistribution:\n\t\t\t{std_opencl}\t{std_python}\t{np.isclose(std_opencl, std_python, atol=atol)}")

    print("\nEstimateForGivenPValue:")
    print("p\tOpenCLGaussianCTE\t\tPythonGaussianCTE")
    for i in range(len(pvals)):
        print(f"{pvals[i]}   \t{EoP_opencl[i]}\t{EoP_python[i]}")
    verbose(EoP_opencl, EoP_python, "", "Estimate for given PValue")

    print("\n=========================================================================")


def test_analytic_distribution_mi_discrete():
	pvals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	m = 'equal'
	bins = 5

	print(
		f"\n\nTesting Discrete MI on discretized gaussian data with cov=0.4\n using discretise_method {m} - {bins} bins\n")
	print(f"testing computeEstimateForGivenPValue for:\n\t{pvals}")

	expected_mi, source, source_uncorr, target = _get_gauss_data(seed=SEED)
	source = source[1:]
	source_uncorr = source_uncorr[1:]
	target = target[:-1]

	EoP_opencl = np.zeros(len(pvals))
	EoP_python = np.zeros(len(pvals))

	settings_opencl = {"discretise_method": m,
					   "n_discrete_bins": bins,
					   'noise_level': 0,
					   'normalise': False, }

	settings_python = {"discretise_method": m,
					   "n_discrete_bins": bins,
					   'noise_level': 0,
					   'normalise': False, }

	est_opencl = OpenCLDiscreteMI(settings_opencl)
	est_python = PythonDiscreteMI(settings_python)

	mi = est_opencl.estimate(source, target)
	C_opencl = est_opencl.computeSignificance()

	mi2 = est_python.estimate(source, target)
	C_python = est_python.computeSignificance()

	mean_opencl = C_opencl.getMeanOfDistribution()
	mean_python = C_python.getMeanOfDistribution()

	mean_uncorr_opencl = C_opencl.getMeanOfUncorrectedDistribution()
	mean_uncorr_python = C_python.getMeanOfUncorrectedDistribution()

	std_opencl = C_opencl.getStdOfDistribution()
	std_python = C_python.getStdOfDistribution()

	count = 0
	for p in pvals:
		EoP_opencl[count] = C_opencl.computeEstimateForGivenPValue(p)
		EoP_python[count] = C_python.computeEstimateForGivenPValue(p)
		count += 1

	print(f"OpenCL computeSignificance object:\ntype: {type(C_opencl)}")
	print(f"Python computeSignificance object:\ntype: {type(C_python)}")

	atol = 1e-06
	print(f"\nSummary OpenCL vs Python DiscreteMI discretised 1D gaussian data using {m}:\n")

	print(f"\t\t\tOpenCL\t\t\tPython\t\t\tclose {atol}")
	print(
		f"actualValue:\n\t\t\t{C_opencl.actualValue}\t{C_python.actualValue}\t{np.isclose(C_opencl.actualValue, C_python.actualValue, atol=atol)}")
	print(
		f"pValue:\n\t\t\t{C_opencl.pValue}\t{C_python.pValue}\t{np.isclose(C_opencl.pValue, C_python.pValue, atol=atol)}")
	print(
		f"getMeanOfDistribution:\n\t\t\t{mean_opencl}\t{mean_python}\t{np.isclose(mean_opencl, mean_python, atol=atol)}")
	print(
		f"getMeanOfUncorrectedDistribution:\n\t\t\t{mean_uncorr_opencl}\t{mean_uncorr_python}\t{np.isclose(mean_uncorr_opencl, mean_uncorr_python, atol=atol)}")
	print(f"StdOfDistribution:\n\t\t\t{std_opencl}\t{std_python}\t{np.isclose(std_opencl, std_python, atol=atol)}")

	print("\nEstimateForGivenPValue:")
	print("p\tOpenCLDiscreteMI\t\tPythonDiscreteMI")
	for i in range(len(pvals)):
		print(f"{pvals[i]}   \t{EoP_opencl[i]}\t{EoP_python[i]}")
	verbose(EoP_opencl, EoP_python, "", "Estimate for given PValue")

	print("\n=========================================================================")

def test_analytic_distribution_cmi_discrete():
	pvals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	m = 'equal'
	bins = 5

	print(
		f"\n\nTesting Discrete CMI on discretized gaussian data with cov=0.4\n using discretise_method {m} - {bins} bins\n")
	print(f"testing computeEstimateForGivenPValue for:\n\t{pvals}")

	expected_mi, source, source_uncorr, target = _get_gauss_data(seed=SEED)
	source = source[1:]
	source_uncorr = source_uncorr[1:]
	target = target[:-1]

	EoP_opencl = np.zeros(len(pvals))
	EoP_python = np.zeros(len(pvals))

	settings_opencl = {"discretise_method": m,
					   "n_discrete_bins": bins,
					   'noise_level': 0,
					   'normalise': False, }

	settings_python = {"discretise_method": m,
					   "n_discrete_bins": bins,
					   'noise_level': 0,
					   'normalise': False, }

	est_opencl = OpenCLDiscreteCMI(settings_opencl)
	est_python = PythonDiscreteCMI(settings_python)

	mi = est_opencl.estimate(source, target, source_uncorr)
	C_opencl = est_opencl.computeSignificance()

	mi2 = est_python.estimate(source, target, source_uncorr)
	C_python = est_python.computeSignificance()

	mean_opencl = C_opencl.getMeanOfDistribution()
	mean_python = C_python.getMeanOfDistribution()

	mean_uncorr_opencl = C_opencl.getMeanOfUncorrectedDistribution()
	mean_uncorr_python = C_python.getMeanOfUncorrectedDistribution()

	std_opencl = C_opencl.getStdOfDistribution()
	std_python = C_python.getStdOfDistribution()

	count = 0
	for p in pvals:
		EoP_opencl[count] = C_opencl.computeEstimateForGivenPValue(p)
		EoP_python[count] = C_python.computeEstimateForGivenPValue(p)
		count += 1

	print(f"OpenCL computeSignificance object:\ntype: {type(C_opencl)}")
	print(f"Python computeSignificance object:\ntype: {type(C_python)}")

	atol = 1e-06
	print(f"\nSummary OpenCL vs Python DiscreteCMI discretised 1D gaussian data using {m}:\n")

	print(f"\t\t\tOpenCL\t\t\tPython\t\t\tclose {atol}")
	print(
		f"actualValue:\n\t\t\t{C_opencl.actualValue}\t{C_python.actualValue}\t{np.isclose(C_opencl.actualValue, C_python.actualValue, atol=atol)}")
	print(
		f"pValue:\n\t\t\t{C_opencl.pValue}\t{C_python.pValue}\t{np.isclose(C_opencl.pValue, C_python.pValue, atol=atol)}")
	print(
		f"getMeanOfDistribution:\n\t\t\t{mean_opencl}\t{mean_python}\t{np.isclose(mean_opencl, mean_python, atol=atol)}")
	print(
		f"getMeanOfUncorrectedDistribution:\n\t\t\t{mean_uncorr_opencl}\t{mean_uncorr_python}\t{np.isclose(mean_uncorr_opencl, mean_uncorr_python, atol=atol)}")
	print(f"StdOfDistribution:\n\t\t\t{std_opencl}\t{std_python}\t{np.isclose(std_opencl, std_python, atol=atol)}")

	print("\nEstimateForGivenPValue:")
	print("p\tOpenCLDiscreteCMI\t\tPythonDiscreteCMI")
	for i in range(len(pvals)):
		print(f"{pvals[i]}   \t{EoP_opencl[i]}\t{EoP_python[i]}")
	verbose(EoP_opencl, EoP_python, "", "Estimate for given PValue")

	print("\n=========================================================================")

def test_analytic_distribution_cmi_nocond_discrete():
	pvals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	m = 'equal'
	bins = 5

	print(
		f"\n\nTesting Discrete CMI on discretized gaussian data (conditional=None) with cov=0.4\n using discretise_method {m} - {bins} bins\n")
	print(f"testing computeEstimateForGivenPValue for:\n\t{pvals}")

	expected_mi, source, source_uncorr, target = _get_gauss_data(seed=SEED)
	source = source[1:]
	source_uncorr = source_uncorr[1:]
	target = target[:-1]

	EoP_opencl = np.zeros(len(pvals))
	EoP_python = np.zeros(len(pvals))

	settings_opencl = {"discretise_method": m,
					   "n_discrete_bins": bins,
					   'noise_level': 0,
					   'normalise': False, }

	settings_python = {"discretise_method": m,
					   "n_discrete_bins": bins,
					   'noise_level': 0,
					   'normalise': False, }

	est_opencl = OpenCLDiscreteCMI(settings_opencl)
	est_python = PythonDiscreteCMI(settings_python)

	mi = est_opencl.estimate(source, target)
	C_opencl = est_opencl.get_analytic_distribution(source, target)

	mi2 = est_python.estimate(source, target)
	C_python = est_python.get_analytic_distribution(source, target)

	mean_opencl = C_opencl.getMeanOfDistribution()
	mean_python = C_python.getMeanOfDistribution()

	mean_uncorr_opencl = C_opencl.getMeanOfUncorrectedDistribution()
	mean_uncorr_python = C_python.getMeanOfUncorrectedDistribution()

	std_opencl = C_opencl.getStdOfDistribution()
	std_python = C_python.getStdOfDistribution()

	count = 0
	for p in pvals:
		EoP_opencl[count] = C_opencl.computeEstimateForGivenPValue(p)
		EoP_python[count] = C_python.computeEstimateForGivenPValue(p)
		count += 1

	print(f"OpenCL computeSignificance object:\ntype: {type(C_opencl)}")
	print(f"Python computeSignificance object:\ntype: {type(C_python)}")

	atol = 1e-06
	print(f"\nSummary OpenCL vs Python DiscreteCMI (no cond) discretised 1D gaussian data using {m}:\n")

	print(f"\t\t\tOpenCL\t\t\tPython\t\t\tclose {atol}")
	print(
		f"actualValue:\n\t\t\t{C_opencl.actualValue}\t{C_python.actualValue}\t{np.isclose(C_opencl.actualValue, C_python.actualValue, atol=atol)}")
	print(
		f"pValue:\n\t\t\t{C_opencl.pValue}\t{C_python.pValue}\t{np.isclose(C_opencl.pValue, C_python.pValue, atol=atol)}")
	print(
		f"getMeanOfDistribution:\n\t\t\t{mean_opencl}\t{mean_python}\t{np.isclose(mean_opencl, mean_python, atol=atol)}")
	print(
		f"getMeanOfUncorrectedDistribution:\n\t\t\t{mean_uncorr_opencl}\t{mean_uncorr_python}\t{np.isclose(mean_uncorr_opencl, mean_uncorr_python, atol=atol)}")
	print(f"StdOfDistribution:\n\t\t\t{std_opencl}\t{std_python}\t{np.isclose(std_opencl, std_python, atol=atol)}")

	print("\nEstimateForGivenPValue:")
	print("p\tOpenCLDiscreteCMI\t\tPythonDiscreteCMI")
	for i in range(len(pvals)):
		print(f"{pvals[i]}   \t{EoP_opencl[i]}\t{EoP_python[i]}")
	verbose(EoP_opencl, EoP_python, "", "Estimate for given PValue")

	print("\n=========================================================================")

def test_analytic_distribution_ais_discrete():
	pvals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	m = 'equal'
	bins = 5

	print(f"\n\nTesting Discrete AIS using 1D AR with history \n using discretise_method {m} - {bins} bins\n")
	print(f"testing computeEstimateForGivenPValue for:\n\t{pvals}")

	source1, source2 = _get_ar_data(seed=SEED)

	EoP_opencl = np.zeros(len(pvals))
	EoP_python = np.zeros(len(pvals))

	settings_opencl = {'history': 2,
					   "discretise_method": m,
					   "n_discrete_bins": bins,
					   'noise_level': 0,
					   'normalise': False, }

	settings_python = {'history': 2,
					   "discretise_method": m,
					   "n_discrete_bins": bins,
					   'noise_level': 0,
					   'normalise': False, }

	est_opencl = OpenCLDiscreteAIS(settings_opencl)
	est_python = PythonDiscreteAIS(settings_python)

	mi = est_opencl.estimate(source1)
	C_opencl = est_opencl.computeSignificance()

	mi2 = est_python.estimate(source1)
	C_python = est_python.computeSignificance()

	mean_opencl = C_opencl.getMeanOfDistribution()
	mean_python = C_python.getMeanOfDistribution()

	mean_uncorr_opencl = C_opencl.getMeanOfUncorrectedDistribution()
	mean_uncorr_python = C_python.getMeanOfUncorrectedDistribution()

	std_opencl = C_opencl.getStdOfDistribution()
	std_python = C_python.getStdOfDistribution()

	count = 0
	for p in pvals:
		EoP_opencl[count] = C_opencl.computeEstimateForGivenPValue(p)
		EoP_python[count] = C_python.computeEstimateForGivenPValue(p)
		count += 1

	print(f"OpenCL computeSignificance object:\ntype: {type(C_opencl)}")
	print(f"Python computeSignificance object:\ntype: {type(C_python)}")

	atol = 1e-06
	print(f"\nSummary OpenCL vs Python DiscreteAIS on AR data with history using {m}:\n")
	print(f"\t\t\tOpenCL\t\t\tPython\t\t\tclose {atol}")
	print(
		f"actualValue:\n\t\t\t{C_opencl.actualValue}\t{C_python.actualValue}\t{np.isclose(C_opencl.actualValue, C_python.actualValue, atol=atol)}")
	print(
		f"pValue:\n\t\t\t{C_opencl.pValue}\t{C_python.pValue}\t{np.isclose(C_opencl.pValue, C_python.pValue, atol=atol)}")
	print(
		f"getMeanOfDistribution:\n\t\t\t{mean_opencl}\t{mean_python}\t{np.isclose(mean_opencl, mean_python, atol=atol)}")
	print(
		f"getMeanOfUncorrectedDistribution:\n\t\t\t{mean_uncorr_opencl}\t{mean_uncorr_python}\t{np.isclose(mean_uncorr_opencl, mean_uncorr_python, atol=atol)}")
	print(f"StdOfDistribution:\n\t\t\t{std_opencl}\t{std_python}\t{np.isclose(std_opencl, std_python, atol=atol)}")

	print("\nEstimateForGivenPValue:")
	print("p\tOpenCLDiscreteAIS\t\tPythonDiscreteAIS")
	for i in range(len(pvals)):
		print(f"{pvals[i]}   \t{EoP_opencl[i]}\t{EoP_python[i]}")
	verbose(EoP_opencl, EoP_python, "", "Estimate for given PValue")

	print("\n=========================================================================")

def test_analytic_distribution_te_discrete():
	pvals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	m = 'equal'
	bins = 5

	print(
		f"\n\nTesting Discrete TE on discretized gaussian data with cov=0.4\n using discretise_method {m} - {bins} bins\n")
	print(f"testing computeEstimateForGivenPValue for:\n\t{pvals}")

	expected_mi, source, source_uncorr, target = _get_gauss_data(seed=SEED)
	source = source[1:]
	source_uncorr = source_uncorr[1:]
	target = target[:-1]

	EoP_opencl = np.zeros(len(pvals))
	EoP_python = np.zeros(len(pvals))

	settings_opencl = {'history_target': 1,
					   "discretise_method": m,
					   "n_discrete_bins": bins,
					   'noise_level': 0,
					   'normalise': False, }

	settings_python = {'history_target': 1,
					   "discretise_method": m,
					   "n_discrete_bins": bins,
					   'noise_level': 0,
					   'normalise': False, }

	est_opencl = OpenCLDiscreteTE(settings_opencl)
	est_python = PythonDiscreteTE(settings_python)

	mi = est_opencl.estimate(source, target)
	C_opencl = est_opencl.computeSignificance()

	mi2 = est_python.estimate(source, target)
	C_python = est_python.computeSignificance()

	mean_opencl = C_opencl.getMeanOfDistribution()
	mean_python = C_python.getMeanOfDistribution()

	mean_uncorr_opencl = C_opencl.getMeanOfUncorrectedDistribution()
	mean_uncorr_python = C_python.getMeanOfUncorrectedDistribution()

	std_opencl = C_opencl.getStdOfDistribution()
	std_python = C_python.getStdOfDistribution()

	count = 0
	for p in pvals:
		EoP_opencl[count] = C_opencl.computeEstimateForGivenPValue(p)
		EoP_python[count] = C_python.computeEstimateForGivenPValue(p)
		count += 1

	print(f"OpenCL computeSignificance object:\ntype: {type(C_opencl)}")
	print(f"Python computeSignificance object:\ntype: {type(C_python)}")

	atol = 1e-06
	print(f"\nSummary OpenCL vs Python DiscreteTE discretised 1D gaussian data using {m}:\n")

	print(f"\t\t\tOpenCL\t\t\tPython\t\t\tclose {atol}")
	print(
		f"actualValue:\n\t\t\t{C_opencl.actualValue}\t{C_python.actualValue}\t{np.isclose(C_opencl.actualValue, C_python.actualValue, atol=atol)}")
	print(
		f"pValue:\n\t\t\t{C_opencl.pValue}\t{C_python.pValue}\t{np.isclose(C_opencl.pValue, C_python.pValue, atol=atol)}")
	print(
		f"getMeanOfDistribution:\n\t\t\t{mean_opencl}\t{mean_python}\t{np.isclose(mean_opencl, mean_python, atol=atol)}")
	print(
		f"getMeanOfUncorrectedDistribution:\n\t\t\t{mean_uncorr_opencl}\t{mean_uncorr_python}\t{np.isclose(mean_uncorr_opencl, mean_uncorr_python, atol=atol)}")
	print(f"StdOfDistribution:\n\t\t\t{std_opencl}\t{std_python}\t{np.isclose(std_opencl, std_python, atol=atol)}")

	print("\nEstimateForGivenPValue:")
	print("p\tOpenCLDiscreteTE\t\tPythonDiscreteTE")
	for i in range(len(pvals)):
		print(f"{pvals[i]}   \t{EoP_opencl[i]}\t{EoP_python[i]}")
	verbose(EoP_opencl, EoP_python, "", "Estimate for given PValue")

	print("\n=========================================================================")


#### Test bi- and multivariate analysis (single target)
def test_single_target_analysis(analysis, est_type, numperm=500, samples=10000):
    """Test multivariate TE estimation from correlated Gaussians."""
    
    measure = analysis[-2:].lower()
    opencl_estimator = f"OpenCL{est_type}CMI"
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

    settings_opencl = {
        'cmi_estimator': opencl_estimator,
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
    
    print("\n#### Analyse single target OpenCL\n")

    itic = time.perf_counter()
    results_opencl = nw.analyse_single_target(
        settings_opencl, data, target=2, sources=[0, 1])
    mi_opencl = results_opencl.get_single_target(2, fdr=False)[measure][0]
    sources_opencl = results_opencl.get_target_sources(2, fdr=False)
    itoc = time.perf_counter()
    time_opencl = itoc-itic

    # Assert that only the correlated source was detected.
    assert len(sources_opencl) == 1, 'Wrong no. inferred sources: {0}.'.format(
        len(sources_opencl))
    assert sources_opencl[0] == 0, 'Wrong inferred source: {0}.'.format(sources_opencl[0])


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
    
    # Compare MultivariateTE() estimate to OpenCL and Python estimate. Mimick realisations used
    # internally by the algorithm.
    settings = {'lag_mi': 0, 'normalise': False, 'noise_level': 0}
    est_opencl = eval(f"{opencl_estimator}(settings)")
    est_python = eval(f"{python_estimator}(settings)")
    
    opencl_mi = est_opencl.estimate(var1=source[1:-1], var2=target[2:])
    python_mi = est_python.estimate(var1=source[1:-1], var2=target[2:])
    
    print(f"Summary of comparing {analysis} using {opencl_estimator} vs {python_estimator}:\n")
    if sources_opencl==sources_python:
        print(f"OpenCL {sources_opencl} and Python {sources_python} found identical target_sources. +++")
    else:
        print(f"opencl {sources_opencl} and Python {sources_python} DID NOT find identical target_sources. !!!!!!!")
    verbose(mi_opencl, opencl_mi, f"OpenCL {analysis} vs core", measure.upper(), atol=1e-03)
    verbose(mi_python, python_mi, f"Python {analysis} vs core", measure.upper(), atol=1e-03)
    verbose(mi_opencl, mi_python, f"OpenCL {analysis} vs Python {analysis}", measure.upper(), atol=1e-03)
    verbose(opencl_mi, python_mi, "OpenCL core vs Python core", measure.upper(), atol=1e-03)

    print("\n calculation times:")
    print(f"single target analysis {analysis} {opencl_estimator} nperms {numperm}: ", np.mean(time_opencl) )
    print(f"single target analysis {analysis} {python_estimator} nperms {numperm}: ", np.mean(time_python) )


#### Test network analysis
def test_network_analysis(analysis, est_type, numperm=300, samples=1000, reps=3):
	
	measure = analysis[-2:].lower()
	opencl_estimator = f"OpenCL{est_type}CMI"
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
	
	print("\n#### Analyse network OpenCL\n")

	settings = {
	    "cmi_estimator": opencl_estimator,
	    "n_perm_max_stat": numperm,
	    "n_perm_min_stat": numperm,
	    "n_perm_omnibus": numperm,
	    "n_perm_max_seq": numperm,
	    "max_lag_sources": 5,
	    "min_lag_sources": 1,
	}

	itic = time.perf_counter()
	results_opencl = network_analysis.analyse_network(settings, data)
	itoc = time.perf_counter()
	time_opencl = itoc - itic
	
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
	target_delays_opencl = [None]*5
	selected_sources_opencl = [None]*5
	selected_sources_te_opencl = [None]*5
	
	target_delays_python = [None]*5
	selected_sources_python = [None]*5
	selected_sources_te_python = [None]*5

	for t in range(5):
		target_delays_opencl[t] = results_opencl.get_target_delays(t, fdr=False)
		target_delays_python[t] = results_python.get_target_delays(t, fdr=False)

		target_opencl = results_opencl.get_single_target(t, fdr=False)
		selected_sources_opencl[t] = target_opencl['selected_vars_sources']
		selected_sources_te_opencl[t] = target_opencl[f'selected_sources_{measure}']
		
		target_python = results_python.get_single_target(t, fdr=False)
		selected_sources_python[t] = target_python['selected_vars_sources']
		selected_sources_te_python[t] = target_python[f'selected_sources_{measure}']

	
	print(f"\nSummary network analysis {analysis} - {opencl_estimator} vs {python_estimator}\n")
	
	print("\nselected sources:\n")
	print("target\tequal")
	for t in range(5):
		print(f"{t}\t\t{selected_sources_opencl[t]==selected_sources_python[t]}\t{opencl_estimator}  : {selected_sources_opencl[t]}\n\t\t\t\t{python_estimator}: {selected_sources_python[t]}")

	atol = 1e-03
	print("\ntarget delays:\n")
	print("target\t\t\t\t\t\tequal")
	for t in range(5):
		if len(target_delays_opencl[t])==len(target_delays_python[t]):
			equal = np.allclose(target_delays_opencl[t], target_delays_python[t], atol=atol)
		else:
			equal = False
		t1 = "\t"
		t2 = "\t\t"
		print(f"{t}\t{opencl_estimator}  :\t{target_delays_opencl[t]}{t1 if len(target_delays_opencl[t])>1 else t2}{equal}\n\t{python_estimator}:\t{target_delays_python[t]}")
	
	print(f"\nselected sources {measure.upper()}:\n")
	print(f"target\tclose {atol}")

	for t in range(5):
		try:
			if len(selected_sources_te_opencl[t])==len(selected_sources_te_python[t]):
				equal = np.allclose(selected_sources_te_opencl[t], selected_sources_te_python[t], atol=atol)	
			else: 
				equal = False
		except:
			equal = False
		print(f"{t}\t\t{equal}\t{opencl_estimator}  : {selected_sources_te_opencl[t]}\n\t\t\t\t{python_estimator}: {selected_sources_te_python[t]}")
	
	print("\nEdge lists:")
	print("OpenCL:")
	results_opencl.print_edge_list("max_te_lag", fdr=False)
	print("Python:")
	results_python.print_edge_list("max_te_lag", fdr=False)

	print("\n calculation times:")
	print(f" network_analysis {analysis} {opencl_estimator} nperms {numperm}: ", np.mean(time_opencl) )
	print(f" network_analysis {analysis} {python_estimator} nperms {numperm}: ", np.mean(time_python) )


#### test nonlinear granger
def test_nonlinear_granger(analysis, est_type, numperm=300, samples=1000, reps=6):
	
	opencl_estimator = f"OpenCL{est_type}"
	python_estimator = f"Python{est_type}"

	print(f"\n\nTesting nonlinear granger analysis via {analysis}")
	print(f"using mute data ({samples} samples, {reps} replications)\n")

	
	data = Data(normalise=False)  # initialise an empty data object
	data.generate_mute_data(n_samples=samples, n_replications=reps)
	data2 = copy.deepcopy(data)

	print("\n#### Analyse network OpenCL\n")

	settings = {
	    "target": 1,   # mandatory in settings for nonlinear single target analysis
	    "sources": 0,  # optional in settings for nonlinear  single targetanalysis
	    "cmi_estimator": opencl_estimator,
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
	
	# perform OpenCLGaussianCMI WITH nonlinear data
	itic = time.perf_counter()
	results_opencl = nonlin_analysis.analyse_network(settings, data)
	itoc = time.perf_counter()
	time_opencl = itoc - itic
	
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

	print(f"\nSummary nonlinear granger network analysis {opencl_estimator} vs {python_estimator}\n")
	
	print("ts = target_sources")
	print("type = type of target sources: 1=lin, 2=nonlin")
	print(f"\t{opencl_estimator}\t\t{python_estimator}")
	print("target\tts\t\ttype\tts\ttype\t\tequal ts\tequal type")
	for t in range(5):
		ts_opencl = results_opencl.get_nonlinear_target_sources(t, fdr=False)
		ts_python = results_python.get_nonlinear_target_sources(t, fdr=False)
		tt_opencl = results_opencl.get_target_source_types(t, fdr=False)
		tt_python = results_python.get_target_source_types(t, fdr=False)

		try:
			equal_ts = ts_opencl==ts_python
		except:
			equal_ts = False
		try:
			equal_tt = tt_opencl==tt_python
		except:
			equal_tt = False

		t1 = "\t"
		t2 = "\t\t"
		print(f"{t}\t\t{ts_opencl}{t2 if len(ts_opencl)<=1 else t1}{tt_opencl}{t2 if len(ts_opencl)<=1 else t1}{ts_python}{t2 if len(ts_python)<=1 else t1}{tt_python}\t\t{equal_ts}{t2 if len(tt_python)<=1 else t1}{equal_tt}")

	print("\n calculation times:")
	print(f" nonlinear Granger via {analysis} {opencl_estimator}: ", np.mean(time_opencl) )
	print(f" nonlinear Granger via {analysis} {python_estimator}: ", np.mean(time_python) )


if __name__ == '__main__':

	#### Test Gaussian OpenCL estimators
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


	#### Test bi- and multivariate analysis (single target)
	# Gaussian CMI

	testhead("BivariateMI GaussianCMI (analyse_single_target)")
	test_single_target_analysis("BivariateMI","Gaussian", samples=10000)
	"""	
		
	testhead("BivariateTE GaussianCMI (analyse_single_target)")
	test_single_target_analysis("BivariateTE","Gaussian", samples=10000)
	
	testhead("MultivariateMI GaussianCMI (analyse_single_target)")
	test_single_target_analysis("MultivariateMI","Gaussian", samples=10000)

	testhead("MultivariateTE GaussianCMI (analyse_single_target)")
	test_single_target_analysis("MultivariateTE","Gaussian", samples=10000)
	"""

	# Discrete CMI
	"""
	testhead("BivariateMI DiscreteCMI (analyse_single_target)")
	test_single_target_analysis("BivariateMI","Discrete", samples=10000)

	testhead("BivariateTE DiscreteCMI (analyse_single_target)")
	test_single_target_analysis("BivariateTE","Discrete", samples=10000)

	testhead("MultivariateMI DiscreteCMI (analyse_single_target)")
	test_single_target_analysis("MultivariateMI","Discrete", samples=10000)

	testhead("MultivariateTE DiscreteCMI (analyse_single_target)")
	test_single_target_analysis("MultivariateTE","Discrete", samples=100000)
	"""

	#### Test network analysis CMI
	
	# Gaussian
	"""
	testhead("network analysis BivariateMI GaussianCMI")
	test_network_analysis("BivariateMI","Gaussian", numperm=500, samples=10000, reps=3)
	
	testhead("network analysis BivariateTE GaussianCMI")
	test_network_analysis("BivariateTE","Gaussian", numperm=500, samples=10000, reps=3)
	
	testhead("network analysis MultivariateMI GaussianCMI")
	test_network_analysis("MultivariateMI","Gaussian", numperm=500, samples=10000, reps=3)

	testhead("network analysis MultivariateTE GaussianCMI")
	test_network_analysis("MultivariateTE","Gaussian", numperm=500, samples=10000, reps=3)
	"""

	# Discrete
	"""
	testhead("network analysis BivariateMI DiscreteCMI")
	test_network_analysis("BivariateMI","Discrete", numperm=300, samples=10000, reps=3)
	
	testhead("network analysis BivariateTE DiscreteCMI")
	test_network_analysis("BivariateTE","Discrete", numperm=300, samples=10000, reps=3)
	
	testhead("network analysis MultivariateMI DiscreteCMI")
	test_network_analysis("MultivariateMI","Discrete", numperm=300, samples=10000, reps=3)
	
	testhead("network analysis MultivariateTE DiscreteCMI")
	test_network_analysis("MultivariateTE","Discrete", numperm=300, samples=10000, reps=3)
	"""

	# Test nonlinear Granger analysis
	"""
	testhead("nonlinear granger network analysis BivariateTE GaussianCMI") 
	test_nonlinear_granger("BivariateTE", "GaussianCMI", numperm=500, samples=10000, reps=3)
	
	testhead("nonlinear granger network analysis MultivariateTE GaussianCMI")
	test_nonlinear_granger("MultivariateTE", "GaussianCMI", numperm=500, samples=10000, reps=3)
	"""
