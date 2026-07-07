



import numpy as np

import time
import sys

from idtxl.estimators_jidt import (JidtKraskovMI, JidtKraskovCMI, JidtKraskovAIS, JidtKraskovTE, JidtKraskovCTE, 
									JidtGaussianMI, JidtGaussianCMI, JidtGaussianTE, JidtGaussianCTE, JidtGaussianAIS, 
									JidtDiscreteMI, JidtDiscreteCMI , JidtDiscreteAIS, JidtDiscreteTE)
from idtxl.estimators_python import (PythonKraskovMI, PythonKraskovCMI, PythonKraskovAIS, PythonKraskovTE, PythonKraskovCTE, 
									PythonGaussianMI, PythonGaussianCMI, PythonGaussianTE, PythonGaussianCTE, PythonGaussianAIS, 
									PythonDiscreteMI, PythonDiscreteCMI, PythonDiscreteAIS, PythonDiscreteTE, 
									PythonSpectralMI, PythonSpectralCMI)

from idtxl.idtxl_utils import calculate_mi
import random as rn

from gen_testdata import _get_gauss_data, _get_ar_data, _generate_mute_data, _get_mem_binary_data, _get_freq_data, _get_cte_test_data


SEED = 42

def verbose(res_jidt, res_python, values, est, rtol=1e-04, atol=1e-04, local=False):

	if local:
		addstring = " local"
	else:
		addstring = ""

	if atol < 1e-03:

		if np.allclose(res_jidt, res_python, rtol=rtol, atol=atol):

			print(f"{values} - all{addstring} {est} results within tolerance (atol = {atol:.0e}) +++")
		else:
		
			rtol=rtol*10
			atol=atol*10
			if np.allclose(res_jidt, res_python, rtol=1e-03, atol=1e-03):
				print(f"{values} - all{addstring} {est} results within tolerance (atol = {atol:.0e}) ---")
			else:
				diff = abs(res_jidt - res_python)
				num = (diff>1e-03).sum()
				try:
					print(f"{values} - {num}/{res_jidt.shape[0]} of{addstring} {est} results are not within tolerance (atol = {atol:.0e}) !!!!!!")
				except:
					print(f"{values} - {est} result is not within tolerance (atol = {atol:.0e}) !!!!!!")

	else:
		if np.allclose(res_jidt, res_python, rtol=rtol, atol=atol):

			print(f"{values} - all{addstring} {est} results within tolerance (atol = {str(atol)}) +++")
		else:
			diff = abs(res_jidt - res_python)
			num = (diff>1e-03).sum()	
			print(f"{values} - {num}/{res_jidt.shape[0]} of{addstring} {est} results are not within tolerance (atol = {str(atol)}) !!!!!!")

def testhead(est):
	print("\n#######################################################################")
	print(f"\n            Compare {est}:\n")
	print("#######################################################################")



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

	print(f"\n\nTesting average MI using 1D gaussian data with covariance 0.4 - uncorrelated conditional vs uncorrelated source:\n")
	print(f"testing settings lag_mi {vals}")

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


	# test 2D

	vals = [0,1,2,3]

	print(f"\n\nTesting average MI using 2D mute data with and without coupling\n")
	print(f"testing settings lag_mi {vals}")

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

	
	# test mixed dimension input
	d = [1, 2, 3, 5]

	print(f"\n\nTesting average MI using mixed dimensions\ntesting dimensions {d} for var1 and var2 each")
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
	
	print(f"\n\nTesting local MI using 1D gaussian data with covariance 0.4 - uncorrelated and uncorrelated\n")
	print(f"testing settings lag_mi {vals}")

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
		mi_jidt = jidt_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_jidt_cor[lags] = itoc - itic

		itic = time.perf_counter()
		mi_python = python_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_python_cor[lags] = itoc - itic

		itic = time.perf_counter()
		mi_jidt = jidt_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		time_jidt_uncor[lags] = itoc - itic
		
		itic = time.perf_counter()
		mi_python = python_estimator.estimate(source2, target)
		itoc = time.perf_counter()
		time_python_uncor[lags] = itoc - itic
		
		verbose(mi_jidt, mi_python, lags, "MI (coupled)", local=True)
		verbose(mi_jidt, mi_python, lags, "MI (not coupled)", local=True)
	
	print("\nmean calculation times:")
	print(" JidtGaussianMI (coupled): ", np.mean(time_jidt_cor) )
	print(" PythonGaussianMI (coupled): ", np.mean(time_python_cor) )
	print(" JidtGaussianMI (not coupled): ", np.mean(time_jidt_uncor) )
	print(" PythonGaussianMI (not coupled): ", np.mean(time_python_uncor) )

	
	# test 2D
	print(f"\n\nTesting local MI using 2D mute data with and without coupling\n")
	print(f"testing settings lag_mi {vals}")

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

		verbose(mi_jidt_cor, mi_python_cor, lags, "MI (coupled) 2D input", local=True, atol=1e-03)

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
		
		verbose(mi_jidt_uncor, mi_python_uncor, lags, "MI (not couled) 2D input", local=True, atol=1e-03)

	
	print("\nmean calculation times:")
	print(" JidtGaussianMI (coupled): ", np.mean(time_jidt_cor) )
	print(" PythonGaussianMI (coupled): ", np.mean(time_python_cor) )
	print(" JidtGaussianMI (not coupled): ", np.mean(time_jidt_uncor) )
	print(" PythonGaussianMI (not coupled): ", np.mean(time_python_uncor) )

def test_gaussian_cmi():

	cmi_jidt = np.zeros(8)
	cmi_python = np.zeros(8)
	time_jidt = np.zeros(8)
	time_python = np.zeros(8)
	
	vals = [0.2, 0.4, 0.6, 0.8]

	print(f"\n\nTesting average CMI using 1D gaussian data with different \ncovariances: {vals} - uncorrelated conditional vs uncorrelated source\n")

	count = 0
	for i in vals:

		expected_mi, source1, source2, target = _get_gauss_data(expand=True, covariance=i, seed=SEED)

		settings={}
		
		jidt_estimator = JidtGaussianCMI(settings)
		python_estimator = PythonGaussianCMI(settings)
		
		itic = time.perf_counter()
		cmi_jidt[count] = jidt_estimator.estimate(source1, target, source2)
		itoc = time.perf_counter()
		time_jidt[count] = itoc - itic

		itic = time.perf_counter()
		cmi_python[count] = python_estimator.estimate(source1, target, source2)
		itoc = time.perf_counter()
		time_python[count] += itoc - itic

		itic = time.perf_counter()
		cmi_jidt[count] = jidt_estimator.estimate(source2, target, source1)
		itoc = time.perf_counter()
		time_jidt[count] += itoc - itic

		itic = time.perf_counter()
		cmi_python[count] = python_estimator.estimate(source2, target, source1)
		itoc = time.perf_counter()
		time_python[count] += itoc - itic

		count += 1 

	print("cov\tJidtGaussianCMI\t\tPythonGaussianCMI")
	print("uncorr conditional")
	for i in range(len(vals)):
		print(f"{vals[i]}\t{cmi_jidt[i]}\t{cmi_python[i]}")
	print("uncorr source")
	for i in range(len(vals)):
		print(f"{vals[i]}\t{cmi_jidt[i+4]}\t{cmi_python[i+4]}")
	
	verbose(cmi_jidt, cmi_python, "", "CMI", local=False)

	print("\nmean calculation times:")
	print(" JidtGaussianCMI: ", np.mean(time_jidt) )
	print(" PythonGaussianCMI: ", np.mean(time_python) )


	# test mixed dimension input
	print(f"\n\nTesting average CMI using 2D mute data - uncorrelated conditional vs uncorrelated source\n")

	data = _generate_mute_data()

	source1 = data[0,:,:]
	target = data[2,:,:]
	source2 = data[4,:,:]

	settings={}
	
	jidt_estimator = JidtGaussianCMI(settings)
	python_estimator = PythonGaussianCMI(settings)
	
	itic = time.perf_counter()
	cmi_jidt = jidt_estimator.estimate(source1, target, source2)
	itoc = time.perf_counter()
	time_jidt = itoc - itic

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


	# test mixed dimension input
	d = [1, 2, 3]

	print(f"\n\nTesting average MI using mixed dimensions\ntesting dimensions {d} for var1, var2 and cond each")
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

	print(f"\n\nTesting local CMI using 1D gaussian data with covariance 0.4 - uncorrelated conditional vs uncorrelated source\n")
	
	expected_mi, source1, source2, target = _get_gauss_data(expand=True, seed=SEED)

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
	
	print("no history")
	for i in range(len(res_jidt_uncor)):
		print(f"{conds[i]}\t{res_jidt_uncor[i]}\t{res_python_uncor[i]}")

	verbose(res_jidt_uncor, res_python_uncor, "", "AIS (no hist)", local=True)
	
	print("\nmean calculation times:")
	print(" JidtGaussianAIS (cor): ", np.mean(time_jidt_cor) )
	print(" PythonGaussianAIS (cor): ", np.mean(time_python_cor) )
	print(" JidtGaussianAIS (uncor): ", np.mean(time_jidt_uncor) )
	print(" PythonGaussianAIS (uncor): ", np.mean(time_python_uncor) )

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

	vals = [1,2,3]

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

	vals = [1,2,3]
	
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

						
						verbose(te_jidt, te_python, [hst, ht, tt, hs, ts], "TE", local=True)

	
	print("\nmean calculation times:")
	print(" JidtGaussianTE: ", np.mean(time_jidt) )
	print(" PythonGaussianTE: ", np.mean(time_python) )


def test_gaussian_cte():

	vals = [1,2,3]

	print(f"\n\nTesting average CTE using 1D mute data - correlated and uncorrelated conditional\n")
	print(f"testing settings history_source, tau_source, history_target, tau_target, history_conditional")
	print(f"tau_conditional, source_target_delay and conditional_target_delay with {vals}")
	
	data = _generate_mute_data(n_replications=1)
	source1 = data[0,:]
	target = data[4,:]
	cond = data[3,:]
	nocond = data[5,:]

	vals = [1,2,3]
	
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
									
									#verbose(cte_jidt_cond, cte_python_cond, f"{hst,cst,ht,tt,hs,ts,hc,tc}", "CTE", local=True, atol=atol) 
									print(f"{hst,cst,ht,tt,hs,ts,hc,tc}\t{cte_jidt_cond}\t{cte_python_cond}\t\t{np.isclose(cte_jidt_cond, cte_python_cond, rtol=atol, atol=atol)}\t\t{np.isclose(cte_jidt_nocond, cte_python_nocond, rtol=atol, atol=atol)}")

	verbose(res_jidt_cond, res_python_cond, "correlated conditional", "CTE", atol=1e-04)
	verbose(res_jidt_nocond, res_python_nocond, "uncorrelated conditional" , "CTE", atol=1e-04)

	print("\nmean calculation times:")
	print(" JidtGaussianCTE (correlated conditional): ", np.mean(time_jidt_cond) )
	print(" PythonGaussianCTE (correlated conditional): ", np.mean(time_python_cond) )
	print(" JidtGaussianCTE (uncorrelated conditional): ", np.mean(time_jidt_nocond) )
	print(" PythonGaussianCTE (uncorrelated conditional): ", np.mean(time_python_nocond) )


# =??????????????????????????????????????????????????????????
def test_gaussian_cte_local_values():
	
	vals = [1,3]

	print(f"\n\nTesting local CTE using 1D mute data - correlated and uncorrelated conditional\n")
	print(f"testing settings history_source, tau_source, history_target, tau_target, history_conditional")
	print(f"tau_conditional, source_target_delay and conditional_target_delay with {vals}")
	
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

	print("hst,ht,tt,hs,ts\t\tJidtGaussianCTE vs PythonGaussianCTE")

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
										"local_values": True}
																		
									jidt_estimator = JidtGaussianCTE(settings)
									
									itic = time.perf_counter()
									cte_jidt_cond = jidt_estimator.estimate(source1, target, cond)
									itoc = time.perf_counter()
									time_jidt_cond += itoc - itic
									
									itic = time.perf_counter()
									cte_jidt_nocond = jidt_estimator.estimate(source1, target, nocond)
									itoc = time.perf_counter()
									time_jidt_nocond += itoc - itic
									
									python_estimator = PythonGaussianCTE(settings)
									
									itic = time.perf_counter()
									cte_python_cond = python_estimator.estimate(source1, target, cond)
									itoc = time.perf_counter()
									time_python_cond += itoc - itic
									
									itic = time.perf_counter()
									cte_python_nocond = python_estimator.estimate(source1, target, nocond)
									itoc = time.perf_counter()
									time_python_nocond += itoc - itic
									
									#print(cte_jidt_cond[:10])
									#print(cte_python_cond[:10])

									verbose(cte_jidt_cond, cte_python_cond, f"{hst,cst,ht,tt,hs,ts,hc,tc} correlated conditional", "CTE", local=True, atol=atol) 
									verbose(cte_jidt_nocond, cte_python_nocond, f"{hst,cst,ht,tt,hs,ts,hc,tc} uncorrelated conditional", "CTE", local=True, atol=atol) 


	print("\nmean calculation times:")
	print(" JidtGaussianCTE (correlated conditional): ", np.mean(time_jidt_cond) )
	print(" PythonGaussianCTE (correlated conditional): ", np.mean(time_python_cond) )
	print(" JidtGaussianCTE (uncorrelated conditional): ", np.mean(time_jidt_nocond) )
	print(" PythonGaussianCTE (uncorrelated conditional): ", np.mean(time_python_nocond) )


	
# Test Kraskov estimators
def test_kraskov_mi():

	# test 1D data
	expected_mi, source1, source2, target = _get_gauss_data(expand=True, seed=SEED)
	source1 = source1[1:]
	source2 = source2[1:]
	target = target[:-1]

	lvals = [0,1,2,3]
	kvals = [2,4,6,8]

	time_jidt_cor = np.empty(np.power(len(kvals),2))
	mi_jidt_cor = np.empty(np.power(len(kvals),2))
	time_python_cor = np.empty(np.power(len(kvals),2))
	mi_python_cor = np.empty(np.power(len(kvals),2))
	time_jidt_uncor = np.empty(np.power(len(kvals),2))
	mi_jidt_uncor = np.empty(np.power(len(kvals),2))
	time_python_uncor = np.empty(np.power(len(kvals),2))
	mi_python_uncor = np.empty(np.power(len(kvals),2))

	conds = np.empty((np.power(len(kvals),2),2))

	print(f"\n\nTesting average MI using 1D gaussian data with covariance 0.4 and lag 1 \ntesting settings kraskov k {kvals} and lags_mi {lvals}\n")
	
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

	# test mixed dimension input
	d = [1, 2, 3, 5]

	print(f"\n\nTesting average MI using mixed dimensions\ntesting dimensions {d} for var1 and var2 each")
	print("Shapes:")
	data = _generate_mute_data(n_replications=5)
	source_o = data[0,:,:]
	target_o = data[2,:,:]
	
	settings2 = {"kraskov_k": k,
				"noise_level": 0,
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

			#print(mi_jidt_cor)
			#print(mi_python_cor)
			
			verbose(mi_jidt_cor, mi_python_cor, cond, "MI")
	

	# test theiler 
	expected_mi, source1, source2, target = _get_gauss_data(expand=True, seed=SEED)
	source1 = source1[1:]
	source2 = source2[1:]
	target = target[:-1]

	tvals = [0,1,2,3]
	
	time_jidt_cor = np.empty(np.power(len(kvals),2))
	mi_jidt_cor = np.empty(np.power(len(kvals),2))
	time_python_cor1 = np.empty(np.power(len(kvals),2))
	mi_python_cor1 = np.empty(np.power(len(kvals),2))
	time_python_cor2 = np.empty(np.power(len(kvals),2))
	mi_python_cor2 = np.empty(np.power(len(kvals),2))
	
	conds = np.empty((np.power(len(kvals),2),2))

	print(f"\n\nTesting average MI using 1D gaussian data with covariance 0.4 and lag 1 \ntesting settings kraskov k {kvals} and theiler t {tvals}\nusing knn_finder scipy_ckdtree and scipy_kdtree\n")
	
	count = 0

	knn = ['scipy_kdtree', 'scipy_ckdtree']

	for k in kvals:
		for t in tvals:
			conds[count,:] = [k, t]
		
			settings_j = {"kraskov_k": k,
						"noise_level": 0,
						"num_threads": "USE_ALL",
						'theiler_t': t}
			settings_p1 = {"kraskov_k": k,
						"noise_level": 0,
						"knn_finder": "scipy_kdtree",
						"num_threads": "USE_ALL",
						"theiler_t": t}

			settings_p2 = {"kraskov_k": k,
						"noise_level": 0,
						"knn_finder": "scipy_ckdtree",
						"num_threads": "USE_ALL",
						"theiler_t": t}

			jidt_estimator = JidtKraskovMI(settings_j)
			python_estimator1 = PythonKraskovMI(settings_p1)
			python_estimator2 = PythonKraskovMI(settings_p2)

			itic = time.perf_counter()
			mi_jidt_cor[count] = jidt_estimator.estimate(source1, target)
			itoc = time.perf_counter()
			time_jidt_cor[count] = itoc - itic

			itic = time.perf_counter()
			mi_python_cor1[count] = python_estimator1.estimate(source1, target)
			itoc = time.perf_counter()
			time_python_cor1[count] = itoc - itic

			itic = time.perf_counter()
			mi_python_cor2[count] = python_estimator2.estimate(source1, target)
			itoc = time.perf_counter()
			time_python_cor2[count] = itoc - itic

			count += 1

	print("k, t\tJidtKraskovMI\t\tPythonKraskovMI\t\tPythonKraskovMI\t\tclose 1e-03")
	print("\t\t\t\tscipy_kdtree\t\tscipy_ckdtree")
	for i in range(len(mi_python_cor1)):
		print(f"{conds[i,:]}\t{mi_jidt_cor[i]}\t{mi_python_cor1[i]}\t{mi_python_cor2[i]}\t{np.isclose(mi_jidt_cor[i], mi_python_cor1[i], atol=1e-03)}\t{np.isclose(mi_jidt_cor[i], mi_python_cor2[i], atol=1e-03)}")
	
	verbose(mi_jidt_cor, mi_python_cor1, "scipy_kdtree", "MI", local=False)
	verbose(mi_jidt_cor, mi_python_cor2, "scipy_ckdtree", "MI", local=False)
	
	print("\nmean calculation times:")
	print(" JidtKraskovMI: ", np.mean(time_jidt_cor) )
	print(" PythonKraskovMI (scipy_kdtree): ", np.mean(time_python_cor1) )
	print(" PythonKraskovMI (scipy_ckdtree): ", np.mean(time_python_cor2) )
	

def test_kraskov_mi_local_values():

	expected_mi, source1, source2, target = _get_gauss_data(expand=True, seed=SEED)
	
	lvals = [0,1,2,3]
	kvals = [2,4,6,8]

	time_jidt_cor = np.empty(np.power(len(kvals),2))
	time_python_cor = np.empty(np.power(len(kvals),2))
	time_jidt_uncor = np.empty(np.power(len(kvals),2))
	time_python_uncor = np.empty(np.power(len(kvals),2))
	
	conds = np.empty((np.power(len(kvals),2),2))

	print(f"\n\nTesting local MI using 1D gaussian data with covariance 0.4 and lag 1 \ntesting settings kraskov k {kvals} and lag_mi {lvals}\n")
	
	print(f"k, lag\t\tJidtKraskovMI vs PythonKraskovMI")
	count = 0
	for k in kvals:
		for l in lvals:
			conds[count,:] = [k, l]
			settings = {}
			settings_j = {"kraskov_k": k,
						"noise_level": 0,
						"local_values": True,
						"num_threads": "USE_ALL",
						"lag_mi": l}
			settings_p = {"kraskov_k": k,
						"noise_level": 0,
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


	# test 2D
	print(f"\n\nTesting local MI using 2D mute data with and without coupling\n")
	
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
						"local_values": True,
						"num_threads": "USE_ALL",
						"lag_mi": l}
			settings_p = {"kraskov_k": k,
						"noise_level": 0,
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

def test_kraskov_cmi():

	# test 1D data
	
	cvals = [0.2, 0.4, 0.6, 0.8]
	kvals = [2,4,6,8]
	
	time_jidt_cor = np.empty(np.power(len(kvals),2))
	mi_jidt_cor = np.empty(np.power(len(kvals),2))
	time_python_cor = np.empty(np.power(len(kvals),2))
	mi_python_cor = np.empty(np.power(len(kvals),2))
	time_jidt_uncor = np.empty(np.power(len(kvals),2))
	mi_jidt_uncor = np.empty(np.power(len(kvals),2))
	time_python_uncor = np.empty(np.power(len(kvals),2))
	mi_python_uncor = np.empty(np.power(len(kvals),2))

	conds = np.empty((np.power(len(kvals),2),2))
	
	print(f"\n\nTesting average CMI using 1D gaussian data with covariances {cvals} \ntesting settings kraskov k {kvals} and uncorrelated conditional and uncorrelated source\n")
	
	count = 0
	for k in kvals:
		for i in cvals:
			conds[count,:] = [k,i]
			expected_mi, source1, source2, target = _get_gauss_data(expand=True, covariance=i, seed=SEED)
			
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
		print(f"{conds[i,:]}\t{mi_jidt_cor[i]}\t{mi_python_cor[i]}")
	print("uncorrelated source")
	for i in range(len(mi_jidt_uncor)):
		print(f"{conds[i,:]}\t{mi_jidt_uncor[i]}\t{mi_python_uncor[i]}")
	
	verbose(mi_jidt_cor, mi_python_cor, "uncorrelated conditional", "CMI", local=False)
	verbose(mi_jidt_uncor, mi_python_uncor, "uncorrelated source", "CMI", local=False)

	print("\nmean calculation times:")
	print(" JidtKraskovCMI: (uncorrelated conditional)", np.mean(time_jidt_cor) )
	print(" PythonKraskovCMI: (uncorrelated conditional)", np.mean(time_python_cor) )
	print(" JidtKraskovCMI: (uncorrelated source)", np.mean(time_jidt_uncor) )
	print(" PythonKraskovCMI: (uncorrelated source)", np.mean(time_python_uncor) )


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
					"num_threads": "USE_ALL"}
		settings_p = {"kraskov_k": k,
					"noise_level": 0,
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

				
				verbose(mi_jidt_cor, mi_python_cor, cond, "MI")
		
	
	# test theiler 
	expected_mi, source1, source2, target = _get_gauss_data(expand=True, seed=SEED)
	
	tvals = [0,1,2,3]
	
	time_jidt_cor = np.empty(np.power(len(kvals),2))
	mi_jidt_cor = np.empty(np.power(len(kvals),2))
	time_python_cor1 = np.empty(np.power(len(kvals),2))
	mi_python_cor1 = np.empty(np.power(len(kvals),2))
	time_python_cor2 = np.empty(np.power(len(kvals),2))
	mi_python_cor2 = np.empty(np.power(len(kvals),2))
	
	conds = np.empty((np.power(len(kvals),2),2))

	print(f"\n\nTesting average CMI using 1D gaussian data with covariance 0.4 \ntesting settings kraskov k {kvals} and theiler t {tvals}\nusing knn_finder scipy_ckdtree and scipy_kdtree\n")
	
	count = 0

	knn = ['scipy_kdtree', 'scipy_ckdtree']

	for k in kvals:
		for t in tvals:
			conds[count,:] = [k, t]
		
			settings_j = {"kraskov_k": k,
						"noise_level": 0,
						"num_threads": "USE_ALL",
						'theiler_t': t}
			settings_p1 = {"kraskov_k": k,
						"noise_level": 0,
						"knn_finder": "scipy_kdtree",
						"num_threads": "USE_ALL",
						"theiler_t": t}

			settings_p2 = {"kraskov_k": k,
						"noise_level": 0,
						"knn_finder": "scipy_ckdtree",
						"num_threads": "USE_ALL",
						"theiler_t": t}

			jidt_estimator = JidtKraskovCMI(settings_j)
			python_estimator1 = PythonKraskovCMI(settings_p1)
			python_estimator2 = PythonKraskovCMI(settings_p2)

			itic = time.perf_counter()
			mi_jidt_cor[count] = jidt_estimator.estimate(source1, target, source2)
			itoc = time.perf_counter()
			time_jidt_cor[count] = itoc - itic

			itic = time.perf_counter()
			mi_python_cor1[count] = python_estimator1.estimate(source1, target, source2)
			itoc = time.perf_counter()
			time_python_cor1[count] = itoc - itic

			itic = time.perf_counter()
			mi_python_cor2[count] = python_estimator2.estimate(source1, target, source2)
			itoc = time.perf_counter()
			time_python_cor2[count] = itoc - itic

			count += 1

	print("k, t\tJidtKraskovCMI\t\tPythonKraskovCMI\tPythonKraskovCMI\tclose 1e-03")
	print("\t\t\t\tscipy_kdtree\t\tscipy_ckdtree")
	for i in range(len(mi_python_cor1)):
		print(f"{conds[i,:]}\t{mi_jidt_cor[i]}\t{mi_python_cor1[i]}\t{mi_python_cor2[i]}\t{np.isclose(mi_jidt_cor[i], mi_python_cor1[i], atol=1e-03)}\t{np.isclose(mi_jidt_cor[i], mi_python_cor2[i], atol=1e-03)}")
	
	verbose(mi_jidt_cor, mi_python_cor1, "scipy_kdtree", "MI", local=False)
	verbose(mi_jidt_cor, mi_python_cor2, "scipy_ckdtree", "MI", local=False)
	
	
	print("\nmean calculation times:")
	print(" JidtKraskovCMI: ", np.mean(time_jidt_cor) )
	print(" PythonKraskovCMI (scipy_kdtree): ", np.mean(time_python_cor1) )
	print(" PythonKraskovCMI (scipy_ckdtree): ", np.mean(time_python_cor2) )
	

def test_kraskov_cmi_local_values():

	# test 1D	
	expected_mi, source1, source2, target = _get_gauss_data(expand=True, seed=SEED)

	vals = [2,4,6,8]
	
	time_jidt_cor = np.zeros(4)
	time_python_cor = np.zeros(4)
	time_jidt_uncor = np.zeros(4)
	time_python_uncor = np.zeros(4)

	print(f"\n\nTesting local CMI using 1D gaussian data with covariances 0.4 \ntesting settings kraskov {vals} and uncorrelated conditional and uncorrelated source\n")
	
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
					"num_threads": "USE_ALL"}

		jidt_estimator = JidtKraskovCMI(settings_j)
		python_estimator = PythonKraskovCMI(settings_p)

		itic = time.perf_counter()
		cmi_jidt_cor = jidt_estimator.estimate(source1, target, source2)
		itoc = time.perf_counter()
		time_jidt_cor[count] = itoc - itic

		itic = time.perf_counter()
		cmi_python_cor = python_estimator.estimate(source1, target, source2)
		itoc = time.perf_counter()
		time_python_cor[count] = itoc - itic
		
		verbose(cmi_jidt_cor, cmi_python_cor, f"{k} uncorrelated conditional", "CMI", local=True, atol=1e-03)

		itic = time.perf_counter()
		cmi_jidt_uncor = jidt_estimator.estimate(source1, target, source2)
		itoc = time.perf_counter()
		time_jidt_uncor[count] = itoc - itic

		itic = time.perf_counter()
		cmi_python_uncor = python_estimator.estimate(source1, target, source2)
		itoc = time.perf_counter()
		time_python_uncor[count] = itoc - itic

		verbose(cmi_jidt_uncor, cmi_python_uncor, f"{k} uncorrelated source", "CMI", local=True, atol=1e-03)

	print("\nmean calculation times:")
	print(" JidtKraskovCMI: (uncorrelated conditional)", np.mean(time_jidt_cor) )
	print(" PythonKraskovCMI: (uncorrelated conditional)", np.mean(time_python_cor) )
	print(" JidtKraskovCMI: (uncorrelated source)", np.mean(time_jidt_uncor) )
	print(" PythonKraskovCMI: (uncorrelated source)", np.mean(time_python_uncor) )

	# test 2D
	
	print(f"\n\nTesting local CMI using 2D mute data\ntesting settings kraskov {vals} and uncorrelated conditional and uncorrelated source\n")
	
	data = _generate_mute_data(n_samples=2000, n_replications=4)

	source1 = data[0,:,:]
	target = data[2,:,:]
	source2 = data[4,:,:]

	time_jidt_cor = np.zeros(4)
	time_python_cor = np.zeros(4)
	time_jidt_uncor = np.zeros(4)
	time_python_uncor = np.zeros(4)


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
					"num_threads": "USE_ALL"}

		jidt_estimator = JidtKraskovCMI(settings_j)
		python_estimator = PythonKraskovCMI(settings_p)

		itic = time.perf_counter()
		cmi_jidt_cor = jidt_estimator.estimate(source1, target, source2)
		itoc = time.perf_counter()
		time_jidt_cor[count] = itoc - itic

		itic = time.perf_counter()
		cmi_python_cor = python_estimator.estimate(source1, target, source2)
		itoc = time.perf_counter()
		time_python_cor[count] = itoc - itic
		
		verbose(cmi_jidt_cor, cmi_python_cor, f"{k} uncorrelated conditional", "CMI", local=True, atol=1e-03)

		itic = time.perf_counter()
		cmi_jidt_uncor = jidt_estimator.estimate(source1, target, source2)
		itoc = time.perf_counter()
		time_jidt_uncor[count] = itoc - itic

		itic = time.perf_counter()
		cmi_python_uncor = python_estimator.estimate(source1, target, source2)
		itoc = time.perf_counter()
		time_python_uncor[count] = itoc - itic

		verbose(cmi_jidt_uncor, cmi_python_uncor, f"{k} uncorrelated source", "CMI", local=True, atol=1e-03)

	print("\nmean calculation times:")
	print(" JidtKraskovCMI: (uncorrelated conditional)", np.mean(time_jidt_cor) )
	print(" PythonKraskovCMI: (uncorrelated conditional)", np.mean(time_python_cor) )
	print(" JidtKraskovCMI: (uncorrelated source)", np.mean(time_jidt_uncor) )
	print(" PythonKraskovCMI: (uncorrelated source)", np.mean(time_python_uncor) )

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
				settings_j = {'kraskov_k': k,'history': h, 'tau': t, 'local_values': True}

				settings_p = {'kraskov_k': k,'history': h, 'tau': t, 'local_values': True}
		
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

	vals = [1,2,3]

	print(f"\n\nTesting average TE using 1D gaussian data with covariance 0.4 and lag 1\n")
	print(f"testing settings history_source (hs), tau_source (ts), history_target (ht), \ntau_target (tt), source_target_delay (std) with {vals} each.\n")

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

def test_kraskov_te_local_values():

	vals = [1,2]
	
	print(f"\n\nTesting average TE using 1D gaussian data with covariance 0.4 and lag 1\n")
	print(f"testing settings history_source (hs), tau_source (ts), history_target (ht), \ntau_target (tt), source_target_delay (std) with {vals} each.\n")

	expected_mi, source1, source2, target = _get_gauss_data(expand=True, seed=SEED)

	source1 = source1[1:]
	source2 = source2[1:]
	target = target[:-1]

	time_jidt = np.empty(np.power(len(vals),5))
	res_jidt = np.empty(np.power(len(vals),5))
	time_python = np.empty(np.power(len(vals),5))
	res_python = np.empty(np.power(len(vals),5))
	
	conds = np.empty((np.power(len(vals),5),5))

	print("hst,ht,tt,hs,ts\t\tJidtKraskovTE vs PythonKraskovTE")

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

						settings_p = {"history_target": ht,
									"history_source": hs,
									"tau_target": tt,
									"tau_source": ts,
									"source_target_delay": hst,
									"local_values": True}

						jidt_estimator = JidtKraskovTE(settings_j)
						python_estimator = PythonKraskovTE(settings_p)
						
						itic = time.perf_counter()
						te_jidt = jidt_estimator.estimate(source=source1, target=target)
						itoc = time.perf_counter()
						time_jidt[count] = itoc-itic
						
						itic = time.perf_counter()
						te_python = python_estimator.estimate(source=source1, target=target)
						itoc = time.perf_counter()
						time_python[count] = itoc-itic
						
						count += 1
						
						verbose(te_jidt, te_python, [hst, ht, tt, hs, ts], "TE", atol=1e-03, local=True)

	print("\nmean calculation times:")
	print(" JidtKraskovTE: ", np.mean(time_jidt) )
	print(" PythonKraskovTE: ", np.mean(time_python) )



def test_Kraskov_cte():

	print(f"\n\nTesting average CTE using 1D mute data - with coupling and no coupling\n")

	#i = 0.4
	#expected_mi, source1, source2, target = _get_gauss_data(expand=True, covariance=i, seed=SEED)

	data = _generate_mute_data(n_replications=1)
	source1 = data[0,:]
	target = data[4,:]
	cond = data[3,:]
	nocond = data[5,:]

	vals = [1,2,3]
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
										"conditional_target_delay": cst}
									
									
									jidt_estimator = JidtKraskovCTE(settings)
									
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
									
									python_estimator = PythonKraskovCTE(settings)
									
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
									
									#verbose(cte_jidt_cond, cte_python_cond, f"{hst,cst,ht,tt,hs,ts,hc,tc}", "CTE", local=True, atol=atol) 
									print(f"{hst,cst,ht,tt,hs,ts,hc,tc}\t{cte_jidt_cond}\t{cte_python_cond}\t\t{np.isclose(cte_jidt_cond, cte_python_cond, rtol=atol, atol=atol)}\t{np.isclose(cte_jidt_nocond, cte_python_nocond, rtol=atol, atol=atol)}")

	verbose(res_jidt_cond, res_python_cond, "", "CTE cond", atol=atol)
	verbose(res_jidt_nocond, res_python_nocond, "", "CTE nocond", atol=atol)

	print("\nmean calculation times:")
	print(" JidtGaussianCTE (cond): ", np.mean(time_jidt_cond) )
	print(" PythonGaussianCTE (cond): ", np.mean(time_python_cond) )
	print(" JidtGaussianCTE (nocond): ", np.mean(time_jidt_nocond) )
	print(" PythonGaussianCTE (nocond): ", np.mean(time_python_nocond) )



############################################################# TODO CTE local


# Test Discrete estimators
def test_discrete_mi():

	vals = [2,5,8,32]
	lvals = [0,1,2,3]

	# test 1D gaussian
	print(f"\n\nTesting average MI using 1D gaussian data with covariance 0.4 - correlated and uncorrelated")
	print(f"testing settings lag_mi {lvals}, n_discrete_bins {vals} and discrete_method max_ent and equal")
	
	expected_mi, source1, source2, target = _get_gauss_data(expand=True, seed=SEED)
	
	for m in ['max_ent','equal']:
		print(f"\n--- discrete_method: {m}\n")
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
				
		print(f"Summary Jidt vs Python DiscreteMI discretised 1D gaussian data using {m}:")

		print("lags, nbins\tJidtDiscreteMI\t\tPythonDiscreteMI")
		print("correlated data:")
		for i in range(len(vals)):
			print(f"{conds[i]}   \t{mi_jidt_cor[i]}\t{mi_python_cor[i]}")

		print("\nuncorrelated data:")
		for i in range(len(vals)):
			print(f"{conds[i]}   \t{mi_jidt_uncor[i]}\t{mi_python_uncor[i]}")
		
		verbose(mi_jidt_cor, mi_python_cor, "correlated", "MI", local=False, atol=1e-03)
		verbose(mi_jidt_uncor, mi_python_uncor, "uncorrelated", "MI", local=False, atol=1e-03)

		print("\nmean calculation times:")
		print(" JidtDiscreteMI (correlated): ", np.mean(time_jidt_cor) )
		print(" PythonDiscreteMI (correlated): ", np.mean(time_python_cor) )
		print(" JidtDiscreteMI (uncorrelated): ", np.mean(time_jidt_uncor) )
		print(" PythonDiscreteMI (uncorrelated): ", np.mean(time_python_uncor) )
		
	
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


	# test 2D
	lvals = [0,1,2,3]

	print(f"\n\nTesting average MI using 2D mute data - correlated and uncorrelated")
	print(f"testing settings lag_mi {lvals}, n_discrete_bins 2 and discrete_method max_ent and equal")
	
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

	atol = 0.05

	vals = [0,1,2,3]
	print(f"\n\nTesting local MI using 1D gaussian data with covariance 0.4 - correlated and uncorrelated")
	print(f"testing settings lag_mi {vals}, n_discrete_bins 2 and discrete_method max_ent")
	
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
		
		verbose(mi_jidt, mi_python, lags, "MI (correlated)", local=True, atol=atol)

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

		verbose(mi_jidt, mi_python, lags, "MI (uncorrelated)", local=True, atol=atol)
	
	print("\nmean calculation times:")
	print(" JidtDiscreteMI: ", np.mean(jidt_time) )
	print(" PythonDiscreteMI: ", np.mean(python_time) )


	# test 2D 
	
	vals = [0,1,2,3]

	print(f"\n\nTesting local MI using 2D mute data - correlated and uncorrelated")
	print(f"testing settings lag_mi {vals}, n_discrete_bins 2 and discrete_method max_ent")
	
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

		print(mi_jidt_cor[:10])
		print(mi_python_cor[:10])

		verbose(mi_jidt_cor[:-100], mi_python_cor[:-100], lags, "MI (correlated) 2D input", local=True, atol=atol)

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
	print(f"testing settings n_discrete_bins {vals} and discrete_method max_ent and equal")
	
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
		
		verbose(mi_jidt, mi_python, i, "CMI (uncorrelated source)", local=True, atol=1e-03)
		
		
	print("\nmean calculation times:")
	print(" JidtDiscreteCMI: ", np.mean(time_jidt_cor) )
	print(" PythonDiscreteCMI: ", np.mean(time_python_cor) )


	print(f"\n\nTesting local CMI using 2D mute data  - uncorrelated \nconditional and uncorrelated source")
	print(f"testing settings n_discrete_bins {vals} and discrete_method max_ent and equal")
	
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
		
		print(mi_jidt[:10])
		print(mi_python[:10])

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
		
		verbose(mi_jidt, mi_python, i, "CMI (uncorrelated source)", local=True)
		
		
	print("\nmean calculation times:")
	print(" JidtDiscreteCMI: ", np.mean(time_jidt_cor) )
	print(" PythonDiscreteCMI: ", np.mean(time_python_cor) )

def test_discrete_ais():
	

	hvals = [1,2,3]
	nvals = [2,4,8]
	tvals = [1,2,3]
	
	print(f"\n\nTesting average AIS using 1D AR with history and noise")
	print(f"testing settings history {hvals},  and tau {tvals}, n_discrete_bins {nvals} and discrete_method max_ent\n")
	
	source1, source2 = _get_ar_data(seed=SEED)

	time_jidt_cor = np.zeros(np.power(len(nvals),3))
	res_jidt_cor = np.zeros(np.power(len(nvals),3))
	time_python_cor = np.zeros(np.power(len(nvals),3))
	res_python_cor = np.zeros(np.power(len(nvals),3))
	time_jidt_uncor = np.zeros(np.power(len(nvals),3))
	res_jidt_uncor = np.zeros(np.power(len(nvals),3))
	time_python_uncor = np.zeros(np.power(len(nvals),3))
	res_python_uncor = np.zeros(np.power(len(nvals),3))
	conds = np.empty((np.power(len(nvals),3),3))

	count = 0
	for h in hvals:
		for t in tvals:
			for i in nvals:
			
					conds[count,:] = [h, t, i]

					settings_j = {'history': h,
								'tau': t,
								'discretise_method': 'max_ent',
								'n_discrete_bins': i}

					settings_p = {'history': h, 
								'tau': t,
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

	print("hist, tau, bins\tJidtDiscreteAIS\t\tPythonDiscreteAIS\t close 0.01")
	print("correlated")
	count = 0
	for i in range(len(res_jidt_cor)):
		print(f"{conds[i,:]}\t{res_jidt_cor[i]}\t{res_python_cor[i]}\t{np.isclose(res_jidt_cor[i], res_python_cor[i] ,atol=0.01)}")
		count += 1
	
	print("uncorrelated")
	count = 0
	for i in range(len(res_jidt_uncor)):
		print(f"{conds[i,:]}\t{res_jidt_uncor[i]}\t{res_python_uncor[i]}\t{np.isclose(res_jidt_uncor[i], res_python_uncor[i] ,atol=0.01)}")
		count += 1

	verbose(res_jidt_cor, res_python_cor, "with history", "AIS", atol=5e-03)
	verbose(res_jidt_uncor, res_python_uncor, "noise", "AIS", atol=5e-03)

	print("\nmean calculation times:")
	print(" JidtDiscreteAIS (with history): ", np.mean(time_jidt_cor) )
	print(" PythonDiscreteAIS (with history): ", np.mean(time_python_cor) )
	print(" JidtDiscreteAIS (noise): ", np.mean(time_jidt_uncor) )
	print(" PythonDiscreteAIS (noise): ", np.mean(time_python_uncor) )





def test_discrete_ais_local_values():
	
	atol = 0.08

	hvals = [1,2,3]
	nvals = [2,4,6]
	tvals = [1,2,3]
	
	print(f"\n\nTesting local AIS using 1D AR with history and noise")
	print(f"testing settings history {hvals},  and tau {tvals}, n_discrete_bins {nvals} and discrete_method max_ent\n")
	
	source1, source2 = _get_ar_data(seed=SEED+1)
	
	min_len = min(len(source1),len(source2))
	source1 = source1[:min_len]
	source2 = source2[:min_len]


	print(min_len)


	time_jidt_cor = np.zeros(np.power(len(nvals),3))
	res_jidt_cor = np.zeros(np.power(len(nvals),3))
	time_python_cor = np.zeros(np.power(len(nvals),3))
	res_python_cor = np.zeros(np.power(len(nvals),3))
	time_jidt_uncor = np.zeros(np.power(len(nvals),3))
	res_jidt_uncor = np.zeros(np.power(len(nvals),3))
	time_python_uncor = np.zeros(np.power(len(nvals),3))
	res_python_uncor = np.zeros(np.power(len(nvals),3))
	conds = np.empty((np.power(len(nvals),3),3))

	print("hist, tau, bins\tJidtDiscreteAIS vs PythonDiscreteAIS")
	count = 0
	for h in hvals:
		for t in tvals:
			for i in nvals:
			
				conds[count,:] = [h, t, i]
				settings = {}
				settings_j = {'history': h,
							'tau': t,
							'discretise_method': 'max_ent',
							'n_discrete_bins': i,
							'local_values': True}
				settings_p = {'history': h, 
							'tau': t,
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
				
				verbose(res_jidt_cor, res_python_cor, f"{conds[count,:]} - with hist ", "AIS", local=True, atol=atol)
				verbose(res_jidt_uncor, res_python_uncor, f"{conds[count,:]} - noise\t", "AIS", local=True, atol=atol)

				count += 1
		
	print("\nmean calculation times:")
	print(" JidtKraskovAIS (with history): ", np.mean(time_jidt_cor) )
	print(" PythonKraskovAIS (with history): ", np.mean(time_python_cor) )
	print(" JidtKraskovAIS (noise): ", np.mean(time_jidt_uncor) )
	print(" PythonKraskovAIS (noise): ", np.mean(time_python_uncor) )

def test_discrete_te():

	vals = [1,2,3]
	nvals = [2,4,6]

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
	
	#time_jidt = 0.0
	#time_python = 0.0
	
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
										'discretise_method': 'max_ent',
										'n_discrete_bins': n}

							settings_p = {"history_target": ht,
										"history_source": hs,
										"tau_target": tt,
										"tau_source": ts,
										"source_target_delay": hst,
										'discretise_method': 'max_ent',
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

	vals = [1,2,3]

	print(f"\n\nTesting average TE using 1D binary data with memory\n")
	print(f"testing settings history_source (hs), tau_source (ts), history_target (ht), \ntau_target (tt), source_target_delay (std) with {vals} each.\nand n_discrete_bins 2\n")

	#expected_mi, source1, source2, target = _get_gauss_data(expand=True, seed=SEED)
	#source1 = source1[1:]
	#source2 = source2[1:]
	#target = target[:-1]

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
									#'discretise_method': 'max_ent',
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
									'opt_source_hist_tau': False,
									#'discretise_method': 'max_ent',
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



# Test Spectral estimators
def test_spectral_mi():
	# test different estimator settings for PythonSpectralMI
	Hz = 40
	lag = 30
	noise = 0.2

	source1, target, source2 = _get_freq_data(sample_rate=10000, duration=1.0, hz=Hz, lag=lag, noise=noise, seed=SEED)
		
	# test different estimator settings
	print(f"\nTest different estimator types of SpectralMI on {Hz} Hz freq data with noise {noise} and lag {lag}")
	
	estimators = ['kraskov','discrete','gaussian','none']
	
	vals = [0,10,20,30,40,50,60,70]

	expected_best = vals.index(lag)
	
	mi_python_cor = np.zeros([len(estimators),len(vals)])
	mi_python_uncor = np.zeros([len(estimators),len(vals)])
	time_python_cor = np.zeros([len(estimators),len(vals)])
	time_python_uncor = np.zeros([len(estimators),len(vals)])

	ecount = 0
	for e in estimators:

		lcount=0
		for lags in vals:
			settings = {"estimator": e,
						"lag_mi": lags,
						}

			python_estimator = PythonSpectralMI(settings)
			itic = time.perf_counter()
			mi_python_cor[ecount,lcount] = python_estimator.estimate(source1, target)
			itoc = time.perf_counter()
			time_python_cor[ecount,lcount] = itoc - itic
			
			python_estimator = PythonSpectralMI(settings)
			itic = time.perf_counter()
			mi_python_uncor[ecount,lcount] = python_estimator.estimate(source1, source2)
			itoc = time.perf_counter()
			time_python_uncor[ecount,lcount] = itoc - itic

			lcount += 1

		ecount += 1

	print(f"Summary PythonSpectralMI lags ({vals}):")

	print("MI values:")
	print("correlated data:")
	print("lag\t\tkraskov\t\tdiscrete\tgaussian\tnone")
	for i in range(len(vals)):
		print(f"{vals[i]}\t{mi_python_cor[0,i]}\t{mi_python_cor[1,i]}\t{mi_python_cor[2,i]}\t{mi_python_cor[3,i]}")
	
	test = []
	for e in range(len(estimators)):
		test.append(expected_best==np.argmax(mi_python_cor[e,:]))
	print(f"found expected lag:\t{test[0]}\t\t{test[1]}\t\t\t{test[2]}\t\t\t{test[3]}")


	print(f"\nuncorrelated data {Hz*0.7} Hz:")
	print("lag\t\tkraskov\t\tdiscrete\tgaussian\tnone")
	for i in range(len(vals)):
		print(f"{vals[i]}\t{mi_python_uncor[0,i]}\t{mi_python_uncor[1,i]}\t{mi_python_uncor[2,i]}\t{mi_python_uncor[3,i]}")
	

	print("\nmean calculation times:")
	print(" PythonSpectralMI kraskov (cor): ", np.mean(time_python_cor[0,:]) )
	print(" PythonSpectralMI discrete (cor): ", np.mean(time_python_cor[1,:]) )
	print(" PythonSpectralMI gaussian (cor): ", np.mean(time_python_cor[2,:]) )
	print(" PythonSpectralMI none (cor): ", np.mean(time_python_cor[3,:]) )
	print(" PythonSpectralMI kraskov (uncor): ", np.mean(time_python_uncor[0,:]) )
	print(" PythonSpectralMI discrete (uncor): ", np.mean(time_python_uncor[1,:]) )
	print(" PythonSpectralMI gaussian (uncor): ", np.mean(time_python_uncor[2,:]) )
	print(" PythonSpectralMI none (uncor): ", np.mean(time_python_uncor[3,:]) )
	




	print("\n\nTest Spectral vs standard MI estimators with orig data")

	# kraskov
	mi_jidt = np.zeros(len(vals))
	time_jidt = np.zeros(len(vals))
	mi_python = np.zeros(len(vals))
	time_python = np.zeros(len(vals))
	mi_python2 = np.zeros(len(vals))
	time_python2 = np.zeros(len(vals))
	
	lcount=0
	for lags in vals:
		settings = {"estimator": 'kraskov',
					"lag_mi": lags,
					}

		python_estimator = PythonSpectralMI(settings)
		itic = time.perf_counter()
		mi_python[lcount] = python_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_python[lcount] = itoc - itic
		
		python_estimator2 = PythonKraskovMI({"lag_mi": lags})
		itic = time.perf_counter()
		mi_python2[lcount] = python_estimator2.estimate(source1, target)
		itoc = time.perf_counter()
		time_python2[lcount] = itoc - itic
		
		jidt_estimator = JidtKraskovMI({"lag_mi": lags})
		itic = time.perf_counter()
		mi_jidt[lcount] = jidt_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_jidt[lcount] = itoc - itic

		lcount += 1

	print(f"\n\nSummary PythonSpectralMI kraskov vs JidtKraskovMI vs PythonKraskovMI:")

	print("MI values:")
	print("correlated data:")
	print("lag\tPythonSpectralMI\t\tJidtKraskovMI\tPythonKraskovMI")
	for i in range(len(vals)):
		print(f"{vals[i]}\t{mi_python[i]}\t{mi_jidt[i]}\t{mi_python2[i]}")

	print(f"found expected lag:\t{expected_best==np.argmax(mi_python)}\t\t{expected_best==np.argmax(mi_jidt)}\t\t\t{expected_best==np.argmax(mi_python2)}")

	print("\nmean calculation times:")
	print(" PythonSpectralMI kraskov: ", np.mean(time_python) )
	print(" JidtKraskovMI: ", np.mean(time_jidt) )
	print(" PythonKraskovlMI: ", np.mean(time_python2) )
	
	# discrete
	mi_jidt = np.zeros(len(vals))
	time_jidt = np.zeros(len(vals))
	mi_python = np.zeros(len(vals))
	time_python = np.zeros(len(vals))
	mi_python2 = np.zeros(len(vals))
	time_python2 = np.zeros(len(vals))
	

	lcount=0
	for lags in vals:
		settings = {"estimator": 'discrete',
					"lag_mi": lags,
					"discretise_method": 'max_ent'}

		python_estimator = PythonSpectralMI(settings)
		itic = time.perf_counter()
		mi_python[lcount] = python_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_python[lcount] = itoc - itic
		
		python_estimator2 = PythonDiscreteMI({"lag_mi": lags, "discretise_method": 'max_ent'})
		itic = time.perf_counter()
		mi_python2[lcount] = python_estimator2.estimate(source1, target)
		itoc = time.perf_counter()
		time_python2[lcount] = itoc - itic
		
		jidt_estimator = JidtDiscreteMI({"lag_mi": lags, "discretise_method": 'max_ent'})
		itic = time.perf_counter()
		mi_jidt[lcount] = jidt_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_jidt[lcount] = itoc - itic

		lcount += 1

	print(f"\n\nSummary PythonSpectralMI discrete vs JidtDiscreteMI vs PythonDiscreteMI:")

	print("MI values:")
	print("correlated data:")
	print("lag\tPythonSpectralMI\t\tJidtDiscreteMI\tPythonDiscreteMI")
	for i in range(len(vals)):
		print(f"{vals[i]}\t{mi_python[i]}\t{mi_jidt[i]}\t{mi_python2[i]}")
	
	print(f"found expected lag:\t{expected_best==np.argmax(mi_python)}\t\t{expected_best==np.argmax(mi_jidt)}\t\t\t{expected_best==np.argmax(mi_python2)}")
	print("\nmean calculation times:")
	print(" PythonSpectralMI discrete: ", np.mean(time_python) )
	print(" JidtDiscreteMI: ", np.mean(time_jidt) )
	print(" PythonDiscreteMI: ", np.mean(time_python2) )
	
	# gaussian
	mi_jidt = np.zeros(len(vals))
	time_jidt = np.zeros(len(vals))
	mi_python = np.zeros(len(vals))
	time_python = np.zeros(len(vals))
	mi_python2 = np.zeros(len(vals))
	time_python2 = np.zeros(len(vals))
	

	lcount=0
	for lags in vals:
		settings = {"estimator": 'gaussian',
					"lag_mi": lags,
					}

		python_estimator = PythonSpectralMI(settings)
		itic = time.perf_counter()
		mi_python[lcount] = python_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_python[lcount] = itoc - itic
		
		python_estimator2 = PythonGaussianMI({"lag_mi": lags})
		itic = time.perf_counter()
		mi_python2[lcount] = python_estimator2.estimate(source1, target)
		itoc = time.perf_counter()
		time_python2[lcount] = itoc - itic
		
		jidt_estimator = JidtGaussianMI({"lag_mi": lags})
		itic = time.perf_counter()
		mi_jidt[lcount] = jidt_estimator.estimate(source1, target)
		itoc = time.perf_counter()
		time_jidt[lcount] = itoc - itic

		lcount += 1

	print(f"\n\nSummary PythonSpectralMI gaussian vs JidtGaussianMI vs PythonGaussianMI:")

	print("MI values:")
	print("correlated data:")
	print("lag\tPythonSpectralMI\t\tJidtGaussianMI\tPythonGaussianMI")
	for i in range(len(vals)):
		print(f"{vals[i]}\t{mi_python[i]}\t{mi_jidt[i]}\t{mi_python2[i]}")
	print(f"found expected lag:\t{expected_best==np.argmax(mi_python)}\t\t{expected_best==np.argmax(mi_jidt)}\t\t\t{expected_best==np.argmax(mi_python2)}")
	print("\nmean calculation times:")
	print(" PythonSpectralMI gaussian: ", np.mean(time_python) )
	print(" JidtGaussianMI: ", np.mean(time_jidt) )
	print(" PythonGaussianMI: ", np.mean(time_python2) )



	# test 2D input
	print("\ntest 2D data input (n,2)")
	
	source1x, targetx, source2x = _get_freq_data(sample_rate=10000, duration=1.0, hz=Hz, lag=lag, noise=noise, seed=SEED+1)
	
	n = source1.shape[0]
	shift = 0
	source1_2d = np.concatenate([source1[0:n-shift,None], source1x[shift:n,None]], axis=1)
	source2_2d = np.concatenate([source2[0:n-shift,None], source2x[shift:n,None]], axis=1)
	target_2d = np.concatenate([target[0:n-shift,None], targetx[shift:n,None]], axis=1)
	
	mi_python_cor = np.zeros([len(estimators),len(vals)])
	mi_python_uncor = np.zeros([len(estimators),len(vals)])
	time_python_cor = np.zeros([len(estimators),len(vals)])
	time_python_uncor = np.zeros([len(estimators),len(vals)])

	ecount = 0
	for e in estimators:

		lcount=0
		for lags in vals:
			settings = {"estimator": e,
						"lag_mi": lags}

			python_estimator = PythonSpectralMI(settings)
			itic = time.perf_counter()
			mi_python_cor[ecount,lcount] = python_estimator.estimate(source1_2d, target_2d)
			itoc = time.perf_counter()
			time_python_cor[ecount,lcount] = itoc - itic
			
			python_estimator = PythonSpectralMI(settings)
			itic = time.perf_counter()
			mi_python_uncor[ecount,lcount] = python_estimator.estimate(source1_2d, source2_2d)
			itoc = time.perf_counter()
			time_python_uncor[ecount,lcount] = itoc - itic

			lcount += 1

		ecount += 1

	print(f"Summary PythonSpectralMI lags ({vals}):")

	print("MI values:")
	print("correlated data:")
	print("lag\t\tkraskov\t\tdiscrete\tgaussian\tnone")
	for i in range(len(vals)):
		print(f"{vals[i]}\t{mi_python_cor[0,i]}\t{mi_python_cor[1,i]}\t{mi_python_cor[2,i]}\t{mi_python_cor[3,i]}")
	
	test = []
	for e in range(len(estimators)):
		test.append(expected_best==np.argmax(mi_python_cor[e,:]))
	print(f"found expected lag:\t{test[0]}\t\t{test[1]}\t\t\t{test[2]}\t\t\t{test[3]}")


	print(f"\nuncorrelated data {Hz*0.7} Hz:")
	print("lag\t\tkraskov\t\tdiscrete\tgaussian\tnone")
	for i in range(len(vals)):
		print(f"{vals[i]}\t{mi_python_uncor[0,i]}\t{mi_python_uncor[1,i]}\t{mi_python_uncor[2,i]}\t{mi_python_uncor[3,i]}")
	

	print("\nmean calculation times:")
	print(" PythonSpectralMI kraskov (cor): ", np.mean(time_python_cor[0,:]) )
	print(" PythonSpectralMI discrete (cor): ", np.mean(time_python_cor[1,:]) )
	print(" PythonSpectralMI gaussian (cor): ", np.mean(time_python_cor[2,:]) )
	print(" PythonSpectralMI none (cor): ", np.mean(time_python_cor[3,:]) )
	print(" PythonSpectralMI kraskov (uncor): ", np.mean(time_python_uncor[0,:]) )
	print(" PythonSpectralMI discrete (uncor): ", np.mean(time_python_uncor[1,:]) )
	print(" PythonSpectralMI gaussian (uncor): ", np.mean(time_python_uncor[2,:]) )
	print(" PythonSpectralMI none (uncor): ", np.mean(time_python_uncor[3,:]) )

def test_spectral_mi_local_values():
	# test different estimator settings for PythonSpectralCMI local values
	Hz = 40
	lag = 30
	noise = 0.2

	source1, target, source2 = _get_freq_data(sample_rate=10000, duration=1.0, hz=Hz, lag=lag, noise=noise, seed=SEED)
		
	# test different estimator settings
	print(f"\nTest different estimator types of SpectralMI (mean of local values) on {Hz} Hz freq data with noise {noise} and lag {lag}")
	
	estimators = ['kraskov','discrete','gaussian']
	
	vals = [0,10,20,30,40,50,60,70]

	expected_best = vals.index(lag)
	
	mi_python_cor = np.zeros([len(estimators),len(vals)])
	mi_python_uncor = np.zeros([len(estimators),len(vals)])
	time_python_cor = np.zeros([len(estimators),len(vals)])
	time_python_uncor = np.zeros([len(estimators),len(vals)])

	ecount = 0
	for e in estimators:

		lcount=0
		for lags in vals:
			settings = {"estimator": e,
						"lag_mi": lags,
						'local_values': True
						}

			python_estimator = PythonSpectralMI(settings)
			itic = time.perf_counter()
			mi_python_cor[ecount,lcount] = np.mean(python_estimator.estimate(source1, target))
			itoc = time.perf_counter()
			time_python_cor[ecount,lcount] = itoc - itic
			
			python_estimator = PythonSpectralMI(settings)
			itic = time.perf_counter()
			mi_python_uncor[ecount,lcount] = np.mean(python_estimator.estimate(source1, source2))
			itoc = time.perf_counter()
			time_python_uncor[ecount,lcount] = itoc - itic

			lcount += 1

		ecount += 1

	print(f"Summary PythonSpectralMI (mean of local values) lags ({vals}):")

	print("MI values:")
	print("correlated data:")
	print("lag\t\tkraskov\t\tdiscrete\tgaussian")
	for i in range(len(vals)):
		print(f"{vals[i]}\t{mi_python_cor[0,i]}\t{mi_python_cor[1,i]}\t{mi_python_cor[2,i]}")
	
	test = []
	for e in range(len(estimators)):
		test.append(expected_best==np.argmax(mi_python_cor[e,:]))
	print(f"found expected lag:\t{test[0]}\t\t{test[1]}\t\t\t{test[2]}")


	print(f"\nuncorrelated data {Hz*0.7} Hz:")
	print("lag\t\tkraskov\t\tdiscrete\tgaussian")
	for i in range(len(vals)):
		print(f"{vals[i]}\t{mi_python_uncor[0,i]}\t{mi_python_uncor[1,i]}\t{mi_python_uncor[2,i]}")
	

	print("\nmean calculation times:")
	print(" PythonSpectralMI kraskov (cor): ", np.mean(time_python_cor[0,:]) )
	print(" PythonSpectralMI discrete (cor): ", np.mean(time_python_cor[1,:]) )
	print(" PythonSpectralMI gaussian (cor): ", np.mean(time_python_cor[2,:]) )
	print(" PythonSpectralMI kraskov (uncor): ", np.mean(time_python_uncor[0,:]) )
	print(" PythonSpectralMI discrete (uncor): ", np.mean(time_python_uncor[1,:]) )
	print(" PythonSpectralMI gaussian (uncor): ", np.mean(time_python_uncor[2,:]) )

def test_spectral_cmi():
	# test different estimator settings for PythonSpectralCMI
	print(f"\nTest different estimator types of SpectralCMI on 40 Hz freq data \nwith noise 0.2 and no lag")
	
	Hz = 40
	lag = 0
	noise = 0.2
	
	estimators = ['kraskov','discrete','gaussian']
	
	# test different estimator settings
	print(f"\nTest different estimator types of SpectralCMI on {Hz} Hz freq data \nwith noise {noise} and no lag")
	
	mi_python_cor = np.zeros(len(estimators))
	mi_python_uncor = np.zeros(len(estimators))
	time_python_cor = np.zeros(len(estimators))
	time_python_uncor = np.zeros(len(estimators))
	
	source1, target, source2 = _get_freq_data(sample_rate=10000, duration=1.0, hz=Hz, lag=lag, noise=noise, seed=SEED)

	
	ecount = 0
	for e in estimators:
		settings = {"estimator": e}

		python_estimator = PythonSpectralCMI(settings)
		itic = time.perf_counter()
		mi_python_cor[ecount] = python_estimator.estimate(source1, target, source2)
		itoc = time.perf_counter()
		time_python_cor[ecount] = itoc - itic
		
		python_estimator = PythonSpectralCMI(settings)
		itic = time.perf_counter()
		mi_python_uncor[ecount] = python_estimator.estimate(source2, target, source1)
		itoc = time.perf_counter()
		time_python_uncor[ecount] = itoc - itic

		ecount += 1
		
	print(f"Summary PythonSpectralCMI:")
	print("CMI values:")
	print("uncorrelated conditional:")
	print("\tkraskov\t\tdiscrete\t\tgaussian")
	print(f"{mi_python_cor[0]}\t{mi_python_cor[1]}\t{mi_python_cor[2]}")
	
	print(f"\nuncorrelated source:")
	print("\tkraskov\t\tdiscrete\t\tgaussian")
	print(f"{mi_python_uncor[0]}\t{mi_python_uncor[1]}\t{mi_python_uncor[2]}")

	print("\nmean calculation times:")
	print(" PythonSpectralMI kraskov (cor): ", np.mean(time_python_cor[0]) )
	print(" PythonSpectralMI discrete (cor): ", np.mean(time_python_cor[1]) )
	print(" PythonSpectralMI gaussian (cor): ", np.mean(time_python_cor[2]) )
	print(" PythonSpectralMI kraskov (uncor): ", np.mean(time_python_uncor[0]) )
	print(" PythonSpectralMI discrete (uncor): ", np.mean(time_python_uncor[1]) )
	print(" PythonSpectralMI gaussian (uncor): ", np.mean(time_python_uncor[2]) )
	

	################################################################################ TODO
	print("\n\nTest Spectral vs standard MI estimators with orig data")


	


if __name__ == '__main__':

	################################ TODO:
	# AIS und TE optimization weg
	# get_dist nur gaussian testen für gaussian estimator? 
	# 			multivariate gaussian mit flag is_tested dann Analytic für Gaussian zulassen.
	# 
    
    # Gaussian
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
	
	# test opt_source  ################################################## TODO
	
	testhead("GaussianTE local values")
	test_gaussian_te_local_values()
	"""
	#testhead("GaussianCTE")
	#test_gaussian_cte()
	
	#testhead("GaussianCTE local values") ################################################## TODO
	#test_gaussian_cte_local_values()
	

	##																	check theiler_T and kraskov string instead int
	## 																	check theiler _T >3
	"""	
	testhead("KraskovMI") ########################################## TODO mixed dim
	test_kraskov_mi()
	
	testhead("KraskovMI local values")
	test_kraskov_mi_local_values()
	
	testhead("KraskovCMI") ########################################## TODO mixed dim
	test_kraskov_cmi()
	
	testhead("KraskovCMI local values")
	test_kraskov_cmi_local_values()
	
	testhead("KraskovAIS")
	test_kraskov_ais()
	
	testhead("KraskovAIS local values") ################################################## TODO
	test_kraskov_ais_local_values()
	
	testhead("KraskovTE")
	test_kraskov_te()
	
	# test opt_source  ################################################## TODO

	testhead("KraskovTE local values")
	test_kraskov_te_local_values()
	"""
	
	#testhead("KraskovCTE")
	#test_Kraskov_cte()
	
	#testhead("GaussianCTE local values") ################################################## TODO
	#test_gaussian_cte_local_values()
	
	
	# Discrete
	
	################################################################ TODO return_calc for all

	#testhead("DiscreteMI")
	#test_discrete_mi()

	#testhead("DiscreteMI local values") ################################################## TODO 2D data
	#test_discrete_mi_local_values()

	#testhead("DiscreteCMI")
	#test_discrete_cmi()

	#testhead("DiscreteCMI local values") ################################################## TODO 2D data
	#test_discrete_cmi_local_values()
	
	#testhead("DiscreteAIS")
	#test_discrete_ais()

	#testhead("DiscreteAIS local values") ############################################## TODO
	#test_discrete_ais_local_values()

	#testhead("DiscreteTE")
	#test_discrete_te()

	# test opt source ############################################## TODO

	#testhead("DiscreteTE local values") ############################################## TODO
	#test_discrete_te_local_values()
	








	#testhead("SpectralMI")
	#test_spectral_mi()
	############################################ TODO 2D performence dest data with or without lags new data 


	#testhead("SpectralMI local values")
	#test_spectral_mi_local_values()

	#testhead("SpectralCMI")
	#test_spectral_cmi()
	############################################ TODO 2D performence dest data with or without lags new data 


	#testhead("SpectralCMI local values") ############################################## TODO
	#test_spectral_cmi_local_values()

	#testhead("SpectralAIS") ############################################## TODO
	#test_spectral_ais()

	#testhead("SpectralAIS local values") ############################################## TODO
	#test_spectral_mi_local_values()

	#testhead("SpectralTE") ############################################## TODO
	#test_spectral_te()

	#testhead("SpectralTE local values") ############################################## TODO
	#test_spectral_te_local_values()
