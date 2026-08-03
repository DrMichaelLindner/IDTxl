""" provide tests for Python Spectral estimators """


import numpy as np

import time
import sys
import copy

from idtxl.estimators_python import (PythonKraskovMI, PythonKraskovCMI, PythonKraskovAIS, PythonKraskovTE, PythonKraskovCTE, 
									PythonGaussianMI, PythonGaussianCMI, PythonGaussianTE, PythonGaussianCTE, PythonGaussianAIS, 
									PythonDiscreteMI, PythonDiscreteCMI, PythonDiscreteAIS, PythonDiscreteTE, 
									PythonSpectralMI, PythonSpectralCMI, PythonSpectralAIS, PythonSpectralTE)


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
	print("\n#######################################################################")
	print(f"\n            Compare {est}:\n")
	print("#######################################################################")




#### Test Spectral estimators
def test_spectral_mi():

	# test different estimator settings for PythonSpectralMI
	Hz = 40
	lag = 30
	noise = 0.2

	estimators = ['kraskov','discrete','gaussian','none']
	
	vals = [0,10,20,30,40,50,60,70]

	source1, target, source2 = _get_freq_data(sample_rate=10000, duration=1.0, hz=Hz, lag=lag, noise=noise, seed=SEED)
	
	# test different estimator settings
	print(f"\nTest average SpectralMI on 1D {Hz} Hz freq data noise {noise} and lag {lag}")
	print(f"different estimator type settings {estimators} and lag_mi {vals}\n")
	
	
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

	print(f"Summary PythonSpectralMI lags ({vals}) (1D data):")

	print("MI values:")
	print("correlated data:")
	print("lag\tkraskov\t\t\tdiscrete\t\tgaussian\t\tnone")
	for i in range(len(vals)):
		print(f"{vals[i]}\t{mi_python_cor[0,i]}\t{mi_python_cor[1,i]}\t{mi_python_cor[2,i]}\t{mi_python_cor[3,i]}")
	
	test = []
	for e in range(len(estimators)):
		test.append(expected_best==np.argmax(mi_python_cor[e,:]))
	print(f"found expected lag:\t{test[0]}\t\t{test[1]}\t\t\t{test[2]}\t\t\t{test[3]}")


	print(f"\nuncorrelated data var2 ={Hz*0.7} Hz:")
	print("lag\tkraskov\t\t\tdiscrete\t\tgaussian\t\tnone")
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

	# Test 2D data input
	reps = 10
	source1_2D = np.zeros((source1.shape[0], reps))
	target_2D = np.zeros((source1.shape[0], reps))
	source2_2D = np.zeros((source1.shape[0], reps))
	
	for i in range(reps):
		source1_2D[:,i], target_2D[:,i], source2_2D[:,i] = _get_freq_data(sample_rate=10000, duration=1.0, hz=Hz, lag=lag, noise=noise)#, seed=SEED)
		
	print(f"\nTest average SpectralMI on 2D {Hz} Hz freq data noise {noise} and lag {lag}")
	print(f"different estimator type settings {estimators} and lag_mi {vals}\n")
	
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
			mi_python_cor[ecount,lcount] = python_estimator.estimate(source1_2D, target_2D)
			itoc = time.perf_counter()
			time_python_cor[ecount,lcount] = itoc - itic
			
			python_estimator = PythonSpectralMI(settings)
			itic = time.perf_counter()
			mi_python_uncor[ecount,lcount] = python_estimator.estimate(source1_2D, source2_2D)
			itoc = time.perf_counter()
			time_python_uncor[ecount,lcount] = itoc - itic

			lcount += 1

		ecount += 1

	print(f"Summary PythonSpectralMI lags ({vals}) (2D data):")

	print("MI values:")
	print("correlated data:")
	print("lag\tkraskov\t\t\tdiscrete\t\tgaussian\t\tnone")
	for i in range(len(vals)):
		print(f"{vals[i]}\t{mi_python_cor[0,i]}\t{mi_python_cor[1,i]}\t{mi_python_cor[2,i]}\t{mi_python_cor[3,i]}")
	
	test = []
	for e in range(len(estimators)):
		test.append(expected_best==np.argmax(mi_python_cor[e,:]))
	print(f"found expected lag:\t{test[0]}\t\t{test[1]}\t\t\t{test[2]}\t\t\t{test[3]}")


	print(f"\nuncorrelated data var2 ={Hz*0.7} Hz:")
	print("lag\tkraskov\t\t\tdiscrete\t\tgaussian\t\tnone")
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
	

	# test spectral vs other estimators
	print(f"\n\nTest Spectral vs standard MI estimators with orig data {Hz} Hz freq data \nwith noise {noise} and lag {lag}")
	print(f"different estimator setting {estimators} and lag_mi {vals}\n")
	
	# kraskov
	mi_jidt = np.zeros(len(vals))
	time_jidt = np.zeros(len(vals))
	mi_python = np.zeros(len(vals))
	time_python = np.zeros(len(vals))
	mi_python2 = np.zeros(len(vals))
	time_python2 = np.zeros(len(vals))
	
	print("- MI estimator: kraskov")
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

	print(f"\n\nSummary PythonSpectralMI 'kraskov' vs JidtKraskovMI vs PythonKraskovMI:")

	print("MI values:")
	print("correlated data:")
	print("lag\tPythonSpectralMI\tJidtKraskovMI\t\tPythonKraskovMI")
	for i in range(len(vals)):
		print(f"{vals[i]}\t{mi_python[i]}\t{mi_jidt[i]}\t{mi_python2[i]}")

	print(f"found expected lag:\t{expected_best==np.argmax(mi_python)}\t\t{expected_best==np.argmax(mi_jidt)}\t\t\t{expected_best==np.argmax(mi_python2)}")

	print("\nmean calculation times:")
	print(" PythonSpectralMI kraskov: ", np.mean(time_python) )
	print(" JidtKraskovMI: ", np.mean(time_jidt) )
	print(" PythonKraskovlMI: ", np.mean(time_python2) )
	
	print("\n\n- MI estimator: discrete")
	
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
	print("lag\tPythonSpectralMI\tJidtDiscreteMI\t\tPythonDiscreteMI")
	for i in range(len(vals)):
		print(f"{vals[i]}\t{mi_python[i]}\t{mi_jidt[i]}\t{mi_python2[i]}")
	
	print(f"found expected lag:\t{expected_best==np.argmax(mi_python)}\t\t{expected_best==np.argmax(mi_jidt)}\t\t\t{expected_best==np.argmax(mi_python2)}")
	print("\nmean calculation times:")
	print(" PythonSpectralMI discrete: ", np.mean(time_python) )
	print(" JidtDiscreteMI: ", np.mean(time_jidt) )
	print(" PythonDiscreteMI: ", np.mean(time_python2) )
	
	print("\n\n- MI estimator: gaussian")
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
	print("lag\tPythonSpectralMI\tJidtGaussianMI\t\tPythonGaussianMI")
	for i in range(len(vals)):
		print(f"{vals[i]}\t{mi_python[i]}\t{mi_jidt[i]}\t{mi_python2[i]}")
	print(f"found expected lag:\t{expected_best==np.argmax(mi_python)}\t\t{expected_best==np.argmax(mi_jidt)}\t\t\t{expected_best==np.argmax(mi_python2)}")
	print("\nmean calculation times:")
	print(" PythonSpectralMI gaussian: ", np.mean(time_python) )
	print(" JidtGaussianMI: ", np.mean(time_jidt) )
	print(" PythonGaussianMI: ", np.mean(time_python2) )

def test_spectral_mi_local_values():
	# test different estimator settings for PythonSpectralCMI local values
	Hz = 40
	lag = 30
	noise = 0.2
	
	estimators = ['kraskov','discrete','gaussian']
	
	vals = [0,10,20,30,40,50,60,70]

	source1, target, source2 = _get_freq_data(sample_rate=10000, duration=1.0, hz=Hz, lag=lag, noise=noise, seed=SEED)
		
	# test different estimator settings
	print(f"\nTest SpectralMI (mean of local values) on 1D {Hz} Hz freq data noise {noise} and lag {lag}")
	print(f"different estimator type settings {estimators} and lag_mi {vals}\n")
	
	
	expected_best = vals.index(lag)
	
	mi_python_cor = np.zeros([len(estimators),len(vals)])
	mi_python_uncor = np.zeros([len(estimators),len(vals)])
	time_python_cor = np.zeros([len(estimators),len(vals)])
	time_python_uncor = np.zeros([len(estimators),len(vals)])

	mi_close_cor = np.zeros([len(estimators),len(vals)])
	mi_close_uncor = np.zeros([len(estimators),len(vals)])
	
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

	print(f"Summary PythonSpectralMI (mean of local values) lags ({vals}) (1D data):")

	print("MI values:")
	print("correlated data:")
	print("lag\t\tkraskov\t\tdiscrete\t\tgaussian")
	for i in range(len(vals)):
		print(f"{vals[i]}\t{mi_python_cor[0,i]}\t{mi_python_cor[1,i]}\t{mi_python_cor[2,i]}")
	
	test = []
	for e in range(len(estimators)):
		test.append(expected_best==np.argmax(mi_python_cor[e,:]))
	print(f"found expected lag:\t{test[0]}\t\t{test[1]}\t\t\t{test[2]}")


	print(f"\nuncorrelated data var2 = {Hz*0.7} Hz:")
	print("lag\t\tkraskov\t\tdiscrete\t\tgaussian")
	for i in range(len(vals)):
		print(f"{vals[i]}\t{mi_python_uncor[0,i]}\t{mi_python_uncor[1,i]}\t{mi_python_uncor[2,i]}")
	

	print("\nmean calculation times:")
	print(" PythonSpectralMI kraskov (cor): ", np.mean(time_python_cor[0,:]) )
	print(" PythonSpectralMI discrete (cor): ", np.mean(time_python_cor[1,:]) )
	print(" PythonSpectralMI gaussian (cor): ", np.mean(time_python_cor[2,:]) )
	print(" PythonSpectralMI kraskov (uncor): ", np.mean(time_python_uncor[0,:]) )
	print(" PythonSpectralMI discrete (uncor): ", np.mean(time_python_uncor[1,:]) )
	print(" PythonSpectralMI gaussian (uncor): ", np.mean(time_python_uncor[2,:]) )




	# Test 2D data input
	reps = 10
	source1_2D = np.zeros((source1.shape[0], reps))
	target_2D = np.zeros((source1.shape[0], reps))
	source2_2D = np.zeros((source1.shape[0], reps))
	
	for i in range(reps):
		source1_2D[:,i], target_2D[:,i], source2_2D[:,i] = _get_freq_data(sample_rate=10000, duration=1.0, hz=Hz, lag=lag, noise=noise)#, seed=SEED)
		
	print(f"\nTest average SpectralMI on 2D {Hz} Hz freq data noise {noise} and lag {lag}")
	print(f"different estimator type settings {estimators} and lag_mi {vals}\n")
	
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
			mi_python_cor[ecount,lcount] = np.mean(python_estimator.estimate(source1_2D, target_2D))
			itoc = time.perf_counter()
			time_python_cor[ecount,lcount] = itoc - itic
			
			python_estimator = PythonSpectralMI(settings)
			itic = time.perf_counter()
			mi_python_uncor[ecount,lcount] = np.mean(python_estimator.estimate(source1_2D, source2_2D))
			itoc = time.perf_counter()
			time_python_uncor[ecount,lcount] = itoc - itic

			lcount += 1

		ecount += 1

	print(f"Summary PythonSpectralMI (mean of local values) lags ({vals}) (2D data):")

	print("MI values:")
	print("correlated data:")
	print("lag\t\tkraskov\t\tdiscrete\t\tgaussian")
	for i in range(len(vals)):
		print(f"{vals[i]}\t{mi_python_cor[0,i]}\t{mi_python_cor[1,i]}\t{mi_python_cor[2,i]}")
	
	test = []
	for e in range(len(estimators)):
		test.append(expected_best==np.argmax(mi_python_cor[e,:]))
	print(f"found expected lag:\t{test[0]}\t\t{test[1]}\t\t\t{test[2]}")


	print(f"\nuncorrelated data var2 = {Hz*0.7} Hz:")
	print("lag\t\tkraskov\t\tdiscrete\t\tgaussian")
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
	
	Hz = 40
	lag = 0
	noise = 0.2
	
	estimators = ['kraskov','discrete','gaussian']
	
	# test different estimator settings
	print(f"\nTest different estimator types of SpectralCMI on 1D {Hz} Hz freq data \nwith noise {noise} and no lag")
	
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
	print(" PythonSpectralCMI kraskov (cor): ", np.mean(time_python_cor[0]) )
	print(" PythonSpectralCMI discrete (cor): ", np.mean(time_python_cor[1]) )
	print(" PythonSpectralCMI gaussian (cor): ", np.mean(time_python_cor[2]) )
	print(" PythonSpectralCMI kraskov (uncor): ", np.mean(time_python_uncor[0]) )
	print(" PythonSpectralCMI discrete (uncor): ", np.mean(time_python_uncor[1]) )
	print(" PythonSpectralCMI gaussian (uncor): ", np.mean(time_python_uncor[2]) )
	

	# Test 2D data input
	reps = 10
	source1_2D = np.zeros((source1.shape[0], reps))
	target_2D = np.zeros((source1.shape[0], reps))
	source2_2D = np.zeros((source1.shape[0], reps))
	
	for i in range(reps):
		source1_2D[:,i], target_2D[:,i], source2_2D[:,i] = _get_freq_data(sample_rate=10000, duration=1.0, hz=Hz, lag=lag, noise=noise)#, seed=SEED)
		
	# test different estimator settings
	print(f"\nTest different estimator types of SpectralCMI on 2D {Hz} Hz freq data \nwith noise {noise} and no lag")
	
	
	mi_python_cor = np.zeros(len(estimators))
	mi_python_uncor = np.zeros(len(estimators))
	time_python_cor = np.zeros(len(estimators))
	time_python_uncor = np.zeros(len(estimators))
	
	
	ecount = 0
	for e in estimators:
		settings = {"estimator": e}

		python_estimator = PythonSpectralCMI(settings)
		itic = time.perf_counter()
		mi_python_cor[ecount] = python_estimator.estimate(source1_2D, target_2D, source2_2D)
		itoc = time.perf_counter()
		time_python_cor[ecount] = itoc - itic
		
		python_estimator = PythonSpectralCMI(settings)
		itic = time.perf_counter()
		mi_python_uncor[ecount] = python_estimator.estimate(source2_2D, target_2D, source1_2D)
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
	print(" PythonSpectralCMI kraskov (cor): ", np.mean(time_python_cor[0]) )
	print(" PythonSpectralCMI discrete (cor): ", np.mean(time_python_cor[1]) )
	print(" PythonSpectralCMI gaussian (cor): ", np.mean(time_python_cor[2]) )
	print(" PythonSpectralCMI kraskov (uncor): ", np.mean(time_python_uncor[0]) )
	print(" PythonSpectralCMI discrete (uncor): ", np.mean(time_python_uncor[1]) )
	print(" PythonSpectralCMI gaussian (uncor): ", np.mean(time_python_uncor[2]) )

def test_spectral_cmi_local_values():
	
	Hz = 40
	lag = 0
	noise = 0.2
	
	estimators = ['kraskov','discrete','gaussian']
	
	# test different estimator settings
	print(f"\nTest different estimator types of SpectralCMI on 1D {Hz} Hz freq data \nwith noise {noise} and no lag")
	
	mi_python_cor = np.zeros(len(estimators))
	mi_python_uncor = np.zeros(len(estimators))
	time_python_cor = np.zeros(len(estimators))
	time_python_uncor = np.zeros(len(estimators))
	
	source1, target, source2 = _get_freq_data(sample_rate=10000, duration=1.0, hz=Hz, lag=lag, noise=noise, seed=SEED)
	
	ecount = 0
	for e in estimators:
		settings = {"estimator": e,
					"local_values": True}

		python_estimator = PythonSpectralCMI(settings)
		itic = time.perf_counter()
		mi_python_cor[ecount] = np.mean(python_estimator.estimate(source1, target, source2))
		itoc = time.perf_counter()
		time_python_cor[ecount] = itoc - itic
		
		python_estimator = PythonSpectralCMI(settings)
		itic = time.perf_counter()
		mi_python_uncor[ecount] = np.mean(python_estimator.estimate(source2, target, source1))
		itoc = time.perf_counter()
		time_python_uncor[ecount] = itoc - itic

		ecount += 1
		
	print(f"Summary (mean of local values) PythonSpectralCMI:")
	print("CMI values:")
	print("uncorrelated conditional:")
	print("\tkraskov\t\tdiscrete\t\tgaussian")
	print(f"{mi_python_cor[0]}\t{mi_python_cor[1]}\t{mi_python_cor[2]}")
	
	print(f"\nuncorrelated source:")
	print("\tkraskov\t\tdiscrete\t\tgaussian")
	print(f"{mi_python_uncor[0]}\t{mi_python_uncor[1]}\t{mi_python_uncor[2]}")

	print("\nmean calculation times:")
	print(" PythonSpectralCMI kraskov (cor): ", np.mean(time_python_cor[0]) )
	print(" PythonSpectralCMI discrete (cor): ", np.mean(time_python_cor[1]) )
	print(" PythonSpectralCMI gaussian (cor): ", np.mean(time_python_cor[2]) )
	print(" PythonSpectralCMI kraskov (uncor): ", np.mean(time_python_uncor[0]) )
	print(" PythonSpectralCMI discrete (uncor): ", np.mean(time_python_uncor[1]) )
	print(" PythonSpectralCMI gaussian (uncor): ", np.mean(time_python_uncor[2]) )

	# Test 2D data input
	reps = 10
	source1_2D = np.zeros((source1.shape[0], reps))
	target_2D = np.zeros((source1.shape[0], reps))
	source2_2D = np.zeros((source1.shape[0], reps))
	
	for i in range(reps):
		source1_2D[:,i], target_2D[:,i], source2_2D[:,i] = _get_freq_data(sample_rate=10000, duration=1.0, hz=Hz, lag=lag, noise=noise)#, seed=SEED)
		
	# test different estimator settings
	print(f"\nTest different estimator types of SpectralCMI on 2D {Hz} Hz freq data \nwith noise {noise} and no lag")
		
	mi_python_cor = np.zeros(len(estimators))
	mi_python_uncor = np.zeros(len(estimators))
	time_python_cor = np.zeros(len(estimators))
	time_python_uncor = np.zeros(len(estimators))
	
	ecount = 0
	for e in estimators:
		settings = {"estimator": e,
					"local_values": True}

		python_estimator = PythonSpectralCMI(settings)
		itic = time.perf_counter()
		mi_python_cor[ecount] = np.mean(python_estimator.estimate(source1_2D, target_2D, source2_2D))
		itoc = time.perf_counter()
		time_python_cor[ecount] = itoc - itic
		
		python_estimator = PythonSpectralCMI(settings)
		itic = time.perf_counter()
		mi_python_uncor[ecount] = np.mean(python_estimator.estimate(source2_2D, target_2D, source1_2D))
		itoc = time.perf_counter()
		time_python_uncor[ecount] = itoc - itic

		ecount += 1
		
	print(f"Summary (mean of local values) PythonSpectralCMI:")
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
	
def test_spectral_ais(): ############################################## TODO
    
    hvals = [1,2,3,4]
    tvals = [10,20,30,30,50]

    Hz = 40
    lag = 20
    noise = 0.2
    estimators = ['kraskov','discrete','gaussian']
    
    print(f"\n\nTest average SpectralAIS on 1D {Hz} Hz freq data")
    print(f"\ntesting settings, history {hvals} and tau {tvals}")
    
    source1, target, source2 = _get_freq_data(sample_rate=10000, duration=1.0, hz=Hz, lag=lag, noise=noise, seed=SEED)
    
    count = 0
    ecount = 0
    
    print("\t\t\t\tPythonSpectralAIS")
        
    print("hist, tau\tkraskov\t\t\tdiscrete\t\tgaussian")
    
    for h in hvals:
        for t in tvals:

            st = f"{h, t}"
            for e in estimators:
                settings_p = {'estimator': e,'history': h, 'tau': t}
        
                python_estimator = PythonSpectralAIS(settings=settings_p)
                res = python_estimator.estimate(source1)

                st += f"\t{res}"
                
            print(st)

def test_spectral_ais_local_values(): ############################################## TODO
    
    hvals = [1,2,3,4]
    tvals = [10,20,30,30,50]

    Hz = 40
    lag = 20
    noise = 0.2
    estimators = ['kraskov','discrete','gaussian']
    
    print(f"\n\nTest average SpectralAIS on 1D {Hz} Hz freq data")
    print(f"\ntesting settings, history {hvals} and tau {tvals}")
    
    source1, target, source2 = _get_freq_data(sample_rate=10000, duration=1.0, hz=Hz, lag=lag, noise=noise, seed=SEED)
    
    count = 0
    ecount = 0
    
    print("\t\t\t\tPythonSpectralAIS")
        
    print("hist, tau\tkraskov\t\t\tdiscrete\t\tgaussian")
    
    for h in hvals:
        for t in tvals:

            st = f"{h, t}"
            for e in estimators:
                settings_p = {'estimator': e,'history': h, 'tau': t, 'local_values': True}
        
                python_estimator = PythonSpectralAIS(settings=settings_p)
                res = python_estimator.estimate(source1)

                st += f"\t{np.mean(res)}"
              
            print(st)

def test_spectral_te(): ############################################## TODO
    # test different estimator settings for PythonSpectralTE
    Hz = 40
    lag = 30
    noise = 0.2

    estimators = ['kraskov','discrete','gaussian']
    
    hvals = [1, 2 ,3]
    tvals = [1,10,20,30]
    hsvals = [0,15,30,45,60]

    source1, target, source2 = _get_freq_data(sample_rate=10000, duration=1.0, hz=Hz, lag=lag, noise=noise, seed=SEED)
    
    # test different estimator settings
    print(f"\nTest average SpectralMI on 1D {Hz} Hz freq data coulpled X->Y (noise {noise} and lag {lag})")
    print(f"Testing different estimator type settings {estimators}, history {hvals}, tau {tvals} \n and source_target_delay {hsvals}\n")
    
    print("hst,ht,tt,hs,ts\t\tkraskov\t\tgaussian\t\tdiscrete")

    count = 0
    for hst in hsvals:

        for ht in hvals:
            for tt in tvals:
                for hs in hvals:
                    for ts in tvals:
                        
                        st = f"{hst, ht, tt, hs, ts}"
                        
                        for e in estimators:
                            settings = {"estimator": e,
                                        "history_target": ht,
                                        "history_source": hs,
                                        "tau_target": tt,
                                        "tau_source": ts,
                                        "source_target_delay": hst}

                            python_estimator = PythonSpectralTE(settings)
                            
                            itic = time.perf_counter()
                            te_python = python_estimator.estimate(source1, target)
                            itoc = time.perf_counter()
                            time_python_cor[ecount,lcount] = itoc - itic
                            
                            st += f"\t{te_python}"

                        print(st)



    """
    ecount = 0
    for e in estimators:

        lcount=0
        for k in kvals:
            for t in tvals:

                settings = {"estimator": e,
                            "kraskov_k": k,
                            }

                python_estimator = PythonSpectralMI(settings)
                itic = time.perf_counter()
                te_python_cor[ecount,lcount] = python_estimator.estimate(source1, target)
                itoc = time.perf_counter()
                time_python_cor[ecount,lcount] = itoc - itic
                
                python_estimator = PythonSpectralMI(settings)
                itic = time.perf_counter()
                te_python_uncor[ecount,lcount] = python_estimator.estimate(source2, source1)
                itoc = time.perf_counter()
                time_python_uncor[ecount,lcount] = itoc - itic

                lcount += 1

            ecount += 1

    print(f"Summary PythonSpectralTE lags ({vals}) (1D data):")

    print("TE values:")
    print("coupled data:")
    print("lag\tkraskov\t\t\tdiscrete\t\tgaussian")
    for i in range(len(vals)):
        print(f"{vals[i]}\t{te_python_cor[0,i]}\t{te_python_cor[1,i]}\t{te_python_cor[2,i]}")
    
    test = []
    for e in range(len(estimators)):
        test.append(expected_best==np.argmax(te_python_cor[e,:]))
    print(f"found expected lag:\t{test[0]}\t\t{test[1]}\t\t\t{test[2]}")


    print(f"\nnot coupled data:")
    print("lag\tkraskov\t\t\tdiscrete\t\tgaussian")
    for i in range(len(vals)):
        print(f"{vals[i]}\t{te_python_uncor[0,i]}\t{te_python_uncor[1,i]}\t{te_python_uncor[2,i]}")
    

    print("\nmean calculation times:")
    print(" PythonSpectralTE kraskov (cor): ", np.mean(time_python_cor[0,:]) )
    print(" PythonSpectralTE discrete (cor): ", np.mean(time_python_cor[1,:]) )
    print(" PythonSpectralTE gaussian (cor): ", np.mean(time_python_cor[2,:]) )
    print(" PythonSpectralTE kraskov (uncor): ", np.mean(time_python_uncor[0,:]) )
    print(" PythonSpectralTE discrete (uncor): ", np.mean(time_python_uncor[1,:]) )
    print(" PythonSpectralTE gaussian (uncor): ", np.mean(time_python_uncor[2,:]) )
    










    # test spectral vs other estimators
    print(f"\n\nTest Spectral vs standard MI estimators with orig data {Hz} Hz freq data \nwith noise {noise} and lag {lag}")
    print(f"different estimator setting {estimators} and lag_mi {vals}\n")
    
    # kraskov
    te_jidt = np.zeros(len(vals))
    time_jidt = np.zeros(len(vals))
    te_python = np.zeros(len(vals))
    time_python = np.zeros(len(vals))
    te_python2 = np.zeros(len(vals))
    time_python2 = np.zeros(len(vals))
    
    print("- TE estimator: kraskov")
    lcount=0
    for lags in vals:
        settings = {"estimator": 'kraskov',
                    "lag_mi": lags,
                    }

        python_estimator = PythonSpectralMI(settings)
        itic = time.perf_counter()
        te_python[lcount] = python_estimator.estimate(source1, target)
        itoc = time.perf_counter()
        time_python[lcount] = itoc - itic
        
        python_estimator2 = PythonKraskovMI({"lag_mi": lags})
        itic = time.perf_counter()
        te_python2[lcount] = python_estimator2.estimate(source1, target)
        itoc = time.perf_counter()
        time_python2[lcount] = itoc - itic
        
        jidt_estimator = JidtKraskovMI({"lag_mi": lags})
        itic = time.perf_counter()
        te_jidt[lcount] = jidt_estimator.estimate(source1, target)
        itoc = time.perf_counter()
        time_jidt[lcount] = itoc - itic

        lcount += 1

    print(f"\n\nSummary PythonSpectralTE 'kraskov' vs JidtKraskovMI vs PythonKraskovMI:")

    print("MI values:")
    print("correlated data:")
    print("lag\tPythonSpectralTE\tJidtKraskovTE\t\tPythonKraskovTE")
    for i in range(len(vals)):
        print(f"{vals[i]}\t{te_python[i]}\t{te_jidt[i]}\t{te_python2[i]}")

    print(f"found expected lag:\t{expected_best==np.argmax(te_python)}\t\t{expected_best==np.argmax(te_jidt)}\t\t\t{expected_best==np.argmax(te_python2)}")

    print("\nmean calculation times:")
    print(" PythonSpectralMI kraskov: ", np.mean(time_python) )
    print(" JidtKraskovMI: ", np.mean(time_jidt) )
    print(" PythonKraskovlMI: ", np.mean(time_python2) )

    """








#def test_spectral_te_local_values(): ############################################## TODO

















if __name__ == '__main__':

	#testhead("SpectralMI")
	#test_spectral_mi()
	
	#testhead("SpectralMI local values")
	#test_spectral_mi_local_values()

	#testhead("SpectralCMI")
	#test_spectral_cmi()
	
	#testhead("SpectralCMI local values")
	#test_spectral_cmi_local_values()

	#testhead("SpectralAIS") ############################################## TODO
	#test_spectral_ais()

	#testhead("SpectralAIS local values") ############################################## TODO
	#test_spectral_ais_local_values()

	testhead("SpectralTE") ############################################## TODO
	test_spectral_te()

	#testhead("SpectralTE local values") ############################################## TODO
	#test_spectral_te_local_values()

