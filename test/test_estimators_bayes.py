""" Provide tests for Python Bayesian Estimators """


import numpy as np

import time
import sys
import copy


from idtxl.estimators_python import PythonBayesianDiscreteMI, PythonDiscreteMI, PythonBayesianDiscreteCMI, PythonDiscreteCMI
from generate_test_data import (_get_gauss_data, _get_ar_data, _generate_mute_data,
                                _get_mem_binary_data, _get_freq_data, generate_continuous_idtxl_data)

SEED=42

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
	print(f"\n            {est}:\n")
	print("#######################################################################")


def test_bayesian_discrete_mi():

	nbins = 2
	lvals = [0,1,2]
	cvals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	pvals = [0.25, 0.5, 0.75, 1.0]
	approaches = ['analytical','numerical']

	samples = 10000

	nsamples = 4000

	atol = 5e-03
	atol2 = 1e-04
	
	print(f"\n\nTesting average MI PythonBayesianDiscreteMI and PythonDiscreteMI using 1D gaussian data")
	print(f"testing covariances {cvals}, {samples} samples, lag = 1 - correlated and uncorrelated")
	print(f"testing settings lag_mi {lvals} and discrete_method=max_ent")
	print(f"dirichlet alphas (dprior) {pvals} - for both approaches: {approaches}")
	print(f"'numerical': nsamples = {nsamples}")
		
	print("\n#### correlated data:\n")
	print(f"atol={atol}\t\t\tPythonBayesianDiscreteMI\t\tPythonDiscreteMI\tPythonDiscreteMI close to")
	print(f"lag,cov,dprior\t\tanalytical\t\tnumerical\tclose\tmi\t\t\tanalytical {atol} {atol2} numerical {atol} {atol2}\n")
	
	time_analytical=np.zeros(len(lvals)*len(cvals)*len(pvals))
	time_numerical=np.zeros(len(lvals)*len(cvals)*len(pvals))
	time_discrete=np.zeros(len(lvals)*len(cvals)*len(pvals))

	count = 0
	for l in lvals:
		for c in cvals:
			expected_mi, source1, source2, target = _get_gauss_data(n=samples+1, expand=True, covariance=c, seed=SEED)
			source1 = source1[1:]
			source2 = source2[1:]
			target = target[:-1]

			for p in pvals:

				s = f"{l,c,p}\t"
				mis = np.zeros(2)
				acount = 0
				for a in approaches:

					settings = {'approach': a,
								'lag_mi': l,
								'dprior': p,
								'n_discrete_bins': nbins,
								'discretise_method': 'max_ent',
								'nsamples': nsamples,
								}
					est = PythonBayesianDiscreteMI(settings)
					
					itic = time.perf_counter()
					mi = est.estimate(source1, target)
					itoc = time.perf_counter()
					if a == 'analytical':
						time_analytical[count] = itoc-itic
					else:
						time_numerical[count] = itoc-itic

					s += f"{mi}\t"
				
					mis[acount] = mi
					acount += 1

				settings2= {'lag_mi': l,
							'n_discrete_bins': nbins,
							'discretise_method': 'max_ent',
							}

				est2 = PythonDiscreteMI(settings2)
					
				itic = time.perf_counter()
				mi2 = est2.estimate(source1, target)
				itoc = time.perf_counter()
				time_discrete[count] = itoc-itic
				
				s += f"{np.isclose(mis[0], mis[1], atol=atol)}\t{mi2}\t{np.isclose(mis[0], mi2, atol=atol)}\t{np.isclose(mis[0], mi2, atol=atol2)}\t{np.isclose(mis[1], mi2, atol=atol)}\t{np.isclose(mis[1], mi2, atol=atol2)}"

				print(s)

				count += 1

	print("\nmean calculation times:")
	print(" PythonBayesianDiscreteMI analytical: ", np.mean(time_analytical) )
	print(" PythonBayesianDiscreteMI numerical: ", np.mean(time_numerical) )
	print(" PythonDiscreteMI: ", np.mean(time_discrete) )
		


	# test pure noise
	atol = 5e-03
	print("\n#### Testing pure random noise:\n")
	print(f"atol={atol}\t\t\tPythonBayesianDiscreteMI\t\tPythonDiscreteMI\tPythonDiscreteMI close to")
	print(f"lag,dprior\t\tanalytical\t\tnumerical\tclose\tmi\t\t\tanalytical {atol} {atol2} numerical {atol} {atol2}\n")
	
	time_analytical=np.zeros(len(lvals)*len(pvals))
	time_numerical=np.zeros(len(lvals)*len(pvals))
	time_discrete=np.zeros(len(lvals)*len(pvals))
	
	count = 0
	for l in lvals:
		source1 = np.random.randn(samples-1)
		source2 = np.random.randn(samples-1)

		for p in pvals:

			s = f"{l,c,p}\t"
			mis = np.zeros(2)
			acount = 0
			for a in approaches:

				settings = {'approach': a,
							'lag_mi': l,
							'dprior': p,
							'n_discrete_bins': nbins,
							'discretise_method': 'max_ent',
							'nsamples': 4000,
							}

				est = PythonBayesianDiscreteMI(settings)
				
				itic = time.perf_counter()
				mi = est.estimate(source1, source2)
				itoc = time.perf_counter()
				if a == 'analytical':
					time_analytical[count] = itoc-itic
				else:
					time_numerical[count] = itoc-itic

				s += f"{mi}\t"
				mis[acount] = mi
				acount += 1

			settings2= {'lag_mi': l,
						'n_discrete_bins': nbins,
						'discretise_method': 'max_ent',
						}

			est2 = PythonDiscreteMI(settings2)
			
			itic = time.perf_counter()
			mi2 = est2.estimate(source1, source2)
			itoc = time.perf_counter()
			time_discrete[count] = itoc-itic

			s += f"{np.isclose(mis[0], mis[1], atol=atol)}\t{mi2}\t{np.isclose(mis[0], mi2, atol=atol)}\t{np.isclose(mis[0], mi2, atol=atol2)}\t{np.isclose(mis[1], mi2, atol=atol)}\t{np.isclose(mis[1], mi2, atol=atol2)}"

			print(s)

			count += 1

	print("\nmean calculation times:")
	print(" PythonBayesianDiscreteMI analytical: ", np.mean(time_analytical) )
	print(" PythonBayesianDiscreteMI numerical: ", np.mean(time_numerical) )
	print(" PythonDiscreteMI: ", np.mean(time_discrete) )
	

	# test 2D
	reps = 2
	
	print(f"\n\nTesting average MI PythonBayesianDiscreteMI and PythonDiscreteMI using 2D mute data")
	print(f"testing covariances {cvals}, {samples} samples, lag = 1 ")
	print(f"testing settings lag_mi {lvals} and discrete_method=max_ent")
	print(f"dirichlet alphas (dprior) {pvals} - for both approaches: {approaches}")
	print(f"'numerical': nsamples = {nsamples}")
	
	atol = 5e-03
	print("\n#### correlated data:\n")
	print(f"atol={atol}\t\t\tPythonBayesianDiscreteMI\t\tPythonDiscreteMI\tPythonDiscreteMI close to")
	print(f"lag,dprior\t\tanalytical\t\tnumerical\tclose\tmi\t\t\tanalytical {atol} {atol2} numerical {atol} {atol2}\n")
	
	time_analytical=np.zeros(len(lvals)*len(pvals))
	time_numerical=np.zeros(len(lvals)*len(pvals))
	time_discrete=np.zeros(len(lvals)*len(pvals))

	count = 0
	for l in lvals:
		data = _generate_mute_data(n_replications=reps)
		source1 = data[0,:,:]
		target = data[2,:,:]
		
		for p in pvals:

			s = f"{l,p}\t"
			mis = np.zeros(2)
			acount = 0
			for a in approaches:

				settings = {'approach': a,
							'lag_mi': l,
							'dprior': p,
							'n_discrete_bins': nbins,
							'discretise_method': 'max_ent',
							'nsamples': nsamples,
							}
				
				est = PythonBayesianDiscreteMI(settings)
				
				itic = time.perf_counter()
				mi = est.estimate(source1, target)
				itoc = time.perf_counter()
				if a == 'analytical':
					time_analytical[count] = itoc-itic
				else:
					time_numerical[count] = itoc-itic

				
				s += f"{mi}\t"
			
				mis[acount] = mi
				acount += 1

			settings2= {'lag_mi': l,
						'n_discrete_bins': nbins,
						'discretise_method': 'max_ent',
						}

			est2 = PythonDiscreteMI(settings2)
			
			itic = time.perf_counter()
			mi2 = est2.estimate(source1, target)
			itoc = time.perf_counter()
			time_discrete[count] = itoc-itic

			s += f"{np.isclose(mis[0], mis[1], atol=atol)}\t{mi2}\t{np.isclose(mis[0], mi2, atol=atol)}\t{np.isclose(mis[0], mi2, atol=atol2)}\t{np.isclose(mis[1], mi2, atol=atol)}\t{np.isclose(mis[1], mi2, atol=atol2)}"

			print(s)

			count += 1

	print("\nmean calculation times:")
	print(" PythonBayesianDiscreteMI analytical: ", np.mean(time_analytical) )
	print(" PythonBayesianDiscreteMI numerical: ", np.mean(time_numerical) )
	print(" PythonDiscreteMI: ", np.mean(time_discrete) )
	
"""
def test_bayesian_discrete_mi_local_values():
	
	nbins = 2
	lvals = [0,1,2]
	cvals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	pvals = [0.25, 0.5, 0.75, 1.0]
	approaches = ['numerical']

	samples = 10000

	nsamples = 4000

	atol = 5e-03
	atol2 = 1e-04
	
	print(f"\n\nTesting average MI PythonBayesianDiscreteMI and PythonDiscreteMI using 1D gaussian data")
	print(f"testing covariances {cvals}, {samples} samples, lag = 1")# - correlated and uncorrelated")
	print(f"testing settings lag_mi {lvals} and discrete_method=max_ent")
	print(f"for both approaches: {approaches}")
	print(f"'numerical': nsamples = {nsamples}")
		
	print("\n#### correlated data:\n")
	#print(f"atol={atol}\t\t\tPythonBayesianDiscreteMI\t\tPythonDiscreteMI\tPythonDiscreteMI close to")
	print(f"atol={atol}\tPythonBayesianDiscreteMI\tPythonDiscreteMI\tPythonDiscreteMI close to")
	#print(f"lag,cov,dprior\t\tanalytical\t\tnumerical\tclose\tmi\t\t\tanalytical {atol} {atol2} numerical {atol} {atol2}\n")
	print(f"lag,cov,dprior\t\tnumerical\t\tmi\t\t\tnumerical {atol} {atol2}\n")
	
	time_analytical=np.zeros(len(lvals)*len(cvals)*len(pvals))
	time_numerical=np.zeros(len(lvals)*len(cvals)*len(pvals))
	time_discrete=np.zeros(len(lvals)*len(cvals)*len(pvals))

	count = 0
	for l in lvals:
		for c in cvals:
			expected_mi, source1, source2, target = _get_gauss_data(n=samples+1, expand=True, covariance=c, seed=SEED)
			source1 = source1[1:]
			source2 = source2[1:]
			target = target[:-1]


			for p in pvals:

				s = f"{l,c,p}\t"
				mis = np.zeros((2, nsamples))
				acount = 0
				for a in approaches:

					settings = {'approach': a,
								'lag_mi': l,
								'dprior': p,
								'n_discrete_bins': nbins,
								'discretise_method': 'max_ent',
								'nsamples': nsamples,
								'local_values': True,
								}
					est = PythonBayesianDiscreteMI(settings)
					
					itic = time.perf_counter()
					mi = est.estimate(source1, target)
					itoc = time.perf_counter()
					if a == 'analytical':
						time_analytical[count] = itoc-itic
					else:
						time_numerical[count] = itoc-itic

					s += f"{np.mean(mi)}\t"
				
					mis[acount,:] = mi
					acount += 1

				settings2= {'lag_mi': l,
							'n_discrete_bins': nbins,
							'discretise_method': 'max_ent',
							'local_values': True,
							}

				est2 = PythonDiscreteMI(settings2)
					
				itic = time.perf_counter()
				mi2 = est2.estimate(source1, target)
				itoc = time.perf_counter()
				time_discrete[count] = itoc-itic
				
				#s += f"{np.allclose(mis[0,:], mis[1,:], atol=atol)}\t{np.mean(mi2)}\t{np.allclose(mis[0,:], np.mean(mi2), atol=atol)}\t{np.allclose(mis[0,:], np.mean(mi2), atol=atol2)}\t{np.allclose(mis[1,:], np.mean(mi2), atol=atol)}\t{np.allclose(mis[1,:], np.mean(mi2), atol=atol2)}"
				s += f"{np.mean(mi2)}\t\t{np.allclose(mis[0,:], np.mean(mi2), atol=atol)}\t{np.allclose(mis[0,:], np.mean(mi2), atol=atol2)}"

				print(s)

				count += 1

	print("\nmean calculation times:")
	#print(" PythonBayesianDiscreteMI analytical: ", np.mean(time_analytical) )
	print(" PythonBayesianDiscreteMI numerical: ", np.mean(time_numerical) )
	print(" PythonDiscreteMI: ", np.mean(time_discrete) )
	


	# test 2D data
	reps = 2
	
	print(f"\n\nTesting average MI PythonBayesianDiscreteMI and PythonDiscreteMI using 2D mute data")
	print(f"testing {samples} samples, {reps} replications, lag = 1")# - correlated and uncorrelated")
	print(f"testing settings lag_mi {lvals} and discrete_method=max_ent")
	print(f"for both approaches: {approaches}")
	print(f"'numerical': nsamples = {nsamples}")
	
	atol = 5e-03
			
	print("\n#### correlated data:\n")
	#print(f"atol={atol}\t\t\tPythonBayesianDiscreteMI\t\tPythonDiscreteMI\tPythonDiscreteMI close to")
	print(f"atol={atol}\tPythonBayesianDiscreteMI\tPythonDiscreteMI\tPythonDiscreteMI close to")
	#print(f"lag,dprior\t\tanalytical\t\tnumerical\tclose\tmi\t\t\tanalytical {atol} {atol2} numerical {atol} {atol2}\n")
	print(f"lag,dprior\t\tnumerical\t\tmi\t\t\tnumerical {atol} {atol2}\n")
	
	time_analytical=np.zeros(len(lvals)*len(cvals)*len(pvals))
	time_numerical=np.zeros(len(lvals)*len(cvals)*len(pvals))
	time_discrete=np.zeros(len(lvals)*len(cvals)*len(pvals))

	count = 0
	for l in lvals:
		data = _generate_mute_data(n_replications=reps)
		source1 = data[0,:,:]
		target = data[2,:,:]

		for p in pvals:

			s = f"{l,p}\t"
			mis = np.zeros((2, nsamples))
			acount = 0
			for a in approaches:

				settings = {'approach': a,
							'lag_mi': l,
							'dprior': p,
							'n_discrete_bins': nbins,
							'discretise_method': 'max_ent',
							'nsamples': nsamples,
							'local_values': True,
							}
				est = PythonBayesianDiscreteMI(settings)
				
				itic = time.perf_counter()
				mi = est.estimate(source1, target)
				itoc = time.perf_counter()
				if a == 'analytical':
					time_analytical[count] = itoc-itic
				else:
					time_numerical[count] = itoc-itic

				s += f"{np.mean(mi)}\t"
			
				mis[acount,:] = mi
				acount += 1

			settings2= {'lag_mi': l,
						'n_discrete_bins': nbins,
						'discretise_method': 'max_ent',
						'local_values': True,
						}

			est2 = PythonDiscreteMI(settings2)
				
			itic = time.perf_counter()
			mi2 = est2.estimate(source1, target)
			itoc = time.perf_counter()
			time_discrete[count] = itoc-itic
			
			print(mi2[:20])
			print(mis[0,:20])


			#s += f"{np.allclose(mis[0,:], mis[1,:], atol=atol)}\t{np.mean(mi2)}\t{np.allclose(mis[0,:], np.mean(mi2), atol=atol)}\t{np.allclose(mis[0,:], np.mean(mi2), atol=atol2)}\t{np.allclose(mis[1,:], np.mean(mi2), atol=atol)}\t{np.allclose(mis[1,:], np.mean(mi2), atol=atol2)}"
			s += f"{np.mean(mi2)}\t\t{np.allclose(mis[0,:], np.mean(mi2), atol=atol)}\t{np.allclose(mis[0,:], np.mean(mi2), atol=atol2)}"

			print(s)

			count += 1

	print("\nmean calculation times:")
	#print(" PythonBayesianDiscreteMI analytical: ", np.mean(time_analytical) )
	print(" PythonBayesianDiscreteMI numerical: ", np.mean(time_numerical) )
	print(" PythonDiscreteMI: ", np.mean(time_discrete) )	
"""

def test_bayesian_discrete_cmi():
	
	nbins = 2
	lvals = [0,1,2]
	cvals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	pvals = [0.25, 0.5, 0.75, 1.0]
	approaches = ['analytical','numerical']

	samples = 10000

	nsamples = 4000

	atol = 5e-03
	atol2 = 1e-04
	
	print(f"\n\nTesting averaget CMI PythonBayesianDiscreteCMI and PythonDiscreteCMI using 1D gaussian data")
	print(f"testing covariances {cvals}, {samples} samples - uncorrelated conditional and uncorrelated source")
	print(f"for both approaches: {approaches}")
	print(f"'numerical': nsamples = {nsamples}")
		
	print("\n#### uncorrelated conditional:\n")
	print(f"atol={atol}\t\t\tPythonBayesianDiscreteCMI\t\tPythonDiscreteCMI\tPythonDiscreteCMI close to")
	print(f"cov,dprior\t\tanalytical\t\tnumerical\tclose\tmi\t\t\tanalytical {atol} {atol2} numerical {atol} {atol2}\n")
	
	time_analytical=np.zeros(len(lvals)*len(cvals)*len(pvals))
	time_numerical=np.zeros(len(lvals)*len(cvals)*len(pvals))
	time_discrete=np.zeros(len(lvals)*len(cvals)*len(pvals))

	count = 0
	for c in cvals:
		expected_mi, source1, source2, target = _get_gauss_data(n=samples, expand=True, covariance=c, seed=SEED)
		source2 = np.random.rand(samples)

		for p in pvals:
			s = f"{c,p}\t"
			mis = np.zeros(2)
			acount = 0
			for a in approaches:

				settings = {'approach': a,
							'dprior': p,
							'n_discrete_bins': nbins,
							'discretise_method': 'max_ent',
							'nsamples': nsamples,
							}
				est = PythonBayesianDiscreteCMI(settings)
				
				itic = time.perf_counter()
				mi = est.estimate(source1, target, source2)
				itoc = time.perf_counter()
				if a == 'analytical':
					time_analytical[count] = itoc-itic
				else:
					time_numerical[count] = itoc-itic

				s += f"{mi}\t"
			
				mis[acount] = mi
				acount += 1

			settings2= {'n_discrete_bins': nbins,
						'discretise_method': 'max_ent',
						}

			est2 = PythonDiscreteCMI(settings2)
				
			itic = time.perf_counter()
			mi2 = est2.estimate(source1, target, source2)
			itoc = time.perf_counter()
			time_discrete[count] = itoc-itic
			
			s += f"{np.isclose(mis[0], mis[1], atol=atol)}\t{mi2}\t{np.isclose(mis[0], mi2, atol=atol)}\t{np.isclose(mis[0], mi2, atol=atol2)}\t{np.isclose(mis[1], mi2, atol=atol)}\t{np.isclose(mis[1], mi2, atol=atol2)}"

			print(s)

			count += 1

	print("\nmean calculation times:")
	print(" PythonBayesianDiscreteCMI analytical: ", np.mean(time_analytical) )
	print(" PythonBayesianDiscreteCMI numerical: ", np.mean(time_numerical) )
	print(" PythonDiscreteCMI: ", np.mean(time_discrete) )
		

	print("\n#### uncorrelated source:\n")
	print(f"atol={atol}\t\t\tPythonBayesianDiscreteCMI\t\tPythonDiscreteCMI\tPythonDiscreteCMI close to")
	print(f"cov,dprior\t\tanalytical\t\tnumerical\tclose\tmi\t\t\tanalytical {atol} {atol2} numerical {atol} {atol2}\n")
	
	time_analytical=np.zeros(len(lvals)*len(cvals)*len(pvals))
	time_numerical=np.zeros(len(lvals)*len(cvals)*len(pvals))
	time_discrete=np.zeros(len(lvals)*len(cvals)*len(pvals))

	count = 0
	for c in cvals:
		expected_mi, source1, source2, target = _get_gauss_data(n=samples, expand=True, covariance=c, seed=SEED)
		source1 = np.random.rand(samples)

		for p in pvals:
			s = f"{c,p}\t"
			mis = np.zeros(2)
			acount = 0
			for a in approaches:

				settings = {'approach': a,
							'dprior': p,
							'n_discrete_bins': nbins,
							'discretise_method': 'max_ent',
							'nsamples': nsamples,
							}
				est = PythonBayesianDiscreteCMI(settings)
				
				itic = time.perf_counter()
				mi = est.estimate(source2, target, source1)
				itoc = time.perf_counter()
				if a == 'analytical':
					time_analytical[count] = itoc-itic
				else:
					time_numerical[count] = itoc-itic

				s += f"{mi}\t"
			
				mis[acount] = mi
				acount += 1

			settings2= {'n_discrete_bins': nbins,
						'discretise_method': 'max_ent',
						}

			est2 = PythonDiscreteCMI(settings2)
				
			itic = time.perf_counter()
			mi2 = est2.estimate(source2, target, source1)
			itoc = time.perf_counter()
			time_discrete[count] = itoc-itic
			
			s += f"{np.isclose(mis[0], mis[1], atol=atol)}\t{mi2}\t{np.isclose(mis[0], mi2, atol=atol)}\t{np.isclose(mis[0], mi2, atol=atol2)}\t{np.isclose(mis[1], mi2, atol=atol)}\t{np.isclose(mis[1], mi2, atol=atol2)}"

			print(s)

			count += 1

	print("\nmean calculation times:")
	print(" PythonBayesianDiscreteCMI analytical: ", np.mean(time_analytical) )
	print(" PythonBayesianDiscreteCMI numerical: ", np.mean(time_numerical) )
	print(" PythonDiscreteCMI: ", np.mean(time_discrete) )
		



	# Test 2D 
	reps = [2,4]
	print(f"\n\nTesting averaget CMI PythonBayesianDiscreteCMI and PythonDiscreteCMI using 2D mute data and noise as conditional")
	print(f"testing {samples} samples, {reps} replications - uncorrelated conditional")
	print(f"for both approaches: {approaches}")
	print(f"'numerical': nsamples = {nsamples}")
		
	print("\n#### uncorrelated conditional:\n")
	print(f"atol={atol}\t\t\tPythonBayesianDiscreteCMI\t\tPythonDiscreteCMI\tPythonDiscreteCMI close to")
	print(f"reps,dprior\t\tanalytical\t\tnumerical\tclose\tmi\t\t\tanalytical {atol} {atol2} numerical {atol} {atol2}\n")
	
	time_analytical=np.zeros(len(lvals)*len(cvals)*len(pvals))
	time_numerical=np.zeros(len(lvals)*len(cvals)*len(pvals))
	time_discrete=np.zeros(len(lvals)*len(cvals)*len(pvals))

	count = 0
	for r in reps:
		data = _generate_mute_data(n_replications=r)
		source1 = data[0,:,:]
		target = data[2,:,:]
		
		#source2 = data[4,:,:]
		source2 = np.random.rand(len(source1),r)
	
		for p in pvals:
			s = f"{r,p}\t"
			mis = np.zeros(2)
			acount = 0
			for a in approaches:

				settings = {'approach': a,
							'dprior': p,
							'n_discrete_bins': nbins,
							'discretise_method': 'max_ent',
							'nsamples': nsamples,
							}
				est = PythonBayesianDiscreteCMI(settings)
				
				itic = time.perf_counter()
				mi = est.estimate(source1, target, source2)
				itoc = time.perf_counter()
				if a == 'analytical':
					time_analytical[count] = itoc-itic
				else:
					time_numerical[count] = itoc-itic

				s += f"{mi}\t"
			
				mis[acount] = mi
				acount += 1

			settings2= {'n_discrete_bins': nbins,
						'discretise_method': 'max_ent',
						}

			est2 = PythonDiscreteCMI(settings2)
				
			itic = time.perf_counter()
			mi2 = est2.estimate(source1, target, source2)
			itoc = time.perf_counter()
			time_discrete[count] = itoc-itic
			
			s += f"{np.isclose(mis[0], mis[1], atol=atol)}\t{mi2}\t{np.isclose(mis[0], mi2, atol=atol)}\t{np.isclose(mis[0], mi2, atol=atol2)}\t{np.isclose(mis[1], mi2, atol=atol)}\t{np.isclose(mis[1], mi2, atol=atol2)}"

			print(s)

			count += 1

	print("\nmean calculation times:")
	print(" PythonBayesianDiscreteCMI analytical: ", np.mean(time_analytical) )
	print(" PythonBayesianDiscreteCMI numerical: ", np.mean(time_numerical) )
	print(" PythonDiscreteCMI: ", np.mean(time_discrete) )

def test_bayesian_discrete_cmi_local_values():
		
	nbins = 2
	lvals = [0,1,2]
	cvals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	pvals = [0.25, 0.5, 0.75, 1.0]
	approaches = ['analytical','numerical']

	samples = 10000

	nsamples = 4000

	atol = 5e-03
	atol2 = 1e-04
	
	print(f"\n\nTesting local CMI PythonBayesianDiscreteCMI and PythonDiscreteCMI using 1D gaussian data")
	print(f"testing covariances {cvals}, {samples} samples - uncorrelated conditional and uncorrelated source")
	print(f"for both approaches: {approaches}")
	print(f"'numerical': nsamples = {nsamples}")
		
	print("\n#### uncorrelated conditional:\n")
	print(f"atol={atol}\t\t\tPythonBayesianDiscreteCMI\t\tPythonDiscreteCMI\tPythonDiscreteCMI allclose to")
	print(f"cov,dprior\t\tanalytical\t\tnumerical\tallclose\tmi\t\t\tanalytical {atol} {atol2} numerical {atol} {atol2}\n")
	
	time_analytical=np.zeros(len(lvals)*len(cvals)*len(pvals))
	time_numerical=np.zeros(len(lvals)*len(cvals)*len(pvals))
	time_discrete=np.zeros(len(lvals)*len(cvals)*len(pvals))

	count = 0
	for c in cvals:
		expected_mi, source1, source2, target = _get_gauss_data(n=samples, expand=True, covariance=c, seed=SEED)
		source2 = np.random.rand(samples)

		for p in pvals:
			s = f"{c,p}\t"
			mis = np.zeros((2,samples))
			acount = 0
			for a in approaches:

				settings = {'approach': a,
							'dprior': p,
							'n_discrete_bins': nbins,
							'discretise_method': 'max_ent',
							'nsamples': nsamples,
							}
				est = PythonBayesianDiscreteCMI(settings)
				
				itic = time.perf_counter()
				mi = est.estimate(source1, target, source2)
				itoc = time.perf_counter()
				if a == 'analytical':
					time_analytical[count] = itoc-itic
				else:
					time_numerical[count] = itoc-itic

				s += f"{np.mean(mi)}\t"
			
				mis[acount,:] = mi
				acount += 1

			settings2= {'n_discrete_bins': nbins,
						'discretise_method': 'max_ent',
						}

			est2 = PythonDiscreteCMI(settings2)
				
			itic = time.perf_counter()
			mi2 = est2.estimate(source1, target, source2)
			itoc = time.perf_counter()
			time_discrete[count] = itoc-itic
			
			s += f"{np.allclose(mis[0,:], mis[1,:], atol=atol)}\t{np.mean(mi2)}\t{np.allclose(mis[0,:], np.mean(mi2), atol=atol)}\t{np.allclose(mis[0,:], np.mean(mi2), atol=atol2)}\t{np.allclose(mis[1,:], np.mean(mi2), atol=atol)}\t{np.allclose(mis[1,:], np.mean(mi2), atol=atol2)}"

			print(s)

			count += 1

	print("\nmean calculation times:")
	print(" PythonBayesianDiscreteCMI analytical: ", np.mean(time_analytical) )
	print(" PythonBayesianDiscreteCMI numerical: ", np.mean(time_numerical) )
	print(" PythonDiscreteCMI: ", np.mean(time_discrete) )
		

	print("\n#### uncorrelated source:\n")
	print(f"atol={atol}\t\t\tPythonBayesianDiscreteCMI\t\tPythonDiscreteCMI\tPythonDiscreteCMI close to")
	print(f"cov,dprior\t\tanalytical\t\tnumerical\tclose\tmi\t\t\tanalytical {atol} {atol2} numerical {atol} {atol2}\n")
	
	time_analytical=np.zeros(len(lvals)*len(cvals)*len(pvals))
	time_numerical=np.zeros(len(lvals)*len(cvals)*len(pvals))
	time_discrete=np.zeros(len(lvals)*len(cvals)*len(pvals))

	count = 0
	for c in cvals:
		expected_mi, source1, source2, target = _get_gauss_data(n=samples, expand=True, covariance=c, seed=SEED)
		source1 = np.random.rand(samples)

		for p in pvals:
			s = f"{c,p}\t"
			mis = np.zeros((2,samples))
			acount = 0
			for a in approaches:

				settings = {'approach': a,
							'dprior': p,
							'n_discrete_bins': nbins,
							'discretise_method': 'max_ent',
							'nsamples': nsamples,
							}
				est = PythonBayesianDiscreteCMI(settings)
				
				itic = time.perf_counter()
				mi = est.estimate(source2, target, source1)
				itoc = time.perf_counter()
				if a == 'analytical':
					time_analytical[count] = itoc-itic
				else:
					time_numerical[count] = itoc-itic

				s += f"{np.mean(mi)}\t"
			
				mis[acount,:] = mi
				acount += 1

			settings2= {'n_discrete_bins': nbins,
						'discretise_method': 'max_ent',
						}

			est2 = PythonDiscreteCMI(settings2)
				
			itic = time.perf_counter()
			mi2 = est2.estimate(source2, target, source1)
			itoc = time.perf_counter()
			time_discrete[count] = itoc-itic
			
			s += f"{np.allclose(mis[0,:], mis[1,:], atol=atol)}\t{np.mean(mi2)}\t{np.allclose(mis[0,:], np.mean(mi2), atol=atol)}\t{np.allclose(mis[0,:], np.mean(mi2), atol=atol2)}\t{np.allclose(mis[1,:], np.mean(mi2), atol=atol)}\t{np.allclose(mis[1,:], np.mean(mi2), atol=atol2)}"

			print(s)

			count += 1

	print("\nmean calculation times:")
	print(" PythonBayesianDiscreteCMI analytical: ", np.mean(time_analytical) )
	print(" PythonBayesianDiscreteCMI numerical: ", np.mean(time_numerical) )
	print(" PythonDiscreteCMI: ", np.mean(time_discrete) )
		

if __name__ == '__main__':

	#### Test Bayesian estimators
	
	testhead("PythonBayesianDiscreteMI")
	test_bayesian_discrete_mi()

	testhead("PythonBayesianDiscreteCMI")
	test_bayesian_discrete_cmi()

	testhead("PythonBayesianDiscreteMI local values")
	test_bayesian_discrete_cmi_local_values()
