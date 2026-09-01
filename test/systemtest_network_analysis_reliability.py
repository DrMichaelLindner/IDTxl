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
                                _get_mem_binary_data, _get_freq_data)


from collections import defaultdict

# Test reliability of network analysis


def test_network_analysis_single(data, analysis, est_type, numperm=500, samples=1000, reps=3, verbose=True, nbins=2):
	
	measure = analysis[-2:].lower()
	jidt_estimator = f"Jidt{est_type}CMI"
	python_estimator = f"Python{est_type}CMI"
	
	
	
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
		'discretise_method': 'none',
        'n_discrete_bins': nbins, 
		"verbose": verbose,
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
		'discretise_method': 'none',
        'n_discrete_bins': nbins, 
		"verbose": verbose,
	}

	itic = time.perf_counter()
	results_python = network_analysis.analyse_network(settings2, data2)
	itoc = time.perf_counter()
	time_python = itoc - itic

	return results_jidt, results_python, time_jidt, time_python


def test_network_analysis_loop(analysis, est_type, outputfile, num_loops=3, numperm=500, samples=1000, reps=3, verbose=True, nbins=2):

	
	measure = analysis[-2:].lower()
	jidt_estimator = f"Jidt{est_type}CMI"
	python_estimator = f"Python{est_type}CMI"

	data = Data(normalise=False)  # initialise an empty data object
	data.generate_mute_data(n_samples=samples, n_replications=reps)
	if est_type == "Discrete":
		est = PythonDiscreteCMI({"discretise_method": "equal", "n_discrete_bins": nbins})
		d = data.data
		for i in range(5):
			d[i,:,:] = est._discretise_vars(var1=d[i,:,:])
		d = d.astype(int)
		data.set_data(d, "psr")


	results_jidt = defaultdict(int)
	results_python = defaultdict(int)
	time_jidt = np.zeros(num_loops)
	time_python = np.zeros(num_loops)

	for i in range(num_loops):

		r_jidt, r_python, t_jidt, t_python = test_network_analysis_single(data, analysis, est_type, numperm, samples, reps, verbose, nbins)

		results_jidt[i] = r_jidt
		results_python[i] = r_python
		time_jidt[i] = t_jidt
		time_python[i] = t_python


	# get results
	all_target_delays_jidt = defaultdict()
	all_selected_sources_jidt = defaultdict()
	all_selected_targets_jidt = defaultdict()
	all_selected_sources_te_jidt = defaultdict()
	all_am_jidt = defaultdict()
		
	all_target_delays_python = defaultdict()
	all_selected_sources_python = defaultdict()
	all_selected_targets_python = defaultdict()
	all_selected_sources_te_python = defaultdict()
	all_am_python = defaultdict()

	loop_source_true_jidt = np.zeros(5)
	loop_source_true_python = np.zeros(5)
	loop_source_te_true_jidt = np.zeros(5)
	loop_source_te_true_python = np.zeros(5)
	loop_target_delay_true_jidt = np.zeros(5)
	loop_target_delay_true_python = np.zeros(5)
	
	loop_source_true_jvsp = np.zeros(5)
	loop_source_te_true_jvsp = np.zeros(5)
	loop_target_delay_true_jvsp = np.zeros(5)
	
	for i in range(num_loops):
		target_delays_jidt = [None]*5
		selected_sources_jidt = [None]*5
		selected_targets_jidt = [None]*5
		selected_sources_te_jidt = [None]*5
		
		target_delays_python = [None]*5
		selected_sources_python = [None]*5
		selected_targets_python = [None]*5
		selected_sources_te_python = [None]*5

		for t in range(5):
			target_delays_jidt[t] = results_jidt[i].get_target_delays(t, fdr=False)
			target_delays_python[t] = results_python[i].get_target_delays(t, fdr=False)

			target_jidt = results_jidt[i].get_single_target(t, fdr=False)
			selected_sources_jidt[t] = target_jidt['selected_vars_sources']
			selected_targets_jidt[t] = target_jidt['selected_vars_target']
			selected_sources_te_jidt[t] = target_jidt[f'selected_sources_{measure}']
			
			target_python = results_python[i].get_single_target(t, fdr=False)
			selected_sources_python[t] = target_python['selected_vars_sources']
			selected_targets_python[t] = target_python['selected_vars_target']
			selected_sources_te_python[t] = target_python[f'selected_sources_{measure}']

		am_jidt = results_jidt[i].get_adjacency_matrix("max_te_lag",fdr=False)
		am_python = results_python[i].get_adjacency_matrix("max_te_lag",fdr=False)

		all_target_delays_jidt[i] = target_delays_jidt
		all_selected_sources_jidt[i] = selected_sources_jidt
		all_selected_targets_jidt[i] = selected_targets_jidt
		all_selected_sources_te_jidt[i] = selected_sources_te_jidt
		all_am_jidt[i] = am_jidt
			
		all_target_delays_python[i] = target_delays_python
		all_selected_sources_python[i] = selected_sources_python
		all_selected_targets_python[i] = selected_targets_python
		all_selected_sources_te_python[i] = selected_sources_te_python
		all_am_python[i] = am_python

	print("\n\n#########################################################################################################", file=outputfile)
	print(f"\nSummary reliability test network analysis {analysis} - {jidt_estimator} and {python_estimator}\n", file=outputfile)
	print(f"\nTested network analysis via {analysis} (nperms: {numperm})", file=outputfile)
	print(f"using mute data ({samples} samples, {reps} replications)\n", file=outputfile)
	print("#########################################################################################################\n", file=outputfile)
	
	print("\n########## selected var sources ##########\n", file=outputfile)
	print(f"\t\t\tJidt{est_type}CMI", file=outputfile)
	#print("\ntarget\tloop\tselected sources\t\tequal")
	print("\ntarget\tequal\tloop\tselected sources", file=outputfile)
	for t in range(5):
		st = f"{t}\t"
		s = ""
		eq = [False]*(num_loops-1)
		for r in range(num_loops):
			if r > 0:
				s += "\n\t\t\t"	
			s += f"{r}\t{all_selected_sources_jidt[r][t]}"
		
			if r > 0:
				if sorted(all_selected_sources_jidt[r][t]) == sorted(all_selected_sources_jidt[0][t]):
					eq[r-1] = True

		loop_source_true_jidt[t] = all(eq)
		se = f"{all(eq)}\t"	
		print(st,se,s,"\n", file=outputfile)

	print(f"\n\t\t\tPython{est_type}CMI", file=outputfile)
	print("\ntarget\tequal\tloop\tselected sources", file=outputfile)
	for t in range(5):
		st = f"{t}\t"
		s = ""
		eq = [False]*(num_loops-1)
		for r in range(num_loops):
			if r > 0:
				s += "\n\t\t\t"	
			s += f"{r}\t{all_selected_sources_python[r][t]}"
		
			if r > 0:
				if sorted(all_selected_sources_python[r][t]) == sorted(all_selected_sources_python[0][t]):
					eq[r-1] = True

		loop_source_true_python[t] = all(eq)
		se = f"{all(eq)}\t"	
		print(st,se,s,"\n", file=outputfile)

	print(f"\nJidt{est_type}CMI vs Python{est_type}CMI equal within replications\n", file=outputfile)
	sf = ""
	for r in range(num_loops):
		eq = [False]*5
		for t in range(5):
			if sorted(all_selected_sources_jidt[r][t])==sorted(all_selected_sources_python[r][t]):
				eq[t] = True

		
		sf += f"\tLoop {r}\t{all(eq)}\n"
	loop_source_true_jvsp[t] = all(eq)
	print(sf, file=outputfile)

	"""
	print("\nselected var target:\n")
	print(f"\t\t\tJidt{est_type}CMI")
	print("\ntarget\tloop\tselected targets\t\tequal")
	for t in range(5):
		s = f"{t}\t"
		eq = True
		for r in range(num_loops):
			if r > 0:
				s += "\n\t"	
			s += f"{r}\t{all_selected_targets_jidt[r][t]}"
		
			if r > 0:
				if sorted(all_selected_targets_jidt[r][t])!=sorted(all_selected_targets_jidt[1][t]):
					eq = False

		s += f"\t{eq}"	
		print(s)

	print(f"\n\t\t\tPython{est_type}CMI")
	print("\ntarget\tloop\tselected targets\t\tequal")
	for t in range(5):
		s = f"{t}\t"
		eq = True
		for r in range(num_loops):
			if r > 0:
				s += "\n\t"	
			s += f"{r}\t{all_selected_targets_python[r][t]}"
		
			if r > 0:
				if sorted(all_selected_targets_python[r][t])!=sorted(all_selected_targets_python[1][t]):
					eq = False

		s += f"\t{eq}"	
		print(s)

	print(f"\nJidt{est_type}CMI and Python{est_type}CMI equal within replications\n")

	sf = ""
	for r in range(num_loops):
		sf += f"\tLoop {r}\t{all_selected_targets_jidt[r]==all_selected_targets_python[r]}\n"
	print(sf)
	"""

	print("\n########## target delays ##########\n", file=outputfile)
	print(f"\t\t\tJidt{est_type}CMI", file=outputfile)
	print("\ntarget\tloop\ttarget delays\t\tequal", file=outputfile)
	for t in range(5):
		s = f"{t}\t"
		eq = [False]*(num_loops-1)
		for r in range(num_loops):
			if r > 0:
				s += "\n\t"	
			s += f"{r}\t{all_target_delays_jidt[r][t]}"
		
			if r > 0:
				try:
					if sorted(all_target_delays_jidt[r][t])==sorted(all_target_delays_jidt[0][t]):
						eq[r-1] = True
				except:
					continue
		loop_target_delay_true_jidt[t] = all(eq)
		s += f"\t\t{all(eq)}"	
		print(s, file=outputfile)

	print(f"\n\t\t\tPython{est_type}CMI", file=outputfile)
	print("\ntarget\tloop\ttarget delays\t\tequal", file=outputfile)
	for t in range(5):
		s = f"{t}\t"
		eq = [False]*(num_loops-1)
		for r in range(num_loops):
			if r > 0:
				s += "\n\t"	
			s += f"{r}\t{all_target_delays_python[r][t]}"
		
			if r > 0:
				try: 
					if sorted(all_target_delays_python[r][t])==sorted(all_target_delays_python[0][t]):
						eq[r-1] = True
				except:
					continue
		loop_target_delay_true_python[t] = all(eq)
		s += f"\t\t{all(eq)}"	
		print(s, file=outputfile)

	print(f"\nJidt{est_type}CMI vs Python{est_type}CMI equal within replications\n", file=outputfile)
	sf = ""
	for r in range(num_loops):
		eq = [False]*5
		for t in range(5):
			if sorted(all_target_delays_jidt[r][t])==sorted(all_target_delays_python[r][t]):
				eq[t] = True
		
		sf += f"\tLoop {r}\t{all(eq)}\n"
	loop_target_delay_true_jvsp[t] = all(eq)
	print(sf, file=outputfile)

	atol = 1e-03
	print(f"\n########## selected sources {measure.upper()} ##########\n", file=outputfile)
	print(f"\t\t\tJidt{est_type}CMI", file=outputfile)
	print(f"\ntarget\tclose {atol}\tloop\tselected sources {measure.upper()}", file=outputfile)
	for t in range(5):
		st = f"{t}\t"
		s = ""
		eq = [False]*(num_loops-1)
		for r in range(num_loops):
			if r > 0:
				s += "\n\t\t\t\t"	
			s += f"{r}\t{all_selected_sources_te_jidt[r][t]}"
		
			if r > 0:
				try:
					eq[r-1] = np.allclose(all_selected_sources_te_jidt[r][t], all_selected_sources_te_jidt[0][t], atol=atol)
				except:
					eq[r-1] = False

		loop_source_te_true_jidt[t] = all(eq)
		se = f"{all(eq)}\t\t"	
		print(st,se,s,"\n", file=outputfile)

	print(f"\n\t\t\tPython{est_type}CMI", file=outputfile)
	print(f"\ntarget\tclose {atol}\tloop\tselected sources {measure.upper()}", file=outputfile)
	for t in range(5):
		st = f"{t}\t"
		s = ""
		eq = [False]*(num_loops-1)
		for r in range(num_loops):
			if r > 0:
				s += "\n\t\t\t\t"	
			s += f"{r}\t{all_selected_sources_te_python[r][t]}"
		
			if r > 0:
				try:
					eq[r-1] = np.allclose(all_selected_sources_te_python[r][t], all_selected_sources_te_python[0][t], atol=atol)
				except:
					eq[r-1] = False

		loop_source_te_true_python[t] = all(eq)
		se = f"{all(eq)}\t\t"	
		print(st,se,s,"\n", file=outputfile)

	print(f"\nJidt{est_type}CMI vs Python{est_type}CMI close {atol} within replications\n", file=outputfile)
	sf = ""
	for r in range(num_loops):
		eq = [False]*5
		for t in range(5):
			try:
				eq[t] = np.allclose(all_selected_sources_te_jidt[r][t], all_selected_sources_te_python[r][t], atol=atol)
			except:
				eq[t] = False
		
		sf += f"\tLoop {r}\t{all(eq)}\n"
	loop_source_te_true_jvsp[t] = all(eq)
	print(sf, file=outputfile)


	####################################################### TODO orint edge lists and am

	print("\n\n mean calculation times:", file=outputfile)
	print(f" network_analysis {analysis} {jidt_estimator} nperms {numperm}: {np.mean(time_jidt)}", file=outputfile)
	print(f" network_analysis {analysis} {python_estimator} nperms {numperm}: {np.mean(time_python)}" , file=outputfile)

	return loop_source_true_jidt, loop_source_true_python, loop_target_delay_true_jidt, loop_target_delay_true_python, loop_source_te_true_jidt, loop_source_te_true_python, loop_source_true_jvsp, loop_target_delay_true_jvsp, loop_source_te_true_jvsp, time_jidt, time_python
	
def final_results_to_file(res, outputfile, sample_list, numperm_list):

	print(f"\t\tnum perm", file=outputfile)
	s = "samples\t"
	for j in numperm_list:
		s += f"{j}\t"

	print(s, file=outputfile)


	scount = 0
	for i in sample_list:
		s = f"{i}\t\t"
		pcount = 0
		for j in numperm_list:
			s += f"{int(res[scount, pcount])}\t"

			pcount += 1

		print(s, file=outputfile)

		scount += 1

	print("\n", file=outputfile)


if __name__ == '__main__':

	num_loops = 5

	analysis = "BivariateMI"
	est_type = "Discrete"
	#numperm = 500
	#samples = 1000
	reps = 3
	verbose = False
	nbins = 5

	sample_list = [100, 250,500,750,1000]

	numperm_list = [21, 100, 300, 500]


	if est_type == "Discrete":
		filename = f"rel_test_network_analysis_{num_loops}loops_{reps}reps_{analysis}_{est_type}_nbins{nbins}.txt"
	else:
		filename = f"rel_test_network_analysis_{num_loops}loops_{reps}reps_{analysis}_{est_type}.txt"
	outputfile = open(filename, 'w')

	source_true_jidt = np.zeros((len(sample_list), len(numperm_list)))
	source_true_python = np.zeros((len(sample_list), len(numperm_list)))
	
	target_delay_true_jidt = np.zeros((len(sample_list), len(numperm_list)))
	target_delay_true_python = np.zeros((len(sample_list), len(numperm_list)))

	source_te_true_jidt = np.zeros((len(sample_list), len(numperm_list)))
	source_te_true_python = np.zeros((len(sample_list), len(numperm_list)))

	source_true_jvsp = np.zeros((len(sample_list), len(numperm_list)))
	target_delay_true_jvsp = np.zeros((len(sample_list), len(numperm_list)))
	source_te_true_jvsp = np.zeros((len(sample_list), len(numperm_list)))
	
	mean_time_jidt = np.zeros((len(sample_list), len(numperm_list)))
	mean_time_python = np.zeros((len(sample_list), len(numperm_list)))
	
	measure = analysis[-2:].lower()
	
	scount = 0
	for samples in sample_list:
		pcount = 0
		for numperm in numperm_list:
	
			loop_source_true_jidt, loop_source_true_python, loop_target_delay_true_jidt, loop_target_delay_true_python, loop_source_te_true_jidt, loop_source_te_true_python, loop_source_true_jvsp, loop_target_delay_true_jvsp, loop_source_te_true_jvsp, time_jidt, time_python = test_network_analysis_loop(analysis, 
				est_type, 
				outputfile=outputfile,
				num_loops=num_loops, 
				numperm=numperm, 
				samples=samples, 
				reps=reps, 
				verbose=verbose,
				nbins=nbins,
				)


			print("loop_source_true_jidt")
			print(loop_source_true_jidt)
			print("loop_source_true_python")
			print(loop_source_true_python)
			
			print("loop_target_delay_true_jidt") 
			print(loop_target_delay_true_jidt) 
			print("loop_target_delay_true_python")
			print(loop_target_delay_true_python)
			
			print(loop_source_te_true_jidt)
			print(loop_source_te_true_python)
			print(loop_source_true_jvsp)
			print(loop_target_delay_true_jvsp)
			print(loop_source_te_true_jvsp)


			source_true_jidt[scount, pcount] = sum(loop_source_true_jidt)
			source_true_python[scount, pcount] = sum(loop_source_true_python)

			target_delay_true_jidt[scount, pcount] = sum(loop_target_delay_true_jidt)
			target_delay_true_python[scount, pcount] = sum(loop_target_delay_true_python)

			source_te_true_jidt[scount, pcount] = sum(loop_source_te_true_jidt)
			source_te_true_python[scount, pcount] = sum(loop_source_te_true_python)

			source_true_jvsp[scount, pcount] = sum(loop_source_true_jvsp)
			target_delay_true_jvsp[scount, pcount] = sum(loop_target_delay_true_jvsp)
			source_te_true_jvsp[scount, pcount] = sum(loop_source_te_true_jvsp)
			
			mean_time_jidt[scount, pcount] = np.mean(time_jidt) 
			mean_time_python[scount, pcount] = np.mean(time_python) 

			pcount += 1

		scount += 1


	print(f"\n\n========================================================================================", file=outputfile)
	print(f"\nFinal summary reliability test network analysis {analysis} - Jidt{est_type}CMI and Python{est_type}CMI\n", file=outputfile)
	print(f"\nTested network analysis via {analysis}", file=outputfile)
	print(f"using mute data ({reps} replications)\n", file=outputfile)
	print(f"Tested number or samples: {sample_list})", file=outputfile)
	print(f"Tested number or permutations: {numperm_list})", file=outputfile)
	print(f"\n========================================================================================\n", file=outputfile)
	


	print(f"\nNumber of targets (max 5) showing CONSISTENT results over all {num_loops} loops", file=outputfile)
	#print(f"All following results printes in {len(sample_list)} samples x {len(numperm_list)} num perms)\n", file=outputfile)
	
	print("### selected sources\n", file=outputfile)
	print(f"\t\tJidt{est_type}CMI:", file=outputfile)
	final_results_to_file(source_true_jidt, outputfile, sample_list, numperm_list)
	print(f"\t\tPython{est_type}CMI:", file=outputfile)
	final_results_to_file(source_true_python, outputfile, sample_list, numperm_list)
	print(f"\tJidt{est_type}CMI vs Python{est_type}CMI:\n", file=outputfile)
	final_results_to_file(source_true_jvsp, outputfile, sample_list, numperm_list)
	
	print("### target delays\n", file=outputfile)
	print(f"\t\tJidt{est_type}CMI:\n", file=outputfile)
	final_results_to_file(target_delay_true_jidt, outputfile, sample_list, numperm_list)
	print(f"\t\tPython{est_type}CMI:", file=outputfile)
	final_results_to_file(target_delay_true_python, outputfile, sample_list, numperm_list)
	print(f"\tJidt{est_type}CMI vs Python{est_type}CMI:\n", file=outputfile)
	final_results_to_file(target_delay_true_jvsp, outputfile, sample_list, numperm_list)
	
	print(f"### selected sources {measure}\n", file=outputfile)
	print(f"\t\tJidt{est_type}CMI:", file=outputfile)
	final_results_to_file(source_te_true_jidt, outputfile, sample_list, numperm_list)
	print(f"\t\tPython{est_type}CMI:", file=outputfile)
	final_results_to_file(source_te_true_python, outputfile, sample_list, numperm_list)
	print(f"\tJidt{est_type}CMI vs Python{est_type}CMI:\n", file=outputfile)
	final_results_to_file(source_te_true_jvsp, outputfile, sample_list, numperm_list)
	
	print(f"\n mean calculation time", file=outputfile)
	print(f"(samples x num perms)\n", file=outputfile)
	print(f"\n\t\tJidt{est_type}CMI:", file=outputfile)
	print(mean_time_jidt, file=outputfile)
	print(f"\n\t\tPython{est_type}CMI:\n", file=outputfile)
	print(mean_time_python, file=outputfile)
	print(f"\n\t\t~ percent faster:\n", file=outputfile)
	print(np.rint(mean_time_python/mean_time_jidt*100), file=outputfile)

	#print(f"", file=outputfile)
	#print(f"", file=outputfile)
