#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 10:23:56 2023

@author: h_k_linh
"""
import os
print(f'working directory: {os.getcwd()}')
# os.chdir('/home/h_k_linh/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test')
# os.getcwd()

# import GenerateData as dataGen
import numpy as np
import Correlation_Surrogate_tests as cst

import sys # to save name passed from cmd
import time

# import dill # load and save data
import pickle # load and save data

def load_results_params(data_name):
    # names = 'caroline_LvCh_FitzHugh_100'
    fi = f"{data_name}/data.pkl"
    with open(fi, 'rb') as file:
        data = pickle.load(file)
    return data['data'], data['datagen_params']

if __name__=="__main__":
    test_list_all = ['randphase', 'bstrap', 'twin', 'tts', 'tts_naive', 'tts_multishift', 'circperm','perm']
    stats_list_all = ['pearson', 'lsa', 'mutual_info', 
                  'granger_y->x', 'granger_x->y', 
                  'ccm_y->x', 'ccm_x->y',
                  'pcc_param', 'granger_param_y->x', 'granger_param_x->y', 'lsa_data_driven']
    
    test_list = ['tts', 'tts_naive']
    stats_list = ['granger_y->x', 'granger_x->y']
    
    data, datagen_param = load_results_params('ts_chaoticLV')
    test_param = {'r_tts': cst.choose_r(datagen_param['N']),
                  'r_naive': cst.choose_r(datagen_param['N']), 
                  'maxlag': 4}

    if 'tts' not in test_list: 
        test_param['r_tts'] = None
    if 'tts_naive' not in test_list: 
        test_param['r_naive'] = None
    
    reps = len(data)
    print('Staring running MPI')
    start = time.time()
    results = []
    for i in range(reps):
        print(i)
        results.append(cst.for_generated_data(i, data[i], ['ts_chaoticLV'], test_list, stats_list, test_param))
        print(i)
        
    print(time.time()-start)

11
12
Processing ['ts_chaoticLV']
Surrogates of Y
Start getting significant values
	Start calculating statistics for significant value: 
		['tts', 'tts_naive']
			Trimming data to remove discontinuity
			Creating tts surrogates
				Calculating granger_stat for tts surrogates
						Scanning best lags with step lags = 2
				Calculating Granger
						lag: x_2 and y_-2
				Calculating Granger
				Calculating Granger
						lag: x_4 and y_-4
				Calculating Granger
				Calculating Granger
						lag: x_6 and y_-6
				Calculating Granger
				Calculating Granger
						lag: x_8 and y_-8
				Calculating Granger
				Calculating Granger
						Scanning best lags with step lags = 2
				Calculating Granger
						lag: x_2 and y_-2
				Calculating Granger
				Calculating Granger
						lag: x_4 and y_-4
				Calculating Granger
				Calculating Granger
						lag: x_6 and y_-6
				Calculating Granger
				Calculating Granger
						lag: x_8 and y_-8
				Calculating Granger
				Calculating Granger
			Creating tts surrogates
				Calculating granger_stat for tts surrogates
						Scanning best lags with step lags = 2
				Calculating Granger
						lag: x_2 and y_-2
				Calculating Granger
				Calculating Granger
						lag: x_4 and y_-4
				Calculating Granger
				Calculating Granger
						lag: x_6 and y_-6
				Calculating Granger
				Calculating Granger
						lag: x_8 and y_-8
				Calculating Granger
				Calculating Granger
						Scanning best lags with step lags = 2
				Calculating Granger
						lag: x_2 and y_-2
				Calculating Granger
				Calculating Granger
						lag: x_4 and y_-4
				Calculating Granger
				Calculating Granger
						lag: x_6 and y_-6
				Calculating Granger
				Calculating Granger
						lag: x_8 and y_-8
				Calculating Granger
				Calculating Granger
	Start calculating statistics for significant value: 
		['tts', 'tts_naive']
			Trimming data to remove discontinuity
			Creating tts surrogates
				Calculating granger_stat for tts surrogates
						Scanning best lags with step lags = 2
				Calculating Granger
						lag: x_2 and y_-2
				Calculating Granger
				Calculating Granger
						lag: x_4 and y_-4
				Calculating Granger
				Calculating Granger
						lag: x_6 and y_-6
				Calculating Granger
				Calculating Granger
						lag: x_8 and y_-8
				Calculating Granger
				Calculating Granger
						Scanning best lags with step lags = 2
				Calculating Granger
						lag: x_2 and y_-2
				Calculating Granger
				Calculating Granger
Traceback (most recent call last):

  File ~/mambaforge/envs/timeseries/lib/python3.11/site-packages/spyder_kernels/py3compat.py:356 in compat_exec
    exec(code, globals, locals)

  File ~/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/untitled0.py:56
    results.append(cst.for_generated_data(i, data[i], ['ts_chaoticLV'], test_list, stats_list, test_param))

  File ~/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/Correlation_Surrogate_tests.py:951 in for_generated_data
    results, runtime = surrX_and_surrY(data, data_name, test_list, stats_list, test_param['maxlag'], test_param['r_tts'],test_param['r_naive'])

  File ~/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/Correlation_Surrogate_tests.py:874 in surrX_and_surrY
    results['surrY'], runtime['surrY'] = benchmark_stats(data[0], data[1], test_list= test_list, stats_list=stats_list, maxlag=maxlag, r_tts=r_tts, r_naive=r_naive)

  File ~/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/Correlation_Surrogate_tests.py:842 in benchmark_stats
    pvals['granger_x->y'], runtime['granger_x->y'] = get_sigf(y,x,granger_stat,test_list=test_list,maxlag=maxlag, r_tts=r_tts, r_naive=r_naive);

  File ~/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/Correlation_Surrogate_tests.py:792 in get_sigf
    B = tts_test(x, y, r, statistic,maxlag);

  File ~/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/Correlation_Surrogate_tests.py:480 in tts
    importance_surr = np.max(scan_lags(xstar, y_surr.T, statistic,maxlag)[1:],axis=1);

  File ~/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/Correlation_Surrogate_tests.py:707 in scan_lags
    score[:,maxlag-i] = statistic(x[:-2*i],y[2*i:],**kw_statistic);

  File ~/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/Correlation_Surrogate_tests.py:663 in granger_stat
    res = model.fit(maxlags=15,ic='aic');

  File ~/mambaforge/envs/timeseries/lib/python3.11/site-packages/statsmodels/tsa/vector_ar/var_model.py:660 in fit
    selections = self.select_order(maxlags=maxlags)

  File ~/mambaforge/envs/timeseries/lib/python3.11/site-packages/statsmodels/tsa/vector_ar/var_model.py:824 in select_order
    result = self._estimate_var(p, offset=maxlags - p, trend=trend)

  File ~/mambaforge/envs/timeseries/lib/python3.11/site-packages/statsmodels/tsa/vector_ar/var_model.py:744 in _estimate_var
    params = np.linalg.lstsq(z, y_sample, rcond=1e-15)[0]

  File <__array_function__ internals>:200 in lstsq

  File ~/mambaforge/envs/timeseries/lib/python3.11/site-packages/numpy/linalg/linalg.py:2285 in lstsq
    x, resids, rank, s = gufunc(a, b, rcond, signature=signature, extobj=extobj)

  File ~/mambaforge/envs/timeseries/lib/python3.11/site-packages/numpy/linalg/linalg.py:101 in _raise_linalgerror_lstsq
    raise LinAlgError("SVD did not converge in Linear Least Squares")

LinAlgError: SVD did not converge in Linear Least Squares