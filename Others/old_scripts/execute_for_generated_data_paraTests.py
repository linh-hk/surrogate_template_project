#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 17:53:11 2023

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

from mpi4py.futures import MPIPoolExecutor

#%% Load generated data
def load_results_params(names):
    # names = 'caroline_LvCh_FitzHugh_100'
    fi = f"{names}/{names}_testing.pkl"
    with open(fi, 'rb') as file:
        results = pickle.load(file)
        
    fi = f"{names}/{names}_parameters.pkl"
    with open(fi, 'rb') as file:
        params = pickle.load(file)
        
    return results, params

if __name__=="__main__":
    results, params = load_results_params(f'{sys.argv[1]}')
    
    # del(names, fi, file)
    #%%
    pvals   =   [ _['pvals'] for _ in results]
    data    =   [ _['XY'] for _ in results]
    # time    =   [ _['runtime'] for _ in results]
    # run_id  =   [ _['run_id'] for _ in results]
    #%%
    test_list_all = ['randphase', 'bstrap', 'twin', 'tts', 'tts_naive', 'tts_multishift', 'circperm','perm']
    stats_list_all = ['pearson', 'lsa', 'mutual_info', 
                  'granger_y->x', 'granger_x->y', 
                  'ccm_y->x', 'ccm_x->y',
                  'pcc_param', 'granger_param_y->x', 'granger_param_x->y', 'lsa_data_driven']

    data_list = data[0].keys()
    test_list = ['tts', 'tts_naive']
    stats_list = ['pcc_param', 'granger_param_y->x', 'granger_param_x->y', 'lsa_data_driven']
    datagen_param = {'r': np.array([1, 0.72, 1.53, 1.27]),
                     'a': np.array([[1, 1.09, 1.52, 0],
                          [0, 1, 0.44, 1.36],
                          [2.33, 0, 1, 0.47],
                          [1.21, 0.51, 0.35, 1]]), 
                     'run_id' : 1, 
                     'dt_s' : 1.25, 
                     'N' : 500, 
                     'noise' : 0.01,
                     'noise_T' : 0.05,
                     'intx' : "predprey",
                     'maxlag': 4}
    reps = len(data)
    print('Staring running MPI')
    start = time.time()
    with MPIPoolExecutor() as executor:
        resultsIter = executor.map(cst.for_generated_data, np.arange(reps), 
                                   data, 
                                   [test_list]*reps, 
                                   [stats_list]*reps, 
                                   [datagen_param]*reps, 
                                   unordered=True)
        # https://stackoverflow.com/questions/3459098/create-list-of-single-item-repeated-n-times
        # create a list of repeated elements - Google
        resultsList = [_ for _ in resultsIter] # list(resultsIter)
    
    prefix = f'{sys.argv[1]}_{sys.argv[2]}'
    os.mkdir(prefix)
        
    filename = f'{prefix}/{prefix}_testing.pkl'
    print(f"Saving all_results list at {filename}"); 
    with open(filename, 'wb') as fi:
        pickle.dump(resultsList, fi);
    print(f'Done saving all_results at {filename}');         
    
    parameters = {'reps':reps, 'data_list':data_list, 'test_list': test_list, 'stats_list':stats_list, 'datagen_params':datagen_param}
    paraname = f'{prefix}/{prefix}_parameters.pkl'
    print(f"Saving parameters at {paraname}"); 
    with open(paraname, 'wb') as fi:
        pickle.dump(parameters, fi);
    print(f'Done saving parameters at {paraname}');
        
            #np.savetxt(fname,resultsList);
    sys.stdout.flush();
    sys.stdout.write('Total time: %5.2f seconds\n' % (time.time() - start));
    sys.stdout.flush();
