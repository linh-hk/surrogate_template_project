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
    stats_list = ['pearson', 'mutual_info']
    
    data, datagen_param = load_results_params(f'{sys.argv[1]}')
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
    with MPIPoolExecutor() as executor:
        resultsIter = executor.map(cst.for_generated_data,
                                   np.arange(reps), 
                                   data, 
                                   [f'{sys.argv[1]}']*reps,
                                   [test_list]*reps, 
                                   [stats_list]*reps,
                                   [test_param]*reps,
                                   unordered=True)
        # https://stackoverflow.com/questions/3459098/create-list-of-single-item-repeated-n-times
        # create a list of repeated elements - Google
        resultsList = [_ for _ in resultsIter] # list(resultsIter)
    saveP = {'pvals' : resultsList,
             'stats_list' : stats_list,
             'test_list' : test_list,
             'test_params' : test_param}
    
    tests = '_'.join(stats_list + test_list + [str(_) for _ in test_param.values() if _ != None])
    with open(f'{sys.argv[1]}/{tests}.pkl', 'wb') as fi:
        pickle.dump(saveP, fi);
            
            #np.savetxt(fname,resultsList);
    sys.stdout.flush();
    sys.stdout.write('Total time: %5.2f seconds\n' % (time.time() - start));
    sys.stdout.flush();
