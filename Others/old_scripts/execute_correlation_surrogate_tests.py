#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 10:48:49 2023

@author: h_k_linh

Multiple processing
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
# fi = 'results_Generated_data.pkl'
# dill.load_session(fi)
# print(f"{fi} session loaded")
#%% Generate data
if __name__=="__main__":
    data_list_all = ['xy_ar_u', 'xy_ar_u2', 'xy_uni_logistic', 'xy_sinewn', 
                  'ts_ar', 'ts_logistic', 'ts_sine_intmt', 'ts_coinflip', 
                  'ts_noise_wv_kurtosis', 'xy_FitzHugh_Nagumo', 'ts_chaoticLV', 
                  'xy_Caroline_LV', 'xy_Caroline_CH'] # 13
    test_list_all = ['randphase', 'bstrap', 'twin', 'tts', 'tts_naive', 'tts_multishift', 'circperm','perm']
    stats_list_all = ['pearson', 'lsa', 'mutual_info', 
                  'granger_y->x', 'granger_x->y', 
                  'ccm_y->x', 'ccm_x->y',
                  'pcc_param', 'granger_param_y->x', 'granger_param_x->y', 'lsa_data_driven']

    data_list = ['xy_Caroline_LV', 'xy_Caroline_CH','xy_FitzHugh_Nagumo']
    test_list = ['tts', 'tts_naive']
    stats_list = ['pearson', 'mutual_info']
    
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
                     'intx' : "competitive",
                     'maxlag': 4}
    reps = int(sys.argv[2])
    print('Staring running MPI')
    start = time.time()
    with MPIPoolExecutor() as executor:
        resultsIter = executor.map(cst.for_replicates, np.arange(reps), 
                                   [data_list]*reps, 
                                   [test_list]*reps, 
                                   [stats_list]*reps, 
                                   [datagen_param]*reps, 
                                   unordered=True)
        # https://stackoverflow.com/questions/3459098/create-list-of-single-item-repeated-n-times
        # create a list of repeated elements - Google
        resultsList = [_ for _ in resultsIter] # list(resultsIter)
    
    os.mkdir(f'{sys.argv[1]}')
        
    filename = f'{sys.argv[1]}/{sys.argv[1]}_testing.pkl'
    print(f"Saving all_results list at {filename}"); 
    with open(filename, 'wb') as fi:
        pickle.dump(resultsList, fi);
    print(f'Done saving all_results at {filename}');         
    
    parameters = {'reps':reps, 'data_list':data_list, 'test_list': test_list, 'stats_list':stats_list, 'datagen_params':datagen_param}
    paraname = f'{sys.argv[1]}/{sys.argv[1]}_parameters.pkl'
    print(f"Saving parameters at {paraname}"); 
    with open(paraname, 'wb') as fi:
        pickle.dump(parameters, fi);
    print(f'Done saving parameters at {paraname}');
        
            #np.savetxt(fname,resultsList);
    sys.stdout.flush();
    sys.stdout.write('Total time: %5.2f seconds\n' % (time.time() - start));
    sys.stdout.flush();
    

# .spydata
# https://docs.spyder-ide.org/current/panes/variableexplorer.html#toolbar-buttons

#%% Load data
# dill.load_session(filename)

# with open('Correlation_simulation.pkl', 'rb') as file:
#     # Call load method to deserialze
#     test = pickle.load(file)
  
#     print(myvar)
 #%%% Save generated data
 # filename = f'results_{sys.argv[1]}_Generated_data.pkl'
 # print(f'Saving Generated data to {filename}')
 # dill.dump_session(filename)
 # print(f'Done saving generated data at {filename}')
