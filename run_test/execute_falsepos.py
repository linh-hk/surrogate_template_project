#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 18:15:37 2023

@author: h_k_linh

qsub    ...             {surr_proc}     {data_name}     {N_0}           {which_data}
qsub    sys.argv[0]     sys.argv[1]     sys.argv[2]     sys.argv[3]     sys.argv[4]

This script is submited to SGE on UCL cluster as a task. 
What it does is:
    Load simulated data that is in ../Simulated_data
        Which data to load is using the first argument passed from qsub script.
        The second argument from the qsub script specifies which pairs of time series (index of simulated data) will be run. This came about because I simulated 1000 simulations in total and only want to run test on 100 simulations per run.
    Run the dependence test on the data in parallel and save results on cluster.
Note:
This script particularly use the randomphase protocol to general surrogate test
The imported data is changed to test for false positive rate in this script
Integrated multiprocessor into the workflow ('new' in file name)
This script uses multiprocessor to excecute on cluster, rather than MPI4py ('2' in file name)

"""
import os
print(f'working directory: {os.getcwd()}')
# os.chdir('/home/h_k_linh/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test')
# os.getcwd()

# import GenerateData as dataGen
import numpy as np
# import Correlation_Surrogate_tests as cst
# from scipy import stats

import sys # to save name passed from cmd
import time

# import dill # load and save data
import pickle # load and save data

# from mpi4py.futures import MPIPoolExecutor
sys.path.append('/home/hoanlinh/Simulation_test/Simulation_code/surrogate_dependence_test')
import main as sdt
#%%
def load_data(data_name, suffix = ''):
    if 'xy_' in data_name:
        sampdir = f'Simulated_data/{data_name}'
    elif '500' in data_name:
        sampdir = f'Simulated_data/LVextra/{data_name}'
    with open(f'{sampdir}/data{suffix}.pkl', 'rb') as fi:
        data = pickle.load(fi)
    # return data['data'], data['datagen_params']
    # for false pos
    num_trials = data['datagen_params']['N']
    data_fp = [[data['data'][_][0], data['data'][0 if _ == num_trials - 1 else _+1][1]] 
              for _ in range(num_trials)]
    return data_fp, data['datagen_params']

def run_each_ts(pair, pair_id, stats_list, test_list, maxlag):
    x = pair[0]
    y = pair[1]
    return {pair_id: sdt.manystats_manysurr(x, y, stats_list, test_list, maxlag),
            'XY': np.array([x,y])}

if __name__=="__main__":
    stats_list = ['pearson', 'lsa', 'mutual_info', 'ccm_y->x', 'ccm_x->y', 'granger_y->x', 'granger_x->y']
    test_list = [sys.argv[1]] # , 'twin','randphase'
    maxlag = 0
    
    data_name = sys.argv[2]
    N_0 = int(sys.argv[3])
    which_data = sys.argv[4]
    print(f"Running {data_name} sample, data{which_data}.pkl {N_0} at {time.time()}, with {' '.join(test_list)} + {' '.join(stats_list)}, falsepos")
    data, datagen_param = load_data(data_name, which_data)
    
    print(f'Sequencing number {N_0} to {N_0+100}')
    data = data[N_0:N_0+100]
    # ARGs = []
    resultsList = []
    start = time.time()
    for series in enumerate(data):
        ARGs = (series[1], N_0+series[0],stats_list, test_list, maxlag)
        # print(ARGs)
        resultsList.append(run_each_ts(*ARGs))
        print(f'Series #{series[0]} run in {time.time()-start}')
    # with MPIPoolExecutor() as executor:
    #     resultsIter = executor.map(run_each_ts, ARGs, unordered=True)
    #     resultsList = [_ for _ in resultsIter]
    
    saveP = {'pvals' : resultsList,
             'stats_list' : stats_list,
             'test_list' : test_list,
             'nsurr' : 99}
    
    if which_data == '':
        tests = '_'.join(test_list+['nolag', 'falsepos']) #, str(N_0)
    else:
        tests = '_'.join(test_list+['nolag', 'falsepos', str(N_0)]) #
    if 'xy_' in data_name:
        fiS = f"Simulated_data/{data_name}/{tests}.pkl"
    elif '500' in data_name: 
        fiS = f'Simulated_data/LVextra/{data_name}/{tests}.pkl'
    print(f'Saving at {fiS}')
    with open(fiS, 'wb') as file:
        pickle.dump(saveP, file);
            
            #np.savetxt(fname,resultsList);
    sys.stdout.flush();
    sys.stdout.write('Total time: %5.2f seconds\n' % (time.time() - start));
    sys.stdout.flush();
