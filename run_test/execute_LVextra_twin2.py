#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 18:15:37 2023

@author: h_k_linh

This script is submited to SGE on UCL cluster as a task. 
What it does is:
    Load simulated data that is in ../Simulated_data
        Which data to load is using the first argument passed from qsub script.
        The second argument from the qsub script specifies which pairs of time series (index of simulated data) will be run. This came about because I simulated 1000 simulations in total and only want to run test on 100 simulations per run.
    Run the dependence test on the data in parallel and save results on cluster.
Note:
This script particularly use the twin protocol to general surrogate test
The imported data is not changed to test for false positive rate in this script
Integrated multiprocessor into the workflow ('new' in file name)
This script uses multiprocessor to excecute on cluster, rather than MPI4py ('2' in file name)

"""
import os
print(f'working directory: {os.getcwd()}')
# os.chdir('/home/h_k_linh/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test')
# os.getcwd()

# import GenerateData as dataGen
# import numpy as np
# import Correlation_Surrogate_tests as cst

import sys # to save name passed from cmd
import time

# import dill # load and save data
import pickle # load and save data

# from mpi4py.futures import MPIPoolExecutor
sys.path.append('/home/hoanlinh/Simulation_test/Simulation_code/surrogate_dependence_test')
import main as sdt
#%%
def load_results_params(data_name):
    # names = 'caroline_LvCh_FitzHugh_100'
    fi = f"Simulated_data/LVextra/{data_name}/data.pkl"
    with open(fi, 'rb') as file:
        data = pickle.load(file)
    return data['data'], data['datagen_params']

def run_each_ts(pair, pair_id, stats_list, test_list, maxlag):
    x = pair[0]
    y = pair[1]
    return {pair_id: sdt.manystats_manysurr(x, y, stats_list, test_list, maxlag)}

if __name__=="__main__":
    stats_list = ['pearson', 'lsa', 'mutual_info', 'ccm_y->x', 'ccm_x->y', 'granger_y->x', 'granger_x->y']
    test_list = ['twin'] # , 'twin','randphase'
    maxlag = 0
    
    # print(f"Loading {sys.argv[1]} data, {int(sys.argv[2])} {time.time()}")
    data, datagen_param = load_results_params(f'{sys.argv[1]}')
    
    # print(f'Sequencing number {sys.argv[2]} to {int(sys.argv[2])+100}')
    # data = data[int(sys.argv[2]):int(sys.argv[2])+100]
    # ARGs = []
    resultsList = []
    start = time.time()
    for series in enumerate(data):
        ARGs = (series[1], series[0],stats_list, test_list, maxlag)
        # print(ARGs)
        resultsList.append(run_each_ts(*ARGs))
        print(f'Series #{series[0]} run in {time.time()-start}')
    # with MPIPoolExecutor() as executor:
    #     resultsIter = executor.map(run_each_ts, ARGs, unordered=True)
    #     resultsList = [_ for _ in resultsIter]
    
    saveP = {'pvals' : resultsList,
             'stats_list' : stats_list,
             'test_list' : test_list}
    
    tests = '_'.join(test_list+['nolag'])
    with open(f'Simulated_data/LVextra/{sys.argv[1]}/{tests}.pkl', 'wb') as fi:
        pickle.dump(saveP, fi);
            
            #np.savetxt(fname,resultsList);
    sys.stdout.flush();
    sys.stdout.write('Total time: %5.2f seconds\n' % (time.time() - start));
    sys.stdout.flush();
