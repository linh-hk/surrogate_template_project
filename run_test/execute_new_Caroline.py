#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 18:15:37 2023

@author: h_k_linh

This script is submited to SGE on UCL cluster as a task. 
What it does is:
    Load data that Caroline sent
        Which data to load is using the first argument passed from qsub script.
        The second argument from the qsub script is the surrogate protocol.
    Run the dependence test on the data in parallel and save results on cluster.
Note: The imported data is not changed to test for false positive rate in this script

"""
import os
print(f'working directory: {os.getcwd()}')
# os.chdir('/home/h_k_linh/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test')
# os.getcwd()

# import GenerateData as dataGen
import numpy as np
# import Correlation_Surrogate_tests as cst

import sys # to save name passed from cmd
import time

# import dill # load and save data
import pickle # load and save data

# from mpi4py.futures import MPIPoolExecutor
sys.path.append('/home/hoanlinh/Simulation_test/Simulation_code/surrogate_dependence_test')
import main as sdt

# list of data_name:
    # ['competitive_xdom_1extinct_fastsampling_data_100reps',
    # 'competitive_xdom_1extinct_slowsampling_data_100reps',
    # 'competitive_xdom_coexist_slowsampling_data_100reps',
    # 'lorenz_logistic_map_data_100reps',
    # 'predprey_data_100reps']
#%%
def load_data(data_name):
    # names = 'caroline_LvCh_FitzHugh_100'
    fi = f"run_this_for_Caroline/{data_name}/data.npy"
    data = np.load(fi)
    return [data[:,:,_] for _ in np.arange(data.shape[2])]
    # if 'comp' in data_name or 'predprey' in data_name:
    #     return [data[:,:-1,_] for _ in np.arange(data.shape[2])]
    # if 'lorenz' in data_name:
    #     return [data[:,:,_] for _ in np.arange(data.shape[2])]

def run_each_ts(pair, pair_id, stats_list, test_list, maxlag):
    x = pair[0]
    y = pair[1]
    return [pair_id, sdt.manystats_manysurr(x, y, stats_list, test_list, maxlag)]

if __name__=="__main__":
    stats_list = ['ccm_y->x', 'ccm_x->y', 'granger_y->x', 'granger_x->y']
    test_list = [sys.argv[2]] # 'tts', 'twin','randphase'
    if 'predprey' in sys.argv[1]:
        maxlag = 5
    else:
        maxlag = 0
    
    print(f"Running {sys.argv[1]} data, {sys.argv[2]}, start time {time.time()}")
    data = load_data(f'{sys.argv[1]}')
    print(stats_list, test_list, maxlag)
    
    # ARGs = []
    resultsList = []
    start = time.time()
    for series in enumerate(data):
        ARGs = (series[1], series[0], stats_list, test_list, maxlag)
        # print(ARGs)
        resultsList.append(run_each_ts(*ARGs))
        print(f'Series #{series[0]} run in {time.time()-start}')
    # with MPIPoolExecutor() as executor:
    #     resultsIter = executor.map(run_each_ts, ARGs, unordered=True)
    #     resultsList = [_ for _ in resultsIter]
    
    saveP = {'pvals' : resultsList,
             'stats_list' : stats_list,
             'test_list' : test_list}
    
    tests = '_'.join(['ccms', 'grangers'] + test_list)
    with open(f'run_this_for_Caroline/{sys.argv[1]}/{tests}_Linh.pkl', 'wb') as fi:
        pickle.dump(saveP, fi);
            
            #np.savetxt(fname,resultsList);
    sys.stdout.flush();
    sys.stdout.write('Total time: %5.2f seconds\n' % (time.time() - start));
    sys.stdout.flush();
