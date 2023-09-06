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
This script particularly use the tts protocol to general surrogate test
The imported data is not changed to test for false positive rate in this script
Integrated multiprocessor into the workflow ('new' in file name)
This script uses multiprocessor to excecute on cluster, rather than MPI4py ('2' in file name)

"""
import os
print(f'working directory: {os.getcwd()}')
# os.chdir('C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/')
# os.chdir('/home/h_k_linh/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test')
# os.getcwd()

# import GenerateData as dataGen
import numpy as np

import sys # to save name passed from cmd
import time

# import dill # load and save data
import pickle # load and save data

# from mpi4py.futures import MPIPoolExecutor

sys.path.append('/home/hoanlinh/Simulation_test/Simulation_code/surrogate_dependence_test')
from GenerateData import generate_lv

def save_data(filename, data):
    if not os.path.exists(f'Simulated_data/LVextra/{filename}/'): 
        os.makedirs(f'Simulated_data/LVextra/{filename}/')
    with open(f'Simulated_data/LVextra/{filename}/data.pkl', 'wb') as fi:
        pickle.dump(data, fi)
#%%
if __name__ == '__main__': 
    
    ARGs= []
    # EComp
    # ARGs = {'mode': 'EComp', 'dt_s': 0.25, 'N': 500, 's0': np.array([2.,0.]), 'mu': np.array([0.7,0.7]), 'M': np.array([[-0.4,-0.5],[-0.5,-0.4]]), 'noise': 0.01, 'noise_T': 0.05}
    # ARGs.append({'mode': 'EComp', 'dt_s': 1.25, 'N': 500, 's0': np.array([2.,0.]), 'mu': np.array([0.7,0.7]), 'M': np.array([[-0.4,-0.5],[-0.5,-0.4]]), 'noise': 0.01, 'noise_T': 0.05})
    # ARGs.append({'mode': 'EComp', 'dt_s': 0.25, 'N': 500, 's0': np.array([1.,1.]), 'mu': np.array([0.7,0.7]), 'M': np.array([[-0.4,-0.5],[-0.5,-0.4]]), 'noise': 0.01, 'noise_T': 0.05})
    # ARGs = {'mode': 'EComp', 'dt_s': 1.25, 'N': 500, 's0': np.array([1.,1.]), 'mu': np.array([0.7,0.7]), 'M': np.array([[-0.4,-0.5],[-0.5,-0.4]]), 'noise': 0.01, 'noise_T': 0.05}
    
    # EMut
    # ARGs.append({'mode': 'EMut', 'dt_s': 0.25, 'N': 500, 's0': np.array([2.,0.]), 'mu': np.array([0.7,0.7]), 'M': np.array([[-0.4,0.3],[0.3,-0.4]]), 'noise': 0.01, 'noise_T': 0.05})
    # ARGs.append({'mode': 'EMut', 'dt_s': 1.25, 'N': 500, 's0': np.array([2.,0.]), 'mu': np.array([0.7,0.7]), 'M': np.array([[-0.4,0.3],[0.3,-0.4]]), 'noise': 0.01, 'noise_T': 0.05})
    # ARGs.append({'mode': 'EMut', 'dt_s': 0.25, 'N': 500, 's0': np.array([1.,1.]), 'mu': np.array([0.7,0.7]), 'M': np.array([[-0.4,0.3],[0.3,-0.4]]), 'noise': 0.01, 'noise_T': 0.05})
    # ARGs = {'mode': , 'dt_s': 1.25, 'N': 500, 's0': np.array([1.,1.]), 'mu': np.array([0.7,0.7]), 'M': np.array([[-0.4,0.3],[0.3,-0.4]]), 'noise': 0.01, 'noise_T': 0.05}
    # ARGs.append({'mode': 'EMut', 'dt_s': 0.25, 'N': 500, 's0': np.array([1.,1.]), 'mu': np.array([0.7,0.7]), 'M': np.array([[-0.4,0.5],[0.5,-0.4]]), 'noise': 0.01, 'noise_T': 0.05})
    # ARGs.append({'mode': 'EMut', 'dt_s': 1.25, 'N': 500, 's0': np.array([1.,1.]), 'mu': np.array([0.7,0.7]), 'M': np.array([[-0.4,0.5],[0.5,-0.4]]), 'noise': 0.01, 'noise_T': 0.05})
    
    # UComp
    # ARGs.append({'mode': 'UComp', 'dt_s': 0.25, 'N': 500, 's0': np.array([2.,0.]), 'mu': np.array([0.8,0.8]), 'M': np.array([[-0.4,-0.5],[-0.9,-0.4]]), 'noise': 0.01, 'noise_T': 0.05})
    # ARGs.append({'mode': 'UComp', 'dt_s': 1.25, 'N': 500, 's0': np.array([2.,0.]), 'mu': np.array([0.8,0.8]), 'M': np.array([[-0.4,-0.5],[-0.9,-0.4]]), 'noise': 0.01, 'noise_T': 0.05})
    # ARGs.append({'mode': 'UComp', 'dt_s': 0.25, 'N': 500, 's0': np.array([1.,1.]), 'mu': np.array([0.8,0.8]), 'M': np.array([[-0.4,-0.5],[-0.9,-0.4]]), 'noise': 0.01, 'noise_T': 0.05})
    # ARGs = {'mode': , 'dt_s': 1.25, 'N': 500, 's0': np.array([1.,1.]), 'mu': np.array([0.8,0.8]), 'M': np.array([[-0.4,-0.5],[-0.9,-0.4]]), 'noise': 0.01, 'noise_T': 0.05}
    
    # UComp2
    # ARGs.append({'mode': 'UComp2', 'dt_s': 0.25, 'N': 500, 's0': np.array([2.,0.]), 'mu': np.array([0.8,0.8]), 'M': np.array([[-1.4,-0.5],[-0.9,-1.4]]), 'noise': 0.01, 'noise_T': 0.05})
    # ARGs.append({'mode': 'UComp2', 'dt_s': 1.25, 'N': 500, 's0': np.array([2.,0.]), 'mu': np.array([0.8,0.8]), 'M': np.array([[-1.4,-0.5],[-0.9,-1.4]]), 'noise': 0.01, 'noise_T': 0.05})
    # ARGs.append({'mode': 'UComp2', 'dt_s': 0.25, 'N': 500, 's0': np.array([1.,1.]), 'mu': np.array([0.8,0.8]), 'M': np.array([[-1.4,-0.5],[-0.9,-1.4]]), 'noise': 0.01, 'noise_T': 0.05})
    # ARGs = {'mode': , 'dt_s': 1.25, 'N': 500, 's0': np.array([1.,1.]), 'mu': np.array([0.8,0.8]), 'M': np.array([[-1.4,-0.5],[-0.9,-1.4]]), 'noise': 0.01, 'noise_T': 0.05}
    
    # UComp3
    ARGs.append({'mode': 'UComp3', 'dt_s': 0.25, 'N': 500, 's0': np.array([1,1]), 'mu': np.array([50.,50.]), 'M': np.array([[-100,-95],[-99,-100]]), 'noise': 0.01, 'noise_T': 0.05})
    # ARGs = ({'mode': 'UComp3', 'dt_s': 1.25, 'N': 500, 's0': np.array([1,1]), 'mu': np.array([50.,50.]), 'M': np.array([[-100,-95],[-99,-100]]), 'noise': 0.01, 'noise_T': 0.05})
    ARGs.append({'mode': 'UComp3', 'dt_s': 0.25, 'N': 500, 's0': np.array([2,0]), 'mu': np.array([50.,50.]), 'M': np.array([[-100,-95],[-99,-100]]), 'noise': 0.01, 'noise_T': 0.05})
    ARGs.append({'mode': 'UComp3', 'dt_s': 1.25, 'N': 500, 's0': np.array([2,0]), 'mu': np.array([50.,50.]), 'M': np.array([[-100,-95],[-99,-100]]), 'noise': 0.01, 'noise_T': 0.05})
    # ARGs.append({'mode': 'UComp3', 'dt_s': 0.25, 'N': 500, 's0': np.array([50.,50.]), 'mu': np.array([50.,50.]), 'M': np.array([[-100,-95],[-99,-100]]), 'noise': 0.01, 'noise_T': 0.05})
    # ARGs.append({'mode': 'UComp3', 'dt_s': 1.25, 'N': 500, 's0': np.array([50.,50.]), 'mu': np.array([50.,50.]), 'M': np.array([[-100,-95],[-99,-100]]), 'noise': 0.01, 'noise_T': 0.05})
    ARGs.append({'mode': 'UComp3', 'dt_s': 0.25, 'N': 500, 's0': np.array([50.,50.]), 'mu': np.array([100.,25.]), 'M': np.array([[-100,-95],[-99,-100]]), 'noise': 0.01, 'noise_T': 0.05})
    ARGs.append({'mode': 'UComp3', 'dt_s': 1.25, 'N': 500, 's0': np.array([50.,50.]), 'mu': np.array([100.,25.]), 'M': np.array([[-100,-95],[-99,-100]]), 'noise': 0.01, 'noise_T': 0.05})
    
    # Pred-prey 
    # ARGs = {'mode': 'predprey', 'dt_s': 1.25, 'N': 500, 's0': np.array([1.,1.]), 'mu': np.array([1.1,-0.4]), 'M': np.array([[0.0,-0.4],[0.1,0.0]]), 'noise': 0.01, 'noise_T': 0.05}

    
    reps = 100
    for ARGS in ARGs:
        data = []
        start = time.time()
        for i in range(reps): 
            data.append(generate_lv(**ARGS))
        runtime = time.time() - start
        filename = filename = '_'.join([','.join([str(j) for j in _.flatten()]) if type(_) == type(np.array([[-0.4,-0.5],[-0.5,-0.4]])) else str(_) for _ in ARGS.values()])
        savethis = {'data': data, 'datagen_params': ARGS}
        save_data(filename, savethis)

    # sys.stdout.flush();
    # sys.stdout.write('Total time: %5.2f seconds\n' % (time.time() - start));
    # sys.stdout.flush();

