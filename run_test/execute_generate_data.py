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
# os.chdir('D:/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/')
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
# sys.path.append('D:/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/Simulation_code/surrogate_dependence_test')
# sys.path.append('/home/h_k_linh/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/Simulation_code/surrogate_dependence_test')
from GenerateData import generate_lv
from multiprocessor import Multiprocessor

def load_streamed_data(tmp_file):
    data = []
    with open(tmp_file, "rb") as f:
        while True:
            try:
                data.append(pickle.load(f))
            except EOFError:
                break
    return data

def save_data(filename, data, tag = '', foldername = 'LVextra'):
    thispath = 'Simulated_data/'+ foldername + '/'
    os.makedirs(thispath, exist_ok= True)
    filepath = thispath + 'data_' + filename + '_' + tag + '.pkl'
    with open(filepath, 'wb') as fi:
        pickle.dump(data, fi)
        
def iter_generatelv(dt_s, N, s0, mu, M, noise, noise_T, time_skip, seed):
    # print((os.getpid() * int(time.time())) % 123456789)
    np.random.seed(seed) # (os.getpid() * int(time.time())) % 123456789
    return generate_lv(dt_s=dt_s, N=N, s0=s0, mu=mu, M=M, noise=noise, noise_T=noise_T, time_skip = time_skip)
#%%
if __name__ == '__main__': 
    
    # Multispecies, multistability, no chaos with process noise
    ## one matrix, 20 random initial condition
    from GenerateData import intrinsic_growth_vector_mu, multistability_crit, im_symmetric_M, initial_conditions_s0
    mat_num = sys.argv[1]
    noise = 0.001
    n_species = 50
    mu = intrinsic_growth_vector_mu(n_species)
    meanmu = 0.5
    sigma_crit = multistability_crit(meanmu, n_species) # 0.05
    sigma = 0.3
    M = im_symmetric_M(S=n_species, meanmu=meanmu, sigma=sigma)
    datagen_params = {'mode': f"multispecies_M{mat_num}",
                      'n_species': n_species,
                      'N': 500,
                      'dt_s': 0.25, 
                      'noise': noise,
                      'noise_T': 0.05,
                      'mu': mu,
                      'meanmu': meanmu,
                      'sigma': sigma,
                      'M': M,
                      'time_skip': 250}
    
    start = time.time()
    for s0_i in range(20): # run 20 random s0
        trial_start_time = time.time()
        datagen_params['s0'] = initial_conditions_s0(datagen_params['n_species'])

        reps = 1000
        ARGS_ = (datagen_params['dt_s'], 
                 datagen_params['N'], 
                 datagen_params['s0'], 
                 datagen_params['mu'], 
                 datagen_params['M'], 
                 datagen_params['noise'], 
                 datagen_params['noise_T'], 
                 datagen_params['time_skip'])
        
        tmp_file = f"temp_{datagen_params['mode']}_s0{s0_i}_pid{os.getpid()}.pkl"
        mp = Multiprocessor(output_file=tmp_file)
        for rep in range(reps):
            seed = (1234567 + s0_i*10007 + rep) % (2**32 - 1)   # reproducible
            mp.add(iter_generatelv, ARGS_ + (seed,))
        mp.run(4) # can only work up to 6. 7 and 8 will freeze the laptop
        # data = mp.results()
        data = load_streamed_data(tmp_file)
        
        no_noise = generate_lv(dt_s=datagen_params['dt_s'], 
                                   N=datagen_params['N'], 
                                   s0=datagen_params['s0'], 
                                   mu=datagen_params['mu'], 
                                   M=datagen_params['M'], 
                                   noise=0.0, 
                                   noise_T=datagen_params['noise_T'], 
                                   time_skip=datagen_params['time_skip'])
        
        runtime = time.time() - trial_start_time
        # filename = '_'.join([','.join([str(j) for j in _.flatten()]) if type(_) == type(np.array([[-0.4,-0.5],[-0.5,-0.4]])) else str(_) for _ in ARGS.values()])
        filename = '_'.join([datagen_params['mode'], f"s0_{s0_i}", 'noise', str(datagen_params['noise'])])
        params_to_save = dict(datagen_params)      # shallow copy
        params_to_save['s0'] = datagen_params['s0'].copy()
        savethis = {'no_noise': no_noise,
                    'data': data, 
                    'datagen_params': params_to_save, 
                    'runtime': runtime}
        
        save_data(filename, savethis, tag=str(reps), foldername = 'multispecies')
        os.remove(tmp_file)
        del ARGS_, data, no_noise, savethis

    sys.stdout.flush();
    sys.stdout.write('Total time: %5.2f seconds\n' % (time.time() - start));
    sys.stdout.flush();
