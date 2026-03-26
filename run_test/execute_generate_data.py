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

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | PID=%(process)d | T=%(threadName)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


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

def append_to_file(filename, obj):
    with open(filename, "ab") as f:
        pickle.dump(obj, f)

def save_data(filename, data, tag = '', foldername = 'LVextra'):
    thispath = 'Simulated_data/'+ foldername + '/'
    os.makedirs(thispath, exist_ok= True)
    filepath = thispath + 'data_' + filename + '_' + tag + '.pkl'
    with open(filepath, 'wb') as fi:
        pickle.dump(data, fi)
        
from statsmodels.tsa.stattools import adfuller, kpss

def adf_p(x, regression="c", autolag="AIC", maxlag=None):
    x = np.asarray(x, dtype=float)
    return adfuller(x, regression=regression, autolag=autolag, maxlag=maxlag)[1]

def kpss_p(x, regression="c", nlags="auto"):
    x = np.asarray(x, dtype=float)
    return kpss(x, regression=regression, nlags=nlags)[1]

def stationary_evaluate(X, alpha=0.05, regression="c", 
                        autolag="AIC", maxlag=None, 
                        nlags="auto"):
    """
    X: (T,S)
    Returns boolean mask (S,) for species that are stationary over the whole window:
      ADF p < alpha  AND  KPSS p > alpha
    """
    X = np.asarray(X, dtype=float)
    S = X.shape[1]
    mask = np.zeros(S, dtype=bool)

    for s in range(S):
        x = X[:, s]
        p_adf = adf_p(x, regression=regression, autolag=autolag, maxlag=maxlag)
        p_kpss = kpss_p(x, regression=regression, nlags=nlags)
        mask[s] = (p_adf < alpha) and (p_kpss > alpha)

    return mask

def iter_generatelv(dt_s, N, s0, mu, M, noise, noise_T, time_skip, seed, sp_list):
    # print((os.getpid() * int(time.time())) % 123456789)
    s0 = s0.copy()
    np.random.seed(seed) # (os.getpid() * int(time.time())) % 123456789
    series = generate_lv(dt_s=dt_s, N=N, s0=s0, mu=mu, M=M, noise=noise, noise_T=noise_T, time_skip = time_skip)
    mask = stationary_evaluate(series[:, sp_list])
    if mask.all():
        return series
    else:
        return None

#%%
if __name__ == '__main__': 
    
    sp_list = [24,39]
    # Multispecies, multistability, no chaos with process noise
    ## one matrix, 20 random initial condition
    from GenerateData import intrinsic_growth_vector_mu# , multistability_crit, im_symmetric_M, initial_conditions_s0
    noise = 0.001
    n_species = 50
    mu = intrinsic_growth_vector_mu(n_species)
    M = np.load('multispecies_parameters/found_matrix.npy')
    datagen_params = {'mode': "multispecies_M_found",
                      'n_species': n_species,
                      'N': 500,
                      'dt_s': 0.25, 
                      'noise': noise,
                      'noise_T': 0.05,
                      'mu': mu,
                      # 'meanmu': meanmu,
                      # 'sigma': sigma,
                      'M': M,
                      'time_skip': 250,
                      'sp_list': sp_list}
    
    reps = 1000
    start = time.time()
    for i in ['A', 'B']: # run 20 random s0
        filename = '_'.join([datagen_params['mode'], f"s0_{i}", 'noise', str(datagen_params['noise'])])
        if os.path.isfile(f"Simulated_data/multispecies/data_{filename}_{reps}.pkl"):
            continue
        log.info(f'Saving at {filename}')
        trial_start_time = time.time()
        datagen_params['s0'] = np.load(file=f'multispecies_parameters/initial_condition_{i}.npy')

        ARGS_ = (datagen_params['dt_s'], 
                 datagen_params['N'], 
                 datagen_params['s0'], 
                 datagen_params['mu'], 
                 datagen_params['M'], 
                 datagen_params['noise'], 
                 datagen_params['noise_T'], 
                 datagen_params['time_skip'])
        
        tmp_file = f"temp_{datagen_params['mode']}_s0_{i}_pid{os.getpid()}.pkl"
        mp = Multiprocessor(output_file=tmp_file)
        
        for rep in range(reps):
            seed = (1234567 + ord(i)*10007 + rep) % (2**32 - 1)   # reproducible
            mp.add(iter_generatelv, ARGS_ + (seed, sp_list,))
        mp.run(4) # can only work up to 6. 7 and 8 will freeze the laptop
        # data = mp.results()
        
        tmp_file2 = f"temp_{datagen_params['mode']}_s0_{i}_pid{os.getpid()}2.pkl"
        count = 0
        data_ = load_streamed_data(tmp_file) 
        for dat in data_:
            if dat is not None:
                append_to_file(tmp_file2, dat)
                count +=1
        os.remove(tmp_file)
        
        while count < reps:
            seed = (1234567 + ord(i)*10007 + rep) % (2**32 - 1) 
            data_ = iter_generatelv(dt_s=datagen_params['dt_s'], 
                                       N=datagen_params['N'], 
                                       s0=datagen_params['s0'], 
                                       mu=datagen_params['mu'], 
                                       M=datagen_params['M'], 
                                       noise=datagen_params['noise'], 
                                       noise_T=datagen_params['noise_T'], 
                                       time_skip=datagen_params['time_skip'],
                                       seed = seed,
                                       sp_list = datagen_params['sp_list'])
            rep += 1
            if data_ is not None:
                append_to_file(tmp_file2, data_)
                count +=1

        no_noise = generate_lv(dt_s=datagen_params['dt_s'], 
                                   N=datagen_params['N'], 
                                   s0=datagen_params['s0'], 
                                   mu=datagen_params['mu'], 
                                   M=datagen_params['M'], 
                                   noise=0.00, 
                                   noise_T=datagen_params['noise_T'], 
                                   time_skip=datagen_params['time_skip'])
        
        data = load_streamed_data(tmp_file2) 
        
        runtime = time.time() - trial_start_time
        # filename = '_'.join([','.join([str(j) for j in _.flatten()]) if type(_) == type(np.array([[-0.4,-0.5],[-0.5,-0.4]])) else str(_) for _ in ARGS.values()])
        # filename = '_'.join([datagen_params['mode'], f"s0_{i}", 'noise', str(datagen_params['noise'])])
        params_to_save = dict(datagen_params)      # shallow copy
        params_to_save['s0'] = datagen_params['s0'].copy()
        savethis = {'no_noise': no_noise,
                    'data': data, 
                    'datagen_params': params_to_save, 
                    'runtime': runtime}
        
        save_data(filename, savethis, tag=str(reps), foldername = 'multispecies')
        os.remove(tmp_file2)
        del ARGS_, data, no_noise, savethis

    sys.stdout.flush();
    sys.stdout.write('Total time: %5.2f seconds\n' % (time.time() - start));
    sys.stdout.flush();
