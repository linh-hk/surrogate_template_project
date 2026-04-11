#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 18:15:37 2023

@author: h_k_linh

This script is submited to SGE on UCL cluster as a task. 
========================= IMPORTANT WARNING =========================
This script assumes multispecies mode by default:
  - s0_list = ['A','B']
  - s0 is loaded from file (initial_condition_*.npy)

If you switch to other models (EComp, EMut, UComp, Vano, etc.):
  1. Comment out the multispecies block
  2. Uncomment the desired datagen_params
  3. Adjust s0 handling:
      - If s0 is defined in datagen_params → DO NOT overwrite it
      - If using Vano → replace s0_list with your own list of arrays
  4. Update filename logic if needed (since s0 is no longer 'A'/'B')

Failure to do this will cause:
  - incorrect s0 being used
  - crashes or silent logical errors
=====================================================================

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

def iter_generatelv(dt_s, N, s0, mu, M, noise, noise_T, time_skip, sp_list):
    # print((os.getpid() * int(time.time())) % 123456789)
    s0 = s0.copy()
    np.random.seed(None)
    series = generate_lv(dt_s=dt_s, N=N, s0=s0, mu=mu, M=M, noise=noise, noise_T=noise_T, time_skip = time_skip)
    mask = stationary_evaluate(series[:, sp_list])
    if mask.all():
        return series
    else:
        return None

#%%
if __name__ == '__main__': 
    # 2-species gLV, predator-prey
    # # EComp
    # datagen_params = {'mode': 'EComp_Fast_20', 'dt_s': 0.25, 'N': 500, 's0': np.array([2.,0.]), 'mu': np.array([0.7,0.7]), 'M': np.array([[-0.4,-0.5],[-0.5,-0.4]]), 'noise': 0.01, 'noise_T': 0.05, 'time_skip': 250, 'sp_list': [0,1]}
    # datagen_params = {'mode': 'EComp_Slow_20', 'dt_s': 1.25, 'N': 500, 's0': np.array([2.,0.]), 'mu': np.array([0.7,0.7]), 'M': np.array([[-0.4,-0.5],[-0.5,-0.4]]), 'noise': 0.01, 'noise_T': 0.05, 'time_skip': 250, 'sp_list': [0,1]}
    # datagen_params = {'mode': 'EComp_Fast_11', 'dt_s': 0.25, 'N': 500, 's0': np.array([1.,1.]), 'mu': np.array([0.7,0.7]), 'M': np.array([[-0.4,-0.5],[-0.5,-0.4]]), 'noise': 0.01, 'noise_T': 0.05, 'time_skip': 250, 'sp_list': [0,1]}
    # # EMut
    # datagen_params = {'mode': 'EMut_Fast_20', 'dt_s': 0.25, 'N': 500, 's0': np.array([2.,0.]), 'mu': np.array([0.7,0.7]), 'M': np.array([[-0.4,0.3],[0.3,-0.4]]), 'noise': 0.01, 'noise_T': 0.05, 'time_skip': 250, 'sp_list': [0,1]}
    # datagen_params = {'mode': 'EMut_Slow_20', 'dt_s': 1.25, 'N': 500, 's0': np.array([2.,0.]), 'mu': np.array([0.7,0.7]), 'M': np.array([[-0.4,0.3],[0.3,-0.4]]), 'noise': 0.01, 'noise_T': 0.05, 'time_skip': 250, 'sp_list': [0,1]}
    # datagen_params = {'mode': 'EMut_Fast_11', 'dt_s': 0.25, 'N': 500, 's0': np.array([1.,1.]), 'mu': np.array([0.7,0.7]), 'M': np.array([[-0.4,0.3],[0.3,-0.4]]), 'noise': 0.01, 'noise_T': 0.05, 'time_skip': 250, 'sp_list': [0,1]}
    # datagen_params = {'mode': 'EMut_Slow_11', 'dt_s': 1.25, 'N': 500, 's0': np.array([1.,1.]), 'mu': np.array([0.7,0.7]), 'M': np.array([[-0.4,0.5],[0.5,-0.4]]), 'noise': 0.01, 'noise_T': 0.05, 'time_skip': 250, 'sp_list': [0,1]}
    # # UComp
    # datagen_params = {'mode': 'UComp_Fast_20', 'dt_s': 0.25, 'N': 500, 's0': np.array([2.,0.]), 'mu': np.array([0.8,0.8]), 'M': np.array([[-0.4,-0.5],[-0.9,-0.4]]), 'noise': 0.01, 'noise_T': 0.05, 'time_skip': 250, 'sp_list': [0,1]}
    # datagen_params = {'mode': 'UComp_Slow_20', 'dt_s': 1.25, 'N': 500, 's0': np.array([2.,0.]), 'mu': np.array([0.8,0.8]), 'M': np.array([[-0.4,-0.5],[-0.9,-0.4]]), 'noise': 0.01, 'noise_T': 0.05, 'time_skip': 250, 'sp_list': [0,1]}
    # datagen_params = {'mode': 'UComp_Fast_11', 'dt_s': 0.25, 'N': 500, 's0': np.array([1.,1.]), 'mu': np.array([0.8,0.8]), 'M': np.array([[-0.4,-0.5],[-0.9,-0.4]]), 'noise': 0.01, 'noise_T': 0.05, 'time_skip': 250, 'sp_list': [0,1]}
    # datagen_params = {'mode': 'UComp_Slow_11', 'dt_s': 1.25, 'N': 500, 's0': np.array([1.,1.]), 'mu': np.array([0.8,0.8]), 'M': np.array([[-0.4,-0.5],[-0.9,-0.4]]), 'noise': 0.01, 'noise_T': 0.05, 'time_skip': 250, 'sp_list': [0,1]}
    # # UComp2
    # datagen_params = {'mode': 'UComp2_Fast_20', 'dt_s': 0.25, 'N': 500, 's0': np.array([2.,0.]), 'mu': np.array([0.8,0.8]), 'M': np.array([[-1.4,-0.5],[-0.9,-1.4]]), 'noise': 0.01, 'noise_T': 0.05, 'time_skip': 250, 'sp_list': [0,1]}
    # datagen_params = {'mode': 'UComp2_Slow_20', 'dt_s': 1.25, 'N': 500, 's0': np.array([2.,0.]), 'mu': np.array([0.8,0.8]), 'M': np.array([[-1.4,-0.5],[-0.9,-1.4]]), 'noise': 0.01, 'noise_T': 0.05, 'time_skip': 250, 'sp_list': [0,1]}
    # datagen_params = {'mode': 'UComp2_Fast_11', 'dt_s': 0.25, 'N': 500, 's0': np.array([1.,1.]), 'mu': np.array([0.8,0.8]), 'M': np.array([[-1.4,-0.5],[-0.9,-1.4]]), 'noise': 0.01, 'noise_T': 0.05, 'time_skip': 250, 'sp_list': [0,1]}
    # datagen_params = {'mode': 'UComp2_Slow_11', 'dt_s': 1.25, 'N': 500, 's0': np.array([1.,1.]), 'mu': np.array([0.8,0.8]), 'M': np.array([[-1.4,-0.5],[-0.9,-1.4]]), 'noise': 0.01, 'noise_T': 0.05, 'time_skip': 250, 'sp_list': [0,1]}
    # # Pred-prey
    # datagen_params = {'mode': 'predprey', 'dt_s': 1.25, 'N': 500, 's0': np.array([1.,1.]), 'mu': np.array([1.1,-0.4]), 'M': np.array([[0.0,-0.4],[0.1,0.0]]), 'noise': 0.01, 'noise_T': 0.05, 'time_skip': 250, 'sp_list': [0,1]}

    # # 4-species, Vano
    # sp_list = [0,1]
    # r = np.array([1, 0.72, 1.53, 1.27])
    # A = np.array([[1, 1.09, 1.52, 0],
    #               [0, 1, 0.44, 1.36],
    #               [2.33, 0, 1, 0.47],
    #               [1.21, 0.51, 0.35, 1]])
    # mu = r.copy()
    # M = -A * np.expand_dims(r, 1) # M = -(r[:, None] * A)
    # datagen_params = {'mode': 'Vano_4sp', 'dt_s': 0.25, 'N': 500, 'mu': mu, 'M': M, 'noise': 0.01, 'noise_T': 0.05, 'time_skip': 250, 'sp_list': sp_list}
    # s0_list = [np.array([0.1, 0.1, 0.1, 0.1]),
    #            np.array([0.9, 0.1, 0.1, 0.1]), 
    #            np.array([0.1, 0.9, 0.1, 0.1]), 
    #            np.array([0.1, 0.1, 0.9, 0.1]), 
    #            np.array([0.1, 0.1, 0.1, 0.9]), 
    #            np.array([0.7, 0.7, 0.1, 0.1]), 
    #            np.array([0.7, 0.1, 0.7, 0.1]), 
    #            np.array([0.7, 0.1, 0.1, 0.7]), 
    #            np.array([0.1, 0.7, 0.7, 0.1]), 
    #            np.array([0.1, 0.7, 0.1, 0.7]), 
    #            np.array([0.1, 0.1, 0.7, 0.7]), 
    #            np.array([0.5, 0.5, 0.5, 0.5]),
    #            np.array([0.3, 0.6, 0.2, 0.4]), 
    #            np.array([0.6, 0.3, 0.4, 0.2]), 
    #            np.array([0.2, 0.4, 0.6, 0.3]), 
    #            np.array([0.4, 0.2, 0.3, 0.6])]
    
    
    # Multispecies, multistability, no chaos with process noise
    ## one matrix, 20 random initial condition
    sp_list = [24,39]
    from GenerateData import intrinsic_growth_vector_mu# , multistability_crit, im_symmetric_M, initial_conditions_s0
    noise = 0.001
    n_species = 50
    mu = intrinsic_growth_vector_mu(n_species)
    parameters_folder = 'multispecies_parameters' # in which, stores found_matrix.npy, initial_condition_{i}.npy
    M = np.load(f'{parameters_folder}/found_matrix.npy')
    datagen_params = {'mode': "multispecies_M_found_GB",
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
                      'sp_list': sp_list,
                      "a_GB": 0.025} # Species abundance below 0.05 (in code) then growth = 1 + a_GB
    
    # If I want to continue a broken run, the temp.file for such run is resume_file
    # Check the params carefully before continuing the run. the params are not saved in temp.file
    resume_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    reps = 1000
    start = time.time()
    foldername = 'multispecies'
    s0_list = ['A', 'B'] # Also involve in naming so be careful.
    #########################NO CHANGE AFTERWARDS #############################
    for i in s0_list: 
        # Check if run is generated
        filename = '_'.join([datagen_params['mode'], f"s0_{i}", 'noise', str(datagen_params['noise'])])
        if os.path.isfile(f"Simulated_data/{foldername}/data_{filename}_{reps}.pkl"):
            continue
        log.info(f'Saving at {filename}')
        trial_start_time = time.time()
        
        # Update s0 in datagen_params
        datagen_params['s0'] = np.load(file=f'{parameters_folder}/initial_condition_{i}.npy')

        # Prep for iter_generatelv
        ARGS_ = (datagen_params['dt_s'], 
                 datagen_params['N'], 
                 datagen_params['s0'], 
                 datagen_params['mu'], 
                 datagen_params['M'], 
                 datagen_params['noise'], 
                 datagen_params['noise_T'], 
                 datagen_params['time_skip'])
        
        # Continue from previous temp.file (set number of finished run count) or create new run, new temp.file
        if resume_file is not None and f"s0_{i}" in resume_file:
            tmp_file2 = resume_file
            data_ = load_streamed_data(tmp_file2) 
            count = len(data_)
            log.info(f"Continue run from {tmp_file2} at count {count}")
        
        else:
            tmp_file = f"temp_{datagen_params['mode']}_s0_{i}_pid{os.getpid()}.pkl"
            mp = Multiprocessor(output_file=tmp_file)
            
            for rep in range(reps):
                # reproducible
                mp.add(iter_generatelv, ARGS_ + (sp_list,))
            mp.run(4) # can only work up to 6. 7 and 8 will freeze the laptop
            # data = mp.results()
            del rep
            
            tmp_file2 = f"temp_{datagen_params['mode']}_s0_{i}_pid{os.getpid()}2.pkl"
            count = 0
            data_ = load_streamed_data(tmp_file) 
            for dat in data_:
                if dat is not None:
                    append_to_file(tmp_file2, dat)
                    count +=1
            os.remove(tmp_file)
        
        while count < reps:
            data_ = iter_generatelv(dt_s=datagen_params['dt_s'], 
                                    N=datagen_params['N'], 
                                    s0=datagen_params['s0'],
                                    mu=datagen_params['mu'], 
                                    M=datagen_params['M'], 
                                    noise=datagen_params['noise'], 
                                    noise_T=datagen_params['noise_T'], 
                                    time_skip=datagen_params['time_skip'],
                                    sp_list = datagen_params['sp_list'])
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
        save_data(filename, savethis, tag=str(reps), foldername = foldername)
        os.remove(tmp_file2)
        del ARGS_, data, no_noise, savethis

    sys.stdout.flush();
    sys.stdout.write('Total time: %5.2f seconds\n' % (time.time() - start));
    sys.stdout.flush();
