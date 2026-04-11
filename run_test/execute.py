#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 18:15:37 2023

@author: h_k_linh

Run surrogate dependence tests on simulated time-series batches on SGE.

Qsub passes args to python:

qsub    ...             {cor_stat}      {surr_proc}     {folder_name}     {file_name}    {N_0}          {pair}
qsub    sys.argv[0]     sys.argv[1]     sys.argv[2]     sys.argv[3]     sys.argv[4]     sys.argv[5]     sys.argv[6]
qsub    qsub.sh         a               a               multispecies    data.pkl        0               "5,7"

Arguments:
    cor_stat: string of statistic initials
        a = all
        p = pearson
        l = lsa
        m = mutual_info
        c = ccm (both directions)
        g = granger (both directions)
        Examples: "a", "plm", "cg", "lg"

    surr_proc: string of surrogate test initials
        a = all ("tts_naive", "twin", "randphase")
        n = tts_naive
        w = twin
        r = randphase
        Examples: "a", "nr", "w"

    folder_name: folder under Simulated_data/ (or Simulated_data/LVextra/)
    file_name:   base filename (without .pkl)
    N0:          start index into the stored list of simulations
    pair:       (optional) for multiplespecies

This script runs on SGE on UCL cluster:
    Load simulated data that is in ../Simulated_data(/LVextra)/folder_name/file_name        
    Run the dependence test on the data in parallel (multiprocessor rather than MPI4py) and save results on cluster.
Note:

"""
import os
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

import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning
warnings.filterwarnings("ignore", category=InterpolationWarning)

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | PID=%(process)d | T=%(threadName)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)
# Optional: silence chatty libraries
logging.getLogger("statsmodels").setLevel(logging.WARNING)
import traceback

# from mpi4py.futures import MPIPoolExecutor
sys.path.append('/home/hoanlinh/Simulation_test/Simulation_code/surrogate_dependence_test')
import main as sdt
#%%
# from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessor import Multiprocessor
OUTER_WORKERS = 4

# inner budget per pair (must sum <= 4-ish; these are “caps”)
NPROC_SCAN  = 1
NPROC_CCM   = 4
NPROC_EMBED = 1
#%% Argument parsing helpers
def parse_corstat(sys_arg):    
    if 'a' in sys_arg:
        return ['pearson', 'lsa', 'mutual_info', 'ccm_y->x', 'ccm_x->y', 'granger_y->x', 'granger_x->y']
    else:
        stats_list = []
        if 'p' in sys_arg:
            stats_list.append('pearson')
        if 'l' in sys_arg:
            stats_list.append('lsa')
        if 'm' in sys_arg:
            stats_list.append('mutual_info')
        if 'c' in sys_arg:
            stats_list += ['ccm_y->x', 'ccm_x->y']
        if 'g' in sys_arg:
            stats_list += ['granger_y->x', 'granger_x->y']
        return stats_list
    
def parse_testlist(sys_arg):
    if 'a' in sys_arg:
        return ['tts_naive', 'twin','randphase'] # 
    else: 
        test_list = []
        if 'n' in sys_arg:
            test_list.append('tts_naive')
        if 'w' in sys_arg:
            test_list.append('twin')
        if 'r' in sys_arg:
            test_list.append('randphase')
        return test_list

def parse_pair(s):
    a, b = s.split(",")
    return [int(a), int(b)]

def get_sample_dir(folder_name):
    base = "Simulated_data/LVextra" if "500" in folder_name else "Simulated_data"
    return f"{base}/{folder_name}"
    
def load_data(folder_name, file_name = 'data'):
    sampdir = get_sample_dir(folder_name=folder_name)
    with open(f'{sampdir}/{file_name}.pkl', 'rb') as fi:
        data = pickle.load(fi)
    return data['data'], data['datagen_params']
    # for false pos
    # num_trials = len(data['data'])
    # data_fp = [[data['data'][_][0], data['data'][0 if _ == num_trials - 1 else _+1][1]] 
    #           for _ in range(num_trials)]
    # return data_fp, data['datagen_params']
    
def name_output(choose_name, cor_stat_arg='a', test_list_arg='a', maxlag=0, note=''):
    parts = [choose_name]
    if note:
        parts.append(note)
    parts += [cor_stat_arg, test_list_arg, "maxlag", str(maxlag)]
    return "_".join(parts) + ".pkl"

#%% Execution
def pair_selection(data, pair, N0=0):
    out = []
    for i, trial in enumerate(data):
        # CASE 1: list of [x, y]
        if isinstance(trial, list):
            series = np.column_stack(trial)   # (T, 2)
        # CASE 2: full matrix (T, N)
        else:
            series = trial[:, pair]           # (T, 2)
        # TEMP FIX: old data has 501 timepoints → trim to 500
        if series.shape[0] == 501:
            series = series[:500]
        series_id = N0 + i
        out.append((series, series_id))
    return out

def run_each_ts(series, series_id, stats_list, test_list, maxlag, nsurr,
                nproc_scan=2, nproc_ccm=2, nproc_embed=1):
    x = series[:, 0]
    y = series[:, 1]
    # manystats_manysurr(x, y, stats_list='all', test_list='all', maxlag=0, steplag=1, n_surr=99, kw_randphase={}, kw_twin={}, r_tts=choose_r, kw_statistic={})
    try:
        res = sdt.manystats_manysurr(x=x, y=y, stats_list=stats_list, test_list=test_list, maxlag=maxlag, n_surr=nsurr,
                                     nproc_scan=nproc_scan, nproc_ccm=nproc_ccm, nproc_embed=nproc_embed)
        return {series_id: res}
    except Exception as e:
        log.error(f"FAILED series_id={series_id}")
        log.error(traceback.format_exc())

        return {
            "FAILED": series_id,
            "error": str(e)
        }
        
if __name__=="__main__":
    
    # test_list = [sys.argv[2] if 'tts' not in sys.argv[2] else 'tts_naive'] # , 'twin','randphase'
    log.info(f'working directory: {os.getcwd()}')
    if len(sys.argv) < 6:
        raise SystemExit(
            "Expected 5 arguments: cor_stat surr_proc folder_name file_name N0 (pair)\n"
            "Example: qsub execute_run_tests.py plm nr myfolder data_foo 0 \"5,7\""
        )
    stats_list = parse_corstat(sys.argv[1])  
    test_list = parse_testlist(sys.argv[2])
    folder_name = sys.argv[3]
    file_name = sys.argv[4]
    N0 = int(sys.argv[5])
    pair = parse_pair(sys.argv[6]) if len(sys.argv) == 7 else [0,1]
    
    import re
    s0_match = re.search(r's0_([^_]+)', file_name) 
    s0 = s0_match.group() if s0_match else "s0_unknown"
    
    # Defaults
    maxlag = 0
    nsurr = 99
    batch_N = 100
    
    data, datagen_params = load_data(folder_name, file_name)
    data = data[N0 : N0 + batch_N]
    mode = datagen_params.get('mode') or folder_name 
    out_name = name_output(choose_name=mode, cor_stat_arg=sys.argv[1], test_list_arg=sys.argv[2], maxlag=maxlag, note=f"{s0}_{N0}")
    
    # Pair-selection / stationarity settings (single source of truth)
    data = pair_selection(data=data, pair=pair, N0=N0)
    log.info(
        "Running: folder=%s, file=%s.pkl, range=[%d:%d), tests=%s, stats=%s, nsurr=%d, maxlag=%d TRUEPOS | ", 
        folder_name, file_name, N0, N0 + batch_N, test_list, stats_list, nsurr, maxlag)
    
    test_config = { "stationarity_filter": { "method": "fixed_pair"
                                            },
                   "pair_selection": {"pair": pair
                                      },
                   "surrogate_test": { "stats_list": stats_list,
                                      "test_list": test_list,
                                      "nsurr": nsurr,
                                      "maxlag": maxlag
                                      },
                   "batch": { "N0": N0,
                             "batch_N": batch_N,
                             "isFALSEPOS": False
                             }
                   }
    saveP = {'test_config': test_config,
             'datagen_params': datagen_params}
    
    out_dir = get_sample_dir(folder_name)
    os.makedirs(out_dir, exist_ok=True)
    fiS = f"{out_dir}/{out_name}"
    if os.path.isfile(fiS):
        raise SystemExit(
            f'File {fiS} already existed. Please delete or recheck'
        )
    log.info(f'Saving at {fiS}')
    
    with open(fiS, 'wb') as file:
        pickle.dump(saveP, file);
        
    start = time.time()
    
    mp = Multiprocessor(output_file=fiS)
    for series, series_id in data:
        ARGs=(series, series_id, stats_list, test_list, maxlag, nsurr,
              NPROC_SCAN, NPROC_CCM, NPROC_EMBED)
        mp.add(run_each_ts, ARGs)
    mp.run(OUTER_WORKERS, per_result_timeout_s=1800) 

    sys.stdout.flush();
    log.info('Total time: %5.2f seconds\n', time.time() - start);
    sys.stdout.flush();
