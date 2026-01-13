#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 18:15:37 2023

@author: h_k_linh

Run surrogate dependence tests on simulated time-series batches on SGE.

Qsub passes args to python:

qsub    ...             {cor_stat}      {surr_proc}     {folder_name}     {file_name}    {N_0}           
qsub    sys.argv[0]     sys.argv[1]     sys.argv[2]     sys.argv[3]     sys.argv[4]     sys.argv[5]
qsub    qsub.sh         a               a               multispecies    fina wo .pkl    0

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

from statsmodels.tsa.stattools import adfuller, kpss

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

# from mpi4py.futures import MPIPoolExecutor
sys.path.append('/home/hoanlinh/Simulation_test/Simulation_code/surrogate_dependence_test')
import main as sdt
#%%
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def choose_pair(trial, alpha=0.05, top_k=2, value="mean", abs_value=False,
                regression="c", autolag="AIC", maxlag=None, nlags="auto"):
    """
    Filters by stationarity over the full window using:
      ADF p < alpha AND KPSS p > alpha

    Returns:
      pair_ts   : (T,2) array
      pair_idx  : (s1,s2) original species indices
      stationary: array of stationary species indices (for trace/debug)
    """
    mask = stationary_evaluate(
        trial, alpha=alpha, regression=regression, autolag=autolag,
        maxlag=maxlag, nlags=nlags
    )
    stationary = np.where(mask)[0]
    if stationary.size < 2:
        return None, (-1, -1), stationary

    # value to rank (not returned)
    if value == "median":
        vals = np.median(trial, axis=0)
    else:
        vals = np.mean(trial, axis=0)

    key = np.abs(vals[stationary]) if abs_value else vals[stationary]
    order = stationary[np.argsort(key)[::-1]]

    s1 = int(order[0])
    s2 = int(order[min(top_k - 1, order.size - 1)])

    pair_ts = trial[:, [s1, s2]]
    return pair_ts, (s1, s2), stationary

def refine_run_data(data, N0,
                    alpha=0.05, top_k=2, value="mean", abs_value=False):
    pairs = []
    series_ids = []
    species_id = []
    stationary_idx = []

    for i, trial in enumerate(data):
        pair, (s1, s2), stationarity_ = choose_pair(trial, alpha=alpha, top_k=top_k, value=value, abs_value=abs_value)
        if pair is None:
            continue
        pairs.append((pair, N0+i)) 
        series_ids.append(N0+i)
        species_id.append((s1, s2))
        stationary_idx.append(stationarity_)

    meta = {"series_ids": series_ids, "species_id": species_id, "stationary_idx": stationary_idx}

    return pairs, meta

# test, test_meta = refine_run_data(data['data'][:100], N0=0, top_k=2)
# for series in test:
#     plot_timeseries_all_species(series[0], title = "top2 both")
# pick top 2 because after looking, it seems that most of them are close to 0 after filtered

def run_each_ts(series, series_id, stats_list, test_list, maxlag, nsurr,
                nproc_scan=2, nproc_ccm=2, nproc_embed=1):
    x = series[:, 0]
    y = series[:, 1]
    # manystats_manysurr(x, y, stats_list='all', test_list='all', maxlag=0, steplag=1, n_surr=99, kw_randphase={}, kw_twin={}, r_tts=choose_r, kw_statistic={})
    res = sdt.manystats_manysurr(x=x, y=y, stats_list=stats_list, test_list=test_list, maxlag=maxlag, n_surr=nsurr,
                                 nproc_scan=nproc_scan, nproc_ccm=nproc_ccm, nproc_embed=nproc_embed)
    return {series_id: res}
        
if __name__=="__main__":
    
    # test_list = [sys.argv[2] if 'tts' not in sys.argv[2] else 'tts_naive'] # , 'twin','randphase'
    log.info(f'working directory: {os.getcwd()}')
    if len(sys.argv) != 6:
        raise SystemExit(
            "Expected 5 arguments: cor_stat surr_proc folder_name file_name N0\n"
            "Example: qsub execute_run_tests.py plm nr myfolder data_foo 0"
        )
    stats_list = parse_corstat(sys.argv[1])  
    test_list = parse_testlist(sys.argv[2])
    folder_name = sys.argv[3]
    file_name = sys.argv[4]
    N0 = int(sys.argv[5])
    
    # Defaults
    maxlag = 0
    nsurr = 99
    batch_N = 100
    
    data, datagen_params = load_data(folder_name, file_name)
    data = data[N0 : N0 + batch_N]
    
    # Pair-selection / stationarity settings (single source of truth)
    alpha = 0.05
    top_k = 2
    value = "mean"
    abs_value = False
    data, meta = refine_run_data(data=data, N0=N0,
                                 alpha=alpha, top_k=top_k, value=value, abs_value=abs_value)
    log.info(
        f"Running: folder={folder_name}, file={file_name}.pkl, "
        f"range=[{N0}:{N0 + batch_N}), "
        f"tests={test_list}, stats={stats_list}, nsurr={nsurr}, maxlag={maxlag}"
    )
    resultsList = []
    start = time.time()
    
    with ThreadPoolExecutor(max_workers=OUTER_WORKERS) as ex:
        futs = []
        for series, series_id in data:
            futs.append(ex.submit(run_each_ts,
                                  series, series_id, stats_list, test_list, maxlag, nsurr,
                                  NPROC_SCAN, NPROC_CCM, NPROC_EMBED))
        for fut in as_completed(futs):
            resultsList.append(fut.result())
    
    # for series, series_id in data:
    #     ARGs = (series, series_id,stats_list, test_list, maxlag, nsurr)
    #     # print(ARGs)
    #     resultsList.append(run_each_ts(*ARGs))
    
    # with MPIPoolExecutor() as executor:
    #     resultsIter = executor.map(run_each_ts, ARGs, unordered=True)
    #     resultsList = [_ for _ in resultsIter]
    test_config = { "stationarity_filter": { "method": "ADF+KPSS",
                                            "alpha": alpha,
                                            "regression": "c",
                                            "autolag": "AIC",
                                            "nlags": "auto"
                                            },
                   "pair_selection": { "top_k": top_k,
                                      "value": value,
                                      "abs_value": abs_value,
                                      },
                   "surrogate_test": { "stats_list": stats_list,
                                      "test_list": test_list,
                                      "nsurr": nsurr,
                                      "maxlag": maxlag
                                      },
                   "batch": { "N0": N0,
                             "batch_N": batch_N
                             }
                   }
    saveP = {'pvals' : resultsList,
             'test_config': test_config,
             'data_meta': meta}
    
    out_name = name_output(choose_name=datagen_params['mode'], cor_stat_arg=sys.argv[1], test_list_arg=sys.argv[2], maxlag=maxlag, note=str(N0))
    
    fiS = f'{get_sample_dir(folder_name)}/{out_name}'
    log.info(f'Saving at {fiS}')
    with open(fiS, 'wb') as file:
        pickle.dump(saveP, file);
            
            #np.savetxt(fname,resultsList);
    sys.stdout.flush();
    log.info('Total time: %5.2f seconds\n', time.time() - start);
    sys.stdout.flush();
