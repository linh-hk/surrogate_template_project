# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 13:46:04 2024

@author: hoang
"""
#%% Load packages
import os
os.getcwd()
os.chdir('your_directory')
# os.chdir("C:/Users/hoang/Dropbox (Personal)/independence_tests/data/Figure5and8")
os.getcwd()
# import glob
import numpy as np
# !pip3 install mergedeep
from mergedeep import merge

import pickle
#%% Define function to load data and to merge data
"""You can load them by these simple codes and then merge them together:"""

# Trials are run in 10 batches each 100. Load all trials:
def load_data(sample, whichrun, excl = [], incl = ['_']):
    # sample and whichrun (true pos or false pos) see in execution
    # excl = [] # list of patterns in file names that you do not want to read data from
    # incl = ['_'] # list of patterns in file names that you want to read data from
    path = f'{sample}/{whichrun}'
    results = []
    for fi in os.listdir(path):
        if any(pat in fi for pat in excl):
            continue
        elif any(pat in fi for pat in incl):
            print(f'Loading:{path}/{fi}')
            with open(f'{path}/{fi}', 'rb') as file:
                results.append(pickle.load(file))
    return results

# Merge results of all batches together. 
"""The structure of the object is a bit tricky because I didn't think this far 
when I generalised them haha but you can use this simple codes to merge them together.
The initial object has the null distributions of each trial but after being merged by 
the following function, it only keeps a list of 1000 p-values from 1000 trials:"""

def merge_data(results, test_list, stat_list):
    maxlag = set([ cst['maxlag']
                  for sublist in results
                  for trials in sublist['pvals']
                  for _ in trials.values() if type(_) != type(np.array([]))
                  for xory in ['surrY', 'surrX']
                  for cst in _['test_params'][xory].values()])
    n_surr = set([sublist['nsurr'] for sublist in results])
    pvalss = {}
    for ml in maxlag:
        intermed = [_ for sublist in results
                    for _ in sublist['pvals']]
        pvalss[f'maxlag{ml}'] = merge(*intermed)          
   
    try:
        runtime = {lagkey: {xory : {cst : {stat : [trial['runtime'][xory][cst][stat]
                                            for idx, trial in lagitm.items()]
                                    for stat in stat_list}
                             for cst in test_list}
                      for xory in ['surrY', 'surrX']}
                 for lagkey, lagitm in pvalss.items()}
    except:
        runtime = 'not available'
    test_params = {lagkey: merge(*[trial['test_params'] for key,trial in lagitm.items() if key != 'XY']) for lagkey, lagitm in pvalss.items()}
   
    pvals = {lagkey: {xory : {cst : {stat : [trial['score_null_pval'][xory][cst][stat]['pval']
                                        for idx, trial in lagitm.items() if idx != 'XY']
                                for stat in stat_list}
                         for cst in test_list}
                  for xory in ['surrY', 'surrX']}
             for lagkey, lagitm in pvalss.items()}
    r = {lagkey: lagitm['surrY']['tts_naive']['surr_params']['r_tts']
         for lagkey, lagitm in test_params.items()}
    tts_pvals = {lagkey: {xory : {'tts': {stat : [B * (2 * r[lagkey] + 1) / (r[lagkey] + 1) for B in lagitm[xory]['tts_naive'][stat] ]
                                          for stat in stat_list}
                                  }
                          for xory in ['surrY', 'surrX']}
                 for lagkey, lagitm in pvals.items()}
    test_list = (*test_list, 'tts')
    pvals = merge(pvals, tts_pvals)
    return pvals, {'maxlag': maxlag, 'runtime' :runtime, 'test_params': test_params, 'test_list': test_list, 'n_surr': n_surr}
#%% Execution
stat_list = ('pearson', 'lsa', 'mutual_info',
                      'granger_y->x', 'granger_x->y',
                      'ccm_y->x', 'ccm_x->y')
test_list = ('randphase', 'twin', 'tts_naive') 

sample = 'Fig5_EComp_0.25_500_2.0,0.0_0.7,0.7_-0.4,-0.5,-0.5,-0.4_0.01_0.05' 
        # or 'Fig8_EMut_1.25_500_1.0,1.0_0.7,0.7_-0.4,0.3,0.3,-0.4_0.01_0.05'
# True positive results
whichrun = 'nolag_surr99' # 'nolag_fp_surr99'
results = load_data(sample, whichrun)
pvals, trial_info = merge_data(results, test_list, stat_list)
# False positive results
whichrun = 'nolag_fp_surr99'
results_fp = load_data(sample, whichrun)
pvals_fp, trial_info_fp = merge_data(results, test_list, stat_list)

