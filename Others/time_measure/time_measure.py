#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 11:19:38 2023

@author: h_k_linh
"""
import os
# os.chdir('/home/h_k_linh/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test')
os.mkdir('time_measure')

import GenerateData as dataGen
import time
import numpy as np
import pandas as pd

import Correlation_Surrogate_tests as cst

#%% Measure data generation time
def time_mess_dataGen(func, r,a, run_id = 1, dt_s = 0.25, N=500, noise=0.05*0.5,noise_T=0.05,intx="competitive"):
    start = time.time()
    if (func == dataGen.generate_chaotic_lv):
        func(N, r, a)
        # print(func,N,r,a)
    elif (func == dataGen.generate_lv or func == dataGen.generate_niehaus):
        func(run_id, dt_s, N, noise, noise_T, intx)
        # print(func,run_id, dt_s, N, noise, noise_T, intx)
    else:
        func(N)
        # print(func,N)
    return time.time() - start

data_list1 = [dataGen.generate_AR1_uni_tau1, dataGen.generate_AR1_uni_tau2, dataGen.generate_uni_logistic_map, dataGen.generate_sine_w_noise,
                 dataGen.generate_ar1, dataGen.generate_logistic_map, dataGen.generate_sinew_intmt_corptxn, dataGen.generate_coinflips_w_changeHeadprob_noise,
                 dataGen.generate_noise_w_periodically_varying_kurtosis, dataGen.generate_FitzHugh_Nagumo, dataGen.generate_chaotic_lv,
                 dataGen.generate_lv, dataGen.generate_niehaus]
out = {'models': ['xy_ar_u', 'xy_ar_u2', 'xy_uni_logistic', 'xy_sinewn', 
              'ts_ar', 'ts_logistic', 'ts_sine_intmt', 'ts_coinflip', 
              'ts_noise_wv_kurtosis', 'xy_FitzHugh_Nagumo', 'ts_chaoticLV', 
              'xy_Caroline_LV', 'xy_Caroline_CH']}

r = np.array([1, 0.72, 1.53, 1.27])
a = np.array([[1, 1.09, 1.52, 0],
     [0, 1, 0.44, 1.36],
     [2.33, 0, 1, 0.47],
     [1.21, 0.51, 0.35, 1]])

# Default
dt_s = 0.25
N=500
noise=0.05*0.5
noise_T=0.05
intx="competitive"
for j in range(5):
    tm=[]
    for i in range(len(data_list1)):
        tm.append( time_mess_dataGen(data_list1[i], r, a) )
    out.update({f'Default{j}' : tm})
    
# Caroline suggestions
dt_s = 1.25
N=500 #
noise=0.01
noise_T=0.05 #
intx="competitive"
for j in range(5):
    tm=[]
    for i in range(len(data_list1)):
        tm.append( time_mess_dataGen(data_list1[i], r, a, dt_s=dt_s, noise=noise, noise_T=noise_T) )
    out.update({f'Caroline_sugg_{j}' : tm})

# Caroline suggestions + 2000
dt_s = 1.25
N=2000 #
noise=0.01
noise_T=0.05 #
intx="competitive"
for j in range(5):
    tm=[]
    for i in range(len(data_list1)):
        tm.append( time_mess_dataGen(data_list1[i], r, a, dt_s=dt_s, noise=noise, noise_T=noise_T) )
    out.update({f'Caroline_sugg_{j}_2000' : tm})

    
out = pd.DataFrame(out)
out['Default_avg']=out.iloc[:, 2:6].mean(axis=1)
out['C_sugg_avg']=out.iloc[:, 7:11].mean(axis=1)
out['C_sugg_2000_avg']=out.iloc[:, 12:16].mean(axis=1)
out.to_csv("time_measure/dataGen_time.csv", index=False)


#%%% measure surrogate time
dt_s = 1.25
N=500 #
noise=0.01
noise_T=0.05 #
intx="competitive"
xy_Caroline_CH = list(dataGen.generate_niehaus(run_id=1, dt_s=dt_s, N=500, noise=noise, noise_T=noise_T)[:,:2].T)
#%%%

# default: maxlag= 2
def time_mess_cst_surrX_and_surrY(data, name, test, stats, maxlag = 4): 
    start = time.time()
    results = cst.surrX_and_surrY(data, name, test, stats, maxlag = maxlag)
    return [time.time() - start, results]

data = xy_Caroline_CH
name = 'xy_Caroline_CH'
test_list = ['tts', 'tts_naive']
stats_list = ['pearson', 'lsa','mutual_info']

# t_cst = {}
# for test in test_list:
#     t_stats ={}
#     for stats in stats_list:
#         t_stats[stats] = time_mess_cst_surrX_and_surrY(data, name, test, stats)
#     t_cst[test] = t_stats
# test = pd.DataFrame(t_cst)

t = {}
dflist = []
for stats in stats_list:
    t_cst = {}
    for test in test_list:
        t_P = {}
        for i in range(10):
            t_P[f'{stats}{i}'] = time_mess_cst_surrX_and_surrY(data, name, test, stats)
        t_cst[test] = t_P
    t[stats] = t_cst
    df = pd.DataFrame(t[stats])
    df.loc['Avg_time'] = df.mean(axis=0)
    dflist.append(df)
    df.to_csv(f'time_measure/time_tts_vs_ttsnaive_{stats}.csv', index = False)
# start = time.time()
# results = cst.return_result_dict_for_mpi4py(xy_Caroline_CH, 'xy_Caroline_CH', test_list, stats_list, maxlag=4)
# time.time() - start

#%%% For multishift
xy_Caroline_CH = list(dataGen.generate_niehaus(run_id=1, dt_s=dt_s, N=N, noise=noise, noise_T=noise_T)[:,:2].T)

data = xy_Caroline_CH
name = 'xy_Caroline_CH'
test_list = ['tts_multishift']
stats_list = ['pearson', 'lsa','mutual_info']

t = {}
dflist = []
resultss = []
for stats in stats_list:    
    t_P = {}
    for i in range(10):
        t_P[f'{stats}{i}'], result = time_mess_cst_surrX_and_surrY(data, name, 'tts_multishift', stats, maxlag = 4)
        resultss.append(result)
    t[stats] = t_P
    df = pd.DataFrame(t[stats], index = [0]).T
    df.loc['Avg_time'] = df.mean(axis=0)
    df.to_csv(f'time_measure/tts_multishift_{stats}.csv')
    dflist.append(df)