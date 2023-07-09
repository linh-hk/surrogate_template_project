#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 14:59:04 2023

@author: h_k_linh

Breaking down granger to understand, try to optimise run time

"""
import time
import numpy as np
from scipy.spatial import distance_matrix
from pandas import DataFrame

import os
os.getcwd()
# os.chdir('C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/')
os.chdir('/home/h_k_linh/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/')
os.getcwd()
import glob
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from mergedeep import merge

import pickle

import GenerateData as dataGen
import numpy as np
from scipy import stats # t-distribution
from scipy.spatial import distance_matrix # for twin
from pyunicorn.timeseries import surrogates #for iaaft
import ccm # for ccm
from statsmodels.tsa.stattools import grangercausalitytests as granger # for granger
from statsmodels.tsa.api import VAR # vector autoregression for granger
from sklearn.feature_selection import mutual_info_regression # for mutual information

# import dill # for saving/loading data
# import pickle # for saving/loading data
# import matplotlib.pyplot as plt

import time
#%%
data = all_models.ts_chaoticLV.series[11]
# first replicate of the LV competitive model
x = data[0]
y = data[1]
#%%
"""
Trouble shooting granger run, run untitled0.py returns a results list of 12 so suppose that the 12th data has problem. set data to data_raw[12]
Save loaded data to data_raw
"""
# data_raw = data

y = data_raw[12][0]
x = data_raw[12][1]

# fig, ax = plt.subplots()
# ax.plot(x)
# ax.plot(y)

#%%%
def tts(x,y,r):
    # time shift
    t = len(x); #number of time points in orig data
    xstar = x[r:(t-r)]; # middle half of data - truncated original X0
    tstar = len(xstar); # length of truncated series
    ystar = y[r:(t-r)]; # truncated original Y0
    t0 = r;             # index of Y0 in matrix of surrY
    
    y_surr = np.tile(ystar,[2*r+1, 1]) # Create empty matrix of surrY
    print("\t\t\tCreating tts surrogates")
    #iterate through all shifts from -r to r
    for shift in np.arange(t0 - r, t0 + r):
        # print(f'shift-t0:{shift-t0}\ty[shift:(shift+tstar)]: {shift}:{shift+tstar}')
        y_surr[shift-t0] = y[shift:(shift+tstar)];
    # print(f"\t\t\t\tCalculating {statistic.__name__} for tts surrogates")
    # importance_true = np.max(scan_lags(xstar, ystar.reshape(-1,1),statistic,maxlag)[1]); # scan_lags returns np.vstack((lags,score))
    # importance_surr = np.max(scan_lags(xstar, y_surr.T, statistic,maxlag)[1:],axis=1);
    # sig = np.mean(importance_surr > importance_true, axis=0);
    return xstar, y_surr;
xstar, y_array = tts(x,y,119)

data = np.vstack((xstar,y_array)).T

#%%%
def granger_stat(x,y,pval=False):
    print("\t\t\t\tCalculating Granger")
    t,N = y.shape; # y is surrY matrix
    if (N == 1):
        data = np.vstack((x,y.T)).T;
        model = VAR(data); # Vector autoregression
        results = model.fit(maxlags=15,ic='aic'); # ??VAR
        maxlag=np.max([results.k_ar, 1])
        #gc = results.test_causality(caused='y1',causing='y2');
        #return np.array(gc.test_statistic).reshape(1);
        if pval:
            return granger(data, [maxlag], verbose=False)[maxlag][0]['ssr_ftest'][1];
        else:
            return np.array(granger(data, [maxlag], verbose=False)[maxlag][0]['ssr_ftest'][0]);
    else:
        result = np.zeros(N);
        for i in range(N):
            data = np.vstack((x,y[:,i].T)).T;
            model = VAR(data);
            res = model.fit(maxlags=15,ic='aic');
            #gc = res.test_causality(caused='y1',causing='y2');
            #result[i] = gc.test_statistic
            maxlag = np.max([res.k_ar, 1])
            if pval:
                result[i] = granger(data, [maxlag], verbose=False)[maxlag][0]['ssr_ftest'][1];
            else:
                print(i)
                result[i] = granger(data, [maxlag], verbose=True)[maxlag][0]['ssr_ftest'][0];
        return result;


#%%
start = time.time()
result_C = granger_stat(xstar,y_array.T,pval=False)
rt_C = time.time()-start
#%%
def run(x, y_i, pvals):
    if pvals:
        j = 1 
    else: 
        j = 0
    data = np.vstack((x,y_i.T)).T;
    model = VAR(data);
    res = model.fit(maxlags=15,ic='aic');
    #gc = res.test_causality(caused='y1',causing='y2');
    #result[i] = gc.test_statistic
    maxlag = np.max([res.k_ar, 1])
    return granger(data, [maxlag], verbose=False)[maxlag][0]['ssr_ftest'][j]

def run2(i, x, y, pvals):
    y=y[i,:]
    if pvals:
        j = 1 
    else: 
        j = 0
    data = np.vstack((x,y.T)).T;
    model = VAR(data);
    res = model.fit(maxlags=15,ic='aic');
    #gc = res.test_causality(caused='y1',causing='y2');
    #result[i] = gc.test_statistic
    maxlag = np.max([res.k_ar, 1])
    return granger(data, [maxlag], verbose=False)[maxlag][0]['ssr_ftest'][j]


def granger_stat(x,y,pval=False):
    print("\t\t\t\tCalculating Granger")
    t,N = y.shape; # y is surrY matrix
    y = y.T
    if (N == 1):
        data = np.vstack((x,y)).T;
        model = VAR(data); # Vector autoregression
        results = model.fit(maxlags=15,ic='aic'); # ??VAR
        maxlag=np.max([results.k_ar, 1])
        #gc = results.test_causality(caused='y1',causing='y2');
        #return np.array(gc.test_statistic).reshape(1);
        if pval:
            return granger(data, [maxlag], verbose=False)[maxlag][0]['ssr_ftest'][1];
        else:
            return np.array(granger(data, [maxlag], verbose=False)[maxlag][0]['ssr_ftest'][0]);
    else:
        # result = np.zeros(N);
        # for i in range(N):
        #     data = np.vstack((x,y[:,i].T)).T;
        #     model = VAR(data);
        #     res = model.fit(maxlags=15,ic='aic');
        #     #gc = res.test_causality(caused='y1',causing='y2');
        #     #result[i] = gc.test_statistic
        #     maxlag = np.max([res.k_ar, 1])
        #     if pval:
        #         result[i] = granger(data, [maxlag], verbose=False)[maxlag][0]['ssr_ftest'][1];
        #     else:
        #         result[i] = granger(data, [maxlag], verbose=False)[maxlag][0]['ssr_ftest'][0];
        result = [run(x,y[i,:],pval) for i in range(N)]
        # result = list(map(run,[x]*len(y_i),y_i, [pval]*len(y_i)))
        return result;
#%%
start = time.time()
result_L = granger_stat(y_array.T,xstar,pval=False)
rt_L = time.time()-start