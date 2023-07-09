#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from scipy import stats
import numpy as np
"""
Created on Fri Mar 17 10:36:28 2023

@author: h_k_linh
"""
"""
Pearson can detect global associations so LSA was designed to detect local associations
LSA was assessed to be the highest performing correlation detection tool for microbial time series
Steps:
    1. Normalise x, y by ranking their values + apply inverse cumulave distribution function
    -> x, y follow standard normal distribution: \hat{x}, \hat{y}
    2. possitive association vector S+ and negative association vector S-
       S+(0)=S_(0) = 0
    3. S+(t) = max[0, S+(t-1) + \hat{x}(t)*\hat{y}(t)]
       S_(t) = max[0, S_(t-1) - \hat{x}(t)*\hat{y}(t)]
    4. LS = max( max(S_(t)) , max(S+(t)) ) / n
"""
data = all_models.xy_Caroline_LV_competitive.series[0]
# first replicate of the LV competitive model
x = data[0]
y = data[1]
    
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
    
#%%
def norm_transform(x):
    return stats.norm.ppf((stats.rankdata(x))/(x.size+1.0))

#alex's LSA from scratch
def lsa_new(x,y_array):
    print("\t\t\t\tCalculating local similarity without lags")
    # lsa with Delay=0 
    # x and y are time series
    # returns: local similarity
    n = x.size
    x = np.copy(norm_transform(x))
    score_P = np.zeros((y_array.shape[1]))
    score_N = np.zeros((y_array.shape[1]))
    
    for j in range(y_array.shape[1]): # for each column/surrogates in y series, column = feature
        y = np.copy(norm_transform(y_array[:,j])) # norm_transform the jth column of y
        P = np.zeros(n+1) # Initialise pos association
        print(P.size)
        N = np.zeros(n+1) # Initialise neg association
        for i in range(x.size): # for each x and y elements do
            P[i+1] = np.max([0, P[i] + x[i] * y[i] ])
            print(i,P[i+1])
            N[i+1] = np.max([0, N[i] - x[i] * y[i] ])
        score_P[j] = np.max(P) / n
        score_N[j] = np.max(N) / n
    sign = np.sign(score_P - score_N)
    return np.max([score_P, score_N], axis=0) * sign

def lsa_new_delay(x,y_array,D=3):
    print("\t\t\t\tCalculating local similarity with lag")
    # lsa with any Delay
    # x and y are time series
    # x is 1D, y is 2D (could be multiple time series)
    # returns: local similarity
    n = x.size # number of time points
    x = np.copy(norm_transform(x)) # normalize x
    score_P = np.zeros((y_array.shape[1])); # P-score for each X,Y pair
    score_N = np.zeros((y_array.shape[1])); # N-score for each X,Y pair
    
    for k in range(y_array.shape[1]):
        y = np.copy(norm_transform(y_array[:,k])); # normalize y
        P = np.zeros((n+1,n+1)) # 2D array
        N = np.zeros((n+1,n+1))
        for i in range(x.size):
            for j in range(x.size):
                if np.abs(i - j) <= D:
                    P[i+1][j+1] = np.max([0, P[i][j] + x[i] * y[j]])
                    N[i+1][j+1] = np.max([0, N[i][j] - x[i] * y[j]])
        score_P[k] = np.max(P) / n;
        score_N[k] = np.max(N) / n;
    # sign = np.sign(score_P - score_N);
    return np.max([score_P, score_N],axis=0); # non-negative values only

def alex(x,y_array):
    start = time.time()
    lsa = lsa_new_delay(x, y_array)
    rt = time.time()-start
    return lsa, rt

alex_lsa2, alex_rt2 = alex(xstar, y_array.T)

#%% Linh ver
def lsa_main(xy_):
    n = len(xy_)
    P = np.zeros(n+1) # Initialise pos association
    # print(P.size)
    N = np.zeros(n+1) # Initialise neg association
    for i in range(xy_.size): # for each x and y elements do
        P[i+1] = np.max([0, P[i] + xy_[i] ])
        # print(i,P[i+1])
        N[i+1] = np.max([0, N[i] - xy_[i] ])
    return {'P':P, 'N':N}

def lsa_new_linh(x,y_array):
    print("\t\t\t\tCalculating local similarity without lags")
    # lsa with Delay=0 
    # x and y are time series
    # returns: local similarity
    # n = x.size
    x_norm = norm_transform(x)
    
    y_norm = np.apply_along_axis(norm_transform, 1, y_array)
    xy = x_norm*y_norm
    
    score_P, score_N = np.apply_along_axis(lsa_main,1,xy).T
    sign = np.sign(score_P - score_N)
    return np.max([score_P, score_N], axis=0) * sign

def lsa_new_delay_linh(x,y_array,D=3):
    n = x.size
    x_norm = norm_transform(x)
    
    y_norm = np.apply_along_axis(norm_transform, 1, y_array)
    
    xy = []
    xy.append(x_norm * y_norm)
    for i in range(1,D+1):
        xy.append(x_norm[i:]*y_norm[:,:-i])
        print(i,x.size,0, x.size-i)
        xy.append(x_norm[:-i]*y_norm[:,i:])
        print(0,x.size-1,i,x.size)
    
    score_PN=[np.apply_along_axis(lsa_main,1,_) for _ in xy]
    score_P = np.divide([np.max([np.max(score_PN[i][_]['P']) 
                        for i in range(len(score_PN))]) 
                for _ in range(len(score_PN[0]))],n)
    score_N = np.divide([np.max([np.max(score_PN[i][_]['N']) 
                        for i in range(len(score_PN))]) 
                for _ in range(len(score_PN[0]))],n)
    return np.max([score_P,score_N], axis = 0);

def linh(x,y_array):
    start = time.time()
    lsa = lsa_new_delay_linh(x, y_array)
    rt = time.time() - start
    return lsa, rt
    
linh_lsa2, linh_rt2 = linh(xstar, y_array)
