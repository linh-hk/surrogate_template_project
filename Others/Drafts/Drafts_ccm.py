#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 13:32:07 2023

@author: h_k_linh

Break down ccm function, try to optimise code

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
#%%
data = all_models.xy_ar_u.series[0]
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

data = np.vstack((xstar,y_array)).T
#%%
def pcorr_multivariate_y(x,y):
    M = np.zeros([x.size, y.shape[1] + 1])
    M[:,0] = x
    M[:,1:] = y
    cov = np.cov(M.T)
    cov_xy = cov[0,:]
    std_y = np.sqrt(np.diag(cov))
    std_x = std_y[0]
    rho = cov_xy / (std_y * std_x)
    return rho[1:]

def pcorr(x,y):
    return pcorr_multivariate_y(x,y.reshape(-1,1))[0]

def get_weights(distances):
    """Performs weighting in KNN for Simplex projection

    Args:
      * distances_ (array) Array of distances

    Returns:
      * weights
    """
    distances = np.array(distances, copy=True).astype(float)
    if len(distances.shape) == 1:
        distances = distances.reshape(1,-1)
    if np.any(distances == 0): # deal with zeros
        distances[distances != 0] = np.inf
        distances[distances == 0] = 1
    min_dist = np.min(distances, axis=1).reshape(-1,1)
    u = np.exp(-distances / min_dist)
    weights = u / np.sum(u, axis=1).reshape(-1,1)
    return weights


def my_random_choice(a, size, n, replace=False):
    """
    Args:
        a (int): number of items to choose from
        size (int): number of items to choose on each replicate trial
        n (int): number of trials
        replace (bool): choose with or without replacement

    Returns:
        an (size x n) matrix where each column is a trial
    """
    if replace:
        return np.random.choice(a, size=[size, n], replace=True)
    else:
        return np.argsort(np.random.random([a, n]), axis=0)[:size, :]

def setup_problem(data, embed_dim, tau, pred_lag=0):
    """Prepares a "standard matrix problem" from the time series data

    Args:
        data (array): A 2d array with two columns where the first column
            will be used to generate features and the second column will
            be used to generate the response variable
        embed_dim (int): embedding dimension (delay vector length)
        tau (int): delay between values
        pred_lag (int): prediction lag
    """
    x = data[:,0] # x
    y = data[:,1] # surr y
    feat = []
    resp = []
    idx_template = np.array([pred_lag] + [-i*tau for i in range(embed_dim)])
    for i in range(x.size):
        if np.min(idx_template + i) >= 0 and np.max(idx_template + i) < x.size:
            feat.append(x[idx_template[1:] + i])
            resp.append(y[idx_template[0] + i])
    return np.array(feat), np.array(resp)

def ccm_loocv(data, embed_dim, tau=1, lib_sizes=[None], replace=False,
              n_replicates=1, weights='exp', pred_lag=0, score=pcorr,
              variable_libs=False):
    """
    Args:
        data (array): array with n rows and m+1 columns, where n is the number
            of time points and m is the number of putative causer variables. The
            first column is the putative causee
        embed_dim (int): embedding dimension
        tau (int): delay for delay embedding
        lib_sizes (iterable): list of library sizes. Default is maximum library size
        replace (False): replacement for libraries chosen from sizes below maximum
        n_replicates (int): number of replicate runs
        weights (string): weighting function. Choices are 'exp' or 'uniform'
        variable_libs (bool): If this parameter is True, and if lib_sizes is
            used, each delay vector uses a different random library.
    Returns:
        Pandas DataFrame. The column 'causer' denotes the index of the putative
            causer
    """
    print(data, embed_dim, tau, lib_sizes, replace,
                  n_replicates, weights, pred_lag, score,
                  variable_libs)
    X, _ = setup_problem(data, embed_dim=embed_dim, tau=tau, pred_lag=pred_lag)
    n_y = data.shape[-1] - 1 # number of replicates/ surrogates
    Y = [setup_problem(data[:,[0,i+1]], embed_dim=embed_dim, tau=tau, pred_lag=pred_lag)[1] for i in range(n_y)]
    # STEP ONE: get true neigbor indices
    KX = distance_matrix(X,X)
    np.fill_diagonal(KX, np.inf) # not allowed to train on self
    (n, embed_dim) = X.shape
    RX = np.zeros_like(KX)
    for loc in range(n):
        RX[loc,:] = np.argsort(KX[loc])                # sort vec indices by dist to loc
    all_xnbrs = np.zeros([n, embed_dim+1])
    result = []
    for y_idx, y in enumerate(Y):
        for lib_size in lib_sizes:
            for rep in range(n_replicates):
                if lib_size is None:
                    all_xnbrs = RX[:, :embed_dim+1]
                else:
                    if variable_libs:
                        rnd_lib = my_random_choice(a=n, size=lib_size, n=n, replace=replace).T
                        for loc in range(n):
                            mask = np.in1d(RX[loc,:], rnd_lib[loc,:])
                            all_xnbrs[loc,:] = RX[loc,:][mask][:embed_dim+1]
                    else:
                        rnd_lib = np.random.choice(n, size=lib_size, replace=replace)
                        for loc in range(n):
                            mask = np.in1d(RX[loc,:], rnd_lib)
                            all_xnbrs[loc,:] = RX[loc,:][mask][:embed_dim+1]
                # STEP TWO: compute observed cross-map skill
                if weights == 'uniform':
                    yhat = np.mean(y[all_xnbrs.astype(int)], axis=1) # compute observed xmap skill
                elif weights == 'exp':
                    yhat = [np.sum( y[all_xnbrs[loc,:].astype(int)] *
                                    get_weights(KX[loc,all_xnbrs[loc,:].astype(int)])
                                  )
                            for loc in range(n)]
                    yhat = np.array(yhat)
                else:
                    raise ValueError('weights must be `exp` or `uniform`.')
                result.append([lib_size, y_idx, score(y,yhat)])
    result = np.array(result)
    return DataFrame(result, columns=['lib_size', 'causer', 'score'])

def choose_embed_params(ts, ed_grid=np.arange(1,9), tau_grid=np.arange(1,9), weights='exp', score=pcorr):
    # ts is x series.
    data = np.zeros([len(ts), 2])
    data[:,0] = ts 
    data[:,1] = ts 
    result = []
    for embed_dim in ed_grid:
        for tau in tau_grid:
            rho = ccm_loocv(data, embed_dim=embed_dim, tau=tau, 
                            pred_lag=1, weights=weights,
                            score=score).score.values.item()
            result.append([embed_dim, tau, rho])
    result = np.array(result)
    winner_idx = np.argmax(result[:,2])
    # return result[winner_idx, 0].astype(int), result[winner_idx, 1].astype(int)
    return result
#%%
start = time.time()
embed_dim0, tau0 = choose_embed_params(x)
time_choose_et = time.time()-start
#%%
start = time.time()
# embed_dim0, tau0 = choose_embed_params(x)
result3 = choose_embed_params(x)
time_choose_et5 = time.time()-start
#%% linh ver

def ccm_loocv_sub_linh(KX, RX,  y_idx, y, lib_sizes, n_replicates, variable_libs, weights, embed_dim, tau, n, replace, score):
    for lib_size in lib_sizes:
        for rep in range(n_replicates):
            if lib_size is None:
                all_xnbrs = RX[:, :embed_dim+1]
            else:
                if variable_libs:
                    rnd_lib = my_random_choice(a=n, size=lib_size, n=n, replace=replace).T
                    for loc in range(n):
                        mask = np.in1d(RX[loc,:], rnd_lib[loc,:])
                        all_xnbrs[loc,:] = RX[loc,:][mask][:embed_dim+1]
                else:
                    rnd_lib = np.random.choice(n, size=lib_size, replace=replace)
                    for loc in range(n):
                        mask = np.in1d(RX[loc,:], rnd_lib)
                        all_xnbrs[loc,:] = RX[loc,:][mask][:embed_dim+1]
            # STEP TWO: compute observed cross-map skill
            if weights == 'uniform':
                yhat = np.mean(y[all_xnbrs.astype(int)], axis=1) # compute observed xmap skill
            elif weights == 'exp':
                yhat = [np.sum( y[all_xnbrs[loc,:].astype(int)] *
                                get_weights(KX[loc,all_xnbrs[loc,:].astype(int)])
                              )
                        for loc in range(n)]
                yhat = np.array(yhat)
            else:
                raise ValueError('weights must be `exp` or `uniform`.')
    return [lib_size, y_idx, embed_dim, tau, score(y,yhat)]
    
    
def ccm_loocv(data, embed_dim, tau=1, lib_sizes=[None], replace=False,
              n_replicates=1, weights='exp', pred_lag=0, score=pcorr,
              variable_libs=False):
    """
    Args:
        data (array): array with n rows and m+1 columns, where n is the number
            of time points and m is the number of putative causer variables. The
            first column is the putative causee
        embed_dim (int): embedding dimension
        tau (int): delay for delay embedding
        lib_sizes (iterable): list of library sizes. Default is maximum library size
        replace (False): replacement for libraries chosen from sizes below maximum
        n_replicates (int): number of replicate runs
        weights (string): weighting function. Choices are 'exp' or 'uniform'
        variable_libs (bool): If this parameter is True, and if lib_sizes is
            used, each delay vector uses a different random library.
    Returns:
        Pandas DataFrame. The column 'causer' denotes the index of the putative
            causer
    """
    # print(data, embed_dim, tau, lib_sizes, replace,
    #               n_replicates, weights, pred_lag, score,
    #               variable_libs)
    X, _ = setup_problem(data, embed_dim=embed_dim, tau=tau, pred_lag=pred_lag)
    n_y = data.shape[-1] - 1 # number of replicates/ surrogates
    Y = [setup_problem(data[:,[0,i+1]], embed_dim=embed_dim, tau=tau, pred_lag=pred_lag)[1] for i in range(n_y)]
    # STEP ONE: get true neigbor indices
    KX = distance_matrix(X,X)
    np.fill_diagonal(KX, np.inf) # not allowed to train on self
    (n, embed_dim) = X.shape
    # RX = np.zeros_like(KX)
    # for loc in range(n):
    #     RX[loc,:] = np.argsort(KX[loc])                # sort vec indices by dist to loc
    RX = np.array([np.argsort(loc) for loc in KX])
    
    result = np.array([ccm_loocv_sub_linh(KX, RX,  y_idx, y, 
                                     lib_sizes, n_replicates, 
                                     variable_libs, weights, 
                                     embed_dim, tau, n, replace, score)
                       for y_idx,y in enumerate(Y)])

    return DataFrame(result, columns=['lib_size', 'causer', 'embed_dim', 'tau',  'score'])

def choose_embed_params(ts, ed_grid=np.arange(1,9), tau_grid=np.arange(1,9), weights='exp', score=pcorr):
    # ts is x series.
    mm = len(ed_grid)
    mmm = mm*mm
    data = np.zeros([len(ts), 2])
    data[:,0] = ts 
    data[:,1] = ts 
    # dat = np.tile(np.array(data),mmm)
    ccm_res = list(map(ccm_loocv, [data]*mmm, np.repeat(ed_grid,mm), np.tile(tau_grid,mm), [[None]]*mmm, [False]*mmm,
                  [1]*mmm, ['exp']*mmm, [1]*mmm, [score]*mmm, 
                  [False]*mmm))
    # ccm_loocv(data, embed_dim, tau=1, lib_sizes=[None], replace=False,
    #               n_replicates=1, weights='exp', pred_lag=0, score=pcorr,
    #               variable_libs=False)
    
    # return ccm_res

    result = np.array([[_['embed_dim'][0], _['tau'][0], _['score'][0]] for _ in ccm_res])
    winner_idx = np.argmax(result[:,2])
    # return result[winner_idx, 0].astype(int), result[winner_idx, 1].astype(int)
    return result
#%%
start = time.time()
embed_dim2, tau2 = choose_embed_params(x)
time_choose_et3 = time.time()-start
#%%
start = time.time()
# embed_dim2, tau2 = choose_embed_params(x)
result4 = choose_embed_params(x)
time_choose_et6 = time.time()-start