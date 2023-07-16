#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 18:09:17 2023

@author: h_k_linh

Surrogate protocols and wrapping around the workflow
(specify which correlation statistics is being used with which surrogate test,
scan lags for the original pair of time series and each pair of surrogate and the other series
calculate p-value)

"""
import os
# os.chdir('C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/')
# os.chdir('/home/h_k_linh/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/')
os.getcwd()

import time
import numpy as np


# from scipy import stats # t-distribution
from scipy.spatial import distance_matrix # for twin
from pyunicorn.timeseries import surrogates #for iaaft

import pickle
import dill
# from multiprocessing import Pool, Process, Queue
import sys
sys.path.append('/home/hoanlinh/Simulation_test/Simulation_code/surrogate_dependence_test')
from multiprocessor import Multiprocessor

#%%% Surrogates
    #%%%% Random shuffle
def get_perm_surrogates(timeseries, n_surr=99):
    print("\t\t\t\tpermutations")
    sz = timeseries.size;
    result = np.zeros([sz,n_surr]); # Create mentioned matrix of surrogates
    for col in range(n_surr):
        result[:,col] = np.random.choice(timeseries,size=sz,replace=False);
    return result;
    #%%%% Stationary bootstrap (block boostraps)
"""
Randomly select consecutive blocks of data to construct surrogate sample
"""
def get_stationary_bstrap_surrogates(timeseries, p_jump=0.05, n_surr=99):
    print("\t\t\t\tstationary bootstrap")
    # According to Caroline p_jump = alpha
    sz = timeseries.size
    
    # Create surrY matrix but first use it to define index of surrYs 
    result = np.zeros([sz, n_surr]);
    
    # Define indices of all surrY_0, replace = T => repeated values 
    # According to Caroline, choose a random interger T(1) in [1,N];
    result[0,:] = np.random.choice(sz, size=n_surr, replace=True)    
    for col in range(n_surr):
        for row in range(1,sz):
            # with probability alpha we pick new T(k) => new block
            if np.random.random() < p_jump:
                result[row,col] = np.random.choice(sz)
            # with probability 1-alpha T(k)=(T(k-1)+1) mod N # => size of block
            else: 
                result[row,col] = (result[row-1,col] + 1) % sz
            # = > total random blocks with total random size
    # after having full index matrices, take values
    for col in range(n_surr):
        for row in range(sz):
            result[row,col] = timeseries[int(result[row,col])]
    return result
    #%%%% IAAFT Iterative Amplitude-adjusted Fourier transform (random phase)
"""
Works for stationary linear Gaussian process.
Wiener-Khinchin
Steps:
    Decompose Y into sine waves = Fourier Transform
    Shift sine wave's phases
    Add shifted waves togethers
AAFT:
    for x(t) = (a(t))^3 with a(t) stationary linear while x(t) is nonlinear, 
    x(t) can be made linear = invertible, time dependent scaling function
IAAFT:
    scaling and rescaling after phase shift might alter the amplitude 
    (power spectrum) of surrY so we iteratively adjust the amplitudes of 
    scaled data.
"""
def get_iaaft_surrogates(timeseries, n_surr=99, n_iter=200):
    print("\t\t\t\trandom phase iaaft")
    # prebuilt algorithm in pyunicorn so dont have to create empty matrix
    results = []
    i = 0
    while i < n_surr:
        obj = surrogates.Surrogates(original_data=timeseries.reshape(1,-1), silence_level=2)
        surr = obj.refined_AAFT_surrogates(original_data=timeseries.reshape(1,-1), n_iterations=n_iter, output='true_spectrum')
        surr = surr.ravel()
        if not np.isnan(np.sum(surr)):
            results.append(surr)
            i += 1
    return np.array(results).T

    #%%%% Circular permutation (time shift)
def get_circperm_surrogates(timeseries):
    print("\t\t\t\tcircular permutations")
    result = np.zeros([timeseries.size, timeseries.size])
    for i in range(timeseries.size):
        result[:,i] = np.roll(timeseries, i)
    return result
    #%%%% Trimming to remove discontinuity then run with IAAFT and Circular permutation
"""
Fourier transform and circular TS can fail if there is discontinuity between 
beginning and end of timeseries.
=> Trimming method from Lancaster 2018
estimate period of timeseries T with fast Fourier transform and set parameter 
p = T/10
"""
def trim_periodic_data(y,p=0):
    print("\t\t\t\t + trimming data to remove discontinuity")
    if (p == 0):
        #Find period of signal (very simple)
        
        #subtract mean to get rid of 0 peak
        y_clean = y - np.mean(y);
        #get fourier transform
        fft = np.fft.rfft(y_clean)
        #first peak of fft
        freq = np.argmax(np.abs(fft)) # Nyquist frequency f = N/2T
        T = len(y)/freq/2 
        p = int(np.ceil(T/10))
    
    #Now, run trimming algorithm
    trunc_max = int(np.floor(len(y)/10));
    match_mat = np.zeros((trunc_max,trunc_max))
    
    #grid search for best k1 and k2
    for k1 in np.arange(trunc_max):
        for k2 in np.arange(len(y) - trunc_max - p + 1, len(y) - p + 1):
            y_left = y[k1:k1+p]
            y_right = y[k2:k2+p]
            match_mat[k1,len(y)-p-k2] = np.sum((y_left - y_right)**2);
    
    k_start, k_end = np.unravel_index(np.argmin(match_mat),match_mat.shape)
    #change from index to actual k2
    k_end = len(y) - p - k_end;
    
    return (k_start,k_end);
    #%%%% Twin method

from ccm_xory import choose_embed_params

"""
Embed data into a delay space:
    Neighbors of a focal point = points within predefined distance.
    Two points are twins if they have same neighbors.
Steps:
    choose random starting points
    if it has no twins, following point = its following points in normal space
    if it has twins in delay space, following point = following point of twins in normal space 
Works with N0 = two time series are independent and Markovian.
Parameters:
    - Embedding dimension and delay chose from grid serach and pick the 
    values that give highest cross map skill of y to itself
    - Neighbor distance chosen so that on average, each point in delay space
    has specified number of neighbors (10% of total)
"""
def create_embed_space(data, embed_dim, tau, pred_lag=0):
    """
    Create embed space, used in  choose_twin_threshold
    Prepares a "standard matrix problem" from the time series data

    Args:
        data (array): Created in choose_twin_threshold, data = np.tile(timeseries, (2,1)).T:
            A 2d array with two columns where the first column
            will be used to generate features and the second column will
            be used to generate the response variable
        embed_dim (int): embedding dimension (delay vector length) , calculated from ccm
        tau (int): delay between values
        pred_lag (int): prediction lag
    """
    # print("Twin embedding")
    x = data[:,0]
    y = data[:,1]
    feat = []
    resp = []
    idx_template = np.array([pred_lag] + [-i*tau for i in range(embed_dim)])
    for i in range(x.size):
        if np.min(idx_template + i) >= 0 and np.max(idx_template + i) < x.size: # make sure indices is not out of bound
            feat.append(x[idx_template[1:] + i])
            resp.append(y[idx_template[0] + i])
            # length = original length - tau *(embed_dim -1)
    return np.array(feat), np.array(resp)

# calculate distance matrices in embed spaces, 
# return the distance matrix of the embed space with the largest distances
def max_distance_matrix(X, Y):
    """
    returns max norm distance matrix

    Args:
        X (array): an m-by-n array where m is the number of vectors and n is the
            vector length
        Y (array): same shape as X
    """
    # print("Generating twin distance matrix")
    n_vecs, n_dims = X.shape # original length -tau*(embed_dim-1); embed_dim
    K_by_dim = np.zeros([n_dims, n_vecs, n_vecs]) 
    # embed_dim matrices of disctance between each embed dim. dim1xdim1, dim2xdim2,...
    for dim in range(n_dims):
        K_by_dim[dim,:,:] = distance_matrix(X[:,dim].reshape(-1,1), Y[:,dim].reshape(-1,1))
    return K_by_dim.max(axis=0) # out of the 5 distance matrices - embed_dim =5

# predefined distance to define twins
# defined so that on average each point in delay space has specified # of neighbors
# (12% total points, later set to 10 % in get_twin_surrogates)
def choose_twin_threshold(timeseries, embed_dim, tau, neighbor_frequency=0.12, distmat_fxn=max_distance_matrix):
    """Given a univariate timeseries, embedding parameters, and a twin frequency,
        choose the twin threshold.

       Args:
         timeseries (numpy array): a univariate time series
         embed_dim (int): embedding dimension
         tau (int): embedding delay
         neighbor_frequency (float): Fraction of the "recurrence plot" to choose
             as neighbors. Note that not all neighbors are twins.

        Returns:
          recurrence distance threshold for twins
    """
    # print("Choosing twin thresholds")
    # timeseries is 1d
    timeseries = np.copy(timeseries.flatten())
    data_ = np.zeros([timeseries.size, 2])
    data_[:,0] = timeseries
    data_[:,1] = timeseries
    # or data_ = np.tile(timeseries, (2,1)).T
    X, y = create_embed_space(data_, embed_dim=embed_dim, tau=tau) # feat, resp
    K = distmat_fxn(X,X) # max_distance_matrix, time series length = 400, embed = 5 , tau = 2 then K.shape = (392,392)
    #np.fill_diagonal(K, np.inf) # self-neighbors are allowed in recurrence plot.
    k = K.flatten()
    k = np.sort(k) # sort all distances
    idx = np.floor(k.size * neighbor_frequency).astype(int)
    # print(f"Twin threshold for distance is {k[idx]}")
    return k[idx]


#twin method
def get_twin_surrogates(timeseries, embed_dim, tau, num_surr=99,
                        neighbor_frequency=0.1, th=None):
    if th is None:
        th = choose_twin_threshold(timeseries, embed_dim, tau, neighbor_frequency)
    results = [] # i=0
    obj = surrogates.Surrogates(original_data=timeseries.reshape(1,-1), silence_level=2)
    for i in range(num_surr):
        surr = obj.twin_surrogates(original_data=timeseries.reshape(1,-1), dimension=embed_dim, delay=tau, threshold=th, min_dist=1)
        surr = surr.ravel()
        results.append(surr)
    return np.array(results).T

def get_twin_wrapper(timeseries,num_surr=99):
    print("\t\t\t\ttwin")
    embed_dim, tau = choose_embed_params(timeseries);
    surrs = get_twin_surrogates(timeseries,embed_dim,tau,num_surr);
    return surrs;
#%%% tts
def choose_r(n):
    delta = (n/4 + 1) % 20
    return int(n/4 - delta)

# Compared to other surrogates fxn, this tts already calculate the statistics
def tts(x,y,r):
    # time shift
    t = len(x); #number of time points in orig data
    xstar = x[r:(t-r)]; # middle half of data - truncated original X0
    tstar = len(xstar); # length of truncated series
    ystar = y[r:(t-r)]; # truncated original Y0
    t0 = r;             # index of Y0 in matrix of surrY
    
    y_surr = np.tile(ystar,[2*r+1, 1]) # Create empty matrix of surrY
    print("\t\t\t\ttts")
    #iterate through all shifts from -r to r
    for shift in np.arange(t0 - r, t0 + r + 1):
        y_surr[shift-t0] = y[shift:(shift+tstar)];
    return xstar, ystar, y_surr[1:].T

#%% Correlation Statistics

from correlation_stats import correlation_Pearson
from correlation_stats import lsa_new_delay_linh
from correlation_stats import mutual_info
from ccm_xory import ccm_predict_surr, ccm_surr_predict
from granger_xory import granger_predict_surr, granger_surr_predict
#%% scan_lags and calculate pvals

def iter_scanlag(i, statistic, x, y, kw_statistic):
    return [i, dill.loads(statistic)(x,y,**kw_statistic)]
    # statistic(x[:-2*i],y[2*i:],**kw_statistic)
    
def scan_lags(x,y,statistic,maxlag=5,kw_statistic={}):
    print("\t\t\t\t\tScanning best lags with step lags = 2")

    score_sim = statistic(x,y,**kw_statistic);
    # rows: # of y rows
    # columns: # of lags to test, # test lags of (+-)2,4,6,8,10
    score = np.zeros((score_sim.size,2*maxlag+1)); 
    score[:,maxlag] = score_sim # middle column = no lag statistics
    lags = np.arange(-2*maxlag,2*maxlag+1,2); 
        
    if (maxlag > 0):
        fun = dill.dumps(statistic)
        mp = Multiprocessor()
        for i in np.arange(1,maxlag+1):
            # print(f"\t\t\t\t\t\tmaxlag {i}.1")
            # print(f"\t\t\t\t\t\tlag: x_{2*i} and y_{-2*i}")
            ARGs = (maxlag+i, fun, x[2*i:],y[:-2*i], kw_statistic)
            mp.add(iter_scanlag, ARGs)
            ARGs = (maxlag-i, fun, x[:-2*i],y[2*i:], kw_statistic)
            mp.add(iter_scanlag, ARGs)
        mp.run(2) 
        rets = mp.results()
        for _ in rets: 
            score[:,_[0]] = np.array(_[1])
    return np.vstack((lags,score));

def create_surr(x, y, surr, r_tts = choose_r):
    print("\t\t\tGet surrogates")
    if surr == 'perm':
        surr_fxn = get_perm_surrogates
    if surr == 'bstrap':
        surr_fxn = get_stationary_bstrap_surrogates
    if surr == 'twin':
        surr_fxn = get_twin_wrapper
    if surr == 'tts_naive':
        surr_fxn = tts
    if surr == 'randphase':
        surr_fxn = get_iaaft_surrogates
    if surr == 'circperm':
        surr_fxn = get_circperm_surrogates
        
    test_params = {}
    if surr_fxn in [get_perm_surrogates, get_stationary_bstrap_surrogates, get_twin_wrapper]:
        xstar = x
        ystar = y
        test_params['no params'] = None
        surr = surr_fxn(y);
    if surr_fxn in [get_circperm_surrogates, get_iaaft_surrogates]:
        k_start,k_end = trim_periodic_data(y);
        test_params['trim kstart_kend'] = [k_start,k_end]
        xstar = x[k_start:k_end+1]
        ystar = y[k_start:k_end+1]
        surr = surr_fxn(ystar);
    if surr_fxn == tts:
        if type(r_tts) is type(1): # if manually pick r
            r = r_tts
            def r_tts(x):
                return r
        r = r_tts(x.size)
        test_params['r_tts'] = r
        xstar, ystar, surr = tts(x, y, r)
    return xstar, ystar, surr, test_params

def sig_test_good(xstar, ystar, surr, statistic, maxlag, shorter=False):     
    #find statistic from original data
    print("\t\t\t\tCalculating original stats with scan_lags")
    scanlag = scan_lags(xstar,ystar.reshape(-1,1),statistic,maxlag)
    score = np.max(scanlag[1]); # no np.max in test_bad
    
    # sometimes twin produce surrogates shorter than original series (example: FitzHugh_Nagumo_cont[14] embed_dim =3, tau = 1, the surr is only 498)
    if xstar.shape[0] != surr.shape[0]:
        shorter = True
        xstar = xstar[:surr.shape[0]]
    #find null statistic for each surrogate 
    #using same maximizing procedure as original
    print("\t\t\t\tCalculating null stats with scan_lags")
    scanlagsurr = scan_lags(xstar,surr,statistic,maxlag)
    null = np.max(scanlagsurr[1:],axis=1);
    
    #one tailed test
    pval = (np.sum(null >= score) + 1) / (np.size(null) + 1);
    if shorter:
        return {'scanlag': scanlag, 'scanlagsurr': scanlagsurr, 'score': score, 'null': null, 'pval': pval, 'surr_length': surr.shape[0]}
    else:
        return {'scanlag': scanlag, 'scanlagsurr': scanlagsurr, 'score': score, 'null': null, 'pval': pval}
"""
these have not calculated u value for tts
"""
#%%
def whichstats(stat, xory): 
    if stat == 'pearson':
        stat_fxn = correlation_Pearson
    if stat == 'lsa': 
        stat_fxn = lsa_new_delay_linh
    if stat == 'mutual_info':
        stat_fxn = mutual_info
    if xory == 'surrY':
        if stat == 'ccm_y->x':
            stat_fxn = ccm_predict_surr
        if stat == 'ccm_x->y':
            stat_fxn = ccm_surr_predict
        if stat == 'granger_y->x':
            stat_fxn = granger_surr_predict
        if stat == 'granger_x->y':
            stat_fxn = granger_predict_surr
    elif xory == 'surrX':
        if stat == 'ccm_y->x':
            stat_fxn = ccm_surr_predict
        if stat == 'ccm_x->y':
            stat_fxn = ccm_predict_surr
        if stat == 'granger_y->x':
            stat_fxn = granger_predict_surr
        if stat == 'granger_x->y':
            stat_fxn = granger_surr_predict
    return stat_fxn
    # 'pcc_param', 'granger_param_y->x', 'granger_param_x->y'
    
def iter_stats(a, b, surr, stats_list, maxlag, xory):#
    # print(a[0], b[0], stats_list, maxlag, xory)
    # res = ['pvals': {}, 'runtimes': {}, 'test_params': {}]
    test_params = {'maxlag': maxlag}
    A, B, SURR, test_params['surr_params'] = create_surr(a, b, surr)
    pvals = {}
    runtimes = {}
    for stat in stats_list:
        print(f'Running {stat} in many stats for Y or X surr')
        stat_fxn = whichstats(stat, xory)
        start = time.time()
        pvals[stat] = sig_test_good(A, B, SURR, stat_fxn, maxlag=maxlag)
        runtimes[stat] = time.time() - start
    return pvals, runtimes, test_params
   
# from multiprocessing import Pool 
def manystats_manysurr(x, y, stats_list='all', test_list='all', maxlag=0, kw_randphase={}, kw_bstrap={}, kw_twin={}, r_tts=choose_r, r_naive=choose_r):
    if test_list == 'all':
        test_list = ['randphase', 'bstrap', 'twin','tts', 'tts_naive', 'circperm','perm']
    if stats_list == 'all':
        stats_list = ['pearson', 'lsa', 'mutual_info', 
                      'granger_y->x', 'granger_x->y', 
                      'ccm_y->x', 'ccm_x->y']
    test_list = set(['tts_naive' if 'tts' in _ else _ for _ in test_list])
    
    # result = []
    print(f'Many stats many surr {stats_list}, {test_list}')
    score_null_pval = {'surrY': {}, 'surrX': {}};
    runtime = {'surrY': {}, 'surrX': {}};
    test_params = {'surrY': {}, 'surrX': {}}
    
    for surr in test_list:
        ARGsurrY = (x, y, surr, stats_list, maxlag, 'surrY')# 
        ARGsurrX = (y, x, surr, stats_list, maxlag, 'surrX')# 
        print(f'Running {surr} for Y')
        score_null_pval['surrY'][surr], runtime['surrY'][surr], test_params['surrY'][surr] = iter_stats(*ARGsurrY)
        print(f'Running {surr} for X') 
        score_null_pval['surrX'][surr], runtime['surrX'][surr], test_params['surrX'][surr] = iter_stats(*ARGsurrX)
    # tts_p = {xory : {'tts': {stat : [B * (2 * r + 1) / (r + 1) for B in pvals[xory]['tts_naive'][stat] ]
    #                         for stat in new_res['stats_list']}
    #                     }
    result = {'score_null_pval': score_null_pval, 'runtime': runtime, 'test_params': test_params, 'status': 'corrected'}
    
    return result 