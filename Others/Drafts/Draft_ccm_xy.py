# -*- coding: utf-8 -*-
"""
Created on Thu May 25 22:10:48 2023

@author: hoang

Breaking down Caroline's help script to understand

"""
import os
os.getcwd()
# os.chdir('C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/')
# os.chdir('/home/h_k_linh/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/')
os.getcwd()

import time
import numpy as np
# import pandas as pd 

# import matplotlib.pyplot as plt
# import seaborn as sns
# from mergedeep import merge

# from scipy import stats # t-distribution
from scipy.spatial import distance_matrix # for twin
from pyunicorn.timeseries import surrogates #for iaaft
# import ccm_xory as ccm # for ccm
# from statsmodels.tsa.stattools import grangercausalitytests as granger # for granger
# from statsmodels.tsa.api import VAR # vector autoregression for granger
# from sklearn.feature_selection import mutual_info_regression # for mutual information

import pickle
import dill
# from multiprocessing import Pool, Process, Queue
from multiprocessor import Multiprocessor

#%% Data
# series = all_models.xy_uni_logistic.series[0]
with open('/home/h_k_linh/Desktop/data.pkl', 'rb') as fi:
    series = pickle.load(fi)
#%% Surrogates
""" Surrogates does not include original series
because in sig_test we calculate original statistics separately
"""
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
    embed_dim, tau, pred_lag = choose_embed_params(timeseries);
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
    for shift in np.arange(t0 - r, t0 + r):
        # print(f'shift-t0:{shift-t0}\ty[shift:(shift+tstar)]: {shift}:{shift+tstar}')
        y_surr[shift-t0] = y[shift:(shift+tstar)];
    # print(f"\t\t\t\tCalculating {statistic.__name__} for tts surrogates")
    # importance_true = np.max(scan_lags(xstar, ystar.reshape(-1,1),statistic,maxlag)[1]); # scan_lags returns np.vstack((lags,score))
    # importance_surr = np.max(scan_lags(xstar, y_surr.T, statistic,maxlag)[1:],axis=1);
    # sig = np.mean(importance_surr > importance_true, axis=0);
    # return sig;
    return xstar, ystar, y_surr[1:].T

#%% Run to generate surr
# surrperm = get_perm_surrogates(series[1])
# surrbstrp = get_stationary_bstrap_surrogates(series[1])
# surrtwin = get_twin_wrapper(series[1])

xstar, ystar, surrtts = tts(series[0], series[1], 119)

# k_start,k_end = trim_periodic_data(series[1]);
# xtrim = series[0][k_start:k_end+1]
# ytrim = series[1][k_start:k_end+1]

# surrcircperm = get_circperm_surrogates(ytrim)
# surrrandphase = get_iaaft_surrogates(ytrim)

#%% Statistics
# start = time.time()

# time.time() - start
from Correlation_Surrogate_tests import correlation_Pearson
from Correlation_Surrogate_tests import lsa_new_delay_linh
from Correlation_Surrogate_tests import mutual_info

# statspearson_star = correlation_Pearson(xstar, ystar.reshape(-1,1))
# statspearson = correlation_Pearson(xstar, surrtts)

# statslsa_start = lsa_new_delay_linh(xstar, ystar.reshape(-1,1))
# statslsa = lsa_new_delay_linh(xstar, surrtts)

# statsmi_star = mutual_info(xstar, ystar.reshape(-1,1))
# statsmi = mutual_info(xstar, surrtts)

from ccm_xory import ccm_predict_surr, ccm_surr_predict

# start = time.time()
# statsccm_star_y_cause_x = ccm_predict_surr(xstar, ystar) 
# time.time() - start
# around 4 sec after pooling 0.2271561622619629 sec
# start = time.time()
# statsccm_y_cause_x_surrY = ccm_predict_surr(xstar, surrtts) 
# time.time() - start
# around 5 mins after pooling 2.6371231079101562 sec

# statsccm_star_x_cause_y = ccm_predict_surr(ystar, xstar)
# statsccm_x_cause_y_surrY = ccm_surr_predict(xstar, surrtts)

from granger_xory import granger_predict_surr, granger_surr_predict

# start = time.time()
# statsgranger_star_y_cause_x = granger_predict_surr(xstar, ystar.reshape(-1,1)) 
# time.time()- start
# # 0.1222529411315918

# start = time.time()
# statsgranger_y_cause_x_surrY = granger_predict_surr(xstar, surrtts) 
# time.time()- start
# # 2.78220534324646

# start = time.time()
# statsgranger_star_x_cause_y = granger_predict_surr(ystar, xstar.reshape(-1,1))
# time.time()- start
# # 0.016766071319580078

# start = time.time()
# statsgranger_x_cause_y_surrY = granger_surr_predict(xstar, surrtts)
# time.time()- start
# # 2.87807035446167

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
        
            
            # score[:,maxlag+i] = statistic(x[2*i:],y[:-2*i],**kw_statistic);
            # print(f"\t\t\t\t\t\tmaxlag {i}.2")
            # # print(f"\t\t\t\t\t\tlag: x_{-2*i} and y_{2*i}")
            # score[:,maxlag-i] = statistic(x[:-2*i],y[2*i:],**kw_statistic);
                         
        # with Pool(5) as p: 
        #     test = list(p.starmap(iter_scanlag, ARGs))
        #     p.close()
        #     p.join()
        # for _ in test: 
        #     score[:,_[0]] = np.array(_[1])
            
    return np.vstack((lags,score));

def create_surr(x, y, surr, r_tts = choose_r):
    print("\t\t\tGet surrogates")
    if surr == 'perm':
        surr_fxn = get_perm_surrogates
    if surr == 'bstrap':
        surr_fxn = get_stationary_bstrap_surrogates
    if surr == 'twin':
        surr_fxn = get_twin_wrapper
    if surr == 'tts':
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
        surr = surr_fxn(y);
    if surr_fxn == tts:
        if type(r_tts) is type(1): # if manually pick r
            r = r_tts
            def r_tts(x):
                return r
        # if type(r_naive) is type(1): # backup plan for r_tts?
        #     r_ = r_naive
        #     def r_naive(x):
        #         return r_
        r = r_tts(x.size)
        test_params['r_tts'] = r
        xstar, ystar, surr = tts(x, y, r)
        return xstar, ystar, surr, test_params

def sig_test_good(xstar, ystar, surr, statistic, maxlag):     
    #find statistic from original data
    print("\t\t\t\tCalculating original stats with scan_lags")
    scanlag = scan_lags(xstar,ystar.reshape(-1,1),statistic,maxlag)
    score = np.max(scanlag[1]); # no np.max in test_bad
    
    #find null statistic for each surrogate 
    #using same maximizing procedure as original
    print("\t\t\t\tCalculating null stats with scan_lags")
    scanlagsurr = scan_lags(xstar,surr,statistic,maxlag)
    null = np.max(scanlagsurr[1:],axis=1);
    
    #one tailed test
    pval = (np.sum(null >= score) + 1) / (np.size(null) + 1);
    return {'scanlag': scanlag, 'scanlagsurr': scanlagsurr, 'score': score, 'null': null, 'pval': pval}
#%%
x= series[0]
y= series[1]

with open('/home/h_k_linh/Desktop/result3.pkl', 'rb') as fi:
    results3 = pickle.load(fi)

# start = time.time()
# scanlag_pear = scan_lags(xstar, ystar.reshape(-1,1), correlation_Pearson)
# time.time()-start
# # 0.0018665790557861328 sec
# start = time.time()
# scanlag_pear_surr = scan_lags(xstar, surrtts, correlation_Pearson)
# time.time()-start
# # 0.008819818496704102 sec

# start = time.time()
# scanlag_lsa = scan_lags(xstar, ystar.reshape(-1,1), lsa_new_delay_linh)
# time.time()-start
# # 0.34944987297058105 sec
# start = time.time()
# scanlag_lsa_surr = scan_lags(xstar, surrtts, lsa_new_delay_linh)
# time.time()-start
# # 42.15380597114563 sec

# start = time.time()
# scanlag_mi = scan_lags(xstar, ystar.reshape(-1,1), mutual_info)
# time.time()-start
# # 0.032531023025512695 sec
# start = time.time()
# scanlag_mi_surr = scan_lags(xstar, surrtts, mutual_info)
# time.time()-start
# # 3.583428144454956 sec

# start = time.time()
# scanlag_ccm = scan_lags(xstar, ystar.reshape(-1,1), ccm_predict_surr)
# time.time()-start
# # 12.223682880401611 sec
# # result of scan_lags(xstar, ystar.reshape(-1,1), ccm_predict_surr) is the same as statsccm_star_y_cause_x # 4.589737415313721 sec

# start = time.time()
# scanlag_ccm_surr = scan_lags(xstar, surrtts, ccm_predict_surr)
# time.time()-start
# # 20.460524559020996 sec
# # results in the middle is the same as statsccm_y_cause_x_surr
# # test = np.max(scanlag_ccm_surr[1:],axis=1); test = np.argmax(scanlag_ccm_surr[1:],axis=1)

# start = time.time()
# scanlag_ccm_long = scan_lags(xstar, ystar.reshape(-1,1), ccm_surr_predict)
# time.time()-start
# # after using Pool 1.7416932582855225 sec, 1.9149584770202637 sec, 1.8591697216033936 sec 1.927781105041504 sec
# # 11.399733066558838 sec
# # result of test = scan_lags(xstar, ystar.reshape(-1,1), ccm_surr_predict) is the same as statsccm_star_y_cause_x



# start = time.time()
# scanlag_ccm_long_surr3 = scan_lags(xstar, surrtts, ccm_surr_predict)
# time.time()-start
# # after using Pool 463.555686712265
# scanlag 2, predsurr 5, surrpred 2,  4.46G RAM, 494.00011563301086 sec
# scanlag 2, predsurr 5, surrpred 3, 4.8G RAM, 526.3753848075867 sec
# turns out that RAM is still 4.45G when not running... so it's not the RAM, it's the cores.

# # Compare scanlag_ccm vs scanlag_ccm_long vs test1 and test2
# # scanlag_ccm = scan_lags(xstar, ystar.reshape(-1,1), ccm_predict_surr)
# # test2 = scan_lags(ystar, xstar.reshape(-1,1), ccm_predict_surr)
# # scanlag_ccm_long = scan_lags(xstar, ystar.reshape(-1,1), ccm_surr_predict)
# # test1 = scan_lags(ystar, xstar.reshape(-1,1), ccm_surr_predict)
# # middle point of test1 and scanlag_ccm are the same, the lags are symmetry about the middle point
# # middle point of test2 and scanlag_ccm_long are the same, the lags are symmetry about the middle point -> correct code
#%%
# start = time.time() 
# res_pearson_tts = sig_test_good(x, y, correlation_Pearson, tts, 5)
# time.time()-start
# # 0.009849071502685547

# start = time.time() 
# res_lsa_tts = sig_test_good(x, y, lsa_new_delay_linh, tts, 5)
# time.time()-start
# # 43.08095693588257

# start = time.time() 
# res_mi_tts = sig_test_good(x, y, mutual_info, tts, 5)
# time.time()-start
# # 3.3142948150634766

# start = time.time()
# res_ccm_tts = sig_test_good(xstar, ystar, surrtts, ccm_predict_surr, 5)
# time.time()-start
# # 24.74372172355652
# after pool 24.009589672088623
# # sum(res_ccm_tts['null'] - np.max(scanlag_ccm_surr[1:],axis=1))

# start = time.time()
# res_ccm_long_tts5 = sig_test_good(xstar, ystar, surrtts, ccm_surr_predict, 5)
# time.time()-start

# for _ in np.arange(11):
#     print(np.sum(np.sort(scanlag_ccm_long_surr2[1:,_])- np.sort(scanlag_ccm_long_surr2[1:,_])))
    
# for _ in np.arange(11):
#     print(np.sum(np.sort(res_ccm_long_tts3['scanlagsurr'][1:,_])- np.sort(scanlag_ccm_long_surr2[1:,_])))
    
# # # 1076.1955161094666 sec 
# # after pool 490.7074406147003 sec, 520.7803153991699
# a = np.sort(np.array(results['statsccm_y_cause_x_surrY']['score']))
# a1= np.sort(scanlag_ccm_long_surr3[1:,5])
# a2= np.max(scanlag_ccm_long_surr3[1:],axis=1)
# a3= np.sort(res_ccm_long_tts['null'])
# a4= np.sort(np.array(results3['scanlag_ccm_long_surr3'][1:,5]))
#%%
def whichstats(stat): 
    if stat == 'pearson':
        stat_fxn = correlation_Pearson
    if stat == 'lsa': 
        stat_fxn = lsa_new_delay_linh
    if stat == 'mutual_info':
        stat_fxn = mutual_info
    if stat == 'ccm_y->x':
        stat_fxn = ccm_predict_surr
    if stat == 'ccm_x->y':
        stat_fxn = ccm_surr_predict
    if stat == 'granger_y->x':
        stat_fxn = granger_surr_predict
    if stat == 'granger_x->y':
        stat_fxn = granger_predict_surr
    return stat_fxn
    # 'pcc_param', 'granger_param_y->x', 'granger_param_x->y'
    
def iter_stats(a, b, surr, stats_list, maxlag):#, xory
    # print(a[0], b[0], stats_list, maxlag, xory)
    # res = ['pvals': {}, 'runtimes': {}, 'test_params': {}]
    test_params = {'maxlag': maxlag}
    A, B, SURR, test_params['surr_params'] = create_surr(a, b, surr)
    pvals = {}
    runtimes = {}
    for stat in stats_list:
        stat_fxn = whichstats(stat)
        start = time.time()
        pvals[stat] = sig_test_good(A, B, SURR, stat_fxn, maxlag=maxlag)
        runtimes[stat] = time.time() - start
    return pvals, runtimes, test_params
   
# from multiprocessing import Pool 
def manystats_manysurr(x, y, stats_list='all', test_list='all', maxlag=0, kw_randphase={}, kw_bstrap={}, kw_twin={}, r_tts=choose_r, r_naive=choose_r):
    if test_list == 'all':
        test_list = ['randphase', 'bstrap', 'twin', 'tts', 'tts_naive', 'circperm','perm']
    if stats_list == 'all':
        stats_list = ['pearson', 'lsa', 'mutual_info', 
                      'granger_y->x', 'granger_x->y', 
                      'ccm_y->x', 'ccm_x->y',
                      'pcc_param', 'granger_param_y->x', 'granger_param_x->y', 'lsa_data_driven']
    # result = []
    score_null_pval = {'surrY': {}, 'surrX': {}};
    runtime = {'surrY': {}, 'surrX': {}};
    test_params = {'surrY': {}, 'surrX': {}}
    
    for surr in test_list:
        ARGsurrY = (x, y, surr, stats_list, maxlag)#, 'surrY'
        ARGsurrX = (y, x, surr, stats_list, maxlag)#, 'surrX'
        score_null_pval['surrY'][surr], runtime['surrY'][surr], test_params['surrY'][surr] = iter_stats(*ARGsurrY)
        score_null_pval['surrX'][surr], runtime['surrX'][surr], test_params['surrX'][surr] = iter_stats(*ARGsurrX)
    result = {'score_null_pval': score_null_pval, 'runtime': runtime, 'test_params': test_params}
    # mp = Multiprocessor()
    # for surr in test_list:
    #     print(f"\t\t\t'Surrogate - {surr}")
    #     ARGsurrY = (x, y, surr, stats_list, maxlag, 'surrY')
    #     ARGsurrX = (y, x, surr, stats_list, maxlag, 'surrX')
    #     mp.add(iter_stats, ARGsurrY)
    #     mp.add(iter_stats, ARGsurrX)
    # mp.run(2)
    # result = mp.results()
        
        # with Pool(2) as p:
        #     results = list(p.starmap(iter_stats, [ARGsurrY, ARGsurrX]))
        #     p.close()
        #     p.join()
        # results = list(map(iter_stats, [ARGsurrY, ARGsurrX]))
        
        # xstar, ystar, ysurr, test_paramsy = create_surr(x, y, surr)
        # for stat in stats_list:
        #     stat_fxn = whichstats(stat)
        #     start = time.time()
        #     score_null_pval['surrY'][stat] = {surr: sig_test_good(xstar, ystar, ysurr, stat_fxn, maxlag=maxlag)}
        #     runtime['surrY'][stat] = {surr: time.time() - start}
    
        # ystar, xstar, xsurr, test_paramsx = create_surr(y, x, surr) 
        # for stat in stats_list:
        #     stat_fxn = whichstats(stat)
        #     start = time.time()
        #     score_null_pval['surrX'][stat] = {surr: sig_test_good(ystar, xstar, xsurr, stat_fxn, maxlag=maxlag)}
        #     runtime['surrX'][stat] = {surr: time.time() - start}
    
    return result #, runtime, [test_paramsy, test_paramsx]

# score_null_pval['surrY'], runtimes['surrY'] = iter_stats()    
#     print("\t\t\tSurrX")
#%%
# start = time.time()
# res_omg_ccm = manystats_manysurr(x, y, stats_list=['ccm_y->x', 'ccm_x->y'], test_list= ['tts'], maxlag=5)
# time.time()-start
# ccm both ways 1129.1264700889587
# 'pearson', 'lsa', 'mutual_info', 'ccm_y->x', 'ccm_x->y', 'granger_y->x', 'granger_x->y'
# 2405.6757962703705
# #%%
# granger both ways 41.275888442993164 sec
# start = time.time()
# res_omg2, runtimes2 = manystats_manysurr(x, y, stats_list=['pearson', 'lsa', 'mutual_info', 'ccm_y->x', 'ccm_x->y'], test_list= ['tts'], maxlag=5)
# time.time()-start

with open('ccms_grangers_randphase_0_100_2.pkl', 'rb') as fi:
    xy_ar_u_ccmsgrangers = pickle.load(fi)
    
score_null_pval = [_[1]['score_null_pval'] for _ in xy_ar_u_ccmsgrangers['pvals'] ]
runtime = score_null_pval = [_[1]['runtime'] for _ in xy_ar_u_ccmsgrangers['pvals'] ]
test_params = score_null_pval = [_[1]['test_params'] for _ in xy_ar_u_ccmsgrangers['pvals'] ]

pvals = {xory: {cst: {stat : [_[xory][cst][stat]['pval'] 
                                  for _ in score_null_pval]
                          for stat in xy_ar_u_ccmsgrangers['stats_list']}
                for cst in xy_ar_u_ccmsgrangers['test_list']} 
          for xory in ['surrY', 'surrX']}