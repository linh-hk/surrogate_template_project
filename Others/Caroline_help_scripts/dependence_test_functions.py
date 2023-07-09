#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 13:41:40 2022

@author: carolc24
"""

#Statistical functions to be used in dependence testing

import numpy as np
import pandas as pd
import ccm
from pyunicorn.timeseries import surrogates
from scipy.spatial.distance import correlation as dcor
from sklearn.feature_selection import mutual_info_regression
from scipy.spatial import distance_matrix
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests as granger
from statsmodels.tsa.api import VAR

#%%Surrogates
#%%% twin
#for twin method
# def setup_problem(data, embed_dim, tau, pred_lag=0):
#     """Prepares a "standard matrix problem" from the time series data

#     Args:
#         data (array): A 2d array with two columns where the first column
#             will be used to generate features and the second column will
#             be used to generate the response variable
#         embed_dim (int): embedding dimension (delay vector length)
#         tau (int): delay between values
#         pred_lag (int): prediction lag
#     """
#     x = data[:,0]
#     y = data[:,1]
#     feat = []
#     resp = []
#     idx_template = np.array([pred_lag] + [-i*tau for i in range(embed_dim)])
#     for i in range(x.size):
#         if np.min(idx_template + i) >= 0 and np.max(idx_template + i) < x.size:
#             feat.append(x[idx_template[1:] + i])
#             resp.append(y[idx_template[0] + i])
#     return np.array(feat), np.array(resp)

# #for twin method
# def get_maxnorm_distmat(X, Y):
#     """
#     returns max norm distance matrix

#     Args:
#         X (array): an m-by-n array where m is the number of vectors and n is the
#             vector length
#         Y (array): same shape as X
#     """
#     n_vecs, n_dims = X.shape
#     K_by_dim = np.zeros([n_dims, n_vecs, n_vecs])
#     for dim in range(n_dims):
#         K_by_dim[dim,:,:] = distance_matrix(X[:,dim].reshape(-1,1), Y[:,dim].reshape(-1,1))
#     return K_by_dim.max(axis=0)

# #for twin method
# def choose_twin_threshold(timeseries, embed_dim, tau, neighbor_frequency=0.12, distmat_fxn=get_maxnorm_distmat):
#     """Given a univariate timeseries, embedding parameters, and a twin frequency,
# 		choose the twin threshold.

# 	   Args:
# 	     timeseries (numpy array): a univariate time series
# 		 embed_dim (int): embedding dimension
# 		 tau (int): embedding delay
# 		 neighbor_frequency (float): Fraction of the "recurrence plot" to choose
# 		 	as neighbors. Note that not all neighbors are twins.

# 		Returns:
# 		  recurrence distance threshold for twins
#     """
#     # timeseries is 1d
#     timeseries = np.copy(timeseries.flatten())
#     data_ = np.zeros([timeseries.size, 2])
#     data_[:,0] = timeseries
#     data_[:,1] = timeseries
#     X, y = setup_problem(data_, embed_dim=embed_dim, tau=tau)
#     K = distmat_fxn(X,X)
#     #np.fill_diagonal(K, np.inf) # self-neighbors are allowed in recurrence plot.
#     k = K.flatten()
#     k = np.sort(k)
#     idx = np.floor(k.size * neighbor_frequency).astype(np.int)
#     return k[idx]


# #twin method
# def get_twin_surrogates(timeseries, embed_dim, tau, num_surr=99,
# 						neighbor_frequency=0.1, th=None):
# 	if th is None:
# 		th = choose_twin_threshold(timeseries, embed_dim, tau, neighbor_frequency)
# 	results = [] # i=0
# 	obj = surrogates.Surrogates(original_data=timeseries.reshape(1,-1), silence_level=2)
# 	for i in range(num_surr):
# 		surr = obj.twin_surrogates(original_data=timeseries.reshape(1,-1), dimension=embed_dim, delay=tau, threshold=th, min_dist=1)
# 		surr = surr.ravel()
# 		results.append(surr)
# 	return np.array(results).T

# def get_twin_wrapper(timeseries,num_surr=99):
#     embed_dim, tau = ccm.choose_embed_params(timeseries);
#     surrs = get_twin_surrogates(timeseries,embed_dim,tau,num_surr);
#     return surrs;


#%%% TTS
#for tts
# def choose_r(n):
# 	delta = (n/4 + 1) % 20
# 	return int(n/4 - delta)

#truncated time shift
# def tts(x,y,r,statistic,maxlag=0):
#     # time shift
#     t = len(x); #number of time points in orig data
#     xstar = x[r:(t-r)]; # middle half of data
#     tstar = len(xstar); #length of middle half of data
#     ystar = y[r:(t-r)];
#     t0 = r;
    
#     importance_true = np.max(delay_scan(xstar, ystar.reshape(-1,1),statistic,maxlag)[1]);
    
#     y_surr = np.tile(ystar,[2*r+1, 1])
    
#     #iterate through all shifts from -r to r
#     for shift in np.arange(t0 - r, t0 + r):
        
#         y_surr[shift-t0] = y[shift:(shift+tstar)];
         
#     importance_surr = np.max(delay_scan(xstar, y_surr.T, statistic,maxlag)[1:],axis=1);
#     sig = np.mean(importance_surr > importance_true, axis=0);
#     return sig;

#truncated time shift
# def tts_bad(x,y,r,statistic,maxlag=0):
#     # time shift
#     t = len(x); #number of time points in orig data
#     xstar = x[r:(t-r)]; # middle half of data
#     tstar = len(xstar); #length of middle half of data
#     ystar = y[r:(t-r)];
#     t0 = r;
    
#     score_list = delay_scan(xstar, ystar.reshape(-1,1),statistic,maxlag);
#     [lag,importance_true] = score_list[:,np.argmax(score_list[1])]
#     lag = int(lag)
    
#     y_surr = np.tile(ystar,[2*r+1, 1])
    
#     #iterate through all shifts from -r to r
#     for shift in np.arange(t0 - r, t0 + r):
        
#         y_surr[shift-t0] = y[shift:(shift+tstar)];
         
#     if lag == 0:
#         importance_surr = statistic(xstar, y_surr.T)
#     elif lag > 0:
#         importance_surr = statistic(xstar[lag:],(y_surr.T)[:-lag])
#     else:
#         importance_surr = statistic(xstar[:lag],(y_surr.T)[-lag:])
#     sig = np.mean(importance_surr > importance_true, axis=0);
#     return sig;

#%%% random phase method IAAFT
#random phase method
# def get_iaaft_surr(data, num_surr=99, n_iter=200):
# 	results = []
# 	i = 0
# 	while i < num_surr:
# 		obj = surrogates.Surrogates(original_data=data.reshape(1,-1), silence_level=2)
# 		surr = obj.refined_AAFT_surrogates(original_data=data.reshape(1,-1), n_iterations=n_iter, output='true_spectrum')
# 		surr = surr.ravel()
# 		if not np.isnan(np.sum(surr)):
# 			results.append(surr)
# 			i += 1
# 	return np.array(results).T

#%%% block bootstrap method
#block bootstrap method
# def get_stationary_bstrap_surrogates(timeseries, p_jump=0.05, n_surr=99):
# 	sz = timeseries.size
# 	result = np.zeros([sz, n_surr])
# 	result[0,:] = np.random.choice(sz, size=n_surr, replace=True)
# 	for col in range(n_surr):
# 		for row in range(1,sz):
# 			if np.random.random() < p_jump:
# 				result[row,col] = np.random.choice(sz)
# 			else:
# 				result[row,col] = (result[row-1,col] + 1) % sz
# 	for col in range(n_surr):
# 		for row in range(sz):
# 			result[row,col] = timeseries[int(result[row,col])]
# 	return result
#%%% random shuffle method
#random shuffle method
# def get_perm_surrogates(timeseries, n_surr=99):
#     sz = timeseries.size;
#     result = np.zeros([sz,n_surr]);
#     for col in range(n_surr):
#         result[:,col] = np.random.choice(timeseries,size=sz,replace=False);
#     return result;
#%%% circular permutation
#circular permutation method
# def get_circperm_surrogates(timeseries):
# 	result = np.zeros([timeseries.size, timeseries.size])
# 	for i in range(timeseries.size):
# 		result[:,i] = np.roll(timeseries, i)
# 	return result

#%% Correlation statistic
#%%% mutual info
#mutual info statistic
# def mutual_info(x,y):
# 	return mutual_info_regression(y, x, n_neighbors=3).flatten()

#%%% local similarity from alex
#for alex's LSA from scratch
# def norm_transform(x):
#     return stats.norm.ppf((stats.rankdata(x))/(x.size+1.0))

# #alex's LSA from scratch
# def lsa_new(x,y_array):
#     # lsa with D=0
#     # x and y are time series
#     # returns: local similarity
#     n = x.size
#     x = np.copy(norm_transform(x))
#     score_P = np.zeros((y_array.shape[1]))
#     score_N = np.zeros((y_array.shape[1]))
    
#     for j in range(y_array.shape[1]):
#         y = np.copy(norm_transform(y_array[:,j]))
#         P = np.zeros(n+1)
#         N = np.zeros(n+1)
#         for i in range(x.size):
#             P[i+1] = np.max([0, P[i] + x[i] * y[i] ])
#             N[i+1] = np.max([0, N[i] - x[i] * y[i] ])
#         score_P[j] = np.max(P) / n
#         score_N[j] = np.max(N) / n
#     sign = np.sign(score_P - score_N)
#     return np.max([score_P, score_N], axis=0) * sign

# def lsa_new_delay(x,y_array,D=0):
#     # lsa with any D
#     # x and y are time series
#     # x is 1D, y is 2D (could be multiple time series)
#     # returns: local similarity
#     n = x.size # number of time points
#     x = np.copy(norm_transform(x)) # normalize x
#     score_P = np.zeros((y_array.shape[1])); # P-score for each X,Y pair
#     score_N = np.zeros((y_array.shape[1])); # N-score for each X,Y pair
    
#     for k in range(y_array.shape[1]):
#         y = np.copy(norm_transform(y_array[:,k])); # normalize y
#         P = np.zeros((n+1,n+1)) # 2D array
#         N = np.zeros((n+1,n+1))
#         for i in range(x.size):
#             for j in range(x.size):
#                 if np.abs(i - j) <= D:
#                     P[i+1][j+1] = np.max([0, P[i][j] + x[i] * y[j]])
#                     N[i+1][j+1] = np.max([0, N[i][j] - x[i] * y[j]])
#         score_P[k] = np.max(P) / n;
#         score_N[k] = np.max(N) / n;
#     sign = np.sign(score_P - score_N);
#     return np.max([score_P, score_N],axis=0); # non-negative values only

#%%% pearson correlation
#pearson correlation statistic
# def pcorr_strength(x,y):
#     M = np.zeros([x.size, y.shape[1] + 1])
#     M[:,0] = x
#     M[:,1:] = y
#     cov = np.cov(M.T)
#     cov_xy = cov[0,:]
#     std_y = np.sqrt(np.diag(cov))
#     std_x = std_y[0]
#     rho = cov_xy / (std_y * std_x)
#     return np.abs(rho[1:])

#%%% cross map skill ccm
#cross map skill statistic
# def ccm_statistic(x,y):
#     embed_dim, tau = ccm.choose_embed_params(x);
#     return np.array(ccm.ccm_loocv(np.vstack((x,y.T)).T, embed_dim, tau)['score']);

#%%% Granger causality
#granger causality statistic
#use aic to pick lag
# def granger_stat(x,y,pval=False):
#     t,N = y.shape;
#     if (N == 1):
#         data = np.vstack((x,y.T)).T;
#         model = VAR(data);
#         results = model.fit(maxlags=15,ic='aic');
#         maxlag=np.max([results.k_ar, 1])
#         #gc = results.test_causality(caused='y1',causing='y2');
#         #return np.array(gc.test_statistic).reshape(1);
#         if pval:
#             return granger(data, [maxlag], verbose=False)[maxlag][0]['ssr_ftest'][1];
#         else:
#             return np.array(granger(data, [maxlag], verbose=False)[maxlag][0]['ssr_ftest'][0]);
#     else:
#         result = np.zeros(N);
#         for i in range(N):
#             data = np.vstack((x,y[:,i].T)).T;
#             model = VAR(data);
#             res = model.fit(maxlags=15,ic='aic');
#             #gc = res.test_causality(caused='y1',causing='y2');
#             #result[i] = gc.test_statistic
#             maxlag = np.max([res.k_ar, 1])
#             if pval:
#                 result[i] = granger(data, [maxlag], verbose=False)[maxlag][0]['ssr_ftest'][1];
#             else:
#                 result[i] = granger(data, [maxlag], verbose=False)[maxlag][0]['ssr_ftest'][0];
#         return result;

#%% parametric test
#for parametric test
# def acorr_bj(x,acov=False):
#     # box-jenkins estimator for autocorrelation
#     # Eq 6 in Pyper and Peterman
#     result = np.zeros(x.size)
#     if acov:
#         denomenator = x.size;
#     else:
#         denomenator = np.sum((x - np.mean(x))**2)
#     for j in range(x.size):
#         numerator = 0
#         for t in range(x.size-j):
#             numerator += (x[t] - np.mean(x)) * (x[t + j] - np.mean(x))
#         result[j] = numerator / denomenator
#     return result

#for parametric test
# def sample_corr(x,y):
#     numerator = np.sum( (x - np.mean(x)) * (y - np.mean(y)) )
#     x_sum_squares = np.sum((x - np.mean(x))**2)
#     y_sum_squares = np.sum((y - np.mean(y))**2)
#     return numerator / np.sqrt(x_sum_squares * y_sum_squares)

#for parametric test
# def ttest(r,df):
#     t = np.abs(r) * np.sqrt(df) / np.sqrt(1 - r ** 2)
#     pval = 1-stats.t.cdf(t, df)
#     return t, 2*pval

#parametric test. use without surrogate data
# def acorr_test(x,y,pval_only=False):
#     # adusted t-test incorporating autocorrelation correction
#     x_acorr = acorr_bj(x)
#     y_acorr = acorr_bj(y)
#     idx = np.arange(1,x.size)
#     weights = (x.size - idx) / x.size
#     df_recip = (1 + 2 * np.sum( weights * x_acorr[1:] * y_acorr[1:])) / x.size # Eq 1 in Pyper and Peterman
#     df = 1/df_recip - 2
#     r = sample_corr(x,y)
#     t, p = ttest(r, df)
#     if (pval_only):
#         return p;
#     else:
#         return np.array([r, p])
    
#z-test that accounts for autocorrelation
# def fisher_acorr_test(x,y):
#     #autocorrelation is not truncated
#     x_acorr = acorr_bj(x)
#     y_acorr = acorr_bj(y)
#     idx = np.arange(1,x.size)
#     weights = (x.size - idx) / x.size
#     #take absolute value of correlation. we don't care about sign here
#     rho = sample_corr(x,y);
#     rho_var = (1 + 2 * np.sum( weights * x_acorr[1:] * y_acorr[1:])) / x.size # Afyouni et al eq 5
#     z_num = np.arctanh(rho);
#     z_denom = np.sqrt(rho_var * (1 - rho**2) ** -2)
#     #final fisherized z-score
#     z_score = z_num / z_denom;
#     #one-tailed test
#     #remember we took absolute value so the null dist should reflect this
#     pval = 2 * (1 - stats.norm.cdf(np.abs(z_score)));
#     return np.array([rho,pval]);
    
# #long run variance estimator
# #from Zhang 2019
# #uses a lag cutoff from andrews 1991
# def lrv(x,y):
#     N = x.size;
#     #elementwise product
#     z = x * y.reshape(-1);
#     resid = z - np.mean(z);
#     rhohat = np.sum(resid[1:]*resid[:-1]) / np.sum(resid[:-1]**2)
#     alphahat = 4*rhohat**2/(1-rhohat**2)**2
#     bw = int(np.ceil(1.1447*(alphahat*N)**(1/3)));
    
#     #autocovariance
#     x_acov = acorr_bj(x,acov=True).reshape(-1)
#     y_acov = acorr_bj(y,acov=True).reshape(-1)
#     z_acov = x_acov * y_acov;
#     idx = np.arange(1,bw+1);
#     weights = (bw - idx) / bw;
    
#     omega = z_acov[0] + 2*np.sum(weights * z_acov[1:bw+1])
#     return omega;

# #theoretical p value estimator for local similarity
# #from Xia 2013
# def lsa_theo(ls, N, var, D):
#     norm_ls = ls/np.sqrt(var*N);
#     if (norm_ls == 0):
#         tApprox_new = 1
#     else:
#         partial_sum = 0
#         tApprox_new = 0
#         i=1;
#         tApprox_diff = 1;
#         while (tApprox_diff > 1e-6):
#             A = norm_ls**2;
#             B = (2*i - 1)**2 * np.pi**2;
#             threshold = (1/A + 1/B)*np.exp(-B/(2*A));
#             partial_sum += threshold;
#             tApprox_old = tApprox_new;
#             tApprox_new = 1-8**(2*D+1)*partial_sum**(2*D+1);
#             tApprox_diff = np.abs(tApprox_new - tApprox_old);
#             i += 1;
#     return tApprox_new;
    
# #data-driven LSA
# #from Zhang 2019
# def dd_lsa(x,y,D=0):
#     N = x.size;
#     #zhang's code uses non-normalized LS so we correct here
#     sd = lsa_new_delay(x,y,D) * N;
#     #normalize x and y
#     x = np.copy(norm_transform(x));
#     y = np.copy(norm_transform(y)).reshape(-1,1);
#     #estimate long run variance
#     try:
#         xOmegay = lrv(x,y.reshape(-1));
#     except ValueError:
#         xOmegay = np.nan;
#     #stand-in for lrv
#     var = np.var(x)*np.var(y);
#     #then use theoretical approximation for pval
#     if (np.isnan(xOmegay)):
#         approximation = lsa_theo(sd, N, var, D)
#     else:
#         approximation = lsa_theo(sd, N, xOmegay, D)
#     return np.array([sd / N, approximation]).reshape(-1);

#%%% Protocols
#adds delays to any statistic
# def delay_scan(x,y,statistic,maxlag=5,kw_statistic={}):

#     score_sim = statistic(x,y,**kw_statistic);
#     # rows: # of y rows
#     # columns: # of lags to test, # test lags of (+-)2,4,6,8,10
#     score = np.zeros((score_sim.size,2*maxlag+1)); 
#     score[:,maxlag] = score_sim # middle column = no lag statistics
#     lags = np.arange(-2*maxlag,2*maxlag+1,2); 
#     if (maxlag > 0):
#         for i in np.arange(1,maxlag+1):
#             score[:,maxlag+i] = statistic(x[2*i:],y[:-2*i],**kw_statistic);
#             score[:,maxlag-i] = statistic(x[:-2*i],y[2*i:],**kw_statistic);
#     return np.vstack((lags,score));

# Everything calculated at the best lag
# def sig_test_good(x,y,statistic,surr_fxn,maxlag): 
#     #find statistic from original data
#     score = np.max(delay_scan(x,y.reshape(-1,1),statistic,maxlag)[1]); # no np.max in test_bad
#     #make surrogate data
#     surr = surr_fxn(y);
#     #truncate original x to match surrogate y if necessary
#     x = x[:surr.shape[0]]
#     #find null statistic for each surrogate 
#     #using same maximizing procedure as original
#     null = np.max(delay_scan(x,surr,statistic,maxlag)[1:],axis=1);
#     #one tailed test
#     pval = (np.sum(null >= score) + 1) / (np.size(null) + 1);
#     return pval;

# def sig_test_bad(x,y,statistic,surr_fxn,maxlag):
#     #find statistic from original data and lag
#     score_list = delay_scan(x,y.reshape(-1,1),statistic,maxlag); # in test_good we have np.max
#     [lag,score] = score_list[:,np.argmax(score_list[1])]
#     lag = int(lag);
#     #get surrogate data
#     surr = surr_fxn(y);
#     #truncate x if necessary
#     x = x[:surr.shape[0]]
#     #find correlation for shifted x and surrogates
#     if lag == 0:    
#         null = statistic(x,surr);
#     elif lag > 0:
#         null = statistic(x[lag:],surr[:-lag]);
#     else:
#         null = statistic(x[:lag],surr[-lag:]);
#     #one tailed test
#     pval = (np.sum(null >= score) + 1) / (np.size(null) + 1);
#     return pval;    

#Algorithm from Lancaster 2018
#Trim data so beginning matches end
# def trim_periodic_data(y,p=0):
#     if (p == 0):
#         #Find period of signal (very simple)
        
#         #subtract mean to get rid of 0 peak
#         y_clean = y - np.mean(y);
#         #get fourier transform
#         fft = np.fft.rfft(y_clean)
#         #first peak of fft
#         freq = np.argmax(np.abs(fft))
#         T = len(y)/freq/2
#         p = int(np.ceil(T/10))
    
#     #Now, run trimming algorithm
#     trunc_max = int(np.floor(len(y)/10));
#     match_mat = np.zeros((trunc_max,trunc_max))
    
#     #grid search for best k1 and k2
#     for k1 in np.arange(trunc_max):
#         for k2 in np.arange(len(y) - trunc_max - p + 1, len(y) - p + 1):
#             y_left = y[k1:k1+p]
#             y_right = y[k2:k2+p]
#             match_mat[k1,len(y)-p-k2] = np.sum((y_left - y_right)**2);
    
#     k_start, k_end = np.unravel_index(np.argmin(match_mat),match_mat.shape)
#     #change from index to actual k2
#     k_end = len(y) - p - k_end;
    
#     return (k_start,k_end);

#from alex
# def multishift_sigf(x, y, test_name, statistic, shift_grid, r_tts):
#     # returns pvalues obtained from shifts and bonferroni correction
#     if (2 * r_tts >= x.size) and (test_name == 'tts'):
#         return np.nan
#     pvals = np.zeros(shift_grid.size)
#     for i, shift in enumerate(shift_grid):
#         if shift >= 0:
#             x_shifted = x[shift:]
#             y_shifted = y[:y.size-shift]
#         else:
#             x_shifted = x[:x.size+shift]
#             y_shifted = y[-shift:]
#         pvals[i] = get_sigf(x_shifted, y_shifted, statistic,r_tts=r_tts, test_list=[test_name])[test_name]
#     return np.min(pvals) * shift_grid.size

# def r_multishift(N,maxlag,alpha=0.05):
#     m = 2*maxlag + 1;
#     # truncated time series should be at least 80 time points or else granger gets mad
#     r_options = np.arange(int(m/alpha) - 1, N/2 - 40, int(m/alpha)); 
#     if r_options.size == 0:
#         print("Time series too short for multishift with maxlag=%d" % (maxlag));
#     return int(np.max(r_options));

# #run suite of SURROGATE tests
# def get_sigf(x, y, statistic, test_list='all', maxlag=0, kw_randphase={}, kw_bstrap={}, kw_twin={}, r_tts=choose_r, r_naive=choose_r):
#     pvals = {};
#     sig_test = sig_test_good;
#     tts_test = tts;
#     if test_list == 'all':
#         test_list = ['randphase', 'bstrap', 'twin', 'tts', 'tts_naive', 'circperm','perm']
#     if type(r_tts) is type(1): # if manually pick r
#         r = r_tts
#         def r_tts(x):
#             return r
#     if type(r_naive) is type(1): # backup plan for r_tts?
#         r_ = r_naive
#         def r_naive(x):
#             return r_
#     k_start,k_end = trim_periodic_data(y);
#     xtrim = x[k_start:k_end+1]
#     ytrim = y[k_start:k_end+1]
#     if 'perm' in test_list:
#             pvals['perm'] = sig_test(x,y,statistic,get_perm_surrogates,maxlag);
# 	# get stationary bootstrap significance
#     if 'bstrap' in test_list:
#     	    pvals['bstrap'] = sig_test(x,y,statistic,get_stationary_bstrap_surrogates,maxlag);
# 	# get twin significance
#     if 'twin' in test_list:
#             pvals['twin'] = sig_test(x,y,statistic,get_twin_wrapper,maxlag);
# 	# get TTS significance
#     if 'tts' in test_list:
#             r = r_tts(x.size)
#             B = tts_test(x, y, r, statistic,maxlag);pvals['tts'] = B * (2 * r + 1) / (r + 1)
# 	# get naive TTS significance
#     if 'tts_naive' in test_list:
#             r = r_naive(x.size)
#             pvals['tts_naive'] = tts_test(x, y, r, statistic, maxlag)
#     #tts multishift
#     if 'tts_multishift' in test_list:
#         r = r_multishift(x.size, maxlag);
#         lag_grid = np.arange(-maxlag,maxlag+1,1);
#         pvals['tts_multishift'] = multishift_sigf(x,y,'tts', statistic, lag_grid, r);
# 	# get circular permutation significance
#     if 'randphase' in test_list:
#             #preprocessing step
#             pvals['randphase'] = sig_test(xtrim,ytrim,statistic,get_iaaft_surr,maxlag);
#     if 'circperm' in test_list:
#             #preprocessing step
#             pvals['circperm'] = sig_test(xtrim,ytrim,statistic,get_circperm_surrogates,maxlag);
#     return pvals

# # Run stats with surrogates
# def benchmark_stats(x,y,test_list='all',maxlag=5):
#     pvals_granger = get_sigf(x,y,granger_stat,test_list=test_list,maxlag=maxlag);
#     pvals_granger_rev = get_sigf(y,x,granger_stat,test_list=test_list,maxlag=maxlag);
#     pvals_lsa = get_sigf(x,y,lsa_new_delay,test_list=test_list,maxlag=maxlag);
#     pvals_pcorr = get_sigf(x,y,pcorr_strength,test_list=test_list,maxlag=maxlag);
#     pvals_ccm = get_sigf(x,y,ccm_statistic,test_list=test_list,maxlag=maxlag);
#     pvals_ccm_rev = get_sigf(y,x,ccm_statistic,test_list=test_list,maxlag=maxlag);
#     pvals_mutual_info = get_sigf(x,y,mutual_info,test_list=test_list,maxlag=maxlag);
#     pval_parametric = {"pearson":(2*maxlag+1)*np.min(delay_scan(x,y,fisher_acorr_test,maxlag=maxlag)[2])};
#     pval_granger_nosurr = {"granger_nosurr":(2*maxlag+1)*np.min(delay_scan(x,y.reshape(-1,1), \
#                         granger_stat,maxlag=maxlag,kw_statistic={'pval':True})[1])};
#     pval_granger_nosurr_rev = {"granger_nosurr":(2*maxlag+1)*np.min(delay_scan(y,x.reshape(-1,1), \
#                         granger_stat,maxlag=maxlag,kw_statistic={'pval':True})[1])};
#     pvals_lsa_dd = {"lsa":(2*maxlag+1)*np.min(delay_scan(x,y.reshape(-1,1),dd_lsa,maxlag=maxlag)[2])};

#     return {'granger_y->x':pvals_granger, \
#             'granger_x->y':pvals_granger_rev, \
#             'lsa':pvals_lsa, \
#             'pearson':pvals_pcorr, \
#             'ccm_y->x':pvals_ccm, \
#             'ccm_x->y':pvals_ccm_rev, \
#             'mutual_info':pvals_mutual_info, \
#             'pcc_param':pval_parametric, \
#             'granger_param_y->x':pval_granger_nosurr, \
#             'granger_param_x->y':pval_granger_nosurr_rev, \
#             'lsa_data_driven':pvals_lsa_dd};

