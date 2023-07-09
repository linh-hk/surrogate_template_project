#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 11:03:15 2023

@author: h_k_linh

This is an old workflow that runs very slow and not optimised.
Most of this has been transcend to surrogate_dependence_test.py (Surrogate tests and wrapping around the workflow)
The surrogate_dependence_test.py is written when I had better understanding of the workflow with integration of multiprocessing.
It will be using the lsa_new_delay_linh, correlation_Pearson and mutual_info from this script

"""
#%% Set working directory
import os
print(f'working directory: {os.getcwd()}')
# os.chdir('/home/h_k_linh/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test')
# os.chdir('C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/')
# os.getcwd()
#%% Import libraries
# import GenerateData as dataGen
import numpy as np
from scipy import stats # t-distribution
# from scipy.spatial import distance_matrix # for twin
# from pyunicorn.timeseries import surrogates #for iaaft
# import ccm # for ccm
# from statsmodels.tsa.stattools import grangercausalitytests as granger # for granger
# from statsmodels.tsa.api import VAR # vector autoregression for granger
from sklearn.feature_selection import mutual_info_regression # for mutual information

# import dill # for saving/loading data
# import pickle # for saving/loading data
# import matplotlib.pyplot as plt

# import time
# import sys # Import sys.argv to save better results
#%% Parametric test
# Box-Jenkins estimator for autocorrelation
# Equation 6 in Pyper and Peterman , look in notion
def autocorrelationcorr_BJ(x,acov=False):
    result = np.zeros(x.size)
    mean = np.mean(x)
    if acov:
        denomenator = x.size;
    else:
        denomenator = np.sum((x - mean)**2)
    # test different distance j from x[t]. the smaller j, the higher result.
    for j in range(x.size):
        numerator = 0
        for t in range(x.size-j):
            numerator += (x[t] - mean) * (x[t + j] - mean)
        result[j] = numerator / denomenator
    return result

# crosscorrelation
def crosscorrelation(x,y):
    numerator = np.sum( (x - np.mean(x)) * (y - np.mean(y)) )
    x_sum_squares = np.sum((x - np.mean(x))**2)
    y_sum_squares = np.sum((y - np.mean(y))**2)
    return numerator / np.sqrt(x_sum_squares * y_sum_squares)

    #%%% t-distribution
# calculate t-value from ttest, small fxn for parametric test
def p_from_ttest(r,df): 
    print("\t\t\t\t\t\tCalculating p-value from t distribution")
    # r = crosscorrelation stats, df = degree of freedom
    t = np.abs(r) * np.sqrt(df) / np.sqrt(1 - r ** 2)
    pval = 1-stats.t.cdf(t, df)
    return t, 2*pval

# the parametric test
def acorr_test_1(x,y,pval_only=False):
    # adusted t-test incorporating autocorrelation correction
    x_acorr = autocorrelationcorr_BJ(x)
    y_acorr = autocorrelationcorr_BJ(y)
    idx = np.arange(1,x.size) #j
    weights = (x.size - idx) / x.size # (N-j)/N according to equation
    df_recip = (1 + 2 * np.sum( weights * x_acorr[1:] * y_acorr[1:])) / x.size 
    # Degree of freedom: Eq 1 in Pyper and Peterman, Afyouni eq 5
    df = 1/df_recip - 2
    r = crosscorrelation(x, y)
    t, p = p_from_ttest(r, df) # derive pvalue of crosscorelation from autocorrelation's t distribution
    if (pval_only):
        return p;
    else:
        return np.array([r, p])
    #%%% normal distribution with modified degree of freedom
#Fisher transformation of Pearson corr to z(normal) distribution # Afyouni et al eq 5
def acorr_test_fisher_z(x,y):
    print("\t\t\t\t\t\tCalculating p-value from z distribution of Fisher transformed Pearson correlation coefficient (parametric test)")
    #autocorrelation is not truncated
    x_acorr = autocorrelationcorr_BJ(x)
    y_acorr = autocorrelationcorr_BJ(y)
    idx = np.arange(1,x.size)
    weights = (x.size - idx) / x.size
    #take absolute value of correlation. we don't care about sign here
    rho = crosscorrelation(x, y);
    rho_var = (1 + 2 * np.sum( weights * x_acorr[1:] * y_acorr[1:])) / x.size # Afyouni et al eq 5
    z_num = np.arctanh(rho);
    z_denom = np.sqrt(rho_var * (1 - rho**2) ** -2)
    #final fisherized z-score
    z_score = z_num / z_denom;
    #one-tailed test
    #remember we took absolute value so the null dist should reflect this
    pval = 2 * (1 - stats.norm.cdf(np.abs(z_score)));
    return np.array([rho,pval]);
    #%%% Data-driven Local similarity
#long run variance estimator
#from Zhang 2019
#uses a lag cutoff from andrews 1991
def lrv(x,y):
    N = x.size;
    #elementwise product
    z = x * y.reshape(-1);
    resid = z - np.mean(z);
    rhohat = np.sum(resid[1:]*resid[:-1]) / np.sum(resid[:-1]**2)
    alphahat = 4*rhohat**2/(1-rhohat**2)**2
    bw = int(np.ceil(1.1447*(alphahat*N)**(1/3)));
    
    #autocovariance
    x_acov = autocorrelationcorr_BJ(x,acov=True).reshape(-1)
    y_acov = autocorrelationcorr_BJ(y,acov=True).reshape(-1)
    z_acov = x_acov * y_acov;
    idx = np.arange(1,bw+1);
    weights = (bw - idx) / bw;
    
    omega = z_acov[0] + 2*np.sum(weights * z_acov[1:bw+1])
    return omega;

#theoretical p value estimator for local similarity
#from Xia 2013
def lsa_theo(ls, N, var, D):
    norm_ls = ls/np.sqrt(var*N);
    if (norm_ls == 0):
        tApprox_new = 1
    else:
        partial_sum = 0
        tApprox_new = 0
        i=1;
        tApprox_diff = 1;
        while (tApprox_diff > 1e-6):
            A = norm_ls**2;
            B = (2*i - 1)**2 * np.pi**2;
            threshold = (1/A + 1/B)*np.exp(-B/(2*A));
            partial_sum += threshold;
            tApprox_old = tApprox_new;
            tApprox_new = 1-8**(2*D+1)*partial_sum**(2*D+1);
            tApprox_diff = np.abs(tApprox_new - tApprox_old);
            i += 1;
    return tApprox_new;
    
#data-driven LSA
#from Zhang 2019
def dd_lsa(x,y,D=0):
    print("\t\t\t\t\t\tCalculating p-value from data-driven local similarity (parametric) test")
    N = x.size;
    #zhang's code uses non-normalized LS so we correct here
    sd = lsa_new_delay(x,y,D) * N;
    #normalize x and y
    x = np.copy(norm_transform(x));
    y = np.copy(norm_transform(y)).reshape(-1,1);
    #estimate long run variance
    try:
        xOmegay = lrv(x,y.reshape(-1));
    except ValueError:
        xOmegay = np.nan;
    #stand-in for lrv
    var = np.var(x)*np.var(y);
    #then use theoretical approximation for pval
    if (np.isnan(xOmegay)):
        approximation = lsa_theo(sd, N, var, D)
    else:
        approximation = lsa_theo(sd, N, xOmegay, D)
    return np.array([sd / N, approximation]).reshape(-1);

#%% Non-parametric
"""
The way to approach this coding scripts:
    Linh's old thinking: 
        Write dgen_fxntion for correlation statistics first.
        Then write function for surrogates of y
        Then run for loop to caculate correlation of x with each surrogates of y.
        On the loop, append the stats to a matrix.
    Instead, Caroline has better ideas:
        Write surrogates to generate all needed data at once:
            Put y and all surrogates of y into a matrix (y.size, number.of.surrogates)
            # columns of surrogates of y
        Write correlation statistics to calculate correlation between:
            x series 
            with each column of surrogate matrix.
    So Linh will follow Caroline and learn to think in matrix
    After readings, I think the scripts could have been written in the other way:
        Because the tts fxn also calculate the statistics, we can still 
            write the correlation stats fxn first (for matrix surrY of course)
            then write surrogates fxn with integrated calculation of correlation stats
            this would lessen down the need for get_sigf fxn, we can go straight to benchmark stats
        Or, we could:
            Separate the tts function to return the surrY instead of sig.
            And adjust the get_sigf fxn.
        The good point of Caroline's structure:
            We can keep track of the protocol. The whole process is summarised in a nice fxn.
"""

# #%%% Surrogates
#     #%%%% Random shuffle
# def get_perm_surrogates(timeseries, n_surr=99):
#     print("\t\t\t\tpermutations")
#     sz = timeseries.size;
#     result = np.zeros([sz,n_surr]); # Create mentioned matrix of surrogates
#     for col in range(n_surr):
#         result[:,col] = np.random.choice(timeseries,size=sz,replace=False);
#     return result;
#     #%%%% Stationary bootstrap (block boostraps)
#     """
#     Randomly select consecutive blocks of data to construct surrogate sample
#     """
# def get_stationary_bstrap_surrogates(timeseries, p_jump=0.05, n_surr=99):
#     print("\t\t\t\tstationary bootstrap")
#     # According to Caroline p_jump = alpha
#     sz = timeseries.size
    
#     # Create surrY matrix but first use it to define index of surrYs 
#     result = np.zeros([sz, n_surr]);
    
#     # Define indices of all surrY_0, replace = T => repeated values 
#     # According to Caroline, choose a random interger T(1) in [1,N];
#     result[0,:] = np.random.choice(sz, size=n_surr, replace=True)    
#     for col in range(n_surr):
#         for row in range(1,sz):
#             # with probability alpha we pick new T(k) => new block
#             if np.random.random() < p_jump:
#                 result[row,col] = np.random.choice(sz)
#             # with probability 1-alpha T(k)=(T(k-1)+1) mod N # => size of block
#             else: 
#                 result[row,col] = (result[row-1,col] + 1) % sz
#             # = > total random blocks with total random size
#     # after having full index matrices, take values
#     for col in range(n_surr):
#         for row in range(sz):
#             result[row,col] = timeseries[int(result[row,col])]
#     return result
#     #%%%% IAAFT Iterative Amplitude-adjusted Fourier transform (random phase)
#     """
#     Works for stationary linear Gaussian process.
#     Wiener-Khinchin
#     Steps:
#         Decompose Y into sine waves = Fourier Transform
#         Shift sine wave's phases
#         Add shifted waves togethers
#     AAFT:
#         for x(t) = (a(t))^3 with a(t) stationary linear while x(t) is nonlinear, 
#         x(t) can be made linear = invertible, time dependent scaling function
#     IAAFT:
#         scaling and rescaling after phase shift might alter the amplitude 
#         (power spectrum) of surrY so we iteratively adjust the amplitudes of 
#         scaled data.
#     """
# def get_iaaft_surrogates(timeseries, n_surr=99, n_iter=200):
#     print("\t\t\t\trandom phase iaaft")
#     # prebuilt algorithm in pyunicorn so dont have to create empty matrix
#     results = []
#     i = 0
#     while i < n_surr:
#         obj = surrogates.Surrogates(original_data=timeseries.reshape(1,-1), silence_level=2)
#         surr = obj.refined_AAFT_surrogates(original_data=timeseries.reshape(1,-1), n_iterations=n_iter, output='true_spectrum')
#         surr = surr.ravel()
#         if not np.isnan(np.sum(surr)):
#             results.append(surr)
#             i += 1
#     return np.array(results).T

#     #%%%% Circular permutation (time shift)
# def get_circperm_surrogates(timeseries):
#     print("\t\t\t\tcircular permutations")
#     result = np.zeros([timeseries.size, timeseries.size])
#     for i in range(timeseries.size):
#         result[:,i] = np.roll(timeseries, i)
#     return result
#     #%%%% Trimming to remove discontinuity then run with IAAFT and Circular permutation
#     """
#     Fourier transform and circular TS can fail if there is discontinuity between 
#     beginning and end of timeseries.
#     => Trimming method from Lancaster 2018
#     estimate period of timeseries T with fast Fourier transform and set parameter 
#     p = T/10
#     """
# def trim_periodic_data(y,p=0):
#     print("\t\t\t\ttrimming data to remove discontinuity")
#     if (p == 0):
#         #Find period of signal (very simple)
        
#         #subtract mean to get rid of 0 peak
#         y_clean = y - np.mean(y);
#         #get fourier transform
#         fft = np.fft.rfft(y_clean)
#         #first peak of fft
#         freq = np.argmax(np.abs(fft)) # Nyquist frequency f = N/2T
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
#     #%%%% Twin method
#     """
#     Embed data into a delay space:
#         Neighbors of a focal point = points within predefined distance.
#         Two points are twins if they have same neighbors.
#     Steps:
#         choose random starting points
#         if it has no twins, following point = its following points in normal space
#         if it has twins in delay space, following point = following point of twins in normal space 
#     Works with N0 = two time series are independent and Markovian.
#     Parameters:
#         - Embedding dimension and delay chose from grid serach and pick the 
#         values that give highest cross map skill of y to itself
#         - Neighbor distance chosen so that on average, each point in delay space
#         has specified number of neighbors (10% of total)
#     """
# def create_embed_space(data, embed_dim, tau, pred_lag=0):
#     """
#     Create embed space, used in  choose_twin_threshold
#     Prepares a "standard matrix problem" from the time series data

#     Args:
#         data (array): Created in choose_twin_threshold, data = np.tile(timeseries, (2,1)).T:
#             A 2d array with two columns where the first column
#             will be used to generate features and the second column will
#             be used to generate the response variable
#         embed_dim (int): embedding dimension (delay vector length) , calculated from ccm
#         tau (int): delay between values
#         pred_lag (int): prediction lag
#     """
#     # print("Twin embedding")
#     x = data[:,0]
#     y = data[:,1]
#     feat = []
#     resp = []
#     idx_template = np.array([pred_lag] + [-i*tau for i in range(embed_dim)])
#     for i in range(x.size):
#         if np.min(idx_template + i) >= 0 and np.max(idx_template + i) < x.size: # make sure indices is not out of bound
#             feat.append(x[idx_template[1:] + i])
#             resp.append(y[idx_template[0] + i])
#             # length = original length - tau *(embed_dim -1)
#     return np.array(feat), np.array(resp)

# # calculate distance matrices in embed spaces, 
# # return the distance matrix of the embed space with the largest distances
# def max_distance_matrix(X, Y):
#     """
#     returns max norm distance matrix

#     Args:
#         X (array): an m-by-n array where m is the number of vectors and n is the
#             vector length
#         Y (array): same shape as X
#     """
#     # print("Generating twin distance matrix")
#     n_vecs, n_dims = X.shape # original length -tau*(embed_dim-1); embed_dim
#     K_by_dim = np.zeros([n_dims, n_vecs, n_vecs]) 
#     # embed_dim matrices of disctance between each embed dim. dim1xdim1, dim2xdim2,...
#     for dim in range(n_dims):
#         K_by_dim[dim,:,:] = distance_matrix(X[:,dim].reshape(-1,1), Y[:,dim].reshape(-1,1))
#     return K_by_dim.max(axis=0) # out of the 5 distance matrices - embed_dim =5

# # predefined distance to define twins
# # defined so that on average each point in delay space has specified # of neighbors
# # (12% total points, later set to 10 % in get_twin_surrogates)
# def choose_twin_threshold(timeseries, embed_dim, tau, neighbor_frequency=0.12, distmat_fxn=max_distance_matrix):
#     """Given a univariate timeseries, embedding parameters, and a twin frequency,
#         choose the twin threshold.

#        Args:
#          timeseries (numpy array): a univariate time series
#          embed_dim (int): embedding dimension
#          tau (int): embedding delay
#          neighbor_frequency (float): Fraction of the "recurrence plot" to choose
#              as neighbors. Note that not all neighbors are twins.

#         Returns:
#           recurrence distance threshold for twins
#     """
#     # print("Choosing twin thresholds")
#     # timeseries is 1d
#     timeseries = np.copy(timeseries.flatten())
#     data_ = np.zeros([timeseries.size, 2])
#     data_[:,0] = timeseries
#     data_[:,1] = timeseries
#     # or data_ = np.tile(timeseries, (2,1)).T
#     X, y = create_embed_space(data_, embed_dim=embed_dim, tau=tau) # feat, resp
#     K = distmat_fxn(X,X) # max_distance_matrix, time series length = 400, embed = 5 , tau = 2 then K.shape = (392,392)
#     #np.fill_diagonal(K, np.inf) # self-neighbors are allowed in recurrence plot.
#     k = K.flatten()
#     k = np.sort(k) # sort all distances
#     idx = np.floor(k.size * neighbor_frequency).astype(int)
#     # print(f"Twin threshold for distance is {k[idx]}")
#     return k[idx]


# #twin method
# def get_twin_surrogates(timeseries, embed_dim, tau, num_surr=99,
#                         neighbor_frequency=0.1, th=None):
#     if th is None:
#         th = choose_twin_threshold(timeseries, embed_dim, tau, neighbor_frequency)
#     results = [] # i=0
#     obj = surrogates.Surrogates(original_data=timeseries.reshape(1,-1), silence_level=2)
#     for i in range(num_surr):
#         surr = obj.twin_surrogates(original_data=timeseries.reshape(1,-1), dimension=embed_dim, delay=tau, threshold=th, min_dist=1)
#         surr = surr.ravel()
#         results.append(surr)
#     return np.array(results).T

# def get_twin_wrapper(timeseries,num_surr=99):
#     print("\t\t\t\ttwin")
#     embed_dim, tau = ccm.choose_embed_params(timeseries);
#     surrs = get_twin_surrogates(timeseries,embed_dim,tau,num_surr);
#     return surrs;
#     #%%%% Truncated time shift (not affected by discontinuity)
#     """
#     Only take subsets of original time series.
#     The series for surrogates must be stationary
#     Needs truncation radius r: x_trunc = x_{1+r},...,x_{n-r}
#     Needs delta (-r<=delta<=r): y_trunc = y_{1+r+delta},...,y_{n-r+delta}
#     """
# # One of the options that is not r = B/u - 1
# def choose_r(n):
#     delta = (n/4 + 1) % 20
#     return int(n/4 - delta)

# # For protocol later
# def r_multishift(N,maxlag,alpha=0.05):
#     print("\t\t\t\t tts_multishift, shifting r")
#     m = 2*maxlag + 1;
#     # truncated time series should be at least 80 time points or else granger gets mad
#     r_options = np.arange(int(m/alpha) - 1, N/2 - 40, int(m/alpha)); 
#     # if r_options.size == 0:
#         # print("Time series too short for multishift with maxlag=%d" % (maxlag));
#     return int(np.max(r_options));

# # For protocols later
# def multishift_sigf(x, y, test_name, statistic, shift_grid, r_tts):
#     # returns pvalues obtained from shifts and bonferroni correction
#     print("\t\t\t\tRunning tts_multishift, returns pvalues obtained from shifts and bonderroni correction")
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

# # Compared to other surrogates fxn, this tts already calculate the statistics
# def tts(x,y,r,statistic,maxlag=0):
#     # time shift
#     t = len(x); #number of time points in orig data
#     xstar = x[r:(t-r)]; # middle half of data - truncated original X0
#     tstar = len(xstar); # length of truncated series
#     ystar = y[r:(t-r)]; # truncated original Y0
#     t0 = r;             # index of Y0 in matrix of surrY
    
#     y_surr = np.tile(ystar,[2*r+1, 1]) # Create empty matrix of surrY
#     print("\t\t\t\ttts")
#     #iterate through all shifts from -r to r
#     for shift in np.arange(t0 - r, t0 + r):
#         # print(f'shift-t0:{shift-t0}\ty[shift:(shift+tstar)]: {shift}:{shift+tstar}')
#         y_surr[shift-t0] = y[shift:(shift+tstar)];
#     print(f"\t\t\t\tCalculating {statistic.__name__} for tts surrogates")
#     importance_true = np.max(scan_lags(xstar, ystar.reshape(-1,1),statistic,maxlag)[1]); # scan_lags returns np.vstack((lags,score))
#     importance_surr = np.max(scan_lags(xstar, y_surr.T, statistic,maxlag)[1:],axis=1);
#     sig = np.mean(importance_surr > importance_true, axis=0);
#     return sig;

# # This was in fact not used at all in Caroline's scripts ...
# # It is a 'bad' version of the tts fxn ...
# def tts_bad(x,y,r,statistic,maxlag=0):
#     # time shift
#     t = len(x); #number of time points in orig data
#     xstar = x[r:(t-r)]; # middle half of data
#     tstar = len(xstar); #length of middle half of data
#     ystar = y[r:(t-r)];
#     t0 = r;
    
#     y_surr = np.tile(ystar,[2*r+1, 1])
#     #iterate through all shifts from -r to r
#     for shift in np.arange(t0 - r, t0 + r):
        
#         y_surr[shift-t0] = y[shift:(shift+tstar)];
    
#     score_list = scan_lags(xstar, ystar.reshape(-1,1),statistic,maxlag);
#     [lag,importance_true] = score_list[:,np.argmax(score_list[1])]
#     lag = int(lag)
#     if lag == 0:
#         importance_surr = statistic(xstar, y_surr.T)
#     elif lag > 0:
#         importance_surr = statistic(xstar[lag:],(y_surr.T)[:-lag])
#     else:
#         importance_surr = statistic(xstar[:lag],(y_surr.T)[-lag:])
#     sig = np.mean(importance_surr > importance_true, axis=0);
#     return sig;
#%%% Correlation Statistics
    #%%%% Pearson correlations
def correlation_Pearson(x, y):
    print("\t\t\t\t\t\tpearson correlation coefficients")
    M = np.zeros([x.size, y.shape[1] + 1])
    M[:,0] = x
    M[:,1:] = y
    cov = np.cov(M.T)
    cov_xy = cov[0,:]
    std_y = np.sqrt(np.diag(cov))
    std_x = std_y[0]
    rho = cov_xy / (std_y * std_x)
    return np.abs(rho[1:])

    #%%%% Local similarity analysis
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
def norm_transform(x):
    return stats.norm.ppf((stats.rankdata(x))/(x.size+1.0))

#alex's LSA from scratch
def lsa_new(x,y_array):
    # print("\t\t\t\t\t\tlocal similarity analysis")
    # lsa with Delay=0 
    # x and y are time series
    # returns: local similarity
    n = x.size
    x = np.copy(norm_transform(x))
    score_P = np.zeros((y_array.shape[1]))
    score_N = np.zeros((y_array.shape[1]))
    
    for j in range(y_array.shape[1]): # for each columns in y
        y = np.copy(norm_transform(y_array[:,j])) # norm_transform the jth column of y
        P = np.zeros(n+1) # Initialise pos association
        N = np.zeros(n+1) # Initialise neg association
        for i in range(x.size): # for each x and y elements do
            P[i+1] = np.max([0, P[i] + x[i] * y[i] ])
            N[i+1] = np.max([0, N[i] - x[i] * y[i] ])
        score_P[j] = np.max(P) / n
        score_N[j] = np.max(N) / n
    sign = np.sign(score_P - score_N)
    return np.max([score_P, score_N], axis=0) * sign

def lsa_new_delay(x,y_array,D=0):
    # print("\t\t\t\tCalculating local similarity with lag")
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

# Linh modified
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
    print("\t\t\t\t\t\tlocal similarity analysis")
    # lsa with Delay=0 
    # x and y are time series
    # returns: local similarity
    # n = x.size
    # y_array should be (replicates,timepoints)
    
    x_norm = norm_transform(x)
    
    y_norm = np.apply_along_axis(norm_transform, 1, y_array)
    xy = x_norm*y_norm
    
    score_P, score_N = np.apply_along_axis(lsa_main,1,xy).T
    sign = np.sign(score_P - score_N)
    return np.max([score_P, score_N], axis=0) * sign

def lsa_new_delay_linh(x,y_arr,D=3):
    print("\t\t\t\t\t\tlocal similarity analysis")
    # y_array should be (replicates,timepoints)
    n = x.size
    y_array = y_arr.T
    x_norm = norm_transform(x)
    
    y_norm = np.apply_along_axis(norm_transform, 1, y_array)
    
    xy = []
    xy.append(x_norm * y_norm)
    for i in range(1,D+1):
        xy.append(x_norm[i:]*y_norm[:,:-i])
        # print(i,x.size,0, x.size-i)
        xy.append(x_norm[:-i]*y_norm[:,i:])
        # print(0,x.size-1,i,x.size)
    
    score_PN=[np.apply_along_axis(lsa_main,1,_) for _ in xy]
    score_P = np.divide([np.max([np.max(score_PN[i][_]['P']) 
                                 for i in range(len(score_PN))]) 
                         for _ in range(len(score_PN[0]))],n)
    score_N = np.divide([np.max([np.max(score_PN[i][_]['N']) 
                                 for i in range(len(score_PN))]) 
                         for _ in range(len(score_PN[0]))],n)
    return np.max([score_P,score_N], axis = 0);

    #%%%% Granger causality
# def granger_stat(x,y,pval=False):
#     print("\t\t\t\t\t\tGranger")
#     t,N = y.shape; # y is surrY matrix
#     if (N == 1):
#         data = np.vstack((x,y.T)).T;
#         model = VAR(data); # Vector autoregression
#         results = model.fit(maxlags=15,ic='aic'); # ??VAR
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

    #%%%% Convergent cross mapping 
# def ccm_statistic(x,y):
#     print("\t\t\t\t\t\tconvergent cross mapping")
#     embed_dim, tau = ccm.choose_embed_params(x);
#     return np.array(ccm.ccm_loocv(np.vstack((x,y.T)).T, embed_dim, tau)['score']);

    #%%%% Mutual information
    """
    The Kullback-Leibler divergence - measures the difference in two probability distributions
    Here, the 2 probability distributions are
        * that of P(X,Y) - joint distribution of X - P1
        * that of P(X)*P(Y) - marginal distribution of X & Y - P(2)
    (how: the expectation (calculated by P(x) instead of weighted means) of logarithmic difference between the probabilities P1 and P2)
    if P(X,Y) == P(X)*P(Y) <=> X,Y no relation <=> P1/P2 = 1
    log( 1 ) = 0 => X and Y are independent.    
    """
def mutual_info(x,y):
    print("\t\t\t\t\t\tmutual information")
    return mutual_info_regression(y, x, n_neighbors=3).flatten()
# #%% Test protocols
#     #%%%% Tests at several lags/ detect lags/scan lags
# def scan_lags(x,y,statistic,maxlag=5,kw_statistic={}):
#     print("\t\t\t\t\tScanning best lags with step lags = 2")

#     score_sim = statistic(x,y,**kw_statistic);
#     # rows: # of y rows
#     # columns: # of lags to test, # test lags of (+-)2,4,6,8,10
#     score = np.zeros((score_sim.size,2*maxlag+1)); 
#     score[:,maxlag] = score_sim # middle column = no lag statistics
#     lags = np.arange(-2*maxlag,2*maxlag+1,2); 
#     if (maxlag > 0):
#         for i in np.arange(1,maxlag+1):
#             print(f"\t\t\t\t\t\tlag: x_{2*i} and y_{-2*i}")
#             score[:,maxlag+i] = statistic(x[2*i:],y[:-2*i],**kw_statistic);
#             print(f"\t\t\t\t\t\tlag: x_{-2*i} and y_{2*i}")
#             score[:,maxlag-i] = statistic(x[:-2*i],y[2*i:],**kw_statistic);
#     return np.vstack((lags,score));

#     #%%%% Calibrate x length, find null stats, calculate p-value
# def sig_test_good(x,y,statistic,surr_fxn,maxlag): 
#     print("\t\t\tCalculating p-value")
#     #find statistic from original data
#     score = np.max(scan_lags(x,y.reshape(-1,1),statistic,maxlag)[1]); # no np.max in test_bad
    
#     #get surrogate data
#     surr = surr_fxn(y);
#     #truncate original x to match surrogate y if necessary
#     x = x[:surr.shape[0]]
    
#     #find null statistic for each surrogate 
#     #using same maximizing procedure as original
#     null = np.max(scan_lags(x,surr,statistic,maxlag)[1:],axis=1);
    
#     #one tailed test
#     pval = (np.sum(null >= score) + 1) / (np.size(null) + 1);
#     return pval;

# def sig_test_bad(x,y,statistic,surr_fxn,maxlag):
#     #find statistic from original data and lag
#     score_list = scan_lags(x,y.reshape(-1,1),statistic,maxlag); # in test_good we have np.max
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

# # %%% Which surr
# def get_sigf(x, y, statistic, test_list='all', maxlag=0, kw_randphase={}, kw_bstrap={}, kw_twin={}, r_tts=choose_r, r_naive=choose_r):
#     pvals = {};
#     runtime = {};
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
        
#     print(f"\tStart calculating statistics for significant value: \n\t\t{test_list}")
    
#     k_start,k_end = trim_periodic_data(y);
#     xtrim = x[k_start:k_end+1]
#     ytrim = y[k_start:k_end+1]
#     if 'perm' in test_list:
#             start = time.time()
#             pvals['perm'] = sig_test(x,y,statistic,get_perm_surrogates,maxlag);
#             runtime['perm'] = time.time() - start
#     # get stationary bootstrap significance
#     if 'bstrap' in test_list:
#             start = time.time()
#             pvals['bstrap'] = sig_test(x,y,statistic,get_stationary_bstrap_surrogates,maxlag);
#             runtime['bstrap'] = time.time() - start
#     # get twin significance
#     if 'twin' in test_list:
#             start = time.time()
#             pvals['twin'] = sig_test(x,y,statistic,get_twin_wrapper,maxlag);
#             runtime['twin'] = time.time() - start
#     # get TTS significance
#     if 'tts' in test_list:
#             start = time.time()
#             r = r_tts(x.size)
#             B = tts_test(x, y, r, statistic,maxlag);
#             pvals['tts'] = B * (2 * r + 1) / (r + 1)
#             runtime['tts'] = time.time() - start
#     # get naive TTS significance
#     if 'tts_naive' in test_list:
#             start = time.time()
#             r = r_naive(x.size)
#             pvals['tts_naive'] = tts_test(x, y, r, statistic, maxlag)
#             runtime['tts_naive'] = time.time() - start
#     #tts multishift
#     if 'tts_multishift' in test_list:
#         start = time.time()
#         r = r_multishift(x.size, maxlag);
#         lag_grid = np.arange(-maxlag,maxlag+1,1);
#         pvals['tts_multishift'] = multishift_sigf(x,y,'tts', statistic, lag_grid, r);
#         runtime['tts_multishift'] = time.time() - start
#     # get circular permutation significance
#     if 'randphase' in test_list:
#             #preprocessing step
#             start = time.time()
#             pvals['randphase'] = sig_test(xtrim,ytrim,statistic,get_iaaft_surrogates,maxlag);
#             runtime['randphase'] = time.time() - start
#     if 'circperm' in test_list:
#             #preprocessing step
#             start = time.time()
#             pvals['circperm'] = sig_test(xtrim,ytrim,statistic,get_circperm_surrogates,maxlag);
#             runtime['circperm'] = time.time() - start
#     return pvals , runtime

#     #%%% Which statistic
# # Run stats with surrogates
# def benchmark_stats(x,y,test_list='all',stats_list='all', maxlag=5, r_tts=choose_r, r_naive=choose_r):
#     pvals = {};
#     runtime = {};
#     if stats_list == 'all':
#         stats_list = ['pearson', 'lsa', 'mutual_info', 
#                       'granger_y->x', 'granger_x->y', 
#                       'ccm_y->x', 'ccm_x->y',
#                       'pcc_param', 'granger_param_y->x', 'granger_param_x->y', 'lsa_data_driven']
#     print("Start getting significant values");
#     if 'pearson' in stats_list:
#         pvals['pearson'], runtime['pearson'] = get_sigf(x,y,correlation_Pearson,test_list=test_list,maxlag=maxlag, r_tts=r_tts, r_naive=r_naive);
#     if 'lsa' in stats_list:
#         # pvals['lsa'], runtime['lsa'] = get_sigf(x,y,lsa_new_delay,test_list=test_list,maxlag=maxlag, r_tts=r_tts, r_naive=r_naive);
#         pvals['lsa'], runtime['lsa'] = get_sigf(x,y,lsa_new_delay_linh,test_list=test_list,maxlag=maxlag, r_tts=r_tts, r_naive=r_naive);
#     if 'mutual_info' in stats_list:
#         pvals['mutual_info'], runtime['mutual_info'] = get_sigf(x,y,mutual_info,test_list=test_list,maxlag=maxlag, r_tts=r_tts, r_naive=r_naive);
#     if 'granger_y->x' in stats_list:
#         pvals['granger_y->x'], runtime['granger_y->x'] = get_sigf(x,y,granger_stat,test_list=test_list,maxlag=maxlag, r_tts=r_tts, r_naive=r_naive);
#     if 'granger_x->y' in stats_list:
#         pvals['granger_x->y'], runtime['granger_x->y'] = get_sigf(y,x,granger_stat,test_list=test_list,maxlag=maxlag, r_tts=r_tts, r_naive=r_naive);
#     if 'ccm_y->x' in stats_list:
#         pvals['ccm_y->x'], runtime['ccm_y->x'] = get_sigf(x,y,ccm_statistic,test_list=test_list,maxlag=maxlag, r_tts=r_tts, r_naive=r_naive);
#     if 'ccm_x->y' in stats_list:
#         pvals['ccm_x->y'], runtime['ccm_x->y'] = get_sigf(y,x,ccm_statistic,test_list=test_list,maxlag=maxlag, r_tts=r_tts, r_naive=r_naive);
#     if 'pcc_param' in stats_list:
#         start = time.time()
#         pvals['pcc_param'] = {"pearson":(2*maxlag+1)*np.min(scan_lags(x,y,acorr_test_fisher_z,maxlag=maxlag)[2])};
#         runtime['pcc_param'] = time.time() - start
#     if 'granger_param_y->x' in stats_list:
#         start = time.time()
#         pvals['granger_param_y->x'], = {"granger_nosurr":(2*maxlag+1)*np.min(scan_lags(x,y.reshape(-1,1), \
#                         granger_stat,maxlag=maxlag,kw_statistic={'pval':True})[1])};
#         runtime['granger_param_y->x'] = time.time() - start
#     if 'granger_param_x->y' in stats_list:
#         start = time.time()
#         pvals['granger_param_x->y'] = {"granger_nosurr":(2*maxlag+1)*np.min(scan_lags(y,x.reshape(-1,1), \
#                         granger_stat,maxlag=maxlag,kw_statistic={'pval':True})[1])};
#         runtime['granger_param_x->y'] = time.time() - start
#     if 'lsa_data_driven' in stats_list:
#         start = time.time()
#         pvals['lsa_data_driven'] = {"lsa":(2*maxlag+1)*np.min(scan_lags(x,y.reshape(-1,1),dd_lsa,maxlag=maxlag)[2])};
#         runtime['lsa_data_driven'] = time.time() - start
#     return pvals, runtime

#     #%%% SurrX and SurrY     
# # To run in for loop for multiple model i . Run surrY for i or surrX for i  
# def surrX_and_surrY(data, name_of_data, test_list, stats_list, maxlag = 2,r_tts=choose_r, r_naive=choose_r): #runid, reps, 
#     results = {}
#     runtime = {}
#     print(f"Processing {name_of_data}")
#     print("Surrogates of Y")
#     results['surrY'], runtime['surrY'] = benchmark_stats(data[0], data[1], test_list= test_list, stats_list=stats_list, maxlag=maxlag, r_tts=r_tts, r_naive=r_naive)
#     print("Surrogates of X")
#     results['surrX'], runtime['surrX'] = benchmark_stats(data[1], data[0], test_list= test_list, stats_list=stats_list, maxlag=maxlag, r_tts=r_tts, r_naive=r_naive)
#     return results, runtime
#     #%%% Which model
# # Default params
# datagen_param = {'r': np.array([1, 0.72, 1.53, 1.27]),
#                  'a': np.array([[1, 1.09, 1.52, 0],
#                       [0, 1, 0.44, 1.36],
#                       [2.33, 0, 1, 0.47],
#                       [1.21, 0.51, 0.35, 1]]),  
#                  'dt_s' : 1.25, 
#                  'N' : 500, 
#                  'noise' : 0.01,
#                  'noise_T' : 0.05,
#                  'intx' : "competitive"}
# test_param = {'r_tts': 75,
#               'r_naive': 75, 
#               'maxlag': 2}

# # Input dgen is string.
# def for_generate_data(run_id, data_list, test_list, stats_list , datagen_param, test_param):
#     data = {} 
#     if ('xy_ar_u' in data_list):
#         data['xy_ar_u'] = dataGen.generate_AR1_uni_tau1(datagen_param['N']) # dependent
#     if ('xy_ar_u2' in data_list):
#         data['xy_ar_u2'] = dataGen.generate_AR1_uni_tau2(datagen_param['N']) # dependent
#     if ('xy_uni_logistic' in data_list):
#         data['xy_uni_logistic'] = dataGen.generate_uni_logistic_map(datagen_param['N']) # dependent
#     if ('xy_sinewn' in data_list):
#         data['xy_sinewn'] = dataGen.generate_sine_w_noise(datagen_param['N']) # dependent
#     if ('ts_ar' in data_list):
#         data['ts_ar'] = [dataGen.generate_ar1(datagen_param['N']), dataGen.generate_ar1(datagen_param['N'])]
#     if ('ts_logistic' in data_list):
#         data['ts_logistic'] = [dataGen.generate_logistic_map(datagen_param['N']), dataGen.generate_logistic_map(datagen_param['N'])]
#     if ('ts_sine_intmt' in data_list):
#         data['ts_sine_intmt'] = [dataGen.generate_sinew_intmt_corptxn(datagen_param['N']), dataGen.generate_sinew_intmt_corptxn(datagen_param['N'])]
#     if ('ts_coinflip' in data_list):
#         data['ts_coinflip'] = [dataGen.generate_coinflips_w_changeHeadprob_noise(datagen_param['N']), dataGen.generate_coinflips_w_changeHeadprob_noise(datagen_param['N'])]
#     if ('ts_noise_wv_kurtosis' in data_list):
#         data['ts_noise_wv_kurtosis'] = [dataGen.generate_noise_w_periodically_varying_kurtosis(datagen_param['N']), dataGen.generate_noise_w_periodically_varying_kurtosis(datagen_param['N'])]
#     # Dependent, x and w dependent on each other
#     if ('xy_FitzHugh_Nagumo' in data_list):
#         data['xy_FitzHugh_Nagumo'] = list(dataGen.generate_FitzHugh_Nagumo(datagen_param['N']).T)
#     # independent - according to Alex so made code of independent
#     if ('ts_chaoticLV' in data_list):
#         data['ts_chaoticLV'] = [dataGen.generate_chaotic_lv(datagen_param['N'], datagen_param['r'], datagen_param['a']), dataGen.generate_chaotic_lv(datagen_param['N'], datagen_param['r'], datagen_param['a'])]
#     # dependent - predator and prey
#     if ('xy_Caroline_LV' in data_list):
#         intx = datagen_param['intx']
#         data[f'xy_Caroline_LV_{intx}'] = list(dataGen.generate_lv(run_id = 1, dt_s = datagen_param['dt_s'], N=datagen_param['N'], noise=datagen_param['noise'], noise_T=datagen_param['noise_T'],intx=intx).T)
#     # dependent - two species competing for a chemical
#     if ('xy_Caroline_CH' in data_list):
#         intx = datagen_param['intx']
#         data['xy_Caroline_CH'] = list(dataGen.generate_niehaus(run_id = 1, dt_s = datagen_param['dt_s'], N=datagen_param['N'], noise=datagen_param['noise'], noise_T=datagen_param['noise_T'],intx=intx)[:,:2].T)
        
#     # Non-stationary
#     # ts_random_walk = [dataGen.generate_random_walk(datagen_param['N']),dataGen.generate_random_walk(datagen_param['N'])]
#     # ts_ar_wtrend = [dataGen.generate_ar1_w_trend(datagen_param['N']), dataGen.generate_ar1_w_trend(datagen_param['N'])]
#     return data

# def for_replicates(run_id, data_list, test_list, stats_list , datagen_param, test_param):
#     start = time.time()
#     data = for_generate_data(run_id, data_list, test_list, stats_list , datagen_param, test_param)
#     results = {}
#     for i in data: 
#         results[i] = surrX_and_surrY(data[i], i, test_list, stats_list, test_param['maxlag'], test_param['r_tts'], test_param['r_naive'])
#     runtime = time.time() - start
#     return {'run_id': run_id, 'pvals': results, 'XY': data, 'runtime': runtime}
# # 'run_id': run_id, 'pvals': results, 'XY': data, 'runtime': runtime


# def for_replicates_test(run_id, data_list, test_list, stats_list , datagen_param):
#     return {'run_id': run_id, 'data_list': data_list, 'test_list': test_list, 'stats_list': stats_list, 'datagen_param': datagen_param}
    

# def for_generated_data(run_id, data, data_name, test_list, stats_list, test_param):
#     results, runtime = surrX_and_surrY(data, data_name, test_list, stats_list, test_param['maxlag'], test_param['r_tts'],test_param['r_naive'])
#     return {'pvals': results, 'runtime': runtime}





    # filename = f'results_{name}.pkl'# f'results_{runid}_{i}_{reps}.pkl'
    # print(f'Saving {name} results to {filename}')
    # with open(filename, 'wb') as fi:
    #     pickle.dump(results, fi)
    # print(f'Done saving {name} results at {filename}') # ref number {reps}
    # return results
    
#%% Generate data
# if __name__=="__main__":
    # xy_ar_u = dataGen.generate_AR1_uni_tau1(2000) # dependent
    # xy_ar_u2 = dataGen.generate_AR1_uni_tau2(2000) # dependent
    # xy_uni_logistic = dataGen.generate_uni_logistic_map(2000) # dependent
    # xy_sinewn = dataGen.generate_sine_w_noise(2000) # dependent
    
    # ts_ar = [dataGen.generate_ar1(2000), dataGen.generate_ar1(2000)]
    # ts_logistic = [dataGen.generate_logistic_map(2000), dataGen.generate_logistic_map(2000)]
    
    # ts_sine_intmt = [dataGen.generate_sinew_intmt_corptxn(2000), dataGen.generate_sinew_intmt_corptxn(2000)]
    # ts_coinflip = [dataGen.generate_coinflips_w_changeHeadprob_noise(2000), dataGen.generate_coinflips_w_changeHeadprob_noise(2000)]
    # ts_noise_wv_kurtosis = [dataGen.generate_noise_w_periodically_varying_kurtosis(2000), dataGen.generate_noise_w_periodically_varying_kurtosis(2000)]
    # # Dependent, x and w dependent on each other
    # xy_FitzHugh_Nagumo = dataGen.generate_FitzHugh_Nagumo(2000)
    # r = np.array([1, 0.72, 1.53, 1.27])
    # a = np.array([[1, 1.09, 1.52, 0],
    #      [0, 1, 0.44, 1.36],
    #      [2.33, 0, 1, 0.47],
    #      [1.21, 0.51, 0.35, 1]])
    # dt_s, N = 0.25, 2000
    # # independent - according to Alex so made code of independent
    # ts_chaoticLV = [dataGen.generate_chaotic_lv(N, r, a), dataGen.generate_chaotic_lv(N, r, a)]
    
    # # dependent - predator and prey
    # xy_Caroline_LV = dataGen.generate_lv(run_id = 1, dt_s = 0.25, N=2000, noise=0.05*0.5,noise_T=0.05,intx="competitive")
    # # dependent - two species competing for a chemical
    # xy_Caroline_CH = dataGen.generate_niehaus(run_id=1, dt_s=0.25, N=2000, noise=0.05*0.5, noise_T=0.05, intx="competitive")[:,:2]
    # #%%% Save generated data
    # filename = 'results_Generated_data.pkl'
    # print(f'Saving Generated data to {filename}')
    # dill.dump_session(filename)
    # print(f'Done saving generated data at {filename}')
#%% Test and save data
# test = benchmark_stats(x, y)
# if __name__=="__main__":
#     test_list = ['tts', 'tts_naive', 'tts_multishift']
#     fi_results = []
#     for i in ['ts_ar_u', 'ts_ar_u2', 'ts_uni_logistic', 'xy_ar', 'xy_logistic', 'xy_sinewn', 'xy_sine_intmt', 'xy_coinflip', 'xy_noise_wv_kurtosis', 'xy_chaoticLV']: # , 'xy_FitzHugh_Nagumo'
#         results = {}
#         print(f"Processing {i}")
#         data = globals()[i]
#         print("Surrogates of Y")
#         results[f'{i}_xy'] = benchmark_stats(data[0], data[1], test_list= test_list, maxlag= 4)
#         print("Surrogates of X")
#         results[f'{i}_yx'] = benchmark_stats(data[1], data[0], test_list= test_list, maxlag= 4)
#         filename = f'results_{i}.pkl'
#         fi_results.append(filename)
#         print(f'Saving {i} results to {filename}')
#         with open(filename, 'wb') as fi:
#             pickle.dump(results, fi)
#         print(f'Done {i} saving results at {filename}')
        
#     with open('results_file_list.pkl', 'wb') as fi:
#         pickle.dump(fi_results, fi)
# # .spydata
# # https://docs.spyder-ide.org/current/panes/variableexplorer.html#toolbar-buttons

# #%%% Save all again

#     filename = 'Correlation_simulation.pkl'
#     dill.dump_session(filename)

# # dill.load_session(filename)

# # with open('results_file_list.pkl', 'rb') as file:
# #     # Call load method to deserialze
# #     fi_results = pickle.load(file)
  
# #     print(myvar)