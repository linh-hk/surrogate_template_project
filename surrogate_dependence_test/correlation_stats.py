#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 11:03:15 2023

@author: h_k_linh

Pearson correlation, lsa, mutual infomation
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
from sklearn.feature_selection import mutual_info_regression # for mutual information

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

# Adapted from Alex's code
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

def lsa_new(x,y_array):
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

def lsa_new_delay(x,y_arr,D=3):
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

#%% Parametric test - not used
# # Box-Jenkins estimator for autocorrelation
# # Equation 6 in Pyper and Peterman , look in notion
# def autocorrelationcorr_BJ(x,acov=False):
#     result = np.zeros(x.size)
#     mean = np.mean(x)
#     if acov:
#         denomenator = x.size;
#     else:
#         denomenator = np.sum((x - mean)**2)
#     # test different distance j from x[t]. the smaller j, the higher result.
#     for j in range(x.size):
#         numerator = 0
#         for t in range(x.size-j):
#             numerator += (x[t] - mean) * (x[t + j] - mean)
#         result[j] = numerator / denomenator
#     return result

# # crosscorrelation
# def crosscorrelation(x,y):
#     numerator = np.sum( (x - np.mean(x)) * (y - np.mean(y)) )
#     x_sum_squares = np.sum((x - np.mean(x))**2)
#     y_sum_squares = np.sum((y - np.mean(y))**2)
#     return numerator / np.sqrt(x_sum_squares * y_sum_squares)

#     #%%% t-distribution
# # calculate t-value from ttest, small fxn for parametric test
# def p_from_ttest(r,df): 
#     print("\t\t\t\t\t\tCalculating p-value from t distribution")
#     # r = crosscorrelation stats, df = degree of freedom
#     t = np.abs(r) * np.sqrt(df) / np.sqrt(1 - r ** 2)
#     pval = 1-stats.t.cdf(t, df)
#     return t, 2*pval

# # the parametric test
# def acorr_test_1(x,y,pval_only=False):
#     # adusted t-test incorporating autocorrelation correction
#     x_acorr = autocorrelationcorr_BJ(x)
#     y_acorr = autocorrelationcorr_BJ(y)
#     idx = np.arange(1,x.size) #j
#     weights = (x.size - idx) / x.size # (N-j)/N according to equation
#     df_recip = (1 + 2 * np.sum( weights * x_acorr[1:] * y_acorr[1:])) / x.size 
#     # Degree of freedom: Eq 1 in Pyper and Peterman, Afyouni eq 5
#     df = 1/df_recip - 2
#     r = crosscorrelation(x, y)
#     t, p = p_from_ttest(r, df) # derive pvalue of crosscorelation from autocorrelation's t distribution
#     if (pval_only):
#         return p;
#     else:
#         return np.array([r, p])
#     #%%% normal distribution with modified degree of freedom
# #Fisher transformation of Pearson corr to z(normal) distribution # Afyouni et al eq 5
# def acorr_test_fisher_z(x,y):
#     print("\t\t\t\t\t\tCalculating p-value from z distribution of Fisher transformed Pearson correlation coefficient (parametric test)")
#     #autocorrelation is not truncated
#     x_acorr = autocorrelationcorr_BJ(x)
#     y_acorr = autocorrelationcorr_BJ(y)
#     idx = np.arange(1,x.size)
#     weights = (x.size - idx) / x.size
#     #take absolute value of correlation. we don't care about sign here
#     rho = crosscorrelation(x, y);
#     rho_var = (1 + 2 * np.sum( weights * x_acorr[1:] * y_acorr[1:])) / x.size # Afyouni et al eq 5
#     z_num = np.arctanh(rho);
#     z_denom = np.sqrt(rho_var * (1 - rho**2) ** -2)
#     #final fisherized z-score
#     z_score = z_num / z_denom;
#     #one-tailed test
#     #remember we took absolute value so the null dist should reflect this
#     pval = 2 * (1 - stats.norm.cdf(np.abs(z_score)));
#     return np.array([rho,pval]);
#     #%%% Data-driven Local similarity
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
#     x_acov = autocorrelationcorr_BJ(x,acov=True).reshape(-1)
#     y_acov = autocorrelationcorr_BJ(y,acov=True).reshape(-1)
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
#     print("\t\t\t\t\t\tCalculating p-value from data-driven local similarity (parametric) test")
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
