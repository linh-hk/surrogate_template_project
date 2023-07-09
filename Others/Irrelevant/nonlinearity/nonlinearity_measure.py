#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 22:32:54 2021

@author: carolc24
"""

import numpy as np
from scipy.spatial import distance_matrix
from scipy.stats import pearsonr

# nonlinear prediction error
#ts: NxM time series vector with N observations and M variables
#embed_dim: scalar int, embedding dimension
#tau: scalar int, time delay
#for more info on embedding, check out Lancaster 2018
#pearson: bool, True if we quantify prediction error as pearson correlation
#instead of as root mean square error
def pred_error(ts, embed_dim, tau=1, pearson=False):
    
    #normalize time series to have zero mean and unit variance
    ts_norm = (ts - np.mean(ts,axis=0)) / np.std(ts, axis=0)
    
    N = ts_norm.shape[0];
    if (ts_norm.ndim > 1):
        M = ts_norm.shape[1];
    else:
        M = 1;
    #embed time series
    #list of delay vectors (subsets of time series at equally spaced points in time)
    X_array = np.zeros((N - (embed_dim-1)*tau - 1,embed_dim,M));
    for i in range(embed_dim):
        X_array[:,i] = ts_norm[tau*i:(N - (embed_dim-1)*tau + tau*i - 1)];
        
    #k-nearest neighbors algorithm
    rms_err = np.zeros(M);
    pcorr = np.zeros(M);
    for j in range(M):
        X = X_array[:,:,j];
        # response variable. one step future of each delay vector's latest point
        X_true = ts_norm[(tau*(embed_dim-1) + 1):,j]; 
        #make list of sorted neighbors
        KX = distance_matrix(X,X);
        
        #exclusion radius 4 (self + nearest 3)
        #we don't use very similar delay vectors in prediction
        for i in range(KX.shape[0] - 1):
            if (i < KX.shape[0] - 4):
                KX[i,i:(i+4)] = np.inf
                KX[i:(i+4),i] = np.inf
            else:
                KX[i,i:] = np.inf
                KX[i:,i] = np.inf
        nbr = np.argsort(KX);
        
        err = np.zeros(X.shape[0]);
        X_pred = np.zeros(X.shape[0]);
        
        #estimate each pt based on neighbors list
        for i in range(X.shape[0]):
            #X_true[i] = X[i+1,0];
            
            #number of neighbors = embed_dim + 1
            #no distance based weighting
            N_nbrs = embed_dim + 1;
            X_nbrs = nbr[i,:N_nbrs];
            
            X_pred[i] = np.mean(X_true[(X_nbrs).astype(int)]);
            err[i] = (X_true[i] - X_pred[i])**2;
            
        rms_err[j] = np.sqrt(np.mean(err,axis=0))
        pcorr[j] = pearsonr(X_pred, X_true)[0]
    
    if pearson:
        return pcorr;
    else:
        return rms_err;

#time irreversibility
def tm_irrev(ts):
    #normalize data to have zero mean unit variance
    ts_norm = (ts - np.mean(ts,axis=0)) / np.std(ts, axis=0)
    return np.mean((ts_norm[1:] - ts_norm[:-1])**3,axis=0);