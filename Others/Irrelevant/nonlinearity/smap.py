#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 22:32:54 2021

@author: carolc24
"""

import numpy as np
from scipy.spatial import distance_matrix
from scipy.stats import pearsonr

#weight neighbors based on distance. for picking embed dim
def get_weights(distances):
    """Performs weighting in KNN for Simplex projection

    Args:
      * distances_ (array) Array of distances

    Returns:
      * weights
    """
    distances = np.array(distances, copy=True).astype(np.float)
    if len(distances.shape) == 1:
        distances = distances.reshape(1,-1)
    if np.any(distances == 0): # deal with zeros
        distances[distances != 0] = np.inf
        distances[distances == 0] = 1
    min_dist = np.min(distances, axis=1).reshape(-1,1)
    u = np.exp(-distances / min_dist)
    weights = u / np.sum(u, axis=1).reshape(-1,1)
    return weights

#nonlinearity test from sugihara (1994)

ts = np.random.random(200); # fake time series for now. 200 points from uniform dist

x = ts[:int(np.floor(ts.size/2))] # library points
y = ts[int(np.floor(ts.size/2)):] # forecasting points

#scan embedding dimensions
rho = np.zeros(21)
for E in np.arange(2,21):
    #embed data
    X = np.zeros((x.size-E+1,E))
    Y = np.zeros((y.size-E+1,E))
    for i in range(E):
        X[:,i] = x[(E-i-1):(x.size-i)]
        Y[:,i] = y[(E-i-1):(y.size-i)]
        
    #find neighbors of Y in X
    XY = distance_matrix(Y,X);
    nbrs = np.argsort(XY)[:,:(E+1)]
    wgts = get_weights(np.sort(XY)[:,:(E+1)])
    
    #predict Y with weighted X neighbors
    yhat = np.sum(nbrs * wgts,axis=1);
    
    #how good is the prediction?
    rho[E] = pearsonr(y[(E-1):],yhat)[0];
    
embed_dim = np.argmax(rho);

X = np.ones((x.size-embed_dim+1,embed_dim))
Y = np.ones((y.size-embed_dim+1,embed_dim+1))
for i in range(embed_dim):
    X[:,i] = x[(embed_dim-i-1):(x.size-i)]
    Y[:,i+1] = y[(embed_dim-i-1):(y.size-i)]
    
#find neighbors of Y in X
XY = distance_matrix(Y[:,1:],X);
dbar = np.mean(XY);
nbrs = np.argsort(XY);

err = np.zeros(21);

for i in range(21):
    theta = 0.05*i;
    yhat = np.zeros(Y.shape[0]);
    #loop through time points
    for t in np.arange(Y.shape[0]-1):
        #we want to predict y[t+1] from Y[t]
        #we build a model with X
        #yhat[t+1] = dot product of C and Y[t]
        #B = AC, find C with svd
        
        A = X[:-(embed_dim-1)] * np.exp(-theta*XY[t:(t+1),:-(embed_dim-1)]/dbar).T
        B = x[(embed_dim):] * np.exp(-theta*XY[t:(t+1),:-(embed_dim-1)]/dbar)
        
        #svd
        u,s,vh = np.linalg.svd(A,full_matrices=False);
        C = vh.T @ np.linalg.inv(np.diag(s)) @ u.T @ B.T;
        
        yhat[t+1] = Y[t] @ C;
    
    err[i] = np.sum((yhat[1:] - y[2:])**2);
