#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 08:27:53 2023

@author: h_k_linh
"""
import scipy as sp
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import random

def lotkaVolterra(t,y,mu,M):
    return y * (mu + M @ y);

def lotkaVolterraSat(t,y,mu,M,K):    
    intx_mat = M/(K + y);
    lv = y * (mu + intx_mat @ y);
    return lv;

# Originally Caroline return tested results. Linh modified to only generate ts
def generate_lv(run_id, dt_s, N, s0, M, mu, noise,noise_T, fn = lotkaVolterra): # ,intx="competitive"
    print('Generating Caroline Lotka-Volterra model')
    dt=0.05;
    lag = int(150/dt);
    sample_period = int(np.ceil(dt_s / dt)); 
    obs = sample_period * N;
    s = np.zeros((lag + obs + 1, 2))
    
    s[0] = s0
    for i in range(lag + obs):
        soln = solve_ivp(fn,[0,dt],s[i],args=args)
        eps = noise*np.random.randn(2)*np.random.binomial(1,dt/noise_T,size=2);
        s[i+1] = soln.y[:,-1] + eps;
        s[i+1][np.where(s[i+1] < 0)] = 0;

    x = s[lag::sample_period,];
    for i in range(x.ndim):
        x[:,i] += 0.001*np.random.randn(x[:,i].size);

    # if run_id % 100 == 0:
    #     print("test " + str(run_id) + " finished");
        
    # plt.plot(x)
    return x

if __name__ == '__main__':
    N = 500
    
    dt_s = 1.25
    
    mu = np.array([0.7,0.7]);
    M = np.zeros((2,2));
    fn = lotkaVolterra;
    s0 = [1.,1.]
    
    if (intx=="competitive"):
        M = [[-0.4,-0.5],
             [-0.5,-0.4]];
        args=(mu,M)
    if (intx=="asym_competitive"):
        mu = [0.8,0.8];
        M = [[-0.4,-0.5],[-0.9,-0.4]];
        args=(mu,M);
    if (intx=="asym_competitive_2"):
        M = [[-1.4,-0.5],
             [-0.9,-1.4]];
        mu = [0.8,0.8];
        args=(mu,M);
    if (intx=="asym_competitive_2_rev"):
        mu = [0.8,0.8];
        M = [[-1.4,-0.9],[-0.5,-1.4]];
        args=(mu,M);  
    if (intx=="asym_competitive_3"):
        mu = [50.,50.];
        M = [[-100.,-95.],[-99.,-100.]]
        args=(mu,M);
    elif (intx=="mutualistic"):
        M = [[-0.4,0.3],
             [0.3,-0.4]];
        args=(mu,M)
    elif (intx=="saturable"):     
        M = np.array([[-10.,0.3],
                      [1.,-10.]]);
        K = np.array([[100.,10.],
                      [20.,100.]]) 
        args=(mu,M,K)
        fn = lotkaVolterraSat;
    elif (intx=="predprey"):              
        mu = np.array([1.1,-0.4])
        M = np.array([[0.0,-0.4],
                      [0.1,0.0]]);
        args=(mu,M)
        
    