#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 08:27:53 2023

@author: h_k_linh
"""
import os
# os.chdir('C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/')
# os.chdir('/home/h_k_linh/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test')
print(f'working directory: {os.getcwd()}')
import scipy as sp
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import random
import time
"""
General form:
dS_i/dt = S_i*[mu_i + \sum_{j=1}^{N}{M_{ij}*S_{j}}]

More specifically for modelling competition between 2 species X and Y:
    dX/dt = X*[mu_x - b_x*X - c_yx*Y]
    dY/dt = Y*[mu_y - c_xy*X - b_y*Y]
    
    * Change in species population from one 'generation' to the next (dt can be infinitely small)
    is dependent on the current state of the population. Without competition, the next generation 
    can increase by X*mu_x (or Y*mu_y for Y species), so we can call mu_x (or mu_y) the maximum
    growth rate (or basal growth rate) of each species.
    Due to competition within the same species, the maximum change is decrease by b_x*X (or b_y*Y)
    because the inhibition effect (b_x or b_y) is proportional to how much the current population is
    (X or Y depending on wich population we are considering).
    Same goes for competition between species, or we call cross-species competition. 

If we change the sign of the coefficients of the inhibitory effect from negative to positive, the 
effect is now beneficial to the species (increase in population change), then the relationship becomes 
mutualistic instead of competitive.
 
To explore the 
"""
def lotkaVolterra(t,y,mu,M):
    return y * (mu + M @ y);

# Originally Caroline return tested results. Linh modified to only generate ts
def generate_lv(dt_s, N, s0, mu, M, noise, noise_T, raise_extinct=0, fn = lotkaVolterra): # ,intx="competitive"
    print('Generating Caroline Lotka-Volterra model')
    dt=0.05; # integration step
    lag = int(150/dt);
    sample_period = int(np.ceil(dt_s / dt)); 
    obs = sample_period * N;
    s = np.zeros((lag + obs + 1, 2))
    
    args = (mu,M);
    s[0] = s0
    # neg_count = 1
    # pos_cumulate =0
    # pos_period = []
    
    for i in range(lag + obs):
        soln = solve_ivp(fn,[0,dt],s[i],args=args)
        eps = noise*np.random.randn(2)*np.random.binomial(1,dt/noise_T,size=2); # process noise. add process noise in every integration steps.
        # eps[1] += raise_extinct;
        s[i+1] = soln.y[:,-1] + eps; # print(s[i+1])
        # if i > lag: 
        #     if np.any(np.where(s[i+1] < 0)): 
        #         neg_count +=1
        #         pos_period.append(pos_cumulate)
        #         pos_cumulate = 0
        #     else: 
        #         pos_cumulate += 1
        s[i+1][np.where(s[i+1] < 0)] = 0;

    x = s[lag:lag+obs:sample_period,]; # measurement noise
    for i in range(x.ndim):
        x[:,i] += 0.001*np.random.randn(x[:,i].size)

    # if run_id % 100 == 0:
    #     print("test " + str(run_id) + " finished");
        
    # plt.plot(x)
    return [x[:,_] for _ in [0,1]] #, neg_count, pos_period]
    # return [x[:,_] for _ in [0,1]]
    
def generate_lv_nocap(dt_s, N, s0, mu, M, noise, noise_T, raise_extinct=0, fn = lotkaVolterra): # ,intx="competitive"
    print('Generating Caroline Lotka-Volterra model')
    dt=0.05; # integration step
    lag = int(150/dt);
    sample_period = int(np.ceil(dt_s / dt)); 
    obs = sample_period * N;
    
    args = (mu,M);
    
    # s = np.zeros((lag + obs + 1, 2))
    scap = np.zeros((lag + obs + 1, 2))
    sncap = np.zeros((lag + obs + 1, 2))
    # s[0] = s0
    scap[0] = s0
    sncap[0] = s0
    # neg_count = 1
    neg_count_cap = 1
    neg_count_ncap = 1
    # pos_cumulate =0
    pos_cumulate_cap =0
    pos_cumulate_ncap =0
    # pos_period = []
    pos_period_cap = []
    pos_period_ncap = []
    
    for i in range(lag + obs):
        soln_cap = solve_ivp(fn,[0,dt],scap[i],args=args)
        soln_ncap = solve_ivp(fn,[0,dt],sncap[i],args=args)
        eps = noise*np.random.randn(2)*np.random.binomial(1,dt/noise_T,size=2); # process noise. add process noise in every integration steps.
        # eps[1] += raise_extinct;
        scap[i+1] = soln_cap.y[:,-1] + eps; # print(s[i+1])
        sncap[i+1] = soln_ncap.y[:,-1] + eps;
        if i > lag: 
            if np.any(np.where(scap[i+1] < 0)): 
                neg_count_cap +=1
                pos_period_cap.append(pos_cumulate_cap)
                pos_cumulate_cap = 0
            else: 
                pos_cumulate_cap += 1
                
            if np.any(np.where(sncap[i+1] < 0)): 
                neg_count_ncap +=1
                pos_period_ncap.append(pos_cumulate_ncap)
                pos_cumulate_ncap = 0
            else: 
                pos_cumulate_ncap += 1
        
        scap[i+1][np.where(scap[i+1] < 0)] = 0;

    return {'s':{'cap': scap, 'ncap': sncap}, 'lag': lag, 'obs': obs, 'sample_period': sample_period, 
            'neg_count': {'cap': neg_count_cap, 'ncap': neg_count_ncap},
            'pos_periods':{'cap': pos_period_cap, 'ncap': pos_period_ncap}}
    
    # x = s[lag:lag+obs:sample_period,]; # measurement noise
    # for i in range(x.ndim):
    #     x[:,i] += 0.001*np.random.randn(x[:,i].size);

    # if run_id % 100 == 0:
    #     print("test " + str(run_id) + " finished");
        
    # plt.plot(x)
    # return [[x[:,_] for _ in [0,1]], neg_count, pos_period]
    # return [x[:,_] for _ in [0,1]]
#%%
if __name__ == '__main__':
    '''
    # Old parameters that Caroline gave
    intx = ''
    if (intx=="competitive"):
        M = [[-0.4,-0.5],
             [-0.5,-0.4]];
    if (intx=="asym_competitive"):
        mu = [0.8,0.8];
        M = [[-0.4,-0.5],[-0.9,-0.4]];
    if (intx=="asym_competitive_2"):
        M = [[-1.4,-0.5],
             [-0.9,-1.4]];
        mu = [0.8,0.8];
    if (intx=="asym_competitive_2_rev"):
        mu = [0.8,0.8];
        M = [[-1.4,-0.9],[-0.5,-1.4]];  
    if (intx=="asym_competitive_3"):
        mu = [50.,50.];
        M = [[-100.,-95.],[-99.,-100.]]
    elif (intx=="mutualistic"):
        M = [[-0.4,0.3],
             [0.3,-0.4]];
    # elif (intx=="saturable"):     
    #     M = np.array([[-10.,0.3],
    #                   [1.,-10.]]);
    #     K = np.array([[100.,10.],
    #                   [20.,100.]]) 
    #     args=(mu,M,K)
    #     fn = lotkaVolterraSat;
    elif (intx=="predprey"):              
        mu = np.array([1.1,-0.4])
        M = np.array([[0.0,-0.4],
                      [0.1,0.0]]);)
    ARGs = {'dt_s': 1.25, 'N': 500, 's0': [1.,1.], 'mu': np.array([0.7, 0.7]), 'M': np.array([[-0.4,-0.5],[-0.5,-0.4]]), 'noise': 0.01, 'noise_T': 0.05}
    # fn = lotkaVolterra;
    ''' 
#%% Trying other parameters
# The case where Caroline saw a difference
    ARGs = {'dt_s': 0.25, 'N': 500, 's0': [2.,0.], 'mu': np.array([0.7,0.7]), 'M': np.array([[-0.4,-0.5],[-0.5,-0.4]]), 'noise': 0.01, 'noise_T': 0.05}
    # change raise noise to make them closer
    # how come s0 make such a difference? if the system is chaos, it should
        # be affected by initial conditions but the initial condition is fixed so 
        # the only thing that's making the difference here is the noise
        # and also, the oscillation around 0 is highly suspicious
        # maybe we should try raising 0, maybe put it to [4.,1.]
        # these should be similar to raising noise but we will see
    # change dt_s but Caroline says it wont change much
    # avoid changing mu unless I want a different scale
    # for this particular case, keep M
    
    # So far, Caroline have also tried:
        # ARGs = {'dt_s': 0.25, 'N': 500, 's0': [2.,0.], 'mu': np.array([50, 50]), 'M': np.zeros([[-100,-90],[-99,-100]]), 'noise': 0.01, 'noise_T': 0.05}
    reps = 3    
    start = time.time()
    # for i in range(reps):
    #     for a in np.arange(0.25,2,0.25): # different growth rate
    #         mu = np.array([a,a])
    #         for b in np.arange(0.25,2,0.5): # different self-inbtn rate, shouldn't be larger than a though ...
    #             for c in np.arange(0.25,2,0.25): # different cross-inbtn rate (at some point will be larger than b which makes gamma/beta >1)
    #                 M = np.array([[-b, -c], [-c, -b]])
    #                 for s0 in [np.array([s1,s2]) for s1 in [0.1, 0.3, 0.5] for s2 in [0.5, 0.7, 0.9]]:
    #                     key = f"a_{a}_b_{b}_c_{c}_s0_{'_'.join([str(x) for x in s0])}"
    #                     print(key)
    #                     data[key] = generate_lv(dt_s, N, s0, M, mu, noise, noise_T)
    time.time() - start
#%%
    # x = np.arange(500)
    # def parse_parameters_in_keys(key):
    #     splt = key.split('_')
    #     return [float(splt[_]) for _ in [1,3,5,7,8]]
    
    # row,col = 0,0
    # for key, val in data.items():
    #     if row == 0 and col == 0:
    #         fig = plt.figure(constrained_layout=True)
    #         gs = fig.add_gridspec(3, 3)
    #     a,b,c,s1,s2 = parse_parameters_in_keys(key)
    #     if a!=0: beta = b/a ; gamma = c/a;
    #     else: beta = 'a=0'; gamma = 'a=0'
    #     ax = fig.add_subplot(gs[row,col])
    #     ax.plot(x,val[:-1,0], color = "blue")
    #     ax.plot(x,val[:-1,1], color = "#ff6600")
    #     ax.set_title(f'a={a}, b={b}, c={c},\n beta={beta}, gamma={gamma}')
    #     if col<2:
    #         col+=1
    #     elif row<2:
    #         col=0
    #         row+=1
    #     else: plt.show(); row=0; col=0
    """
    After just comparing them I realised that it really doesn't matter the range.
    (too many a, b, c all just or one type of (X,Y) )
    It matters the number of stable points that we calculated. 
    So we can just pick 2 cases of each stable point and make the stable point cahnges.
    And then varies the parameters. changing s0 is so far not making big impact.
    """
#%%
    
        
    