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

import random
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.major.size'] = 7
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.major.size'] = 7

mpl.rcParams['hatch.color'] = 'white'
mpl.rcParams['hatch.linewidth'] = 3

font = {'fontsize' : 24, 'fontweight' : 'bold', 'fontname' : 'arial'}
# font_data = {'fontsize' : 20, 'fontweight' : 'bold', 'fontname' : 'arial','color':'white'}
font_data = {'fontsize' : 26, 'fontweight' : 'bold', 'fontname' : 'arial','color':'black'}

#%%
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

# LV no external perturbabtions, no sampling period
def generate_lv(N, mu, M, s0, fn=lotkaVolterra):
    print('Generating Caroline Lotka-Volterra model')
    dt=0.05;    
    args=(mu,M)    
    s = np.zeros((N+1, 2))
    s[0] = s0
    for i in range(N):
        soln = solve_ivp(fn,[0,dt],s[i],args=args)
        s[i+1] = soln.y[:,-1];
        s[i+1][np.where(s[i+1] < 0)] = 0; 
    return [s[:,_] for _ in [0,1]]
    # return s

def vis_data(data):
    if len(data) ==2:
        fig,ax = plt.subplots()
        ax.plot(data[0])
        ax.plot(data[1])
        plt.show()
    else:
        fig, ax = plt.subplots(2,1,sharex= True)
        for dat in data:
            ax[0].plot(dat[0])
            ax[1].plot(dat[1])
        ax[0].set_title('Species 1')
        ax[1].set_title('Species2')
            
    
def vis_phase(data):
    if len(data) == 2:
        fig,ax = plt.subplots()
        ax.scatter(data[0], data[1], s=2)
        plt.show()
    else:
        fig, ax = plt.subplots()
        for _ in data:
            ax.scatter(_[0],_[1], s=2)
        plt.show()
            
#%%
# BSGoh 1976
mu = np.array([-2,5]);
M = np.array([[1,1],
              [-3,-2]]);
s0 = np.array([1.6,1.])
# S1 will go to infty very fast so N=32 is enough 33 is a bit too much,
# and S2 will go to 0
ARGs = {'N': 500, 'mu': mu, 'M': M, 's0':s0}
s = generate_lv(**ARGs)
vis_data([_[:32] for _ in s]) #33 is bad : )
# pred-prey
mu = np.array([1.1,-0.4]);
M = np.array([[0.0,-0.4],
              [0.1,0.0]]);
s0 = np.array([1.,1.])
ARGs = {'N': 500, 'mu': mu, 'M': M, 's0':s0}
s = generate_lv(**ARGs)
vis_phase(s)


# draw pred-prey phase
many_s = []
for S1_0 in range(1,4,1):
    for S2_0 in range(1,4,1):
        s0 = np.array([S1_0, S2_0])
        ARGs = {'N': 500, 'mu': mu, 'M': M, 's0':s0}
        many_s.append(generate_lv(**ARGs))
        
vis_phase(many_s)
    
#%% draw phase for different parameters Caroline gave:
mu = np.array([0.7,0.7]);

# competitive
M = [[-0.4,-0.5],
     [-0.5,-0.4]];
many_comp = []
for S1_0 in np.arange(1,5,1):
    for S2_0 in np.arange(0.1,1,0.1):
        s0 = np.array([S1_0, S2_0])
        ARGs = {'N': 500, 'mu': mu, 'M': M, 's0':s0}
        many_comp.append(generate_lv(**ARGs))
vis_phase(many_comp)

# "mutualistic"
M = [[-0.4,0.3],
     [0.3,-0.4]];
many_mut = []
for S1_0 in np.arange(1,5,1):
    for S2_0 in np.arange(1,5,1):
        s0 = np.array([S1_0, S2_0])
        ARGs = {'N': 500, 'mu': mu, 'M': M, 's0':s0}
        many_mut.append(generate_lv(**ARGs))
vis_phase(many_mut)
many_mut = []
for S1_0 in np.arange(6.9,7,0.01):
    for S2_0 in np.arange(6.9,7,0.01):
        s0 = np.array([S1_0, S2_0])
        ARGs = {'N': 500, 'mu': mu, 'M': M, 's0':s0}
        many_mut.append(generate_lv(**ARGs))
vis_phase(many_mut)

# "mutualistic2"
M = [[-0.4,0.5],
     [0.5,-0.4]];
many_mut2 = []
for S1_0 in np.arange(1,5,1):
    for S2_0 in np.arange(1,5,1):
        s0 = np.array([S1_0, S2_0])
        ARGs = {'N': 500, 'mu': mu, 'M': M, 's0':s0}
        many_mut2.append(generate_lv(**ARGs))
vis_phase(many_mut2)
many_mut = []
for S1_0 in np.arange(6.9,7,0.01):
    for S2_0 in np.arange(6.9,7,0.01):
        s0 = np.array([S1_0, S2_0])
        ARGs = {'N': 500, 'mu': mu, 'M': M, 's0':s0}
        many_mut.append(generate_lv(**ARGs))
vis_phase(many_mut)

#%%
mu = [0.8,0.8]

#"asym_competitive"
M = [[-0.4,-0.5],[-0.9,-0.4]];
many_asymcomp = []
for S1_0 in np.arange(1,5,1):
    for S2_0 in np.arange(0.1,1,0.1):
        s0 = np.array([S1_0, S2_0])
        ARGs = {'N': 500, 'mu': mu, 'M': M, 's0':s0}
        many_asymcomp.append(generate_lv(**ARGs))
vis_phase(many_asymcomp)

#"asym_competitive_2"
M = [[-1.4,-0.5],
     [-0.9,-1.4]];
many_asymcomp2 = []
for S1_0 in np.arange(1,5,1):
    for S2_0 in np.arange(0.1,1,0.1):
        s0 = np.array([S1_0, S2_0])
        ARGs = {'N': 500, 'mu': mu, 'M': M, 's0':s0}
        many_asymcomp2.append(generate_lv(**ARGs))
vis_phase(many_asymcomp2)


# "asym_competitive_2_rev"
M = [[-1.4,-0.9],[-0.5,-1.4]];
many_asymcomp_2_rev = []
for S1_0 in np.arange(1,5,1):
    for S2_0 in np.arange(0.1,1,0.1):
        s0 = np.array([S1_0, S2_0])
        ARGs = {'N': 500, 'mu': mu, 'M': M, 's0':s0}
        many_asymcomp_2_rev.append(generate_lv(**ARGs))
vis_phase(many_asymcomp_2_rev)  

# "asym_competitive_3"
mu = [50.,50.];
M = [[-100.,-95.],[-99.,-100.]]
many_asymcomp3 = []
for S1_0 in np.arange(1,5,1):
    for S2_0 in np.arange(0.1,1,0.1):
        s0 = np.array([S1_0, S2_0])
        ARGs = {'N': 500, 'mu': mu, 'M': M, 's0':s0}
        many_asymcomp3.append(generate_lv(**ARGs))
vis_phase(many_asymcomp3)
many_asymcomp3_ = []
for S1_0 in np.arange(0.1,1,0.1):
    for S2_0 in np.arange(0.1,1,0.1):
        s0 = np.array([S1_0, S2_0])
        ARGs = {'N': 500, 'mu': mu, 'M': M, 's0':s0}
        many_asymcomp3_.append(generate_lv(**ARGs))
vis_phase(many_asymcomp3_)

#%% Add sampling frequency, add perturbations
def generate_lv(N, mu, M, s0, dt_s, noise, noise_T, raise_extinct=0, fn = lotkaVolterra): # ,intx="competitive"
    print('Generating Caroline Lotka-Volterra model')
    dt=0.05; # integration step
    lag = int(150/dt);
    sample_period = int(np.ceil(dt_s / dt)); 
    obs = sample_period * N;
    
    args = (mu,M);
    s = np.zeros((lag + obs + 1, 2))
    s[0] = s0
    
    for i in range(lag + obs):
        soln = solve_ivp(fn,[0,dt],s[i],args=args)
        eps = noise*np.random.randn(2)*np.random.binomial(1,dt/noise_T,size=2); # process noise. add process noise in every integration steps.
        # eps[1] += raise_extinct;
        s[i+1] = soln.y[:,-1] + eps; # print(s[i+1])
        s[i+1][np.where(s[i+1] < 0)] = 0;

    x = s[lag:lag+obs:sample_period,]; # measurement noise
    # for i in range(x.ndim):
    #     x[:,i] += 0.001*np.random.randn(x[:,i].size)

    return [x[:,_] for _ in [0,1]] #, neg_count, pos_period]
    # return [x[:,_] for _ in [0,1]]
    
#"asym_competitive"
mu = [0.8,0.8]
M = [[-0.4,-0.5],[-0.9,-0.4]];
many_asymcomp = []
for S1_0 in np.arange(1,5,1):
    for S2_0 in np.arange(0.1,1,0.1):
        s0 = np.array([S1_0, S2_0])
        ARGs = {'N': 500, 'mu': mu, 'M': M, 's0':s0, 
                'dt_s': 0.25, 'noise': 0, 'noise_T':0.25}
        many_asymcomp.append(generate_lv(**ARGs))
vis_phase(many_asymcomp)

#%%
from Simulation_code.surrogate_dependence_test.main import manystats_manysurr
