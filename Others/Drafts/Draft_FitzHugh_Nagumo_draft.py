#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 06:06:22 2023

@author: h_k_linh
"""
import scipy as sp
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle

def d_FitzHugh_Nagumo(t, x_w):
    x = x_w[0]
    w = x_w[1]
    epsilon_t = np.random.normal(loc = 1, scale = 0.2)
    dx_t = 0.7* (x - x**3 /3 - w + epsilon_t)
    dw_t = 0.7*0.08* (x + 0.7 -0.8 * w)
    return np.array([dx_t, dw_t])

# Turns out that 0.7 is Alex's delta t so it seems that we dont need 0.7 in the equation.
# Maybe we'll generate a time series that does not have 0.7 in the differential equation.
# Let that equation be xy_FitzHugh_Nagumo_0.5

def d_FitzHugh_Nagumo_(t, x_w):
    x = x_w[0]
    w = x_w[1]
    epsilon_t = np.random.normal(loc = 1, scale = 0.2)
    dx_t = x - x**3 /3 - w + epsilon_t
    dw_t = 0.08* (x + 0.7 -0.8 * w)
    return np.array([dx_t, dw_t])

def generate_FitzHugh_Nagumo(N, dxy_dt = d_FitzHugh_Nagumo_, dt_s = 0.25):
    print('Generating FitzHugh-Nagumo')
    dt=0.05;
    sample_period = int(np.ceil(dt_s / dt));
    
    lag = int(150/dt);
    obs = sample_period * N;
    # epsilon = np.random.standard_normal(size = length + 1000)
    s = np.zeros((lag + obs + 1,2))
    for t in range(lag + obs):
        soln = sp.integrate.solve_ivp(dxy_dt, t_span= [0,dt], y0= s[t])
        # print(soln.y[:,-1])
        s[t+1] = soln.y[:,-1] # + eps;
        # print(s[t+1])
        
    x = s[lag::sample_period,];    
    return x[-N:,]
    # return s

# Investigate epsilon t effect
# epsilon_t is supposed to be R*I_external in the system
# epsilon_t = 0
# epsilon_t = 0.1
# epsilon_t = 0.4
# Starting from 0.4 it showed patterns
# epsilon_t = -100
# It seems that the epsilon can not be negative since they all generate same curve
# s = generate_FitzHugh_Nagumo(500, dxy_dt=d_FitzHugh_Nagumo)
# vis_data(s.T,"FN0")

s = generate_FitzHugh_Nagumo(500, dxy_dt=d_FitzHugh_Nagumo_, dt_s= 0.25)
vis_data(s.T,"FN1")
#%% discrete time version of FitzHugh Nagumo
# Let this equation be xy_FitzHugh_Nagumo_1
def generate_discrete_FitzHugh_Nagumo(size, delta_t = 0.7):
    x = np.zeros(size + 10000)
    w = np.zeros(size + 10000)
    # epsilon_t_x = np.random.normal(loc = 0, scale = 1, size = size + 1000)
    epsilon_t_x = np.random.normal(loc = 1, scale = 0.2, size = size + 10000)
    # epsilon_t_w = np.random.normal(size = size + 1000)
    for i in range(1,size + 10000):
        x[i] = x[i-1] + delta_t*(x[i-1] - x[i-1]**3 /3 - w[i-1] + epsilon_t_x[i-1])
        w[i] = w[i-1] + delta_t*0.08*(x[i-1] + 0.7 -0.8 * w[i-1])
        # print(f'xt = {x[i-1]}\txt+1 = {x[i]}\twt = {w[i-1]}\twt+1 = {w[i]}\tepsilon = {epsilon_t_x[i-1]}')
    # return x[-N:,], w[-N:,]
    return x, w    

#%%
def noisy_fitzhugh(n=10500, speed=0.7):
    t = np.arange(1, n+1)
    v = np.zeros(n)
    w = np.zeros(n)
    for i in range(n-1):
        delta_v = speed * (v[i] - ((v[i] ** 3)/3) - w[i] + np.random.normal(1, 0.2))
        delta_w = speed * 0.08 * (v[i] + 0.7 - 0.8*w[i])
        v[i+1] = v[i] + delta_v
        w[i+1] = w[i] + delta_w
    return v,w

#%%
def vis_data(ts, titl = ""):
    fig, ax = plt.subplots(figsize = (10, 2.7))
    ax.set_title(titl)
    for i in range(len(ts)):
        ax.plot(ts[i])
        
def wrapper(N, delta_t):
    x, y = generate_discrete_FitzHugh_Nagumo(N, delta_t= delta_t)
    vis_data([x[-N:],y[-N:]], f'#timepoints {N} + 1000, delta_t = {delta_t}')
    return x, y
#%%
x, y = wrapper(500, delta_t= 0.7)
# RuntimeWarning: overflow encountered in scalar power
# Lots of nan
# while i < 100:
# x, y = wrapper(500, delta_t= 0.001)
# x, y = wrapper(3500, delta_t= 0.001)
# x, y = wrapper(500, delta_t= 0.05)
# x, y = wrapper(3500, delta_t= 0.05)
# x, y = wrapper(3500, delta_t= 0.07)
# x, y = wrapper(0, delta_t= 0.07 )
# x, y = wrapper(0, delta_t= 0.01)
    # i+= 1
#%%
v, w = noisy_fitzhugh()
# vis_data([v[1400:2400],w[1400:2400]])
vis_data([v[-500:], w[-500:]])

#%% Generate FitzHugh Nagumo Alex ver
with open('xy_FitzHugh_Nagumo/data.pkl','rb') as fi:
    test = pickle.load(fi)
    
datagen_param = test['datagen_params']
datagen_param['a'] = None
datagen_param['dt_s'] = 0.25
datagen_param['intx'] = None
datagen_param['noise'] = None
datagen_param['noise_T'] = None
datagen_param['r'] = None
datagen_param['delta_t'] = 0.7

xy_FitzHugh_Nagumo_cont = [list(generate_FitzHugh_Nagumo(datagen_param['N'], dxy_dt = d_FitzHugh_Nagumo_, dt_s = datagen_param['dt_s']).T) for i in range(1000)]
# xy_FitzHugh_Nagumo_cont = [[_.T[0], _.T[1]] for _ in np.copy(xy_FitzHugh_Nagumo_cont)]

xy_FitzHugh_Nagumo_disc = [list(generate_discrete_FitzHugh_Nagumo(datagen_param['N'], delta_t= datagen_param['delta_t'])) for i in range(1000)]
# xy_FitzHugh_Nagumo_disc = [[_[0][-500:], _[1][-500:]] for _ in np.copy(xy_FitzHugh_Nagumo_disc)]

data = {'data': xy_FitzHugh_Nagumo_cont, 'datagen_params' : datagen_param}

with open('xy_FitzHugh_Nagumo_cont/data.pkl', 'wb') as fi:
    pickle.dump(data,fi)
    
data = {'data': xy_FitzHugh_Nagumo_disc1, 'datagen_params' : datagen_param}

with open('xy_FitzHugh_Nagumo_disc/data.pkl', 'wb') as fi:
    pickle.dump(data,fi)

#%% Check generated data
with open('xy_FitzHugh_Nagumo_cont/data.pkl','rb') as fi:
    cont = pickle.load(fi)

with open('xy_FitzHugh_Nagumo_disc/data.pkl','rb') as fi:
    disc = pickle.load(fi)