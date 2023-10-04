# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 19:11:05 2023

@author: hoang

Visualise data
"""

import os
os.getcwd()
os.chdir('C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/')
# os.chdir('/home/h_k_linh/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/')
os.getcwd()
# import glob
import numpy as np
# import pandas as pd 
import scipy
# from mergedeep import merge

import pickle
#%% Draw heatmap as Caroline
import matplotlib as mpl
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gs
# import matplotlib.patheffects as pe
# import matplotlib.patches as patches
from matplotlib.collections import LineCollection
# import seaborn as sns
#%%
# Caroline configuration
plt.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['ytick.major.size'] = 5

mpl.rcParams['hatch.color'] = 'white'
mpl.rcParams['hatch.linewidth'] = 2

font = {'fontsize' : 22, 'fontweight' : 'bold', 'fontname' : 'arial'}
# font_data = {'fontsize' : 20, 'fontweight' : 'bold', 'fontname' : 'arial','color':'white'}
font_data = {'fontsize' : 22, 'fontweight' : 'bold', 'fontname' : 'arial','color':'black'}
# plt.plot(x, y, **font)
# texts = annotate_heatmap(im, valfmt="{x:.0f}", threshold=0, **font_data)

#%% Load data
def load_data(sample):
    if 'xy_' in sample:
        sampdir = f'Simulated_data/{sample}'
    elif '500' in sample:
        sampdir = f'Simulated_data/LVextra/{sample}'
    with open(f'{sampdir}/data.pkl', 'rb') as fi:
        data = pickle.load(fi)
    return data
if __name__ == "__main__":
    data = {}
    for sample in os.listdir('Simulated_data'):
        if 'xy_' in sample:
            data[sample] = load_data(sample)
            print(sample)
    for sample in os.listdir('Simulated_data/LVextra'):
        if '500' in sample:
            data[sample] = load_data(sample)
            print(sample)
#%% 
def vis_data(XY, title):
    X = XY[0][0:100]
    Y = XY[1][0:100]
    tmp = np.arange(len(X))
    
    fig, ax = plt.subplots(figsize=(5.25,4.125))
    fig.set_tight_layout(True)
    
    ax.plot(X, color = "blue")
    ax.plot(Y, color = "#ff6600")
    ax.legend(["X","Y"], fontsize = 20)
    ax.set_ylim(-0.25, 2.25)
    ax.set_xlabel('Time index', **font)
    ax.set_ylabel('X and Y value', **font)
    ax.set_title(f'{title}', **font_data)
    ax.scatter(tmp, X, color = "blue", s = 5)
    ax.scatter(tmp, Y, color = "#ff6600", s = 5)

    ax.tick_params(labelsize = 17)
    
    plt.savefig(f"Simulated_data/Figures/data/{title}.svg") 
    plt.show()
    
#%%
def vis_valuedistr(XY, title):
    fig,ax = plt.subplots(1,2, figsize=(10,5), constrained_layout=True)
    
    for i in range(len(ax)):
        if i == 0: color = "blue"; xory = "X"
        else: color = "#ff6600"; xory = "Y"
        counts, bins = np.histogram(XY[i], bins = 50)
        ax[i].hist(bins[:-1], bins, weights=counts, color = color)
        # ax[i].stairs(counts, bins, color = color)
        ax[i].set_xlabel(f'{xory} values', **font)
        ax[i].set_ylabel('Counts', **font)
        ax[i].set_title(f'Value distribution of {xory}', **font_data)
        ax[i].tick_params(labelsize = 17)
    
    fig.suptitle(title, **font_data)
    plt.savefig(f"Simulated_data/Figures/value_dist/{title}.svg") 
    # plt.show() 
    
#%%
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
# separately
def vis_acf(XY, title = ''):
    acf = [autocorrelationcorr_BJ(_) for _ in XY] 
    
    fig,ax = plt.subplots(1,2, figsize=(10,5), constrained_layout=True)
    
    tau = r"$\tau$"
    for i in range(len(ax)):
        if i == 0: color = "blue"; xory = "X"
        else: color = "#ff6600"; xory = "Y"
        lines = LineCollection([[(_,0),(_,acf[i][_])] for _ in np.arange(len(acf[i]))],
                               colors = color, linewidths = 0.5, linestyle = 'solid')
        ax[i].add_collection(lines)#, s = 5
        ax[i].scatter(np.arange(len(acf[i])), acf[i], color = color, s = 1.75)
        ax[i].set_xlabel(f'Time lag ({tau})', **font)
        ax[i].set_ylabel('autocorrelation', **font)
        ax[i].set_title(f'Autocorrelation in {xory}', **font_data)
        ax[i].tick_params(labelsize = 17)
    
    fig.suptitle(title, **font_data)
    plt.savefig(f"Simulated_data/Figures/autocorrelation/{title}.svg") 
    # plt.show() 
    
#%% Power spectrum
def vis_powerspectrum(XY, title):
    acf = [autocorrelationcorr_BJ(_) for _ in XY] 
    
    Amp = [scipy.fft.fft(_) for _ in acf]
    f = scipy.fft.fftfreq(len(acf[0]))
    
    fig,ax = plt.subplots(1,2, figsize=(10,5), constrained_layout=True)
    
    for i in range(len(ax)):
        if i == 0: color = "blue"; xory = "X"
        else: color = "#ff6600"; xory = "Y"
        lines = LineCollection([[(f[_],0),(f[_],Amp[i][_])] for _ in np.arange(len(f))],
                               colors = color, linewidths = 0.5, linestyle = 'solid')
        ax[i].add_collection(lines)#, s = 5
        ax[i].scatter(f, Amp[i], color = color, s = 1.75)
        ax[i].set_xlabel("Frequency (f)", **font)
        ax[i].set_ylabel('Power spectral density', **font)
        ax[i].set_title(f'Power spectrum in {xory}', **font_data)
        ax[i].tick_params(labelsize = 17)
    
    fig.suptitle(title, **font_data)
    plt.savefig(f"Simulated_data/Figures/powerspectrum/{title}.svg") 
    # plt.show() 

#%%
from scipy.spatial.distance import pdist, squareform
def recurrence_matrix(state, eps=0.10, steps=10):
    # https://github.com/laszukdawid/recurrence-plot/blob/master/plot_recurrence.py
    # https://www.kaggle.com/code/tigurius/recuplots-and-cnns-for-time-series-classification
    d = pdist(state[:,None])
    d = np.floor(d/eps)
    d[d>steps] = steps
    Z = squareform(d)
    return Z
def vis_recurrence_plot(XY, title, eps=0.10, steps=10):
    recmat = [recurrence_matrix(_, eps = eps, steps = steps) for _ in XY] 
    
    fig,ax = plt.subplots(1,2, figsize=(12,6), constrained_layout=True)
    obj = []
    cbar = []
    for i in range(len(ax)):
        if i == 0: cmap = mpl.cm.winter; xory = "X"
        else: cmap = mpl.cm.autumn; xory = "Y"
        
        obj.append(ax[i].imshow(recmat[i], cmap = cmap))#
        cbar.append(fig.colorbar(obj[i], ax=ax[i], anchor=(0, 0.5), shrink=0.75))
        cbar[i].ax.tick_params(labelsize=17)
        ax[i].set_xlabel("Time", **font)
        ax[i].set_ylabel('Time', **font)
        ax[i].set_title(f'Recurrence plot of {xory}, eps={eps}, steps = {steps}', **font_data)
        ax[i].tick_params(labelsize = 17)
    
    fig.suptitle(title, **font_data)
    plt.savefig(f"Simulated_data/Figures/recurrence_plot/{title}_{eps}_{steps}.svg") 
    # plt.show() 

# vis_recurrence_plot(data['xy_sinewn']['data'][0], 'xy_ar_u', eps=0.25, steps=10)
 #%% 
if __name__ == "__main__":    
    for key, val in data.items():
        vis_data(val['data'][0], key)
        # vis_valuedistr(val['data'][0], key)
        # vis_acf(val['data'][0], key)
        # vis_powerspectrum(val['data'][0], key)
        # vis_recurrence_plot(val['data'][0], key, eps=0.01, steps=20)