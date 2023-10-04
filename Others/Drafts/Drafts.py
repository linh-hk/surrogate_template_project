#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 23:09:25 2023

@author: h_k_linh
"""
    #%% Visualising after running surrogate tests
        #%%%Load data from server
    # fi = 'qsub_itered_results/results_Generated_data.pkl'
    fi = 'results_firstrun_Generated_data.pkl'
    dill.load_session(fi)
    
    datalist = ['ts_ar_u', 'ts_ar_u2', 'xy_ar', 'xy_logistic', 'xy_sinewn', 'xy_sine_intmt', 'xy_coinflip', 'xy_noise_wv_kurtosis', 'xy_chaoticLV'] # , 'ts_uni_logistic','xy_Caroline_CH', 'xy_Caroline_LV']
    datalist1 = [f'qsub_itered_results/results_{i}.pkl' for i in datalist]
    results = {}
    for fi in datalist1:
        with open(fi, 'rb') as file:
            # Call load method to deserialze
            results[fi] = pickle.load(file)

    #%%% visualise each data
data = all_models.xy_Caroline_CH_competitive.series[0]

fig, ax = plt.subplots(figsize = (7, 2.2))
ax.set_title("Growth mass of two species competing for same resource")
ax.plot(data[0], color = "blue", linewidth =1)
ax.plot(data[1], color = "#ffc000", linewidth = 1)
ax.legend(["X","Y"])
vis_data(data)
vis_data(data[:;0:100])
data = [_[0:100] for _ in data]
#%% Vis surrogates test
import os 
# os.chdir('/home/h_k_linh/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/')
import numpy as np
import pandas as pd
import pickle
# with open('xy_Caroline_LV_mutualistic/data.pkl','rb') as fi:
#     data = pickle.load(fi)
# # data = all_models.xy_Caroline_LV_mutualistic.series[0]
# data = [_[0:100] for _ in data['data'][0]]
# # vis_data(data)
from scipy.stats import pearsonr
# pearsonr(data[0], data[1]).statistic

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import animation
from functools import partial
from matplotlib import patches

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
with open('xy_Caroline_CH_mutualistic/data.pkl', 'rb') as fi:
    dataori = pickle.load(fi)
     
data = dataori['data'][0]
data = [_[0:100] for _ in dataori['data'][0]]

from pyunicorn.timeseries import surrogates #for iaaft
def get_iaaft_surrogates(timeseries, n_surr=99, n_iter=200):
    print("\t\tCreating random phase iaaft surrogates")
    # prebuilt algorithm in pyunicorn so dont have to create empty matrix
    results = []
    i = 0
    while i < n_surr:
        obj = surrogates.Surrogates(original_data=timeseries.reshape(1,-1), silence_level=2)
        surr = obj.refined_AAFT_surrogates(original_data=timeseries.reshape(1,-1), n_iterations=n_iter, output='true_spectrum')
        surr = surr.ravel()
        if not np.isnan(np.sum(surr)):
            results.append(surr)
            i += 1
    return np.array(results).T

y_surr_randphase = get_iaaft_surrogates(data[1], n_surr=99, n_iter=200)
x_surr_randphase = get_iaaft_surrogates(data[0], n_surr=99, n_iter=200)
pcor_record_randphase_x = np.zeros(x_surr_randphase.shape[1])
for i in range(0, x_surr_randphase.shape[1]):
    pcor_record_randphase_x[i] = round(float(pearsonr(x_surr_randphase[:,i], data[1]).statistic), 3)
        
# vis_data(data)
# for i in range(99): 
#     vis_data([data[0],y_surr[:,i]])

import pandas as pd
def correlation_Pearson(x, y):
    x = pd.Series(x)
    y = pd.Series(y)
    return x.corr(y)
correlation_Pearson(data[0], data[1])

from scipy.stats import pearsonr
# pearsonr(data[0], data[1]).statistic

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import animation
from functools import partial
from matplotlib import patches

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


tmp = np.arange(len(data[0]))
pcor = round(float(pearsonr(data[0], data[1]).statistic), 3)

fig, ax = plt.subplots(figsize=(5,5))
fig.set_tight_layout(True)
ax.plot(data[0], color = "blue")
ax.plot(data[1], color = "#ff6600")
ax.set_ylim(6.9,7.15)
ax.legend(["X",f"Original Y"], fontsize = 20, loc = "upper right")
ax.set_xlabel('Time point (t)', fontsize = 22, fontweight = 'bold', fontname = 'arial')
ax.set_ylabel('X and Y value', fontsize = 22, fontweight = 'bold', fontname = 'arial')

ax.scatter(tmp, data[0], color = "blue", s = 5)
ax.scatter(tmp, data[1], color = "#ff6600", s = 5)

ax.tick_params(labelsize = 17)
ax.text(0, 7.1525, r'$\rho =$' + f"{pcor}", fontsize = 24, color = 'green')
# plt.savefig(f"C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/COURSES/BIOS0010 Biosciences Research Skills/xyori4.svg")
# C:/Users/hoang , /home/h_k_linh
plt.savefig(f"/home/h_k_linh/OneDrive/Desktop/UCL_MRes_Biosciences_2022/COURSES/BIOS0010 Biosciences Research Skills/Assignment2 Video log/xyori.svg")

pcor_record = np.zeros(99)

fig, ax = plt.subplots(figsize=(5,5))

writer = animation.PillowWriter()
with writer.saving(fig, f"C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/COURSES/BIOS0010 Biosciences Research Skills/Assignment2 Video log/surrogates.gif", 180):
    for i in range(99):
        # pcor_record[i] = round(float(pearsonr(data[0], y_surr[:,i]).statistic), 3)
        #
        ax.clear()
        # 
        ax.plot(data[0], color = "blue")
        ax.plot(y_surr[:,i], color = "orange")
        ax.set_ylim(6.9,7.15)
        ax.legend(["X",f"Y surrogate #{i}"], fontsize = 17, loc = "upper right")
        ax.scatter(tmp, data[0], color = "blue", s = 3)
        ax.scatter(tmp, y_surr[:,i], color = "orange", s = 3)
        ax.tick_params(labelsize = 15)
        ax.text(0, 7.125, r'$\rho =$' + f"{pcor_record[i]}", fontsize = 19, color = 'red')

        writer.grab_frame()
        
for i in range(4):
    # i=98
    # pcor_record[i] = round(float(pearsonr(data[0], y_surr[:,i]).statistic), 3)
    #
    # ax.clear()
    # 
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(data[0], color = "blue")
    ax.plot(y_surr[:,i], color = "#ff6600")
    ax.set_ylim(6.9,7.15)
    ax.legend(["X",f"Y surrogate #{i+1}"], fontsize = 20, loc = "upper right")
    ax.set_xlabel('Time point (t)', fontsize = 22, fontweight = 'bold', fontname = 'arial')
    ax.set_ylabel('X and Y value', fontsize = 22, fontweight = 'bold', fontname = 'arial')
    ax.scatter(tmp, data[0], color = "blue", s = 3)
    ax.scatter(tmp, y_surr[:,i], color = "#ff6600", s = 3)
    ax.tick_params(labelsize = 17)
    ax.text(0, 6.91, r'$\rho =$' + f"{pcor_record[i]}", fontsize = 24, color = 'red')
    # plt.show()
    plt.savefig(f"C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/COURSES/BIOS0010 Biosciences Research Skills/xysurr{i+1}.svg")

# pcor_record = np.append(pcor_record, pcor)
drawthis = np.absolute(pcor_record)
hist_bins = np.arange(min(drawthis),max(drawthis)+0.005, 0.002)
fig, ax = plt.subplots(figsize=(5,5))
fig.set_tight_layout(True)
ax.hist(drawthis, bins = hist_bins, density = False, color = 'red')
ax.tick_params(labelsize = 15)
plt.yticks([0, 1, 2, 3, 4])
ax.set_xlabel('Absolute ' + r'$\rho$', **font)
ax.set_ylabel('Frequencies', **font)
ax.axvline(x=pcor, color='green', linestyle='dashed', linewidth=.5)
ax.text(pcor*0.65, 3, r'$\rho =$' +  f"{pcor}", color = "green", fontsize = 19, fontstyle = 'italic')
plt.savefig('C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/COURSES/BIOS0010 Biosciences Research Skills/Assignment2 Video log/pcor_dist.svg')

#%% Use animation.FuncAnimation
def draw_each_frame(frame_num, xstar, y_surr, pcor_record):
    i = frame_num
    tmp = np.arange(len(xstar))
    # pcor_record[i] = round(float(pearsonr(xstar, y_surr[:,i]).statistic), 3)

    ax.clear()
    
    ax.plot(data[0], color = "blue")
    ax.plot(y_surr[:,i], color = "#ff6600")
    ax.set_ylim(6.9,7.15)
    ax.legend(["X",f"Y surrogate #{i+1}"], fontsize = 20, loc = "upper right")
    ax.set_xlabel('Time point (t)', fontsize = 22, fontweight = 'bold', fontname = 'arial')
    ax.set_ylabel('X and Y value', fontsize = 22, fontweight = 'bold', fontname = 'arial')

    # ax.scatter(tmp, xstar, color = "blue", s = 5)
    ax.scatter(tmp, y_surr[:,i], color = "#ff6600", s = 5)

    ax.tick_params(labelsize = 17)
    ax.text(0, 7.1525, r'$\rho =$' + f"{pcor_record[i]}", fontsize = 24, color = 'red')
    return ax

fig, ax = plt.subplots(figsize=(5,5))
fig.set_tight_layout(True)
anim = animation.FuncAnimation(fig, draw_each_frame, fargs = (data[0], y_surr, pcor_record), frames = 99)# , blit = True, interval = 500
# read about blit https://stackoverflow.com/questions/62784710/the-animation-function-must-return-a-sequence-of-artist-objects
# the interval will mean smt if fps is not decided in save function below (open the anim object and look for save method)
# anim.save('C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/COURSES/BIOS0010 Biosciences Research Skills/Assignment2 Video log/test.mp4', writer = 'ffmpeg', fps = 2)
anim.save('C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/COURSES/BIOS0010 Biosciences Research Skills/Assignment2 Video log/test.gif', writer = 'Pillow', fps = 4)
# MovieWriter ffmpeg unavailable; using Pillow instead.
# Download ffmpeg 
# conda install -c conda-forge ffmpeg
#%% Vis randphase or twin
def draw_each_frame(frame_num, xstar, y_surr, pcor_record):
    i = frame_num
    tmp = np.arange(len(xstar))
    # pcor_record[i] = round(float(pearsonr(xstar, y_surr[:,i]).statistic), 3)

    ax.clear()
    ax.set_xlim(0,100)
    
    ax.plot(xstar, color = "blue")
    ax.plot(y_surr[:,i], color = "#ff6600")
    ax.set_ylim(6.9,7.15)
    ax.legend(["X",f"Y surrogate #{i+1}"], fontsize = 20, loc = "upper right")
    ax.set_xlabel('Time point (t)', fontsize = 22, fontweight = 'bold', fontname = 'arial')
    # ax.set_ylabel('Y value', fontsize = 22, fontweight = 'bold', fontname = 'arial')
    ax.set_ylabel('X and Y value', fontsize = 22, fontweight = 'bold', fontname = 'arial')

    ax.scatter(tmp, xstar, color = "blue", s = 5)
    ax.scatter(tmp, y_surr[:,i], color = "#ff6600", s = 5)

    ax.tick_params(labelsize = 17)
    ax.text(0, 7.1525, r'$\rho =$' + f"{pcor_record[i]}", fontsize = 24, color = 'red')
    return ax

def draw_each_frame_surrx(frame_num, ystar, x_surr, pcor_record):
    i = frame_num
    tmp = np.arange(len(ystar))
    # pcor_record[i] = round(float(pearsonr(xstar, y_surr[:,i]).statistic), 3)

    ax.clear()
    ax.set_xlim(0,100)
    
    ax.plot(x_surr[:,i], color = "blue")
    ax.plot(ystar, color = "#ff6600")
    ax.set_ylim(6.9,7.15)
    ax.legend([f"X surrogate #{i+1}", "Y"], fontsize = 20, loc = "upper right")
    ax.set_xlabel('Time point (t)', fontsize = 22, fontweight = 'bold', fontname = 'arial')
    # ax.set_ylabel('X value', fontsize = 22, fontweight = 'bold', fontname = 'arial')
    ax.set_ylabel('X and Y value', fontsize = 22, fontweight = 'bold', fontname = 'arial')

    ax.scatter(tmp, x_surr[:,i], color = "blue", s = 5)
    ax.scatter(tmp, ystar, color = "#ff6600", s = 5)

    ax.tick_params(labelsize = 17)
    ax.text(0, 7.1525, r'$\rho =$' + f"{pcor_record[i]}", fontsize = 24, color = 'red')
    return ax

def draw_each_frame_one_y(frame_num, xstar, y_surr, pcor_record):
    i = frame_num
    tmp = np.arange(len(xstar))
    # pcor_record[i] = round(float(pearsonr(xstar, y_surr[:,i]).statistic), 3)

    ax.clear()
    ax.set_xlim(0,100)
    
    # ax.plot(xstar, color = "blue")
    ax.plot(y_surr[:,i], color = "#ff6600")
    ax.set_ylim(6.9,7.15)
    ax.legend([f"Y surrogate #{i+1}"], fontsize = 20, loc = "upper right")
    ax.set_xlabel('Time point (t)', fontsize = 22, fontweight = 'bold', fontname = 'arial')
    ax.set_ylabel('Y value', fontsize = 22, fontweight = 'bold', fontname = 'arial')
    # ax.set_ylabel('X and Y value', fontsize = 22, fontweight = 'bold', fontname = 'arial')

    # ax.scatter(tmp, xstar, color = "blue", s = 5)
    ax.scatter(tmp, y_surr[:,i], color = "#ff6600", s = 5)

    ax.tick_params(labelsize = 17)
    # ax.text(0, 7.1525, r'$\rho =$' + f"{pcor_record[i]}", fontsize = 24, color = 'red')
    return ax

def draw_each_frame_one_x(frame_num, ystar, x_surr, pcor_record):
    i = frame_num
    tmp = np.arange(len(ystar))
    # pcor_record[i] = round(float(pearsonr(xstar, y_surr[:,i]).statistic), 3)

    ax.clear()
    ax.set_xlim(0,100)
    
    ax.plot(x_surr[:,i], color = "blue")
    # ax.plot(ystar, color = "#ff6600")
    ax.set_ylim(6.9,7.15)
    ax.legend([f"X surrogate #{i+1}"], fontsize = 20, loc = "upper right")
    ax.set_xlabel('Time point (t)', fontsize = 22, fontweight = 'bold', fontname = 'arial')
    ax.set_ylabel('X value', fontsize = 22, fontweight = 'bold', fontname = 'arial')
    # ax.set_ylabel('X and Y value', fontsize = 22, fontweight = 'bold', fontname = 'arial')

    ax.scatter(tmp, x_surr[:,i], color = "blue", s = 5)
    # ax.scatter(tmp, ystar, color = "#ff6600", s = 5)

    ax.tick_params(labelsize = 17)
    # ax.text(0, 7.1525, r'$\rho =$' + f"{pcor_record[i]}", fontsize = 24, color = 'red')
    return ax
#%%
fig, ax = plt.subplots(figsize=(5,5))
fig.set_tight_layout(True)

anim_y = animation.FuncAnimation(fig, draw_each_frame, fargs = (data[0], y_surr_randphase, pcor_record_randphase_y), frames = 99)
# C:/Users/hoang , /home/h_k_linh
anim_y.save('/home/h_k_linh/OneDrive/Desktop/UCL_MRes_Biosciences_2022/COURSES/BIOS0010 Biosciences Research Skills/Assignment2 Video log/surr_randphase_y.gif', writer = 'pillow', fps = 8)
anim_x = animation.FuncAnimation(fig, draw_each_frame_surrx, fargs = (data[1], x_surr_randphase, pcor_record_randphase_x), frames = 99)
anim_x.save('/home/h_k_linh/OneDrive/Desktop/UCL_MRes_Biosciences_2022/COURSES/BIOS0010 Biosciences Research Skills/Assignment2 Video log/surr_randphase_x.gif', writer = 'pillow', fps = 8)
#%% Vis twin
from cluster_xory import get_twin_wrapper

y_surr_twin = get_twin_wrapper(data[1])
xtwin = data[0][:y_surr_twin.shape[0]]

x_surr_twin = get_twin_wrapper(data[0])
ytwin = data[1][:x_surr_twin.shape[0]]

from scipy.stats import pearsonr
# pcor_twin = pearsonr(data[0], data[1]).statistic
pcor_record_twin_y = np.zeros(y_surr_twin.shape[1])
for i in range(0,y_surr_twin.shape[1]):
        pcor_record_twin_y[i] = round(float(pearsonr(xtwin, y_surr_twin[:,i]).statistic), 3)
        
pcor_record_twin_x = np.zeros(x_surr_twin.shape[1])
for i in range(0,x_surr_twin.shape[1]):
        pcor_record_twin_x[i] = round(float(pearsonr(x_surr_twin[:,i], ytwin).statistic), 3)

#%%
fig, ax = plt.subplots(figsize=(5,5))
fig.set_tight_layout(True)

anim_y = animation.FuncAnimation(fig, draw_each_frame, fargs = (xtwin, y_surr_twin, pcor_record_twin_y), frames = 99)
anim_y.save('/home/h_k_linh/OneDrive/Desktop/UCL_MRes_Biosciences_2022/COURSES/BIOS0010 Biosciences Research Skills/Assignment2 Video log/surr_twin_y.gif', writer = 'pillow', fps = 8)

anim_x = animation.FuncAnimation(fig, draw_each_frame_surrx, fargs = (ytwin, x_surr_twin, pcor_record_twin_x), frames = 99)
anim_x.save('/home/h_k_linh/OneDrive/Desktop/UCL_MRes_Biosciences_2022/COURSES/BIOS0010 Biosciences Research Skills/Assignment2 Video log/surr_twin_x.gif', writer = 'pillow', fps = 8)

#%%
fig, ax = plt.subplots(figsize=(5,5))
fig.set_tight_layout(True)

anim_y = animation.FuncAnimation(fig, draw_each_frame_one_y, fargs = (data[0], y_surr_randphase, pcor_record_randphase_y), frames = 99)
# C:/Users/hoang , /home/h_k_linh
anim_y.save('C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/COURSES/BIOS0010 Biosciences Research Skills/Assignment2 Video log/surr_y_randphase.gif', writer = 'pillow', fps = 8)
anim_x = animation.FuncAnimation(fig, draw_each_frame_one_x, fargs = (data[1], x_surr_randphase, pcor_record_randphase_x), frames = 99)
anim_x.save('C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/COURSES/BIOS0010 Biosciences Research Skills/Assignment2 Video log/surr_x_randphase.gif', writer = 'pillow', fps = 8)

anim_y = animation.FuncAnimation(fig, draw_each_frame_one_y, fargs = (xtwin, y_surr_twin, pcor_record_twin_y), frames = 99)
anim_y.save('C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/COURSES/BIOS0010 Biosciences Research Skills/Assignment2 Video log/surr_y_twin.gif', writer = 'pillow', fps = 8)

anim_x = animation.FuncAnimation(fig, draw_each_frame_one_x, fargs = (ytwin, x_surr_twin, pcor_record_twin_x), frames = 99)
anim_x.save('C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/COURSES/BIOS0010 Biosciences Research Skills/Assignment2 Video log/surr_x_twin.gif', writer = 'pillow', fps = 8)
#%% Vis tts

def choose_r(n):
    delta = (n/4 + 1) % 20
    return int(n/4 - delta)
# use data load from COURSES/COURSES/BIOS0010 Biosciences Research Skills/Assignment2 Video log/surr_series.spydata
# choose_r(100) = 19

def tts(x,y,r): # from old scripts
    # time shift
    t = len(x); #number of time points in orig data
    xstar = x[r:(t-r)]; # middle half of data - truncated original X0
    tstar = len(xstar); # length of truncated series
    ystar = y[r:(t-r)]; # truncated original Y0
    t0 = r;             # index of Y0 in matrix of surrY
    
    y_surr = np.tile(ystar,[2*r+1, 1]) # Create empty matrix of surrY
    y_pos = np.tile(ystar,[2*r+1, 1])
    print("\t\t\tCreating tts surrogates")
    #iterate through all shifts from -r to r
    for shift in np.arange(t0 - r, t0 + r + 1):
        # print(f'shift-t0:{shift-t0}\ty[shift:(shift+tstar)]: {shift}:{shift+tstar}')
        y_surr[shift-t0] = y[shift:(shift+tstar)];
        y_pos[shift-t0] = np.arange(shift, shift+tstar)
    return xstar, y_surr, y_pos;
xstar, y_surr_tts, pos = tts(data[0],data[1],19)
# xpos = y_pos[0,:]

ystar, x_surr_tts, pos = tts(data[1],data[0],19)
# y_pos = x_pos[0,:]

from scipy.stats import pearsonr
pearsonr(data[0], data[1]).statistic
pcor_record_tts_y = np.zeros(y_pos.shape[0]) # code old tts does not return .T and ystar is in y_surr_tts
for i in range(0,pos.shape[0]):
        pcor_record_tts_y[i] = round(float(pearsonr(xstar, y_surr_tts[i,:]).statistic), 3)
pcor_record_tts_x = np.zeros(pos.shape[0]) # code old tts does not return .T and ystar is in y_surr_tts
for i in range(0,pos.shape[0]):
        pcor_record_tts_x[i] = round(float(pearsonr(x_surr_tts[i,:], ystar).statistic), 3)


#%%
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import animation
from functools import partial
from matplotlib import patches

def draw_each_frame_tts(frame_num, ax, data, y_surr, y_pos, pcor_record):
    i = frame_num
    if i == 0:
        leg = "Original Y"
        colorr = "green"
    else:
        leg = "Y surrogate"
        colorr = "red"
    # pcor_record[i] = round(float(pearsonr(data[0], y_surr[:,i]).statistic), 3)
    ax[1].clear()
    ax[1].set_xlim(0,100)
    #         # 
    ax[1].plot(data[1], color = "#ff6600")
    ax[1].set_ylim(6.92,7.13)
    
    ax[1].set_xlabel('Time point (t)', fontsize = 22, fontweight = 'bold', fontname = 'arial')
    ax[1].tick_params(labelsize = 17)
            
    ax[1].plot(pos[i,:],y_surr[i,:], color = "red")
    ax[1].scatter(pos[i,:], y_surr[i,:], color = "red", s = 5)
    ax[1].legend(["Y",leg], fontsize = 20, loc = "upper left")
    #         #
    rect = patches.Rectangle((pos[i,0], 6.925), 62, 0.2, edgecolor='r', facecolor='none')#, linewidth=1
    ax[1].add_patch(rect)
    #         # ax.tick_params(labelsize = 12)
    ax[1].text(pos[i,0]+12, 7.1325, r'$\rho =$' + f"{pcor_record[i]}", fontsize = 24, color = colorr)
    # ax[1].tick_params(labelsize = 17)
    return ax

fig, ax = plt.subplots(1, 2, figsize=(9,4.8))
fig.set_tight_layout(True)
ax[0].set_xlim(0,100)
# ax[1].set_xlim(0,100)

# pcor = round(float(pearsonr(xstar, y_surr[0,:]).statistic), 3)
ax[0].plot(data[0], color = "blue")#, linewidth = 1
ax[0].legend(["X"], fontsize = 20, loc = "upper right")
ax[0].set_ylim(6.92,7.13)
ax[0].set_xlabel('Time point (t)', fontsize = 22, fontweight = 'bold', fontname = 'arial')
# ax[0].set_ylabel('X and Y value', fontsize = 22, fontweight = 'bold', fontname = 'arial')
ax[0].tick_params(labelsize = 17)

# ax[1].plot(data[1], color = "#ff6600")#, linewidth = 1
# ax[1].set_ylim(6.9,7.15)
# ax[1].legend(["Y"], fontsize = 20, loc = "upper left")
# ax[1].set_xlabel('Time point (t)', fontsize = 22, fontweight = 'bold', fontname = 'arial')
# ax[1].tick_params(labelsize = 17)

ax[0].plot(pos[0,:], xstar, color = "red")#, linewidth = 1.1
ax[0].scatter(pos[0,:], xstar, color = "red", s = 5)
rect = patches.Rectangle((pos[0,0], 6.925), 62, 0.2, edgecolor='r', facecolor='none')#, linewidth=1
ax[0].add_patch(rect)

# 0, 18, 19, 20, 21
ax = draw_each_frame_tts(21, ax, data, y_surr_tts, pos, pcor_record_tts_y)
plt.savefig(f"plots/tts_21.svg")
    
# anim_y = animation.FuncAnimation(fig, draw_each_frame_tts, fargs = (ax, data, y_surr_tts, pos, pcor_record_tts_y), frames = 39)# , blit = True, interval = 500
# # anim.save('C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/COURSES/BIOS0010 Biosciences Research Skills/Assignment2 Video log/test_tts.mp4', writer = 'ffmpeg', fps = 2)
# anim_y.save('C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/COURSES/BIOS0010 Biosciences Research Skills/Assignment2 Video log/surr_tts_y.gif', writer = 'pillow', fps = 8)
# # anim.save('/home/h_k_linh/OneDrive/Desktop/UCL_MRes_Biosciences_2022/COURSES/BIOS0010 Biosciences Research Skills/Assignment2 Video log/surr_tts.mp4', writer = 'ffmpeg', fps = 8)
# # anim_y.save('/home/h_k_linh/OneDrive/Desktop/UCL_MRes_Biosciences_2022/COURSES/BIOS0010 Biosciences Research Skills/Assignment2 Video log/surr_tts_y.gif', writer = 'pillow', fps = 8)
#%%

def draw_each_frame_tts_surrx(frame_num, ax, data, x_surr, x_pos, pcor_record):
    i = frame_num
    if i == 0:
        leg = "Original X"
        colorr = "green"
    else:
        leg = "X surrogate"
        colorr = "red"
    # pcor_record[i] = round(float(pearsonr(data[0], y_surr[:,i]).statistic), 3)
    ax[0].clear()
    ax[0].set_xlim(0,100)
    #         # 
    ax[0].plot(data[0], color = "blue")
    ax[0].set_ylim(6.9,7.15)
    
    ax[0].set_xlabel('Time point (t)', fontsize = 22, fontweight = 'bold', fontname = 'arial')
    # ax[0].set_ylabel('X and Y value', fontsize = 22, fontweight = 'bold', fontname = 'arial')
    ax[0].tick_params(labelsize = 17)
         
    ax[0].plot(pos[i,:],x_surr[i,:], color = "red")
    ax[0].scatter(pos[i,:], x_surr[i,:], color = "red", s = 5)
    ax[0].legend(["X", leg], fontsize = 20, loc = "upper left")
    #         #
    rect = patches.Rectangle((x_pos[i,0], 6.925), 62, 0.2, edgecolor='r', facecolor='none')#, linewidth=1
    ax[0].add_patch(rect)
    #         # ax.tick_params(labelsize = 12)
    ax[0].text(pos[i,0]+12, 7.1525, r'$\rho =$' + f"{pcor_record[i]}", fontsize = 24, color = colorr)
    # ax[1].tick_params(labelsize = 17)
    return ax

fig, ax = plt.subplots(1, 2, figsize=(9, 5))
fig.set_tight_layout(True)
# ax[0].set_xlim(0,100)
ax[1].set_xlim(0,100)

# pcor = round(float(pearsonr(xstar, y_surr[0,:]).statistic), 3)
# ax[0].plot(data[0], color = "blue")#, linewidth = 1
# ax[0].legend(["X"], fontsize = 20, loc = "upper right")
# ax[0].set_ylim(6.9,7.15)
# ax[0].set_xlabel('Time point (t)', fontsize = 22, fontweight = 'bold', fontname = 'arial')
# # ax[0].set_ylabel('X and Y value', fontsize = 22, fontweight = 'bold', fontname = 'arial')
# ax[0].tick_params(labelsize = 17)

ax[1].plot(data[1], color = "#ff6600")#, linewidth = 1
ax[1].set_ylim(6.9,7.15)
ax[1].legend(["Y"], fontsize = 20, loc = "upper left")
ax[1].set_xlabel('Time point (t)', fontsize = 22, fontweight = 'bold', fontname = 'arial')
ax[1].tick_params(labelsize = 17)

ax[1].plot(pos[0,:], ystar, color = "red")#, linewidth = 1.1
ax[1].scatter(pos[0,:], ystar, color = "red", s = 5)
rect = patches.Rectangle((pos[0,0], 6.925), 62, 0.2, edgecolor='r', facecolor='none')#, linewidth=1
ax[1].add_patch(rect)
    
anim_x = animation.FuncAnimation(fig, draw_each_frame_tts_surrx, fargs = (ax, data, x_surr_tts, pos, pcor_record_tts_x), frames = 39)# , blit = True, interval = 500
# anim.save('C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/COURSES/BIOS0010 Biosciences Research Skills/Assignment2 Video log/test_tts.mp4', writer = 'ffmpeg', fps = 2)
anim_x.save('C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/COURSES/BIOS0010 Biosciences Research Skills/Assignment2 Video log/surr_tts_x.gif', writer = 'pillow', fps = 8)
# anim.save('/home/h_k_linh/OneDrive/Desktop/UCL_MRes_Biosciences_2022/COURSES/BIOS0010 Biosciences Research Skills/Assignment2 Video log/surr_tts.mp4', writer = 'ffmpeg', fps = 8)
# anim_x.save('/home/h_k_linh/OneDrive/Desktop/UCL_MRes_Biosciences_2022/COURSES/BIOS0010 Biosciences Research Skills/Assignment2 Video log/surr_tts_x.gif', writer = 'pillow', fps = 8)

#%%
line, = ax.plot(data[1])

def animate(i):
    line.set_data(range(100),y_surr[:,i])  # update the data.
    return line,

writer = animation.PillowWriter()

ani = animation.FuncAnimation(
    fig, animate, interval=1000, blit=True, save_count=50)


import aspose.words as aw

def vis_data(ts, titl = ""):
    fig, ax = plt.subplots(figsize = (5, 5))
    ax.set_title(titl)
    for i in range(len(ts)):
        ax.plot(ts[i])
    ax.legend(["X","Y"])

for i in range(len(datalist)):
    vis_data(globals()[datalist[i]],datalist[i])
# vis_data(ts_ar_u, 'ts_ar_u')

fig, ax = plt.subplots(figsize = (5, 5))
ax.scatter(data[0], data[1], color = 'green')

doc = aw.Document()
builder = aw.DocumentBuilder(doc)

shape = builder.insert_image("plots/data_time.png")
shape.image_data.save("plots/data_time.svg")
#%% Draw beautiful series to compare models for slides
fig, axes = plt.subplots(2,4, figsize = (21,11))
fig.tight_layout(pad=9.0)
axes[0,0].plot(all_models.xy_Caroline_LV_competitive.series[0][0][0:100], color = "blue")
axes[0,0].plot(all_models.xy_Caroline_LV_competitive.series[0][1][0:100], color = "red")
axes[0,0].set_title("LV equally competitive", fontsize = 20)
axes[0,0].set_xlabel("Time (t)",fontsize = 20)
axes[0,0].set_ylabel("Species growth mass", fontsize = 20)
axes[0,0].legend(["Species 1", "Species 2"], fontsize = 20, loc = "center right")
axes[0,0].tick_params(labelsize =12)

axes[0,1].plot(all_models.xy_Caroline_LV_asym_competitive.series[0][0][0:100], color = "blue")
axes[0,1].plot(all_models.xy_Caroline_LV_asym_competitive.series[0][1][0:100], color = "red")
axes[0,1].set_title("LV unequally competitive", fontsize = 20)
axes[0,1].set_xlabel("Time (t)", fontsize = 20)
axes[0,1].set_ylabel("Species growth mass", fontsize = 20)
axes[0,1].legend(["Species 1", "Species 2"], fontsize = 20, loc = "center right")
axes[0,1].tick_params(labelsize =12)

axes[0,2].plot(all_models.xy_Caroline_LV_mutualistic.series[0][0][0:100], color = "blue")
axes[0,2].plot(all_models.xy_Caroline_LV_mutualistic.series[0][1][0:100], color = "red")
axes[0,2].set_title("LV mutualistic", fontsize = 20)
axes[0,2].set_xlabel("Time (t)", fontsize = 20)
axes[0,2].set_ylabel("Species growth mass", fontsize = 20)
axes[0,2].legend(["Species 1", "Species 2"], fontsize = 20, loc = "upper right")
axes[0,2].tick_params(labelsize =12)

axes[0,3].plot(all_models.xy_Caroline_LV_predprey.series[0][0][0:100], color = "blue")
axes[0,3].plot(all_models.xy_Caroline_LV_predprey.series[0][1][0:100], color = "red")
axes[0,3].set_title("LV predator prey", fontsize = 20)
axes[0,3].set_xlabel("Time (t)", fontsize = 20)
axes[0,3].set_ylabel("Species growth mass", fontsize = 20)
axes[0,3].legend(["Species 1", "Species 2"], fontsize = 20, loc = "upper right")
axes[0,3].tick_params(labelsize =12)

axes[1,0].plot(all_models.xy_FitzHugh_Nagumo.series[0][0][0:100], color = "blue")
axes[1,0].plot(all_models.xy_FitzHugh_Nagumo.series[0][1][0:100], color = "red")
axes[1,0].set_title("Excitable systems \n(FitzHugh Nagumo)", fontsize = 20)
axes[1,0].set_xlabel("Time (t)",fontsize = 20)
axes[1,0].set_ylabel("Membrane voltage (v) \nand recovery variable (w)", fontsize = 20)
axes[1,0].legend(["v", "w"], fontsize = 20, loc = "center right")
axes[1,0].tick_params(labelsize =12)

axes[1,1].plot(all_models.xy_ar_u.series[0][0][0:100], color = "blue")
axes[1,1].plot(all_models.xy_ar_u.series[0][1][0:100], color = "red")
axes[1,1].set_title("Unidirectional coupled \nautoregressive processes", fontsize = 20)
axes[1,1].set_xlabel("Time (t)", fontsize = 20)
axes[1,1].set_ylabel("X and Y values", fontsize = 20)
axes[1,1].legend(["X", "Y"], fontsize = 20, loc = "lower right")
axes[1,1].tick_params(labelsize =12)

axes[1,2].plot(all_models.xy_uni_logistic.series[0][0][0:100], color = "blue")
axes[1,2].plot(all_models.xy_uni_logistic.series[0][1][0:100], color = "red")
axes[1,2].set_title("Unidirectional coupled \nlogistic map processes", fontsize = 20)
axes[1,2].set_xlabel("Time (t)", fontsize = 20)
axes[1,2].set_ylabel("X and Y values", fontsize = 20)
axes[1,2].legend(["X", "Y"], fontsize = 20, loc = "lower right")
axes[1,2].tick_params(labelsize =12)
#%% Vis tts


    #%%% visualise surrogate tests results
def draw_heatmap1(x_or_y_surr,axes, n, cbar_ax):
    heatmap_drawthis = pd.DataFrame.from_dict(x_or_y_surr, orient = 'index')
    sns.heatmap(heatmap_drawthis, ax = axes[n], annot = True, cbar = True, cbar_ax= cbar_ax)
    # Control color in seaborn heatmap
    # https://www.python-graph-gallery.com/92-control-color-in-seaborn-heatmaps

def draw_heatmap2(results_from_server):
    n = 0
    fig, axes = plt.subplots(1,2, sharey= True, figsize = (7.5,3))
    cbar_ax = fig.add_axes([1,.1,.03,.8])#rightness 0-1, upness [-1,1], width, height # https://www.geeksforgeeks.org/matplotlib-figure-figure-add_axes-in-python/
    # https://matplotlib.org/stable/tutorials/intermediate/tight_layout_guide.html
    # fig.subplots_adjust(wspace = 0.05)
    fig.tight_layout() 
    for key in results_from_server:
        x_or_y_surr = results_from_server[key]
        draw_heatmap1(x_or_y_surr, axes, n, cbar_ax)
        axes[n].set_title(key)
        n +=1
    # plt.show()
for i in results: 
    draw_heatmap2(results[i])
    
#%%  Checking if FitzHugh Nagumo codes
def d_FitzHugh_Nagumo(t, x_w):
    epsilon_t = np.random.standard_normal()
    dx_t = x_w[0] - x_w[0]**3 - x_w[1] + 0.23
    dw_t = 0.05* (x_w[0] + 0.3 -1.4 * x_w[1])
    return np.array([dx_t, dw_t])

def generate_FitzHugh_Nagumo(N, dt_s = 0.25):
    print('Generating FitzHugh-Nagumo')
    dt=1;
    sample_period = int(np.ceil(dt_s / dt));
    
    lag = int(50/dt);
    obs = sample_period * N;
    # epsilon = np.random.standard_normal(size = length + 1000)
    s = np.zeros((lag + obs + 1,2))
    for t in range(lag + obs):
        soln = sp.integrate.solve_ivp(d_FitzHugh_Nagumo,t_span= [0,dt],y0= s[t])
        # print(soln.y[:,-1])
        s[t+1] = soln.y[:,-1] # + eps;
        # print(s[t+1])
        
    x = s;    
    return x

test1 = data[0]['xy_FitzHugh_Nagumo']
test1 = list(test1.T)
fig, axes = plt.subplots()
for i in test1: axes.plot(i)


def d_FitzHugh_Nagumo(t, x_w):
    epsilon_t = np.random.standard_normal()
    dx_t = 0.7* (x_w[0] - x_w[0]**3 /3 - x_w[1] + epsilon_t)
    dw_t = 0.7*0.08* (x_w[0] + 0.7 -0.8 * x_w[1])
    return np.array([dx_t, dw_t])

def generate_FitzHugh_Nagumo(N, dt_s = 0.25):
    print('Generating FitzHugh-Nagumo')
    dt=1;
    sample_period = int(np.ceil(dt_s / dt));
    
    lag = int(50/dt);
    obs = sample_period * N;
    # epsilon = np.random.standard_normal(size = length + 1000)
    s = np.zeros((lag + obs + 1,2))
    for t in range(lag + obs):
        soln = sp.integrate.solve_ivp(d_FitzHugh_Nagumo,t_span= [0,dt],y0= s[t])
        # print(soln.y[:,-1])
        s[t+1] = soln.y[:,-1] # + eps;
        # print(s[t+1])
        
    x = s[lag::sample_period,];    
    return x

#%%%
print([chr(_) for _ in range(945,969)])

#%%%
def vis_data(data, name_data, rep_num=1, input_stat = 'test_reps_series', graph_type = 'as2series'): 
    if input_stat == 'test_one_series':
        draw = data
    elif input_stat == 'test_reps_series': 
        draw = data[rep_num-1]
        
    if graph_type == 'as2series':
        fig, ax = plt.subplots(figsize = (10, 2.7))
        ax.set_title(f'{name_data} replicate number {rep_num}')
        for i in range(len(draw)): # for x, for y
            ax.plot(draw[i])
            # ax.plot([draw[i].mean()]*len(draw[i]))
    elif graph_type == 'ascordinates':
        fig,ax = plt.subplots()
        ax.scatter(draw[0],draw[1])
        ax.set_title(f'{name_data} replicate number {rep_num}')

vis_data(ts_ar['data']['data'],'ts_ar',rep_num=5)

for name_data in data[0].keys():
    for i in range(1,5): 
        vis_data(data,name_data,rep_num=i)
        
for i in range(len(pvals)): 
    vis_data(data,'xy_Caroline_LV',rep_num=i)

import Caroline_help_scripts.benchmark_mpi_coop.test_stats_lv as test_stats_lv
test1 = test_stats_lv(1, dt_s = 0.25, N=500, noise=0.01, noise_T=0.05, intx='competitive')
vis_data(test1, 'xy_Caroline_LV', input_stat='test_one_series')

vis_data(data, 'xy_Caroline_LV', rep_num=5, graph_type='ascordinates')


#%%%
def convert_format(pvals): 
    obj = pd.concat(
        [
            pd.DataFrame({
                f'{xory}_{stats}_{cst}': 
                    [ pvals[stats_test]['pvals'][i]['pvals'][xory][stats][cst] for i in range(len(pvals['data']['data']))] for stats_test in pvals.keys() if stats_test != 'data' for xory in ['surrY', 'surrX'] for stats in pvals[stats_test]['stats_list'] for cst in pvals[stats_test]['test_list']
                    }) ], axis = 1)
    return obj

# params['data_list']
# LV = convert_format(pvals, 'xy_Caroline_LV')
# CH = convert_format(pvals, 'xy_Caroline_CH')
# FN = convert_format(pvals, 'xy_FitzHugh_Nagumo')
# (test1 <= 0.05).sum()/len(test1)

def pvals_hist(pvals, subttl):  # data is convert_format(data) # subttl = name of model
    draw_hist = convert_format(pvals)
    for data in draw_hist.values():
        pd = int(data.shape[1]/2) # pair distance
        # hist_bins = [data.min().min(), 0.05, data.max().max()]
        hist_bins = np.linspace(data.min().min(),data.max().max(), 20)
        fig, ax = plt.subplots(pd, figsize = (10, 10))
        
#%% twin
def create_embed_space(data, embed_dim, tau, pred_lag=0):
    """
    Create embed space, used in  choose_twin_threshold
    Prepares a "standard matrix problem" from the time series data

    Args:
        data (array): Created in choose_twin_threshold, data = np.tile(timeseries, (2,1)).T:
            A 2d array with two columns where the first column
            will be used to generate features and the second column will
            be used to generate the response variable
        embed_dim (int): embedding dimension (delay vector length) , calculated from ccm
        tau (int): delay between values
        pred_lag (int): prediction lag
    """
    print("Twin embedding")
    x = data[:,0]
    y = data[:,1]
    feat = []
    resp = []
    idx_template = np.array([pred_lag] + [-i*tau for i in range(embed_dim)])
    for i in range(x.size):
        if np.min(idx_template + i) >= 0 and np.max(idx_template + i) < x.size: # make sure indices is not out of bound
            feat.append(x[idx_template[1:] + i])
            resp.append(y[idx_template[0] + i])
            # length = original length - tau *(embed_dim -1)
    return np.array(feat), np.array(resp)

# calculate distance matrices in embed spaces, 
# return the distance matrix of the embed space with the largest distances
def max_distance_matrix(X, Y):
    """
    returns max norm distance matrix

    Args:
        X (array): an m-by-n array where m is the number of vectors and n is the
            vector length
        Y (array): same shape as X
    """
    print("Generating twin distance matrix")
    n_vecs, n_dims = X.shape # original length -tau*(embed_dim-1); embed_dim
    K_by_dim = np.zeros([n_dims, n_vecs, n_vecs]) 
    # embed_dim matrices of disctance between each embed dim. dim1xdim1, dim2xdim2,...
    for dim in range(n_dims):
        K_by_dim[dim,:,:] = distance_matrix(X[:,dim].reshape(-1,1), Y[:,dim].reshape(-1,1))
    return K_by_dim.max(axis=0) # out of the 5 distance matrices - embed_dim =5

# predefined distance to define twins
# defined so that on average each point in delay space has specified # of neighbors
# (12% total points, later set to 10 % in get_twin_surrogates)
def choose_twin_threshold(timeseries, embed_dim, tau, neighbor_frequency=0.12, distmat_fxn=max_distance_matrix):
    """Given a univariate timeseries, embedding parameters, and a twin frequency,
        choose the twin threshold.

       Args:
         timeseries (numpy array): a univariate time series
         embed_dim (int): embedding dimension
         tau (int): embedding delay
         neighbor_frequency (float): Fraction of the "recurrence plot" to choose
             as neighbors. Note that not all neighbors are twins.

        Returns:
          recurrence distance threshold for twins
    """
    print("Choosing twin thresholds")
    # timeseries is 1d
    timeseries = np.copy(timeseries.flatten())
    data_ = np.zeros([timeseries.size, 2])
    data_[:,0] = timeseries
    data_[:,1] = timeseries
    # or data_ = np.tile(timeseries, (2,1)).T
    X, y = create_embed_space(data_, embed_dim=embed_dim, tau=tau) # feat, resp
    K = distmat_fxn(X,X) # max_distance_matrix, time series length = 400, embed = 5 , tau = 2 then K.shape = (392,392)
    #np.fill_diagonal(K, np.inf) # self-neighbors are allowed in recurrence plot.
    k = K.flatten()
    k = np.sort(k) # sort all distances
    idx = np.floor(k.size * neighbor_frequency).astype(int)
    print(f"Twin threshold for distance is {k[idx]}")
    return k[idx]


#twin method
def get_twin_surrogates(timeseries, embed_dim, tau, num_surr=99,
                        neighbor_frequency=0.1, th=None):
    if th is None:
        th = choose_twin_threshold(timeseries, embed_dim, tau, neighbor_frequency)
    results = [] # i=0
    obj = surrogates.Surrogates(original_data=timeseries.reshape(1,-1), silence_level=2)
    for i in range(num_surr):
        surr = obj.twin_surrogates(original_data=timeseries.reshape(1,-1), dimension=embed_dim, delay=tau, threshold=th, min_dist=1)
        surr = surr.ravel()
        results.append(surr)
    return np.array(results).T

def get_twin_wrapper(timeseries,num_surr=99):
    print("\t\tCreating twin surrogates")
    embed_dim, tau = ccm.choose_embed_params(timeseries);
    surrs = get_twin_surrogates(timeseries,embed_dim,tau,num_surr);
    return surrs;
import ccm
y_surr = get_twin_wrapper(data[1])

y_surr = get_iaaft_surrogates(data[1], n_surr=99, n_iter=200)
#%% drawing heatmap?
        for a in range(pd):
            ax[a].hist(data.iloc[:,[a, a+pd]], bins = hist_bins,  label = data.columns[[a, a+pd]], density = True) 
            ax[a].legend(loc = 'upper left')
        fig.suptitle(t = subttl,   fontsize = 20, x=0.25, y=0.915) # 'center' 'top'
        # https://www.geeksforgeeks.org/how-to-create-a-single-legend-for-all-subplots-in-matplotlib/
        # https://matplotlib.org/stable/gallery/text_labels_and_annotations/legend_demo.html
    
# pvals_hist(CH, 'xy_Caroline_CH')
# pvals_hist(LV, 'xy_Caroline_LV')
# pvals_hist(FN, 'xy_FitzHugh_Nagumo')

pvals_hist(ts_ar_pvals, "ts_ar")

#%% Old structure
# def load_results_params(names):
#     # names = 'caroline_LvCh_FitzHugh_100'
#     fi = f"{names}/{names}_testing.pkl"
#     with open(fi, 'rb') as file:
#         results = pickle.load(file)
        
#     fi = f"{names}/{names}_parameters.pkl"
#     with open(fi, 'rb') as file:
#         params = pickle.load(file)
        
#     return results, params
# results, params = load_results_params('caroline_Lv_asym_comp2_xysinewn_tsar_1000')
# 'caroline_Lv_asym_comp2_xysinewn_tsar_1000'
# 'caroline_Lv_asym_comp3_tscoin_1000'
# 'caroline_LvCh_FitzHugh_1000_5', 'caroline_Lv_predprey_tsLV_1000', 'caroline_Lv_asym_comp3_tscoin_1000'
# del(names, fi, file)
# pvals   =   [ _['pvals'] for _ in results]
# data    =   [ _['XY'] for _ in results]
# time    =   [ _['runtime'] for _ in results]
# run_id  =   [ _['run_id'] for _ in results]
# old structure
# def calc_true_pos_rate(pvals, params): 
#     obj = {model : {
#         xory : {
#             stats : {
#                 cst : sum( _ <= 0.05 for _ in [pvals[i][model][f'{model}_{xory}'][stats][cst] for i in range(len(pvals))] )/len(pvals) for cst in params['test_list'] } for stats in params['stats_list']} for xory in ['surrY', 'surrX']} for model in params['data_list']   } 
#     return obj

# test = calc_true_pos_rate(pvals)
#%%% Reorganise stuff into workdir/model/pvals_parameters.

# def save_data(results, params):
#     data    =   [ _['XY'] for _ in results]
#     for model in data[0].keys(): 
#         saveD = {'data': [data[_][model] for _ in range(len(data))],
#                  'datagen_params' : params['datagen_params']}
        
#         test_param = {'r_tts' : 'choose_r',
#                       'r_tts' : 'choose_r',
#                       'maxlag' : params['datagen_params']['maxlag']}
#         saveP = {'pvals' : [{'pvals' : results[i]['pvals'][f'{model}'],
#                              'runtime' : results[i]['runtime']} for i in range(len(results))],
#                  'stats_list' : params['stats_list'],
#                  'test_list' : params['test_list'],
#                  'test_params' : test_param}

        
#         if model != 'xy_Caroline_LV': 
#             name_data = f'{model}'
#         else:
#             intx = params['datagen_params']['intx']
#             name_data = f'xy_Caroline_LV_{intx}'
            
#         print(f'{name_data}')
#         if not os.path.exists(f'{name_data}'): 
#             os.mkdir(f'{name_data}')
        
#         with open(f'{name_data}/data.pkl', 'wb') as fi:
#             pickle.dump(saveD, fi);
        
        
#         tests = '_'.join(params['stats_list'] + params['test_list'] + [str(_) for _ in test_param.values()])
#         with open(f'{name_data}/{tests}.pkl', 'wb') as fi:
#             pickle.dump(saveP, fi);
    
# def re_organise(model_name):
#     results, params = load_results_params(model_name)
#     save_data(results, params)
    
# for directry in glob.glob('caroline*'):
#     re_organise(directry)
    
# xy_FitzHugh_Nagumo
# xy_Caroline_LV_competitive
# xy_Caroline_CH
# xy_ar_u
# xy_ar_u2
# xy_uni_logistic
# xy_Caroline_LV_asym_competitive
# ts_logistic
# ts_sine_intmt
# xy_Caroline_LV_asym_competitive_2_rev
# xy_ar_u
# xy_ar_u2
# xy_uni_logistic
# xy_Caroline_LV_asym_competitive
# xy_sinewn
# ts_ar
# xy_Caroline_LV_asym_competitive_2
# ts_chaoticLV
# xy_Caroline_LV_predprey
# ts_noise_wv_kurtosis
# xy_Caroline_LV_mutualistic
# ts_coinflip
# xy_Caroline_LV_asym_competitive_3
# data_list_all = sorted(glob.glob('xy_*') + glob.glob('ts_*'))

#%%
'''
Models with xy prefix are dependent series -> pvals are expected to be <= 0.05.
True positive rate = proportion of pvals <= 0.05, rate as high as possible
False negative rate = proportion of pvals > 0.05, rate as low as possible
'''
class test:
    def __init__(self, i):
        self.name = i
        with open(f'{i}/pearson_mutual_info_tts_tts_naive_choose_r_4.pkl', 'rb') as fi:
            setattr(self, 'raw', pickle.load(fi))
    def save(self):
        print(f"{self.name}/pearson_mutual_info_tts_tts_naive_choose_r_4.pkl")
        with open(f"{self.name}/test1_pearson_mutual_info_tts_tts_naive_choose_r_4.pkl", 'wb') as fi:
            pickle.dump(self.raw, fi)
 
class test1:
    def check_xory(self):
        self.ls = []
        for keys, vals in self.__dict__.items():
            if keys == 'check_xory':
                continue
            if isinstance(vals, test): 
                for xory in list(vals.raw['pvals'][0]['pvals'].keys()):
                    if len(xory) == len('surrY'): 
                        print(f'{keys} all clear')
                        continue
                    else:
                        if keys not in self.ls: self.ls.append(keys)
                        for reps in list(vals.raw['pvals']):
                            reps['pvals'][xory.split('_')[-1]] = reps['pvals'][xory]
                            del reps['pvals'][xory]
                        print(f'{keys} {xory} added not deleted')
                    print(vals.raw['pvals'][0]['pvals'].keys())
                # vals.save()
                print(f"{vals.name}/pearson_mutual_info_tts_tts_naive_choose_r_4.pkl")
        print(self.ls)
                
    def save_list(self, ls):        
        for i in self.ls:
            obj = getattr(self, i)
            obj.save()
                
    def add_r_naive(self):
        for keys, vals in self.__dict__.items():
            if keys in ['check_xory', 'save_list', 'add_r_naive']:
                continue
            if isinstance(vals, test): 
                vals.raw['test_params']['r_naive'] = 'choose_r'
        
                 
                    # print(keys)
OMG = test1()
for i in ['xy_Caroline_CH', 'xy_Caroline_LV_competitive', 'xy_FitzHugh_Nagumo']:
    setattr(OMG, i, test(i))

# OMG.add_r_naive()
# OMG.xy_Caroline_CH.save()
# OMG.xy_Caroline_LV_competitive.save()
# OMG.xy_FitzHugh_Nagumo.save()
        
#%%
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
# Run summarise results.py and load the data
# all_models.xy_Caroline_CH.vis_data(100)
# drawthis = all_models.xy_Caroline_CH.series[100]
# Create rectangle patches
def draw_pearson(x, y):
    recpos = []
    recneg = []
    x_mean = x.mean()
    y_mean = y.mean()
    x_prime = x - x_mean
    y_prime = y - y_mean
    recvar = Rectangle((0,0), np.sqrt(np.sum(x_prime**2)),np.sqrt(np.sum(y_prime**2)), linewidth = 1, 
                                                         edgecolor = 'yellow', facecolor = 'yellow', alpha = 0.25)
    sumpos , sumneg = 0 ,0
    for i in range(len(x)):
        if x_prime[i]*y_prime[i] > 0 :
            recpos.append(Rectangle((x_mean, y_mean), x_prime[i], y_prime[i], linewidth = 1))
            sumpos = sumpos + x_prime[i]*y_prime[i]
        if x_prime[i]*y_prime[i] < 0 :
            recneg.append(Rectangle((x_mean, y_mean), x_prime[i], y_prime[i], linewidth = 1))
            sumneg = sumneg + abs(x_prime[i]*y_prime[i])

    recsumpos = Rectangle((0,0), np.sqrt(np.sum(x_prime**2)), sumpos/np.sqrt(np.sum(x_prime**2)), linewidth = 1 ,
                           edgecolor = 'green', facecolor = 'green', alpha = 1)
    recsumneg = Rectangle((0,0), np.sqrt(np.sum(x_prime**2)), sumneg/np.sqrt(np.sum(x_prime**2)), linewidth = 1 ,
                           edgecolor = 'red', facecolor = 'red', alpha = 1)
            
    fig,ax = plt.subplots()
    ax.scatter(x,y, s=3,)
    # ax.set_xlim(x.min(), np.sqrt(np.sum(x_prime**2))-x.min())
    # ax.set_ylim(y.min(), np.sqrt(np.sum(y_prime**2))-y.min())
    ax.axvline(x=x_mean,color = 'pink', linestyle='--',linewidth = 3, label = r'$\bar{x}$')
    ax.axhline(y=y_mean, color = 'orange', linestyle='--',linewidth = 3, label = r'$\bar{y}$')
    ax.add_collection(PatchCollection(recpos, edgecolor = 'green', facecolor = 'green', alpha = 0.05))
    ax.add_collection(PatchCollection(recneg, edgecolor = 'red', facecolor = 'red', alpha = 0.05))
    # ax.add_patch(recvar)
    ax.legend(loc = 'upper right')
    ax.set_title('Species 1 and Species 2 Growth', fontsize=20, y = 1.05)
    ax.set_xlabel('Species 1 mass', fontsize=17)
    ax.set_ylabel('Species 2 mass', fontsize=17)
    plt.show()
    plt.close()
    
    fig2,ax2 = plt.subplots()
    ax2.set_xlim(0, np.sqrt(np.sum(x_prime**2)))
    ax2.set_ylim(0, np.sqrt(np.sum(y_prime**2)))
    ax2.add_patch(recvar)
    ax2.add_patch(recsumneg)
    ax2.add_patch(recsumpos)    
    ax2.set_title('xy_Caroline_CH replicate number 100', fontsize=20, y=1.05)
    ax2.set_xlabel(r'$\sqrt{\sum{(\|x-\bar{x}\|})^2}$', fontsize=17)
    ax2.set_ylabel(r'$\sqrt{\sum{(\|y-\bar{y}\|})^2}$', fontsize=17)
    plt.show()
    plt.close()
    
draw_pearson(drawthis[0][19:-19], drawthis[1][19:-19])
draw_pearson(drawthis[0][19:-19], drawthis[1][10:-28])
draw_pearson(drawthis[0][19:-19], drawthis[1][0:-38])
#%% For LSA
# drawthis = all_models.ts_chaoticLV.series[0]
# drawthis = all_models.xy_Caroline_LV_predprey.series[0]
drawthis = all_models.xy_Caroline_CH_competitive.series[0]
drawthis = [_[150:250] for _ in drawthis]

xpos = np.arange(len(drawthis[0]))
fig, ax = plt.subplots(figsize = (6,4.5))
ax.set_title('Simulation of X and Y', fontsize = 19, fontweight = "bold")
ax.set_xlabel("Time", fontsize = 20)
ax.set_ylabel('Growth (mass)', fontsize = 20)
ax.plot(drawthis[0], color = "blue", linewidth = 1)
ax.plot(drawthis[1], color = "orange", linewidth = 1)
ax.legend(["X","Y"],loc='best', fontsize = 19)
ax.scatter(xpos, drawthis[0], color = "blue", s =2)
ax.scatter(xpos, drawthis[1], color = "orange", s =2)
ax.tick_params(labelsize = 12)
plt.show()
plt.close()

#%%%

# (stats.rankdata(x))/(x.size+1.0)
# plt.plot((stats.rankdata(x))/(x.size+1.0),x)
# plt.scatter((stats.rankdata(x))/(x.size+1.0),stats.norm.ppf((stats.rankdata(x))/(x.size+1.0)))
# plt.scatter((stats.rankdata(x))/(x.size+1.0),x)

# plt.scatter(x,(stats.rankdata(x))/(x.size+1.0))
# plt.set_title('Values vs Ranked')

# fig, ax = plt.subplots()
# ax.scatter(x,(stats.rankdata(x))/(x.size+1.0))
# ax.set_title('Values vs Ranked')

# fig, ax = plt.subplots()
# ax.scatter(stats.norm.ppf((stats.rankdata(x))/(x.size+1.0)),(stats.rankdata(x))/(x.size+1.0))
# ax.set_title('Values vs Ranked')

# fig, ax = plt.subplots()
# ax.scatter(stats.norm.ppf((stats.rankdata(x))/(x.size+1.0)),(stats.rankdata(x))/(x.size+1.0))
# ax.set_title('icdf transformed vs Ranked')

# fig, ax = plt.subplots()
# ax.scatter(stats.norm.ppf((stats.rankdata(x))/(x.size+1.0)),(stats.rankdata(x))/(x.size+1.0))
# ax.set_title('ICDF-transformed values vs Ranked')

#%%% LSA ranking to transform
x = drawthis[0]
y = drawthis[1]

from scipy import stats

fig, ax = plt.subplots(1,2, figsize = (10,4), sharey = True)
fig.tight_layout( pad =2) 
ax[0].scatter(x, stats.rankdata(x))
ax[0].set_ylabel("Ranking", fontsize = 20)
ax[0].set_xlabel("X", fontsize = 20)
ax[0].set_title("Ranking X", fontsize = 20, fontweight="bold")
ax[0].tick_params(labelsize = 17)
ax[1].scatter(y, stats.rankdata(y), color = "orange")
# ax[1].set_ylabel("Rank of mass", fontsize = 17)
ax[1].set_xlabel("Y", fontsize = 20)
ax[1].set_title("Ranking Y", fontsize = 20, fontweight="bold")
ax[1].tick_params(labelsize = 17)

fig, ax = plt.subplots(1,2, figsize = (10,4), sharey = True)
fig.tight_layout( pad = 2) 
ax[0].scatter(x, stats.rankdata(x)/(x.size+1))
ax[0].set_ylabel("Probability of \nfinding smaller values", fontsize = 20)
ax[0].set_xlabel("X", fontsize = 20)
ax[0].set_title("Ranking X", fontsize = 20, fontweight="bold")
ax[0].tick_params(labelsize = 17)
ax[1].scatter(y, stats.rankdata(y)/(y.size+1), color = "orange")
# ax[1].set_ylabel("Probability of \nfinding smaller values", fontsize = 15)
ax[1].set_xlabel("Y", fontsize = 20)
ax[1].set_title("Ranking Y", fontsize = 20, fontweight="bold")
ax[1].tick_params(labelsize = 17)

#%% LSA cumulative normal distribution
x = np.linspace(-4, 4, 1000)
y = stats.norm.cdf(x)
fig, ax = plt.subplots()
ax.scatter(x,y, s=1, color = "black")
ax.set_ylabel("Probability of \nfinding smaller values", fontsize = 20)
ax.set_xlabel("Normally distributed values", fontsize = 20)
ax.set_title("Cumulative distribution of \nnormally distributed values", fontsize = 17, fontweight = "bold")
ax.axvline(x=0, linewidth = 0.75, linestyle=":", color = "red")
ax.tick_params(labelsize = 17)

#%% LSA transform
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

def norm_transform(x):
    return stats.norm.ppf((stats.rankdata(x))/(x.size+1.0))

x_tr = norm_transform(drawthis[0])
y_tr = norm_transform(drawthis[1])


recx, recy = [], []
for i in range(len(x_tr)):
    recx.append(Rectangle((i,0), 0, x_tr[i], linewidth=1))
    recy.append(Rectangle((i,0), 0, y_tr[i], linewidth=1))
fig, ax = plt.subplots(figsize = (7,4.5 ))
ax.set_title('Transformed data by \ninverse cdf of normal distribution', fontsize = 20, fontweight = "bold")
ax.set_xlabel("Time", fontsize = 20)
ax.set_ylabel(r'Normalised X and Y $(\hat{X},\hat{Y})$', fontsize = 17)
ax.plot(x_tr, label = r'$(\hat{X})$')
ax.scatter(np.arange(len(x_tr)), x_tr, color = "blue", s = 5)
ax.add_collection(PatchCollection(recx, edgecolor = 'blue', facecolor = 'blue', alpha = 0.5))
ax.plot(y_tr, label = r'$(\hat{Y})$')
ax.scatter(np.arange(len(y_tr)), y_tr, color = "orange", s=5)
ax.add_collection(PatchCollection(recy, edgecolor = 'orange', facecolor = 'orange', alpha = 0.5))
    # ax.plot([draw[i].mean()]*len(draw[i]))
ax.legend(loc='best', fontsize = 19)
ax.tick_params(labelsize = 17)
ax.axhline(y=0, color="black", linestyle = ":", linewidth=0.75)
plt.show()
plt.close()

# LSA multiplying and calculating LS
x_y_tr = x_tr*y_tr
recstick = []
for i in range(len(x_y_tr)):
    recstick.append(Rectangle((i,0), 0, x_y_tr[i], linewidth = 1))
fig,ax = plt.subplots()
ax.scatter(np.arange(len(x_y_tr)),x_y_tr,s=2, color = 'purple')
ax.plot(x_y_tr, color = 'pink', linewidth = 0.75)
ax.add_collection(PatchCollection(recstick, edgecolor = 'pink'))
ax.set_title(r"Multiplication of $\hat{X}$ and $\hat{Y}$", fontsize = 20, fontweight = "bold")
ax.set_ylabel(r'$\hat{X}*\hat{Y}$', fontsize = 20)
ax.set_xlabel(r'Time', fontsize = 20)
ax.axhline(y=0, linestyle = ":", linewidth = 0.75, color = "black")
ax.tick_params(labelsize = 17)

def lsa_new(x_tr, y_tr):
    n = x_tr.size
    P = np.zeros(n+1)
    N = np.zeros(n+1)
    for i in range(x_tr.size):
        P[i+1] = np.max([0, P[i] + x_tr[i] * y_tr[i] ])
        N[i+1] = np.max([0, N[i] - x_tr[i] * y_tr[i] ])
    score_P = np.max(P) / n
    score_N = np.max(N) / n
    sign = np.sign(score_P - score_N)
    # return np.max([score_P, score_N], axis=0) * sign
    return [P, N, score_P, score_N, np.max([score_P, score_N], axis=0) * sign]

cul = lsa_new(x_tr, y_tr)

fig, ax = plt.subplots()
ax.stairs(cul[0], baseline=None, color = "green", linewidth = 1, label = r"Positive association $S^{+}$")
ax.axhline(y=cul[0].max(), color = "green", linewidth = 1, linestyle = ":")# , label = r"$max S^{+}$"
ax.text(0, cul[0].max()-1, r"$max (S^{+})$", size = 15, color = "green")
ax.stairs(cul[1], baseline=None, color = "red", linewidth =1, label = r"Negative association $S^{-})$")
ax.axhline(y=cul[1].max(), color = "red", linewidth = 1, linestyle = ":")# , label = r"$max S^{-}$"
ax.text(0, cul[1].max()-1, r"$max (S^{-})$", size = 15, color = "red")
ax.annotate(r"$MaxScore(S^{+},S^{-})$", xy=(np.argmax(cul[1]), np.max(cul[1])), xytext=(np.argmax(cul[1])-50, np.max(cul[1])-2),
            arrowprops=dict(arrowstyle="->"), size = 15)
ax.set_xlabel("Time", fontsize = 20)
ax.set_ylabel("Association scores", fontsize = 20)
ax.set_title("Possitive and Negative Association Scores", fontsize = 20, fontweight = "bold")
ax.legend(bbox_to_anchor=(0.5, -0.15), loc="center",
                bbox_transform=fig.transFigure, ncol=1,  fontsize = 15)
ax.tick_params(labelsize = 15)
#%% LSA interval
idx_pos = np.arange(len(drawthis[0]))[cul[0][1:]>0]
idx_neg = np.arange(len(drawthis[0]))[cul[1][1:]>0]
pos = []
start, length = 0, 0
for a in range(len(idx_pos)-1):
    print(a)
    if a == 0:
        start = idx_pos[a]
        length = 0  
    if idx_pos[a+1] - idx_pos[a] == 1:
        length +=1
    else:
        pos.append(np.array([start, length]))
        print(pos)
        start = idx_pos[a+1]
        length =0
    if a == len(idx_pos)-2:
        pos.append(np.array([start, length]))
        print(pos)
    
neg = []
start, length = 0, 0
for a in range(len(idx_neg)-1):
    print(a)
    if a == 0:
        start = idx_neg[a]
        length = 0  
    if idx_neg[a+1] - idx_neg[a] == 1:
        length +=1
    else:
        neg.append(np.array([start, length]))
        print(neg)
        start = idx_neg[a+1]
        length =0
    if a == len(idx_neg)-2:
        neg.append(np.array([start, length]))
        print(neg)
        
from matplotlib import patches
fig, ax = plt.subplots(figsize = (6,4.5))
ax.set_title('Simulation of X and Y', fontsize = 19, fontweight = "bold")
ax.set_xlabel("Time", fontsize = 20)
ax.set_ylabel('Growth (mass)', fontsize = 20)
ax.plot(drawthis[0], color = "blue", linewidth = 1)
ax.plot(drawthis[1], color = "orange", linewidth = 1)
ax.legend(["X","Y"],loc='best', fontsize = 19)
ax.scatter(xpos, drawthis[0], color = "blue", s =2)
ax.scatter(xpos, drawthis[1], color = "orange", s =2)
ax.tick_params(labelsize = 12)
for i in range(len(pos)):
    rect = patches.Rectangle((pos[i][0],1), pos[i][1], 1, linewidth=3, edgecolor='green', facecolor = 'none', alpha = 0.25)
    ax.add_patch(rect)
for i in range(len(neg)):
    rect = patches.Rectangle((neg[i][0],1), neg[i][1], 1, linewidth=3, edgecolor='red',  facecolor = 'none', alpha = 0.25)
    ax.add_patch(rect)
    
# box height = 0.55 for ts_chaoticLV and 
# 22 for LV_predprey
# 1 for CH_competitive
#%%
Xpos = 
#%% Generate xy_Caroline_mutualisitic data
import GenerateData as dataGen

with open('xy_Caroline_CH_competitive/data.pkl','rb') as fi:
    test1 = pickle.load(fi)
    
datagen_param = test['datagen_params']
datagen_param['intx'] = 'mutualistic'

list(dataGen.generate_niehaus(run_id = 1, dt_s = datagen_param['dt_s'], N=datagen_param['N'], noise=datagen_param['noise'], noise_T=datagen_param['noise_T'],intx=datagen_param['intx'])[:,:2].T)

xy_Caroline_CH_mutualistic = [list(dataGen.generate_niehaus(run_id = i, dt_s = datagen_param['dt_s'], N=datagen_param['N'], noise=datagen_param['noise'], noise_T=datagen_param['noise_T'],intx=datagen_param['intx'])[:,:2].T) for i in range(1000)]

data = {'data': xy_Caroline_CH_competitive, 'datagen_params' : datagen_param}

# with open('xy_Caroline_CH_mutualistic/data.pkl', 'wb') as fi:
#     pickle.dump(data,fi)

    

#%% Create subtraction table presentation slide
                
def power_subtraction(all_models, test = 'tts_119', stats_list = 'default'):
    power_subtr = {}
    
    for keys, vals in all_models.__dict__.items(): 
        if isinstance(vals, all_output):
            if stats_list == 'default':
                stats_list = vals.summary_output.rejection_rate_sbs.keys()
            power_subtr[keys] = {stats : np.subtract(vals.summary_output.rejection_rate_sbs[stats][test]['surrY'], vals.summary_output.rejection_rate_sbs[stats][test]['surrX'])
                   for stats in stats_list}
                  # [stats]['tts_119']['surrY'].keys() for stats in vals.summary_output.rejection_rate_sbs.keys())
    #         power_subtr[keys] = {stats: 
    #                              vals.summary_output.rejection_rate_sbs[stats]['tts_119']['surrY'] - 
    #                              vals.summary_output.rejection_rate_sbs[stats]['tts_119']['surrY'] 
    #                                for stats in vals.summary_output.rejection_rate_sbs.keys()}
            
    power_subtr = pd.DataFrame(power_subtr).T
    return power_subtr

power_subtr_tts = power_subtraction(all_models, stats_list=['granger_y->x','granger_x->y',
                                                    'ccm_y->x','ccm_x->y',
                                                    'mutual_info','lsa','pearson'])
power_subtr_randphase = power_subtraction(all_models, stats_list=['granger_y->x','granger_x->y',
                                                    'ccm_y->x','ccm_x->y',
                                                    'mutual_info','lsa','pearson'], 
                                          test = 'randphase')
power_subtr_twin = power_subtraction(all_models, stats_list=['granger_y->x','granger_x->y',
                                                    'ccm_y->x','ccm_x->y',
                                                    'mutual_info','lsa','pearson'], 
                                          test = 'twin')
power_subtr_tts.to_csv("power_subtr_tts.csv", header = True)
sns.heatmap(power_subtr_tts, annot = True, linewidths = 2)
power_subtr_randphase.to_csv("power_subtr_randphase.csv", header = True)
sns.heatmap(power_subtr_randphase, annot = True, linewidths = 2)
power_subtr_twin.to_csv("power_subtr_twin.csv", header = True)
sns.heatmap(power_subtr_twin, annot = True, linewidths = 2)
#%%% Check ccm and ccm2
fi = "xy_sinewn/ccm_y.x_ccm_x.y_randphase_4_test.pkl"
with open(fi, 'rb') as file:
    # Call load method to deserialze
    old = pickle.load(file)

fi = "xy_sinewn/ccm_y->x_ccm_x->y_randphase_4.pkl"
with open(fi, 'rb') as file:
    # Call load method to deserialze
    new = pickle.load(file)
    
pvals_old = { xory : 
              { stats: 
               { 'randphase': 
                [_['pvals'][xory][stats]['randphase'] for _ in old['pvals']]} 
                   for stats in old['stats_list']} 
                  for xory in ['surrY', 'surrX']}
    
pvals_old = { xory : 
              { stats: 
               { 'randphase': 
                [_['pvals'][xory][stats]['randphase'] for _ in old['pvals']]} 
                   for stats in old['stats_list']} 
                  for xory in ['surrY', 'surrX']}
    
pvals_new = { xory : 
              { stats: 
               { 'randphase': 
                [_['pvals'][xory][stats]['randphase'] for _ in new['pvals']]} 
                   for stats in new['stats_list']} 
                  for xory in ['surrY', 'surrX']}
    
def draw_hist(pvals, plottype = 'integrate', stats_list = 'default', test_list = 'default'):
    if stats_list == 'default':
        stats_list = pvals['surrY'].keys()
    if test_list =='default':
        test_list = pvals['surrY']['ccm_y->x'].keys()
        
    ncol = 4
    totl_items = sum([len([_ for _ in pvals[xory][stat] if _ in test_list]) for xory in ['surrY','surrX'] for stat in stats_list])

    if plottype == 'separate':
        totl_plots = totl_items
        
        nrow = int(np.ceil(totl_plots/4))
        if nrow < 1: nrow = 1
        fig, ax = plt.subplots(nrows= nrow, ncols = ncol, figsize = (20, 3*nrow), constrained_layout=True)
        fig.tight_layout(h_pad=8, w_pad = 4.5) 
        ax = ax.flatten()
        
        idx = 0
        for xory in ['surrY', 'surrX']:
            for stats in stats_list:
                for cst in test_list:
                    # rowax = idx // 4
                    # colax = idx % 4
                    if cst in pvals[xory][stats]:
                        data = pvals[xory][stats][cst]
                    else: continue
                    hist_bins = np.linspace(min(data),max(data), 20)
                    ax[idx].hist(data, bins = hist_bins, label = xory, density = True, color = ["green", "orange"])
                    ax[idx].legend(fontsize = 17)
                    # ax[idx].legend(bbox_to_anchor=(0, -0.5, 1, 0.2), loc="lower left", fontsize =20,
                    #                borderaxespad=0, ncol=2)
                    # ax[idx].legend(bbox_to_anchor=(1, 0), loc="lower right",
                    #                bbox_transform=fig.transFigure, ncol=2,  fontsize = 17)
                    ax[idx].set_title(f"{stats} with {cst}",   fontsize = 20, y = 1.05)
                    ax[idx].set_xlabel('p-values',   fontsize = 20)
                    ax[idx].tick_params(labelsize = 12)
                    ax[idx].set_ylabel('counts of p-values',   fontsize = 20)
                    ax[idx].axvline(x=0.05,color='red', linestyle='dashed', linewidth=1)
                    xmin, xmax, ymin, ymax = ax[idx].axis()
                    ax[idx].text(0.05*1.1, ymax*0.9, "pvals = 0.05", color = "red", fontsize = 20)
                    # ax[rowax,colax].hist(data, bins = hist_bins, label = xory, density = True)
                    # ax[rowax, colax].legend(loc = 'upper left')
                    # ax[rowax, colax].set_title(f"{stats} with {cst}")
                    # ax[rowax, colax].set_xlabel('p-values')
                    # ax[rowax, colax].set_ylabel('counts of p-values')
                    idx +=1
        fig.suptitle(t = "Null distribution of correlation statistics",   fontsize = 20, y = 0.98)
        make_space_above(ax, topmargin=0.5)
        plt.show()
        plt.close()
            
    elif plottype == 'integrate':
        totl_plots = totl_items/2

        nrow = int(np.ceil(totl_plots/4))
        if nrow < 1: nrow = 1
        fig, ax = plt.subplots(nrows= nrow, ncols = ncol, figsize = (21, 5*nrow), constrained_layout=True)
        fig.tight_layout(h_pad=9, w_pad = 9)
        ax = ax.flatten()

        idx = 0
        
        for stats in stats_list:
            for cst in test_list:
                # rowax = idx // 4
                # colax = idx % 4
                if cst in pvals['surrY'][stats]:
                    data = [pvals[xory][stats][cst] for xory in ['surrY', 'surrX']]
                else: continue
                minmin = min(min(data))
                maxmax = max(max(data))
                
                if minmin == maxmax :
                    hist_bins = 1
                else: 
                    hist_bins = np.linspace(minmin,maxmax, 20)
                
                ax[idx].hist(data, bins = hist_bins, label = ['surrY', 'surrX'], density = True, color = ["pink", "orange"])
                ax[idx].set_xlim(0,0.5)
                ax[idx].legend(fontsize=17, loc = "lower right")
                # ax[idx].legend(bbox_to_anchor=(0, -0.5, 1, 0.2), loc="center", 
                #                borderaxespad=0, ncol=2)
                # ax[idx].legend(bbox_to_anchor=(1, 0), loc="lower right",
                #                bbox_transform=fig.transFigure, ncol=2,  fontsize = 17)
                # ax[idx].legend(loc = 'upper left',   fontsize = 17)
                ax[idx].set_title(f"{stats} with {cst}",   fontsize = 20, y = 1.05)
                ax[idx].set_xlabel('p-values',   fontsize = 20)
                ax[idx].tick_params(labelsize = 13)
                ax[idx].set_ylabel('counts of p-values',   fontsize = 20)
                ax[idx].axvline(x=0.05,color='red', linestyle='dashed', linewidth=1)
                xmin, xmax, ymin, ymax = ax[idx].axis()
                ax[idx].text(0.05*1.1, ymax*0.9, "pvals = 0.05", color = "red", fontsize = 20)
                # ax[rowax, colax].hist(data, bins = hist_bins, label = ['surrY', 'surrX'], density = True)
                # ax[rowax, colax].legend(loc = 'upper left')
                # ax[rowax, colax].set_title(f"{stats} with {cst}")
                # ax[rowax, colax].set_xlabel('p-values')
                # ax[rowax, colax].set_ylabel('counts of p-values')
                idx +=1
        fig.suptitle(t = "Null distribution of correlation statistics",   fontsize = 20, y = 0.98)
        make_space_above(ax, topmargin=1)
        plt.show()
        plt.close()
        
draw_hist(pvals_old)
draw_hist(pvals_new)

#%% Combine onetenth files
class onetenth:
    def add_results(self, filename, foldername, testname = ''):
        with open(f'{foldername}/{filename}', 'rb') as file:
            test = pickle.load(file)
        if testname == '':
            if 'granger' in filename or 'ccm' in filename:
                testname = '_'.join(filename.split('->')[0:3]).split('.')[0]
            
        print(filename, foldername, testname)
        setattr(self, testname, test)

ot_obj = onetenth()        
for fold in os.listdir('xy_FitzHugh_Nagumo_cont'):
    if os.path.isdir(f'xy_FitzHugh_Nagumo_cont/{fold}'):
        for fi in os.listdir(f'xy_FitzHugh_Nagumo_cont/{fold}'):
            ot_obj.add_results(fi, f'xy_FitzHugh_Nagumo_cont/{fold}')
listgranger = []      
listccm = []                  
for key, val in ot_obj.__dict__.items():
    if key.split(sep="_")[0] == 'granger':
        listgranger.extend(val['pvals'])
    if key.split(sep="_")[0] == 'ccm2':
        listccm.extend(val['pvals'])

    
iteg_ccm = {'pvals': listccm, 'stats_list': ['ccm_y->x', 'ccm_x->y'],
            'test_list': ['tts', 'tts_naive'],
            'test_params': {'r_tts': 119, 'r_naive': 119, 'maxlag': 4}}
iteg_granger = {'pvals': listgranger, 'stats_list': ['granger_y->x', 'granger_x->y'],
            'test_list': ['tts', 'tts_naive'],
            'test_params': {'r_tts': 119, 'r_naive': 119, 'maxlag': 4}}
with open('xy_FitzHugh_Nagumo_cont/ccm2_y_x_ccm2_x_y_tts_tts_naive_119_119_4_onetenth.pkl', 'wb') as file:
    pickle.dump(iteg_ccm,file)
with open('xy_FitzHugh_Nagumo_cont/granger_y_x_granger_x_y_tts_tts_naive_119_119_4_onetenth.pkl', 'wb') as file:
    pickle.dump(iteg_granger,file)
    
#%%
with open('xy_ar_u/ccm2_y_x_ccm2_x_y_tts_tts_naive_119_119_4_onetenth.pkl', 'rb') as file:
    onetenth = pickle.load(file)
    
with open('xy_ar_u/ccm2_y.x_ccm2_x.y_tts_tts_naive_119_119_4.pkl', 'rb') as file:
    full = pickle.load(file)
    
pvals_onetenth = { xory : 
              { stats: 
               { (f"{cst}_{onetenth['test_params']['r_tts']}" if 'tts' in cst else cst): 
                [_['pvals'][xory][stats][cst] for _ in onetenth['pvals']] 
                for cst in onetenth['test_list']} 
                   for stats in ['ccm_y->x', 'ccm_x->y']} 
                  for xory in ['surrY', 'surrX']}
    
pvals_full = { xory : 
              { stats: 
               { (f"{cst}_{full['test_params']['r_tts']}" if 'tts' in cst else cst): 
                [_['pvals'][xory][stats][cst] for _ in full['pvals']] 
                for cst in full['test_list']} 
                   for stats in ['ccm_y->x', 'ccm_x->y']} 
                  for xory in ['surrY', 'surrX']} 
    
sum(np.subtract(np.sort(pvals_full['surrY']['ccm_y->x']['tts_119']), np.sort(pvals_onetenth['surrY']['ccm_y->x']['tts_119'])))
# equal 0 so they are equal

[np.where(_ == pvals_full['surrY']['ccm_y->x']['tts_119']) for _ in pvals_onetenth['surrY']['ccm_y->x']['tts_119']]

#%% binomial cutoff
from scipy.stats import binom
# reps is number of trials  
reps  = 10000
false_cutoff = binom.ppf(0.95,reps,0.05);

#%%
fig, ax = plt.subplots()
ax.plot(another_model.xy_FitzHugh_Nagumo_cont.series[0][0][0:200], color = "blue")
ax.plot(another_model.xy_FitzHugh_Nagumo_cont.series[0][1][0:200], color = "red")
ax.set_title("Excitable systems \n(FitzHugh Nagumo)", fontsize = 20)
ax.set_xlabel("Time (t)",fontsize = 20)
ax.set_ylabel("Membrane voltage (v) \nand recovery variable (w)", fontsize = 20)
ax.legend(["v", "w"], fontsize = 20, loc = "center right")
ax.tick_params(labelsize =12)
plt.savefig('xy_FitzHugh_Nagumo_cont_series.svg')

another_model = all_model(['xy_Caroline_LV_asym_competitive_2','xy_Caroline_LV_asym_competitive_3'])
fig, ax = plt.subplots()
ax.plot(another_model.xy_Caroline_LV_asym_competitive_2.series[0][0][0:200], color = "blue")
ax.plot(another_model.xy_Caroline_LV_asym_competitive_2.series[0][1][0:200], color = "red")
ax.set_title("LV unequally competitive 2", fontsize = 20)
ax.set_xlabel("Time (t)",fontsize = 20)
ax.set_ylabel("Species growth mass", fontsize = 20)
ax.legend(["Species 1", "Species 2"], fontsize = 20, loc = "center right")
ax.tick_params(labelsize =12)
plt.savefig('xy_Caroline_LV_asym_competitive_2_series.svg')

fig, ax = plt.subplots()
ax.plot(another_model.xy_Caroline_LV_asym_competitive_3.series[0][0][0:200], color = "blue")
ax.plot(another_model.xy_Caroline_LV_asym_competitive_3.series[0][1][0:200], color = "red")
ax.set_title("LV unequally competitive 3", fontsize = 20)
ax.set_xlabel("Time (t)",fontsize = 20)
ax.set_ylabel("Species growth mass", fontsize = 20)
ax.legend(["Species 1", "Species 2"], fontsize = 20, loc = "center right")
ax.tick_params(labelsize =12)
plt.savefig('xy_Caroline_LV_asym_competitive_3_series.svg')

another_model.add_model('xy_Caroline_LV_asym_competitive_2_rev')
fig, ax = plt.subplots()
ax.plot(another_model.xy_Caroline_LV_asym_competitive_2_rev.series[0][0][0:200], color = "blue")
ax.plot(another_model.xy_Caroline_LV_asym_competitive_2_rev.series[0][1][0:200], color = "red")
ax.set_title("LV unequally competitive 2 reverse", fontsize = 20)
ax.set_xlabel("Time (t)",fontsize = 20)
ax.set_ylabel("Species growth mass", fontsize = 20)
ax.legend(["Species 1", "Species 2"], fontsize = 20, loc = "center right")
ax.tick_params(labelsize =12)
#%%
x = all_models.xy_Caroline_LV_mutualistic.series[0][0][-100:]
y = all_models.xy_Caroline_LV_mutualistic.series[0][1][-100:]
fig, ax = plt.subplots()
ax.plot(x, color = "red", linewidth = 1.5)
ax.set_xticks([0, 25, 50, 75, 100], labels = [0, 25, 50, 75, 100], **font)
ax.plot(y, color = "blue", linewidth = 1.5)
ax.set_yticks(ax.get_yticks(), labels = ["%.3f"% _ for _ in ax.get_yticks()],**font)
ax.set_title("LV mutualistic model", **font)
ax.set_xlabel("Time (t)",**font)
ax.set_ylabel("population density", **font)
ax.legend(["Species 1", "Species 2"], loc = "center right")
ax.scatter(np.arange(len(x)),x, color = "red", s=7)
ax.scatter(np.arange(len(y)),y, color = "blue", s=7)
ax.tick_params(labelsize =16)
plt.savefig('plots/xy_Caroline_LV_mutualistic_series.svg')
#%%
x = all_models.xy_uni_logistic.series[0][0][-100:]
y = all_models.xy_uni_logistic.series[0][1][-100:]
fig, ax = plt.subplots()
ax.plot(x, color = "red", linewidth = 1.5)
ax.set_xticks([0, 25, 50, 75, 100], labels = [0, 25, 50, 75, 100], **font)
ax.plot(y, color = "blue", linewidth = 1.5)
ax.set_yticks(ax.get_yticks(), labels = ["%.1f"% _ for _ in ax.get_yticks()],**font)
ax.set_title("Unidirectional coupled logistic model", **font)
ax.set_xlabel("Time (t)",**font)
ax.set_ylabel("Species growth mass", **font)
ax.legend(["Species 1", "Species 2"], loc = "center right")
ax.scatter(np.arange(len(x)),x, color = "red", s=7)
ax.scatter(np.arange(len(y)),y, color = "blue", s=7)
ax.tick_params(labelsize =16)
plt.savefig('plots/xy_uni_logistic_series.svg')
#%%
with open('xy_Caroline_LV_asym_competitive_2/granger_y.x_granger_x.y_tts_tts_naive_119_119_4.pkl', 'rb') as fi:
    resultsG = pickle.load(fi)
        
repvals = [{'pvals': 
            {xory: 
             {stats : 
              {'tts_119': _['pvals'][xory][stats]['tts'], 
               'tts_naive_119': _['pvals'][xory][stats]['tts_naive']} 
              for stats in results['stats_list']}
                 for xory in ['surrY','surrX']}, 
                'runtime': _['runtime']} for _ in results['pvals']]
    
retestparams = {'r_tts': 119, 'r_naive': 119, 'maxlag': 4}
    
reresults = {'pvals': results['pvals'],
             'stats_list': results['stats_list'],
             'test_list': ['tts', 'tts_naive'],
             'test_params': retestparams}

with open('xy_Caroline_LV_asym_competitive_2/pearson_mutual_info_from_choose_r_4.pkl', 'wb') as fi:
    pickle.dump(reresults, fi)
    
#%% Check model parameters
import os, sys, pickle
os.chdir('C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/')

def load_data(fold):
    with open(f'Simulated_data/{fold}/data.pkl', 'rb') as fi:
        data = pickle.load(fi)
    return data
# ccms_grangers_randphase_100_200_final_2.pkl

data = {}
for fold in os.listdir('Simulated_data'):
    if 'xy_' in fold:
        data[fold] = load_data(fold)
        print(fold)
        
# Vis autocorrelation
import numpy
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
def vis_acf(data, name = ''):
    acf = autocorrelationcorr_BJ(data)
    fig,ax = plt.subplots()
    ax.scatter(range(len(acf)),acf)
    ax.set_title(name)
    plt.show()

for key, val in data.items():
    vis_acf(val['data'][0][0],key)
   
# all in one
def vis_acf(ax, data, name = ''):
    acf = autocorrelationcorr_BJ(data)
    ax.scatter(range(len(acf)),acf, s = 2)
    ax.set_title(name)
    return ax
fig = plt.figure(figsize= (15,4.5), constrained_layout=True)
gs = fig.add_gridspec(2, 5)
ax=[]
for x in range(2):
    for y in range(5):
        ax.append(fig.add_subplot(gs[x,y]))
x=0
for key, val in data.items():
    vis_acf(ax[x], val['data'][0][0],key)
    x+=1
plt.show()
#%% FFT
import scipy

fig,ax = plt.subplots()
ax.plot(data['xy_ar_u']['data'][0][0])
ax.set_title('xy_ar_u_0')

Amp = scipy.fft.fft(data['xy_ar_u']['data'][0][0])
f = scipy.fft.fftfreq(len(data['xy_ar_u']['data'][0][0]))
# fig,ax = plt.subplots()
# ax.plot(scipy.fft.ifft(Amp))

fig, ax = plt.subplots()
ax.plot(f, Amp)
plt.show()      

import cmath # complex math, methematics for complex numbers
import numpy as np
import matplotlib.pyplot as plt
def graph_periodic_series(Amp,f,T): 
    """
        exp(ix)=cosx+isinx
    Substitute x = wt with w is angular frequency, then
        exp(iwt) = cos(wt) + isin(wt)
    Substitute w = 2*pi*f with f is ordinary frequency, then
        exp(i2*pi*f*t) = cos(2*pi*f*t) + isin(2*pi*f*t)
    IDK why I want negative sign but I guess I just used it from some source...
    """
    ps = []
    for t in np.arange(T):
        ps.append(Amp*cmath.exp(2*np.pi*f*t*1j))
    
    return np.array(ps)
all_ps = []
for i in range(len(testfreq)):
    all_ps.append(graph_periodic_series(Amp[i], f[i], 500))
    # fig,ax = plt.subplots()
    # ax.plot(all_ps[i])
    # ax.set_title(f'i: {i}, Amp: {Amp[i]}, f: {f[i]}')
    # plt.show()

cul_sum = []
cul_sum.append(all_ps[0]/ 500)
for i in range(1,len(all_ps)):
    cul_sum.append(np.add(cul_sum[i-1],all_ps[i]/ 500))

i=499
fig,ax = plt.subplots()
ax.plot(cul_sum[i])
ax.set_title(f'i: 0..{i}, Amp: 0..{Amp[i]}, f: 0..{f[i]}')
plt.show()

# Verify:
# np.sum(data['xy_ar_u']['data'][0][0]-cul_sum[499].real)
# Out[115]: -5.226583055240042e-13

# np.sum(data['xy_ar_u']['data'][0][0]-scipy.fft.ifft(Amp).real)
# Out[116]: -1.7884131675582893e-14

# np.sum(scipy.fft.ifft(Amp).real-cul_sum[499].real)
# Out[117]: -5.047741738484213e-13

#%% Granger
x, y = data['xy_ar_u']['data'][0]
t = y.shape; # y is surrY matrix

import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests as granger # for granger
from statsmodels.tsa.api import VAR # vector autoregression for granger

XY = np.vstack((x,y.T)).T;
model = VAR(XY); # Vector autoregression
VARres = model.fit(maxlags=15,ic='aic'); # ??VAR
# VARres.summary()
maxlag=np.max([VARres.k_ar, 1])
#gc = VARres.test_causality(caused='y1',causing='y2');
#return np.array(gc.test_statistic).reshape(1);
GCres = granger(XY, [maxlag], verbose=False)

recX_x = GCres[1][1][0].params[0]*x[:-1] + GCres[1][1][0].params[1]
recX_xy = GCres[1][1][1].params[0]*x[:-1] + GCres[1][1][1].params[1]*y[:-1] + GCres[1][1][1].params[2]
    
fig, ax = plt.subplots()
ax.plot(x[251:301], color = 'green')
ax.plot(recX_x[250:300], color = 'red')
ax.plot(recX_xy[250:300], color = 'orange')
plt.show()

fig, ax = plt.subplots()
ax.plot(recX_xy)
plt.show()

ssr_x = np.sum((x[1:]-recX_x)**2)
ssr_xy = np.sum((x[1:]-recX_xy)**2)

(500-4)*(ssr_x-ssr_xy)/ssr_xy
#%%
# Sampling every dt_s
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

        
        # not put negative values to 0 then observe how many 'negative cases could have happened':

        
        # On average array([  0.  , 237.57]), quantile [0.25,0.5,0.75] of Y are [ 42.25,  84.5 , 126.75], but can not confirm that it's its fault because in such non-linear system, flipping like that would make big difference.
        # But actually, if we don't need the biological meaning of the system and only care about why we observed such difference between the directions
        # then we might just try the tests on these dataset with negative values.
        
        # Otherwise, we can still let it flip but count the number of flipped. initial count is 1 because initial value of one of them does have 0 ...
        # Over all integration steps (lag+obs+1) then mean = 377.28, quantile = [326.  , 371.5 , 414.25].
        # After the lag then mean = 174.42, quantile = [139.75, 172.5 , 202.75]. Can the distribution of these much 'disruption' over 500 points lower the pvalue of certain direction?
        
        # So I decided to just run the tests on the negatives.
        
        # Change initial values to [2, 1], should still goes around fixed point, neg count has mean 166.25 , quantile [127. , 160.5, 201. ]
        # Seeing longer periods of positive values so trying to check
        # Use code similar to LSA
        # Over all integrations,
        summary_pos2_0 = np.array([[len(_[2]), np.max(np.array(_[2]))] for _ in data2_0])
        np.mean(summary_pos2_0,axis = 0)
        np.max(summary_pos2_0,axis = 0)
        # there are 387.7 positive period on average. The average length of positive period are 494.72
        # the most positive periods with no disruption is 544 and can go up to 1012 length
        # For the last 2500 integrations
        summary_pos2_0 = np.array([[len(_[2]), np.max(np.array(_[2]))] for _ in data2_0])
        # 170.49 for mean and 278 for max .# # of disruptions in 2500 integration points
        # 423.34 for mean and 837 for max .# length of pos_period in 2500 integration points
        np.quantile(summary_pos2_0[:,0], [.25,.5,.75])
        np.quantile(summary_pos2_0[:,1], [.25,.5,.75])    
        # [140., 170., 198.] - # of disruptions
        # [348.75, 434.5 , 539.75] - # length of pos_period
        # Vis:
        pos_period2_0 = [i[2] for i in data2_0]
        # x = np.range(np.max(summary_pos[0]))
        fig, ax = plt.subplots(figsize = (10,5))
        ax.set_xlim(0,np.max(summary_pos2_0[:,0]))
        for i in pos_period2_0:
            ax.plot(np.arange(len(i)),np.array(i))
        # Percentage of positive period that < 100
        posperiperr2_0 = [np.sum(np.array(_)<100)/len(_) for _ in pos_period2_0] 
        # On average the number of positive  period that are less than 100 takes up 0.962 of total # positive periods.
        # per 100 periods, around 25 points are sampled so we might have interuptions every 25 sample points
        #
        
        # 
        
    #%%    
        # raise the extinct species density closer to the dominating ones
        ARGs = {'dt_s': 0.25, 'N': 500, 's0': np.array([2.,1.]), 'mu': np.array([0.7,0.7]), 'M': np.array([[-0.4,-0.5],[-0.5,-0.4]]), 'noise': 0.01, 'noise_T': 0.05}
        # tried raise_extinct = 0.01, 0.02, 0.05, 1; at some point, adding 'extinct' too much gives it too much advantages that it overdominates the other
        # tried raise initial value of the 0 one to 1
        data2_1 = []
        start = time.time()
        for i in range(reps): 
            data2_1.append(generate_lv(**ARGs))
        runtime = time.time() - start
        filename = '_'.join(['equal']+ [','.join([str(j) for j in _.flatten()]) if type(_) == type(np.array([[-0.4,-0.5],[-0.5,-0.4]])) else str(_) for _ in ARGs.values()])
        savethis = {'data': data2_1, 'datagen_params': ARGs}
        vis_data(data2_1[0])
        save_data(filename, savethis, filename)
        # Neg count of last 2500 integ points (with zero caps): 
        # 2_0 and 2_1 are kinda similar
        neg_count2_1 = [i[1] for i in data2_1]
        
        summary_pos2_1 = np.array([[len(_[2]), np.max(np.array(_[2]))] for _ in data2_1])
        np.mean(summary_pos2_1,axis = 0)
        np.max(summary_pos2_1,axis = 0)
        # 2_0: mean 170.49 # positive period; mean 423.34 length of positive period 
        # 2_1:mean 169.75 # positive period; mean 417.94 length of positive period 
        # 2_0 and 2_1 have similar avg and max
        # the most positive periods with no disruption is 278 and can go up to 837 length
        # the most positive periods with no disruption is 284 and can go up to 1234 length
        # ==> 2_1 in general has less negatives than 2_0 (last 2500 integration points)
        
        np.quantile(summary_pos2_1[:,0], [.25,.5,.75])
        np.quantile(summary_pos2_1[:,1], [.25,.5,.75])    
        # 2_0: [140., 170., 198.] - # of disruptions
        # 2_1: [131.25, 173.5 , 201.  ]
        # 2_0: [348.75, 434.5 , 539.75] - # length of pos_period
        # 2_1: [295.75, 382.5 , 500.25]
        
        # Vis:
        pos_period2_1 = [i[2] for i in data2_1]
        # x = np.range(np.max(summary_pos[0]))
        fig, ax = plt.subplots(figsize = (10,5))
        ax.set_xlim(0,np.max(summary_pos2_1[:,0]))
        for i in pos_period2_1:
            ax.plot(np.arange(len(i)),np.array(i))
        # Percentage of positive period that < 100
        posperiperr2_1 = [np.sum(np.array(_)<100)/len(_) for _ in pos_period2_1] 
        # On average the number of positive  period that are less than 100 takes up 0.959 of total # positive periods.
        # per 100 periods, around 25 points are sampled so we might have interuptions every 25 sample points
        #
        
        # In general changing (2,0) to (2,1) doesn't change too much as expected.
        
#%%
def load_data(sample, sampleres):
    with open(f'Simulated_data/{sample}/{sampleres}', 'rb') as fi:
        res = pickle.load(fi)
    return res
def change(res, pmilsa, test):
    pvals = res['pvals']
    
    for thing in pvals:
        for xory in ['surrX', 'surrY']:
            
            thing[1]['score_null_pval'][xory][test]['pearson'] = {'pval': pmilsa['pvals']['pvals'][xory][test]['pearson'][thing[0]]}
            thing[1]['score_null_pval'][xory][test]['mutual_info'] = {'pval': pmilsa['pvals']['pvals'][xory][test]['mutual_info'][thing[0]]}
            thing[1]['score_null_pval'][xory][test]['lsa'] = {'pval': pmilsa['pvals']['pvals'][xory][test]['lsa'][thing[0]]}
            
            thing[1]['runtime'][xory][test]['pearson'] = pmilsa['pvals']['runtime'][xory][test]['pearson'][thing[0]]
            thing[1]['runtime'][xory][test]['mutual_info'] = pmilsa['pvals']['runtime'][xory][test]['mutual_info'][thing[0]]
            thing[1]['runtime'][xory][test]['lsa'] = pmilsa['pvals']['runtime'][xory][test]['lsa'][thing[0]]
    pvalss = [{_[0]:_[1]} for _ in pvals]
    
    res['pvals'] = pvalss
    res['stats_list'].extend(['pearson','mutual_info','lsa'])
    return res

def resafe(sample, sampleres, ress):
    with open(f'Simulated_data/{sample}/{sampleres}', 'wb') as fi:
        pickle.dump(ress, fi);
    

samplelist = ['xy_ar_u',
              'xy_Caroline_LV_asym_competitive',
              'xy_Caroline_LV_asym_competitive_2',
              # 'xy_Caroline_LV_asym_competitive_3',
              # 'xy_Caroline_LV_competitive',
              # 'xy_Caroline_LV_mutualistic',
              'xy_Caroline_LV_predprey',
              'xy_FitzHugh_Nagumo_cont',
              # 'xy_sinewn',
              'xy_uni_logistic']

# sample = 'xy_Caroline_LV_asym_competitive'
for sample in samplelist:
    pmilsa = load_data(sample, 'pearson_mi_lsa_randphase_twin_tts.pkl')

    # os.listdir(f'Simulated_data/{sample}')
    for test in ['randphase', 'tts_naive', 'twin']:

        sampleres = f'ccms_grangers_{test}_100_200_final_2.pkl'
        # test = 'tts_naive'
        if os.path.isfile(f'Simulated_data/{sample}/{sampleres}'):
            print(sample, sampleres)
            res = load_data(sample, sampleres)
            ress = change(res, pmilsa, test)
            resafe(sample, sampleres, ress)
            os.rename(f'Simulated_data/{sample}/{sampleres}', f'Simulated_data/{sample}/{test}_100_200.pkl')
#%%
hahadist = []
for key,val in Res.items():
    hahadist.extend(val.df['SurrX'])
    hahadist.extend(val.df['SurrY'])
for key,val in Res1.items():
    hahadist.extend(val.df['SurrX'])
    hahadist.extend(val.df['SurrY'])
plt.hist(hahadist, bins=[0,25,50,75,100])
plt.show()
#%%
for sample in os.listdir('Simulated_data'):
    if 'xy_' in sample:
        sampdir = f'Simulated_data/{sample}'
        for fi in os.listdir(sampdir):
            if 'falsepos' in fi:                
                print(f'Loading:{sampdir}/{fi}')
                with open(f'{sampdir}/{fi}', 'rb') as file:
                    falsepos = pickle.load(file)
                newpvals = [{_[0]: _[1]} for _ in falsepos['pvals']]
                falsepos['pvals'] = newpvals
                
                with open(f'{sampdir}/{fi}', 'wb') as file:
                    pickle.dump(falsepos, file)