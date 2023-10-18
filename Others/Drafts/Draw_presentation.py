# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 15:12:18 2023

@author: hoang

"""
import os
os.getcwd()
os.chdir('C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/')
# os.chdir('/home/h_k_linh/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/')
os.getcwd()
import numpy as np
import pandas as pd
from scipy import stats 

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import animation
from functools import partial
from matplotlib import patches

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

#%%
# Find download link at https://www2.nau.edu/lrm22/lessons/predator_prey/predator_prey.html
# hare_lynx = pd.read_excel("C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Dissertation/Thesis_defence/hare_lynx_data.xlsx", usecols='A:C', skiprows=1 )
fig, ax = plt.subplots(figsize = (9,5))
fig.set_tight_layout(True)
ax.plot(hare_lynx['Year'], hare_lynx['Hare'], color = "blue")#, linewidth = 1
ax.plot(hare_lynx['Year'], hare_lynx['Lynx'], color = "#ff6600")
ax.legend(["Hare", "Lynx"], fontsize = 22, loc = "upper right")
ax.set_xlabel('Year', fontsize = 22, fontweight = 'bold', fontname = 'arial')
ax.set_ylabel('Population (thousands)', fontsize = 22, fontweight = 'bold', fontname = 'arial')
ax.tick_params(labelsize = 17)
ax.scatter(hare_lynx['Year'], hare_lynx['Hare'], color = "blue")
ax.scatter(hare_lynx['Year'], hare_lynx['Lynx'], color = "#ff6600")
ax.set_xticks(np.arange(1845, 1936, 10))
# ax.text(0, 7.1525, r'$/rho =$' + f"{pcor}", fontsize = 24, color = "green")
ax.set_title("Hare and Lynx population\n from 1845 - 1935", fontsize = 28, fontweight = 'bold', fontname = 'arial')
# plt.show()
plt.savefig("C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Dissertation/Thesis_defence/hare_lynx_data.png")

#%% Draw many mini trajectories of X and Y
# import pickle
# with open('Simulated_data/xy_Caroline_LV_mutualistic/data.pkl','rb') as fi:
#     load_obj = pickle.load(fi)
data = load_obj['data']
for i, d in enumerate(data):
    fig, ax = plt.subplots(nrows=1, ncols = 2, figsize = (4,1))
    ax[0].plot(d[0][:50], color = "blue")
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].plot(d[1][:50], color = "#ff6600")
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    plt.savefig(f"C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Dissertation/Thesis_defence/XY_{i}.svg")
    if i == 6:
        break

#%% Draw normal distribution, two-tailed test
# https://stackoverflow.com/questions/10138085/how-to-plot-normal-distribution
x = np.linspace(-4, 4, 1000)
y = stats.norm.pdf(x, loc=0, scale=1)
x_ = stats.norm.ppf(1-0.05/2, loc=0, scale=1)
fig, ax = plt.subplots()
fig.set_tight_layout(True)
ax.scatter(x,y, s=1, color = "black")
ax.set_ylabel("Probability density", fontsize = 22, fontweight = 'bold', fontname = 'arial')
ax.set_xlabel("Correlation values", fontsize = 22, fontweight = 'bold', fontname = 'arial')
ax.set_title("An example of a\nnull distribution", fontsize = 28, fontweight = "bold", fontname = 'arial')
ax.axvline(x=0, linewidth = 1, linestyle="--", color = "black")
ax.fill_between(x = x, y1 = y, where = (x<-x_)|(x_<x), color = 'blue', alpha = 0.2)
ax.axvline(x=x_, linewidth = 0.75, linestyle=":", color = "blue")
ax.axvline(x=-x_, linewidth = 0.75, linestyle=":", color = "blue")
ax.fill_between(x = x, y1 = y, where = (x<-2.25)|(2.25<x), color="none", hatch="////", edgecolor="green", linewidth=0.0)
ax.axvline(x=2.25, linewidth = 0.75, linestyle=":", color = "green")
ax.axvline(x=-2.25, linewidth = 0.75, linestyle=":", color = "green")
ax.tick_params(labelsize = 17)
# plt.show()
plt.savefig("C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Dissertation/Thesis_defence/Example_null_distribution.svg")
#%% Green hatch box
fig, ax = plt.subplots(figsize = (2,1))
fig.set_tight_layout(True)
ax.add_patch(patches.Rectangle((0,0), 30, 20, linewidth=0.0, facecolor="none", hatch = '///', edgecolor='green'))
ax.set_yticks([])
ax.set_xticks([])
plt.savefig("C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Dissertation/Thesis_defence/Hatch_box.svg")
plt.show()
#%% Load data from "C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/COURSES/BIOS0010 Biosciences Research Skills/Assignment2 Video log/surr_series.spydata"

tmp = np.arange(len(data[0]))
pcor_record = pcor_record_randphase_y

fig = plt.figure(figsize = (15.5,4.8), constrained_layout=True)
gs = fig.add_gridspec(1, 3, wspace=0.175, width_ratios = [1,1,1.1])
ax=[]
ax.append(fig.add_subplot(gs[0,0]))
ax.append(fig.add_subplot(gs[0,1]))
ax.append(fig.add_subplot(gs[0,2]))

ax[0].set_xlim(0,100)
ax[0].set_ylim(6.9,7.15)
ax[0].plot(data[0], color = "blue")#, linewidth = 1
ax[0].plot(data[1], color = "#ff6600")
ax[0].legend(["X", "Original Y"], fontsize = 22, loc = "upper right")
ax[0].set_xlabel('Time point (t)', fontsize = 22, fontweight = 'bold', fontname = 'arial')
ax[0].set_ylabel('X and Y value', fontsize = 22, fontweight = 'bold', fontname = 'arial')
ax[0].tick_params(labelsize = 17)
ax[0].text(0, 7.1525, r'$\rho =$' + f"{pcor}", fontsize = 24, color = "green")
ax[0].set_title("X and Y", fontsize = 28, fontweight = 'bold', fontname = 'arial', y = 1.09)

# def binning(val, hist_bins):
    
pcor_record_abs = np.absolute(pcor_record)
hist_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# hist_bins = np.arange(min(pcor_record_abs),max(pcor_record_abs)+0.005, 0.002)
# hist_y, edges_ticks = np.histogram(pcor_record_abs, hist_bins)
# hist_x = edges_ticks[:-1]+0.001
drawthis = pd.DataFrame({'pcor_record_abs':pcor_record_abs})
drawthis['binned'] = pd.cut(drawthis['pcor_record_abs'], bins = hist_bins, right = False)
drawthis['hist_x'] = drawthis['binned'].apply(lambda x: x.mid)
drawthis['hist_y'] = drawthis.groupby(['binned'])['binned'].cumcount()+1
# test = drawthis.groupby(['binned'])['hist_y'].max().fillna(value=0)

def draw_each_frame(frame_num, ax, data, y_surr, y_pos, drawthis, pcor): # , hist_bins
    i = frame_num
    # if i == 0:
    #     leg = "Original Y"
    #     colorr = "green"
    # else:
    leg = "Y surrogate"
    colorr = "red"
    # pcor_record[i] = round(float(pearsonr(data[0], y_surr[:,i]).statistic), 3)
    ax[1].clear()
    ax[1].set_xlim(0,100)
    ax[1].set_ylim(6.9,7.15)
    ax[1].plot(data[0], color = "blue")#, linewidth = 1
    ax[1].plot(y_surr[i,:], color = "#ff6600")
    ax[1].legend(["X", leg], fontsize = 22, loc = "upper right")
    ax[1].set_xlabel('Time point (t)', fontsize = 22, fontweight = 'bold', fontname = 'arial')
    ax[1].set_ylabel('X and Y value', fontsize = 22, fontweight = 'bold', fontname = 'arial')
    ax[1].tick_params(labelsize = 17)
    ax[1].text(0, 7.1525, r'$\rho =$' + f"{pcor_record[i]}", fontsize = 24, color = colorr)
    ax[1].set_title("X and surrogates of Y", fontsize = 28, fontweight = 'bold', fontname = 'arial', y = 1.09)
    
    ax[2].clear()  
    ax[2].set_ylim(0, (np.floor(max(drawthis['hist_y'])+0.005)/5)*5)
    ax[2].set_xlim(-0.05, 0.8)
    ax[2].scatter(drawthis['pcor_record_abs'][:i], drawthis['hist_y'][:i], color = 'red', s = 15)
    ax[2].scatter(drawthis['pcor_record_abs'][i], drawthis['hist_y'][i], color = 'red', s = 60, marker = '*')
    ax[2].scatter(pcor, 1, color = 'green', s = 15)
    ax[2].set_xlabel('Absolute ' + r'$\rho$', fontsize = 22, fontweight = 'bold', fontname = 'arial')
    ax[2].set_ylabel('Occurrences', fontsize = 22, fontweight = 'bold', fontname = 'arial')
    # ax[2].set_ylabel('Frequencies', **font)
    ax[2].axvline(x=pcor, color='green', linestyle='dashed', linewidth=.5)
    ax[2].text(pcor*0.55, 33, r'$\rho =$' +  f"{pcor}", color = "green", fontsize = 19, fontstyle = 'italic')
    ax[2].tick_params(labelsize = 15)
    ax[2].set_yticks([])
    # ax[2].set_yticks(np.arange(0, max(drawthis['hist_y'])+0.005, np.floor(max(drawthis['hist_y'])/5)))
    ax[2].set_xticks(hist_bins)
    ax[2].set_title("Null distribution", fontsize = 28, fontweight = 'bold', fontname = 'arial', y = 1.09)
    
    return ax

ax = draw_each_frame(98, ax, data, y_surr_randphase, pos, drawthis, pcor)
# plt.show()
# 0, 18, 19, 20, 21
# ax = draw_each_frame_tts(21, ax, data, y_surr_tts, pos, pcor_record_tts_y)
# plt.savefig("C:/Users/hoang/OneDrive/Desktop/test.png")

anim = animation.FuncAnimation(fig, draw_each_frame, fargs = (ax, data, y_surr_randphase, pos, drawthis, pcor), frames = 99, repeat=False)# , blit = True, interval = 500
anim.save('C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Dissertation/Thesis_defence/randphase.gif', writer = 'Pillow', fps = 4)
