# -*- coding: utf-8 -*-
"""
Created on Mon May  1 19:50:19 2023

@author: hoang
"""
import os
os.getcwd()
# os.chdir('C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/')
# os.chdir('/home/h_k_linh/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/')
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
import dill
import shelve

import sys
sys.path.append('/home/hoanlinh/Simulation_test/Simulation_code/surrogate_dependence_test')
from main import manystats_manysurr
import time
start = time.time()

#%% Retieve data and preprocessing
"""
Download the eclectrical Suplementary materials at https://royalsocietypublishing.org/doi/suppl/10.1098/rspb.2017.0722#secSuppl
"""
# Test codes for loading from xlsx file
## test = pd.read_excel("rspb20170722supp2.xlsx", sheet_name = None)
## test is a dictionary of the two sheets. there must be sheet_name, otherwise it only reads first sheet

# Load SCOR
SCOR = pd.read_excel("Real_data/Common_species_link_global_ecosystems_to_climate_change/rspb20170722supp2.xlsx", sheet_name="SCOR", skiprows=6)
# Representative time points at the middle of each 1/3Myr (1/6, 3/6, 5/6)
# round to forth decimal to synchronise with DOT
SCOR['Age (Ma)'] = SCOR['Age (Ma)'].round(4)
## SCOR.sort_values(by = "Age (Ma)", inplace = True, ignore_index = True)

# Linear interpolation of SCOR
SCOR['SCOR'].interpolate(method = 'linear', axis = 0, inplace = True)

# Load DOT
DOT = pd.read_excel("Real_data/Common_species_link_global_ecosystems_to_climate_change/rspb20170722supp2.xlsx", sheet_name="DOT", skiprows=4)

# Averaging DOT in time bins
# Group by time bins and averaging
DOT_new = DOT.groupby(pd.cut(DOT['Age (Ma)'],bins=np.arange(0,65.1,1/3).round(10))).DOT.mean().reset_index()
# Synchronise representative time points with SCOR
DOT_new['Age (Ma)']=np.arange(1/6,65.1,1/3).round(4) # or [i.mid for i in DOT_new['Age_bins']]
## DOT_new.sort_values(by = "Age (Ma)", inplace = True, ignore_index = True)

# Merge SCOR and DOT into one dataframe
SCOR_DOT = pd.merge(SCOR[["Age (Ma)", "SCOR"]], DOT_new, on = "Age (Ma)")

"""
Initial data - not detrended - not normalised
"""

#%% Visualising initial data
# Caroline configuration
plt.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['ytick.major.size'] = 5

mpl.rcParams['hatch.color'] = 'white'
mpl.rcParams['hatch.linewidth'] = 2

font = {'fontsize' : 16, 'fontweight' : 'bold'}# , 'fontname' : 'arial'
# font_data = {'fontsize' : 20, 'fontweight' : 'bold', 'fontname' : 'arial','color':'white'}
font_data = {'fontsize' : 20, 'fontweight' : 'bold', 'fontname' : 'arial','color':'black'}
# plt.plot(x, y, **font)
# texts = annotate_heatmap(im, valfmt="{x:.0f}", threshold=0, **font_data)
def visualise(df, processed_status = "", title = ""):
    fig, ax = plt.subplots()
    ax.plot(df["Age (Ma)"], df["SCOR"], color = "black")
    ax.set_xlabel("Age (Ma)", **font)
    ax.set_ylabel("SCOR", color = "black", **font)
    ax.set_xlim(65,0)
    ax1 = ax.twinx()
    ax1.plot(df["Age (Ma)"], df["DOT"], color = "red")
    reg = r"$\degree{C}$"
    ax1.set_ylabel(f'DOT ({reg})', color = "red",
                   **font)
    ax1.set_xlim(65,0)
    ## ax.set_xlim(ax.get_xlim()[::-1])
    fig.suptitle(title, **font)
    # plt.show()
    plt.savefig(f"Real_data/Common_species_link_global_ecosystems_to_climate_change/{processed_status}_data.svg")
visualise(SCOR_DOT, processed_status="Initial", title = "Initial data - not detrended - not normalised")
#%% Normalising to zero mean and unit standard deviation
def norm_transform(x):
    return stats.norm.ppf((stats.rankdata(x))/(x.size+1.0)) #, loc = 0, scale = 1
"""
Normalised without detrending data
"""
SCOR_DOT_N = SCOR_DOT.copy()
# SCOR_DOT_N["SCOR"], SCOR_DOT_N["DOT"] = [norm_transform(SCOR_DOT_N[x]) for x in ["SCOR","DOT"]]
SCOR_DOT_N[["SCOR","DOT"]] = SCOR_DOT_N[["SCOR","DOT"]].apply(norm_transform, axis = 0, raw = False, result_type = 'broadcast')


# Visualise Normalised without detrending data
visualise(SCOR_DOT_N, processed_status="normalised", title = "Normalised without detrending data")

#%% Detrending with secon-order polynomial
def detrend_polynomial(y, x, order=2):
    coeff = np.polyfit(x, y, order)
    predicted_y = np.polyval(coeff, x)
    plt.plot(x,y, color = "red")
    plt.plot(x, predicted_y, color = "green")
    plt.show()
    return y - predicted_y

def detrend_normalise(y, x, order=2):
    coeff = np.polyfit(x, y, order)
    predicted_y = np.polyval(coeff, x)
    # fig,ax = plt.subplots()
    # ax.plot(x,y, color = "red")
    # ax.plot(x, predicted_y, color = "green")
    # plt.show()
    return norm_transform(y - predicted_y)

SCOR_DOT_D = SCOR_DOT.copy()
SCOR_DOT_D[["SCOR","DOT"]] = SCOR_DOT_D[["SCOR","DOT"]].apply(detrend_polynomial, args = (SCOR_DOT_D["Age (Ma)"], 2), axis = 0, raw = False, result_type = 'broadcast')
visualise(SCOR_DOT_D, processed_status="detrended", title="Detrended without normalisation")

"""
Detrended and normalised data
"""
SCOR_DOT_DN = SCOR_DOT.copy()
SCOR_DOT_DN[["SCOR","DOT"]] = SCOR_DOT_DN[["SCOR","DOT"]].apply(detrend_normalise, args = (SCOR_DOT_DN["Age (Ma)"], 2), axis = 0, raw = False, result_type = 'broadcast')
visualise(SCOR_DOT_DN, processed_status="detrended and normalised", title="Detrended and normalised data")

#%% Apply the tests locally

stats_list = "all"
test_list = ['randphase', 'twin', 'tts']
maxlag = 4

def _wrapper(df, stats_list=stats_list, test_list=test_list, maxlag=maxlag):
    x = np.array(df["SCOR"])
    y = np.array(df["DOT"])
    return (manystats_manysurr(x, y, stats_list, test_list, maxlag))

res = _wrapper(SCOR_DOT)
res_N = _wrapper(SCOR_DOT_N)
res_D = _wrapper(SCOR_DOT_D)
res_DN = _wrapper(SCOR_DOT_DN)

#%% On cluster, run these
filename = 'Real_data/Common_species_link_global_ecosystems_to_climate_change/SCOR_DOT_cluster_dill_ml4.pkl'
dill.dump_session(filename)
print(f"Saved at {filename}")
print(time.time()-start)
# results, runtime = test
