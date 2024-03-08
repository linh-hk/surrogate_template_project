# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 19:37:26 2023

@author: hoang
"""

import os
print(f'working directory: {os.getcwd()}')
os.chdir('C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/')
# os.chdir('/home/h_k_linh/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test')
# os.getcwd()

# import GenerateData as dataGen
import numpy as np
# import Correlation_Surrogate_tests as cst
from scipy import stats

import sys # to save name passed from cmd
import time

# import dill # load and save data
import pickle # load and save data

# from mpi4py.futures import MPIPoolExecutor
sys.path.append("C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/Simulation_code/surrogate_dependence_test")
# sys.path.append('/home/hoanlinh/Simulation_test/Simulation_code/surrogate_dependence_test')
# import main as sdt
from ccm_xory import choose_embed_params
# import matplotlib.pyplot as plt
#%% Load data
def load_data(sample):
    if 'xy_' in sample:
        sampdir = f'Simulated_data/{sample}'
    elif '500' in sample:
        sampdir = f'Simulated_data/LVextra/{sample}'
    with open(f'{sampdir}/data.pkl', 'rb') as fi:
        data = pickle.load(fi)
    return data

#%%
sys.path.append("C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/Simulation_code/after_cluster")
from Visualise_data import recurrence_matrix, vis_recurrence_plot
from Visualise_results import results

if __name__ == "__main__":
    data = {}
    Res = {}
    for sample in os.listdir('Simulated_data'):
        if 'xy_' in sample:
            # data[sample] = load_data(sample)
            Res[sample] = results(sample)
            print(sample)
    for sample in os.listdir('Simulated_data/LVextra'):
        if '500' in sample:
            # data[sample] = load_data(sample)
            Res[sample] = results(sample)
            print(sample)
#%%       
    # for key, val in data.items():
    key = 'EComp_0.25_500_2.0,0.0_0.7,0.7_-0.4,-0.5,-0.5,-0.4_0.01_0.05'
    val = data[key]
    i = 0
    for XY in val['data']:
        vis_recurrence_plot(XY, title = f'ECompFast_20_{i}', saveto='C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Extended', filextsn= 'png')
        print(f'{key}_{i}')
        i+=1