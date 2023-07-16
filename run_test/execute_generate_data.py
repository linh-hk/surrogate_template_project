#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 18:15:37 2023

@author: h_k_linh

This script is submited to SGE on UCL cluster as a task. 
What it does is:
    Load simulated data that is in ../Simulated_data
        Which data to load is using the first argument passed from qsub script.
        The second argument from the qsub script specifies which pairs of time series (index of simulated data) will be run. This came about because I simulated 1000 simulations in total and only want to run test on 100 simulations per run.
    Run the dependence test on the data in parallel and save results on cluster.
Note:
This script particularly use the tts protocol to general surrogate test
The imported data is not changed to test for false positive rate in this script
Integrated multiprocessor into the workflow ('new' in file name)
This script uses multiprocessor to excecute on cluster, rather than MPI4py ('2' in file name)

"""
import os
print(f'working directory: {os.getcwd()}')
# os.chdir('C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/')
# os.chdir('/home/h_k_linh/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test')
# os.getcwd()

# import GenerateData as dataGen
import numpy as np
# import Correlation_Surrogate_tests as cst
from matplotlib import pyplot as plt

import sys # to save name passed from cmd
import time

# import dill # load and save data
import pickle # load and save data

# from mpi4py.futures import MPIPoolExecutor

# sys.path.append('/home/hoanlinh/Simulation_test/Simulation_code/surrogate_dependence_test')
from Simulation_code.surrogate_dependence_test.GenerateData2_GLV import generate_lv
from Simulation_code.surrogate_dependence_test.GenerateData2_GLV import generate_lv_nocap

from Simulation_code.surrogate_dependence_test.GenerateData import vis_data
def save_data(filename, data, fi):
    if not os.path.exists(f'Simulated_data/LVextra/{filename}/'): 
        os.makedirs(f'Simulated_data/LVextra/{filename}/')
    with open(f'Simulated_data/LVextra/{filename}/data.pkl', 'wb') as fi:
        pickle.dump(data, fi)
#%%
if __name__ == '__main__':
    reps = 100 
    
    ARGs = {'dt_s': 0.25, 'N': 500, 's0': np.array([2.,0.]), 'mu': np.array([0.7,0.7]), 'M': np.array([[-0.4,-0.5],[-0.5,-0.4]]), 'noise': 0.01, 'noise_T': 0.05}
    data2_0 = []
    start = time.time()
    for i in range(reps): 
        data2_0.append(generate_lv(**ARGs))
    runtime = time.time() - start
    filename = '_'.join(['equal']+ [','.join([str(j) for j in _.flatten()]) if type(_) == type(np.array([[-0.4,-0.5],[-0.5,-0.4]])) else str(_) for _ in ARGs.values()])
    vis_data(data2_0[0])
    savethis = {'data': data2_0, 'datagen_params': ARGs}
    save_data(filename, savethis, filename)
    # not put negative values to 0 then observe how many 'negative cases could have happened':
    neg_count2_0 = [i[1] for i in data2_0]
    
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
#%% Generate LV series with flipping and not flipping points
if __name__ == '__main__':
    # from Simulation_code.surrogate_dependence_test.GenerateData2_GLV import generate_lv_nocap
    # solve fxn but not sampled yet.
    def sample(s, lag, obs, sample_period):
        x = s[lag:lag+obs:sample_period,]; # measurement noise
        for i in range(x.ndim):
            x[:,i] += 0.001*np.random.randn(x[:,i].size);
        return [x[:,_] for _ in [0,1]]
    reps = 100 
    
    ARGs = {'dt_s': 0.25, 'N': 500, 's0': np.array([2.,0.]), 'mu': np.array([0.7,0.7]), 'M': np.array([[-0.4,-0.5],[-0.5,-0.4]]), 'noise': 0.01, 'noise_T': 0.05}
    data = []
    start = time.time()
    for i in range(reps): 
        data.append(generate_lv_nocap(**ARGs))
    runtime = time.time() - start
    filename = '_'.join(['equal','cap']+ [','.join([str(j) for j in _.flatten()]) if type(_) == type(np.array([[-0.4,-0.5],[-0.5,-0.4]])) else str(_) for _ in ARGs.values()])
    savecap = {'data': [sample(_['s']['cap'], _['lag'], _['obs'], _['sample_period']) for _ in data],
               'datagen_params': ARGs, '(n)cap': 'cap'}
    savencap = {'data': [sample(_['s']['ncap'], _['lag'], _['obs'], _['sample_period']) for _ in data],
               'datagen_params': ARGs, '(n)cap': 'ncap'}
    vis_data(savecap['data'][0])
    vis_data(savencap['data'][0])
    filename = '_'.join(['equal','cap']+ [','.join([str(j) for j in _.flatten()]) if type(_) == type(np.array([[-0.4,-0.5],[-0.5,-0.4]])) else str(_) for _ in ARGs.values()])
    save_data(filename, savecap, filename)
    filename = '_'.join(['equal','ncap']+ [','.join([str(j) for j in _.flatten()]) if type(_) == type(np.array([[-0.4,-0.5],[-0.5,-0.4]])) else str(_) for _ in ARGs.values()])
    save_data(filename, savencap, filename)
    # sys.stdout.flush();
    # sys.stdout.write('Total time: %5.2f seconds\n' % (time.time() - start));
    # sys.stdout.flush();

