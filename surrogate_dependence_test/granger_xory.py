#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 28 19:40:55 2023

@author: h_k_linh
"""
import os
# os.chdir('C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/')
# os.chdir('/home/h_k_linh/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/')
print(f'working directory: {os.getcwd()}')
# os.getcwd()
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests as granger # for granger
from statsmodels.tsa.api import VAR # vector autoregression for granger

# from Simulation_code.surrogate_dependence_test.multiprocessor import Multiprocessor
# import time
#%%
def granger_surr_predict(x,y,pval=False):
    print("\t\t\t\t\t\tGranger")
    t,N = y.shape; # y is surrY matrix
    if (N == 1):
        data = np.vstack((x,y.T)).T;
        model = VAR(data); # Vector autoregression
        results = model.fit(maxlags=15,ic='aic'); # ??VAR
        # results.summary()
        maxlag=np.max([results.k_ar, 1])
        #gc = results.test_causality(caused='y1',causing='y2');
        #return np.array(gc.test_statistic).reshape(1);
        if pval:
            return granger(data, [maxlag], verbose=False)[maxlag][0]['ssr_ftest'][1];
        else:
            return np.array(granger(data, [maxlag], verbose=False)[maxlag][0]['ssr_ftest'][0]);
    else:
        result = np.zeros(N);
        for i in range(N):
            data = np.vstack((x,y[:,i].T)).T;
            model = VAR(data);
            res = model.fit(maxlags=15,ic='aic');
            #gc = res.test_causality(caused='y1',causing='y2');
            #result[i] = gc.test_statistic
            maxlag = np.max([res.k_ar, 1])
            if pval:
                result[i] = granger(data, [maxlag], verbose=False)[maxlag][0]['ssr_ftest'][1];
            else:
                result[i] = granger(data, [maxlag], verbose=False)[maxlag][0]['ssr_ftest'][0];
        return result;

def granger_predict_surr(x,y,pval=False):
    print("\t\t\t\t\t\tGranger")
    t,N = y.shape; # y is surrY matrix
    if (N == 1):
        data = np.vstack((y.T,x)).T;
        model = VAR(data); # Vector autoregression
        results = model.fit(maxlags=15,ic='aic'); # ??VAR
        # results.summary()
        maxlag=np.max([results.k_ar, 1])
        #gc = results.test_causality(caused='y1',causing='y2');
        #return np.array(gc.test_statistic).reshape(1);
        if pval:
            return granger(data, [maxlag], verbose=False)[maxlag][0]['ssr_ftest'][1];
        else:
            return np.array(granger(data, [maxlag], verbose=False)[maxlag][0]['ssr_ftest'][0]);
    else:
        result = np.zeros(N);
        for i in range(N):
            data = np.vstack((y[:,i].T,x)).T;
            model = VAR(data);               
            res = model.fit(maxlags=15,ic='aic');
            #gc = res.test_causality(caused='y1',causing='y2');
            #result[i] = gc.test_statistic
            maxlag = np.max([res.k_ar, 1])
            if pval:
                result[i] = granger(data, [maxlag], verbose=False)[maxlag][0]['ssr_ftest'][1];
            else:
                result[i] = granger(data, [maxlag], verbose=False)[maxlag][0]['ssr_ftest'][0];
        return result;
#%%
# start = time.time()
# statsgranger_star_y_cause_x = granger_predict_surr(xstar, ystar.reshape(-1,1)) 
# time.time()- start
# # 0.1222529411315918

# start = time.time()
# statsgranger_y_cause_x_surrY = granger_predict_surr(xstar, surrtts) 
# time.time()- start
# # 2.78220534324646

# start = time.time()
# statsgranger_star_x_cause_y = granger_predict_surr(ystar, xstar.reshape(-1,1))
# time.time()- start
# # 0.016766071319580078

# start = time.time()
# statsgranger_x_cause_y_surrY1 = granger_surr_predict(xstar, surrtts)
# time.time()- start
# # 2.87807035446167
