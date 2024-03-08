#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 14:00:56 2023

@author: h_k_linh

Optimsed runtime using Multiprocessor.
Old script is 'ccm.py' saved at ../Others/old_scripts
"""
import os
# os.chdir('C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/')
# os.chdir('/home/h_k_linh/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/')
print(f'working directory: {os.getcwd()}')

# import time
import numpy as np
from scipy.spatial import distance_matrix
import pandas as pd

# import pickle
from multiprocessing import Pool # , Process, Queue

#%% Define multiprocessor object
import sys
sys.path.append('/home/hoanlinh/Simulation_test/Simulation_code/surrogate_dependence_test')
from multiprocessor import Multiprocessor
# class Multiprocessor:

#     def __init__(self):
#         self.processes = []
#         self.queue = Queue()
    
#     def info(title):
#         inf = {'fxn name': title, 'from module': __name__, 'parent process': os.getppid(), 'process id': os.getpid()}
#         print(inf)

#     @staticmethod
#     def _wrapper(func, queue, args):
#         ret = func(*args)
#         queue.put(ret)
#         # self.info(func.__name__)
        

#     def add(self, func, args):
#         args2 = [func, self.queue, args]
#         p = Process(target=self._wrapper, args=args2)
#         self.processes.append(p)
#         # print(func.__name__)
        
#     def run(self, num_proc):
#         tot_proc = len(self.processes)
#         print(tot_proc)
#         for proc in np.arange(tot_proc):
#             self.processes[proc].start()
#             print(f'Proc {proc} running')
#             if proc % num_proc == num_proc-1:
#                 for i in np.arange(num_proc):
#                     if num_proc-i <tot_proc:
#                         self.processes[proc-i].join()
#                         print(f'Waiting for proc {proc+i}')

#     def results(self):
#         rets = []
#         for p in self.processes:
#             ret = self.queue.get()
#             rets.append(ret)
#         return rets
#%% Shared function between predict_surr and surr_predict

def get_weights(distances):
    """Performs weighting in KNN for Simplex projection

    Args:
      * distances_ (array) Array of distances

    Returns:
      * weights
    """
    distances = np.array(distances, copy=True).astype(float)
    if len(distances.shape) == 1:
        distances = distances.reshape(1,-1)
    if np.any(distances == 0): # deal with zeros
        distances[distances != 0] = np.inf
        distances[distances == 0] = 1
    min_dist = np.min(distances, axis=1).reshape(-1,1)
    u = np.exp(-distances / min_dist)
    weights = u / np.sum(u, axis=1).reshape(-1,1)
    return weights

def true_neighbour_ind(embY):
    KX = distance_matrix(embY,embY)
    np.fill_diagonal(KX,np.inf)
    RX = np.array([np.argsort(loc) for loc in KX])
    return [KX, RX]

def pcorr_multivariate_y(x,y):
    M = np.zeros([x.size, y.shape[1] + 1])
    M[:,0] = x
    M[:,1:] = y
    cov = np.cov(M.T)
    cov_xy = cov[0,:]
    std_y = np.sqrt(np.diag(cov))
    std_x = std_y[0]
    rho = cov_xy / (std_y * std_x)
    return rho[1:]

def pcorr(x,y):
    return pcorr_multivariate_y(x,y.reshape(-1,1))[0]

#%% predict_surr

def ccm_predict_surr_setup_problem(x, ysurr, embed_dim, tau, pred_lag):
    """embed X and find a bunch of resp Y from surr """
    # data = np.vstack((xstar, surrtts.T)).T
    # x = data[:,0]
    # y = data[:,1:]
    feat = []
    resp = []
    idx_template = np.array([pred_lag] + [-i*tau for i in range(embed_dim)])
    for i in range(x.size):
        if np.min(idx_template + i) >= 0 and np.max(idx_template + i) < x.size:
            feat.append(x[idx_template[1:] + i])
            resp.append(ysurr[idx_template[0] + i])
    if type(resp[0]) == np.float64:
        return np.array(feat), [np.array(resp)]
    else:
        return np.array(feat), [np.array(_) for _ in np.array(resp).T]

def ccm_predict_surr_iter(embX, Y, Yidx, KX, RX, n, embed_dim, tau, pred_lag, weights, score):
    all_xnbrs = RX[:, :embed_dim+1]
    if weights == 'uniform':
        yhat = np.mean(Y[all_xnbrs.astype(int)], axis=1) # compute observed xmap skill
    elif weights == 'exp':
        yhat = [np.sum( Y[all_xnbrs[loc,:].astype(int)] *
                        get_weights(KX[loc,all_xnbrs[loc,:].astype(int)])
                      )
                for loc in range(n)]
        yhat = np.array(yhat)
    else:
        raise ValueError('weights must be `exp` or `uniform`.')
        
    return [Yidx, embed_dim, tau, n, score(Y,yhat)]
    
def ccm_predict_surr(x, ysurr, embed_dim = None, tau = None, lib_sizes=[None], replace=False,
              n_replicates=1, weights='exp', score=pcorr,
              variable_libs=False, res = True):# , pred_lag=0
    # data = np.vstack((x, ysurr.T)).T
    # choose embed params
    if res:
        print('\t\t\t\t\t\tCCM predict surr')
        print('\t\t\t\t\t\t\tChoosing embed')
        pred_lag = 0
    else:
        pred_lag = 1
    if embed_dim == None and tau == None: 
        embed_dim, tau = choose_embed_params(x)
    
    if res: 
        print("\t\t\t\t\t\t\tOfficial ccm")
    # set up problem
    X, Y = ccm_predict_surr_setup_problem(x, ysurr, embed_dim, tau, pred_lag)
    # X is embeded feature, Y has 238 responses
    (n, embed_dim) = X.shape # n=#timepoints

    # STEP ONE: get true neighbor indices
    KX, RX = true_neighbour_ind(X)
    # ITERATE through each response surrogates
    if len(Y)==1:
        result = [list(ccm_predict_surr_iter(X, Y[0], 0, KX, RX, n, embed_dim, tau, pred_lag, weights, score))]
    else:
        mp = Multiprocessor()
        for each_PS in enumerate(Y):
            ARGs=(X, each_PS[1], each_PS[0], KX, RX, n, embed_dim, tau, pred_lag, weights, score)
            mp.add(ccm_predict_surr_iter, ARGs)
        mp.run(5) 
        result = mp.results()
    
    result_sorted = pd.DataFrame(result, columns=['surr_id', 'embed_dim', 'tau', 'n', 'score']).sort_values('surr_id')

    # result = list(map(lambda each_PS: ccm_predict_surr_iter(X, each_PS[1], each_PS[0], KX, RX, n, embed_dim, tau, pred_lag, weights, score), enumerate(Y)))
    if res:
        return np.array(result_sorted['score'])
    else:
        return result_sorted

# start = time.time()
# statsccml = ccm_predict_surr(xstar, ysurr)
# time.time() - start
#%% surr_predict

def ccm_surr_predict_iter(x, ysingle, yid, weights, score): 
    """ X is array of resp and Y is embeded feat"""
    # choose embed dim and tau
    pred_lag=0
    embed_dim, tau = choose_embed_params(ysingle)
    feat = []
    resp = []
    idx_template = np.array([pred_lag] + [-i*tau for i in range(embed_dim)])
    for i in range(x.size):
        if np.min(idx_template + i) >= 0 and np.max(idx_template + i) < x.size:
            feat.append(ysingle[idx_template[1:] + i])
            resp.append(x[idx_template[0] + i])
    X = np.array(resp) 
    embY = np.array(feat)
    (n, embed_dim) = embY.shape
    
    KX, RX = true_neighbour_ind(embY)
    all_ynbrs = RX[:, :embed_dim+1]
    if weights == 'uniform':
        xhat = np.mean(X[all_ynbrs.astype(int)], axis=1) # compute observed xmap skill
    elif weights == 'exp':
        xhat = [np.sum( X[all_ynbrs[loc,:].astype(int)] *
                        get_weights(KX[loc,all_ynbrs[loc,:].astype(int)])
                      )
                for loc in range(n)]
        xhat = np.array(xhat)
    else:
        raise ValueError('weights must be `exp` or `uniform`.')
        
    return [yid, embed_dim, tau, n, score(X,xhat)]

# def setup_problem_surr_predict_(x, ysurr, embed_dim, tau, pred_lag=0):
#     """resp X and find a bunch of embed Y from surr """
#     # data = np.vstack((xstar, surrtts.T)).T
#     # x = data[:,0]
#     # y = data[:,1:]
#     feat = []
#     resp = []
#     idx_template = np.array([pred_lag] + [-i*tau for i in range(embed_dim)])
#     for i in range(x.size):
#         if np.min(idx_template + i) >= 0 and np.max(idx_template + i) < x.size:
#             resp.append(x[idx_template[0] + i])
#             feat.append(np.array([y[idx_template[1:] + i,j] for j in np.arange(y.shape[1])]))
            
#     return np.array(resp), [np.array([x[_] for x in feat]) for _ in np.arange(len(feat[0]))]


# def ccm_iterate_surr_predict(Xomg, embY, Yidx, n, embed_dim, tau, weights, score):
#     KX, RX = true_neighbour_ind(embY)
#     all_xnbrs = RX[:, :embed_dim+1]
#     if weights == 'uniform':
#         xhat = np.mean(Xomg[all_xnbrs.astype(int)], axis=1) # compute observed xmap skill
#     elif weights == 'exp':
#         xhat = [np.sum( Xomg[all_xnbrs[loc,:].astype(int)] *
#                         get_weights(KX[loc,all_xnbrs[loc,:].astype(int)])
#                       )
#                 for loc in range(n)]
#         xhat = np.array(xhat)
#     else:
#         raise ValueError('weights must be `exp` or `uniform`.')
        
#     return [Yidx, embed_dim, tau, score(Xomg,xhat)]

def ccm_surr_predict(x, ysurr, lib_sizes=[None], replace=False,
              n_replicates=1, weights='exp', score=pcorr,
              variable_libs=False): # , pred_lag=0
    # # choose embed params
    # id_ed_tau = list(map(lambda x: choose_embed_params(x), enumerate(ysurr))
    # # set up matrices
    # Xomg, Yomg = setup_problem_surr_predict_(x, ysurr, embed_dim, tau)
    # # X is array of response, Y has 238 embeded feature
    # (n, embed_dim) = Yomg[0].shape # n=#timepoints
    
    # ITERATE through each embeded surrogates
    
    # result = [ccm_surr_predict_iter(x, ysingle, yid, weights, score, pred_lag = 0) for yid, ysingle in enumerate(ysurr.T)]
    # 317.0869941711426 sec
    # 256.24677634239197
    
    print('\t\t\t\t\t\tCCM surr predict')
    
    ysurrs = [np.array(ysurr[:,_]) for _ in np.arange(ysurr.shape[1])]
    
    mp = Multiprocessor()
    for each_ES in enumerate(ysurrs):
        ARGs = (x, each_ES[1], each_ES[0], weights, score)# , pred_lag
        mp.add(ccm_surr_predict_iter, ARGs)
    mp.run(3) 
    result = mp.results()
    # print('what s happenning', result[0], 'got results')
    result_sorted = pd.DataFrame(result, columns=['surr_id', 'embed_dim', 'tau', 'n', 'score']).sort_values('surr_id')
    # # ARGs = []
    # p = []
    # for each_ES in enumerate(ysurrs):
    #     ARGs = (x, each_ES[1], each_ES[0], weights, score, pred_lag)
    #     p.append(Process(target=ccm_surr_predict_iter, args = ARGs))
    # for pool5 in np.arange(0,len(ysurrs),5):
    #     for i in np.arange(5):
    #       if pool5 + i < len(ysurrs):
    #           p[pool5+i].start()
    #     for i in np.arange(5):
    #       if pool5 + i < len(ysurrs):
    #           p[pool5+i].join()
    #     p[pool5].start()
    #     p[pool5+1].start()
    #     p[pool5+2].start()
    #     p[pool5+3].start()
    #     p[pool5+4].start()
    #     p[pool5].join()
    #     p[pool5+1].join()
    #     p[pool5+2].join()
    #     p[pool5+3].join()
    #     p[pool5+4].join()
    
    # with Pool() as p:
    #     result = list(p.starmap(ccm_surr_predict_iter, ARGs))
    #     p.close()
    #     p.join()
    
    # result = list(map(lambda each_ES: ccm_surr_predict_iter(x, each_ES[1], each_ES[0], weights, score, pred_lag = 0), enumerate(ysurrs)))
    # 294.8649673461914 sec
    # 257.86770701408386
    
    # return pd.DataFrame(result, columns=['surr_id', 'embed_dim', 'tau', 'pred_lag', 'n', 'score'])
    return np.array(result_sorted['score'])
    # return result

# [yid, embed_dim, tau, n, score(X,xhat)]

# start = time.time()    
# statsccm2 = ccm_surr_predict(xstar, ysurr)
# time.time()-start
#%% choose embed params
# omg = 100
def choose_embed_params(emb_series, ed_grid=np.arange(1,5), tau_grid=np.arange(1,9),
                        lib_sizes = [None], replace = [False], n_replicates=1, 
                        weights='exp', score=pcorr, 
                        variable_libs=False, res = False): # , pred_lag=1
    # ts is x series.
    # mm = len(ed_grid)
    # mmm = len(tau_grid)
    # mmmm = len(pred_lag)
    # mall = mm*mmm*mmmm
    # data = np.zeros([len(emb_series), 2])
    # data[:,0] = emb_series 
    # data[:,1] = emb_series 
    # dat = np.tile(np.array(data),mmm)
    # print(np.array([np.repeat(ed_grid,mmm*mmmm), np.tile(tau_grid,mm*mmmm), np.tile(np.repeat(pred_lag,mmm),mm)]).T)
    
    ARGs = []
    for ed in ed_grid:
        for tau in tau_grid:
            ARGs.append((emb_series, emb_series, ed, tau, lib_sizes, replace, n_replicates, weights, score, variable_libs, res))# , lag
    with Pool(5) as p:
        ccm_res = list(p.starmap(ccm_predict_surr, ARGs))
        p.close()
        p.join()
        
    # ccm_res = list(map(ccm_predict_surr, 
    #                     [emb_series]*mall, [emb_series]*mall,
    #                     np.repeat(ed_grid,mmm*mmmm),
    #                     np.tile(tau_grid,mm*mmmm), 
    #                     [lib_sizes]*mall, [replace]*mall,
    #                     [n_replicates]*mall, 
    #                     [weights]*mall, 
    #                     np.tile(np.repeat(pred_lag,mmm),mm), 
    #                     [score]*mall, 
    #                     [variable_libs]*mall, 
    #                     [res]*mall))
    
    result = pd.concat(ccm_res)
    winner_idx = np.argmax(result['score'])
    # global omg
    # omg = result
    # print(result , globals().items())
    # print(winner_idx)
    return result['embed_dim'].iloc[winner_idx].astype(int), result['tau'].iloc[winner_idx].astype(int) # , result['pred_lag'].iloc[winner_idx].astype(int)
    # return result
    
