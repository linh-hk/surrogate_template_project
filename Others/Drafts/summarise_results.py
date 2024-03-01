#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 20:01:55 2023

@author: h_k_linh
"""
import os
os.getcwd()
os.chdir('C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/')
# os.chdir('/home/h_k_linh/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/')
os.getcwd()
import glob
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from mergedeep import merge

import pickle

#%% Load data ver 2
''' 
Data run in new structure
'''
class correlation_surrogate_test:
    def __init__(self, model_name, filename = ''):
        if filename == '':
            self.model_name = model_name
            self.pvals = {}
            self.stats_list = []
            self.test_list = []
            self.test_params = {}
            self.runtime = {}
        else:
            with open(f'{model_name}/{filename}', 'rb') as file:
                test_results = pickle.load(file)
            self.model_name = model_name
            self.stats_list = ['lsa' if _ == 'lsa2' else _ for _ in test_results['stats_list']]
            # self.stats_list = test_results['stats_list']
            self.test_list = test_results['test_list']
            self.test_params = test_results['test_params']
            self.pvals = { xory : 
                          { stats: 
                           { (f"{cst}_{self.test_params['r_tts']}" if 'tts' in cst else cst): 
                            [_['pvals'][xory][stats][cst] for _ in test_results['pvals']] 
                            for cst in self.test_list} 
                               for stats in self.stats_list} 
                              for xory in ['surrY', 'surrX']} 
            self.runtime = [_['runtime'] for _ in test_results['pvals']]
            self.test_list = [(f"{cst}_{self.test_params['r_tts']}" if 'tts' in cst else cst) 
                              for cst in self.test_list]
    
    def count_rejection_rate(self):
        self.rejection_rate = { xory : { stats: { cst: sum( _ <= 0.05 for _ in self.pvals[xory][stats][cst]) / len(self.pvals[xory][stats][cst]) 
                                                 for cst in self.pvals[xory][stats].keys()} 
                                        for stats in self.stats_list} for xory in ['surrY', 'surrX']}
        
    def count_sbs(self):
        if not hasattr(self, 'rejection_rate'):
            self.count_rejection_rate()
        self.rejection_rate_sbs = {stats : 
                {cst : 
                 {xory : self.rejection_rate[xory][stats][cst] 
                  if cst in self.rejection_rate['surrY'][stats].keys() else float("NaN") 
                  for xory in ['surrY','surrX']} 
                 for cst in self.test_list} 
                    for stats in self.stats_list}
        
    def draw_heatmap_xory(self, xory_pvals, test_list, axes, ax_idx, cbar_ax):
        heatmap_drawthis = pd.DataFrame.from_dict(xory_pvals, orient = 'index')
        if test_list != 'default':
            row = [_ for _ in test_list if _ in heatmap_drawthis.index.values]
            col = [_ for _ in test_list if _ in heatmap_drawthis.columns.values]
            if len(row) != 0: 
                heatmap_drawthis = heatmap_drawthis.loc[row,:]
            if len(col) != 0:
                heatmap_drawthis = heatmap_drawthis.loc[:,col]                
                
        sns.heatmap(heatmap_drawthis, 
                    ax = axes[ax_idx], 
                    vmin = 0, vmax = 1,
                    annot = True, 
                    cbar = True, cbar_ax= cbar_ax)
        
    def vis_pvals(self, rep_num = 0): # 
        # just pick one value from list of pvals
        data = {xory : { stats : {cst : self.pvals[xory][stats][cst][rep_num] 
                                  for cst in self.pvals[xory][stats].keys()} 
                        for stats in self.stats_list}
                for xory in ['surrY', 'surrX']}
        ax_idx = 0
        # plt.rcParams['text.usetex'] = True
        fig, axes = plt.subplots(1,2, 
                                  sharey= True, 
                                  figsize = (7.5,3))
        cbar_ax = fig.add_axes([1,.1,.03,.8]) # See Drafts.py
        fig.tight_layout() 
        for xory in data:
            self.draw_heatmap_xory(data[xory], 'default', axes, ax_idx, cbar_ax)
            axes[ax_idx].set_title(xory)
            ax_idx +=1
        fig.suptitle(f"{self.model_name}, maxlag: {self.test_params['maxlag']}\npvals rep: {rep_num}", fontsize = 14)
        fig.subplots_adjust(top=0.80)
        plt.show()
        plt.close()        

    def draw_heatmap(self, rate_type = r'$N_0$ rejection rate'):
        if not hasattr(self, 'rejection_rate'):
            self.count_rejection_rate()
        ax_idx = 0
        # plt.rcParams['text.usetex'] = True
        fig, axes = plt.subplots(1,2, 
                                 sharey= True, 
                                 figsize = (7.5,3))
        cbar_ax = fig.add_axes([1,.1,.03,.8]) # See Drafts.py
        fig.tight_layout() 
        for xory in self.rejection_rate:
            self.draw_heatmap_xory(self.rejection_rate[xory], axes, ax_idx, cbar_ax)
            axes[ax_idx].set_title(xory)
            ax_idx +=1
        fig.suptitle(f"{rate_type} of tests on {self.model_name}, maxlag: {self.test_params['maxlag']}", fontsize = 14)
        fig.subplots_adjust(top=0.83)
        plt.show()
        plt.close()
        
# https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot#:~:text=To%20place%20the%20legend%20outside,left%20corner%20of%20the%20legend.&text=places%20the%20legend%20outside%20the,%2C%201)%20in%20axes%20coordinates.

    def draw_heatmap_sbs(self, stats_list = 'default', test_list = 'default', rate_type = r'$N_0$ rejection rate'):
        if not hasattr(self, 'rejection_rate_sbs'):
            self.count_sbs()
        if self.model_name.startswith('ts', 0, 2):
            rate_type = 'false positive rate'
        if self.model_name.startswith('xy', 0, 2):
            rate_type = 'true positive rate (power)'
            
        if stats_list == 'default':
            stats_list = self.stats_list
        
        ax_idx = 0
        fig, axes = plt.subplots(1,len(stats_list), 
                                 sharey= True, 
                                 figsize = (2*len(stats_list),1*len(test_list)))
        cbar_ax = fig.add_axes([1,.1,.03,.8]) # See Drafts.py
        fig.tight_layout() 
        for stats in stats_list:
            self.draw_heatmap_xory(self.rejection_rate_sbs[stats], test_list, axes, ax_idx, cbar_ax)
            axes[ax_idx].set_title(stats)
            ax_idx +=1
        fig.suptitle(f"{self.model_name}, {rate_type}, maxlag: {self.test_params['maxlag']}", fontsize = 14)
        fig.subplots_adjust(top=0.85)
        plt.show()
        plt.close()
        
    def draw_hist(self, plottype = 'integrate', stats_list = 'default', test_list = 'default'):
        if stats_list == 'default':
            stats_list = self.stats_list
        if test_list =='default':
            test_list = self.test_list
            
        ncol = 4
        totl_items = sum([len([_ for _ in self.pvals[xory][stat] if _ in test_list]) for xory in ['surrY','surrX'] for stat in stats_list])

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
                        if cst in self.pvals[xory][stats]:
                            data = self.pvals[xory][stats][cst]
                        else: continue
                        hist_bins = np.linspace(min(data),max(data), 20)
                        ax[idx].hist(data, bins = hist_bins, label = xory, density = True)
                        ax[idx].legend(bbox_to_anchor=(0, -0.5, 1, 0.2), loc="lower left", 
                                       borderaxespad=0, ncol=2)
                        # ax[idx].legend(bbox_to_anchor=(1, 0), loc="lower right",
                        #                bbox_transform=fig.transFigure, ncol=2,  fontsize = 17)
                        ax[idx].set_title(f"{stats} with {cst}",   fontsize = 17, y = 1.05)
                        ax[idx].set_xlabel('p-values',   fontsize = 17)
                        ax[idx].set_ylabel('counts of p-values',   fontsize = 17)
                        ax[idx].axvline(x=0.05,color='red', linestyle='dashed', linewidth=1)
                        xmin, xmax, ymin, ymax = ax[idx].axis()
                        ax[idx].text(0.05*1.1, ymax*0.9, "pvals = 0.05", color = "red")
                        # ax[rowax,colax].hist(data, bins = hist_bins, label = xory, density = True)
                        # ax[rowax, colax].legend(loc = 'upper left')
                        # ax[rowax, colax].set_title(f"{stats} with {cst}")
                        # ax[rowax, colax].set_xlabel('p-values')
                        # ax[rowax, colax].set_ylabel('counts of p-values')
                        idx +=1
            fig.suptitle(t = f"{self.model_name}\np-values/u distribution of tests (r: {self.test_params['r_tts']}, maxlag: {self.test_params['maxlag']})",   fontsize = 20, y = 1.1)
            plt.show()
            plt.close()
                
        elif plottype == 'integrate':
            totl_plots = totl_items/2

            nrow = int(np.ceil(totl_plots/4))
            if nrow < 1: nrow = 1
            fig, ax = plt.subplots(nrows= nrow, ncols = ncol, figsize = (20, 3*nrow), constrained_layout=True)
            fig.tight_layout(h_pad=8, w_pad = 4.5)
            ax = ax.flatten()

            idx = 0
            
            for stats in stats_list:
                for cst in test_list:
                    # rowax = idx // 4
                    # colax = idx % 4
                    if cst in self.pvals['surrY'][stats]:
                        data = [self.pvals[xory][stats][cst] for xory in ['surrY', 'surrX']]
                    else: continue
                    minmin = min(min(data))
                    maxmax = max(max(data))
                    
                    if minmin == maxmax :
                        hist_bins = 1
                    else: 
                        hist_bins = np.linspace(minmin,maxmax, 20)
                    
                    ax[idx].hist(data, bins = hist_bins, label = ['surrY', 'surrX'], density = True)
                    ax[idx].legend(bbox_to_anchor=(0, -0.5, 1, 0.2), loc="center", 
                                   borderaxespad=0, ncol=2)
                    # ax[idx].legend(bbox_to_anchor=(1, 0), loc="lower right",
                    #                bbox_transform=fig.transFigure, ncol=2,  fontsize = 17)
                    # ax[idx].legend(loc = 'upper left',   fontsize = 17)
                    ax[idx].set_title(f"{stats} with {cst}",   fontsize = 17, y = 1.05)
                    ax[idx].set_xlabel('p-values',   fontsize = 17)
                    ax[idx].set_ylabel('counts of p-values',   fontsize = 17)
                    ax[idx].axvline(x=0.05,color='red', linestyle='dashed', linewidth=1)
                    xmin, xmax, ymin, ymax = ax[idx].axis()
                    ax[idx].text(0.05*1.1, ymax*0.9, "pvals = 0.05", color = "red")
                    # ax[rowax, colax].hist(data, bins = hist_bins, label = ['surrY', 'surrX'], density = True)
                    # ax[rowax, colax].legend(loc = 'upper left')
                    # ax[rowax, colax].set_title(f"{stats} with {cst}")
                    # ax[rowax, colax].set_xlabel('p-values')
                    # ax[rowax, colax].set_ylabel('counts of p-values')
                    idx +=1
            fig.suptitle(t = f"{self.model_name}\np-values/u distribution of tests (r: {self.test_params['r_tts']}, maxlag: {self.test_params['maxlag']})",   fontsize = 20, y = 1.1)
            plt.show()
            plt.close()

# class summary_output:
#     def __init__(self, tests):
#         self.pearson = {}
#         self.mutual_info = {}
#         self.lsa = {}
#         self.granger_y_c_x = {}
#         self.granger_x_c_y = {}
#         self.ccm_y_p_x = {}
#         self.ccm_x_p_y = {}
        
#     def add_results(stats_name, )
###############################################################################
class all_output:
    
    def vis_data(self, rep_num=1, graph_type = 'as2series'): 
        draw = self.series[rep_num-1]   
        if graph_type == 'as2series':
            fig, ax = plt.subplots(figsize = (10, 2.7))
            ax.set_title(f'{self.model_name} replicate number {rep_num}')
            ax.set_xlabel(f"time order {r'(t_{1},t_{2},...,t_{500}'}")
            ax.set_ylabel('values')
            for i in range(len(draw)): # for x, for y
                ax.plot(draw[i])
                # ax.plot([draw[i].mean()]*len(draw[i]))
            plt.show()
            plt.close()
                
        elif graph_type == 'ascordinates':
            fig,ax = plt.subplots()
            ax.scatter(draw[0],draw[1])
            ax.set_title(f'{self.model_name} replicate number {rep_num}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            plt.show()
            plt.close()

    def add_test_results(self, filename, testname=''):
        # testname to rename test to make it short. 
        # instead of making it pearson_mutual_info_tts_tts_naive_59_59_4 can make it test 1, 2, 3, 4
        test = correlation_surrogate_test(self.model_name, filename)
        if testname == '':
            testname = filename.split('.')[0]
        setattr(self, testname, test)
        
    def load_all_results(self):  
        for fi in os.listdir(f'{self.model_name}'): 
            if fi == 'data.pkl':                
                continue 
            # if 'choose_r' in fi:
            #     continue
            # if 'lsa2' in fi:
            #     continue
            else:
                self.add_test_results(fi)
        self.summarise_output()
                
    def summarise_output(self):
        self.summary_output = correlation_surrogate_test(self.model_name)
        for keys, values in self.__dict__.items(): 
            if keys == 'summary_output':
                continue
            if isinstance(values, correlation_surrogate_test): 
                # pvals = {f"maxlag_{values.test_params['maxlag']}" : values.pvals}
                merge(self.summary_output.pvals, values.pvals)
                self.summary_output.stats_list.extend([_ for _ in values.stats_list if _ not in self.summary_output.stats_list] )
                self.summary_output.test_list.extend([_ for _ in values.test_list if _ not in self.summary_output.test_list ] )
                merge(self.summary_output.test_params, values.test_params)
                # self.summary_output.test_params[keys] = values.test_params
                self.summary_output.runtime[keys] = values.runtime
                self.summary_output.count_rejection_rate()
                self.summary_output.count_sbs()
                
    def __init__(self, model_name):
        self.model_name = model_name
        with open(f'{model_name}/data.pkl', 'rb') as file:
            gData = pickle.load(file)
        self.series = gData['data']
        self.datagen_params = gData['datagen_params']
        self.load_all_results()

###############################################################################           
# automated so that in __init_ load_all_results and in load_all_results summarise_output                    
class all_model:
    def __init__(self, model_lists):
        for model in model_lists:
            setattr(self, model, all_output(model))
            
    # def count_rejection_rate(self):
    #     for keys, vals in self.__dict__.items():
    #         if isinstance(vals, all_output):
    #             vals.summary_output.count_rejection_rate()
    
    # def count_sbs(self):
    #     for keys, vals in self.__dict__.items():
    #         if isinstance(vals, all_output):
    #             vals.summary_output.count_sbs()
            
    def vis_data(self, rep_num=1, graph_type = 'as2series'):
        for keys, vals in self.__dict__.items():
            if isinstance(vals, all_output):
                vals.vis_data(rep_num, graph_type)
            
    def draw_heatmaps(self):
        for keys, vals in self.__dict__.items():
            if isinstance(vals, all_output):
                vals.summary_output.draw_heatmap()
                
    def draw_heatmaps_sbs(self, stats_list = 'default', test_list = 'default', rate_type = r'$N_0$ rejection rate'):
        for keys, vals in self.__dict__.items():
            if isinstance(vals, all_output):
                vals.summary_output.draw_heatmap_sbs(stats_list, test_list, rate_type)
                
    def draw_hists(self, plottype = 'integrate', stats_list = 'default', test_list = 'default'):
        for keys, vals in self.__dict__.items():
            if isinstance(vals, all_output):
                # vals.vis_data()
                vals.summary_output.draw_hist(plottype, stats_list, test_list)
                
if __name__ == "__main__":
    data_list_all = sorted(glob.glob('xy_*')) + ['ts_chaoticLV'] # + glob.glob('ts_*'))   
    all_models = all_model(data_list_all)
    
    # all_models.draw_hists()
    # all_models.draw_hists(test_list = ['tts_119','tts_naive_119','randphase'])
    # all_models.draw_heatmaps_sbs()
    # all_models.draw_heatmaps_sbs(test_list = ['tts_119','tts_naive_119','randphase'])
    # all_models.vis_data()
    # all_models.testing()
    # all_models.count_rejection_rate()
    
    # all_models.ts_ar.summary_output.draw_heatmap_sbs()
    # all_models.ts_ar.summary_output.draw_heatmap_sbs(test_list = ['tts_119','tts_naive_119','randphase'])
    # all_models.ts_ar.summary_output.draw_hist()
    # all_models.ts_ar.summary_output.draw_hist(test_list = ['tts_119','tts_naive_119','randphase'])

# ts_ar.summary_output.draw_heatmap_sbs()
# ['ts_ar',
#  'ts_chaoticLV',
#  'ts_coinflip',
#  'ts_logistic',
#  'ts_noise_wv_kurtosis',
#  'ts_sine_intmt',
#  'xy_Caroline_CH',
#  'xy_Caroline_LV_asym_competitive',
#  'xy_Caroline_LV_asym_competitive_2',
#  'xy_Caroline_LV_asym_competitive_2_rev',
#  'xy_Caroline_LV_asym_competitive_3',
#  'xy_Caroline_LV_competitive',
#  'xy_Caroline_LV_mutualistic',
#  'xy_Caroline_LV_predprey',
#  'xy_FitzHugh_Nagumo',
#  'xy_ar_u',
#  'xy_ar_u2',
#  'xy_sinewn',
#  'xy_uni_logistic']