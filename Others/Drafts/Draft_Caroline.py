#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 23:36:43 2023

@author: h_k_linh

Modify result's structure and add tts

"""
import os
print(f'working directory: {os.getcwd()}')
# os.chdir('/home/h_k_linh/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test')
# os.chdir('C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/')
# os.getcwd()

# import GenerateData as dataGen
import numpy as np
import pandas as pd
# import Correlation_Surrogate_tests as cst

import sys # to save name passed from cmd
import time

# import dill # load and save data
import pickle # load and save data
from mergedeep import merge

# from mpi4py.futures import MPIPoolExecutor
# import cluster_xory as cl

# list of data_name:
    # ['competitive_xdom_1extinct_fastsampling_data_100reps',
    # 'competitive_xdom_1extinct_slowsampling_data_100reps',
    # 'competitive_xdom_coexist_slowsampling_data_100reps',
    # 'lorenz_logistic_map_data_100reps',
    # 'predprey_data_100reps']
#%%
# def load_results(data_name):
#     result = []
#     for fi in os.listdir(f"run_this_for_Caroline/{data_name}/"):
#         # if 'data' in fi:
#         #     continue
#         # if 'ccms_grangers_randphase_twin_tts' in fi:
#         #     continue
#         # # else:
#         if 'falsepos' in fi:
#             with open(f'run_this_for_Caroline/{data_name}/{fi}', 'rb') as file:
#                 result.append(pickle.load(file))
#     return result

# # comp_xdom_1extinct_fast = load_results('competitive_xdom_1extinct_fastsampling_data_100reps')
# # comp_xdom_1extinct_slow = load_results('competitive_xdom_1extinct_slowsampling_data_100reps')
# # comp_xdom_coexist = load_results('competitive_xdom_coexist_slowsampling_data_100reps')
# # lorenz = load_results('lorenz_logistic_map_data_100reps')
# # predprey = load_results('predprey_data_100reps')

# comp_xdom_1extinct_fast_falsepos = load_results('competitive_xdom_1extinct_fastsampling_data_100reps')
# comp_xdom_1extinct_slow_falsepos = load_results('competitive_xdom_1extinct_slowsampling_data_100reps')
# comp_xdom_coexist_falsepos = load_results('competitive_xdom_coexist_slowsampling_data_100reps')
# lorenz_falsepos = load_results('lorenz_logistic_map_data_100reps')
# predprey_falsepos = load_results('predprey_data_100reps')
#%% Does not account for tts_naive, does not account for mislabel
# def merge_results(res):
#     stats_list = list(set([_ for sublist in res for _ in sublist['stats_list']]))
#     test_list = list(set([_ for sublist in res for _ in sublist['test_list']]))
#     intermed = [{_[0] : _[1] for _ in cst['pvals']} for cst in res]
#     pvals = merge(*intermed)
#     return {'stats_list': stats_list, 'test_list': test_list, 'pvals': pvals}
# NO comp_xdom_1extinct_fast_merged_NO = merge_results(comp_xdom_1extinct_fast)
# NO comp_xdom_1extinct_slow_merged_NO = merge_results(comp_xdom_1extinct_slow)
# NO comp_xdom_coexist_merged_NO = merge_results(comp_xdom_coexist)
# NO lorenz_merged_NO = merge_results(lorenz)
# NO predprey_merged_NO = merge_results(predprey)
#%%
def merge_new_res(res):
    stats_list = list(set([_ for sublist in res for _ in sublist['stats_list']]))
    test_list = list(set(['tts_naive' if _ == 'tts' else _ for sublist in res for _ in sublist['test_list']]))
    intermed = [{_[0] : _[1] for _ in sublist['pvals']} 
                for sublist in res ]
    pvals = merge(*intermed)
    return {'stats_list': stats_list, 'test_list': test_list, 'pvalss': pvals}
# test = merge_new_res(xy_ar_u)

def extract_pval_from_new_res(new_res):
    new_res['runtime'] = {xory : {cst : 
                                  {stat : [trial['runtime'][xory][cst][stat] 
                                           for idx, trial in new_res['pvalss'].items()]
                            for stat in new_res['stats_list']}
                     for cst in new_res['test_list']} 
             for xory in ['surrY', 'surrX']}
    new_res['test_params'] = merge(*[trial['test_params'] for idx, trial in new_res['pvalss'].items()])
    
    pvals = {xory : {cst : {stat : [trial['score_null_pval'][xory][cst][stat]['pval'] 
                                    for idx, trial in new_res['pvalss'].items()]
                            for stat in new_res['stats_list']}
                     for cst in new_res['test_list']} 
             for xory in ['surrY', 'surrX']}
    r = new_res['test_params']['surrY']['tts_naive']['surr_params']['r_tts']
    tts_pvals = {xory : {'tts': {stat : [B * (2 * r + 1) / (r + 1) for B in pvals[xory]['tts_naive'][stat] ] 
                                 for stat in new_res['stats_list']}
                        } 
                 for xory in ['surrY', 'surrX']}
    new_res['test_list'].append('tts')
    new_res['pvals'] = merge(pvals, tts_pvals)
    
# extract_pval_from_new_res(test)
def wrapper2(res):
    merged = merge_new_res(res)
    extract_pval_from_new_res(merged)
    return {key: val for key, val in merged.items() if key != 'pvalss'}

# comp_xdom_1extinct_fast_merged = wrapper2(comp_xdom_1extinct_fast)
# comp_xdom_1extinct_slow_merged = wrapper2(comp_xdom_1extinct_slow)
# comp_xdom_coexist_merged = wrapper2(comp_xdom_coexist)
# lorenz_merged = wrapper2(lorenz)
# predprey_merged = wrapper2(predprey)

comp_xdom_1extinct_fast_merged = wrapper2(comp_xdom_1extinct_fast_falsepos)
comp_xdom_1extinct_slow_merged = wrapper2(comp_xdom_1extinct_slow_falsepos)
comp_xdom_coexist_merged = wrapper2(comp_xdom_coexist_falsepos)
lorenz_merged = wrapper2(lorenz_falsepos)
predprey_merged = wrapper2(predprey_falsepos)

#%%
# # def save_file(data_name, results):
# #     with open(f'run_this_for_Caroline/{data_name}/ccms_grangers_randphase_twin_tts_Linh.pkl', 'wb') as file:
# #         pickle.dump(results, file)
        
# def save_file(data_name, results):
#     with open(f'run_this_for_Caroline/{data_name}/ccms_grangers_randphase_twin_tts_falsepos_Linh.pkl', 'wb') as file:
#         pickle.dump(results, file)
        
# save_file('competitive_xdom_1extinct_fastsampling_data_100reps', comp_xdom_1extinct_fast_merged)
# save_file('competitive_xdom_1extinct_slowsampling_data_100reps', comp_xdom_1extinct_slow_merged)
# save_file('competitive_xdom_coexist_slowsampling_data_100reps', comp_xdom_coexist_merged)
# save_file('lorenz_logistic_map_data_100reps', lorenz_merged)
# save_file('predprey_data_100reps', predprey_merged)
#%%
def load_results(data_name):
    with open(f'run_this_for_Caroline/{data_name}/ccms_grangers_randphase_twin_tts_Linh.pkl', 'rb') as file:
        results = pickle.load(file)
    return results

comp_xdom_1extinct_fast = load_results('competitive_xdom_1extinct_fastsampling_data_100reps')
comp_xdom_1extinct_slow = load_results('competitive_xdom_1extinct_slowsampling_data_100reps')
comp_xdom_coexist = load_results('competitive_xdom_coexist_slowsampling_data_100reps')
lorenz = load_results('lorenz_logistic_map_data_100reps')
predprey = load_results('predprey_data_100reps')

def load_results(data_name):
    with open(f'run_this_for_Caroline/{data_name}/ccms_grangers_randphase_twin_tts_falsepos_Linh.pkl', 'rb') as file:
        results = pickle.load(file)
    return results

comp_xdom_1extinct_fast_falsepos = load_results('competitive_xdom_1extinct_fastsampling_data_100reps')
comp_xdom_1extinct_slow_falsepos = load_results('competitive_xdom_1extinct_slowsampling_data_100reps')
comp_xdom_coexist_falsepos = load_results('competitive_xdom_coexist_slowsampling_data_100reps')
lorenz_falsepos = load_results('lorenz_logistic_map_data_100reps')
predprey_falsepos = load_results('predprey_data_100reps')

#%%
# def remerge_list(res):
#     new_sls = list(set([_ for sublist in res['stats_list'] for _ in sublist]))
#     new_tls = list(set([_ for sublist in res['test_list'] for _ in sublist]))
#     res['stats_list'] = new_sls
#     res['test_list'] = new_tls
# remerge_list(comp_xdom_1extinct_fast)
# remerge_list(comp_xdom_1extinct_slow)
# remerge_list(comp_xdom_coexist)
# remerge_list(lorenz)
# remerge_list(predprey)
# save_file('competitive_xdom_1extinct_fastsampling_data_100reps', comp_xdom_1extinct_fast)
# save_file('competitive_xdom_1extinct_slowsampling_data_100reps', comp_xdom_1extinct_slow)
# save_file('competitive_xdom_coexist_slowsampling_data_100reps', comp_xdom_coexist)
# save_file('lorenz_logistic_map_data_100reps', lorenz)
# save_file('predprey_data_100reps', predprey)
#%%
# from Draft_summary_ttest_redraw_graphs_new_run import heatmap
#%% Draw heatmap as Caroline
import matplotlib as mpl
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gs
import matplotlib.patheffects as pe
# import matplotlib.patches as patches
# import seaborn as sns
#%%
# Caroline configuration
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['ytick.major.size'] = 5

mpl.rcParams['hatch.color'] = 'white'
mpl.rcParams['hatch.linewidth'] = 2

font = {'fontsize' : 16, 'fontweight' : 'bold', 'fontname' : 'arial'}
# font_data = {'fontsize' : 20, 'fontweight' : 'bold', 'fontname' : 'arial','color':'white'}
font_data = {'fontsize' : 20, 'fontweight' : 'bold', 'fontname' : 'arial','color':'black'}
# plt.plot(x, y, **font)
# texts = annotate_heatmap(im, valfmt="{x:.0f}", threshold=0, **font_data)

# Take from web
def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw): 
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            text.set_path_effects([pe.Stroke(linewidth=3, foreground='white'),
                       pe.Normal()])
            texts.append(text)

    return texts

def heatmap(df_all, model_list, stats_list, cst_list):
    df = df_all.melt(ignore_index= False, var_name = "xory", value_name = "pvals").pivot_table(values = "pvals", index = ["models", "stats"], columns = ["surrogate_test", "xory"])
    for model in model_list:
        row = [(model, _) for _ in stats_list]
        col = [(_, xory) for _ in cst_list for xory in ["SurrY", "SurrX"]]
        draw = df.loc[row,col]
        # chi2 = df.loc[row, [(_, "chi2_pvalues") for _ in cst_list]]
        
        test = df_all.loc[[model]].melt(ignore_index= False, var_name = "xory", value_name = "pvals").pivot_table(values = "pvals", index = ["models", "stats"], columns = ["surrogate_test", "xory"])
        test = df_all.loc[[model]].pivot_table(values = ["SurrY", "SurrX"], index = ["models", "stats"], columns = ["surrogate_test"])
        test.xs('twin', level='surrogate_test', axis=1)
        
        fig = plt.figure(figsize = (7,5.5), constrained_layout = True) # figsize = (,), constrained_layout = True
        grid = fig.add_gridspec(nrows = 1, ncols = len(cst_list)+1, width_ratios = [1,1,1,0.25],
                                hspace = 0, wspace = 0.05)
        # axs = fig.subplots(1, len(cst_list), sharey = True, 
                           # subplot_kw = dict(aspect = 1), # equal to separately put ax.set_aspect("equal")
                           # gridspec_kw = dict(hspace = 0, wspace = 0.05))
        # https://matplotlib.org/stable/gallery/text_labels_and_annotations/demo_text_rotation_mode.html
        
        for i, cst in enumerate(cst_list):
            axs = fig.add_subplot(grid[i])
            axs.set_aspect("equal")
            im = axs.imshow(draw[[cst]], cmap = "cool", vmin = 0, vmax = 100)
            axs.set_xlabel(cst, **font)
            # axs.set_title(cst, **font)
            
            axs.set_xticks([0,1], labels = ["surrY", "surrX"], **font)
            axs.tick_params(top = True, labeltop=True, bottom = False, labelbottom=False)
            # ax.xaxis.set_ticks_position('top')
            plt.setp(axs.get_xticklabels(), rotation=30, ha="left", rotation_mode = "anchor")
            
            # # Chi2 squared test add hatch
            # chi2 = np.array(df.loc[row, (cst, "chi2_pvalues")])
            # print(model, cst, chi2)
            # chi2loc = np.where(chi2 <= 0.05)[0]
            # for j in chi2loc:
            #     # print(j)
            #     axs.add_patch(patches.Rectangle((-0.49,j-0.49),2,1, hatch = '--', edgecolor = 'white', fill = False))
                
            texts = annotate_heatmap(im, valfmt="{x:.0f}", threshold=0, **font_data)
            
            if i == 0:
                axs.set_yticks(np.arange(len(stats_list)), labels = stats_list, **font)
            else:
                axs.set_yticks([])
                
            axs.spines[:].set_visible(False)
            axs.set_xticks(np.arange(3)-0.5, minor = True)
            axs.set_yticks(np.arange(len(stats_list)+1)-0.5, minor = True)
            axs.grid(which = "minor", color='black', linestyle='-', linewidth=2)
            axs.tick_params(which =  "minor", bottom = False, left = False)
            
            
        fig.suptitle(f"{model}", **font) 
        # fig.subplots_adjust(right=0.8) 
        cbar_ax = fig.add_subplot(grid[-1]) 
        cbar = fig.colorbar(im, cax=cbar_ax, ticks = [0,25,50,75,100])
        cbar.ax.tick_params(labelsize = 16)
        # plt.show()

        plt.savefig(f"run_this_for_Caroline/falsepos_{model}.svg")
        
def into_df(combined): # faster than into_df2 - runtime= 0.21683335304260254
    index = []
    surrY = []
    surrX = []
    # if model_list == 'default':
    #     model_list = all_modls.model_list
    for model, mdl in combined.items():
        for cst in mdl['test_list']:
            for stat in mdl['stats_list']: 
                index.append(np.array([model, stat, cst]))
                if stat in ['pearson', 'mutual_info', 'lsa']:
                    surrY.append(sum( _ <= 0.05 for _ in mdl['pvals']['surrY'][cst][stat][:100]))
                    surrX.append(sum( _ <= 0.05 for _ in mdl['pvals']['surrX'][cst][stat][:100]))
                else:
                    surrY.append(sum( _ <= 0.05 for _ in mdl['pvals']['surrY'][cst][stat]))
                    surrX.append(sum( _ <= 0.05 for _ in mdl['pvals']['surrX'][cst][stat]))
    indices = pd.MultiIndex.from_arrays(np.array(index).T, names = ("models", "stats", "surrogate_test"))
    df = pd.DataFrame({"SurrY": surrY, "SurrX": surrX}, index = indices, dtype="float")
    return df
#%%
combine = {}
combine['comp_xdom_1extinct_fast'] = comp_xdom_1extinct_fast_merged
combine['comp_xdom_1extinct_slow'] = comp_xdom_1extinct_slow_merged
combine['comp_xdom_coexist'] = comp_xdom_coexist_merged
combine['lorenz'] = lorenz_merged
df_all = into_df(combine)

model_list = ['comp_xdom_1extinct_fast', 'comp_xdom_1extinct_slow', 'comp_xdom_coexist', 'lorenz', 'predprey_merged']
stats_list = ['granger_y->x', 'granger_x->y', 'ccm_y->x', 'ccm_x->y']
cst_list = ['randphase', 'twin', 'tts']
heatmap(df_all, model_list, stats_list, cst_list)

combine = {}
combine['comp_xdom_1extinct_fast_falsepos'] = comp_xdom_1extinct_fast_merged
combine['comp_xdom_1extinct_slow_falsepos'] = comp_xdom_1extinct_slow_merged
combine['comp_xdom_coexist_falsepos'] = comp_xdom_coexist_merged
combine['lorenz_falsepos'] = lorenz_merged
combine['predprey_falsepos'] = predprey_merged
df_all = into_df(combine)

model_list = ['comp_xdom_1extinct_fast_falsepos', 'comp_xdom_1extinct_slow_falsepos', 'comp_xdom_coexist_falsepos', 'lorenz_falsepos', 'predprey_falsepos']
stats_list = ['granger_y->x', 'granger_x->y', 'ccm_y->x', 'ccm_x->y']
cst_list = ['randphase', 'twin', 'tts']
heatmap(df_all, model_list, stats_list, cst_list)
