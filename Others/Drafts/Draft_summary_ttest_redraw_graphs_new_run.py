#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 16:08:09 2023

@author: h_k_linh
"""
import os
os.getcwd()
os.chdir('C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/')
# os.chdir('/home/h_k_linh/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/')
os.getcwd()
import numpy as np
import pandas as pd 
from mergedeep import merge

import pickle
#%%
# class output:
#     def __init__(self, model_name, filename = ''):
#         if filename == '':
#             self.model_name = model_name
#             self.pvals = {}
#             self.stats_list = []
#             self.test_list = []
#             self.test_params = {}
#             self.runtime = {}
#         elif 'ccms' in filename:
#             with open(f'{model_name}/{filename}', 'rb') as file:
#                 test_results = pickle.load(file)
#             self.model_name = model_name
#             self.stats_list = test_results['stats_list']
#             self.test_list = test_results['test_list']
#             # self.test_params = merge(*[_[1]['test_params'] for _ in test_results['pvals']])
#             # self.runtime = merge(*[_[1]['runtime'] for _ in test_results['pvals']])
            
#             self.pvals = {xory : {stat: {cst : [_[1]['score_null_pval'][xory][cst][stat]['pval'] for _ in test_results['pvals']] for cst in self.test_list} for stat in self.stats_list} for xory in ['surrY','surrX']}
            
#         else:
#             with open(f'{model_name}/{filename}', 'rb') as file:
#                 test_results = pickle.load(file)
#             self.model_name = model_name
#             self.stats_list = ['lsa' if _ == 'lsa2' else 'ccm_y->x' if _ == 'ccm2_y->x' else 'ccm_x->y' if _ == 'ccm2_x->y' else _ for _ in test_results['stats_list']]
#             # self.stats_list = test_results['stats_list']
#             self.test_list = test_results['test_list']
#             self.test_params = test_results['test_params']
#             self.pvals = { xory : 
#                           { stat: 
#                            { cst: 
#                             [_['pvals'][xory][stat][cst] for _ in test_results['pvals']] 
#                             for cst in self.test_list} 
#                                for stat in self.stats_list} 
#                               for xory in ['surrY', 'surrX']} 
#             self.runtime = [_['runtime'] for _ in test_results['pvals']]
#             self.test_list = [(f"{cst}_{self.test_params['r_tts']}" if 'tts' in cst else cst) 
#                               for cst in self.test_list]
            
# class all_output:
#     def add_test_results(self, filename, testname=''):
#         # testname to rename test to make it short. 
#         # instead of making it pearson_mutual_info_tts_tts_naive_59_59_4 can make it test 1, 2, 3, 4
#         test = output(self.model_name, filename)
#         if testname == '':
#             if 'granger' in filename or 'ccm' in filename:
#                 testname = '_'.join(filename.split('.')[:-1])
#             else:
#                 testname = filename.split('.')[0] 
#         setattr(self, testname, test)
        
# class all_output:
#     def add_test_results(self, filename, testname=''):
#         # testname to rename test to make it short. 
#         # instead of making it pearson_mutual_info_tts_tts_naive_59_59_4 can make it test 1, 2, 3, 4
#         test = output(self.model_name, filename)
#         # if testname == '':
#         #     if 'granger' in filename or 'ccm' in filename:
#         #         testname = '_'.join(filename.split('.')[:-1])
#         #     else:
#         #         testname = filename.split('.')[0] 
#         testname = filename.split('.')[0]
#         setattr(self, testname, test)
        
#     def load_all_results(self):  
#         for fi in os.listdir(f'{self.model_name}'): 
#             if fi == 'data.pkl':                
#                 continue 
#             # if 'choose_r' in fi:
#             #     continue
#             # if 'lsa2' in fi:
#             #     continue
#             if '_19_' in fi or '_59_' in fi:
#                 continue
#             if os.path.isdir(f'{self.model_name}/{fi}'):
#                 continue
#             else:
#                 self.add_test_results(fi)
#         self.summarise_output()
        
#     def summarise_output(self):
#         self.summary_output = output(self.model_name)
#         for keys, values in self.__dict__.items(): 
#             if keys == 'summary_output':
#                 continue
#             if isinstance(values, output): 
#                 # pvals = {f"maxlag_{values.test_params['maxlag']}" : values.pvals}
#                 merge(self.summary_output.pvals, values.pvals)
#                 self.summary_output.stats_list.extend([_ for _ in values.stats_list if _ not in self.summary_output.stats_list] )
#                 self.summary_output.test_list.extend([_ for _ in values.test_list if _ not in self.summary_output.test_list ] )
#                 # merge(self.summary_output.test_params, values.test_params)
#                 # self.summary_output.test_params[keys] = values.test_params
#                 # self.summary_output.runtime[keys] = values.runtime
#                 # self.summary_output.count_rejection_rate()
#                 # self.summary_output.count_sbs()
                
#     def __init__(self, model_name):
#         self.model_name = model_name
#         with open(f'{model_name}/data.pkl', 'rb') as file:
#             gData = pickle.load(file)
#         self.series = gData['data']
#         self.datagen_params = gData['datagen_params']
#         self.load_all_results()
# class all_model:            
#     def add_model(self, model_name):
#         setattr(self, model_name, all_output(model_name))
        
#     def summarise_models(self):
#         self.pvals = {}
#         # self.stats_list = []
#         # self.test_list = []
#         self.test_params = {}
#         self.runtime = {}
#         for keys, vals in self.__dict__.items():
#             if isinstance(vals, all_output):
#                 self.pvals[keys] = vals.summary_output.pvals
#                 # self.stats_list.extend([_ for _ in vals.summary_output.stats_list if _ not in self.stats_list] )
#                 # self.test_list.extend([_ for _ in vals.summary_output.test_list if _ not in self.test_list ] )
#                 self.test_params[keys] = vals.summary_output.test_params
#                 self.runtime[keys] = vals.summary_output.runtime
                
#     def __init__(self, model_list):
#         self.model_list = model_list
#         for model in model_list:
#             self.add_model(model)
#         self.summary_models = self.summarise_models()
        
# def into_df(all_modls, model_list = 'default'): # faster than into_df2 - runtime= 0.21683335304260254
#     index = []
#     surrY = []
#     surrX = []
#     if model_list == 'default':
#         model_list = all_modls.model_list
#     for models in model_list:
#         for stat in all_modls.pvals[models]['surrY']: 
#             for cst in all_modls.pvals[models]['surrY'][stat]:
#                 index.append(np.array([models, stat, cst]))
#                 surrY.append(sum( _ <= 0.05 for _ in all_modls.pvals[models]['surrY'][stat][cst]))
#                 surrX.append(sum( _ <= 0.05 for _ in all_modls.pvals[models]['surrX'][stat][cst]))
#     indices = pd.MultiIndex.from_arrays(np.array(index).T, names = ("models", "stats", "surrogate_test"))
#     df = pd.DataFrame({"SurrY": surrY, "SurrX": surrX}, index = indices, dtype="float")
#     return df

# #%%
# # xy_uni_logistic = all_output('xy_uni_logistic')
# model_list = ['xy_uni_logistic', 'xy_Caroline_LV_asym_competitive', 'xy_Caroline_LV_asym_competitive_2', 'xy_Caroline_LV_predprey', 'xy_FitzHugh_Nagumo_cont']#, 'xy_Caroline_LV_asym_competitive_3'
# all_models = all_model(model_list)
# df_all = into_df(all_models)
# stats_list = ['granger_y->x', 'granger_x->y', 'ccm_y->x', 'ccm_x->y']
# cst_list = ['randphase', 'twin', 'tts']
# heatmap(df_all, model_list, stats_list, cst_list)

#%% 
def load_results(model_name, fi_ls = "all"):
    if fi_ls == "all":
        fi_ls = [_ for _ in os.listdir(f'{model_name}') if not os.path.isdir(f'{model_name}/{_}') and "data" not in _ ]# and _ != 'pearson_mi_lsa_randphase_twin_tts.pkl' # saved merged results here so had to skipped it from loading
    results = {}
    for fi in fi_ls:
        name = fi.split('.')[0]
        with open(f'{model_name}/{fi}', 'rb') as file:
            results[name] = pickle.load(file)
    return results
# fi_ls = ['']
xy_uni_logistic = load_results('xy_uni_logistic')
xy_Caroline_LV_asym_competitive = load_results('xy_Caroline_LV_asym_competitive')
xy_Caroline_LV_asym_competitive_2 = load_results('xy_Caroline_LV_asym_competitive_2')
xy_Caroline_LV_predprey = load_results('xy_Caroline_LV_predprey')
xy_FitzHugh_Nagumo_cont = load_results('xy_FitzHugh_Nagumo_cont')
xy_ar_u = load_results('xy_ar_u')
#%%
# def merge_old_res(the_model):
#     old_ls = [_ for _ in the_model if 'ccms_grangers' not in _]
#     oldres = {}
#     oldres['stats_list'] = list(set(['lsa' if stat == 'lsa2' else stat for res in old_ls for stat in the_model[res]['stats_list']]))
#     oldres['test_list'] = list(set([ cst for res in old_ls for cst in the_model[res]['test_list']]))
    
#     intermed = []
#     for res in old_ls:
#         intermed.append({cst : the_model[res]['test_params']
#                          for cst in the_model[res]['test_list']})
#     oldres['test_params'] = merge(*intermed)
                
#     intermed = []
#     for res in old_ls:
#         intermed.append({'pvals': {xory : {cst : {'lsa' if stat == 'lsa2' else stat : [_['pvals'][xory]['lsa' if stat == 'lsa2' else stat][cst] for _ in the_model[res]['pvals']]
#                                                   for stat in the_model[res]['stats_list']}
#                                            for cst in the_model[res]['test_list']} 
#                                    for xory in ['surrY', 'surrX']},
#                          'runtime': {'xory': {'tts': {'pearson_mi': _['runtime']}} for _ in the_model[res]['pvals']} if "choose" in res 
#                          else {xory : {cst : {'lsa' if stat == 'lsa2' else stat : [_['runtime'][xory]['lsa' if stat == 'lsa2' else stat][cst] for _ in the_model[res]['pvals']]
#                                                     for stat in the_model[res]['stats_list']}
#                                              for cst in the_model[res]['test_list']} 
#                                      for xory in ['surrY', 'surrX']}})
#     oldres['pvals'] = merge(*intermed)
    
#     return oldres
# test1 = merge_old_res(test)
#%% 
# def wrapper(model_name):
#     test = load_results(model_name)
#     # test1 = merge_old_res(test)
#     # with open(f'{model_name}/pearson_mi_lsa_randphase_twin_tts.pkl', 'wb') as fi:
#     #     pickle.dump(test1, fi)
#     return merge_old_res(test)
# # model_list = ['xy_uni_logistic', 'xy_Caroline_LV_asym_competitive', 'xy_Caroline_LV_asym_competitive_2', 'xy_Caroline_LV_predprey', 'xy_FitzHugh_Nagumo_cont']
# xy_uni_logistic = wrapper('xy_uni_logistic')
# xy_Caroline_LV_asym_competitive = wrapper('xy_Caroline_LV_asym_competitive')
# xy_Caroline_LV_asym_competitive_2 = wrapper('xy_Caroline_LV_asym_competitive_2')
# xy_Caroline_LV_predprey = wrapper('xy_Caroline_LV_predprey')
# xy_FitzHugh_Nagumo_cont = wrapper('xy_FitzHugh_Nagumo_cont')
# xy_ar_u = wrapper('xy_ar_u')
# wrapper('xy_uni_logistic')
# wrapper('xy_Caroline_LV_asym_competitive')
# wrapper('xy_Caroline_LV_asym_competitive_2')
# wrapper('xy_Caroline_LV_predprey')
# wrapper('xy_FitzHugh_Nagumo_cont')
#%%
# def concat_list_of_dict(list_of_dict):
#     new_dict = {}
#     for _ in list_of_dict: new_dict.update(_)
#     return new_dict
def merge_new_res(the_model):
    stats_list = list(set([_ for res in the_model for _ in the_model[res]['stats_list'] if 'ccms_grangers' in res]))
    test_list = list(set(['tts_naive' if _ == 'tts' else _ for res in the_model for _ in the_model[res]['test_list'] if 'ccms_grangers' in res]))
    intermed = [{_[0] : _[1] for _ in the_model[res]['pvals']} 
                # if type(the_model[res]['pvals'][0]) == list 
                # else concat_list_of_dict(the_model[res]['pvals'])
                for res in the_model if 'ccms_grangers' in res]
    pvals = merge(*intermed)
    return {'stats_list': stats_list, 'test_list': test_list, 'pvalss': pvals}
# test = merge_new_res(xy_ar_u)

# Accounted for tts_naive and mislabelling of surrX, also have to adjust choose_embed
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
    
    new_res['pvals'] = merge(pvals, tts_pvals)
    new_res['test_list'].append('tts')


# Accounted for tts_naive but does not account for mislabelling of surrX
# def extract_pval_from_new_res(new_res):
#     new_res['runtime'] = {xory : {cst : {stat : [trial['runtime'][xory][cst][stat] for idx, trial in new_res['pvalss'].items()]
#                             for stat in new_res['stats_list']}
#                      for cst in new_res['test_list']} 
#              for xory in ['surrY', 'surrX']}
#     new_res['test_params'] = merge(*[trial['test_params'] for idx, trial in new_res['pvalss'].items()])
    
#     pvals = {xory : {'tts_naive' if cst =='tts' else cst : {stat : [trial['score_null_pval'][xory][cst][stat]['pval'] 
#                                                                     for idx, trial in new_res['pvalss'].items()]
#                             for stat in new_res['stats_list']}
#                      for cst in new_res['test_list']} 
#              for xory in ['surrY', 'surrX']}
#     r = new_res['test_params']['surrY']['tts']['surr_params']['r_tts']
#     tts_pvals = {xory : {'tts': {stat : [B * (2 * r + 1) / (r + 1) for B in pvals[xory]['tts_naive'][stat] ]
#                             for stat in new_res['stats_list']}
#                         } 
#                 for xory in ['surrY', 'surrX']}
    
#     new_res['pvals'] = merge(pvals, tts_pvals)
    
# extract_pval_from_new_res(test)
def wrapper2(the_model):
    merged = merge_new_res(the_model)
    extract_pval_from_new_res(merged)
    return merged
xy_uni_logistic_cg = wrapper2(xy_uni_logistic)
xy_Caroline_LV_asym_competitive_cg = wrapper2(xy_Caroline_LV_asym_competitive)
xy_Caroline_LV_asym_competitive_2_cg = wrapper2(xy_Caroline_LV_asym_competitive_2)
xy_Caroline_LV_predprey_cg = wrapper2(xy_Caroline_LV_predprey)
xy_FitzHugh_Nagumo_cont_cg = wrapper2(xy_FitzHugh_Nagumo_cont)
# xy_ar_u_cg = wrapper2(xy_ar_u)

# xy_uni_logistic = wrapper('xy_uni_logistic'
#%%
def combine_old_new(the_model, the_model_cg):
    stats_list = list(set(the_model['pearson_mi_lsa_randphase_twin_tts']['stats_list'] + the_model_cg['stats_list']))
    test_list = list(set(the_model['pearson_mi_lsa_randphase_twin_tts']['test_list'] + the_model_cg['test_list']))
    test_params = merge(the_model['pearson_mi_lsa_randphase_twin_tts']['test_params'], the_model_cg['test_params'])
    pvals = merge(the_model['pearson_mi_lsa_randphase_twin_tts']['pvals']['pvals'], the_model_cg['pvals'])
    return {'pvals': pvals, 'stats_list': stats_list, 'test_list': test_list, 'test_params': test_params}

combine = {}
combine['xy_uni_logistic'] = combine_old_new(xy_uni_logistic, xy_uni_logistic_cg)
combine['xy_Caroline_LV_asym_competitive'] = combine_old_new(xy_Caroline_LV_asym_competitive, xy_Caroline_LV_asym_competitive_cg)
combine['xy_Caroline_LV_asym_competitive_2'] = combine_old_new(xy_Caroline_LV_asym_competitive_2, xy_Caroline_LV_asym_competitive_2_cg)
combine['xy_Caroline_LV_predprey'] = combine_old_new(xy_Caroline_LV_predprey, xy_Caroline_LV_predprey_cg)
combine['xy_FitzHugh_Nagumo_cont'] = combine_old_new(xy_FitzHugh_Nagumo_cont, xy_FitzHugh_Nagumo_cont_cg)
# combine['xy_ar_u'] = combine_old_new(xy_ar_u, xy_ar_u_cg)

#%%
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

df_all = into_df(combine)
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

        plt.savefig(f"im_ccm_granger_rerun/{model}.svg")
model_list = ['xy_uni_logistic', 'xy_Caroline_LV_asym_competitive', 'xy_Caroline_LV_asym_competitive_2', 'xy_Caroline_LV_predprey']# , 'xy_ar_u'
stats_list = ['pearson', 'lsa', 'mutual_info', 'granger_y->x', 'granger_x->y', 'ccm_y->x', 'ccm_x->y']
cst_list = ['randphase', 'twin', 'tts']
heatmap(df_all, model_list, stats_list, cst_list)
