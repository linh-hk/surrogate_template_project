# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 08:21:34 2023

@author: Linh Hoang

Load test results, use tts_naive to calculate tts.
Visualise
"""
import os
os.getcwd()
# os.chdir('C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/')
os.chdir('/home/h_k_linh/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/')
os.getcwd()
import glob
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from mergedeep import merge

import pickle
#%%
ficap = 'equal_cap_0.25_500_2.0,0.0_0.7,0.7_-0.4,-0.5,-0.5,-0.4_0.01_0.05'
fiuncap = 'equal_ncap_0.25_500_2.0,0.0_0.7,0.7_-0.4,-0.5,-0.5,-0.4_0.01_0.05'
class results:
    def __init__(self, sample):
        results = []
        for fi in os.listdir(f'Simulated_data/LVextra/{sample}/'):
            if 'data' in fi:
                continue
            else:
                with open(f'Simulated_data/LVextra/{sample}/{fi}', 'rb') as file:
                    results.append(pickle.load(file))
        
        stat_list = ('pearson', 'lsa', 'mutual_info', 
                      'granger_y->x', 'granger_x->y', 
                      'ccm_y->x', 'ccm_x->y')
        test_list = ('randphase', 'twin', 'tts_naive') 
        intermed = [_ for sublist in results for _ in sublist['pvals'] ]
        pvalss = merge(*intermed)
        
        runtime = {xory : {cst : 
                                      {stat : [trial['runtime'][xory][cst][stat] 
                                               for idx, trial in pvalss.items()]
                                for stat in stat_list}
                         for cst in test_list} 
                 for xory in ['surrY', 'surrX']}
        test_params = merge(*[trial['test_params'] for idx, trial in pvalss.items()])
        
        pvals = {xory : {cst : {stat : [trial['score_null_pval'][xory][cst][stat]['pval'] 
                                        for idx, trial in pvalss.items()]
                                for stat in stat_list}
                         for cst in test_list} 
                 for xory in ['surrY', 'surrX']}
        r = test_params['surrY']['tts_naive']['surr_params']['r_tts']
        tts_pvals = {xory : {'tts': {stat : [B * (2 * r + 1) / (r + 1) for B in pvals[xory]['tts_naive'][stat] ] 
                                     for stat in stat_list}
                            } 
                     for xory in ['surrY', 'surrX']}
        test_list = (*test_list, 'tts')
        pvals = merge(pvals, tts_pvals)
        
        # self.results = {'stat_list': stat_list, 'test_list': test_list, 'test_params': test_params, 'runtime': runtime, 'pvals': pvals}
        self.stat_list = stat_list
        self.test_list = test_list
        self.test_params = test_params
        self.runtime = runtime
        self.pvals = pvals
        
    def into_df(self, model_name = ''): # faster than into_df2 - runtime= 0.21683335304260254
        index = []
        surrY = []
        surrX = []
        # if model_list == 'default':
        #     model_list = all_modls.model_list
        
        for cst in self.test_list:
            for stat in self.stat_list: 
                index.append([stat, cst])
                # if stat in ['pearson', 'mutual_info', 'lsa']:
                    # surrY.append(sum( _ <= 0.05 for _ in self.pvals['surrY'][cst][stat][:100]))
                    # surrX.append(sum( _ <= 0.05 for _ in self.pvals['surrX'][cst][stat][:100]))
                # else:
                surrY.append(sum( _ <= 0.05 for _ in self.pvals['surrY'][cst][stat]))
                surrX.append(sum( _ <= 0.05 for _ in self.pvals['surrX'][cst][stat]))
        
        indices = pd.MultiIndex.from_arrays(np.array(index).T, names = ("stats", "surrogate_test"))
        if model_name != '':
            idf = indices.to_frame()
            idf.insert(0, 'model', model_name) 
            indices = pd.MultiIndex.from_frame(idf, names = ("models", "stats", "surrogate_test"))
        self.df = pd.DataFrame({"SurrY": surrY, "SurrX": surrX}, index = indices, dtype="float")
        
cap = results(ficap)
cap.into_df('cap')
uncap = results(fiuncap)
uncap.into_df('uncap')
#%%
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

# https://matplotlib.org/stable/tutorials/colors/colorbar_only.html
def heatmap(pvaldf, title, test_list, stat_list):
    df = pvaldf.melt(ignore_index= False, var_name = "xory", value_name = "pvals").pivot_table(values = "pvals", index = ["stats"], columns = ["surrogate_test", "xory"])
    col = [(_, xory) for _ in test_list for xory in ["SurrY", "SurrX"]]
    draw = df.loc[:,col]
    
    fig = plt.figure(figsize = (7,5.5), constrained_layout = True) # figsize = (,), constrained_layout = True
    grid = fig.add_gridspec(nrows = 1, 
                            ncols = len(test_list)+1, 
                            width_ratios = [1]*len(test_list)+[0.25],
                            hspace = 0, wspace = 0.05)
    # cmap = mpl.cm.cool.with_extremes(over='0.25', under="0.75")
    cmap = (mpl.colors.ListedColormap(['magenta'])).with_extremes(under='cyan')
    bounds = [0.0500000001, 1]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    for i, cst in enumerate(test_list):
        axs = fig.add_subplot(grid[i])
        axs.set_aspect("equal")
        im = axs.imshow(draw[[cst]], cmap = cmap, norm = norm)
        # im = axs.imshow(draw[[cst]], cmap = "cool", vmin = 0, vmax = 1)
        axs.set_xlabel(cst, **font)
        # axs.set_title(cst, **font)
        
        axs.set_xticks([0,1], labels = ["surrY", "surrX"], **font)
        axs.tick_params(top = True, labeltop=True, bottom = False, labelbottom=False)
        # ax.xaxis.set_ticks_position('top')
        plt.setp(axs.get_xticklabels(), rotation=30, ha="left", rotation_mode = "anchor")

        texts = annotate_heatmap(im, valfmt="{x:.0f}", threshold=0, **font_data)
    
        if i == 0:
            axs.set_yticks(np.arange(len(stat_list)), labels = stat_list, **font)
        else:
            axs.set_yticks([])
            
        axs.spines[:].set_visible(False)
        axs.set_xticks(np.arange(3)-0.5, minor = True)
        axs.set_yticks(np.arange(len(stat_list)+1)-0.5, minor = True)
        axs.grid(which = "minor", color='black', linestyle='-', linewidth=2)
        axs.tick_params(which =  "minor", bottom = False, left = False)
        
    
    fig.suptitle(title, **font) 
    # fig.subplots_adjust(right=0.8) 
    cbar_ax = fig.add_subplot(grid[-1]) 
    cbar = fig.colorbar(im,
                        cax=cbar_ax, 
                        extend='min', 
                        ticks=bounds, 
                        spacing='proportional', 
                        label='pvalue')
    # cbar = fig.colorbar(im, cax=cbar_ax, ticks = [0.05, 0.25,0.50,0.75,1])
    cbar.ax.tick_params(labelsize = 16)
    plt.show()
#%%
heatmap(cap.df, 'cap', ['randphase', 'twin', 'tts'], cap.stat_list)
heatmap(uncap.df, 'uncap', ['randphase', 'twin', 'tts'], uncap.stat_list)
