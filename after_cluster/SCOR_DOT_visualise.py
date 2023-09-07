# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 09:52:12 2023

@author: hoang
"""
import os
os.getcwd()
os.chdir('C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/')
# os.chdir('/home/h_k_linh/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/')
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
# from scipy import stats
import dill
# from surrogate_dependence_test import manystats_manysurr

#%% Load data
dill.load_session('Real_data/Common_species_link_global_ecosystems_to_climate_change/SCOR_DOT_cluster_final.pkl')

stats_list = ['pearson', 'lsa', 'mutual_info', 
              'granger_y->x', 'granger_x->y', 
              'ccm_y->x', 'ccm_x->y']
test_list = ['randphase', 'twin', 'tts_naive']
def into_df(res, stats_list=stats_list, test_list=test_list):
    index = []
    surrY = []
    surrX = []
    for stat in stats_list: 
        for cst in test_list:
            index.append(np.array([stat, cst]))
            surrY.append(res['score_null_pval']['surrY'][cst][stat]['pval'])
            surrX.append(res['score_null_pval']['surrX'][cst][stat]['pval'])
    for stat in stats_list:
        index.append(np.array([stat, 'tts']))
        r = res['test_params']['surrY']['tts_naive']['surr_params']['r_tts']
        surrY.append(res['score_null_pval']['surrY']['tts_naive'][stat]['pval'] * (2 * r + 1) / (r + 1))
        r = res['test_params']['surrX']['tts_naive']['surr_params']['r_tts']
        surrX.append(res['score_null_pval']['surrX']['tts_naive'][stat]['pval'] * (2 * r + 1) / (r + 1))
    indices = pd.MultiIndex.from_arrays(np.array(index).T, names = ("stats", "surrogate_test"))
    df = pd.DataFrame({"SurrY": surrY, "SurrX": surrX}, index = indices, dtype="float")
    return df

pvals = into_df(res)
pvals_N = into_df(res_N)
pvals_D = into_df(res_D)
pvals_DN = into_df(res_DN)
#%% Draw heatmap as Caroline
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gs
import matplotlib.patheffects as pe
# import matplotlib.patches as patches
# import seaborn as sns

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

# https://matplotlib.org/stable/tutorials/colors/colorbar_only.html
def heatmap(pvaldf, title, test_list = test_list):
    df = pvaldf.melt(ignore_index= False, var_name = "xory", value_name = "pvals").pivot_table(values = "pvals", index = ["stats"], columns = ["surrogate_test", "xory"])
    col = [(_, xory) for _ in test_list for xory in ["SurrY", "SurrX"]]
    draw = df.loc[:,col]
    
    fig = plt.figure(figsize = (7,5.5), constrained_layout = True) # figsize = (,), constrained_layout = True
    grid = fig.add_gridspec(nrows = 1, 
                            ncols = len(test_list)+1, 
                            width_ratios = [1,1,1,0.25],
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

        texts = annotate_heatmap(im, valfmt="{x:.02f}", threshold=0, **font_data)
    
        if i == 0:
            axs.set_yticks(np.arange(len(stats_list)), labels = stats_list, **font)
        else:
            axs.set_yticks([])
            
        axs.spines[:].set_visible(False)
        axs.set_xticks(np.arange(3)-0.5, minor = True)
        axs.set_yticks(np.arange(len(stats_list)+1)-0.5, minor = True)
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
    
    plt.savefig(f"Real_data/Common_species_link_global_ecosystems_to_climate_change/SCOR_DOT_heatmap/{title}_retts.svg")
#%%
stats_list = ['pearson', 'lsa', 'mutual_info', 
              'granger_y->x', 'granger_x->y', 
              'ccm_y->x', 'ccm_x->y']
test_list = ['randphase', 'twin', 'tts']
heatmap(pvals, "Initial data - not detrended - not normalised", test_list=test_list)
heatmap(pvals_N, "Normalised without detrending data", test_list=test_list)
heatmap(pvals_D, "Detrended without normalisation", test_list=test_list)
heatmap(pvals_DN, "Detrended and normalised data", test_list=test_list)

# def draw_heatmap_xory(xory_pvals, test_list, axes, ax_idx, cbar_ax):
#     heatmap_drawthis = pd.DataFrame.from_dict(xory_pvals, orient = 'index')
#     if test_list != 'default':
#         row = [_ for _ in test_list if _ in heatmap_drawthis.index.values]
#         col = [_ for _ in test_list if _ in heatmap_drawthis.columns.values]
#         if len(row) != 0: 
#             heatmap_drawthis = heatmap_drawthis.loc[row,:]
#         if len(col) != 0:
#             heatmap_drawthis = heatmap_drawthis.loc[:,col]                
            
#     sns.heatmap(heatmap_drawthis, 
#                 ax = axes[ax_idx], 
#                 vmin = 0, vmax = 1,
#                 annot = True, 
#                 cbar = True, cbar_ax= cbar_ax,
#                 linewidths = 2,
#                 cmap="Purples")

# fig, axes = plt.subplots(1,2, 
#                          sharey= True, 
#                          figsize = (7.5,3))
# cbar_ax = fig.add_axes([1,.1,.03,.8]) # See Drafts.py
# fig.tight_layout() 
# ax_idx = 0
# for key, xory_val in results.items():
#     draw_heatmap_xory(xory_val, test_list=['randphase', 'twin', 'tts', 'tts_naive'], axes = axes, ax_idx = ax_idx, cbar_ax = cbar_ax)
#     axes[ax_idx].set_title(key)
#     ax_idx +=1