# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 08:21:34 2023

@author: Linh Hoang

Load test results, use tts_naive to calculate tts.
Visualise
"""
import os
os.getcwd()
os.chdir('D:/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/')
# os.chdir('/home/h_k_linh/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/')
os.getcwd()
import glob
import numpy as np
import pandas as pd 
import scipy as sp
import matplotlib.pyplot as plt
# import seaborn as sns
from mergedeep import merge

import pickle
#%% Draw heatmap as Caroline
import matplotlib as mpl
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gs
import matplotlib.patheffects as pe
import matplotlib.patches as patches
# import seaborn as sns
#%%
# Caroline configuration
plt.rcParams['svg.fonttype'] = 'none'
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

#%% Load results
from sklearn.metrics import cohen_kappa_score
# https://www.statology.org/cohens-kappa-python/
def CohenKappa(X, Y):
    # https://www.statisticshowto.com/cohens-kappa-statistic/
    if len(X) == len(Y):
        total = len(X)
        bothyes     = 0 
        bothno      = 0 
        Xyes        = 0
        Yyes        = 0
        for i in range(len(X)):
            if X[i] <= 0.05:
                Xyes += 1
                if Y[i] <= 0.05:
                    Yyes += 1
                    bothyes +=1
            else:
                if Y[i] <= 0.05:
                    Yyes += 1
                else:
                    bothno += 1
        P_o = (bothyes+bothno)/total
        P_e = (Xyes*Yyes + (total - Xyes)*(total-Yyes))/total**2
        if 1 - P_e !=0 :
            return bothyes, bothno, (P_o-P_e)/(1-P_e)
        else:
            return bothyes, bothno, None
    else:
        print("X and Y must be same length")

def chi_test2(cstY_cstX, n_trial):
    # print("new")
    # YX = cstY_cstX.sort_index(ascending = False, level = ["xory"])
                # print(YX.all(axis = 0))
    cstY_cstX = cstY_cstX[[0,1]]
    lala = np.array([cstY_cstX,n_trial-cstY_cstX]) 
    print(lala)
    # if 1000 in lala:
    #     return 1000
    try:
        return sp.stats.chi2_contingency(lala, correction = False).pvalue #
    except:
        return n_trial
    
def rename(sample):
    # match sample: # only works for python 3.10 onward :<
    #     case 'xy_Caroline_LV_asym_competitive':
    #         return 'UComp_1.25_500_1.0,1.0_0.8,0.8_-0.4,-0.5,-0.9,-0.4_0.01_0.05'
    #     case 'xy_Caroline_LV_asym_competitive_2':
    #         return 'UComp2_1.25_500_1.0,1.0_0.8,0.8_-1.4,-0.5,-0.9,-1.4_0.01_0.05'
    #     case 'xy_Caroline_LV_asym_competitive_3':
    #         return 'UComp3_1.25_500_1.0,1.0_50.0,50.0_-100,-95,-99,-100_0.01_0.05'
    #     case 'xy_Caroline_LV_competitive':
    #         return 'EComp_1.25_500_1.0,1.0_0.7,0.7_-0.4,-0.5,-0.5,-0.4_0.01_0.05'
    #     case 'xy_Caroline_LV_mutualistic':
    #         return 'EMut_1.25_500_1.0,1.0_0.7,0.7_-0.4,0.3,0.3,-0.4_0.01_0.05'
    #     case _:
    #         return sample
    if sample == 'xy_Caroline_LV_asym_competitive':
        return 'UComp_1.25_500_1.0,1.0_0.8,0.8_-0.4,-0.5,-0.9,-0.4_0.01_0.05'
    if sample == 'xy_Caroline_LV_asym_competitive_2':
        return 'UComp2_1.25_500_1.0,1.0_0.8,0.8_-1.4,-0.5,-0.9,-1.4_0.01_0.05'
    if sample == 'xy_Caroline_LV_asym_competitive_3':
        return 'UComp3_1.25_500_1.0,1.0_50.0,50.0_-100,-95,-99,-100_0.01_0.05'
    if sample == 'xy_Caroline_LV_competitive':
        return 'EComp_1.25_500_1.0,1.0_0.7,0.7_-0.4,-0.5,-0.5,-0.4_0.01_0.05'
    if sample == 'xy_Caroline_LV_mutualistic':
        return 'EMut_1.25_500_1.0,1.0_0.7,0.7_-0.4,0.3,0.3,-0.4_0.01_0.05'
    else:
        return sample
    
def merge_data(results, test_list, stat_list):
    maxlag = set([sublist['pvals'][0][list(sublist['pvals'][0])  [0]]['test_params']['surrY'][sublist['test_list'][0]]['maxlag']
                  for sublist in results])
    pvalss = {}
    for ml in maxlag:
        intermed = [_ for sublist in results for _ in sublist['pvals'] if 
                    sublist['pvals'][0][ list(sublist['pvals'][0])[0] ]['test_params']['surrY'][sublist['test_list'][0]]['maxlag'] == ml]
        pvalss[f'maxlag{ml}'] = merge(*intermed)           
    
    try:
        runtime = {lagkey: {xory : {cst : {stat : [trial['runtime'][xory][cst][stat] 
                                            for idx, trial in lagitm.items()]
                                    for stat in stat_list}
                             for cst in test_list}
                      for xory in ['surrY', 'surrX']}
                 for lagkey, lagitm in pvalss.items()}
    except:
        runtime = 'not available'
    test_params = {lagkey: merge(*[trial['test_params'] for key,trial in lagitm.items() if key != 'XY']) for lagkey, lagitm in pvalss.items()}
    
    pvals = {lagkey: {xory : {cst : {stat : [trial['score_null_pval'][xory][cst][stat]['pval'] 
                                        for idx, trial in lagitm.items() if idx != 'XY']
                                for stat in stat_list}
                         for cst in test_list}
                  for xory in ['surrY', 'surrX']}
             for lagkey, lagitm in pvalss.items()}
    r = {lagkey: lagitm['surrY']['tts_naive']['surr_params']['r_tts']
         for lagkey, lagitm in test_params.items()}
    tts_pvals = {lagkey: {xory : {'tts': {stat : [B * (2 * r[lagkey] + 1) / (r[lagkey] + 1) for B in lagitm[xory]['tts_naive'][stat] ]
                                          for stat in stat_list}
                                  }
                          for xory in ['surrY', 'surrX']}
                 for lagkey, lagitm in pvals.items()}
    test_list = (*test_list, 'tts')
    pvals = merge(pvals, tts_pvals)
    return maxlag, runtime, test_params, test_list, pvals
    
class results:
    def __init__(self, sample, excl = ['falsepos'], incl = ['normalised']):
        if 'xy_' in sample:
            sampdir = f'Simulated_data/{sample}/normalised'
        elif '500' in sample:
            sampdir = f'Simulated_data/LVextra/{sample}/normalised'
        
        results = []
        excl.append('data')
        for fi in os.listdir(sampdir):
            if any(pat in fi for pat in excl):
                continue
            elif any(pat in fi for pat in incl):
                print(f'Loading:{sampdir}/{fi}')
                with open(f'{sampdir}/{fi}', 'rb') as file:
                    results.append(pickle.load(file))
        '''falsepos = []
        for fi in os.listdir(sampdir):
            if 'falsepos' in fi:                
                print(f'Loading:{sampdir}/{fi}')
                with open(f'{sampdir}/{fi}', 'rb') as file:
                    falsepos.append(pickle.load(file))'''
        
        stat_list = ('pearson', 'lsa', 'mutual_info', 
                      'granger_y->x', 'granger_x->y', 
                      'ccm_y->x', 'ccm_x->y')
        test_list = ('randphase', 'twin', 'tts_naive') 
        # maxlag = set([sublist['pvals'][0][list(sublist['pvals'][0])  [0]]['test_params']['surrY'][sublist['test_list'][0]]['maxlag']
        #               for sublist in results])
        # pvalss = {}
        # for ml in maxlag:
        #     intermed = [_ for sublist in results for _ in sublist['pvals'] if 
        #                 sublist['pvals'][0][ list(sublist['pvals'][0])[0] ]['test_params']['surrY'][sublist['test_list'][0]]['maxlag'] == ml]
        #     pvalss[f'maxlag{ml}'] = merge(*intermed)           
        
        # try:
        #     runtime = {lagkey: {xory : {cst : {stat : [trial['runtime'][xory][cst][stat] 
        #                                         for idx, trial in lagitm.items()]
        #                                 for stat in stat_list}
        #                          for cst in test_list}
        #                   for xory in ['surrY', 'surrX']}
        #              for lagkey, lagitm in pvalss.items()}
        # except:
        #     runtime = 'not available'
        # test_params = {lagkey: merge(*[trial['test_params'] for trial in lagitm.values()]) for lagkey, lagitm in pvalss.items()}
        
        # pvals = {lagkey: {xory : {cst : {stat : [trial['score_null_pval'][xory][cst][stat]['pval'] 
        #                                     for idx, trial in lagitm.items()]
        #                             for stat in stat_list}
        #                      for cst in test_list}
        #               for xory in ['surrY', 'surrX']}
        #          for lagkey, lagitm in pvalss.items()}
        # r = {lagkey: lagitm['surrY']['tts_naive']['surr_params']['r_tts']
        #      for lagkey, lagitm in test_params.items()}
        # tts_pvals = {lagkey: {xory : {'tts': {stat : [B * (2 * r[lagkey] + 1) / (r[lagkey] + 1) for B in lagitm[xory]['tts_naive'][stat] ]
        #                                       for stat in stat_list}
        #                               }
        #                       for xory in ['surrY', 'surrX']}
        #              for lagkey, lagitm in pvals.items()}
        # test_list = (*test_list, 'tts')
        # pvals = merge(pvals, tts_pvals)
        
        maxlag, runtime0, test_params, test_list0, pvals = merge_data(results, test_list, stat_list)
        ''' fp_maxlag, fp_runtime, fp_test_params, fp_test_list, fp_pvals = merge_data(falsepos, test_list, stat_list)'''
        
        # self.results = {'stat_list': stat_list, 'test_list': test_list, 'test_params': test_params, 'runtime': runtime, 'pvals': pvals}
        self.stat_list = stat_list
        self.test_list = test_list0
        self.test_params = test_params
        self.runtime = runtime0
        self.maxlag = list(maxlag)
        self.pvals = pvals
        '''self.fp_maxlag = fp_maxlag
        self.fp_test_list = fp_test_list
        self.fp_runtime = fp_runtime
        self.fp_test_params = fp_test_params
        self.fp_pvals = fp_pvals'''
        self.n_trial = list(set([len(sublist['pvals']) for sublist in results]))
        
    def into_df(self, model_name = '', falsepos = False): # faster than into_df2 - runtime= 0.21683335304260254
        if falsepos:
            pvals = self.fp_pvals
            maxlag = self.fp_maxlag
            test_list = self.fp_test_list
        else:
            pvals = self.pvals
            maxlag = self.maxlag
            test_list = self.test_list
        index = []
        surrY = []
        surrX = []
        for ml in maxlag:
            for cst in test_list:
                for stat in self.stat_list: 
                    index.append([ml, stat, cst])
                    # if stat in ['pearson', 'mutual_info', 'lsa']:
                        # surrY.append(sum( _ <= 0.05 for _ in self.pvals['surrY'][cst][stat][:100]))
                        # surrX.append(sum( _ <= 0.05 for _ in self.pvals['surrX'][cst][stat][:100]))
                    # else:
                    surrY.append(sum( _ <= 0.05 for _ in pvals[f'maxlag{ml}']['surrY'][cst][stat]))
                    surrX.append(sum( _ <= 0.05 for _ in pvals[f'maxlag{ml}']['surrX'][cst][stat]))
            
        indices = pd.MultiIndex.from_arrays(np.array(index).T, names = ("maxlag", "stats", "surrogate_test"))
        if model_name != '':
            idf = indices.to_frame()
            idf.insert(0, 'model', model_name) 
            indices = pd.MultiIndex.from_frame(idf, names = ("models", "maxlag", "stats", "surrogate_test"))
        if falsepos:
            self.fp_df = pd.DataFrame({"SurrY": surrY, "SurrX": surrX}, index = indices, dtype="float")
        else:
            self.df = pd.DataFrame({"SurrY": surrY, "SurrX": surrX}, index = indices, dtype="float")
        # Res['xy_ar_u'].df.loc[('0', 'pearson', 'randphase'),"SurrX"]
        
    def runtime_df(self, model_name = '', summhow = np.mean, falsepos = False):
        index = []
        surrY = []
        surrX = []
        
        test_list = ('randphase', 'twin', 'tts_naive') 
        
        if falsepos:
            runtime = self.fp_runtime
            maxlag = self.fp_maxlag
            
        else:
            runtime = self.runtime
            maxlag = self.maxlag
            
        
        for ml in maxlag:
            for cst in test_list:
                for stat in self.stat_list: 
                    index.append([ml, stat, cst])
                    # if stat in ['pearson', 'mutual_info', 'lsa']:
                        # surrY.append(sum( _ <= 0.05 for _ in self.pvals['surrY'][cst][stat][:100]))
                        # surrX.append(sum( _ <= 0.05 for _ in self.pvals['surrX'][cst][stat][:100]))
                    # else:
                    surrY.append(summhow(runtime[f'maxlag{ml}']['surrY'][cst][stat]))
                    surrX.append(summhow(runtime[f'maxlag{ml}']['surrX'][cst][stat]))
        indices = pd.MultiIndex.from_arrays(np.array(index).T, names = ("maxlag", "stats", "surrogate_test"))
        if model_name != '':
            idf = indices.to_frame()
            idf.insert(0, 'model', model_name) 
            indices = pd.MultiIndex.from_frame(idf, names = ("models", "maxlag", "stats", "surrogate_test"))
        if falsepos:     
            self.fp_df_runtime = pd.DataFrame({"SurrY": surrY, "SurrX": surrX}, index = indices, dtype="float")
        else:
            self.df_runtime = pd.DataFrame({"SurrY": surrY, "SurrX": surrX}, index = indices, dtype="float")
            
    def add_CohenKappa(self):
        ck = []
        for ml,stat,cst in self.df.index:
            # ck.append(CohenKappa(self.pvals[f'maxlag{ml}']['surrY'][cst][stat], 
            #                      self.pvals[f'maxlag{ml}']['surrX'][cst][stat])) 
            ck.append(cohen_kappa_score(np.array(self.pvals[f'maxlag{ml}']['surrY'][cst][stat]) <= 0.05,
                                        np.array(self.pvals[f'maxlag{ml}']['surrX'][cst][stat]) <= 0.05))
        self.df['CohenKappa'] = ck
        self.df['CohenKappa'] = self.df['CohenKappa'].fillna(1)
        
    def add_chi2(self):
        self.df['chi2'] = self.df.apply(chi_test2, n_trial = self.n_trial[0], axis = 1)
        
def testing(results):
    ck = []
    for ml,stat,cst in results.df.index:
        ck.append(CohenKappa(results.pvals[f'maxlag{ml}']['surrY'][cst][stat], 
                              results.pvals[f'maxlag{ml}']['surrX'][cst][stat])) 
        # ck.append(cohen_kappa_score(np.array(results.pvals[f'maxlag{ml}']['surrY'][cst][stat]) <= 0.05,
        #                             np.array(results.pvals[f'maxlag{ml}']['surrX'][cst][stat]) <= 0.05))
    return [_[0] for _ in ck], [_[1] for _ in ck], [_[2] for _ in ck]

#%% Load results
if __name__ == "__main__":
    # ignore = ['EComp_0.25_500_1.0,1.0_0.7,0.7_-0.4,-0.5,-0.5,-0.4_0.01_0.05', 
    #           'EMut_0.25_500_2.0,0.0_0.7,0.7_-0.4,0.3,0.3,-0.4_0.01_0.05', 
    #           'equal_ncap_0.25_500_2.0,0.0_0.7,0.7_-0.4,-0.5,-0.5,-0.4_0.01_0.05']
    Res = {}
    # for sample in os.listdir('Simulated_data'):
    #     if 'xy' in sample and 'predprey' not in sample:
    #         renamed = rename(sample)
    #         print(sample)
    #         Res[renamed]=results(sample)
    # for sample in os.listdir('Simulated_data/LVextra'):
    # for sample in ['EComp_0.25_500_1.0,1.0_0.7,0.7_-0.4,-0.5,-0.5,-0.4_0.01_0.05','EMut_0.25_500_2.0,0.0_0.7,0.7_-0.4,0.3,0.3,-0.4_0.01_0.05']:
    for sample in ['equal_ncap_0.25_500_2.0,0.0_0.7,0.7_-0.4,-0.5,-0.5,-0.4_0.01_0.05']:
        Res[sample]=results(sample)
    #     if '500' in sample:# and sample not in ignore # load falsepos
    #         print(sample)
    #         Res[sample]=results(sample)
    
    for key, val in Res.items():
    #     print(key)
        val.into_df()
        val.into_df(falsepos = True)
    #     val.add_CohenKappa()
    #     val.add_chi2()
        
    # test1 = testing(Res['xy_ar_u'])
    # for key, val in Res.items():
    #     testing(val)

#%% Draw heatmaps
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

def add_hash_ckchi2(df, axs):
    # Chi2 squared test add hatch
    ckchi2_cordinates = np.where(df <= 0.05)[0]
    for j in ckchi2_cordinates:
        # print(j)
        axs.add_patch(patches.Rectangle((-0.49,j-0.49),2,1, hatch = '--', edgecolor = 'white', fill = False))
    # return axs

# https://matplotlib.org/stable/tutorials/colors/colorbar_only.html
def heatmap(pvaldf, title, test_list, stat_list, ckchi2 = False, falsepos = False, savehere = '', filextsn = 'svg'):
    df = pvaldf.melt(ignore_index= False, var_name = "xory", value_name = "pvals").pivot_table(values = "pvals", index = ["maxlag", "stats"], columns = ["surrogate_test", "xory"])#.groupby("maxlag")
    maxlag = set(df.index.get_level_values(0))
    col = [(_, xory) for _ in test_list for xory in ["SurrY", "SurrX"]]
    print(title)
    
    for ml in maxlag: 
        row = [(ml, _) for _ in stat_list]
        draw = df.loc[row,col]
        print(draw, "oh yea\n")
    
        fig = plt.figure(figsize = (7,5.5), constrained_layout = True) # figsize = (,), constrained_layout = True
        grid = fig.add_gridspec(nrows = 1, 
                                ncols = len(test_list)+1, 
                                width_ratios = [1]*len(test_list)+[0.25],
                                hspace = 0, wspace = 0.05)
        # cmap = mpl.cm.cool.with_extremes(over='0.25', under="0.75")
        # cmap = (mpl.colors.ListedColormap(['magenta']))#.with_extremes(under='cyan')
        # bounds = [0, 100]
        # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cmap = mpl.cm.cool #viridis 
        bounds = [0,10.001,20.001, 30.001, 40.001, 50.001, 60.001, 70.001, 80.001, 90.001, 100.001] 
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
            
            if ckchi2:
                if ckchi2 == 'CohenKappa' or ckchi2 == 'ck':
                    colckchi2 = [(cst, 'CohenKappa') for _ in test_list for xory in ["SurrY", "SurrX"]]
                if ckchi2 == 'chi2':
                    colckchi2 = [(cst, 'chi2') for _ in test_list for xory in ["SurrY", "SurrX"]]
                add_hash_ckchi2(df.loc[row,colckchi2], axs)
        
            if i == 0:
                axs.set_yticks(np.arange(len(stat_list)), labels = stat_list, **font)
            else:
                axs.set_yticks([])
                
            axs.spines[:].set_visible(False)
            axs.set_xticks(np.arange(3)-0.5, minor = True)
            axs.set_yticks(np.arange(len(stat_list)+1)-0.5, minor = True)
            axs.grid(which = "minor", color='black', linestyle='-', linewidth=2)
            axs.tick_params(which =  "minor", bottom = False, left = False)
            
        
        if falsepos:
            fig.suptitle(f'{title}_fp, maxlag {ml}', **font) 
        else:
            fig.suptitle(f'{title}, maxlag {ml}', **font) 
        # fig.subplots_adjust(right=0.8) 
        cbar_ax = fig.add_subplot(grid[-1]) 
        cbar = fig.colorbar(im,
                            cax=cbar_ax, 
                            # extend='min', 
                            ticks=bounds, 
                            spacing='proportional' #, label='pvalue'
                            )
        # cbar = fig.colorbar(im, cax=cbar_ax, ticks = [0.05, 0.25,0.50,0.75,1])
        cbar.ax.tick_params(labelsize = 16)
        cbar.ax.set_ylabel('true positive counts',**font)
        
        
        if savehere == '':
            if ckchi2:
                savehere = f"Simulated_data/Figures/{heatmap}_{ckchi2}"
            if falsepos:
                title = title + '_fp' # load falsepos
                savehere = "Simulated_data/Figures/falsepos"
            else:
                savehere = "Simulated_data/Figures/heatmap_results"
        plt.savefig(f"{savehere}/{title}_ml{ml}.{filextsn}") 
        plt.show()
        
def heatmap2(pvaldf, title, test_list, stat_list): # , falsepos = False
    df = pvaldf.melt(ignore_index= False, var_name = "xory", value_name = "pvals").pivot_table(values = "pvals", index = ["maxlag", "stats"], columns = ["surrogate_test", "xory"])#.groupby("maxlag")
    maxlag = set(df.index.get_level_values(0))
    col = [(_, xory) for _ in test_list for xory in ["SurrY", "SurrX"]]
    print(title)
    
    for ml in maxlag: 
        row = [(ml, _) for _ in stat_list]
        draw = df.loc[row,col]
        # print(draw, "oh yea\n")
    
        fig = plt.figure(figsize = (7,5.5), constrained_layout = True) # figsize = (,), constrained_layout = True
        grid = fig.add_gridspec(nrows = 1, 
                                ncols = 3, 
                                width_ratios = [1,1,0.25],
                                hspace = 0, wspace = 0.05)
        # cmap = mpl.cm.cool.with_extremes(over='0.25', under="0.75")
        # cmap = (mpl.colors.ListedColormap(['magenta']))#.with_extremes(under='cyan')
        # bounds = [0, 100]
        # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cmap = mpl.cm.cool #viridis 
        bounds = [0,10.001,20.001, 30.001, 40.001, 50.001, 60.001, 70.001, 80.001, 90.001, 100.001] 
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        for i, xory in enumerate(['SurrY', 'SurrX']):
            axs = fig.add_subplot(grid[i])
            axs.set_aspect("equal")
            im = axs.imshow(draw.loc(axis=1)[:,xory], cmap = cmap, norm = norm)
            
            # axs.set_xlabel(xory, **font)
            axs.set_title(xory, **font)
            
            axs.set_xticks([0,1,2], labels = test_list, **font)
            axs.tick_params(top = False, labeltop=False, bottom = True, labelbottom=True)
            # ax.xaxis.set_ticks_position('bottom')
            plt.setp(axs.get_xticklabels(), rotation=-30, ha="left", rotation_mode = "anchor")
    
            texts = annotate_heatmap(im, valfmt="{x:.0f}", threshold=0, **font_data)
            
            if i == 0:
                axs.set_yticks(np.arange(len(stat_list)), labels = stat_list, **font)
            else:
                axs.set_yticks([])
                
            axs.spines[:].set_visible(False)
            axs.set_xticks(np.arange(4)-0.5, minor = True)
            axs.set_yticks(np.arange(len(stat_list)+1)-0.5, minor = True)
            axs.grid(which = "minor", color='black', linestyle='-', linewidth=2)
            axs.tick_params(which =  "minor", bottom = False, left = False)
            
        
        fig.suptitle(f'{title}, maxlag {ml}', **font) 
        # fig.subplots_adjust(right=0.8) 
        cbar_ax = fig.add_subplot(grid[-1]) 
        cbar = fig.colorbar(im,
                            cax=cbar_ax, 
                            # extend='min', 
                            ticks=bounds, 
                            spacing='proportional' #, label='pvalue'
                            )
        # cbar = fig.colorbar(im, cax=cbar_ax, ticks = [0.05, 0.25,0.50,0.75,1])
        cbar.ax.tick_params(labelsize = 16)
        cbar.ax.set_ylabel('true positive counts',**font)

        savehere = f"Simulated_data/Figures/heatmap2_results/{title}_ml{ml}.svg"
        plt.savefig(f"{savehere}")
        print(f'saved at {savehere}')
        plt.show()
        
def heatmap3(pvaldf, title, test_list, stat_list): # , falsepos = False
    df = pvaldf.melt(ignore_index= False, var_name = "xory", value_name = "pvals").pivot_table(values = "pvals", index = ["maxlag", "stats"], columns = ["surrogate_test", "xory"])#.groupby("maxlag")
    maxlag = set(df.index.get_level_values(0))
    col = [(_, xory) for _ in test_list for xory in ["SurrY", "SurrX"]]
    print(title)
    
    for ml in maxlag: 
        row = [(ml, _) for _ in stat_list]
        draw = df.loc[row,col]
        # print(draw, "oh yea\n")
    
        fig = plt.figure(figsize = (8,5.5), constrained_layout = True) # figsize = (,), constrained_layout = True
        grid = fig.add_gridspec(nrows = 1, 
                                ncols = len(col) +1, 
                                width_ratios = [1]*len(col) + [0.25],
                                hspace = 0, wspace = 0.05)
        # cmap = mpl.cm.cool.with_extremes(over='0.25', under="0.75")
        # cmap = (mpl.colors.ListedColormap(['magenta']))#.with_extremes(under='cyan')
        # bounds = [0, 100]
        # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cmap = mpl.cm.cool #viridis 
        bounds = [0,10.001,20.001, 30.001, 40.001, 50.001, 60.001, 70.001, 80.001, 90.001, 100.001] 
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        
        for i, oh in enumerate(draw.columns):
            axs = fig.add_subplot(grid[i])
            axs.set_aspect("equal")
            im = axs.imshow(draw[[oh]], cmap = cmap, norm = norm)
            
            # axs.set_xlabel(' '.join(oh), **font)
            # axs.set_title(' '.join(oh), **font)
            
            axs.set_xticks([0], labels = [' '.join(oh)], **font)
            axs.tick_params(top = False, labeltop=False, bottom = True, labelbottom=True)
            # ax.xaxis.set_ticks_position('bottom')
            plt.setp(axs.get_xticklabels(), rotation=90, ha="left", rotation_mode = "anchor")
    
            texts = annotate_heatmap(im, valfmt="{x:.0f}", threshold=0, **font_data)
            
            if i == 0:
                axs.set_yticks(np.arange(len(stat_list)), labels = stat_list, **font)
            else:
                axs.set_yticks([])
                
            axs.spines[:].set_visible(False)
            axs.set_xticks(np.arange(2)-0.5, minor = True)
            axs.set_yticks(np.arange(len(stat_list)+1)-0.5, minor = True)
            axs.grid(which = "minor", color='black', linestyle='-', linewidth=2)
            axs.tick_params(which =  "minor", bottom = False, left = False)
            
        
        fig.suptitle(f'{title}, maxlag {ml}', **font) 
        # fig.subplots_adjust(right=0.8) 
        cbar_ax = fig.add_subplot(grid[-1]) 
        cbar = fig.colorbar(im,
                            cax=cbar_ax, 
                            # extend='min', 
                            ticks=bounds, 
                            spacing='proportional' #, label='pvalue'
                            )
        # cbar = fig.colorbar(im, cax=cbar_ax, ticks = [0.05, 0.25,0.50,0.75,1])
        cbar.ax.tick_params(labelsize = 16)
        cbar.ax.set_ylabel('true positive counts',**font)

        savehere = f"Simulated_data/Figures/heatmap3_results/{title}_ml{ml}.svg"
        plt.savefig(f"{savehere}")
        # print(f'saved at {savehere}')
        plt.show()
        
#%% Draw heatmaps for
if __name__ == "__main__":
    stat_list = ('pearson', 'lsa', 'mutual_info', 'granger_y->x', 'granger_x->y','ccm_y->x', 'ccm_x->y')
    test_list = ('randphase', 'twin', 'tts')
    # heatmap(cap.df, 'cap', ['randphase', 'twin', 'tts'], cap.stat_list)
    # heatmap(uncap.df, 'uncap', ['randphase', 'twin', 'tts'], uncap.stat_list)
    for key, val in Res.items():
        heatmap(val.fp_df, key, test_list, stat_list, falsepos = True)
        heatmap(val.df, key, ('randphase', 'twin', 'tts'), stat_list)
        # if 'UComp3' in key:
        #     heatmap(val.df, key, ('randphase', 'twin', 'tts'), stat_list) # , ckchi2='ck' or 'chi2'
        # if 'EComp' in key:
            # heatmap3(val.df, key, ('randphase', 'twin', 'tts'), stat_list) # , ckchi2='ck' or 'chi2'
#%% Visualise normalised special cases:
    """ 
    For codes of results() before adding 'normalised' to path/to/fi
    """
    samplelist = ['xy_Caroline_LV_asym_competitive', 
                  'EComp_0.25_500_2.0,0.0_0.7,0.7_-0.4,-0.5,-0.5,-0.4_0.01_0.05',
                  'UComp_0.25_500_1.0,1.0_0.8,0.8_-0.4,-0.5,-0.9,-0.4_0.01_0.05']
    Res = {}
    for sample in samplelist:
        Res[sample] = results(sample)
        print(sample)
        
    for key, val in Res.items():
        val.into_df()
        
    stat_list = ('pearson', 'lsa', 'mutual_info', 'granger_y->x', 'granger_x->y','ccm_y->x', 'ccm_x->y')
    test_list = ('randphase', 'twin', 'tts')
    # heatmap(cap.df, 'cap', ['randphase', 'twin', 'tts'], cap.stat_list)
    # heatmap(uncap.df, 'uncap', ['randphase', 'twin', 'tts'], uncap.stat_list)
    for key, val in Res.items():
        # heatmap(val.fp_df, key, test_list, stat_list, falsepos = True)
        heatmap(val.df, key, ('randphase', 'twin', 'tts'), stat_list, 
                savehere='C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Extended/normalised',
                filextsn='svg')
#%%
    samplelist = ['xy_Caroline_LV_asym_competitive', 
                  'EComp_0.25_500_2.0,0.0_0.7,0.7_-0.4,-0.5,-0.5,-0.4_0.01_0.05',
                  'UComp_0.25_500_1.0,1.0_0.8,0.8_-0.4,-0.5,-0.9,-0.4_0.01_0.05']
    whichnorm = 'normzscore' # 'normminmax' 'normrank'
    Res = {}
    for sample in samplelist:
        Res[sample] = results(sample, excl=['falsepos'], incl=[whichnorm])
        print(sample)
        
    for key, val in Res.items():
        val.into_df()
        
    stat_list = ('pearson', 'lsa', 'mutual_info', 'granger_y->x', 'granger_x->y','ccm_y->x', 'ccm_x->y')
    test_list = ('randphase', 'twin', 'tts')
    # heatmap(cap.df, 'cap', ['randphase', 'twin', 'tts'], cap.stat_list)
    # heatmap(uncap.df, 'uncap', ['randphase', 'twin', 'tts'], uncap.stat_list)
    for key, val in Res.items():
        # heatmap(val.fp_df, key, test_list, stat_list, falsepos = True)
        heatmap(val.df, f'{key} {whichnorm}', ('randphase', 'twin', 'tts'), stat_list, 
                savehere=f'D:/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Extended/{whichnorm}',
                filextsn='svg')
