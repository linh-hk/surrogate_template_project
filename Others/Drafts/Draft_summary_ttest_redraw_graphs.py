# -*- coding: utf-8 -*-
"""
Created on Fri May  5 23:31:26 2023

@author: hoang
"""

import os
os.getcwd()
# os.chdir('C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/')
# os.chdir('/home/h_k_linh/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/')
os.getcwd()
import glob
import numpy as np
import pandas as pd 
import scipy as sp
from mergedeep import merge

import pickle

#%% Don't really have to read this part. 
"""
Load the results into objects and put them in dataframe format.
"""
class output:
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
            self.stats_list = ['lsa' if _ == 'lsa2' else 'ccm_y->x' if _ == 'ccm2_y->x' else 'ccm_x->y' if _ == 'ccm2_x->y' else _ for _ in test_results['stats_list']]
            # self.stats_list = test_results['stats_list']
            self.test_list = test_results['test_list']
            self.test_params = test_results['test_params']
            self.pvals = { xory : 
                          { stat: 
                           { (f"{cst}_{self.test_params['r_tts']}" if 'tts' in cst else cst): 
                            [_['pvals'][xory][stat][cst] for _ in test_results['pvals']] 
                            for cst in self.test_list} 
                               for stat in self.stats_list} 
                              for xory in ['surrY', 'surrX']} 
            self.runtime = [_['runtime'] for _ in test_results['pvals']]
            self.test_list = [(f"{cst}_{self.test_params['r_tts']}" if 'tts' in cst else cst) 
                              for cst in self.test_list]
            
class all_output:
    def add_test_results(self, filename, testname=''):
        # testname to rename test to make it short. 
        # instead of making it pearson_mutual_info_tts_tts_naive_59_59_4 can make it test 1, 2, 3, 4
        test = output(self.model_name, filename)
        if testname == '':
            if 'granger' in filename or 'ccm' in filename:
                testname = '_'.join(filename.split('.')[:-1])
            else:
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
            if os.path.isdir(f'{self.model_name}/{fi}'):
                continue
            else:
                self.add_test_results(fi)
        self.summarise_output()
        
    def summarise_output(self):
        self.summary_output = output(self.model_name)
        for keys, values in self.__dict__.items(): 
            if keys == 'summary_output':
                continue
            if isinstance(values, output): 
                # pvals = {f"maxlag_{values.test_params['maxlag']}" : values.pvals}
                merge(self.summary_output.pvals, values.pvals)
                self.summary_output.stats_list.extend([_ for _ in values.stats_list if _ not in self.summary_output.stats_list] )
                self.summary_output.test_list.extend([_ for _ in values.test_list if _ not in self.summary_output.test_list ] )
                merge(self.summary_output.test_params, values.test_params)
                # self.summary_output.test_params[keys] = values.test_params
                self.summary_output.runtime[keys] = values.runtime
                # self.summary_output.count_rejection_rate()
                # self.summary_output.count_sbs()
                
    def __init__(self, model_name):
        self.model_name = model_name
        with open(f'{model_name}/data.pkl', 'rb') as file:
            gData = pickle.load(file)
        self.series = gData['data']
        self.datagen_params = gData['datagen_params']
        self.load_all_results()
        
class all_model:            
    def add_model(self, model_name):
        setattr(self, model_name, all_output(model_name))
        
    def summarise_models(self):
        self.pvals = {}
        # self.stats_list = []
        # self.test_list = []
        self.test_params = {}
        self.runtime = {}
        for keys, vals in self.__dict__.items():
            if isinstance(vals, all_output):
                self.pvals[keys] = vals.summary_output.pvals
                # self.stats_list.extend([_ for _ in vals.summary_output.stats_list if _ not in self.stats_list] )
                # self.test_list.extend([_ for _ in vals.summary_output.test_list if _ not in self.test_list ] )
                self.test_params[keys] = vals.summary_output.test_params
                self.runtime[keys] = vals.summary_output.runtime
                
    def __init__(self, model_list):
        self.model_list = model_list
        for model in model_list:
            self.add_model(model)
        self.summary_models = self.summarise_models()
        
def into_df(all_modls, model_list = 'default'): # faster than into_df2 - runtime= 0.21683335304260254
    index = []
    surrY = []
    surrX = []
    if model_list == 'default':
        model_list = all_modls.model_list
    for models in model_list:
        for stat in all_modls.pvals[models]['surrY']: 
            for cst in all_modls.pvals[models]['surrY'][stat]:
                index.append(np.array([models, stat, cst]))
                surrY.append(sum( _ <= 0.05 for _ in all_modls.pvals[models]['surrY'][stat][cst]))
                surrX.append(sum( _ <= 0.05 for _ in all_modls.pvals[models]['surrX'][stat][cst]))
    indices = pd.MultiIndex.from_arrays(np.array(index).T, names = ("models", "stats", "surrogate_test"))
    df = pd.DataFrame({"SurrY": surrY, "SurrX": surrX}, index = indices, dtype="float")
    return df
    # all_models.pvals['xy_Caroline_LV_asym_competitive_2']['surrY']['lsa']['tts_119']
    
    # def into_df2(all_modls): # very slow - runtime = 8.5038480758667
    #     arr = np.array([[models, xory, stat, cst, pval] 
    #                        for models in all_modls.pvals
    #                        for xory in all_modls.pvals[models]
    #                        for stat in all_modls.pvals[models][xory]
    #                        for cst in all_modls.pvals[models][xory][stat]
    #                        for pval in all_modls.pvals[models][xory][stat][cst]])
    #     df = pd.DataFrame(arr, columns = ["models", "xory", "stats", "surrogate_test", "pvals"])
    #     tbl = df.pivot_table(values="pvals", index = ["models", "stats"], columns = ["surrogate_test", "xory"], aggfunc= lambda x: sum( float(_) <= 0.05 for _ in x))
    #     return tbl

#%% Chi-statistic calculation     
"""
Reading Caroline's reference and write chi-test function:
    https://www.bmj.com/about-bmj/resources-readers/publications/statistics-square-one/8-chi-squared-tests
    
                       surrY   surrX     Total
    less than 0.005      a1      b1      a1+b1
    more than 0.005      a2      b2      a2+b2
                       1000    1000      2000
    Compare distribution of a categorical variable in two different samples.
    Distribution of pval<=0.05 in two different samples (samples of using X and samples of using Y) 
        Null hypothesis: proportion of pval<=0.5 in surrY is equally in surrX.
        Expected a1 = expected b1: (a1+b1)*1000/2000 = (a1+b1)/2
        Expected a2 = expected b2: (a2+b2)*1000/2000 = (a2+b2)/2
        Observed - Expected: a1-(a1+b1)/2 and b1-(a1+b1)/2 <=> (a1-b1)/2 and (b1-a1)/2
                             a2-(a2+b2)/2 and b2-(a2+b2)/2 <=> (a2-b2)/2 and (b2-a2)/2
        x^2 = sum(((O-E)^2/E))
            = ((a1-b1)/2)^2/((a1+b1)/2) +
              ((b1-a1)/2)^2/((a1+b1)/2) +
              ((a2-b2)/2)^2/((a2+b2)/2) +
              ((b2-a2)/2)^2/((a2+b2)/2)
            = ((a1-b1)^2)/((a1+b1)) +
              ((a2-b2)^2)/(a2+b2)+
              
        In general needs a1+b1, a2+b2 = 2000-a1-b1,
                         a1-b1, a2-b2 = 1000-a1-1000+b1 = b1-a1
        Pandas DataFrame groupby ["models", "stats", "surrogate_test"] 

While reading other resources pick up the chi2_contingency function from scipy and write chi_test2.

Tested that chi_test and chi_test2 give same results.   
    In chi2_contingency function, Yates’ correction for continuity is set to True by default. It is used when testing for independence (in certain cases of contingency e.g. polling response vs nationality then we want to see whether response is linked to nationality) and when a cell or the total of all cells of the contingency table is very small (<20) (more details in Caroline's source)
                                                                                                                                                                                                                                                                                                                                                                    
    Here, we are using chi2 test as a test of homogeneity so we can turn correction off.            
"""
def chi_test(cstY_cstX): # written as derived above
    print("new")
    # YX = cstY_cstX[[0,1]].sort_index(ascending = False, level = ["xory"])
    cstY_cstX = cstY_cstX[[0,1]]
    lala = np.array([cstY_cstX,1000-cstY_cstX]) 
    print(lala)
    # add is [a1+b1, a2+b2]
    # if 1000 in lala:
    #     return 1000
    add = lala[:,0]+lala[:,1] # np.sum(lala, axis=0)
                # print(add)
    # subtract_squared is [a1-b1, a2-b2]
    subtract_squared = (lala[:,0]-lala[:,1])**2
                # print(sum(subtract_squared/add))
    return sp.stats.distributions.chi2.sf(sum(subtract_squared/add),1)
    # return YX, sum(subtract_squared/add), sp.stats.distributions.chi2.sf(sum(subtract_squared/add),1)
    
def chi_test2(cstY_cstX):
    print("new")
    # YX = cstY_cstX.sort_index(ascending = False, level = ["xory"])
                # print(YX.all(axis = 0))
    cstY_cstX = cstY_cstX[[0,1]]
    lala = np.array([cstY_cstX,1000-cstY_cstX]) 
    print(lala)
    # if 1000 in lala:
    #     return 1000
    try:
        return sp.stats.chi2_contingency(lala, correction = False).pvalue #
    except:
        return 1000
        # return np.array(cstY_cstX)
        
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
        draw = df.loc[row,col]/10
        chi2 = df.loc[row, [(_, "chi2_pvalues") for _ in cst_list]]
        
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

        plt.savefig(f"images_draft1000/{model}.svg")
        
# heatmap(df_all, model_list, stats_list, cst_list)
#%% Run codes
if __name__ == "__main__":
    data_list_all = sorted(glob.glob('xy_*')) # + ['ts_chaoticLV'] # + glob.glob('ts_*'))
    all_models = all_model(data_list_all)
    df_all = into_df(all_models)
    df_all['chi2_pvalues'] = df_all.apply(chi_test2, axis = 1)
    
    model_list = ['xy_Caroline_CH_competitive', 'xy_Caroline_CH_mutualistic', 
              'xy_Caroline_LV_asym_competitive', 'xy_Caroline_LV_asym_competitive_2',
              'xy_Caroline_LV_asym_competitive_2_rev', 'xy_Caroline_LV_asym_competitive_3',
              'xy_Caroline_LV_competitive', 'xy_Caroline_LV_mutualistic',
              'xy_Caroline_LV_predprey', 'xy_ar_u',
              'xy_uni_logistic', 'xy_FitzHugh_Nagumo_cont']
    stats_list = ['pearson', 'lsa', 'mutual_info', 'granger_y->x', 'granger_x->y', 'ccm_y->x', 'ccm_x->y']
    cst_list = ['randphase', 'twin', 'tts_119']
    heatmap(df_all, model_list, stats_list, cst_list)
    
    # all_models.add_model('xy_FitzHugh_Nagumo_cont')
    # all_models.add_model('xy_uni_logistic')
    # df_others = into_df(all_models, ['xy_FitzHugh_Nagumo_cont', 'xy_uni_logistic'])
    # df_others['chi2_pvalues'] = df_others.apply(chi_test2, axis = 1)
    # heatmap(df_others, ['xy_FitzHugh_Nagumo_cont', 'xy_uni_logistic'],
    #         stats_list, cst_list) 
    
    # another_model = all_model(['xy_Caroline_LV_asym_competitive_2'])
    # df_another = into_df(another_model)
    # df_another['chi2_pvalues'] = df_another.apply(chi_test2, axis = 1)
    # heatmap(df_another, ['xy_Caroline_LV_asym_competitive_3'], stats_list, cst_list)
#%% Drafts and old dataframe    
    # test = df_all.groupby(level=["models", "xory", "stats", "surrogate_test"]).apply(lambda x: sum([ _ <= 0.05 for _ in x['pvals']]))
    # test0 = test.groupby(level=["models", "stats", "surrogate_test"]).apply(chi_test) # return YX
    # test1 = test.groupby(level=["models", "stats", "surrogate_test"]).apply(chi_test) # return x^2 statistic
    # test2 = test.groupby(level=["models", "stats", "surrogate_test"]).apply(chi_test2) # chi2_contingency set correction = True by default - not transposed
    # test3 = test.groupby(level=["models", "stats", "surrogate_test"]).apply(chi_test2) # transpose lala
    # test4 = test.groupby(level=["models", "stats", "surrogate_test"]).apply(chi_test2) # chi2_contingency(correction = False) - not transposed
    ### Yates’ correction for continuity used when testing for independence (in certain cases of contingency e.g. polling response vs nationality then we want to see whether response is linked to nationality).
    ### Here, we are using chi2 test as a test of homogeneity so we can turn correction off.
    
    # lambda x: sum([ _ <= 0.05 for _ in x['pvals']])
    # type(x['pvals'][0])) # 
    # del test, test0, test1, test2, test3, test4, testn
