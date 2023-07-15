# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 15:11:43 2023

@author: hoang
"""

import pandas as pd
import pickle

with open('Simulated_data/xy_ar_u/ccms_grangers_tts_0_100_2.pkl','rb') as fi:
    cg_tts = pickle.load(fi)
nccmxy = [_[1]['runtime']['surrY']['tts']['ccm_x->y'] for _ in cg_tts['pvals']]
nccmyx = [_[1]['runtime']['surrY']['tts']['ccm_y->x'] for _ in cg_tts['pvals']]
ngxy = [_[1]['runtime']['surrY']['tts']['granger_x->y'] for _ in cg_tts['pvals']]
ngyx = [_[1]['runtime']['surrY']['tts']['granger_y->x'] for _ in cg_tts['pvals']]

with open('Simulated_data/xy_ar_u/old_results/ccm2_y.x_ccm2_x.y_tts_tts_naive_119_119_4.pkl','rb') as fi:
    ccm_otts = pickle.load(fi)
occmxy = [_['runtime']['surrY']['ccm_x->y']['tts'] for _ in ccm_otts['pvals']]
occmyx = [_['runtime']['surrY']['ccm_y->x']['tts'] for _ in ccm_otts['pvals']]    

with open('Simulated_data/xy_ar_u/old_results/granger_y.x_granger_x.y_tts_tts_naive_119_119_4.pkl','rb') as fi:
    granger_otts = pickle.load(fi)
ogxy = [_['runtime']['surrY']['granger_x->y']['tts'] for _ in granger_otts['pvals']]
ogyx = [_['runtime']['surrY']['granger_y->x']['tts'] for _ in granger_otts['pvals']]    

with open('Simulated_data/xy_ar_u/pearson_mi_lsa_randphase_twin_tts.pkl','rb') as fi:
    pmil_twin = pickle.load(fi)
np = pmil_twin['pvals']['runtime']['surrY']['tts']['pearson']
nmi = pmil_twin['pvals']['runtime']['surrY']['tts']['mutual_info']
nlsa = pmil_twin['pvals']['runtime']['surrY']['tts']['lsa']    

with open('Simulated_data/xy_ar_u/old_results/lsa_tts_tts_naive_119_119_4.pkl','rb') as fi:
    lsa_otts = pickle.load(fi)
olsa = [_['runtime']['surrY']['lsa']['tts'] for _ in lsa_otts['pvals']]

df = pd.DataFrame(data={'occmxy':occmxy[:100], 
                        'nccmxy': nccmxy[:100],
                        'occmyx': occmyx[:100], 
                        'nccmyx': nccmyx[:100],
                        'ogxy': ogxy[:100], 
                        'ngxy': ngxy[:100],
                        'ogyx': ogyx[:100], 
                        'ngyx': ngyx[:100],
                        'olsa': olsa[:100], 
                        'nlsa': nlsa[:100]})
df.mean()
df.sum()