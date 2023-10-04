# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 19:07:38 2023

@author: hoang
"""

import numpy as np

def x(beta, g_yx, g_xy):
    return (g_yx-beta)/(beta**2-g_xy*g_yx)

def y(beta, g_yx, g_xy):
    return (g_xy-beta)/(beta**2-g_xy*g_yx)

def sign_lambd(beta, g_yx, g_xy):
    s = x(beta, g_yx, g_xy) + y(beta, g_yx, g_xy)
    p = x(beta, g_yx, g_xy) * y(beta, g_yx, g_xy)
    lbd_neg = np.sign(np.real(beta*s/2 - np.emath.sqrt(beta**2*s**2 - 4*(beta**2 - g_xy*g_yx)*p)/2))
    lbd_pos = np.sign(np.real(beta*s/2 + np.emath.sqrt(beta**2*s**2 - 4*(beta**2 - g_xy*g_yx)*p)/2))
    
    return lbd_neg, lbd_pos, -1/beta , (g_yx-beta)/(beta**2-g_yx*g_xy), (g_xy-beta)/(beta**2-g_yx*g_xy)

def sign(a, b, c_yx, c_xy):
    beta = b/a
    g_yx = c_yx/a
    g_xy = c_xy/a
    return sign_lambd(beta, g_yx, g_xy), beta, g_yx, g_xy#, x(beta,g_yx, g_xy), y(beta, g_yx,g_xy),1-g_yx/beta

print('EMut',sign(0.7, -0.4, 0.3, 0.3))
print('EComp',sign(0.7, -0.4, -0.5, -0.5))
print('UComp1',sign(0.8, -0.4, -0.5, -0.9))
print('UComp2',sign(0.8, -1.4, -0.5, -0.9))
print('UComp3',sign(50, -100, -95, -99))
