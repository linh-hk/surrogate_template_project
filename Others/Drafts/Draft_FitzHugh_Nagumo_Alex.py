# -*- coding: utf-8 -*-
"""FHN_granger.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xFI5vrJU057Ch2VYSom1lx4Ahu1FnUA9

Alex script for discrete FitzHugh Nagumo

"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests

def noisy_fitzhugh(n=10400, speed=0.7):
    t = np.arange(1, n+1)
    v = np.zeros(n)
    w = np.zeros(n)
    for i in range(n-1):
        delta_v = speed * (v[i] - ((v[i] ** 3)/3) - w[i] + np.random.normal(1, 0.2))
        delta_w = speed * 0.08 * (v[i] + 0.7 - 0.8*w[i]) + np.random.normal(0, 0.05) # adding a little bit of noise here
        v[i+1] = v[i] + delta_v
        w[i+1] = w[i] + delta_w
    return v,w

x, y = noisy_fitzhugh()
x = x[-400:]
y = y[-400:]

plt.plot(x)
plt.plot(y)
plt.show()

xy = np.zeros([x.size, 2])
xy[:,0] = x
xy[:,1] = y

result = grangercausalitytests(xy, maxlag=4)

result = grangercausalitytests(xy[:,(1,0)], maxlag=4)

