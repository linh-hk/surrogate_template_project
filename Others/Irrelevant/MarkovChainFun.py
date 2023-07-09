# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 22:42:33 2022

@author: hoang
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

mat = np.array([[0, 0.5, 0.5, 0],
                [0.5, 0, 0, 0.5],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])
print(mat)

###Failed
# scipy.linalg.eig(mat, right= False, left= True)
# print("left eigen vectors = \n", left, "\n")
# print("eigen values = \n", values)
# pi = left[:,0]
# pi_normalized = [(x*np.sum(pi)).real for x in pi]
# print(pi_normalized)
###Failed

steps = 10**6
mat_n = mat 
i=0
while i<steps:
    mat_n = np.matmul(mat_n, mat)
    i+=1
    
print("mat_n = \n", mat_n, "\n")
print("pi = ", mat_n[0])

#%% Con lac don giao dong dieu hoa
def genx(A,f,t):
    return A*np.cos(2*np.pi*f*t + np.pi)

time = np.arange(1, 5, 0.0005)
x_t = np.zeros(len(time))
for t in range(len(time)):
    x_t[t] = genx(2.5,5,time[t])
    
plt.plot(time, x_t)

freq = np.arange(1, 5, 0.0005)
x_t = np.zeros(len(freq))
for f in range(len(freq)):
    x_t[f] = genx(2.5,f,10)
plt.plot(freq, x_t)

def graph_exp(t):
    return np.exp(t)

t = np.arange(-5,5,0.0005)
x_t = np.exp(t)
plt.plot(t,x_t)

#%% Visualise Fourier Transform
# http://paulbourke.net/miscellaneous/dft/#:~:text=The%20highest%20positive%20(or%20negative,to%20yield%20%22uncorrupted%22%20results.
import cmath # complex math, methematics for complex numbers
import numpy as np
import matplotlib.pyplot as plt
def graph_part_fourier(f,t): 
    """
        exp(ix)=cosx+isinx
    Substitute x = wt with w is angular frequency, then
        exp(iwt) = cos(wt) + isin(wt)
    Substitute w = 2*pi*f with f is ordinary frequency, then
        exp(i2*pi*f*t) = cos(2*pi*f*t) + isin(2*pi*f*t)
    IDK why I want negative sign but I guess I just used it from some source...
    """
    # return cmath.exp(complex(0,-2*np.pi*f*t))
    return cmath.exp(-2*np.pi*f*t*1j)

t = np.arange(-5,5,0.0005)
x_t = np.ndarray(len(t), dtype=np.complex128)
freq = [1, 2.5, 5]
# The way this is drawn is wrong because it's taking the real part of the equation 
# which is the cos part but it's fine... I guess ...
fig, ax = plt.subplots(3,figsize = (10,2.5))
for f in range(3):
    x_t = np.ndarray(len(t), dtype=np.complex128)
    for i in range(len(t)): 
        x_t[i] = graph_part_fourier(freq[f], t[i])
    ax[f].plot(t, x_t)
plt.show()

fig, ax = plt.subplots(3,figsize = (10,2.5))
for f in range(3):
    ax[f].plot(x_t)
#%%
# generate x series from another scripts
x = data.generate_ar1(400)
k = 1
for k in range(x):
    X_k = x[k]*cmath.exp(-1j*k*2*np.pi*n/len(x))

for i in 