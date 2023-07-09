# -*- coding: utf-8 -*-
"""
Time series handbook
https://phdinds-aim.github.io/time_series_handbook/00_Introduction/00_Introduction.html
"""

#%% Load libraries
import numpy as np
import numpy.random as rng
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as tg
# %matplotlib inline
# %matplotlib auto # or 
"""
You can switch the matplotlib's backend by %matplotlib <backend>. To switch back to your system's default backend use %matplotlib auto or just simply %matplotlib.
There are many backends available such as gtk, qt, notebook, etc. I personally highly recommend the notebook (a.k.a. nbagg) backend. It is similar to inline but interactive, allowing zooming/panning from inside Jupyter.
For more info try: ?%matplotlib inside an IPython/Jupyter or IPython's online documentation

https://stackoverflow.com/questions/30878666/matplotlib-python-inline-on-off
"""
"""
 Magic commands in IPython
 %lsmagic
 %matplotlib? # or
 %%html?
"""
plt.rcParams['figure.figsize'] = [15,2]

#%% Chapter 0
#%%% Temperature data 
data = pd.read_csv("C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/Simulation_test/jena_climate_2009_2016.csv")
data = data.iloc[:, 1:].astype(float).to_numpy()
temp = data[:, 1]
# plt.figure(figsize = (15,2))
plt.plot(range(len(temp)), temp)
plt.ylabel('Temperature\n (degree Celcius)')
plt.xlabel('Time (every 10 mins)')
plt.show()
#%%% Stochastic processes
#%%%% Example 1 Bernoulli trials
bern_outcomes = [-1.,1.]
size = 100
# Generate 100 Bernoulli trials
def generate_data(size):
    flips = rng.choice(bern_outcomes, size = size)
    series = pd.Series(flips)
    return series
series = generate_data(size)
fig, ax = plt.subplots(figsize = (15,2))
series.plot() # implicitly index 'them', let index stand for time
plt.show()
#%%%% Example 2 Gaussian  white noise 
size = 1000
wn = st.norm.rvs(size = size)
wn_series = pd.Series(wn)
# wn_series.plot()
fig, ax = plt.subplots(figsize = (15,2))
wn_series.plot(ax = ax)
plt.show()
#%%%% Example 3 Brownian motion in one dimension
size = 100000
def draw_generated(size = size):
    bm_ave, bm_msq = generate_data(size)
    bms_series = pd.Series(bm_ave)
    bms_msq_series = pd.Series(bm_msq)
    fig, ax = plt.subplots(figsize = (15,2))
    # ax.plot(bms_series, label = "Displacement")
    # ax.plot(bms_msq_series, label = "MSD")
    bms_series.plot(label = "Displacement", ax = ax)
    bms_msq_series.plot(label = "MSD", ax = ax)
    plt.legend()
    plt.show()
    
def generate_data(size):
    bms = []
    bms_msq = []
    for i in range(100):
        wn = st.norm.rvs(size = size)
        bm = np.cumsum(wn)
        bms.append(bm)
        bms_msq.append([x**2 for x in bm])
    bm_ave = np.mean(bms, axis = 1)
    bm_msq = np.mean(bms_msq, axis = 1)
    return bm_ave, bm_msq
    
if __name__ == "__main__":
    draw_generated()
#%%%% Example 4 Moving average
Size = 1000
def draw_generated(size):
    wn_series, ma2_series = generate_ma2(size)
    fig, ax = plt.subplots()    
    ax.plot(wn_series, label = "WN(0,1))")
    ax.plot(ma2_series, label = r"MA(2)")
    plt.legend(loc = "best")
    plt.show()
def generate_ma2(size):
    ma2 = []
    whitenoise = generate_whitenoise(size)
    ma2.append(whitenoise[0]) # first element is just epsilon[0]
    ma2.append(whitenoise[1] + 0.5*whitenoise[0]) # second element only take 1st and 2nd
    for t in range(2, len(whitenoise)):
        ma2.append(whitenoise[t] + 0.5*whitenoise[t-1] + 0.25*whitenoise [t-2])
    return whitenoise, pd.Series(ma2)
def generate_whitenoise(x): # name it as x to see the important of size =x in the st.norm.rvs, can replace "x" by "size"
    wn = st.norm.rvs(size = x)
    return pd.Series(wn)
  
if __name__ == "__main__":
    draw_generated(Size)
# del wn, ma2, Size
    
#%%%% Autoregressive process
size = 1000
def draw_generated(wn, ar2):
    fig, ax = plt.subplots()
    ax.plot(wn , label = "WN(0,1)")
    ax.plot(ar2, label = "AR(2)")
    plt.legend(loc = "best")
    plt.show()
def generate_ar2(size):
    wn = generate_whitenoise(size)
    x = []
    x.append(wn[0])
    x.append(0.6*x[0] + wn[1])
    for t in range(2,len(wn)):
        x.append(0.6*x[t-1] + -0.2*x[t-2] + wn[t])
    return wn, pd.Series(x)
def generate_whitenoise(x):
    wn = st.norm.rvs(size = x)
    return pd.Series(wn)

if __name__ == "__main__":
    wn_series, ar2_series = generate_ar2(size)
    draw_generated(wn_series, ar2_series)

#%%%% Autocorrelation function 


