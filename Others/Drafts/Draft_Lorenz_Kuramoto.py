# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:31:08 2023

@author: hoang
"""
import os
os.getcwd()
# os.chdir('C:/Users/hoang/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/')
os.chdir('/home/h_k_linh/OneDrive/Desktop/UCL_MRes_Biosciences_2022/MyProject/Simulation_test/')
import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# def lorenz_ODEs(t, xyz, sigma = 10, rho = 28, beta = 8/3):
#     x, y, z = xyz
#     x_dot = sigma*(y - x) #-sigma, sigma, 0
#     y_dot = x*rho - x*z - y #rho, -1, 1
#     z_dot = x*y - beta*z # 1,0, -beta
#     return np.array([x_dot, y_dot, z_dot])
def lorenz_ODEs2(t, xyz, sigma = 10, rho = 28, beta = 8/3):
    A = np.array([[-sigma, sigma, 0], [rho, -1, -xyz[0]], [xyz[1], 0, -beta]])
    return np.dot(A, xyz)

#%% according to matplotlib
def generate_discrete_lorenz(N, delta_t = 0.05):
    xyzs = np.empty((N,3))
    # xyzs[0] = (0.0, 1.0, 1.05)
    xyzs[0] = np.random.random(size = 3)
    
    for i in range(xyzs.shape[0]-1):
        xyzs[i+1] = xyzs[i] + lorenz_ODEs2(0, xyzs[i])*delta_t
    
    return xyzs
        
def generate_continuous_lorenz(N, dt_s = 0.25, ODEs = lorenz_ODEs2):
    print('Generating Lorenz attractor')
    dt=0.05;
    sample_period = int(np.ceil(dt_s / dt));
    
    lag = int(150/dt);
    obs = sample_period * N;
    # epsilon = np.random.standard_normal(size = length + 1000)
    xyzs = np.empty((lag + obs + 1,3))
    # xyzs[0] = (0.0, 1.0, 1.05)
    xyzs[0] = np.random.random(size = 3)
    for t in range(lag + obs):
        soln = solve_ivp(ODEs, t_span= [0,dt], y0= xyzs[t])
        # print(soln.y[:,-1])
        xyzs[t+1] = soln.y[:,-1] # + eps;
        # print(s[t+1])
        if t % 100 == 0:
            print(t, "\t", xyzs[t+1])
        
    x = xyzs[lag::sample_period,];    
    return x[-N:,]
    # return xyzs
#%%
# interactive graphic
# https://stackoverflow.com/questions/49981313/rotate-interactively-a-3d-plot-in-python-matplotlib-jupyter-notebook

# xyzs_disc = generate_discrete_lorenz(10000)
# xyzs_cont = generate_continuous_lorenz(500)
# xyzs_disc = [generate_discrete_lorenz(10000) for _ in range(5)]
xyzs_cont = [generate_continuous_lorenz(500) for _ in range(5)]
xyzs_cont2 = [generate_continuous_lorenz(500, dt_s=0.15) for _ in range(5)]
# All 5 replicates give identical series
# for i in range (5):
#     xyzs_disc = generate_discrete_lorenz(10000)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(*xyzs_cont[0].T)
# ax.plot(*xyzs.T, lw=0.5)
# ax.scatter(*xyzs.T, color = "red")
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor for sigma = 10, rho = 28, beta = 2.667")
plt.show()

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(np.arange(len(xyzs_cont[0].T[1])), xyzs_cont[0].T[1])
ax.plot(np.arange(len(xyzs_cont[0].T[2])), xyzs_cont[0].T[2])
ax.plot(np.arange(len(xyzs_cont[0].T[2])), xyzs_cont[0].T[0])
# ax.plot(*xyzs.T, lw=0.5)
# ax.scatter(*xyzs.T, color = "red")
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
# ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor for sigma = 10, rho = 28, beta = 2.667")
plt.show()

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(np.arange(500), xyzs_cont[0].T[1][-500:])
ax.plot(np.arange(500), xyzs_cont[0].T[2][-500:])
# ax.plot(*xyzs.T, lw=0.5)
# ax.scatter(*xyzs.T, color = "red")
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
# ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor for sigma = 10, rho = 28, beta = 2.667")
plt.show()

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(np.arange(500), xyzs_cont2[0].T[1])
ax.plot(np.arange(500), xyzs_cont2[0].T[2])
# ax.plot(*xyzs.T, lw=0.5)
# ax.scatter(*xyzs.T, color = "red")
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
# ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor for sigma = 10, rho = 28, beta = 2.667")

plt.show()

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(np.arange(500), xyzs_disc[0].T[1][-500:])
ax.plot(np.arange(500), xyzs_disc[0].T[2][-500:])
# ax.plot(*xyzs.T, lw=0.5)
# ax.scatter(*xyzs.T, color = "red")
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
# ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor for sigma = 10, rho = 28, beta = 2.667")

plt.show()

#%% Kuramoto oscillator
def kuramoto_ODEs(t,theta, omega, K):
    print(omega)
    dtheta_1 = omega[0] + K*np.sin(theta[1]-theta[0])/2
    dtheta_2 = omega[1] + K*np.sin(theta[0]-theta[1])/2
    return np.array([dtheta_1, dtheta_2])

def generate_continuous_kuramoto(N, dt_s = 0.25, ODEs = kuramoto_ODEs):
    print('Generating Kuramoto oscilator')
    dt=0.01;
    sample_period = int(np.ceil(dt_s / dt));
    
    lag = int(150/dt);
    obs = sample_period * N;
    # epsilon = np.random.standard_normal(size = length + 1000)
    
    theta = np.empty((lag + obs + 1,2))
    theta[0] = np.random.random()*2*np.pi
    # xyzs[0] = (0.0, 1.0, 1.05)
    
    K=1
    
    # # frequency = np.random.randint(3,9,size = 2)
    # # omega = 2*np.pi*frequency
    # # # omega = np.random.normal(0.5,0.5, size = 2) # Wiki?
    # omega = np.random.randint(3,9,size = 2)
    omega= np.array([np.random.normal(3,K),np.random.normal(3+K+2,K)]) # reduce having chances of fixed points
    """
    Reduce the chance of having fixed points so that the fact that the two series are coupled is not obvious to the surrogate test ...
    """
    params = (omega, K)
    for t in range(lag + obs):
        soln = solve_ivp(ODEs, t_span= [0,dt], y0= theta[t], args=params)
        print(soln.y[:,-1])
        theta[t+1] = soln.y[:,-1] # + eps;
        # print(s[t+1])
        if t % 100 == 0:
            print(t, "\t", theta[t+1])
        
    x = theta[lag::sample_period,];    
    return x[-N:,], omega
    # return theta, omega

# theta_test, omega_test = generate_continuous_kuramoto(500)
theta_test_sampling, omega_test_sampling = generate_continuous_kuramoto(100, dt_s= 0.05)

x= np.sin(theta_test_sampling.T[0])
y= np.sin(theta_test_sampling.T[1])

fig, ax = plt.subplots(figsize = (12,3))
ax.plot(np.arange(len(x)), x, linewidth = .5)
ax.plot(np.arange(len(y)), y, linewidth = .5)

fig, ax = plt.subplots()
ax.plot(np.arange(100), x[-100:], linewidth = .5)
ax.plot(np.arange(100), y[-100:], linewidth = .5)