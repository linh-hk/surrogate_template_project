# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 16:19:27 2022

@author: hoang
"""
#%% Import libraries
# import numpy as np
# import pandas as pd
import scipy as sp
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import random

#%% Generate data, two dependent series
def generate_AR1_uni_tau1(size):
    print('Generating AR1_uni_tau1')
    x, y = sp.stats.norm.rvs(size = size + 1), sp.stats.norm.rvs(size = size + 1)
    # x_series, y_series = [] , []
    # x_series.append(wn_x[0])
    # y_series.append(wn_y[0])
    for t in range(size):
        x[t+1] = 0.55*x[t] + -0.25*y[t] + x[t+1]
        y[t+1] = 0.85*y[t]              + y[t+1]
    return [x[:-1], y[:-1]]

# Unidirectional AR(1)
def generate_AR1_uni_tau2(size):
    print('Generating AR1_uni_tau2')
    x,y = sp.stats.norm.rvs(size = size + 2), sp.stats.norm.rvs(size = size + 2)
    for t in range(1, size+1):
        x[t+1] = 0.4 * x[t] + 0.3*y[t-1] + x[t+1]
        y[t+1] = 0.4 * y[t]              + y[t+1] 
    return [ x[1:-1], y[1:-1] ]

# Unidirectional logistic map
def generate_uni_logistic_map(size):
    print('Generating uni_logistic_map')
    x,y = np.zeros(size + 500), np.zeros(size + 500)
    x[0] , y[0] = np.random.uniform(0.2, 0.8) , np.random.uniform(0.2, 0.8)
    for t in range(1, size+500-1):
        x[t] = x[t-1] * (3.5 - 3.5*x[t-1] - 0.1*y[t-1])
        y[t] = y[t-1] * (3.8 - 3.8*y[t-1] )
    return [ x[500:], y[500:] ]

# Nonlinearly coupled autoregressive process
# Bidirectionally coupled processes with random coupling delays
# Autoregressive process
# Logistic map
#%% Draw
def draw_x_y(X, Y):
    fig, ax = plt.subplots(figsize = (15,2))
    ax.plot(X, label = "x_series")
    ax.plot(Y, label = "y_series")
    plt.legend(loc = "best")
    plt.show()
# draw_x_y(x_series, y_series)
#%% Generate indepdendent data
#%%% Stationary
def generate_ar1(length):
    print('Generating normal AR1')
    x = np.zeros(length+1);
    x[0]= np.random.normal(loc =0, scale=(1-0.7**2)**(-1/2));
    for i in range(length):
        x[i+1] = 0.7*x[i] + np.random.standard_normal();
    return x[1:];
        
def generate_logistic_map(length):
    print('Generating normal logistic map')
    x = np.zeros(length+1);
    x[0] = np.random.beta(0.5, 0.5);
    for i in range(length):
        x[i+1] = 4*x[i]*(1-x[i]);
    return x;

def generate_sine_w_noise(length):
    print('Generating sine wave with noise')
    t = range(1,length + 1)
    phi_a = np.random.randint(0,2799, size = length)
    # or ? phi_a = np.random.uniform(0,2799, size = length)
    phi_x = np.random.randint(0,21, size = length)
    # or ? phi_x = np.random.uniform(0,21, size = length)
    phi_y = np.random.randint(0,21, size= length)
    # or ? phi_y = np.random.uniform(0,21, size= length)
    epsilon_t = np.random.beta(1/3, 1/3, size= length)
    a_t = np.mod(t + phi_a, 2800) / 7000
    x_t = np.sin(2*np.pi*(t + phi_x) / 22)
    y_t = np.sin(2*np.pi*(t + phi_y) / 22) + a_t * epsilon_t - 0.5
    # test = np.prod([a_t,epsilon_t], axis =0) = a_t * epsilon_t
    return [x_t, y_t]
    
def generate_sinew_intmt_corptxn(length):
    print('Generating sine wave with intermitment coruption')
    t= np.arange(1,length + 1)
    phi = np.random.uniform(0, 2*np.pi, size = length)
    epsilon_t = np.random.uniform(0,1, size = length)
    lamda_t = np.array(random.choices([0.15, 0.95], [2/3,1/3], k = length)) # [1/2,1/2]
    a_t = np.sin(np.pi * t *1/4 +phi) /2 + 1/2
    x_t = (1-lamda_t) * a_t + lamda_t * epsilon_t
    return x_t

def generate_coinflips_w_changeHeadprob_noise(length):
    print('Generating coinflips with changing of Head probability with noise')
    t = np.arange(1, length + 1)
    phi_1, phi_2 = np.random.uniform(0, 2*np.pi,size = length), np.random.uniform(0, 2*np.pi,size = length)
    a_t_1 = ((1/2)**7) * ((np.sin(phi_1 + 2*np.pi*t /75) + 1)**6)
    a_t_2 = (1/24) * (np.sin(phi_2 + 2*np.pi*t / (31*np.sqrt(2))) + 1)
    a_t = a_t_1 + a_t_2
    
    x = np.zeros(length)
    for t in range(0,len(x)):
        x[t] = random.choices([1,0], [ a_t[t] , 1 - a_t[t] ])[0] + np.random.normal(0,0.15)
    
    return x

def generate_noise_w_periodically_varying_kurtosis(length):
    print('Generating noise with periodically varying kurtosis')
    epsilon = np.random.uniform(-np.sqrt(3.03),np.sqrt(3.03), size = length)
    v = np.random.normal(0, 0.1, size = length)
    beta = random.choices([-1,1],[0.5,0.5], k = length)
    
    a = np.mod(range(length+1),2)
    a_rand = random.choice([0,1])
    a = a[a_rand : length + a_rand]
    
    x = a*epsilon + (1-a)*(v+beta)
    return x

#%%% FitzHugh Nagumo
def d_FitzHugh_Nagumo(t, x_w):
    x = x_w[0]
    w = x_w[1]
    epsilon_t = np.random.standard_normal()
    dx_t =      0.7*(x - x**3 /3 - w + epsilon_t)
    dw_t = 0.08*0.7*(-0.8*w      + x + 0.7 )
    return np.array([dx_t, dw_t])

# Turns out that 0.7 is Alex's delta t so it seems that we dont need 0.7 in the equation.
# Maybe we'll generate a time series that does not have 0.7 in the differential equation.
# Let that equation be xy_FitzHugh_Nagumo_0.5

def d_FitzHugh_Nagumo_(t, x_w):
    x = x_w[0]
    w = x_w[1]
    epsilon_t = np.random.standard_normal()
    dx_t =          x - x**3 /3 - w + epsilon_t
    dw_t = 0.08* ( -0.8 * w     + x + 0.7)
    return np.array([dx_t, dw_t])

def generate_FitzHugh_Nagumo(N, dxy_dt = d_FitzHugh_Nagumo, dt_s = 0.25):
    print('Generating FitzHugh-Nagumo')
    dt=0.05;
    sample_period = int(np.ceil(dt_s / dt));
    
    lag = int(50/dt);
    obs = sample_period * N;
    # epsilon = np.random.standard_normal(size = length + 1000)
    s = np.zeros((lag + obs + 1,2))
    for t in range(lag + obs):
        soln = sp.integrate.solve_ivp(dxy_dt, t_span= [0,dt], y0= s[t])
        # print(soln.y[:,-1])
        s[t+1] = soln.y[:,-1] # + eps;
        # print(s[t+1])
        
    x = s[lag::sample_period,];    
    return x[-N:,]

# discrete time version of FitzHugh Nagumo
# Let this equation be xy_FitzHugh_Nagumo_1
def generate_discrete_FitzHugh_Nagumo(size, delta_t = 0.05):
    x = np.zeros(size + 1000)
    w = np.zeros(size + 1000)
    epsilon_t_x = np.random.normal(loc = 0, scale = 1, size = size + 1000)
    # epsilon_t_w = np.random.normal(size = size + 1000)
    for i in range(1,size + 1000):
        x[i] = x[i-1] + delta_t*(x[i-1] - x[i-1]**3 /3 - w[i-1] + epsilon_t_x[i-1])
        w[i] = w[i-1] + delta_t*0.08*(x[i-1] + 0.7 -0.8 * w[i-1])
        # print(f'xt = {x[i-1]}\txt+1 = {x[i]}\twt = {w[i-1]}\twt+1 = {w[i]}\tepsilon = {epsilon_t_x[i-1]}')
    # return x[-N:,], w[-N:,]
    return x, w
    
#%%% Chaotic Lotka Volterra
def LV_function(t,y,r,a):
    """
    length is the number of observations/ length of series
    r is an array of i length = innate 'growth' rate of i species
    a is a matrix of ixi dimension = interaction between i species
    """
    lamda = np.subtract(1 , a@y ) # 4x4 * 4x1 = 4x1
    np.array(lamda)
    s_prime = 1.5*r*y*lamda
    # print(f't = {t} \n y = {y.shape} \n r = {r} \n a = {a} \n lamda = {lamda} \n s_prime = {s_prime}')
    return s_prime

def generate_chaotic_lv(N, r, a, dt_s = 0.25): #, noise,noise_T):
    # dt_s is time interval between observations
    print("Generating Alex's chaotic Lotka-Volterra")
    dt=0.05; # dt for formulas
    sample_period = int(np.ceil(dt_s / dt)); # period in terms of number of observations
    lag = int(150/dt); # observations in lag phase which will be remove
    obs = sample_period * N ; # number of observations between periods * number of periods
    s = np.zeros((lag + obs + 1, r.size)) # create 2 (r.size) series s of ... observations
    s[0] = np.random.uniform(0.1, 0.5, size = r.size)
    for t in range(lag + obs):
        # t = 0
        soln = sp.integrate.solve_ivp(LV_function,t_span= [0,dt],y0= s[t],args= (r, a))
        # eps = noise*np.random.randn(2)*np.random.binomial(1,dt/noise_T,size=2);
        # print(f'soln = {soln}')
        s[t+1] = soln.y[:,-1] # + eps;
        s[t+1][np.where(s[t+1] < 0)] = 0;

    x = s[lag::sample_period,];
    for i in range(x.ndim):
        x[:,i] += 0.001*np.random.randn(x[:,i].size);
    # plt.plot(x)
    # x and y are independent realisation of s4 only so ...
    return x[:,-1]
ARGs = {'N': 100, 'r' : np.array([1, 0.72, 1.53, 1.27]),
        'a' : np.array([[1, 1.09, 1.52, 0], 
                        [0, 1, 0.44, 1.36], 
                        [2.33, 0, 1, 0.47], 
                        [1.21, 0.51, 0.35, 1]]),
        'dt_s': 0.25}

# LVuncut, LVcut = generate_lv(r,a,0.25,100)
# test = generate_chaotic_lv(400, r, a, 0.25)

#%%% Non stationary
def generate_random_walk(length):
    print('Generating random walk')
    epsilon = np.random.standard_normal(length)
    x = np.zeros(length)
    for t in range(length):
        x[t+1] = x[t] + epsilon
    return x

def generate_ar1_w_trend(length):
    print('Generating AR1 with trend')
    a = generate_ar1(length = length)
    return a + np.divide(range(1,length),60)

#%%% Caroline chemical mediated models
def dSCdt(SC, num_spec, r0, K, alpha, beta, rho_plus, rho_minus):
    """
    Parameters:

    SC (array): an array of species and chemical abundances in which species
        are listed before chemicals
    num_spec (int): number of species
    r0 (2d numpy.array): num_spec x 1 array of intrinsic growth rates
    K (2d numpy.array): num_spec x num_chem array of K values
    alpha (2d numpy.array): num_chem x num_spec array of consumption constants
    beta (2d numpy.array): num_chem x num_spec array of production constants
    rho_plus (2d numpy.array): num_spec x num_chem array of positive influences
    rho_minus (2d numpy.array): num_spec x num_chem array of negative influences
    """

    S = np.reshape(SC[:num_spec], [num_spec,1])
    C = np.reshape(SC[num_spec:], [len(SC) - num_spec, 1])
    # compute K_star
    K_star = K + C.T
    # compute K_dd
    K_dd = rho_plus * np.reciprocal(K_star)
    # compute lambda
    Lambda = np.matmul(K_dd - rho_minus, C) #According to the manuscript we would have (K_dd - rho_minus/K)*C
    # compute dS/dt
    S_prime = (r0 + Lambda) * S
    # compute K_dag
    C_broadcasted = np.zeros_like(K.T) + C
    K_dag = np.reciprocal(C_broadcasted + K.T) * C_broadcasted # Why calculate C_broadcasted instead of using C
    # compute dC/dt
    C_prime = np.matmul(beta - (alpha * K_dag), S)
    SC_prime = np.vstack((S_prime, C_prime))
    return SC_prime

#for chemically mediated interactions (add constant rsrc flux)
def sc_prime_rsrc(t, y, num_spec, r0, K, alpha, beta, rho_plus, rho_minus,r_flux):
    dy = dSCdt(y, num_spec, r0, K, alpha, beta, rho_plus, rho_minus);
    dy[-1] += r_flux;
    return np.reshape(dy, dy.size).tolist()

# Caroline sent: run nonlinearity test, Linh modified to only generate ts
def generate_niehaus(run_id, dt_s, N, noise, noise_T, intx="competitive"):
    print('Generating Caroline bacterial chemically regulated model')
    dt=0.05;
    sample_period = int(np.ceil(dt_s / dt));
    
    lag = int(150/dt);
    obs = sample_period * N;
    # choose interaction modes
    if (intx=="competitive"):
        num_spec = 2
        num_chem = 1
        r0 = np.array([[-1.6],
                       [-1.6]])
        K = np.array([[5.0],
                      [5.0]])
        alpha = np.array([[4.0,4.0]])
        beta = np.array([[0.0,0.0]])
        rho_plus = np.array([[4.8],
                      [4.8]])
        rho_minus = np.zeros((num_spec,num_chem))
        r_flux = 4;
    if (intx=="mutualistic"):
        num_spec = 2
        num_chem = 2
        r0 = np.array([[-2.],[-2.]])
        K = np.array([[5.0,1.0],
                      [1.0,5.0]])
        alpha = np.array([[6.0,0.8],
                          [0.8,6.0]])
        beta = np.array([[0.0,2.0],
                         [2.0,0.0]])
        rho_plus = np.array([[10.0,0.0],
                      [0.0,10.0]])
        rho_minus = np.array([[0.0,0.4],
                              [0.4,0.0]])
        r_flux = 0;
    
    params = (num_spec, r0, K, alpha, beta, rho_plus, rho_minus, r_flux);
    
    s = np.zeros((lag + obs + 1, num_spec + num_chem))
    s[0] = np.zeros((num_spec + num_chem))
    s[0,:2] = 1;
    
    for i in range(lag + obs):
        soln = solve_ivp(sc_prime_rsrc,[0,dt],s[i],args=params);
        eps = noise*np.random.randn(num_spec + num_chem) * \
            np.random.binomial(1,dt/noise_T,size=num_spec + num_chem);
        s[i+1] = soln.y[:,-1] + eps;
        s[i+1][np.where(s[i+1]<0)] = 0;

    x = s[lag:lag+obs:sample_period,].copy()
    
    if (np.any(x < 2.)):
        print("Crossover?")
    
    # return test_with_surr(x.reshape(-1,1));
    return x
#%%% Caroline AR
# AR model for control
def ar_control(noise=1.0):
    print('Generating Caroline AR model for control')
    x = np.zeros(601);
    for i in range(600):
        x[i+1] = 0.8*x[i] + noise*np.random.randn();
      
    x = x[100:];
    return x
#%%% Caroline LV
# Lotka Volterra
def lotkaVolterra(t,y,mu,M):
    return y * (mu + M @ y);  

def lotkaVolterraSat(t,y,mu,M,K):    
    intx_mat = M/(K + y);
    lv = y * (mu + intx_mat @ y);
    return lv;

def generate_lv(dt_s, N, s0, mu, M, noise, noise_T, fn = lotkaVolterra, measurement_noise = 0): # ,intx="competitive"
    print('Generating Caroline Lotka-Volterra model')
    dt=0.05; # integration step
    lag = int(150/dt);
    sample_period = int(np.ceil(dt_s / dt)); 
    obs = sample_period * N;
    n = len(mu)
    s = np.zeros((lag + obs + 1, n))
    
    args = (mu,M);
    s[0] = s0
    
    for i in range(lag + obs):
        soln = solve_ivp(fn,[0,dt],s[i],args=args)
        eps = noise*np.random.randn(n)*np.random.binomial(1,dt/noise_T,size=n); # process noise/external perturbation = allow migration over time.
        # s[i+1] = soln.y[:,-1] + eps; # print(s[i+1])
        # s[i+1][np.where(s[i+1] < 0)] = 0;
        nxt = soln.y[:, -1] + eps
        nxt[nxt<0] = 0 
        s[i+1] = nxt

    x = s[lag:lag+obs:sample_period,]; 
    for i in range(x.shape[1]):
        x[:,i] += measurement_noise*np.random.randn(x[:,i].size) # measurement noise = 0.001 for my thesis

    # return [x[:,_] for _ in [0,1]] # for 2 species
    return x # for 4 species

#%% Multiple species - hihger-order system
# def interaction_matrix_M(n_species, mu=30.0, sigma=4.0, gamma=-0.5, rng=None):
#     """
#     Construct interaction matrix M.
#     May's ecological stability theory. For a GLV or random large ecological network, if interaction coefficients scale like 1/sqrt(S), the system has well-defined stability threshold

#     Parameters
#     ----------
#     n_species : int
#         Number of species (S).
#     mu : float
#         Mean interaction parameter (unnormalised).
#     sigma : float
#         Std dev of interaction parameter (unnormalised).
#     gamma : float
#         Correlation between a_ij and a_ji.
#     rng : np.random.Generator or None
#         Random generator for reproducibility.

#     Returns
#     -------
#     AA : (S, S) numpy array
#         Interaction matrix.
#     """
#     if rng is None:
#         rng = default_rng()

#     A_mean = mu / n_species
#     A_std  = sigma / np.sqrt(n_species)

#     Mean  = np.array([A_mean, A_mean])
#     Sigma = np.array([[A_std**2, gamma * A_std**2],
#                       [gamma * A_std**2, A_std**2]])

#     # sample S*S correlated pairs (a_ij, a_ji)
#     R = rng.multivariate_normal(Mean, Sigma, size=(n_species * n_species))
#     R1 = R[:, 0].reshape(n_species, n_species)
#     R2 = R[:, 1].reshape(n_species, n_species)

#     A1u = np.triu(R1, 1)        # upper triangle (excluding diag)
#     A2u = np.triu(R2, 1)        # another upper triangle → used for lower

#     AA  = A1u + A2u.T + np.eye(n_species)  # symmetric-ish + self = 1
#     return AA

# def initial_conditions_s0():
#     return

# def intrinsic_growth_vector_mu():
#     return

#%%%Drafts
def vis_data(ts, titl = ""):
    fig, ax = plt.subplots(figsize = (10, 2.7))
    ax.set_title(titl)
    for i in range(len(ts)):
        ax.plot(ts[i])

#%% Run
if __name__=="__main__":
    
    N=500;
    noise=0.05*0.5;
    noise_T=0.05;
    dt_s=0.25;
    intx="competitive";
    reps = 1;
    ts_ar = ar_control()
    ts_lv = generate_lv(run_id = 1, dt_s = dt_s, N=N, noise=noise,noise_T=noise_T,intx="competitive")
    ts_ch = generate_niehaus(run_id=1, dt_s=dt_s, N=N, noise=noise, noise_T=noise_T, intx="competitive")
    # func=generate_niehaus
    # Draw separately
    for x in ['ts_lv', 'ts_ch']:
        ts = globals()[x];
        for i in range(ts.shape[1]):
            plt.plot(ts[:,i])
            plt.title(x)
    # plt.show()
        plt.show()
    plt.plot(ts_ar)
    plt.title("ts_ar")
    
    # Draw in one figure
    draw_these = ['LVuncut','LVcut','ts_lv', 'ts_ch']
    fig, axes = plt.subplots(len(draw_these),1, layout = "constrained", figsize = (10,10))
    for i in range(len(draw_these)) :
        titl = draw_these[i]
        draw_this = globals()[titl]
        axes[i].set_title(titl)
        for series in range(draw_this.shape[1]):
            axes[i].plot(draw_this[:,series])
    for ax in axes.flat:
        ax.set(xlabel='time(t)', ylabel='Values')
    