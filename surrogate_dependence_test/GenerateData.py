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
#%%% Caroline gLV
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
    lag = int(250/dt) # int(150/dt) # 30000
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
        nxt[nxt<10e-8] = 0
        # nxt[nxt<0] = 0 
        s[i+1] = nxt

    x = s[lag:lag+obs:sample_period,]; 
    x = x + measurement_noise * np.random.randn(*x.shape) # measurement noise = 0.001 for my thesis

    # return [x[:,_] for _ in [0,1]] # for 2 species
    return x # for multispecies
    # return s # checking the dynamic

#%% Vano 4-species strange attractor
mu_vano = np.array([1, 0.72, 1.53, 1.27])
M_vano = np.array([[1,    1.09, 1.52, 0   ],
                   [0,    1,    0.44, 1.36], 
                   [2.33, 0,    1,    0.47], 
                   [1.21, 0.51, 0.35, 1   ]])
'''
    gLV form:
        dy/dt = y ⊙ (mu + M y)
    ----------
    Vano form:
        dy/dt = r ⊙ y ⊙ (1 - A y)
        - mu = r
        - M = −diag(r) · A
    ----------
    | Code              | What it does        | Mathematical meaning        |
    |-------------------|---------------------|-----------------------------|
    | r[:, None] * A    | Scales rows of A    | (r_i · A_ij)                |
    | A * r             | Scales columns of A | (A_ij · r_j)                |
    | np.diag(r) @ A    | Scales rows of A    | (r_i · A_ij)                |
    | A @ np.diag(r)    | Scales columns of A | (A_ij · r_j)                |
'''
def scale_offdiag(A, s):
    '''
    A : is the interaction matrix M but in Vano annotation...
    s : coupling-scaling bifurcation parameter
    ----------
    Ascaled : A_{ij} -> s*A_{ij} with i!=j

    '''
    Ascaled = s * A.copy()
    np.fill_diagonal(Ascaled, np.diag(A))
    # D = np.diag(np.diag(A))
    # return s * A + (1 - s) * D
    return Ascaled

def convertAvano2M(A, mu):
    '''
    M : interaction matrix
    mu : base growth rate
    '''
    return -A * np.expand_dims(mu, 1) # M = -(r[:, None] * A)
#%% Multiple species
from numpy.random import default_rng

def initial_conditions_s0(S):
    # S : number of species in community. Returns random s0 of size S
    # in Polo codes, the makepool and makeplate contribute to this function
    # in this study, I don't propagate the culture so I'm randomly generating initial condition
    return np.random.uniform(0.0, 1.0, size=S)

def intrinsic_growth_vector_mu(S):
    # S : number of species in community. Returns vector of 1s with S entries
    return np.ones(S)

##### Polo Matlab extracted ver - May's work
'''
    Construct interaction matrix M.
    May's ecological stability theory. For a GLV or random large ecological network, if interaction coefficients scale like 1/sqrt(S), the system has well-defined stability threshold
    ----------
    Idea of May:
        - Simple question: what happens to stability when a system gets large and complex?
        - He considered: S species, random interactions, no structure, asked when equilibria are typically stable
    ----------
    Why scale by squrt(S)?
        - If naively draw A_{ij} ~ Normal(mu, sigma^2)
        Sum(A_{ij}Nj) would grow as S increases, the dynamics blows up
        - So May introduced scaling purely for mathematical reasons
            sigma_eff = sigma/squrt(S)
        - These parameters existed because May was studying random matrix spectra, not ecological realism
    ----------
        dy/dt = y ⊙ (K - y - Ay) = y ⊙ (K-(I+A)y)
    Normalised logistic form of gLV
        dy/dt = y ⊙ (1 - (I+A) @ y)
        - A_{ij} with i!=j, the diagonal has been taken out as -y = -Iy
        - mu = K = 1 (intrinsic_growth_vector_mu())
'''
def im_may_M(S, meanmu=30.0, sigma=4.0, gamma=-0.5, rng=None):
    """
    ----------
    S : int, number of species (S).
    meanmu : float, mean interaction parameter (unnormalised).
    sigma : float, std dev of interaction parameter (unnormalised).
    gamma : float, correlation between a_ij and a_ji.
    rng : np.random.Generator or None, Random generator for reproducibility.
    -------
    M : (S, S) numpy array, NEGATED Interaction matrix A.
    """
    rng = default_rng() if rng is None else rng

    mean_scaled = meanmu / S
    std_scaled  = sigma / np.sqrt(S)

    Mean  = np.array([mean_scaled, mean_scaled])
    Sigma = np.array([[std_scaled**2, gamma * std_scaled**2],
                      [gamma * std_scaled**2, std_scaled**2]])

    # sample S*S correlated pairs (a_ij, a_ji)
    R = rng.multivariate_normal(Mean, Sigma, size=(S * S))
    R1 = R[:, 0].reshape(S, S)
    R2 = R[:, 1].reshape(S, S)

    A1u = np.triu(R1, 1)        # upper triangle (excluding diag)
    A2u = np.triu(R2, 1)        # another upper triangle → used for lower

    A  = A1u + A2u.T + np.eye(S)  # symmetric-ish + self = 1
    return -A

##### Akshit multistability
'''
    + symmetric matrix can not lead to chaos.
    + https://arxiv.org/pdf/2511.06697
    + how easy it is to know the threshold for stochasticity?
    + Akshit says noone knows that so we have to tune the schochasticity in slowly
    + non zeros entries will create the same outcome as discussed
    + what we observed from the 50 species seed(0) is chaos
    ----------
        dy/dt = y ⊙ (K - y - Ay) = y ⊙ (K-(I+A)y)
    The paper consider strong interaction regime: both meanmu and sigma are finite constants that do not scale with S number of species
        - contrast to May's scaling
        - fix K_i = 1
        - boundary for multistability
            sigma = frac{-mu+1}{squrt{2S}}
    ----------
    Multistability boundary
        \sigma=\frac{1-\mu}{\squrt{2S}}
'''
def im_symmetric_M(S, meanmu=0.5, sigma=0.3, rng=None):
    """
    S : int, number of species (S).
    meanmu : float, positive, mean interaction parameter (unnormalised).
    sigma : float, positive, std dev of interaction parameter (unnormalised).
    rng : np.random.Generator or None, Random generator for reproducibility.
    -------
    M : (S, S) numpy array, NEGATED Interaction matrix A.
    """
    rng = default_rng() if rng is None else rng
    A = rng.normal(loc=meanmu, scale=sigma, size=(S,S))
    while np.any(A == 0.0):
        loc = (A == 0.0)
        A[loc] = rng.normal(loc=meanmu, scale=sigma, size=loc.sum())
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 1.0)
    return -A

def multistability_crit(meanmu, S):
    """
    Patro et al. (Eq. 2): multistability boundary for symmetric random GLV.
        sigma_c = (1 - mu) / sqrt(2S)
    """
    return(1-meanmu)/np.sqrt(2.0*S)
#%%%Drafts
def vis_data(ts, titl = ""):
    fig, ax = plt.subplots(figsize = (10, 2.7))
    ax.set_title(titl)
    for i in range(len(ts)):
        ax.plot(ts[i])

#%% Run
if __name__=="__main__":
    datagen_params = {'mode': "competitive",
                      'dt_s': 0.25, 
                      'N': 500,
                      's0': s0, 
                      'noise_T': 0.05,
                      'noise': 0.05*0.5}
    reps = 1;
    ts_ar = ar_control()
    ts_lv = generate_lv(run_id = 1, dt_s = datagen_params['dt_s'], N=datagen_params['N'], noise=datagen_params['noise'],noise_T=datagen_params['noise_T'],intx="competitive")
    ts_ch = generate_niehaus(run_id=1, dt_s=datagen_params['dt_s'], N=datagen_params['N'], noise=datagen_params['noise'], noise_T=datagen_params['noise_T'], intx=datagen_params['mode'])
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
        
    #%% Testing Vano and s = 1.06875
    s = 1.06875
    mu = mu_vano.copy()
    Ascaled = scale_offdiag(M_vano, s) 
    M = convertAvano2M(Ascaled, mu) 
    datagen_params = {'mode': "vano_s",
                      's': s,
                      'dt_s': 0.25, 
                      'N': 500,
  #                   'noise': 0.001,
                      'noise_T': 0.05,
                      'mu': mu,
                      'M': M}
    s0_list = (np.array([0.1, 0.1, 0.1, 0.1]),
               np.array([0.9, 0.1, 0.1, 0.1]), 
               np.array([0.1, 0.9, 0.1, 0.1]), 
               np.array([0.1, 0.1, 0.9, 0.1]), 
               np.array([0.1, 0.1, 0.1, 0.9]), 
               np.array([0.7, 0.7, 0.1, 0.1]), 
               np.array([0.7, 0.1, 0.7, 0.1]), 
               np.array([0.7, 0.1, 0.1, 0.7]), 
               np.array([0.1, 0.7, 0.7, 0.1]), 
               np.array([0.1, 0.7, 0.1, 0.7]), 
               np.array([0.1, 0.1, 0.7, 0.7]), 
               np.array([0.5, 0.5, 0.5, 0.5]),
               np.array([0.3, 0.6, 0.2, 0.4]), 
               np.array([0.6, 0.3, 0.4, 0.2]), 
               np.array([0.2, 0.4, 0.6, 0.3]), 
               np.array([0.4, 0.2, 0.3, 0.6]))
    data = {'data': [],
            'data_noise': [],
            's0': s0_list}
    for s0 in s0_list:  
        data['data'].append(generate_lv(dt_s=datagen_params['dt_s'], N=datagen_params['N'], s0=s0, mu=datagen_params['mu'], M=datagen_params['M'], noise=0, noise_T=datagen_params['noise_T']))
        data['data_noise'].append(generate_lv(dt_s=datagen_params['dt_s'], N=datagen_params['N'], s0=s0, mu=datagen_params['mu'], M=datagen_params['M'], noise=0.001, noise_T=datagen_params['noise_T']))
        
    # datagen_params = {'mode': 'Vano 4 species',
    #                   'dt_s': 0.25,
    #                   'N': 500,
    #                   's0': s0_list,
    #                   'mu': mu,
    #                   'M': M,
    #                   'noise': 0.001,
    #                   'noise_T': 0.05}
    data['datagen_params'] = datagen_params
    del s, mu, Ascaled, M, s0, s0_list
    
    #%% Polo matlab model    
    np.random.seed(0) 
    noise = 0.001
    n_species = 50
    M = im_may_M(n_species)
    mu = intrinsic_growth_vector_mu(n_species)
    datagen_params = {'mode': "multispecies_may",
                      'dt_s': 0.25, 
                      'N': 500,
                      'noise_T': 0.05,
                      'n_species': n_species,
                      'noise': noise,
                      'mu': mu,
                      'M': M}
    data = {'data': [], 
            'data_noise': [],
            's0' : []}
    for i in range(5):
        s0 = initial_conditions_s0(n_species)
        data['s0'].append(s0)
        data['data'].append(generate_lv(dt_s=datagen_params['dt_s'], N=datagen_params['N'], s0=s0, mu=mu, M=M, noise=0.0, noise_T=datagen_params['noise_T']))
        data['data_noise'].append(generate_lv(dt_s=datagen_params['dt_s'], N=datagen_params['N'], s0=s0, mu=mu, M=M, noise=noise, noise_T=datagen_params['noise_T']))
    data['datagen_params'] = datagen_params
    del noise, n_species, M, mu, s0, i
        
    #%% Akshit model - akshit_50
    np.random.seed(0)
    noise = 0.005
    n_species = 50
    mu = intrinsic_growth_vector_mu(n_species)
    meanmu = 0.5
    sigma_crit = multistability_crit(meanmu, n_species) # 0.05
    sigma = 0.3
    # M = im_symmetric_M(S=n_species, meanmu=meanmu, sigma=sigma)
    datagen_params = {'mode': "multispecies_akshit",
                      'dt_s': 0.25, 
                      'N': 500,
                      'noise_T': 0.05,
                      'n_species': n_species,
                      'noise': noise,
                      'mu': mu,
                      # 'meanmu': meanmu,
                      # 'sigma': sigma,
                      'M': M}
    # datagen_params = akshit_50['datagen_params']
    data = {'data': [], 
            'data_noise': [],
            's0' : []}
    # for i in [0,2,3]:
    #     data['s0'].append(akshit_50['s0'][i])
    for s0 in data['s0']:
        for i in range(3):
            # s0 = initial_conditions_s0(datagen_params['n_species'])
            # data['s0'].append(s0)
            data['data'].append(generate_lv(dt_s=datagen_params['dt_s'], N=datagen_params['N'], s0=s0, mu=datagen_params['mu'], M=datagen_params['M'], noise=0.0, noise_T=datagen_params['noise_T']))
            data['data_noise'].append(generate_lv(dt_s=datagen_params['dt_s'], N=datagen_params['N'], s0=s0, mu=datagen_params['mu'], M=datagen_params['M'], noise=noise, noise_T=datagen_params['noise_T']))
    data['datagen_params'] = datagen_params
    del noise, n_species, mu, meanmu, sigma_crit, sigma, M, s0, i
    # akshit_50[0,2,3]