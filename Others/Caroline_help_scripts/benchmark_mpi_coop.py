#Create time series and submit dependence test jobs

import numpy as np
#import matplotlib.pyplot as plt
from dependence_test_functions import benchmark_stats
from scipy.integrate import solve_ivp
from mpi4py.futures import MPIPoolExecutor
import time

def test_stats(run_id):
    x = 0.1*np.ones(200);
    y = 0.1*np.ones(200);
    for i in range(199):
        #putative causee. if this equation has y in it, test should come back true
        x[i+1] = 0.5*x[i] + 0.2*y[i] + 0.1 * np.random.random()
        #putative causer
        y[i+1] = 0.3 * y[i] + 0.4 * x[i] + 0.1 * np.random.random()

    pvals = benchmark_stats(x,y);
    return pvals;

def lotkaVolterra(t,y,mu,M):
    return y * (mu + M @ y);

def test_stats_lv(run_id, dt_s, N, noise,noise_T,intx="competitive"):
    dt=0.05;
    mu = np.tile([0.7,0.7],2); # two sets of interacting pairs
    M = np.zeros((4,4));
    if (intx=="competitive"):
        M[:2,:2] = [[-0.4,-0.5],[-0.5,-0.4]];
        M[2:,2:] = M[:2,:2];
    if (intx=="asym_competitive"):
        mu = np.tile([0.8,0.8],2);
        M[:2,:2] = [[-0.4,-0.5],[-0.9,-0.4]];
        M[2:,2:] = M[:2,:2];
    if (intx=="asym_competitive_2"):
        mu = np.tile([0.8,0.8],2);
        M[:2,:2] = [[-1.4,-0.5],[-0.9,-1.4]];
        M[2:,2:] = M[:2,:2];
    if (intx=="asym_competitive_2_rev"):
        mu = np.tile([0.8,0.8],2);
        M[:2,:2] = [[-1.4,-0.9],[-0.5,-1.4]];
        M[2:,2:] = M[:2,:2];   
    if (intx=="asym_competitive_3"):
        mu = np.tile([50.,50.],2);
        M[:2,:2] = [[-100.,-95.],[-99.,-100.]]
        M[2:,2:] = M[:2,:2];
    if (intx=="mutualistic"):
        M[:2,:2] = [[-0.4,0.3],[0.3,-0.4]];
        M[2:,2:] = M[:2,:2];
    if (intx=="predprey"): # pred-prey
        mu = np.tile([1.1,-0.4],2);
        M[:2,:2] = [[0.0,-0.4],[0.1,0.0]];
        M[2:,2:] = M[:2,:2];
    sample_period = int(np.ceil(dt_s / dt));
    
    lag = int(150/dt);
    obs = sample_period * N;

    if run_id % 100 == 0:
        print("test " + str(run_id) + " starting");
    s = np.zeros((lag + obs + 1, 4))
    s[0] = 2.*np.random.random(4);
    #s[0] = [2.,0.1,2.,0.1]; # ensure y is dominant and x is close to extinct
    for i in range(lag + obs):
        soln = solve_ivp(lotkaVolterra,[0,dt],s[i],args=(mu,M))
        eps = noise*np.random.randn(4)*np.random.binomial(1,dt/noise_T,size=4);
        s[i+1] = soln.y[:,-1] + eps;
        s[i+1][np.where(s[i+1] < 0)] = 0

    s += 0.1*np.random.randn(s.size).reshape(s.shape);
    x = s[lag::sample_period,0]
    y = s[lag::sample_period,1]
    yf = s[lag::sample_period,3]
    
    tic = time.time();

    # testing individual statistics
    pvals_true = benchmark_stats(x,y,maxlag=10,test_list=['tts_multishift'])
    pvals_false = benchmark_stats(x,yf,maxlag=10,test_list=['tts_multishift'])

    if run_id % 100 == 0:
        print("test %d finished at %.2f sec" % (run_id, time.time() - tic));
    
    return {"true":pvals_true,"false":pvals_false};
#     return [x,y]
# test1 = test_stats_lv(1, dt_s = 0.25, N=500, noise=0.01, noise_T=0.05, intx='competitive')
# vis_data(test1, 'xy_Caroline_LV',1)

#data generation
# diffeq
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
    Lambda = np.matmul(K_dd - rho_minus, C)
    # compute dS/dt
    S_prime = (r0 + Lambda) * S
    # compute K_dag
    C_broadcasted = np.zeros_like(K.T) + C
    K_dag = np.reciprocal(C_broadcasted + K.T) * C_broadcasted
    # compute dC/dt
    C_prime = np.matmul(beta - (alpha * K_dag), S)
    SC_prime = np.vstack((S_prime, C_prime))
    return SC_prime

#for chemically mediated interactions (add constant rsrc flux)
def sc_prime_rsrc(t, y, num_spec, r0, K, alpha, beta, rho_plus, rho_minus,r_flux):
    dy = dSCdt(y, num_spec, r0, K, alpha, beta, rho_plus, rho_minus);
    dy[-1] += r_flux;
    return np.reshape(dy, dy.size).tolist()

def test_stats_niehaus(run_id, dt_s, N, noise, noise_T, intx="competitive"):
    dt=0.05;
    sample_period = int(np.ceil(dt_s / dt));
    
    lag = int(150/dt);
    obs = sample_period * N;
    
    if (intx=="competitive"):
        num_spec = 2
        num_chem = 1
        r0 = np.array([[-1.6],[-1.6]])
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
        r0 = np.array([[-2.4],[-2.4]])
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
    
    if run_id % 100 == 0:
        print("test " + str(run_id) + " starting");
    s = np.zeros((lag + obs + 1, num_spec + num_chem))
    s[0] = np.zeros((num_spec + num_chem))
    s[0,:2] = 1;

    tic = time.time()
    
    for i in range(lag + obs):
        soln = solve_ivp(sc_prime_rsrc,[0,dt],s[i],args=params);
        s[i+1] = soln.y[:,-1] + noise*np.random.binomial(1,dt/noise_T,size=num_spec+num_chem) \
                                * np.random.random(num_spec + num_chem);
        s[i+1][np.where(s[i+1] < 0)] = 0

    s += 0.001*np.random.randn(s.size).reshape(s.shape)
    x1 = s[lag::sample_period,0].copy()
    #y is index 1 for dependent, 3 for independent
    y1 = s[lag::sample_period,1].copy()

    pvals_true = benchmark_stats(x1,y1,test_list=[''],maxlag=0);
    
    #second, independent rep
    for i in range(lag + obs):
        soln = solve_ivp(sc_prime_rsrc,[0,dt],s[i],args=params);
        s[i+1] = soln.y[:,-1] + noise*np.random.binomial(1,dt/noise_T,size=num_spec+num_chem)\
                 *np.random.random(num_spec + num_chem);
        s[i+1][np.where(s[i+1] < 0)] = 0
    
    s += 0.001*np.random.randn(s.size).reshape(s.shape)
    y2 = s[lag::sample_period,1].copy() # different realization of y
    
    pvals_false = benchmark_stats(x1,y2,test_list=[''],maxlag=0);
    
    if run_id % 100 == 0:
        print("test %d finished at %.2f sec" % (run_id, time.time() - tic));
    
    return {"true":pvals_true,"false":pvals_false};
    
    
if __name__=="__main__":

    pval_count_granger = {'randphase':0, 'bstrap':0, 'twin':0, 'tts':0, 'tts_naive':0, 'circperm':0,'perm':0,'tts_multishift':0}
    pval_count_lsa_new = {'randphase':0, 'bstrap':0, 'twin':0, 'tts':0, 'tts_naive':0, 'circperm':0,'perm':0,'tts_multishift':0}
    pval_count_pcorr = {'randphase':0, 'bstrap':0, 'twin':0, 'tts':0, 'tts_naive':0, 'circperm':0,'perm':0,'tts_multishift':0}
    pval_count_ccm = {'randphase':0, 'bstrap':0, 'twin':0, 'tts':0, 'tts_naive':0, 'circperm':0,'perm':0,'tts_multishift':0}
    pval_parametric = {'pearson':0}
    pval_granger_nosurr = {'granger_nosurr':0}
    pval_lsa_nosurr = {'lsa':0}

    pval_count_true = {"granger_y->x":pval_count_granger.copy(), \
              "granger_x->y":pval_count_granger.copy(), \
                  "lsa":pval_count_lsa_new.copy(), \
                  "pearson":pval_count_pcorr.copy(), \
                  "ccm_y->x":pval_count_ccm.copy(), \
                  "ccm_x->y":pval_count_ccm.copy(), \
                  "mutual_info":pval_count_ccm.copy(), \
                  "pcc_param":pval_parametric.copy(), \
                  "granger_param_y->x":pval_granger_nosurr.copy(), \
                  "granger_param_x->y":pval_granger_nosurr.copy(), \
                  "lsa_data_driven":pval_lsa_nosurr.copy()};
    pval_count_false = {"granger_y->x":pval_count_granger.copy(), \
              "granger_x->y":pval_count_granger.copy(), \
                  "lsa":pval_count_lsa_new.copy(), \
                  "pearson":pval_count_pcorr.copy(), \
                  "ccm_y->x":pval_count_ccm.copy(), \
                  "ccm_x->y":pval_count_ccm.copy(), \
                  "mutual_info":pval_count_ccm.copy(), \
                  "pcc_param":pval_parametric.copy(), \
                  "granger_param_y->x":pval_granger_nosurr.copy(), \
                  "granger_param_x->y":pval_granger_nosurr.copy(), \
                  "lsa_data_driven":pval_lsa_nosurr.copy()};

    reps = 1;
    noise=0.01;
    noise_T=0.05;
    dt_s=0.25;
    n_t =1000;
    
    with MPIPoolExecutor() as executor:
        resultsIter = map(test_stats_lv, np.arange(reps),
                          np.tile(dt_s, reps), # sample rate
                          np.tile(n_t, reps),  # number of time points
                          np.tile(noise,reps), # noise
                          np.tile(noise_T,reps), # noise period
                          np.tile("predprey",reps))
                          #unordered=True) 
    
        resultsList = [_ for _ in resultsIter]

    for res in resultsList:
        for i in res["true"].keys(): #statistic
            for k in res["true"][i].keys(): # surrogate method
                pval_count_true[i][k] += (res["true"][i][k] < 0.05);
                pval_count_false[i][k] += (res["false"][i][k] < 0.05);

    print("glv predprey, {0} noise at {1} frequency, {2} trials, n={3}, sample rate={4}".format(noise, noise_T, reps, n_t, dt_s))
    print("True positive rate: \n" + str(pval_count_true) + \
          "\nFalse positive rate:\n" + str(pval_count_false))

    #f = open("results.txt","w");
    #f.write("1e4 replicates\nExpected result: False\n " + str(pval_count_elsa) + "\nAlex's LSA: " + str(pval_count_lsa_new))
    #f.close()
