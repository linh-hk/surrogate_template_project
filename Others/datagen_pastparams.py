# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 23:00:49 2026

@author: hoang

Data generation past parameters
"""
ARGs= []

# EComp
# ARGs.append({'mode': 'EComp', 'dt_s': 0.25, 'N': 500, 's0': np.array([2.,0.]), 'mu': np.array([0.7,0.7]), 'M': np.array([[-0.4,-0.5],[-0.5,-0.4]]), 'noise': 0.01, 'noise_T': 0.05})
# ARGs.append({'mode': 'EComp', 'dt_s': 1.25, 'N': 500, 's0': np.array([2.,0.]), 'mu': np.array([0.7,0.7]), 'M': np.array([[-0.4,-0.5],[-0.5,-0.4]]), 'noise': 0.01, 'noise_T': 0.05})
# ARGs.append({'mode': 'EComp', 'dt_s': 0.25, 'N': 500, 's0': np.array([1.,1.]), 'mu': np.array([0.7,0.7]), 'M': np.array([[-0.4,-0.5],[-0.5,-0.4]]), 'noise': 0.01, 'noise_T': 0.05})
# ARGs = {'mode': 'EComp', 'dt_s': 1.25, 'N': 500, 's0': np.array([1.,1.]), 'mu': np.array([0.7,0.7]), 'M': np.array([[-0.4,-0.5],[-0.5,-0.4]]), 'noise': 0.01, 'noise_T': 0.05}

# EMut
# ARGs.append({'mode': 'EMut', 'dt_s': 0.25, 'N': 500, 's0': np.array([2.,0.]), 'mu': np.array([0.7,0.7]), 'M': np.array([[-0.4,0.3],[0.3,-0.4]]), 'noise': 0.01, 'noise_T': 0.05})
# ARGs.append({'mode': 'EMut', 'dt_s': 1.25, 'N': 500, 's0': np.array([2.,0.]), 'mu': np.array([0.7,0.7]), 'M': np.array([[-0.4,0.3],[0.3,-0.4]]), 'noise': 0.01, 'noise_T': 0.05})
# ARGs.append({'mode': 'EMut', 'dt_s': 0.25, 'N': 500, 's0': np.array([1.,1.]), 'mu': np.array([0.7,0.7]), 'M': np.array([[-0.4,0.3],[0.3,-0.4]]), 'noise': 0.01, 'noise_T': 0.05})
# ARGs = {'mode': , 'dt_s': 1.25, 'N': 500, 's0': np.array([1.,1.]), 'mu': np.array([0.7,0.7]), 'M': np.array([[-0.4,0.3],[0.3,-0.4]]), 'noise': 0.01, 'noise_T': 0.05}
# ARGs.append({'mode': 'EMut', 'dt_s': 0.25, 'N': 500, 's0': np.array([1.,1.]), 'mu': np.array([0.7,0.7]), 'M': np.array([[-0.4,0.5],[0.5,-0.4]]), 'noise': 0.01, 'noise_T': 0.05})
# ARGs.append({'mode': 'EMut', 'dt_s': 1.25, 'N': 500, 's0': np.array([1.,1.]), 'mu': np.array([0.7,0.7]), 'M': np.array([[-0.4,0.5],[0.5,-0.4]]), 'noise': 0.01, 'noise_T': 0.05})

# UComp
# ARGs.append({'mode': 'UComp', 'dt_s': 0.25, 'N': 500, 's0': np.array([2.,0.]), 'mu': np.array([0.8,0.8]), 'M': np.array([[-0.4,-0.5],[-0.9,-0.4]]), 'noise': 0.01, 'noise_T': 0.05})
# ARGs.append({'mode': 'UComp', 'dt_s': 1.25, 'N': 500, 's0': np.array([2.,0.]), 'mu': np.array([0.8,0.8]), 'M': np.array([[-0.4,-0.5],[-0.9,-0.4]]), 'noise': 0.01, 'noise_T': 0.05})
# ARGs.append({'mode': 'UComp', 'dt_s': 0.25, 'N': 500, 's0': np.array([1.,1.]), 'mu': np.array([0.8,0.8]), 'M': np.array([[-0.4,-0.5],[-0.9,-0.4]]), 'noise': 0.01, 'noise_T': 0.05})
# ARGs = {'mode': , 'dt_s': 1.25, 'N': 500, 's0': np.array([1.,1.]), 'mu': np.array([0.8,0.8]), 'M': np.array([[-0.4,-0.5],[-0.9,-0.4]]), 'noise': 0.01, 'noise_T': 0.05}

# UComp2
# ARGs.append({'mode': 'UComp2', 'dt_s': 0.25, 'N': 500, 's0': np.array([2.,0.]), 'mu': np.array([0.8,0.8]), 'M': np.array([[-1.4,-0.5],[-0.9,-1.4]]), 'noise': 0.01, 'noise_T': 0.05})
# ARGs.append({'mode': 'UComp2', 'dt_s': 1.25, 'N': 500, 's0': np.array([2.,0.]), 'mu': np.array([0.8,0.8]), 'M': np.array([[-1.4,-0.5],[-0.9,-1.4]]), 'noise': 0.01, 'noise_T': 0.05})
# ARGs.append({'mode': 'UComp2', 'dt_s': 0.25, 'N': 500, 's0': np.array([1.,1.]), 'mu': np.array([0.8,0.8]), 'M': np.array([[-1.4,-0.5],[-0.9,-1.4]]), 'noise': 0.01, 'noise_T': 0.05})
# ARGs = {'mode': , 'dt_s': 1.25, 'N': 500, 's0': np.array([1.,1.]), 'mu': np.array([0.8,0.8]), 'M': np.array([[-1.4,-0.5],[-0.9,-1.4]]), 'noise': 0.01, 'noise_T': 0.05}

# UComp3
# ARGs.append({'mode': 'UComp3', 'dt_s': 0.25, 'N': 500, 's0': np.array([1,1]), 'mu': np.array([50.,50.]), 'M': np.array([[-100,-95],[-99,-100]]), 'noise': 0.01, 'noise_T': 0.05})
# ARGs = ({'mode': 'UComp3', 'dt_s': 1.25, 'N': 500, 's0': np.array([1,1]), 'mu': np.array([50.,50.]), 'M': np.array([[-100,-95],[-99,-100]]), 'noise': 0.01, 'noise_T': 0.05})
# ARGs.append({'mode': 'UComp3', 'dt_s': 0.25, 'N': 500, 's0': np.array([2,0]), 'mu': np.array([50.,50.]), 'M': np.array([[-100,-95],[-99,-100]]), 'noise': 0.01, 'noise_T': 0.05})
# ARGs.append({'mode': 'UComp3', 'dt_s': 1.25, 'N': 500, 's0': np.array([2,0]), 'mu': np.array([50.,50.]), 'M': np.array([[-100,-95],[-99,-100]]), 'noise': 0.01, 'noise_T': 0.05})
# ARGs.append({'mode': 'UComp3', 'dt_s': 0.25, 'N': 500, 's0': np.array([50.,50.]), 'mu': np.array([50.,50.]), 'M': np.array([[-100,-95],[-99,-100]]), 'noise': 0.01, 'noise_T': 0.05})
# ARGs.append({'mode': 'UComp3', 'dt_s': 1.25, 'N': 500, 's0': np.array([50.,50.]), 'mu': np.array([50.,50.]), 'M': np.array([[-100,-95],[-99,-100]]), 'noise': 0.01, 'noise_T': 0.05})
# ARGs.append({'mode': 'UComp3', 'dt_s': 0.25, 'N': 500, 's0': np.array([100,25]), 'mu': np.array([50.,50.]), 'M': np.array([[-100,-95],[-99,-100]]), 'noise': 0.01, 'noise_T': 0.05})
# ARGs.append({'mode': 'UComp3', 'dt_s': 1.25, 'N': 500, 's0': np.array([100,25]), 'mu': np.array([50.,50.]), 'M': np.array([[-100,-95],[-99,-100]]), 'noise': 0.01, 'noise_T': 0.05})

# Pred-prey 
# ARGs = {'mode': 'predprey', 'dt_s': 1.25, 'N': 500, 's0': np.array([1.,1.]), 'mu': np.array([1.1,-0.4]), 'M': np.array([[0.0,-0.4],[0.1,0.0]]), 'noise': 0.01, 'noise_T': 0.05}

# Vano et al 4-species competitive model
# r = np.array([1, 0.72, 1.53, 1.27])
# A = np.array([[1,    1.09, 1.52, 0   ],
#               [0,    1,    0.44, 1.36], 
#               [2.33, 0,    1,    0.47], 
#               [1.21, 0.51, 0.35, 1   ]])
# mu = r.copy()
# M = -A * np.expand_dims(r, 1) # M = -(r[:, None] * A)
# s0_list = [np.array([0.1, 0.1, 0.1, 0.1]),
#            np.array([0.9, 0.1, 0.1, 0.1]), 
#            np.array([0.1, 0.9, 0.1, 0.1]), 
#            np.array([0.1, 0.1, 0.9, 0.1]), 
#            np.array([0.1, 0.1, 0.1, 0.9]), 
#            np.array([0.7, 0.7, 0.1, 0.1]), 
#            np.array([0.7, 0.1, 0.7, 0.1]), 
#            np.array([0.7, 0.1, 0.1, 0.7]), 
#            np.array([0.1, 0.7, 0.7, 0.1]), 
#            np.array([0.1, 0.7, 0.1, 0.7]), 
#            np.array([0.1, 0.1, 0.7, 0.7]), 
#            np.array([0.5, 0.5, 0.5, 0.5]),
#            np.array([0.3, 0.6, 0.2, 0.4]), 
#            np.array([0.6, 0.3, 0.4, 0.2]), 
#            np.array([0.2, 0.4, 0.6, 0.3]), 
#            np.array([0.4, 0.2, 0.3, 0.6])]
# for s0 in s0_list: 
#     ARGs.append({'mode': 'Vano 4 species', 
#              'dt_s': 0.25, 
#              'N': 500, 
#              's0': s0, 
#              'mu': mu, 
#              'M': M, 
#              'noise': 0, 
#              'noise_T': 0.05})
#     ARGs.append({'mode': 'Vano 4 species', 
#              'dt_s': 0.25, 
#              'N': 500, 
#              's0': s0, 
#              'mu': mu, 
#              'M': M, 
#              'noise': 0.01, 
#              'noise_T': 0.05})

# Multispecies, multistability, no chaos with process noise
# from GenerateData import intrinsic_growth_vector_mu
# n_species = 50
# mu = intrinsic_growth_vector_mu(n_species)
# M = np.load('multispecies_parameters/matrix.npy')
# # for i in [0,2,3]:
# #     s0 = np.load(file=f'multispecies_parameters/s0_{i}.npy')
# #     ARGs.append({'mode': f'multispecies_symmetrical_competition_s0_{i}', 
# #                  'n_species': n_species,
# #                  'dt_s': 0.25, 
# #                  'N': 500,
# #                  's0': s0,
# #                  'mu': mu,
# #                  'M': M,
# #                  'noise': 0.001, # tested for 0.01, they will all oscilate around 0.1 eventually : )
# #                  'noise_T': 0.05})
# # s0 = np.load(file='multispecies_parameters/s0_1.npy')
# # ARGs.append({'mode': 'multispecies_symmetrical_competition_s0_1', 
# #              'n_species': n_species,
# #              'dt_s': 0.25, 
# #              'N': 500,
# #              's0': s0,
# #              'mu': mu,
# #              'M': M,
# #              'noise': 0.001, # tested for 0.01, they will all oscilate around 0.1 eventually : )
# #              'noise_T': 0.05})
# # After stabilising the system as Akshit suggested
# # for i in [1,2,3]:
# #     s0 = np.load(file=f'multispecies_parameters/s0_{i}.npy')
# #     ARGs.append({'mode': f'stabilised_multispecies_s0_{i}', 
# #                  'n_species': n_species,
# #                  'dt_s': 0.25, 
# #                  'N': 500,
# #                  's0': s0,
# #                  'mu': mu,
# #                  'M': M,
# #                  'noise': 0.001, # tested for 0.01, they will all oscilate around 0.1 eventually : )
# #                  'noise_T': 0.05})
# # # Added time skip for s_3 because some of them wiggled
# # s0 = np.load(file='multispecies_parameters/s0_3.npy')
# # ARGs.append({'mode': 'stabilised_multispecies_s0_3', 
# #              'n_species': n_species,
# #              'dt_s': 0.25, 
# #              'N': 500,
# #              's0': s0,
# #              'mu': mu,
# #              'M': M,
# #              'noise': 0.001, # tested for 0.01, they will all oscilate around 0.1 eventually : )
# #              'noise_T': 0.05,
# #              'time_skip': 275})
# meanmu = 0.5
# sigma_crit = multistability_crit(meanmu, n_species) # 0.05
# sigma = 0.3
# # M = im_symmetric_M(S=n_species, meanmu=meanmu, sigma=sigma)

# Make sure to state in the paper that the parameters are the same
