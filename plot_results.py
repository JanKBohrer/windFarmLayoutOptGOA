"""
@author: bohrer
based on GA algorithm by mendez
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from helper_functions import cm2inch

plt.rcParams.update(plt.rcParamsDefault)
# %matplotlib inline
# plt.rc('text', usetex=True)      # This is for plot customization
plt.rc('figure', dpi=600)
SMALL_SIZE = 8
MEDIUM_SIZE = 8
BIGGER_SIZE = 8
plt.rc('font', family='sans')
plt.rc('font', size=MEDIUM_SIZE) # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE) # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE) # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE) # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE) # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE) # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE) # fontsize of the figure title

#%%

path_input_folder = '/path/to/folder'

name_file_boundaries = 'boundaries_for_16_wt.txt'
boundaries = np.loadtxt(path_input_folder + os.sep + name_file_boundaries)

# number of populations to initialize -> will be overwritten, if pops are loaded from file
N_pop = 100
# number of wind turbines (there will be two variables per wind turbine (x,y))
# will be overwritten, if pops are loaded from file
N_wt = 16
hub_height = 146 # hub height in m
rotor_diameter = 236 # rotor diameter in m
min_dist = 5 * rotor_diameter # any two turbines cannot be closer than this distance
init_min_dist = min_dist * 1.01 # min distance for the initial configurations

# Set the resolution of points on each turbine rotor plane (3 is minimum, i.e. 3x3 in x,y)
turbine_grid_points = 3
wt_power = 15 # 5, 10 or 15 MW

# number of wind direction bins for optimization
no_wdir_bins_opt = 72 # available are 36, 45, 60, 72, 90, 120, 180
# simplification: use one single effective wind speed
wspeed_opt = 10.5 # m/s

wake_model = 'gch'
# wake_model = 'cc'
# wake_model = 'jens'

# number of grid points covering the shorter side of the bounding box around the domain
N_grid_points_short_side = 500

# number of processors for the multiprocessing pool function
N_PROCESSES_PARALLEL = 16

# GOA parameters (see function def)
N_ITER = 100
mu_I = 0.3
mu_F = 0.5
p_M = 0.5
n_E = 0.05

# max iterations in the cross-over loops
# if no cross-over can be found in the domain for a certain pair of populations,
# the population with the lower error is chosen as target
max_iter_cross_over = 10000

# simulation ID
sim_ID = 3
load_init_pops_from_file = False
# load_init_pops_from_file = True
# number of init. pops to plot
N_plot_init_pops = 8

path_base = '/home/jj/Nextcloud/VKI/course-notes/DDFM/DDFM_project/DDFM_python/outputs'
path_output_folder = path_base + os.sep + f'Npop_{N_pop}_Nwt_{N_wt}_Niter_{N_ITER}_simID_{sim_ID}'
path_fig_folder = path_output_folder + os.sep + 'figures'

#%%

Err_Best = np.loadtxt(path_output_folder + os.sep + 'Err_Best.txt')
Err_Mean = np.loadtxt(path_output_folder + os.sep + 'Err_Mean.txt')

Iter_list = np.arange(0, len(Err_Best))


MS = 3
fig, ax = plt.subplots(figsize=cm2inch(8,6))
# ax.fill(*boundaries.T, fill = False, edgecolor='orange', zorder=99, lw=0.5)
# ax.plot(X_S_init, Y_S_init, 'bo', ms=MS, label='init best', fillstyle='none')
# ax.plot(X_S, Y_S, 'rd', ms=MS, label='optimized best', fillstyle='none')
# ax.axis('equal')
ax.plot(Iter_list, Err_Best*100)
ax.set_xlabel('Iteration')
ax.set_ylabel('Best error function (%)')
# lgnd = ax.legend(loc='center left', bbox_to_anchor=((1.0, 0.5)), markerscale=1)
ax.set_ylim(-100, -99.4)
fig.tight_layout()
name_fig = 'Err_Best_vs_Iter.png'
fig.savefig(path_fig_folder + os.sep + name_fig, dpi=600)
plt.show()

MS = 3
fig, ax = plt.subplots(figsize=cm2inch(8,6))
# ax.fill(*boundaries.T, fill = False, edgecolor='orange', zorder=99, lw=0.5)
# ax.plot(X_S_init, Y_S_init, 'bo', ms=MS, label='init best', fillstyle='none')
# ax.plot(X_S, Y_S, 'rd', ms=MS, label='optimized best', fillstyle='none')
# ax.axis('equal')
ax.plot(Iter_list, Err_Mean*100)
ax.set_xlabel('Iteration')
ax.set_ylabel('Mean error function (%)')
# lgnd = ax.legend(loc='center left', bbox_to_anchor=((1.0, 0.5)), markerscale=1)
# ax.set_ylim(-100, -99.4)
fig.tight_layout()
name_fig = 'Err_Mean_vs_Iter.png'
fig.savefig(path_fig_folder + os.sep + name_fig, dpi=600)
plt.show()


