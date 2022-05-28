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

scale_km2m = 1e3
scale_m2km = 1e-3

from functions_GOA import initialize_GOA, initialize_wind_statistic, initialize_floris_interfaces
from functions_GOA import optimize_layout_GOA

#%% SET PARAMETERS

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
sim_ID = 1
load_init_pops_from_file = False
# load_init_pops_from_file = True
# number of init. pops to plot
N_plot_init_pops = 8

# SET
path_base = '/path/to/folder'
path_output_folder = path_base + os.sep + f'Npop_{N_pop}_Nwt_{N_wt}_Niter_{N_ITER}_simID_{sim_ID}'
path_fig_folder = path_output_folder + os.sep + 'figures'

if not os.path.exists(path_output_folder): 
    os.makedirs(path_output_folder)
if not os.path.exists(path_fig_folder): 
    os.makedirs(path_fig_folder)

params = [N_pop, N_wt, hub_height, rotor_diameter, min_dist, init_min_dist,
         turbine_grid_points, wt_power, no_wdir_bins_opt, wspeed_opt, wake_model,
         N_grid_points_short_side, N_PROCESSES_PARALLEL, N_ITER, mu_I, mu_F, p_M, n_E,
         max_iter_cross_over, sim_ID]
params_str = 'N_pop, N_wt, hub_height, rotor_diameter, min_dist, init_min_dist, ' \
+ 'turbine_grid_points, wt_power, no_wdir_bins_opt, wspeed_opt, wake_model, ' \
+ 'N_grid_points_short_side, N_PROCESSES_PARALLEL, N_ITER, mu_I, mu_F, p_M, n_E, ' \
+ 'max_iter_cross_over, sim_ID'
with open(path_output_folder + os.sep + 'params.txt', 'w') as f:
    f.write(params_str)
    f.write('\n')
    for el in params:
        f.write(str(el))
        f.write(' ')

#%% init pops
x_init_all_pops, y_init_all_pops = \
    initialize_GOA(load_init_pops_from_file, path_output_folder,
                   path_output_folder,
                   boundaries,
                   N_pop, N_wt, min_dist, init_min_dist,
                   N_plot_init_pops, path_fig_folder,
                   N_iter_init_main_loop=100, N_iter_init_sub_loop=10000)

#%% init wind statistic
# '_opt' files are used for optmization
# '_power' files are used for power calc
# for now, the simplification is used that '_opt' and '_power' use the same files
wdir_bin_centers_opt, wspeed_bin_centers_opt, hist_wdir_wspeed_opt,\
power_fraction_wdir_wspeed_opt, wdir_bin_centers_power, wspeed_bin_centers_power,\
hist_wdir_wspeed_power, power_fraction_wdir_wspeed_power = \
    initialize_wind_statistic(path_input_folder + os.sep + 'wind_data',
                              no_wdir_bins_opt, wspeed_opt, hub_height)

#%% init FLORIS interfaces
fi_list = initialize_floris_interfaces(x_init_all_pops, y_init_all_pops,
                                       wake_model, turbine_grid_points, wt_power,
                                       path_input_folder)

#%% optimization loop (N_ITER iterations)
X_S_init, Y_S_init, X_S, Y_S, X_U, Y_U, X_V, Y_V, ind_best_layout_list,\
    Err_Best, Err_Mean, farm_power_no_wake =\
        optimize_layout_GOA(x_init_all_pops, y_init_all_pops, fi_list,
                            boundaries, min_dist, N_grid_points_short_side,
                            wdir_bin_centers_power, wspeed_bin_centers_power,
                            power_fraction_wdir_wspeed_power,
                            power_fraction_wdir_wspeed_opt,
                            N_PROCESSES_PARALLEL,
                            N_ITER, mu_I, mu_F, p_M, n_E, max_iter_cross_over
                            )

#%% save files
np.savetxt(path_output_folder + os.sep + 'x_init_all_pops.txt', x_init_all_pops)
np.savetxt(path_output_folder + os.sep + 'y_init_all_pops.txt', y_init_all_pops)
np.savetxt(path_output_folder + os.sep + 'x_opt_all_pops.txt', X_V)
np.savetxt(path_output_folder + os.sep + 'y_opt_all_pops.txt', Y_V)
np.savetxt(path_output_folder + os.sep + 'Err_Best.txt', Err_Best)
np.savetxt(path_output_folder + os.sep + 'Err_Mean.txt', Err_Mean)
np.savetxt(path_output_folder + os.sep + 'ind_best_layout_list.txt', ind_best_layout_list)

#%% plot init and final best layout

MS = 3
fig, ax = plt.subplots(figsize=cm2inch(12,8))
ax.fill(*boundaries.T, fill = False, edgecolor='orange', zorder=99, lw=0.5)
ax.plot(X_S_init, Y_S_init, 'bo', ms=MS, label='init best', fillstyle='none')
ax.plot(X_S, Y_S, 'rd', ms=MS, label='optimized best', fillstyle='none')
ax.axis('equal')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
lgnd = ax.legend(loc='center left', bbox_to_anchor=((1.0, 0.5)), markerscale=1)
fig.tight_layout()
name_fig = 'best_config.png'
fig.savefig(path_fig_folder + os.sep + name_fig, dpi=600)
plt.show()
