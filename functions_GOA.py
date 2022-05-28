"""
@author: bohrer
based on GA algorithm by mendez
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import multiprocessing as mp
from shapely.geometry.polygon import Polygon
from scipy.spatial.distance import cdist
from floris.tools import FlorisInterface

from geometries import Grid, Domain
from helper_functions import cm2inch, get_farm_power_no_wake, get_farm_power_single
from helper_functions import project_onto_eigvec_2D, draw_random_position_in_domain
from helper_functions import progress
from helper_functions import update_mask_domain_and_min_dist
from helper_functions import draw_rnd_pt_from_allowed_domain
from helper_functions import gen_kernel_mask_min_dist, gen_mask_domain, gen_mask_domain_and_min_dist

#%%
def gen_init_pops(N_pop, N_wt, min_dist, init_min_dist, boundaries,
                  path_output_folder, path_fig_folder,
                  N_iter_init_main_loop=100, N_iter_init_sub_loop=10000):
    """
    Parameters
    ----------
    N_pop : int
        number of populations
    N_wt : int
        number of wind turbines
    min_dist : float
        min. distance between any two turbines 
    init_min_dist : float
        min. distance between any two turbines for initialization configs
    boundaries : N_points x 2 array
        domain boundary corner points [ [x0, y0], [x1, y1], ... ] 
    path_output_folder : string
        path to the output FOLDER
    path_fig_folder : string
        path to the output FOLDER
    N_iter_init_main_loop : int, optional
        max. number of iterations for the main loop. The default is 100.
    N_iter_init_sub_loop : int, optional
        max. number of iterations for the sub loop. The default is 10000.

    Returns
    -------
    x_init_all_pops : N_wt x N_pop array
        init. x-positions of the wind turbines for all pops
    y_init_all_pops : N_wt x N_pop array
        init. y-positions of the wind turbines

    """    
    boundaries_ = np.copy(boundaries)

    # scale domain by the minimum distance
    boundaries_ /= min_dist
    # boundaries_orig_scaled = np.copy(boundaries_)
    
    # for initialization: find the 'long' and 'short' directions of the domain using PCA
    pca = PCA(n_components=2, copy=True)
    pca.fit(boundaries_)
    # normed eigenvector of the main direction (largest variance of data)
    eigvec0 = pca.components_[0] 
    
    # project, such that the main PCA direction is the new x-axis
    # this is done to increase the hit rate of coordinates drawn from the 
    # surrounding bounding box to actually hit the domain
    boundaries_ = project_onto_eigvec_2D(boundaries_.T, eigvec0).T
    
    # Polygon object for further processing
    boundary_polygon = Polygon(boundaries_)
        
    # find a random initial configuration inside the domain, respecting the minimum distance
    
    # domain limits
    domain_limits = np.array([
        [boundaries_[:,0].min(), boundaries_[:,0].max()],
        [boundaries_[:,1].min(), boundaries_[:,1].max()]
        ])
    domain_dimensions = np.array([
        domain_limits[0,1]-domain_limits[0,0],
        domain_limits[1,1]-domain_limits[1,0],
        ])
    
    x_init_all_pops = []
    y_init_all_pops = []
    
    for pop_n in range(N_pop):
        print()
        print('### Init. of Pop', pop_n, '###')
        x_init = []
        y_init = []
        
        iter_main_n = 0
        # rand_cnt = 0
        init_successful = False
        while iter_main_n < N_iter_init_main_loop:
            x_init = []
            y_init = []
            iter_main_n += 1
            
            # x_init.append(rand_x[rand_cnt])
            # y_init.append(rand_y[rand_cnt])
            # rand_cnt +=1
            x_, y_ = draw_random_position_in_domain(domain_dimensions,
                                                    domain_limits,
                                                    boundary_polygon)
            x_init.append(x_)
            y_init.append(y_)
            
            for wt_n in range(1, N_wt):
                iter_sub_n = 0
                sub_loop_successful = False
                while iter_sub_n < N_iter_init_sub_loop:
                    iter_sub_n += 1
                    # repeat_draw = True
                    # x_ = rand_x[rand_cnt]
                    # y_ = rand_y[rand_cnt]
                    # rand_cnt += 1
                    x_, y_ = draw_random_position_in_domain(domain_dimensions,
                                                            domain_limits,
                                                            boundary_polygon)
                    if np.sum(cdist( list(zip(x_init, y_init)), [[x_, y_]]  ) < init_min_dist/min_dist) == 0:
                        x_init.append(x_)
                        y_init.append(y_)
                        iter_sub_n = N_iter_init_sub_loop
                        sub_loop_successful = True
                if not sub_loop_successful:
                    print('Init. of Pop', pop_n,
                          ': sub-loop failed in main loop iteration', iter_main_n,
                          'at turbine number', wt_n, ', starting retry of main loop')
                    break
                
            if len(x_init) == N_wt:
                iter_main_n = N_iter_init_main_loop
                init_successful = True
        
        if init_successful:
            x_init_all_pops.append(x_init)
            y_init_all_pops.append(y_init)
            print('Init. of Pop', pop_n, 'sucessful')
        
        else:
            print('XXX Careful: Init of Pop', pop_n, 'failed, continuing with next Pop')
   
    # rotate back to real domain
    x_init_all_pops_rot = np.array(x_init_all_pops)
    y_init_all_pops_rot = np.array(y_init_all_pops)
        
    x_init_all_pops, y_init_all_pops =\
        project_onto_eigvec_2D([x_init_all_pops_rot, y_init_all_pops_rot],
                               [eigvec0[0], -eigvec0[1]])
    
    # transpose here, to have N_wt x N_pop arrays
    x_init_all_pops = x_init_all_pops.T
    y_init_all_pops = y_init_all_pops.T
    # saving the coordinates in meter for general usage (independent of min_dist)
    x_init_all_pops *= min_dist
    y_init_all_pops *= min_dist
    
    np.savetxt(path_output_folder + os.sep + 'x_init_all_pops.txt', x_init_all_pops)
    np.savetxt(path_output_folder + os.sep + 'y_init_all_pops.txt', y_init_all_pops)
    
    return x_init_all_pops, y_init_all_pops
    
def initialize_GOA(load_init_pops_from_file, path_output_folder, path_input_folder=None,
                   boundaries = np.array(((0,0),(1e4,0),(1e4,1e4),(0,1e4))),
                   N_pop=10, N_wt=16, min_dist=1500, init_min_dist=1515,
                   N_plot_init_pops=0, path_fig_folder=None,
                   N_iter_init_main_loop=100, N_iter_init_sub_loop=10000):
    """
    Parameters
    ----------
    load_init_pops_from_file : bool
        load init. layouts from file or generate them randomly here
    path_output_folder : string
        folder, where init. layouts will be stored
    path_input_folder : string, optional
        folder, where init. layouts will be loaded from. The default is None.
    boundaries : N_points x 2 array
        domain boundary corner points [ [x0, y0], [x1, y1], ... ] 
    N_pop : int
        number of populations
    N_wt : int
        number of wind turbines
    min_dist : float
        min. distance between any two turbines 
    init_min_dist : float
        min. distance between any two turbines for initialization configs
    N_plot_init_pops : int, optional
        how many init. layouts should be plotted. The default is 0.
    path_fig_folder : string, optional
        folder, where figure are stored. The default is None.
    N_iter_init_main_loop : int, optional
        max. number of iterations for the main loop. The default is 100.
    N_iter_init_sub_loop : int, optional
        max. number of iterations for the sub loop. The default is 10000.

    Returns
    -------
    x_init_all_pops : N_wt x N_pop array
        init. x-positions of the wind turbines for all pops
    y_init_all_pops : N_wt x N_pop array
        init. y-positions of the wind turbines

    """
    if load_init_pops_from_file:
        x_init_all_pops = np.loadtxt(path_input_folder + os.sep + 'x_init_all_pops.txt')
        y_init_all_pops = np.loadtxt(path_input_folder + os.sep + 'y_init_all_pops.txt')
        N_wt, N_pop = x_init_all_pops.shape
    else:
        x_init_all_pops, y_init_all_pops =\
            gen_init_pops(N_pop, N_wt, min_dist, init_min_dist, boundaries,
                          path_output_folder, path_fig_folder,
                          N_iter_init_main_loop, N_iter_init_sub_loop)

    if N_plot_init_pops > 0:
        NCOLS = 4
        NROWS = int(np.ceil(N_plot_init_pops / NCOLS))
        figsize = cm2inch(6 * NCOLS , 6 * NROWS)
        fig, axes = plt.subplots(NROWS, NCOLS, figsize=figsize)
        axes = axes.flatten()
        for ax_n in range(N_plot_init_pops):
            ax = axes[ax_n]
            x_init = x_init_all_pops[:,ax_n]
            y_init = y_init_all_pops[:,ax_n]
            ax.fill(*boundaries.T, fill = False, edgecolor='grey')
            ax.plot(x_init, y_init, 'k.')
            ax.axis('equal')    
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.set_title('initial populations')
        fig.tight_layout()
        
        name_fig = f'init_{N_plot_init_pops}_pops.png'
        fig.savefig(path_fig_folder + os.sep + name_fig, dpi=300)
        
        plt.show()
    
    return x_init_all_pops, y_init_all_pops

#%%

def initialize_wind_statistic(path_data_folder, no_wdir_bins_opt,
                              wspeed_opt, hub_height):
    """

    Parameters
    ----------
    path_data_folder : string
        DESCRIPTION.
    no_wdir_bins_opt : string
        DESCRIPTION.
    wspeed_opt : float
        wind speed used for optimization (in this version, a single effective speed is used)
    hub_height : float
        WT hub height

    Returns
    -------
    wdir_bin_centers_opt : numpy.array
        wind direction bin centers used for optimization
    wspeed_bin_centers_opt : numpy.array
        wind speed bin centers used for optimization
    hist_wdir_wspeed_opt : numpy.array
        normalized occurrence histogram used for optimization
    power_fraction_wdir_wspeed_opt : numpy.array
        normalized power fraction histogram used for optimization
    wdir_bin_centers_power : numpy.array
        wind direction bin centers used for init and final power calculation
    wspeed_bin_centers_power : numpy.array
        wind speed bin centers used for init and final power calculation
    hist_wdir_wspeed_power : numpy.array
        normalized occurrence histogram used for init and final power calculation
    power_fraction_wdir_wspeed_power : numpy.array
        normalized power fraction histogram used for init and final power calculation

    """
    name_file = f'wdir_bin_centers_{no_wdir_bins_opt:.0f}.txt'
    wdir_bin_centers_opt = np.loadtxt(path_data_folder + os.sep + name_file)

    # 'bin centers' used by floris
    wspeed_bin_centers_opt = np.array([wspeed_opt])

    name_file = f'histo_norm_Nwdir_{no_wdir_bins_opt:.0f}_Nwspeed_29_H{hub_height:.0f}.txt'
    hist_wdir_wspeed_opt = np.loadtxt(path_data_folder + os.sep + name_file)
    name_file = f'power_fraction_Nwdir_{no_wdir_bins_opt:.0f}_Nwspeed_29_H{hub_height:.0f}.txt'
    power_fraction_wdir_wspeed_opt = np.loadtxt(path_data_folder + os.sep + name_file)

    # hist_wdir_opt = hist_wdir_wspeed_opt.sum(axis=1)
    power_fraction_wdir_opt = power_fraction_wdir_wspeed_opt.sum(axis=1)

    if wspeed_bin_centers_opt.size == 1:
        power_fraction_wdir_wspeed_opt = power_fraction_wdir_opt[:, None]

    # for now, use the same number of direction / speed bins for power calc as for optimization
    power_fraction_wdir_wspeed_power = power_fraction_wdir_wspeed_opt
    hist_wdir_wspeed_power = hist_wdir_wspeed_opt
    wdir_bin_centers_power = wdir_bin_centers_opt
    wspeed_bin_centers_power = wspeed_bin_centers_opt

    # hist_wspeed_power = hist_wdir_wspeed_power.sum(axis=0)
    # hist_wdir_power = hist_wdir_wspeed_power.sum(axis=1)
    # power_fraction_wspeed_power = power_fraction_wdir_wspeed_power.sum(axis=0)
    # power_fraction_wdir_power = power_fraction_wdir_wspeed_power.sum(axis=1)
    
    return wdir_bin_centers_opt, wspeed_bin_centers_opt, hist_wdir_wspeed_opt, power_fraction_wdir_wspeed_opt,\
           wdir_bin_centers_power, wspeed_bin_centers_power, hist_wdir_wspeed_power, power_fraction_wdir_wspeed_power

#%%
def initialize_floris_interfaces(x_init_all_pops, y_init_all_pops,
                                 wake_model, turbine_grid_points, wt_power,
                                 path_input_folder):
    """
    Parameters
    ----------
    x_init_all_pops : N_wt x N_pop array
        init. x-positions of the wind turbines for all pops
    y_init_all_pops : N_wt x N_pop array
        init. y-positions of the wind turbines
    wake_model : string
        FLORIS wake model, one of [gch, cc, jens]
    turbine_grid_points : int
        FLORIS resolution of the rotor blade area ( N x N points per rotor area )
    wt_power : float
        wind turbine power in MW
    path_input_folder : string
        folder, where the FLORIS config files lie

    Returns
    -------
    fi_list : TYPE
        DESCRIPTION.

    """
    N_wt, N_pop = x_init_all_pops.shape
    
    if wake_model == 'gch':
        file_floris_config = 'gch_input01.yaml'
    elif wake_model == 'cc':
        file_floris_config = 'cc_input01.yaml'
    elif wake_model == 'jens':
        file_floris_config = 'jensen_input01.yaml'
    
    fi_list = []
    
    for cnt in range(N_pop):
        fi_list.append(FlorisInterface(path_input_folder + os.sep + file_floris_config))
    
        solver_settings = {
          "type": "turbine_grid",
          "turbine_grid_points": turbine_grid_points
        }
        
        if wt_power == 5:
            turbine_type = ['nrel_5MW']
        elif wt_power in [10,15]:
            turbine_type = [f'iea_{wt_power}MW']
        else: print('!invalid turbine power!')
    
        layout_x = x_init_all_pops[:, cnt]
        layout_y = y_init_all_pops[:, cnt]
        
        fi_list[cnt].reinitialize(solver_settings=solver_settings,
                                  layout=(layout_x, layout_y),
                                  turbine_type=turbine_type)    
    return fi_list

#%%
def Update_POP(X_V, Y_V, Err_1, grid, domain, min_dist,
               mask_domain, mask_domain_and_min_dist, kernel_mask_min_dist, N_cells_in_forbidden_radius,
               n_I, N_ITER, mu_I=0.3, mu_F=0.5, p_M=0.5, n_E=0.05, max_iter_cross_over=1000): 
    """
    

    Parameters
    ----------
    X_V : N_wt x N_pop array
        Input Population x-coordinates. Every column contains an individual
    Y_V : N_wt x N_pop array
        Input Population y-coordinates. Every column contains an individual    
    Err_1 :  N_pop x 1 array
        Cost of every individual
    grid : Grid class object
        contains information abou the discretized grid.    
    domain : Domain class object
        contains information about the domain.
    min_dist : float
        min. distance between any two turbines 
    mask_domain : numpy.array, dtype = bool
        masks the valid grid cells inside the domain
    mask_domain_and_min_dist : numpy.array, dtype = bool
        masks the valid grid cells, respecting the domain and min. dist. between turbines
    kernel_mask_min_dist : numpy.array, dtype = bool
        masks the allowed grid cells around any individual turbine (excludes forbidden radius)
    N_cells_in_forbidden_radius : int
        number of forbidden grid cells around each individual turbine

    n_I : int 
        Number of current iteration
    N_ITER : int 
        Number of iterations that will run    
    mu_I : float (default: 0.3, i.e. 30%)
        Initial portion of the population subject to Mutation
    mu_F : float (default: 0.5, i.e. 50%)
        Final portion of the population subject to Mutation
    p_M : float (default: 0.5, i.e. 50%)
        Portions of the Chromosomes subject to Mutations    
    n_E : float (default: 0.05, i.e. 5%)
        Portion of the population subject to Elitism. 
        This excludes the mutations!        
    max_iter_cross_over : int, optional
        max. iterations during the cross-over phase.
        It might be the case, that no valid position can be found during cross-over.
        In that case, the loop breaks after this number of iterations.
        The default is 1000.

    Returns
    -------
    X_V_n : N_wt x N_pop array
        Output Population x-coordinates. Every column contains an individual
    Y_V_n : N_wt x N_pop array
        Output Population y-coordinates. Every column contains an individual

    """
    # Optional: Introduce an update bar
    progress(n_I,N_ITER)    
    print("\n Best:  %s Mean %s" % (np.min(Err_1), np.mean(Err_1)))
    
    ### Sort the Population and bring forward elitism and mutations
    ## NOTE that the actual number of coordinates per population is 2*N_wt (x and y coords)
    N_wt,n_p=X_V.shape; # Number of features and Pop size
    
    index=Err_1.argsort(axis=0) # Sorted index 
    # Mutated Elements
    alpha=1/N_ITER*np.log(mu_F/mu_I) # Exp Coefficient
    Mut=mu_I*np.exp(alpha*n_I) # Number of mutate elements (float)
    N_M=int(np.round(Mut*n_p)) # Number of mutate elements (int)
    
    N_E=int((n_p-N_M)*n_E) # Number of Elite Elements
    N_C=int(n_p-N_M-N_E) # Number of Cross-over Elements
    
    # print('N_E, N_M, N_C')
    # print(N_E, N_M, N_C)
    
    print(" Elites:%s Mutated:%s Cross-over:%s" % (N_E, N_M, N_C))
    ### Perform Genetic Operations
    # 1. Elitism -------------------------------------------------------------------
    """ the copied elites are automatically inside the allowed domain """
    X_V_E=X_V[:,index[0:N_E,0]]
    Y_V_E=Y_V[:,index[0:N_E,0]]
    # 2. Mutations -----------------------------------------------------------------
    # Number of chromosomes that will mutate
    P_M=int(p_M*N_wt)
    # We mutate over the best n_M Individuals 
    # Take randomly the chromosomes that will mutate
    X_V_M=np.zeros((N_wt,N_M))
    Y_V_M=np.zeros((N_wt,N_M))
    for m in range(N_M):
        X_V_M[:,m]=X_V[:,index[m,0]] # Take the Best N_M
        Y_V_M[:,m]=Y_V[:,index[m,0]] # Take the Best N_M
        print('Mutation ' +str(m))
        for mm in range(P_M):
            Ind_M = np.random.randint(0,N_wt)
            print('Change entry ' + str(Ind_M))
            """ draw here only from allowed domain  -> also these points will be inside domain"""
            mask_domain_and_min_dist[:,:] = mask_domain[:,:]
            # update allowed cell mask from the layout, excluding the turbine, which is mutating
            wt_ind_list = np.delete(np.arange(N_wt), Ind_M) 
            update_mask_domain_and_min_dist(X_V_M[:,m], Y_V_M[:,m],
                                            mask_domain, kernel_mask_min_dist,
                                            N_cells_in_forbidden_radius, grid,
                                            mask_domain_and_min_dist, wt_ind_list=wt_ind_list)
            # draw random point in the allowed domain
            if mask_domain_and_min_dist.sum() == 0:
                pass
            else:
                X_V_M[Ind_M,m], Y_V_M[Ind_M,m] = draw_rnd_pt_from_allowed_domain(grid, mask_domain_and_min_dist)
      
    # 3. Cross-Over ----------------------------------------------------------------
    X_V_C=np.zeros((N_wt, N_C))
    Y_V_C=np.zeros((N_wt, N_C))
    for k in range(0,N_C):
        repeat_SEL_draw = True
        while repeat_SEL_draw:
            SEL=np.random.triangular(0, 0, N_C, 2).astype(int)
            if SEL[0] != SEL[1]: repeat_SEL_draw = False
        # print('cross over number', k)
        # print('SEL =', SEL)
        # the two selected populations:
        X_V_1 = np.copy(X_V[:,index[SEL[0],0]])
        Y_V_1 = np.copy(Y_V[:,index[SEL[0],0]])
        X_V_2 = np.copy(X_V[:,index[SEL[1],0]])
        Y_V_2 = np.copy(Y_V[:,index[SEL[1],0]])
        """ if crossover fails more than max_iter_cross_over times:
            take the pop with the lowest cost as it is """
        
        err1_ = Err_1[index[SEL[0],0],0]
        err2_ = Err_1[index[SEL[1],0],0]

        continue_cross_over = True
        j = 0
        # for j in range(0,N_wt):
        while continue_cross_over:
            cross_over_success = False
            """ always cross with the closest turbine of the second population
            (otherwise respecting the min dist. will take very long time...) """
            #find closest turbine
            x_ = X_V_1[j]
            y_ = Y_V_1[j]
            
            dists = cdist( list(zip(X_V_2, Y_V_2)), [[x_, y_]] )
            ind_min_ = np.argmin(dists)
            
            # (x_, y_) will cross with (x2_, y2_)             
            x2_ = X_V_2[ind_min_]
            y2_ = Y_V_2[ind_min_]
            
            # now, try to find a new spot from crossing, which respects the allowed domain
            draw_cnt = 0
            if j == 0:
                while draw_cnt < max_iter_cross_over:
                    draw_cnt += 1
                    a = np.random.uniform(0,1,1)
                    x_new = a * x_  + (1-a) * x2_
                    y_new = a * y_  + (1-a) * y2_
                    
                    ind_x, ind_y = grid.get_cell_index(x_new, y_new)
                    
                    # check, if new point is inside the domain
                    if mask_domain[ind_x, ind_y]:
                        X_V_C[j,k] = x_new
                        Y_V_C[j,k] = y_new
                        draw_cnt = max_iter_cross_over
                        cross_over_success = True
                # after while loop closes, increase j by 1
                j += 1

                # check, if point is in the domain
                # if cross_over_success:
                #     grid.get_cell_index(x, y)

            else:
                while draw_cnt < max_iter_cross_over:
                    draw_cnt += 1
                    a = np.random.uniform(0,1,1)
                    x_new = a * x_  + (1-a) * x2_
                    y_new = a * y_  + (1-a) * y2_
                    update_mask_domain_and_min_dist(X_V_C[0:j,k], Y_V_C[0:j,k],
                                                    mask_domain, kernel_mask_min_dist,
                                                    N_cells_in_forbidden_radius, grid,
                                                    mask_domain_and_min_dist)
                    ind_x, ind_y = grid.get_cell_index(x_new, y_new)
                    
                    if mask_domain_and_min_dist[ind_x, ind_y]:
                        X_V_C[j,k] = x_new
                        Y_V_C[j,k] = y_new
                        draw_cnt = max_iter_cross_over
                        cross_over_success = True
                    ### second possibility: check domain and distance separately
                    # if mask_domain[ind_x, ind_y]:
                        # dists = cdist( list(zip(X_V_C[0:j,k], Y_V_C[0:j,k])),
                        #               [[x_new, y_new]] )
                        # if dists.min() >= min_dist:
                        # check, if new point is inside allowed cells (domain + min. dist)
                        
                # after while loop closes, increase j by 1
                j += 1
                # when reaching the last turbine, break the loop
                if j == N_wt:
                    continue_cross_over = False
            # if the cross over failed to produce a point in valid domain, break the loop
            if not cross_over_success: continue_cross_over = False
        
        if not cross_over_success:
            print('cross over number', k+1)
            print('cross over was not successful, replacing target with lower error pop')
            if err1_ < err2_:
                X_V_C[:,k] = X_V_1
                Y_V_C[:,k] = Y_V_1
            else:
                X_V_C[:,k] = X_V_2
                Y_V_C[:,k] = Y_V_2
    
    ### Final Concatenation
    X_V_n=np.concatenate([X_V_E, X_V_M, X_V_C], axis=1) 
    Y_V_n=np.concatenate([Y_V_E, Y_V_M, Y_V_C], axis=1) 

    return X_V_n, Y_V_n

def optimize_layout_GOA(x_init_all_pops, y_init_all_pops, fi_list,
                        boundaries, min_dist, N_grid_points_short_side,
                        wdir_bin_centers_power, wspeed_bin_centers_power,
                        power_fraction_wdir_wspeed_power, power_fraction_wdir_wspeed_opt,
                        N_PROCESSES_PARALLEL,
                        N_ITER, mu_I, mu_F, p_M, n_E, max_iter_cross_over
                        ): 
    """
    Parameters
    ----------
    x_init_all_pops : N_wt x N_pop array
        init. x-positions of the wind turbines for all pops
    y_init_all_pops : N_wt x N_pop array
        init. y-positions of the wind turbines    
    fi_list : list of floris.tool.FlorisInterface objects
        list with floris objects must be initialized beforehand
    boundaries : N_points x 2 array
        domain boundary corner points [ [x0, y0], [x1, y1], ... ]    
    min_dist : float
        min. distance between any two turbines 
    N_grid_points_short_side : int
        number of grid points covering the shorter side of the bounding box around the domain
    wdir_bin_centers_power : numpy.array
        wind direction bin centers used for init and final power calculation
    wspeed_bin_centers_power : numpy.array
        wind speed bin centers used for init and final power calculation
    power_fraction_wdir_wspeed_power : numpy.array
        normalized power fraction histogram used for init and final power calculation        
    power_fraction_wdir_wspeed_opt : numpy.array
        normalized power fraction histogram used for optimization
    N_PROCESSES_PARALLEL : int
        number of processes used during multiprocessing pool computation
    N_ITER : int 
        Number of iterations that will run    
    mu_I : float (default: 0.3, i.e. 30%)
        Initial portion of the population subject to Mutation
    mu_F : float (default: 0.5, i.e. 50%)
        Final portion of the population subject to Mutation
    p_M : float (default: 0.5, i.e. 50%)
        Portions of the Chromosomes subject to Mutations    
    n_E : float (default: 0.05, i.e. 5%)
        Portion of the population subject to Elitism. 
        This excludes the mutations!        
    max_iter_cross_over : int, optional
        max. iterations during the cross-over phase.
        It might be the case, that no valid position can be found during cross-over.
        In that case, the loop breaks after this number of iterations.
        The default is 1000.

    Returns
    -------
    X_S_init : array of N_wt elements
        x-coordinates of best init. layout.
    Y_S_init : array of N_wt elements
        y-coordinates of best init. layout.
    X_S : array of N_wt elements
        x-coordinates of best optimized. layout.
    Y_S : array of N_wt elements
        x-coordinates of best optimized. layout.
    X_U : array of N_wt elements
        Solution Uncertainty (std in each entry) for x-coordinates.
        NOTE that in the present form, this variable does not have much meaning,
        because the wind turbines points can be anywhere in the domain and are not ordered
    Y_U : array of N_wt elements
        Solution Uncertainty (std in each entry) for y-coordinates.
        NOTE that in the present form, this variable does not have much meaning,
        because the wind turbines points can be anywhere in the domain and are not ordered
    X_V : N_wt x N_pop ( entire Population)    
        x-coords.
    Y_V : N_wt x N_pop ( entire Population)    
        y-coords.
    ind_best_layout_list : list of integer
        gives the indices of the best layout for each iteration.
    Err_Best : array of N_ITER elements
        best err-value for each iteration.
    Err_Mean : array of N_ITER elements
        mean err-value for each iteration.
    farm_power_no_wake : float
        farm power without wake (this is a constant for the whole optimization)

    """
    
    N_wt, N_pop = x_init_all_pops.shape

    # N_wt x n_p array: Input Population x-coordinates. Every column contains an individual
    X_V = np.copy(x_init_all_pops)
    # N_wt x n_p array: Input Population y-coordinates. Every column contains an individual
    Y_V = np.copy(y_init_all_pops)
    
    domain = Domain(boundaries)
    grid = Grid(domain, min_dist, N_grid_points_short_side=300)

    mask_domain = gen_mask_domain(grid, domain)
    kernel_mask_min_dist = gen_kernel_mask_min_dist(grid, domain, min_dist)
    N_cells_in_forbidden_radius = int( len(kernel_mask_min_dist) / 2)
    mask_domain_and_min_dist =\
        gen_mask_domain_and_min_dist(x_init_all_pops[:,0], y_init_all_pops[:,0],
                                     mask_domain, kernel_mask_min_dist, 
                                     N_cells_in_forbidden_radius, grid)

    farm_power_no_wake = get_farm_power_no_wake(fi_list[0],
                                                wdir_bin_centers_power,
                                                wspeed_bin_centers_power,
                                                power_fraction_wdir_wspeed_power,
                                                N_wt)

    # Prepare Some Stats
    Err_Best=np.zeros((N_ITER,1))
    Err_Mean=np.zeros((N_ITER,1))    
    
    ind_best_layout_list = []
    
    freq_opt = power_fraction_wdir_wspeed_opt
    # for parallelization:
    # need one dir/speed 'frequency' (histogram/power_fraction) per population
    # they are the same for all pops
    freq_list = len(X_V[0]) * [freq_opt]
    
    # prepare for parallelization
    pool = mp.Pool(processes=N_PROCESSES_PARALLEL)
    
    print()
    print('Preparing the loop...')
    for iter_n in range(N_ITER): 
        # parallelized error fct is the farm power weighted by farm power without wake 
        # The Err_1 is always <= 1
        Err_1 = \
            -np.array(pool.starmap(get_farm_power_single, zip(*(fi_list, X_V.T, Y_V.T, freq_list))))[:,None] \
            / farm_power_no_wake
        Err_Best[iter_n]=np.min(Err_1)
        Err_Mean[iter_n]=np.mean(Err_1)         
        Index=Err_1.argmin()
        ind_best_layout_list.append(Index)
        
        if iter_n == 0:
            X_S_init = np.copy(X_V[:,Index])
            Y_S_init = np.copy(Y_V[:,Index])
        
        X_V, Y_V = Update_POP(X_V, Y_V, Err_1, grid, domain, min_dist,
                       mask_domain, mask_domain_and_min_dist, kernel_mask_min_dist, N_cells_in_forbidden_radius,
                       iter_n, N_ITER, mu_I, mu_F, p_M, n_E, max_iter_cross_over)
      
        # X_V=Update_POP(X_V,Err_1,X_Bounds,k,N_ITER,\
                     # mu_I=mu_I,mu_F=mu_F,p_M=p_M,n_E=n_E) 
            
        
    # Finally give the answer
    Index=Err_1.argmin()
    X_S=X_V[:,Index]
    Y_S=Y_V[:,Index]
    X_U=np.std(X_V,axis=1)
    Y_U=np.std(Y_V,axis=1)
    print()
    print('Optimization finished')
    return X_S_init, Y_S_init, X_S, Y_S, X_U, Y_U, X_V, Y_V, ind_best_layout_list, Err_Best, Err_Mean, farm_power_no_wake
    