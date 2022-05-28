"""
@author: bohrer
based on GA algorithm by mendez
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from scipy.spatial.distance import cdist

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

#%%

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def rotate_around_origin_2D(xy, radians):
    """rotate a point around the origin (0, 0)."""
    x, y = xy
    xx = x * np.cos(radians) - y * np.sin(radians)
    yy = x * np.sin(radians) + y * np.cos(radians)

    return xx, yy

def project_onto_eigvec_2D(xy, n):
    """projects onto the given normalized vector n,
       such that n is the new x-axis"""
    x, y = xy
    xx = n[0] * x + n[1] * y
    yy = -n[1] * x + n[0] * y
    return np.array((xx, yy))

def draw_rnd_pt_from_allowed_domain(grid, mask_domain_and_min_dist):

    valid_index_pairs = np.argwhere(mask_domain_and_min_dist)
    N_valid_ind = len(valid_index_pairs)

    # 1. select a cell randomly
    cell_n = np.random.randint(0,N_valid_ind)
    cell_ind = valid_index_pairs[cell_n]
    
    # 2. inside the selected cell, get a random relative coordinate
    rnd_rel = np.random.rand(2)
    
    # 3. generate coordinates from the random parameters
    x_ = grid.x_min + (cell_ind[0] + rnd_rel[0]) * grid.spacing
    y_ = grid.y_min + (cell_ind[1] + rnd_rel[1]) * grid.spacing
    
    return x_, y_

def draw_random_position_in_domain(domain_dimensions, domain_limits, boundary_polygon):
    """
    boundary polygon is a shapely.geometry.polygon class object, containing the domain

    """    
    
    repeat_draw = True
    while repeat_draw:
        x_ = np.random.rand(1) * domain_dimensions[0] + domain_limits[0,0]
        y_ = np.random.rand(1) * domain_dimensions[1] + domain_limits[1,0]
        if boundary_polygon.contains(Point(x_, y_)):
            repeat_draw = False
    return x_[0], y_[0]

def progress(count, total, suffix=''):
    bar_len = 40
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()  # As suggested by Rom Ruben

def plot_power_curve(fi, file_out):
    fi.reinitialize(
        layout=[[0.0], [0.0]],
        wind_directions=[270],
        wind_speeds=np.arange(0,31,0.5)
        )
    
    layout_x = np.copy(fi.layout_x)
    layout_y = np.copy(fi.layout_y)
    
    fi.calculate_wake()
    
    farm_power_array = fi.get_farm_power()
    
    fi.reinitialize(
        layout=(layout_x, layout_y)
        )
    
    plt.figure(figsize=cm2inch(20,8))
    plt.plot(np.arange(0,31,0.5), farm_power_array[0]/1e6)
    plt.xticks(np.arange(0,31,1))
    plt.yticks(np.arange(0,16,1))
    plt.grid()
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Power (MW)')
    plt.savefig(file_out)
    plt.show()

def get_farm_power_no_wake(fi, wind_directions, wind_speeds, frequencies, N_wt):
    # calc power for single turbine
    
    layout_x = np.copy(fi.layout_x)
    layout_y = np.copy(fi.layout_y)
    
    fi.reinitialize(
        layout=[[0.0], [0.0]],
        wind_directions=wind_directions,
        wind_speeds=wind_speeds
        )

    fi.calculate_wake()
    
    farm_power_array = fi.get_farm_power()
    farm_power_per_turbine_no_wake = np.sum(farm_power_array * frequencies)
    farm_power_no_wake = farm_power_per_turbine_no_wake * N_wt
    
    fi.reinitialize(
        layout=(layout_x, layout_y)
        )
    
    return farm_power_no_wake

def reinit(fi, x, y):
    fi.reinitialize(layout=(x, y))

# power calc for single config, usd in the parallelization below
def get_farm_power_single(fi, layout_x, layout_y, freq_opt):
    fi.reinitialize(layout=(layout_x, layout_y))
    fi.calculate_wake()
    return np.sum(fi.get_farm_power() * freq_opt)

def gen_mask_domain(grid, domain):
    
    allowed_corner_ind_domain = np.zeros((grid.Nc[0], grid.Nc[1]), dtype=bool)
    
    for i in range(grid.Nc[0]):
        for j in range(grid.Nc[1]):
            point = Point(grid.corners_x[i,j], grid.corners_y[i,j])
            if domain.polygon.contains(point):
                allowed_corner_ind_domain[i,j] = True
    
    # get allowed cells, which are fully included in domain (all four corners of the cell)
    mask_domain = np.zeros_like(allowed_corner_ind_domain)
    for i in range(1, grid.Nc[0]-1):
        for j in range(1, grid.Nc[1]-1):
            # if all four corners of a cell are included in the domain -> set cell valid
            if (allowed_corner_ind_domain[i:i+2,j:j+2].sum() == 4):
                mask_domain[i,j] = True
    return mask_domain

def gen_kernel_mask_min_dist(grid, domain, min_dist):
    x_ = domain.dimensions[0]/2
    y_ = domain.dimensions[1]/2
    ind_x = ( (x_ - grid.x_min) / grid.spacing).astype(int) 
    ind_y = ( (y_ - grid.y_min) / grid.spacing).astype(int)
    
    N_cells_in_forbidden_radius = np.ceil(min_dist / grid.spacing).astype(int)
    
    kernel_mask_min_dist = np.zeros((2*N_cells_in_forbidden_radius+1,
                                         2*N_cells_in_forbidden_radius+1), dtype=bool)
    
    x_corners_base = grid.corners_x[ind_x:ind_x+2, ind_y:ind_y+2].flatten()
    y_corners_base = grid.corners_y[ind_x:ind_x+2, ind_y:ind_y+2].flatten()
    
    for cnt_i,i in enumerate(range(ind_x-N_cells_in_forbidden_radius,
                                   ind_x+N_cells_in_forbidden_radius+1)):
        for cnt_j,j in enumerate(range(ind_y-N_cells_in_forbidden_radius,
                                       ind_y+N_cells_in_forbidden_radius+1)):
            # check if all corner points of the cells surrounding our cell are 
            x_corners_tg = grid.corners_x[i:i+2,j:j+2].flatten()
            y_corners_tg = grid.corners_y[i:i+2,j:j+2].flatten()
            if cdist(list(zip(x_corners_base, y_corners_base)),
                     list(zip(x_corners_tg, y_corners_tg))).min() < min_dist:
                pass
            else: kernel_mask_min_dist[cnt_i, cnt_j] = True
    return kernel_mask_min_dist

def gen_mask_domain_and_min_dist(layout_x, layout_y,
                                 mask_domain, kernel_mask_min_dist, 
                                 N_cells_in_forbidden_radius, grid):
    mask_domain_and_min_dist = np.copy(mask_domain)
    
    N_wt = len(layout_x)
    
    # get cell indices of all turbines
    ind_x, ind_y = grid.get_cell_index(layout_x, layout_y)
    
    # remove allowed cells in vicinity of present wind turbines, according to the mapping kernel
    for wt_n in range(N_wt):
        i = ind_x[wt_n]
        j = ind_y[wt_n]
        mask_domain_and_min_dist[ 
            i-N_cells_in_forbidden_radius:i+N_cells_in_forbidden_radius+1,
            j-N_cells_in_forbidden_radius:j+N_cells_in_forbidden_radius+1] =\
        np.logical_and(mask_domain_and_min_dist[ 
                       i-N_cells_in_forbidden_radius:i+N_cells_in_forbidden_radius+1,
                       j-N_cells_in_forbidden_radius:j+N_cells_in_forbidden_radius+1],
                       kernel_mask_min_dist)
    return mask_domain_and_min_dist

def update_mask_domain_and_min_dist(layout_x, layout_y,
                                    mask_domain, kernel_mask_min_dist,
                                    N_cells_in_forbidden_radius, grid,
                                    mask_domain_and_min_dist,
                                    wt_ind_list=None):
    """ layout_x and layout_y hold the x,y coordinates of all wind turbines """
    """ only the wind turbines from wt_ind_list are included in the mask """
    
    if wt_ind_list is None: wt_ind_list=np.arange(len(layout_x))
    
    mask_domain_and_min_dist[:,:] = mask_domain[:,:]
    
    # get cell indices of all turbines
    ind_x, ind_y = grid.get_cell_index(layout_x, layout_y)
    
    # remove allowed cells in vicinity of present wind turbines, according to the mapping kernel
    for wt_n in wt_ind_list:
        i = ind_x[wt_n]
        j = ind_y[wt_n]
        
        mask_domain_and_min_dist[ 
            i-N_cells_in_forbidden_radius:i+N_cells_in_forbidden_radius+1,
            j-N_cells_in_forbidden_radius:j+N_cells_in_forbidden_radius+1] =\
        np.logical_and(mask_domain_and_min_dist[ 
                       i-N_cells_in_forbidden_radius:i+N_cells_in_forbidden_radius+1,
                       j-N_cells_in_forbidden_radius:j+N_cells_in_forbidden_radius+1],
                       kernel_mask_min_dist)

#%% snippets

# if act_plotting:
#     # plot boundaries init
#     figsize = cm2inch(7,7)
#     fig, ax = plt.subplots(figsize=figsize)
#     # XMIN = 0
#     # XMAX = 40
#     # YMIN = 0
#     # YMAX = 40
#     # LW = 2
#     ax.fill(*boundaries.T, fill = False, edgecolor='grey')
#     ax.plot(*boundaries.T, 'kx')
#     ax.plot(x_init, y_init, 'k.')
#     # ax.plot( *zip( (0,0), -7*eigvec0 ) )
#     # for i, elt in enumerate(boundaries * scale_m2km):
#     #     ax.annotate(str(i), elt, xytext=(4,0), textcoords='offset points')
#     ax.axis('equal')    
#     ax.set_xlabel('x/min_dist')
#     ax.set_ylabel('y/min_dist')
#     fig.tight_layout()

#     name_fig = 'domain_init.pdf'
#     fig.savefig(path_fig + os.sep + name_fig)
    
#     plt.show()
     

# if init_successful:
#     for wt_n in range(N_wt):
#         point = Point(x_init[wt_n], y_init[wt_n])
#         if boundary_polygon.contains(point):
#             print(wt_n, 'yea') 

# generate a pool of random numbers in the domain

# N_max = N_iter_init_main_loop * (N_wt-1) * N_iter_init_sub_loop + 1
# rand_x = np.zeros(N_max)
# rand_y = np.zeros(N_max)
# cnt = 0
# while cnt < N_max: 
#     x_ = np.random.rand(1) * domain_dimensions[0] + domain_limits[0,0]
#     y_ = np.random.rand(1) * domain_dimensions[1] + domain_limits[1,0]
#     if boundary_polygon.contains(Point(x_, y_)):
#         rand_x[cnt] = x_
#         rand_y[cnt] = y_
#         cnt += 1

#%% SNIPPETS
# MEW = 0.1
# MS = 0.3
# fig, ax = plt.subplots(figsize=cm2inch(10,6))
# ax.plot(grid.corners_x.flatten() + 0.5 * grid.spacing,
#         grid.corners_y.flatten() + 0.5 * grid.spacing,
#         's', c='grey', fillstyle='none', mew=MEW, ms=MS, label='full grid')
# # ax.plot(grid.corners_x[allowed_corner_ind_domain],
# #         grid.corners_y[allowed_corner_ind_domain], '.', c='blue', ms=MS)
# ax.plot(grid.corners_x[mask_domain].flatten() + 0.5 * grid.spacing,
#         grid.corners_y[mask_domain].flatten() + 0.5 * grid.spacing,
#         's', c='blue', fillstyle='none', mew=MEW, ms=MS, label='allowed grid')
# ax.fill(*boundaries.T, fill = False, edgecolor='black', zorder=99, lw=0.5, label='domain')
# ax.axis('equal')
# ax.set_xlabel('x [m]')
# ax.set_ylabel('y [m]')

# lgnd = ax.legend(loc='center left', bbox_to_anchor=((1.0, 0.5)), markerscale=6)
# fig.tight_layout()

# name_fig = 'domain_gridded_fine.png'
# # fig.savefig(path_fig + os.sep + name_fig, dpi=600)

# plt.show()

###
# N_cells_in_forbidden_radius = int( len(kernel_mask_min_dist) / 2)
# x_ = np.arange(-N_cells_in_forbidden_radius, N_cells_in_forbidden_radius+1, 1.0)
# y_ = np.arange(-N_cells_in_forbidden_radius, N_cells_in_forbidden_radius+1, 1.0)
# vis_mapping_kernel_x, vis_mapping_kernel_y = np.meshgrid(x_, y_, indexing='ij')

# vis_mapping_kernel_x *= grid.spacing
# vis_mapping_kernel_y *= grid.spacing

# MS = 1.1
# MEW = 0.3
# fig, ax = plt.subplots(figsize=cm2inch(10,6))
# ax.plot(vis_mapping_kernel_x.flatten() + 0.5 * grid.spacing,
#         vis_mapping_kernel_y.flatten() + 0.5 * grid.spacing,
#         's', c='grey', ms=MS, fillstyle='none', mew=MEW, label='exclude')
# ax.plot(vis_mapping_kernel_x[kernel_mask_min_dist].flatten() + 0.5 * grid.spacing,
#         vis_mapping_kernel_y[kernel_mask_min_dist].flatten() + 0.5 * grid.spacing,
#         's', c='blue', ms=MS, fillstyle='none', mew=MEW, label='include')
# circle1 = plt.Circle((0.5 * grid.spacing, 0.5 * grid.spacing), min_dist + 0.5*grid.spacing,
#                      color='orange', fill=False, label='min dist', zorder=99,lw=1)
# ax.plot( 0.5 * grid.spacing, 0.5 * grid.spacing, 'ko', label='WT', ms=1)
# ax.add_patch(circle1)
# lgnd = ax.legend(loc='center left', bbox_to_anchor=((1.0, 0.5)), markerscale=4)
# ax.axis('equal')
# ax.set_xlabel('x [m]')
# ax.set_ylabel('y [m]')
# fig.tight_layout()
# plt.show()
# name_fig = 'kernel_mask_min_dist.png'
# # fig.savefig(path_fig + os.sep + name_fig, dpi=600)

###
# MS = 0.5
# fig, ax = plt.subplots(figsize=cm2inch(12,8))
# ax.plot(grid.corners_x.flatten(), grid.corners_y.flatten(),
#         '.', c='lightgrey', ms=MS, label='forbidden by domain')
# ax.plot(grid.corners_x[mask_domain].flatten(),
#         grid.corners_y[mask_domain].flatten(),
#         '.', c='dimgrey', ms=MS, label='forbidden by min dist')
# ax.plot(grid.corners_x[mask_domain_and_min_dist].flatten(),
#         grid.corners_y[mask_domain_and_min_dist].flatten(),
#         '.', c='blue', ms=MS, label='allowed cells')
# ax.fill(*boundaries.T, fill = False, edgecolor='orange', zorder=99, lw=0.5)
# ax.plot(layout_x, layout_y, 'ko', ms=3)
# ax.axis('equal')
# ax.set_xlabel('x [m]')
# ax.set_ylabel('y [m]')
# lgnd = ax.legend(loc='center left', bbox_to_anchor=((1.0, 0.5)), markerscale=10)
# fig.tight_layout()
# name_fig = 'forbidden_cells_domain_with_WT.png'
# # fig.savefig(path_fig + os.sep + name_fig, dpi=600)
# plt.show()

###
# list_x = []
# list_y = []
# for cnt in range(100):
    
#     x_, y_ = draw_rnd_pt_from_allowed_domain(grid, mask_domain_and_min_dist)

#     list_x.append(x_)
#     list_y.append(y_)

# MS = 0.5
# fig, ax = plt.subplots(figsize=cm2inch(12,8))
# ax.plot(grid.corners_x.flatten(), grid.corners_y.flatten(),
#         '.', c='lightgrey', ms=MS, label='forbidden by domain')
# ax.plot(grid.corners_x[mask_domain].flatten(),
#         grid.corners_y[mask_domain].flatten(),
#         '.', c='dimgrey', ms=MS, label='forbidden by min dist')
# ax.plot(grid.corners_x[mask_domain_and_min_dist].flatten(),
#         grid.corners_y[mask_domain_and_min_dist].flatten(),
#         '.', c='blue', ms=MS, label='allowed cells')
# ax.plot(list_x, list_y, 'o', c='red', label='random dots', ms=1)
# ax.fill(*boundaries.T, fill = False, edgecolor='orange', zorder=99, lw=0.5)
# ax.plot(layout_x, layout_y, 'ko', ms=3)
# ax.axis('equal')
# ax.set_xlabel('x [m]')
# ax.set_ylabel('y [m]')
# lgnd = ax.legend(loc='center left', bbox_to_anchor=((1.0, 0.5)), markerscale=10)
# lgnd.legendHandles[-1].set(markersize=2)
# fig.tight_layout()
# name_fig = 'random_allowed_dots.png'
# # fig.savefig(path_fig + os.sep + name_fig, dpi=600)
# plt.show()

#%% TEST update mask

# mask_domain_and_min_dist_ = np.copy(mask_domain_and_min_dist)

# wt_ind_list = np.array([0,1,2,3,4,5,6])

# update_mask_domain_and_min_dist(layout_x, layout_y,
#                                 mask_domain, kernel_mask_min_dist,
#                                 N_cells_in_forbidden_radius, grid,
#                                 mask_domain_and_min_dist_, wt_ind_list=wt_ind_list)

# MS = 0.5
# fig, ax = plt.subplots(figsize=cm2inch(12,8))
# ax.plot(grid.corners_x.flatten(), grid.corners_y.flatten(),
#         '.', c='lightgrey', ms=MS, label='forbidden by domain')
# ax.plot(grid.corners_x[mask_domain].flatten(),
#         grid.corners_y[mask_domain].flatten(),
#         '.', c='dimgrey', ms=MS, label='forbidden by min dist')
# ax.plot(grid.corners_x[mask_domain_and_min_dist_].flatten(),
#         grid.corners_y[mask_domain_and_min_dist_].flatten(),
#         '.', c='blue', ms=MS, label='allowed cells')
# # ax.plot(list_x, list_y, 'o', c='red', label='random dots', ms=1)
# ax.fill(*boundaries.T, fill = False, edgecolor='orange', zorder=99, lw=0.5)
# ax.plot(layout_x[wt_ind_list], layout_y[wt_ind_list], 'ko', ms=3)
# ax.axis('equal')
# ax.set_xlabel('x [m]')
# ax.set_ylabel('y [m]')
# lgnd = ax.legend(loc='center left', bbox_to_anchor=((1.0, 0.5)), markerscale=10)
# lgnd.legendHandles[-1].set(markersize=2)
# fig.tight_layout()
# name_fig = 'allowed_domain_after_update.png'
# fig.savefig(path_fig + os.sep + name_fig, dpi=600)
# plt.show()


#%% test if a random point is in any allowed cell

# # test for random numbers, if they fall in allowed domain cells
# for cnt in range(100):
#     x_ = np.random.uniform(domain_limits[0,0], domain_limits[0,1])
#     y_ = np.random.uniform(domain_limits[1,0], domain_limits[1,1])
    
#     ind_x, ind_y = get_cell_index(x_, y_, grid.x_min, grid.y_min, grid_spacing)
    
#     print(x_,y_)
#     print(mask_domain_and_min_dist[ind_x, ind_y])

#%% MORE PLOTTING SNIPPETS
# MS = 0.5
# fig, ax = plt.subplots(figsize=cm2inch(10,10))
# # ax.plot(grid.corners_x.flatten(), grid.corners_y.flatten(),
# #         '.', c='lightgrey', ms=MS, label='forbidden by domain')
# # ax.plot(grid.corners_x[mask_domain].flatten(),
# #         grid.corners_y[mask_domain].flatten(),
# #         '.', c='dimgrey', ms=MS, label='forbidden by min dist')
# # ax.plot(grid.corners_x[mask_domain_and_min_dist].flatten(),
# #         grid.corners_y[mask_domain_and_min_dist].flatten(),
# #         '.', c='blue', ms=MS, label='allowed cells')
# ax.fill(*boundaries.T, fill = False, edgecolor='orange', zorder=99, lw=1)
# ax.plot(X_V[:,0], Y_V[:,0], 'ko', ms=3, label='WT before update')

# for cnt in range(N_wt):
#     x_ = X_V[cnt,0]
#     y_ = Y_V[cnt,0]
#     circle1 = plt.Circle((x_, y_), min_dist,
#                          color='grey', fill=False, zorder=99, lw=1)
#     ax.add_patch(circle1)

# ax.axis('equal')
# ax.set_xlabel('x [m]')
# ax.set_ylabel('y [m]')
# lgnd = ax.legend(loc='center', bbox_to_anchor=((0.5, 1.05)), markerscale=1)
# fig.tight_layout()
# name_fig = 'test_before_update_Mut.png'
# # fig.savefig(path_fig + os.sep + name_fig, dpi=600)
# plt.show()



# MS = 0.5
# fig, ax = plt.subplots(figsize=cm2inch(10,10))
# # ax.plot(grid.corners_x.flatten(), grid.corners_y.flatten(),
# #         '.', c='lightgrey', ms=MS, label='forbidden by domain')
# # ax.plot(grid.corners_x[mask_domain].flatten(),
# #         grid.corners_y[mask_domain].flatten(),
# #         '.', c='dimgrey', ms=MS, label='forbidden by min dist')
# # ax.plot(grid.corners_x[mask_domain_and_min_dist].flatten(),
# #         grid.corners_y[mask_domain_and_min_dist].flatten(),
# #         '.', c='blue', ms=MS, label='allowed cells')
# ax.fill(*boundaries.T, fill = False, edgecolor='orange', zorder=99, lw=1)
# ax.plot(X_V_n[:,0], Y_V_n[:,0], 'ro', ms=3, label='WT after update')

# for cnt in range(N_wt):
#     x_ = X_V_n[cnt,0]
#     y_ = Y_V_n[cnt,0]
#     circle1 = plt.Circle((x_, y_), min_dist,
#                          color='salmon', fill=False, zorder=99, lw=1)
#     ax.add_patch(circle1)

# ax.axis('equal')
# ax.set_xlabel('x [m]')
# ax.set_ylabel('y [m]')
# lgnd = ax.legend(loc='center', bbox_to_anchor=((0.5, 1.05)), markerscale=1)
# fig.tight_layout()
# name_fig = 'test_after_update_Mut.png'
# # fig.savefig(path_fig + os.sep + name_fig, dpi=600)
# plt.show()


# MS = 0.5
# fig, ax = plt.subplots(figsize=cm2inch(10,10))
# # ax.plot(grid.corners_x.flatten(), grid.corners_y.flatten(),
# #         '.', c='lightgrey', ms=MS, label='forbidden by domain')
# # ax.plot(grid.corners_x[mask_domain].flatten(),
# #         grid.corners_y[mask_domain].flatten(),
# #         '.', c='dimgrey', ms=MS, label='forbidden by min dist')
# # ax.plot(grid.corners_x[mask_domain_and_min_dist].flatten(),
# #         grid.corners_y[mask_domain_and_min_dist].flatten(),
# #         '.', c='blue', ms=MS, label='allowed cells')
# ax.fill(*boundaries.T, fill = False, edgecolor='orange', zorder=99, lw=1)
# ax.plot(X_V[:,0], Y_V[:,0], 'ko', ms=3, label='WT before update')
# ax.plot(X_V_n[:,0], Y_V_n[:,0], 'ro', ms=3, label='WT after update')

# # for cnt in range(N_wt):
# #     x_ = X_V[cnt,0]
# #     y_ = Y_V[cnt,0]
# #     circle1 = plt.Circle((x_, y_), min_dist,
# #                          color='grey', fill=False, zorder=99, lw=1)
# #     ax.add_patch(circle1)

# ax.axis('equal')
# ax.set_xlabel('x [m]')
# ax.set_ylabel('y [m]')
# lgnd = ax.legend(loc='center', bbox_to_anchor=((0.5, 1.05)), markerscale=1)
# fig.tight_layout()
# name_fig = 'test_before_and_after_update_Mut.png'
# # fig.savefig(path_fig + os.sep + name_fig, dpi=600)
# plt.show()

#%% draw random point from allowed domain and check if it is too close to any other point from the populations

# pop_n = 0


# for cnt in range(100):

#     x_, y_ = draw_random_position_in_domain(domain_dimensions,
#                                             domain_limits,
#                                             boundary_polygon)
#     print(cnt, x_, y_)
#     dist_to_point_arr = cdist( list(zip(x_init, y_init)), [[x_, y_]])
#     dist_to_point_min = np.min(dist_to_point_arr)
#     print(dist_to_point_min)
#     # if np.sum(cdist( list(zip(x_init, y_init)), [[x_, y_]]  ) < min_dist) == 0:
#     if dist_to_point_min < min_dist:
#         print('bad')
#     else: print('clear')
    
