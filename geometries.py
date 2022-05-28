"""
@author: bohrer
based on GA algorithm by mendez
"""

import numpy as np
from shapely.geometry.polygon import Polygon

class Domain:
    # initialize with arguments in paranthesis of __init__
    def __init__(self,
                 boundaries # (x, y) coordinates of the boundary points
                 ):
        self.boundaries = boundaries
        
        self.polygon = Polygon(boundaries)
        
        domain_limits = np.array([
            [boundaries[:,0].min(), boundaries[:,0].max()],
            [boundaries[:,1].min(), boundaries[:,1].max()]
            ])
        self.limits = domain_limits
        self.dimensions = np.array([
            domain_limits[0,1]-domain_limits[0,0],
            domain_limits[1,1]-domain_limits[1,0],
            ])

class Grid:
    # initialize with arguments in paranthesis of __init__
    def __init__(self,
                 domain, # domain class object
                 min_dist,
                 N_grid_points_short_side=100, # grid cells will be quadratic
                 ):
        # N_grid_points_short_side = 300
        
        domain_dimensions = domain.dimensions
        domain_limits = domain.limits
        
        if domain_dimensions[0] < domain_dimensions[1]:
            # scale grid spacing slightly, such that outermost grid points are included in domain
            grid_spacing = 0.9999 * (domain_limits[0,1] - domain_limits[0,0])\
                           / N_grid_points_short_side
        else:
            # scale grid spacing slightly, such that outermost grid points are included in domain
            grid_spacing = 0.9999 * (domain_limits[1,1] - domain_limits[1,0])\
                           / N_grid_points_short_side
        self.spacing = grid_spacing
        
        # add grid points surrounding the domain to avoid index problems later
        # add enough points to cover min_dist of turbines at the domain boundaries
        domain_grid_x_left_corner = np.arange(domain_limits[0,0] - min_dist*1.01,
                                              domain_limits[0,1] + min_dist*1.01 + grid_spacing,
                                              grid_spacing)
        domain_grid_y_bottom_corner = np.arange(domain_limits[1,0] - min_dist*1.01,
                                                domain_limits[1,1] + min_dist*1.01 + grid_spacing,
                                                grid_spacing)
    
        # to include the outermost grid points in domain
        domain_grid_x_left_corner += 1e-5*grid_spacing
        domain_grid_y_bottom_corner += 1e-5*grid_spacing
        
        self.Nc = np.array((len(domain_grid_x_left_corner), len(domain_grid_y_bottom_corner)))
        # cells are defined by indices [i,j]  the position is given by the borders + grid_spacing
        domain_grid_x_corners, domain_grid_y_corners =\
            np.meshgrid(domain_grid_x_left_corner, domain_grid_y_bottom_corner, indexing='ij')
        
        self.corners_x = domain_grid_x_corners
        self.corners_y = domain_grid_y_corners
        
        self.x_min = domain_grid_x_corners.min()
        self.y_min = domain_grid_y_corners.min()
        
    def get_cell_index(self, x, y):
        ind_x = ((x - self.x_min) / self.spacing).astype(int)
        ind_y = ((y - self.y_min) / self.spacing).astype(int)
        
        return ind_x, ind_y      