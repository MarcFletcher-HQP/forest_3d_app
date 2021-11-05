# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 02:12:40 2021

@author: fletchm


"""

import numpy as np
import warnings
from itertools import chain
from collections import deque


class IndexedPCD:

    def __init__(self, np_arr, gridSize=[1, 1], method = "Grid"):

        if(len(np_arr.shape) != 2):
            raise ValueError("input array should have dimension 2")

        self.xmin, self.ymin = np.min(np_arr, axis = 0)[:2]
        self.xmax, self.ymax = np.max(np_arr, axis = 0)[:2]

        if method == "Grid":
            self.index = Grid(np_arr, gridSize)    
        elif method == "QuadTree":
            self.index = QuadTree.from_np_array(np_arr, gridSize)
        else:
            raise ValueError("Invalid method: %s" % str(method))
        
        self.data = np_arr
    
    
    def __len__(self):
        return len(self.data)
    
    
    def points_index_in_box(self, xmin, ymin, xmax, ymax):
        box = AOI(xmin, ymin, xmax, ymax)
        return self.points_in_aoi_idx(box)
    
    def points_in_box(self, xmin, ymin, xmax, ymax):
        box = AOI(xmin, ymin, xmax, ymax)
        return self.points_in_aoi(box)
    
    def points_in_aoi(self, aoi):
        point_idx = self.points_in_aoi_idx(aoi)
        return self.data[point_idx, :]
    
    
    def points_in_aoi_idx(self, aoi):
        
        # If the AOI is much larger than the grid/leaf-node size then it's 
        # beneficial (although not by much) to only check the points lying in
        # nodes that intersect the boundary and skip checking all points in 
        # internal nodes. For smaller AOI's it's just wasted time.
        if aoi.area() > (42 * self.index.block_area()):
            idx_contained = np.array(self.index.aoi_contains_proper_nodes(aoi), dtype = np.uint32)
            idx_boundary = np.array(self.index.aoi_boundary_nodes(aoi), dtype = np.uint32)
            
            if len(idx_boundary) > 0:
                data = self.data[idx_boundary, :]
                point_idx = np.where(aoi.contains_proper_point(data[:,0], data[:,1]))[0]
                if(len(point_idx) > 0):
                    idx_boundary = idx_boundary[point_idx.astype(int)]
        else:
            idx_contained = np.array([], dtype = np.uint32)
            idx_boundary = np.array(list(self.index.points_in_aoi(aoi)), dtype = np.uint32)
        
        
        if len(idx_boundary) > 0:
            data = self.data[idx_boundary, :]
            point_idx = np.where(aoi.contains_proper_point(data[:,0], data[:,1]))[0]
            if(len(point_idx) > 0):
                idx_boundary = idx_boundary[point_idx.astype(int)]
            
        return np.concatenate((idx_contained, idx_boundary))

    





# Implementation of quad-tree, used specifically to speed up area of interest
# queries in ALS point-clouds. Rather than splitting nodes based on the 
# number of points, this version splits until a minimum spatial extent is 
# reached.
class QuadTree:
  
    # Initialisers
    def __init__(self, extent, minSize = [1, 1]):
        
        self.min_width, self.min_height = minSize
        self.xmin, self.ymin, self.xmax, self.ymax = AOI.coords(extent)
        self.extent = extent
        self.has_children = False
        self.minimum_extent = False
        self.index = deque([])
    
    
    @classmethod
    def from_np_array(cls, np_arr, minSize = [1,1]):
        if np_arr.ndim != 2:
          raise ValueError("np_arr.ndim must equal 2")
        
        if np_arr.shape[1] < 2 or np_arr.shape[1] > 3:
          raise ValueError("np_arr must have at least 2 columns and no more than 3.")
        
        xmin, ymin = np.min(np_arr, axis = 0)[:2]
        xmax, ymax = np.max(np_arr, axis = 0)[:2]
        aoi = AOI(xmin, ymin, xmax, ymax)
        
        tree = cls(aoi, minSize)
        
        idx = np.array(range(len(np_arr)))
        tree.bulk_insert(np_arr[:,0], np_arr[:,1], idx)
        
        return tree
    
    
    def block_area(self):
        
        # technically a lower-bound, would prefer to avoid searching the whole
        # tree for the smallest leaf-node.
        return self.min_width * self.min_height
    
    
    def points_in_aoi(self, aoi):
    
        found_points = deque([])
    
        if not self.extent.intersects(aoi):
          return found_points
      
        
        if aoi.contains(self.extent) and self.has_children:
            
            for child in self.children():
                found_points += child.get_points()
        
        elif self.has_children:
            
            for child in self.children():
                found_points += child.points_in_aoi(aoi)
            
        else:
            found_points += self.index

        return found_points
    
    
    def aoi_contains_proper_nodes(self, aoi):
        
        found_points = deque([])
        
        if not self.extent.intersects(aoi):
          return found_points
      
        if aoi.contains_proper(self.extent):
            
            if self.has_children:
                for child in self.children():
                    found_points += child.get_points()
            else:
                found_points += self.index
        
        elif self.has_children:
            
            for child in self.children():
                found_points += child.aoi_contains_proper_nodes(aoi)

        return found_points
    
    
    def aoi_boundary_nodes(self, aoi):
        
        found_points = deque([])
        
        if not aoi.contains_proper(self.extent):
           
            if self.has_children:
                for child in self.children():
                    if child.extent.intersects(aoi):
                        found_points += child.aoi_boundary_nodes(aoi)
            else:
                found_points += self.index

        return found_points



    ## Internal Methods ##
    
    def children(self):
        return self.sw, self.nw, self.ne, self.se
  
    
    def get_points(self):
        
        found_points = deque([])
        
        if self.has_children:
            for child in self.children():
                found_points += child.get_points()
        else:
            found_points += self.index
        
        return found_points
    
    
    def can_split(self):
    
        if(not self.has_children):
          width = self.extent.width()/2
          height = self.extent.height()/2
          self.minimum_extent = width <= self.min_width or height <= self.min_height
          return (not self.has_children and not self.minimum_extent)
    
    
    def split(self):
    
        # Get centre of the bounding box
        sw, nw, ne, se = self.extent.quadrants()
      
        # Create the 4 children; label them by the cardinal direction of the childs
        # centre, relative to the parents centre.
        self.sw = QuadTree(sw, [self.min_width, self.min_height])
        self.nw = QuadTree(nw, [self.min_width, self.min_height])
        self.ne = QuadTree(ne, [self.min_width, self.min_height])
        self.se = QuadTree(se, [self.min_width, self.min_height])
      
        self.has_children = True
      
        return None
        
        
    def child_containing(self, x, y):
    
        child = None
      
        if self.has_children:
            for child in self.children():
                if child.extent.contains_point(x, y):
                    continue
      
        return child
    
    
    def insert(self, x, y, i):
    
        if(not self.extent.contains_point(x, y)):
          return False
      
        if self.can_split():
          self.split()
      
        if self.has_children:
          child = self.child_containing(x, y)
          child.insert(x, y, i)
        else:
            self.index.append(int(i))
      
        return True
    
    
    def bulk_insert(self, x, y, i):
        
        if self.can_split():
          self.split()
      
        if self.has_children:
          for child in self.children():
              idx = np.where(child.extent.contains_point(x, y))[0]
              idx = idx.astype(int)
              if len(idx) > 0:
                  child.bulk_insert(x[idx], y[idx], i[idx])
        else:
            [self.index.append(pos) for pos in i]
        
        return True
        
        
        





class Grid:
    
    def __init__(self, np_arr, gridSize = [1, 1]):
        
        self.xmin, self.ymin = np.floor(np.min(np_arr, axis = 0))[:2]
        self.xmax, self.ymax = np.ceil(np.max(np_arr, axis = 0))[:2]
        
        self.res_x, self.res_y = gridSize
        
        self.steps_x = (self.xmax // self.res_x) - (self.xmin // self.res_x)
        self.steps_y = (self.ymax // self.res_y) - (self.ymin // self.res_y)

        grid_idx = self.point_in_cell(np_arr[:,0], np_arr[:,1])

        pt_order = grid_idx.argsort()
        grid_idx.sort()

        self.cells = np.unique(grid_idx).astype(int)       
        self.index = {i: pt_order[grid_idx == i] for i in self.cells}

    
    ## Public facing methods ##
    
    def block_area(self):
        return self.res_x * self.res_y
    
    def points_in_aoi(self, aoi):
        
        cells_list = self.cells_in_aoi(aoi)
        
        index = np.array([])
        if(len(cells_list) > 0):
            index = np.concatenate([self.index[cell] for cell in cells_list])
        
        return index.astype(int)
    
    
    def aoi_contains_proper_nodes(self, aoi):
        
        cells_list = list(set(self.cells_in_aoi(aoi)) - set(self.cells_on_boundary(aoi)))
        
        index = np.array([])
        if(len(cells_list) > 0):
            index = np.concatenate([self.index[cell] for cell in cells_list])
        
        return index.astype(int)
    
    
    def aoi_boundary_nodes(self, aoi):
        
        cells_list = self.cells_on_boundary(aoi)
        
        index = np.array([])
        if(len(cells_list) > 0):
            index = np.concatenate([self.index[cell] for cell in cells_list])
        
        return index.astype(int)
    
    
    
    ## Internal Methods ##
    
    def point_in_cell(self, x, y):
        
        pt_idx = (x - self.xmin) // self.res_x +        \
            (self.steps_x + 1) * ((y - self.ymin) // self.res_y)
        
        return pt_idx
    
    
    def cells_in_aoi(self, aoi):
        
        idx_first = self.point_in_cell(aoi.xmin, aoi.ymin)

        if (idx_first < 0 and ((round(aoi.xmin, 0) < self.xmin) or (round(aoi.ymin) < self.ymin))):
            warnings.warn("Vertex of AOI is outside grid extent: (%.4f, %.4f)" % (aoi.xmin, aoi.ymin))
            idx_first = 0

        size_x = (aoi.xmax - self.xmin) // self.res_x -     \
            (aoi.xmin - self.xmin) // self.res_x
        size_y = (aoi.ymax - self.ymin) // self.res_y -     \
            (aoi.ymin - self.ymin) // self.res_y

        if (size_x < 0) or (size_y < 0):
            raise ValueError("extent of box is negative in grid-coords?")

        # 1. List all cells (in the box) along the first row along the x-axis
        # 2. List all cells (in the box) above each cell listed in step 1.
        cells_list = list(
            chain.from_iterable(
                [np.arange(x, x + (self.steps_x + 1) * (size_y + 1), self.steps_x + 1)
                 for x in (idx_first + np.arange(size_x + 1))]
            )
        )

        cells_list = np.unique(cells_list).astype(int)
        return cells_list[np.isin(cells_list, self.cells)]
    
    
    def cells_on_boundary(self, aoi):
        
        idx_first = self.point_in_cell(aoi.xmin, aoi.ymin)

        if (idx_first < 0 and ((round(aoi.xmin, 0) < self.xmin) or (round(aoi.ymin) < self.ymin))):
            warnings.warn("Vertex of AOI is outside grid extent: (%.4f, %.4f)" % (aoi.xmin, aoi.ymin))
            idx_first = 0

        size_x = (aoi.xmax - self.xmin) // self.res_x -     \
            (aoi.xmin - self.xmin) // self.res_x
        size_y = (aoi.ymax - self.ymin) // self.res_y -     \
            (aoi.ymin - self.ymin) // self.res_y

        if (size_x < 0) or (size_y < 0):
            raise ValueError("extent of box is negative in grid-coords?")

        
        base = idx_first + np.arange(size_x + 1)
        top  = (self.steps_x + 1) * size_y + base
        left = idx_first + np.arange(0, (self.steps_x + 1) * size_y + 1, self.steps_x + 1)
        right = idx_first + size_x + np.arange(0, (self.steps_x + 1) * size_y + 1, self.steps_x + 1)
        

        cells_list = np.concatenate((base, left, right, top))
        cells_list = np.unique(cells_list).astype(int)
        return cells_list[np.isin(cells_list, self.cells)]





# Class to represent a (2D) spatial extent.
class AOI:

    # Initialisers
    def __init__(self, xmin, ymin, xmax, ymax):
        swap_x = xmax < xmin
        swap_y = ymax < ymin

        if(swap_x):
            xmin, xmax = xmax, xmin

        if(swap_y):
            ymin, ymax = ymax, ymin

        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
    
    
    @staticmethod
    def coords(aoi):
        try: 
            xmin, ymin, xmax, ymax = aoi.xmin, aoi.ymin, aoi.xmax, aoi.ymax
        except AttributeError:
            xmin, ymin, xmax, ymax = aoi
        
        return xmin, ymin, xmax, ymax
    
    
    def quadrants(self):
        xmid, ymid = self.centre()
        return (AOI(self.xmin, self.ymin, xmid, ymid), 
                AOI(self.xmin, ymid, xmid, self.ymax), 
                AOI(xmid, ymid, self.xmax, self.ymax), 
                AOI(xmid, self.ymin, self.xmax, ymid))
    

    def width(self):
        return self.xmax - self.xmin

    def height(self):
        return self.ymax - self.ymin

    def centre(self):
        return (self.xmin + self.width()/2), (self.ymin + self.height()/2)
    
    def area(self):
        return self.width() * self.height()


    # Comparators
    def contains_point(self, x, y):
        return (self.xmin <= x) & (self.ymin <= y) & (self.xmax >= x) & (self.ymax >= y)
    
    def contains_proper_point(self, x, y):
        return (self.xmin < x) & (self.ymin < y) & (self.xmax > x) & (self.ymax > y)
    
    def contains(self, other):
        try:
            xmin, ymin, xmax, ymax = AOI.coords(other)
        except AttributeError:
            xmin, ymin, xmax, ymax = other
        
        return self.contains_point(xmin, ymin) and self.contains_point(xmax, ymax)
    
    def contains_proper(self, other):
        try:
            xmin, ymin, xmax, ymax = AOI.coords(other)
        except AttributeError:
            xmin, ymin, xmax, ymax = other
        
        return self.contains_proper_point(xmin, ymin) and self.contains_proper_point(xmax, ymax)

    def intersects(self, other):
        xmin, ymin, xmax, ymax = AOI.coords(other)
        
        if (xmin >= self.xmax) or (self.xmin >= xmax):
            return False
        
        if (ymin >= self.ymax) or (self.ymin >= ymax):
            return False
        
        return True
        
    
    def intersection(self, other):
        
        if not self.intersects(other):
            return None
        
        xmin, ymin, xmax, ymax = AOI.coords(other)
        
        if xmin < self.xmin:
            xmin = self.xmin
        
        if ymin < self.ymin:
            ymin = self.ymin
        
        if xmax < self.xmax:
            xmax = self.xmax
        
        if ymax < self.ymax:
            ymax = self.ymax
        
        return AOI(xmin, ymin, xmax, ymax)
    
    def iou(self, other):
        
        common = self.intersection(other)
        areaUnion = self.area() + other.area()
                
        if common is not None:
            areaIntersection = common.area()
        else:
            areaIntersection = 0.0
        
        return areaIntersection / (areaUnion - areaIntersection)
            
        
