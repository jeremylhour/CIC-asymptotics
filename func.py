#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 13:43:00 2020

@author: jeremylhour
"""

import numpy as np

def ranks_and_antiranks(points):
    """
    ranks_and_antiranks:
        returns ranks and antiranks for array of points
        
    :param points: np.array, vector of points of dimension 1
    """
    ranks = np.array([sorted(points).index(i) for i in points])
    return ranks, np.argsort(ranks)


def smoothed_ecdf(new_points, x):
    """
    smoothed_ecdf:
        Smoothed empirical CDF
        as in Shorack and Wellner (p. 86)
    
    :param new_points: new points for which the value is returned
    :param x: points used to compute the smoothed empirical CDF
    """
    n_points = len(x)
    sorted_x = np.zeros(n_points+1)
    sorted_x[1:] =  np.array(sorted(x))
    sorted_x = np.append(sorted_x,1) # add 0 first and 1 at the end
    
    y = []
    for new_point in new_points:
            index = np.where(sorted_x  <= new_point)[0][-1]
            rank_bounds = [index, index+1]
            bounds = sorted_x[rank_bounds]
    
            b = 1/((n_points+1)*(bounds[1]-bounds[0]))
            a = rank_bounds[0]/(n_points+1) - b*bounds[0]
            y.append(a + b*new_point)
        
    return np.array(y)