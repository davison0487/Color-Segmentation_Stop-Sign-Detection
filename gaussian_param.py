# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 19:01:43 2020

@author: A53295675 Yunhsiu Wu 

Put npz files at ./npz
"""

import os
import numpy as np

folder = r'./npz'

def gaussian_parameter(color) :
    """
    Calculate gaussian parameter
    Argument: class name        
    Return: mean dict, covariance dict and prior dict
            If empty, return 0 and array([0.])
    """
    #data storage
    color_BGR = np.zeros((1,3))
    other_BGR = np.zeros((1,3))
    
    #read out data
    for filename in os.listdir(folder):
        npz_file = os.path.join(folder, filename)
        np_data = np.load(npz_file)
        for key in np_data:
            data = np_data[key]
            if data.size == 0 or key == 'MASK_STOP_SIGN':
                continue
            #switching RGB to BGR
            temp = np.array(data[:, 0])
            data[:, 0] = data[:, 2]
            data[:, 2]= temp
            #add to data storage
            if key == color:
                color_BGR = np.append(color_BGR, data, 0)
            else:
                other_BGR = np.append(other_BGR, data, 0)
    
    #take away initialization zeros
    color_BGR = np.array(np.delete(color_BGR, 0, 0))
    other_BGR = np.array(np.delete(other_BGR, 0, 0))
    
    #calculate parameters
    mean ,cov , prior = {}, {}, {}
    mean[color] = np.mean(color_BGR, 0)
    mean['other'] = np.mean(other_BGR, 0)
    cov[color] = np.cov(color_BGR.T)
    cov['other'] = np.cov(other_BGR.T)
    prior[color] = color_BGR.shape[0] / (color_BGR.shape[0] + other_BGR.shape[0])
    prior['other'] =other_BGR.shape[0] / (color_BGR.shape[0] + other_BGR.shape[0])
    
    return mean, cov, prior

if __name__ == '__main__':
    """
    color class = { COLOR_STOP_SIGN_RED, COLOR_OTHER_RED, COLOR_BROWN, COLOR_ORANGE, COLOR_BLUE, COLOR_OTHER }
    """
    COLOR_STOP_SIGN_RED_mean, COLOR_STOP_SIGN_RED_cov, COLOR_STOP_SIGN_RED_prior = {}, {}, {}
    COLOR_STOP_SIGN_RED_mean, COLOR_STOP_SIGN_RED_cov, COLOR_STOP_SIGN_RED_prior = gaussian_parameter('COLOR_STOP_SIGN_RED')
    print(COLOR_STOP_SIGN_RED_mean)
    print(COLOR_STOP_SIGN_RED_cov)
    print(COLOR_STOP_SIGN_RED_prior)

