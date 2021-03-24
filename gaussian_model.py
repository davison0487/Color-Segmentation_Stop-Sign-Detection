# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 00:59:23 2020

@author: A53295675 Yunhsiu Wu 
"""
import numpy as np

def Gaussian_model(x, mean, cov, prior):
    """
    Get gaussian prediction
    Arguments: pixel, mean, covariance, prior
    Return: prediction    
    ( 1 / sqrt( ( (2*pi) ^dim ) * det(cov) ) ) * exp( -0.5 * (x-mean) * inv(cov) * transpose(x-mean) );
    
    """
    dimension = x.shape[1]
    cov_inv = np.linalg.cholesky( np.linalg.inv(cov) ) #credit to Arthur Hsieh
    coefficient = 1 / np.sqrt( (2*np.pi) ** dimension * np.linalg.det(cov) )
    exp_term = np.exp( -0.5 * np.sum( np.square( np.dot(x - mean, cov_inv) ), axis=1 ) )
    
    return coefficient * exp_term * prior