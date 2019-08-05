# -*- coding: utf-8 -*-
"""
Created on Sat May  4 13:14:44 2019

@author: Thanh Tung Khuat

This is a file to define all utility functions for measuring the performance 
"""

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score


def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    """
        AUC ROC Curve Scoring Function for Multi-class Classification
    """
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    
    return roc_auc_score(y_test, y_pred, average=average)

def manhattan_distance(X, Y):
    """
        Compute Manhattan distance of two matrices X and Y
        
        Input:
            X, Y: two numpy arrays (1D or 2D)
            
        Output:
            A numpy array containing manhattan distance. If X and Y are 1-D arrays, the output only contains one element
            The number of elements in the output array is the number of rows of X or Y
    """
    if X.ndim > 1:
        return (np.abs(X - Y)).sum(1)
    else:
        return (np.abs(X - Y)).sum()
    
    


