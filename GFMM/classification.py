# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 11:22:08 2018

@author: Thanh Tung Khuat

GFMM Predictor

"""

import numpy as np
from functionhelper.measurehelper import manhattan_distance
from functionhelper.membershipcalc import memberG
from functionhelper.bunchdatatype import Bunch
from functionhelper import device, GPU_Computing_Threshold, is_Have_GPU, UNLABELED_CLASS

def predict(V, W, classId, XlT, XuT, patClassIdTest, gama = 1, oper = 'min'):
    """
    GFMM classifier (test routine)

      result = predict(V,W,classId,XlT,XuT,patClassIdTest,gama,oper)

    INPUT
      V                 Tested model hyperbox lower bounds
      W                 Tested model hyperbox upper bounds
      classId	          Input data (hyperbox) class labels (crisp)
      XlT               Test data lower bounds (rows = objects, columns = features)
      XuT               Test data upper bounds (rows = objects, columns = features)
      patClassIdTest    Test data class labels (crisp)
      gama              Membership function slope (default: 1)
      oper              Membership calculation operation: 'min' or 'prod' (default: 'min')

   OUTPUT
      result           A object with Bunch datatype containing all results as follows:
                          + summis           Number of misclassified objects
                          + misclass         Binary error map
                          + sumamb           Number of objects with maximum membership in more than one class
                          + out              Soft class memberships
                          + mem              Hyperbox memberships

    """
	if len(XlT.shape) == 1:
        XlT = XlT.reshape(1, -1)
    if len(XuT.shape) == 1:
        XuT = XuT.reshape(1, -1)
		
    #initialization
    yX = XlT.shape[0]
    misclass = np.zeros(yX)

    # classifications
    for i in range(yX):
        mem = memberG(XlT[i, :], XuT[i, :], V, W, gama, oper) # calculate memberships for all hyperboxes
        bmax = mem.max()	                                          # get max membership value
        maxVind = np.nonzero(mem == bmax)[0]                         # get indexes of all hyperboxes with max membership

        if bmax == 0:
            print('zero maximum membership value')                     # this is probably bad...
            misclass[i] = True
        else:
            if len(np.unique(classId[maxVind])) > 1:
                #print('Input is in the boundary')
                misclass[i] = True
            else:
                if np.any(classId[maxVind] == patClassIdTest[i]) == True or patClassIdTest[i] == UNLABELED_CLASS:
                    misclass[i] = False
                else:
                    misclass[i] = True
                #misclass[i] = ~(np.any(classId[maxVind] == patClassIdTest[i]) | (patClassIdTest[i] == 0))

    # results
    summis = np.sum(misclass).astype(np.int64)

    result = Bunch(summis = summis, misclass = misclass)
    return result
	
def predict_with_manhattan(V, W, classId, XlT, XuT, patClassIdTest, gama = 1, oper = 'min'):
    """
    GFMM classifier (test routine): Using Manhattan distance in the case of many hyperboxes with different classes having the same maximum membership value

      result = predict(V,W,classId,XlT,XuT,patClassIdTest,gama,oper)

    INPUT
      V                 Tested model hyperbox lower bounds
      W                 Tested model hyperbox upper bounds
      classId	          Input data (hyperbox) class labels (crisp)
      XlT               Test data lower bounds (rows = objects, columns = features)
      XuT               Test data upper bounds (rows = objects, columns = features)
      patClassIdTest    Test data class labels (crisp)
      gama              Membership function slope (default: 1)
      oper              Membership calculation operation: 'min' or 'prod' (default: 'min')

   OUTPUT
      result           A object with Bunch datatype containing all results as follows:
                          + summis           Number of misclassified objects
                          + misclass         Binary error map
                          + numSampleInBoundary     The number of samples in decision boundary

    """
	if len(XlT.shape) == 1:
        XlT = XlT.reshape(1, -1)
    if len(XuT.shape) == 1:
        XuT = XuT.reshape(1, -1)
		
    #initialization
    yX = XlT.shape[0]
    misclass = np.zeros(yX)
    numPointInBoundary = 0
    # classifications
    for i in range(yX):
        if patClassIdTest[i] == UNLABELED_CLASS:
            misclass[i] = False
        else:          
            mem = memberG(XlT[i, :], XuT[i, :], V, W, gama, oper) # calculate memberships for all hyperboxes
            bmax = mem.max()	                                          # get max membership value
            maxVind = np.nonzero(mem == bmax)[0]                         # get indexes of all hyperboxes with max membership
    
            if bmax == 0:
                print('zero maximum membership value')                     # this is probably bad...
                misclass[i] = True
            else:
                if len(np.unique(classId[maxVind])) > 1:
                    numPointInBoundary = numPointInBoundary + 1
                    #print("Using Manhattan function")
                    if (XlT[i] == XuT[i]).all() == False:
                        XlT_mat = np.ones((len(maxVind), 1)) * XlT[i]
                        XuT_mat = np.ones((len(maxVind), 1)) * XuT[i]
                        XgT_mat = (XlT_mat + XuT_mat) / 2
                    else:
                        XgT_mat = np.ones((len(maxVind), 1)) * XlT[i]
                    # Find all average points of all hyperboxes with the same membership value
                    avg_point_mat = (V[maxVind] + W[maxVind]) / 2
                    # compute the manhattan distance from XgT_mat to all average points of all hyperboxes with the same membership value
                    maht_dist = manhattan_distance(avg_point_mat, XgT_mat)
                    id_min_dist = maht_dist.argmin()
                    
                    if classId[maxVind[id_min_dist]] == patClassIdTest[i]:
                        misclass[i] = False
                    else:
                        misclass[i] = True
                else:
                    if np.any(classId[maxVind] == patClassIdTest[i]) == True:
                        misclass[i] = False
                    else:
                        misclass[i] = True
                    #misclass[i] = ~(np.any(classId[maxVind] == patClassIdTest[i]) | (patClassIdTest[i] == 0))

    # results
    summis = np.sum(misclass).astype(np.int64)

    result = Bunch(summis = summis, misclass = misclass, numSampleInBoundary = numPointInBoundary)
    
    return result
