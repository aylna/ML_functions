#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 00:32:00 2018

@author: aylin
"""

#cross-entropy

import numpy as np 

Y=[1,0,1,1] 
P=[0.4,0.6,0.1,0.5]


#a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    sumCE = 0
    for i,j in enumerate(Y):
        sum_i = Y[i]*np.log(P[i]) + (1-Y[i])*(np.log(1-P[i]))
        sumCE = sumCE + sum_i
        
    return sumCE*(-1)

print(cross_entropy(Y,P))