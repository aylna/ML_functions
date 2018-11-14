#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 22:52:00 2018

@author: aylin
"""


import numpy as np

#softmax function

L =np.array([1,2,3]) 



def softmax(L):
    k=[]
      
    for i in L:
        i = np.exp(i)/sum(np.exp(L))
        k.append(i)
        
   
    return k

KS = []
KS = softmax(L)

print(KS)
