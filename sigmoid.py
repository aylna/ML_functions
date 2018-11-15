#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 00:53:31 2018

@author: aylin
"""

import numpy as np 

def sigmoid(x):
    f = 1/(1+np.exp(-x))
    
    return f



print(sigmoid(5))