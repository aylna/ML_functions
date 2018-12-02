#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 01:10:05 2018

@author: aylin
"""

#linear regression : 1D prediction

import torch

w = torch.tensor(2.0, requires_grad = True)
b = torch.tensor(-1.0, requires_grad = True)

def forward(x):
    yhat = w*x + b
    return yhat

#predict y = 2x-1 at x = 1
x = torch.tensor([[1.0]])
yhat = forward(x)
print("The prediction: ", yhat)


x = torch.tensor([[1.0], [2.0]])
print("The shape of x: ", x.shape)

#make the prediction of y=2x-1 at x =[1,2]

yhat = forward(x)
print("The prediction: ", yhat)

#class linear

from torch.nn import Linear

torch.manual_seed(1)

#create linear regression model, and print out the parameters


#parameters are randomly created
lr = Linear(in_features=1, out_features=1, bias=True)
print("Parameters w and b: ", list(lr.parameters()))

x = torch.tensor([[1.0]])
yhat = lr(x)
print(yhat)


x = torch.tensor([[1.0],[2.0]])
yhat = lr(x)
print(yhat)


#build custom modules

from torch import nn

#customize linear regressşon class

class LR(nn.Module):
    
    #constructor
    
    def __init__(self, input_size, output_size):
        
        #inherit from parent
        super(LR, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        out = self.linear(x)
        return out
    
    
lr = LR(1,1)
print(list(lr.parameters()))
print(lr.linear)


x = torch.tensor([[1.0]])
yhat = lr(x)
print("The prediction: ", yhat)

x = torch.tensor([[1.0], [2.0]])
yhat = lr(x)
print("The prediction: ", yhat)