#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 18:35:54 2018

@author: aylin
"""

#linear regression 1D: training One Parameter

import numpy as np
import torch
import matplotlib.pyplot as plt

class plot_diagram():
    

    #constructor
    
    def __init__(self, X, Y, w, stop, go = False):
        
        start = w.data
        self.error = []
        self.parameter =[]
        self.X = X.numpy()
        self.Y = Y.numpy()
        self.parameter_values = torch.arange(start, stop)
        self.Loss_function = [criterion(forward(X), Y) for w.data in self.parameter_values]
        w.data = start
        
    #executor
    
    def __call__(self, Yhat, w, error, n):
        self.error.append(error)
        self.parameter.append(w.data)
        plt.subplot(212)
        plt.plot(self.X, Yhat.detach().numpy())
        plt.plot(self.X, self.Y, 'ro')
        plt.xlabel("A")
        plt.ylim(-20,20)
        plt.subplot(211)
        plt.title("Data Space (top) Estimated Line (bottom) Iteration " + str(n))
        plt.plot(self.parameter_values.numpy(), self.Loss_function)
        plt.plot(self.parameter, self.error, 'ro')
        plt.xlabel("B")
        plt.figure()
        
    #destructor
    
    def __del__(self):
        plt.close('all')
        
        
        
#make some data
        
        
#create the f(x) with a slope of -3
        
X = torch.arange(-3, 3, 0.1).view(-1,1)
f = -3 * X



plt.plot(X.numpy(),f.numpy(), label = 'f')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

Y = f + 0.1 * torch.randn(X.size())

plt.plot(X.numpy(), Y.numpy(), 'rx', label = 'Y')
plt.plot(X.numpy(), f.numpy(), label = 'f')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
        
        

#create model and cost function

def forward(x):
    return w*x


#define the cost function using mean square error
    
def criterion(yhat, y):
    return torch.mean((yhat - y)**2)

#create learning rate and an empty list to record the loss for each iteration
    
lr = 0.1
LOSS =[]

#create a model  parameter by setting the argyment requires_grad = True because the system must learn it

w = torch.tensor(-10.0, requires_grad = True)

#create a plot_diagram object to visualize the data space and the parameter space for each 
#iteration during training


gradient_plot = plot_diagram(X, Y, w, stop=5)



#train model

def train_model(iter):
    for epoch in range(iter):
        
        Yhat = forward(X)
        
        #calculate the iteration
        
        loss = criterion(Yhat, Y)
        
        #plot the diagram for have better idea

        gradient_plot(Yhat, w, loss.item(), epoch)
        
        #store the loss into list
        
        LOSS.append(loss)
        
        #backward pass: compute gradient of the loss wrt all the learnable path
        
        loss.backward()
        
        #update parameters
        
        w.data = w.data - lr * w.grad.data
        
        #zero the gradient before running the backward pass
        
        w.grad.data.zero_()
        
 


# Give 4 iterations for training the model here.

train_model(4)  


# Plot the loss for each iteration

plt.plot(LOSS)
plt.tight_layout()
plt.xlabel("Epoch/Iterations")
plt.ylabel("Cost")     