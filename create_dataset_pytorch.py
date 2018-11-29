#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 00:07:08 2018

@author: aylin
"""

# These are the libraries will be used for this lab.

import torch
from torch.utils.data import Dataset
torch.manual_seed(1)
#The torch.manual_seed() is for forcing the random function to give the same number every time we try to recompile it.

#define class for dataset

class toy_set(Dataset):
    
    #constructor with default values
    
    def __init__(self, length = 100, transform = None):
        self.len = length
        self.x = 2*torch.ones(length, 2)
        self.y = torch.ones(length,1)
        self.transform = transform
        
    #getter
    
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    #get length
    
    def __len__(self):
        return self.len
    
        
our_dataset = toy_set()
print(our_dataset)
print(our_dataset[0])
print(len(our_dataset))


for i in range(3):
    x, y = our_dataset[i]
    print(i)
    print(x)
    print(y)
    print("***")
    
#create transform class add_mult
    
class add_mult(object):
    
    #constructor
    def __init__(self, addx = 1, muly=2):
        self.addx = addx
        self.muly = muly
        
        
    #executor
    
    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x + self.addx
        y = y * self.muly
        sample = x ,y
        return sample
    
    
# Create an add_mult transform object, and an toy_set object

a_m = add_mult()
data_set = toy_set()

# Use loop to print out first 10 elements in dataset

for i in range(10):
    x, y = data_set[i]
    print('Index: ', i, 'Original x: ', x, 'Original y: ', y)
    x_, y_ = a_m(data_set[i])
    print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)
    
    
# Create a new data_set object with add_mult object as transform

cust_data_set = toy_set(transform = a_m)

# Use loop to print out first 10 elements in dataset

for i in range(10):
    x, y = data_set[i]
    print('Index: ', i, 'Original x: ', x, 'Original y: ', y)
    x_, y_ = cust_data_set[i]
    print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)
    
    
from torchvision import transforms

# Create tranform class mult

class mult(object):
    
    # Constructor
    def __init__(self, mult = 100):
        self.mult = mult
        
    # Executor
    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x * self.mult
        y = y * self.mult
        sample = x, y
        return sample
    
    
# Combine the add_mult() and mult()

data_transform = transforms.Compose([add_mult(), mult()])
print("The combination of transforms (Compose): ", data_transform)


# Create a new toy_set object with compose object as transform

compose_data_set = toy_set(transform = data_transform)


# Use loop to print out first 3 elements in dataset

for i in range(3):
    x, y = data_set[i]
    print('Index: ', i, 'Original x: ', x, 'Original y: ', y)
    x_, y_ = cust_data_set[i]
    print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)
    x_co, y_co = compose_data_set[i]
    print('Index: ', i, 'Compose Transformed x_co: ', x_co ,'Compose Transformed y_co: ',y_co)
    
    
    
#PREBUILT DATASETS
    
    
import torch 
import matplotlib.pylab as plt
import numpy as np
torch.manual_seed(0)


# Show data by diagram

def show_data(data_sample, shape = (28, 28)):
    plt.imshow(data_sample[0].numpy().reshape(shape), cmap='gray')
    plt.title('y = ' + str(data_sample[1].item()))
    
    
import torchvision.transforms as transforms
import torchvision.datasets as dsets

#import prebuilt dataset into variable dataset


dataset = dsets.MNIST(
        root = './data',
        train = False,
        download = True,
        transform = transforms.ToTensor()
        )


print(type(dataset[0]))
print(len(dataset[0]))
print(dataset[0][0].shape)
print(type(dataset[0][0]))

show_data(dataset[0])


# Combine two transforms: crop and convert to tensor. Apply the compose to MNIST dataset

croptensor_data_transform = transforms.Compose([transforms.CenterCrop(20), transforms.ToTensor()])
dataset = dsets.MNIST(root = './data', train = False, download = True, transform = croptensor_data_transform)
print("The shape of the first element in the first tuple: ", dataset[0][0].shape)

show_data(dataset[0],shape = (20, 20))

show_data(dataset[1],shape = (20, 20))


# Construct the compose. Apply it on MNIST dataset. Plot the image out.

fliptensor_data_transform = transforms.Compose([transforms.RandomHorizontalFlip(p = 1),transforms.ToTensor()])
dataset = dsets.MNIST(root = './data', train = False, download = True, transform = fliptensor_data_transform)
show_data(dataset[1])