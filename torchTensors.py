#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 00:37:51 2018

@author: aylin
"""

#torch tensors in 1D


import torch 
import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt


int_to_tensor = torch.tensor([0,1,2,3,4])

print(int_to_tensor.type())
print(int_to_tensor.dtype)

float_to_tensor = torch.tensor([0.0,1.0,2.0])

print(float_to_tensor.type())
print(float_to_tensor.dtype)

floatTensor = torch.FloatTensor([0,1,2])

print(floatTensor.type())
print(floatTensor.dtype)

tensor1 = torch.tensor([0,1,2])
tensor2 = tensor1.type(torch.FloatTensor)
print(tensor2.type())

#size & dim

print(tensor2.size())
print(tensor2.ndimension())

tensor2D = tensor2.view(3,1)

print(tensor2D.size())
print(tensor2D.ndimension())

#if the tensor has a dynamic size -> reshape with -1

tensor2D_2 = tensor2.view(-1,1)

print(tensor2D_2.size())
print(tensor2D_2.ndimension())


#convert a tensor to a numpy array
numpy_array = np.array([0.0,1.1,2.2])
new_tensor = torch.from_numpy(numpy_array)

print(new_tensor.size())
print(new_tensor.ndimension())


#convert a tensor to a numpy array


back_to_numpy = new_tensor.numpy()

#set all elements in numpy array to zero
# back_to_numpy and numpy_array both will change 


numpy_array[:] = 0



#convert panda series to a tensor 

panda_series = pd.Series([0.1, 2, 0.3, 10.1])
new_tensor = torch.from_numpy(panda_series.values)
print(new_tensor)
print(new_tensor.dtype)
print(new_tensor.type())



#indexing and slicing

index_tensor = torch.tensor([0.2,1.2,2.2,3.2])

print(index_tensor[1])

tensor_sample = torch.tensor([20,1,4])

print(tensor_sample[0])

tensor_sample[0] = 100
print(tensor_sample[0])



#slice tensor sample

subset_tensor_sample = tensor_sample[1:2]
print(tensor_sample)
print(subset_tensor_sample)


selected_index = [1,2]
subset_tensor_sample = tensor_sample[selected_index]

print(tensor_sample)
print(subset_tensor_sample)


#using variable to assign the value to the selected indexes

tensor_sample[selected_index] = 10000
print(tensor_sample)




#TENSOR FUNCTIONS

math_tensor = torch.tensor([1.0,-1.0,1,-1])
print(math_tensor)

#standard deviation

standard_deviation = math_tensor.std()
print(standard_deviation)

max_val = math_tensor.max()
print(max_val)

min_val = math_tensor.min()
print(min_val)
#sin

sin_tensor = math_tensor.sin()
print(sin_tensor)

#linspace

len_5_tensor = torch.linspace(-2,2, steps = 5)
print(len_5_tensor)


#TENSOR OPERATIONS
u = torch.tensor([1,0])
v = torch.tensor([0,1])

w = u + v

print(w)

# Plot u, v, w

def plotVec(vectors):
    ax = plt.axes()
    
    # For loop to draw the vectors
    for vec in vectors:
        ax.arrow(0, 0, *vec["vector"], head_width = 0.05,color = vec["color"], head_length = 0.1)
        plt.text(*(vec["vector"] + 0.1), vec["name"])
    
    plt.ylim(-2,2)
    plt.xlim(-2,2)

plotVec([
    {"vector": u.numpy(), "name": 'u', "color": 'r'},
    {"vector": v.numpy(), "name": 'v', "color": 'b'},
    {"vector": w.numpy(), "name": 'w', "color": 'g'}
])
    
    
#tensor + scalar
    
k = u + 1
print(k)

#tensor * scalar

l = k * 2
print(l)

# tensor * tensor 

s = u * v
print(s)

#dot product

t = torch.dot(k,v)
print(t)



#2D TORCH TENSOR


#convert 2D List to 2D Tensor 

twoD_list = [[11,12,12], [21,22,23], [31,32,33]]
twoD_tensor = torch.tensor(twoD_list)
print(twoD_list)
print(twoD_tensor)


print(twoD_tensor.ndimension())
print(twoD_tensor.shape)
print(twoD_tensor.size())


#convert tensor to numpy array and tensor to numoy

twoD_numpy = twoD_tensor.numpy()
print(twoD_numpy)
print(twoD_numpy.dtype)


new_twoD_tensor = torch.from_numpy(twoD_numpy)
print(new_twoD_tensor)
print(new_twoD_tensor.dtype)


#convert panda dataframe to tensor

df = pd.DataFrame({'a':[11,21,31], 'b':[12,22,312]})

df_to_tensor = torch.from_numpy(df.values)
print(df_to_tensor.dtype) 


#indexing and slicing

tensor_example = torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]])

#What is the value on 2nd-row 3rd-column?

print(tensor_example[1,2])
print(tensor_example[1][2])

# both 1st-row 1st-column and 1st-row 2nd-column


print(tensor_example[0, 0:2])
print(tensor_example[0][0:2])

# Give an idea on tensor_obj[number: number][number]

tensor_example = torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]])
sliced_tensor_example = tensor_example[1:3]
print("1. Slicing step on tensor_example: ")
print("Result after tensor_example[1:3]: ", sliced_tensor_example)
print("Dimension after tensor_example[1:3]: ", sliced_tensor_example.ndimension())
print("================================================")
print("2. Pick an index on sliced_tensor_example: ")
print("Result after sliced_tensor_example[1]: ", sliced_tensor_example[1])
print("Dimension after sliced_tensor_example[1]: ", sliced_tensor_example[1].ndimension())
print("================================================")
print("3. Combine these step together:")
print("Result: ", tensor_example[1:3][1])

print("Dimension: ", tensor_example[1:3][1].ndimension())


#What is the value on 3rd-column last two rows?

print(tensor_example[1:3, 2])



#TENSOR OPERATIONS

#addition

X = torch.tensor([[1, 0],[0, 1]]) 
Y = torch.tensor([[2, 1],[1, 2]])
X_plus_Y = X + Y
print(X_plus_Y)


#SCALAR MULT.

Y = torch.tensor([[2, 1], [1, 2]]) 
two_Y = 2 * Y
print("The result of 2Y: ", two_Y)

#Element-wise Product/Hadamard Product

X = torch.tensor([[1, 0], [0, 1]])
Y = torch.tensor([[2, 1], [1, 2]]) 
X_times_Y = X * Y
print("The result of X * Y: ", X_times_Y)


#matrix mult.


A = torch.tensor([[0, 1, 1], [1, 0, 1]])
B = torch.tensor([[1, 1], [1, 1], [-1, 1]])
A_times_B = torch.mm(A,B)
print("The result of A * B: ", A_times_B)



#DERIVATIVES

import torch.functional as F


x = torch.tensor(2.0, requires_grad = True)
print("The tensor x: ", x)


y = x ** 2

print(y)

# Take the derivative. Try to print out the derivative at the value x = 2

y.backward()

print("The dervative at x = 2: ", x.grad)



# Calculate the y = x^2 + 2x + 1, then find the derivative 

x = torch.tensor(2.0, requires_grad = True)
y = x ** 2 + 2 * x + 1
print("The result of y = x^2 + 2x + 1: ", y)
y.backward()
print("The dervative at x = 2: ", x.grad)



#partial derivatives
# Calculate f(u, v) = v * u + u^2 at u = 1, v = 2

u = torch.tensor(1.0,requires_grad=True)
v = torch.tensor(2.0,requires_grad=True)
f = u * v + u ** 2
print("The result of v * u + u^2: ", f)


f.backward()
print(u.grad)


# Calculate the derivative with multiple values

x = torch.linspace(-10, 10, 10, requires_grad = True)
Y = x ** 2
y = torch.sum(x ** 2)


y.backward()

plt.plot(x.detach().numpy(), Y.detach().numpy(), label = 'function')
plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label = 'derivative')
plt.xlabel('x')
plt.legend()
plt.show()


import torch.nn.functional as F

# Take the derivative of Relu with respect to multiple value. Plot out the function and its derivative

x = torch.linspace(-3, 3, 100, requires_grad = True)
Y = F.relu(x)
y = Y.sum()
y.backward()
plt.plot(x.detach().numpy(), Y.detach().numpy(), label = 'function')
plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label = 'derivative')
plt.xlabel('x')
plt.legend()
plt.show()
