#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 21:57:33 2018

@author: aylin
"""

import torch 
import numpy as np

from torchvision import datasets
import torchvision.transforms as transforms

#number of subprocesses to use for data loading 
num_workers  = 0 

#how many samples per batch to load 

batch_size = 20  #number of training images that will be seen in one training iteration

#one training iteration means one time that a network make some mistakes and learn from them using back propagation

#convert data to torch.FloatTensor

transform  = transforms.ToTensor()

#choose the training and test datasets 

train_data = datasets.MNIST(root = 'data', train = True, download = True, transform = transform)

test_data = datasets.MNIST(root = 'data', train = False, download = True, transform = transform)


#prepare data loader

train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, num_workers = num_workers)

test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, num_workers = num_workers)


#VISUALIZE DATA

import matplotlib.pyplot as plt

#obtain one batch of training images 

dataiter = iter(train_loader)

images, labels = dataiter.next()

images = images.numpy()

#plot the images in the batch, along with corresponding labels 

fig = plt.figure(figsize=(10,2))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx + 1, xticks =[], yticks = [])
    ax.imshow(np.squeeze(images[idx]), cmap = 'gray')
    #print out the correct label for each image
    #.item() gets the value contained in a Tensor
    
    ax.set_title(str(labels[idx].item()))
    

#view images more detail
    
    
img = np.squeeze(images[1])

fig = plt.figure(figsize=(12,12))

ax = fig.add_subplot(111)


ax.imshow(img, cmap = 'gray')


width, height = img.shape


thresh = img.max()/2.5

for x in range(width):
    for y in range(height):
        val = round(img[x][y], 2) if img[x][y] != 0 else 0
        ax.annotate(str(val), xy = (y,x), horizontalalignment = 'center', verticalalignment = 'center',
        color = 'white' if img[x][y]<thresh else 'black')
        
        
#define nn architecture
        
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    
 
    def __init__(self):
        super(Net, self).__init__()
        #number of hidden nodes in each layer 512
        hidden_1 = 512
        hidden_2 = 512
        #linear layer 784-> hidden_1
        self.fc1 = nn.Linear(28*28, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, 10)
        
        #droput layer (p = 0.2) for prevents overfitting
        
        self.dropout = nn.Dropout(0.2)
        
        
        
        
    def forward(self, x):
        #flatten image input
         x = x.view(-1, 28 * 28)
         
         #add hidden layer with ReLU activation function
         
         x = F.relu(self.fc1(x))
         
         x = self.dropout(x)
         
         x = F.relu(self.fc2(x)) 
         x = self.dropout(x)
         
         #add output layer
         x = self.fc3(x)
         return x
     
        
         
         
         
         
 #initialize model
 
model = Net()
print(model)
         


#specify the loss function and optimizer

#loss function with categorical cross-entropy

criterion = nn.CrossEntropyLoss()

#optimizer with stochastic gradient descent, learning rate = 0.01

optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)



# number of epochs to train the model
n_epochs = 50

model.train() # prep model for training

for epoch in range(n_epochs):
    # monitor training loss
    train_loss = 0.0
    
    ###################
    # train the model #
    ###################
    for data, target in train_loader:
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()*data.size(0)
             
    # print training statistics 
    # calculate average loss over an epoch
    train_loss = train_loss/len(train_loader.dataset)

    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch+1, 
        train_loss
        ))
    
    
    
#test the trained network
    
    
#initialize list to monitor test loss and accuracy 

test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

for data, target in test_loader:
     #forward pass: compute predicted output by passing inputs to the model 
     output = model(data)
     
     #calculate the loss 
     loss = criterion(output, target)
     
     #update test loss
     
     test_loss += loss.item()* data.size(0)
     
     #convert output probabilities to predicted class
     
     _, pred = torch.max(output, 1)

     
     correct = np.squeeze(pred.eq(target.data.view_as(pred)))
    # calculate test accuracy for each object class
     for i in range(batch_size):
         label = target.data[i]
         class_correct[label] += correct[i].item()
         class_total[label] += 1


# calculate and print avg test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            str(i), 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))


print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
     
     

# obtain one batch of test images
dataiter = iter(test_loader)
images, labels = dataiter.next()

# get sample outputs
output = model(images)
# convert output probabilities to predicted class
_, preds = torch.max(output, 1)
# prep images for display
images = images.numpy()

# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    ax.set_title("{} ({})".format(str(preds[idx].item()), str(labels[idx].item())),
                 color=("green" if preds[idx]==labels[idx] else "red"))    
     