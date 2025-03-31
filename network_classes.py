# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 15:47:07 2025

Author: Adam Coxson, PhD student, University of Liverpool
Department of Chemistry, Materials Innovation Factory, Levershulme Research Centre
Module: network_classes.py
Local dependencies: None
For the dML workshop Mar 2025

# https://medium.com/@shashankshankar10/introduction-to-neural-networks-build-a-single-layer-perceptron-in-pytorch-c22d9b412ccf
# Numpy only perceptron: https://sebastianraschka.com/Articles/2015_singlelayer_neurons.html
# https://www.kaggle.com/code/pinocookie/pytorch-simple-mlp

Contains PyTorch classes for a 1-layer, 2-layer and variable multi-layer feedforward neural networks

"""

import torch.nn as nn
from torch.nn.modules import Module


################ NETWORK CLASSES ################

class SingleLayerPerceptron(nn.Module):
    """
    A simple single-layer perceptron model with one hidden layer.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_dim : int, optional
        Number of neurons in the hidden layer (default is 100).
    output_dim : int, optional
        Number of output neurons (default is 1).
    activation_func : torch.nn.Module, optional
        Activation function applied after the hidden layer (default is None).

    Methods
    -------
    forward(x)
        Performs forward propagation through the network. Returns X, torch.Tensor object.
    """
    def __init__(self,
                 input_dim:  int,
                 hidden_dim: int = 100,
                 output_dim: int = 1,
                 activation_func=None
                 ): 
        super(SingleLayerPerceptron, self).__init__()
        
        # Define the layer variables and activation functions within class
        self.activation_func=activation_func
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x): # Propagate data through layers when class forward method is called
        x = x.view(x.size(0), -1)  # Flatten input if necessary
        x = self.hidden_layer(x)
        if self.activation_func is not None:
            x = self.activation_func(x)
        x = self.output_layer(x)
        return x  


class MLP_2layer(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) with two hidden layers.
    
    Parameters
    ----------
    input_dim : int
        Number of input features.
    output_dim : int, optional
        Number of output neurons.
    neurons : list, optional
        List containing the number of neurons in each hidden layer (default is [200, 100]).
    activation_func : torch.nn.Module, optional
        Activation function applied after each hidden layer (default is ReLU).
    
    Methods
    -------
    forward(x)
        Performs forward propagation through the network. Returns X, torch.Tensor object.
    """
    
    def __init__(self,
                 input_dim:  int,
                 output_dim: int,
                 neurons:    list = [200,100],
                 activation_func  = None
                 ):
        super(MLP_2layer, self).__init__()
        if activation_func is None: activation_func=nn.ReLU()
        
        # Building the network object by hardcoding 2 layers
        self.layers = nn.Sequential( # Sequential pytorch container, forward method can be directly applied 
            nn.Linear(input_dim, neurons[0]),   # Add 1st hidden layer
            activation_func,                    # Apply activation to layer
            nn.Linear(neurons[0], neurons[1]),  # Add 2nd hidden layer
            activation_func,                    # Apply activation to layer
            nn.Linear(neurons[1], output_dim) # Linear output layer with no activation function
        )
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)        # Apply data to Sequential object, that passes data through each layer for you
        return x

class MLP(nn.Module):
    """
    A flexible Multi-Layer Perceptron (MLP) that supports an arbitrary number of layers and activation functions.
    Note, the output neurons are linear and have no activation function (In some cases, like classification, sigmoid is used).
    Also applies dropout to the first 3 layers (Hardcoded in, feel free to change to make more flexible). Arguably, dropout
    is most impactful on the layers with most parameters. You can still apply it anywhere and eveywhere, but test it first.
    
    https://discuss.pytorch.org/t/how-to-create-mlp-model-with-arbitrary-number-of-hidden-layers/13124/5
    
    Parameters
    ----------
    input_dim : int
        Number of input features.
    output_dim : int
        Number of output neurons.
    layers_data : list
        List of tuples, each containing a layer size and an activation function (e.g., [(layer1, nn.ReLU()), (layer2, nn.ReLU()), ...]).
        Note, this does not account for the final layer, which is just a single final linear neuron.
    dropout : float, optional
        Dropout probability applied to the first three layers (default is 0).

    Methods
    -------
    forward(X)
        Performs forward propagation of data through the network. Returns X, torch.Tensor object.
    """

    def __init__(self,
                 input_dim:   int,
                 output_dim:  int,
                 layers_data: list,  #format [(layer1, nn.ReLU()), (layer2, nn.ReLU()), (output_size, nn.Sigmoid())]
                 dropout_prob:     float = 0
                 ):
        super().__init__()
        self.layers = nn.ModuleList() # Pytorch ModuleList does not have a forward method like nn.sequential
        self.dropout = nn.Dropout(p=dropout_prob)

        # Build the layers up sequentially (add nn.linear, add activation, add dropout, then loop to next layer)
        for i, (size, activation) in enumerate(layers_data): 
            self.layers.append(nn.Linear(input_dim, size))
            if i in [0,1,2]: # DROPOUT CURRENTLY BROKEN MAYBE?
                self.layers.append(self.dropout) # Apply dropout to output of first 3 hidden layers (currently hardcoded)
            input_dim = size  # Update input dimension for the next layer
            if activation is not None:
                assert isinstance(activation, Module), \
                    "Each tuple should contain a layer size (int) and a torch.nn.modules.Module."
                self.layers.append(activation)
        self.output = nn.Linear(input_dim, output_dim) # Final layer is a layer of linear neurons without an activation function

    def forward(self, X):
        for layer in self.layers: # Loop to propagate data over elements in ModuleList
            X = layer(X)
        #X = torch.sigmoid(self.output(X)) # Sometimes you could apply sigmoid or some other activation function on the output layer
        X = self.output(X)
        return X
    
