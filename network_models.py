# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 19:44:39 2025

Author: Adam Coxson, PhD student, University of Liverpool
Department of Chemistry, Materials Innovation Factory, Levershulme Research Centre
Module: network_models.py
Local dependencies: network_classes.py, stats_and_plot_functions.py
For the dML workshop Mar 2025

# https://medium.com/@shashankshankar10/introduction-to-neural-networks-build-a-single-layer-perceptron-in-pytorch-c22d9b412ccf
# Numpy only perceptron: https://sebastianraschka.com/Articles/2015_singlelayer_neurons.html
# https://www.kaggle.com/code/pinocookie/pytorch-simple-mlp

Trains a neural network on 1D regression data and visualises the results.
"""

############## PACKAGE IMPORTS ###############
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

############### LOCAL IMPORTS ###############
from network_classes import SingleLayerPerceptron, MLP_2layer, MLP 
from stats_and_plot_functions import compute_y, sample_x_values, normalize_data, mean_squared_error, plot_train_loss, doubleplot, calc_num_net_parameters


torch.manual_seed(42) # 42 for reproducibility

# Uncomment to show the True function of ((0.4*x + 0.5*np.sin(5*x) + np.sin(3*x)) + 10*np.cos(x)*np.exp(-0.1*x)) + 7
x_min, x_max, dx = 0, 30, 0.001
num_samples = 50
x_values = np.linspace(x_min, x_max, 300)
y_values = compute_y(x_values)
# Randomly sample x values and compute corresponding y values
sampled_x = sample_x_values(x_min, x_max, dx, num_samples)
sampled_y = compute_y(sampled_x)
# doubleplot(x_values, y_values, sampled_x, sampled_y,labels=["x","y","True function","50 Random 'training' points",""],lims=[[0,30],[0,20]])
# exit()

# Hyperparameters
input_dim = 1 # We only have one input dimension (the scalar x value)
output_dim = 1  # We only have one output dimension (the scalar y value)
learning_rate = 0.001
num_epochs = 200
batch_size = 32
neurons=[100,50,20,10]
#neurons=5000
#neurons=[100, 50]
# See list of different activation functions https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
activation = [nn.SELU(), nn.ReLU(), nn.Sigmoid()][1] 
dropout=0.0  # Careful with this as it may currently be broken



# Initialize model
#model = SingleLayerPerceptron(input_dim=1, hidden_dim=neurons, output_dim=1,activation_func=activation)
#model = MLP_2layer(input_dim=1,output_dim=1,neurons=neurons,activation_func=activation)
layers=[] # Format layers for variable MLP class
for n in neurons:
    layers.append((n, activation))
model = MLP(input_dim,output_dim,layers,dropout_prob=dropout)

if type(neurons) is list:
    nparams= calc_num_net_parameters(neurons.copy(), output_size=1)
else:
    nparams= calc_num_net_parameters(neurons, output_size=1)

# Training data params
training_samples=20000 
x_min, x_max=0,30

# Create and preprocess training data into 32 bit floats for PyTorch torch.Tensor objects
x_vals_train=sample_x_values(x_min, x_max, dx=0.00001, num_samples=training_samples).astype(np.float32)
y_vals_train = compute_y(x_vals_train).astype(np.float32)
x_vals_train_normalized, x_mean, x_std = normalize_data(x_vals_train)
y_vals_train_normalized, y_mean, y_std = normalize_data(y_vals_train)
X_train = torch.tensor(x_vals_train_normalized.reshape(len(x_vals_train),1), device='cpu')
y_train = torch.tensor(y_vals_train_normalized.reshape(len(y_vals_train),1), device='cpu')

# Create Validation data. In this simple case we are just sampling the true function, but realisitically it is taken from
# shuffled training data, i.e. for k-fold cross validation.
x_vals_valid=np.arange(x_min, x_max + 0.01, 0.1).astype(np.float32)
y_vals_valid = compute_y(x_vals_valid).astype(np.float32)
x_vals_valid_normalised = (x_vals_valid - x_mean) / x_std # DELIBERATELY Using normalisation from TRAINING data
y_vals_valid_normalised = (y_vals_valid - y_mean) / y_std # DELIBERATELY Using normalisation from TRAINING data
X_valid = torch.tensor(x_vals_valid_normalised.reshape(len(x_vals_valid),1), device='cpu')

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
criterion = nn.MSELoss()  
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop with validation tracking
train_losses = []
valid_losses = []
best_valid_loss, best_train_loss, best_epoch = float("inf"), float("inf"), 0
best_model_state = None  # To store the best model state dict
for epoch in range(num_epochs):
    total_loss = 0.0
    model.train() # Tell model we are in training mode, weights can be modified, later we switch to evaluation mode
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        batch_X, batch_y = batch_X.float(), batch_y.float()
        outputs = model(batch_X).squeeze()  # Ensure output shape matches target
        loss = criterion(outputs, batch_y.squeeze())  # Compute loss
        loss.backward()  # Apply backpropagation
        optimizer.step()  # Update weights
        total_loss += loss.item()
    
    model.eval() # Important to tell the model we're in evaluation model, ensures no weights are changed by accident
    with torch.no_grad():
        valid_pred_normalized = np.squeeze(model(X_valid).detach().numpy()) # Predict validation data
    train_loss = total_loss / len(train_loader)
    valid_loss = mean_squared_error(y_vals_valid_normalised, valid_pred_normalized)  # Loss of true norm vs pred norm validation data
    train_losses.append(train_loss), valid_losses.append(valid_loss)

    # Save the best model based on validation loss
    if valid_loss < best_valid_loss:
        best_epoch = epoch
        best_valid_loss, best_train_loss = valid_loss, train_loss
        best_model_state = model.state_dict().copy()  # Save model state, not the whole model object

    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.5f}, Valid Loss: {valid_loss:.5f}")
print("\nTraining complete.\n")
print("Number of training samples:",training_samples)
print("Neurons:",neurons,"("+str(nparams)+" parameters)","\nNum Epochs:", num_epochs,
      "\nLearning Rate:",learning_rate,"\nBatch Size:",batch_size,"\nActivation:",activation)
# Restore the best model before evaluation
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"Best model from Epoch {best_epoch+1} restored with Training Loss: {best_train_loss:.5f} and Validation Loss: {best_valid_loss:.5f}")     
plot_train_loss(train_losses, valid_losses, labels=["Epoch", "MSE Loss", "Training and Validation Loss"])

# Final evaluation using the best model
model.eval() # Important to tell the model we're in evaluation model, ensures no weights are changed by accident
with torch.no_grad():
    valid_pred_normalized = np.squeeze(model(X_valid).detach().numpy())
y_vals_valid_predicted = (valid_pred_normalized * y_std) + y_mean # Unnormalise 

"""
####################### VISUALISE AND COMPARE THE TRUE AND PREDICTED DATA #######################

How good was the network at predicting the true function? Try different hyperparamters and samplings of the data.
"""
doubleplot(x_vals_valid, y_vals_valid, x_vals_valid, y_vals_valid_predicted, labels=["x","y","True validation data","Pred validation data",""])

    
    
    
