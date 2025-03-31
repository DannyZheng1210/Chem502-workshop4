# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 15:47:07 2025

Author: Adam Coxson, PhD student, University of Liverpool
Department of Chemistry, Materials Innovation Factory, Levershulme Research Centre
Module: stats_and_plot_functions.py
Local dependencies: None
For the dML workshop Mar 2025

Contains:
    1D and 2D equations for the workshop materials 
    Functions to sample input data for said functions
    Statistical measures for normalisation and error loss calcs
    Plotting functions to visualise data
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from sklearn.preprocessing import StandardScaler

################## STATS AND SAMPLING FUNCTIONS ##################
def compute_y(x): # For 1D example
    return ((0.4*x + 0.5*np.sin(5*x) + np.sin(3*x)) + 10*np.cos(x)*np.exp(-0.1*x)) + 7

# For 1D example. 
def sample_x_values(x_min, x_max, dx, num_samples):
    """
    Randomly sample x values within a specified range and step size.
    
    Parameters
    ----------
    x_min : float
        Minimum x value.
    x_max : float
        Maximum x value.
    dx : float
        Step size for x values.
    num_samples : int
        Number of samples to draw.
    
    Returns
    -------
    np.array
        Randomly sampled x values.
    """
    possible_x = np.arange(x_min, x_max + dx, dx)  # generate contiguous possible x values
    return np.random.choice(possible_x, num_samples, replace=False)  # Randomly sample x vals

def compute_z(x, y): # For 2D example
    #z = 2 * np.exp(-0.2 * x) * (np.sin(x) ** 2) + (np.sin(2 * y) ** 3) + 1
    z = 0.8*np.exp(-0.2 * x) * (np.sin(x) ** 3) + 0.6*np.exp(-0.15 * y)*(np.sin(2 * y) ** 2) + 1.5
    return z
    

# Function to randomly sample x values within a range and a given step size
def sample_xy_values(x_min_max=[-5,5], y_min_max=[-5,5], dxdy=[0.001,0.001], num_samples=10000):
    possible_x = np.arange(x_min_max[0], x_min_max[1] + dxdy[0], dxdy[0])  # Generate possible x values
    possible_y = np.arange(y_min_max[0], y_min_max[1] + dxdy[1], dxdy[1])  # Generate possible x values
    choice_x= np.random.choice(possible_x, num_samples, replace=False)  # Randomly sample
    choice_y= np.random.choice(possible_y, num_samples, replace=False)  # Randomly sample
    return choice_x, choice_y

class CustomScaler:
    def __init__(self, num_scaled_cols=2):
        """
        Custom scaler that applies StandardScaler independently to the first `num_scaled_cols` columns
        while leaving the remaining columns unchanged.

        Parameters:
        num_scaled_cols (int): Number of columns to scale.
        """
        self.num_scaled_cols = num_scaled_cols
        self.scalers = [StandardScaler() for _ in range(num_scaled_cols)]

    def fit(self, X):
        """
        Fits the scalers independently to each of the first `num_scaled_cols` columns of X.

        Parameters:
        X (numpy array): The input data.
        """
        for i in range(self.num_scaled_cols):
            self.scalers[i].fit(X[:, i].reshape(-1, 1))
        return self

    def transform(self, X):
        """
        Transforms each of the first `num_scaled_cols` columns independently, keeping the rest unchanged.

        Parameters:
        X (numpy array): The input data.

        Returns:
        numpy array: The transformed data.
        """
        X_transformed = X.copy()
        for i in range(self.num_scaled_cols):
            X_transformed[:, i] = self.scalers[i].transform(X[:, i].reshape(-1, 1)).flatten()
        return X_transformed

    def fit_transform(self, X):
        """
        Fits the scalers and transforms the data.

        Parameters:
        X (numpy array): The input data.

        Returns:
        numpy array: The transformed data.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """
        Inverses the transformation for the first `num_scaled_cols` columns.

        Parameters:
        X (numpy array): The transformed data.

        Returns:
        numpy array: The original scaled data.
        """
        X_original = X.copy()
        for i in range(self.num_scaled_cols):
            X_original[:, i] = self.scalers[i].inverse_transform(X[:, i].reshape(-1, 1)).flatten()
        return X_original

def normalize_data(vals): 
    """
    Very important to regularise data before training, don't forget to unnormalize any prediction data. 
    Can also use Sklearns StandardScaler. 
    Parameters
    ----------
    vals : np.array
        1D array of values to be normalise
    Returns
    -------
    norm_vals : np.array
        Normalised values.
    mean : float 
         Mean of the original values, returned for unscaling data later.
    std : float
        Standard deviation of the original values, returned for unscaling data later.

    """
    mean, std = vals.mean(), vals.std()
    norm_vals = ((vals - mean) / std).astype(np.float32)

    return norm_vals, mean, std

def mean_squared_error(y_true, y_pred): 
    return np.mean((y_true - y_pred) ** 2)

def calc_rhoc(x,y):
    ''' 
    Concordance Correlation Coefficient
    https://nirpyresearch.com/concordance-correlation-coefficient/
    '''
    sxy = np.sum((x - x.mean())*(y - y.mean()))/x.shape[0]
    rhoc = 2*sxy / (np.var(x) + np.var(y) + (x.mean() - y.mean())**2)
    return rhoc

def calc_num_net_parameters(layer_config, output_size=1):
    if type(layer_config) is int:
        input_neurons = layer_config
        total_params = input_neurons * output_size + output_size
        return total_params
    
    elif type(layer_config[0]) is int and len(layer_config)==1:
        input_neurons = layer_config[0]
        output_neurons = output_size
        total_params = input_neurons * output_size + output_size
        return total_params
    
    if type(layer_config[0]) is not int:
        neurons=[]
        for n in layer_config:
            neurons.append(n[0])
        layer_config=neurons
    if layer_config[-1] != output_size:
        layer_config.append(1)
        
    total_params = 0
    for i in range(len(layer_config) - 1):
        input_neurons = layer_config[i]
        output_neurons = layer_config[i + 1]

        # Calculate weights and biases for the current layer
        weights = input_neurons * output_neurons
        biases = output_neurons

        # Add to the total parameters
        total_params += weights + biases

    return total_params

######### PLOTTING FUNCTIONS ###########

def plot_train_loss(train_loss, valid_loss=None, labels=["Epoch","MSE loss",""],figsize=(6,4)):
    """
    Plot training loss over epochs.
    
    Parameters
    ----------
    train_loss : list
        List of training loss values.
    valid_loss : list, optional
        List of validation loss values (default is None).
    labels : list, optional
        Labels for the plot axes and title (default is ["Epoch", "MSE loss", ""]).
    """
    plt.figure(figsize=figsize)
    x=np.arange(1,len(train_loss)+1) # Number of Epochs
    plt.plot(x, train_loss, 'r-',label='Train loss')
    if valid_loss is not None:
        plt.plot(x, valid_loss, 'b-',label='Valid loss')
        plt.legend()
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title(labels[2])
    plt.grid(visible=True, which='major', axis='both')
    plt.minorticks_on()
    plt.show()
    return None

def plot_loss_and_r(train_loss, valid_loss, valid_r, figsize=(6,4)):
  
    x=np.arange(1,len(valid_loss)+1)
    fig, ax1 = plt.subplots(figsize=figsize)
    # Plotting the loss curves
    ax1.plot(x, train_loss, 'r-', label='Train Loss')
    ax1.plot(x, valid_loss, 'b-', label='Valid Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE loss', color='black')
    ax1.tick_params('y', colors='black')
    ax1.grid(visible=True, which='major', axis='both')
    ax1.minorticks_on()

    # Creating a secondary y-axis for correlation
    ax2 = ax1.twinx()
    ax2.plot(x, valid_r, 'g-', label='Correlation')
    ax2.set_ylabel('Correlation', color='black')
    ax2.tick_params('y', colors='black')
    ax2.minorticks_on()
            
    best_epoch=np.argmin(valid_loss)
    ax1.axvline(x=best_epoch+1, color='k', linestyle='-', alpha=0.8)

    # Adding legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    lines = lines_1 + lines_2
    labels = labels_1 + labels_2
    ax1.legend(lines, labels, loc='center right')

    # plt.title(title)
    # if save==True:
    #     plt.savefig(fname=savedir+title+'loss_plot.png')
    # if show == True:
    #     plt.show()
    return fig
    
def doubleplot(x1,y1,x2,y2,labels=["x","y","A","B",""],figsize=(8,6),lims=None):
    """
    Create a plot by overplotting two sets of data.
    
    Parameters
    ----------
    x1, y1 : array-like
        Data for the first plot (line).
    x2, y2 : array-like
        Data for the second plot (scatter).
    labels : list
        Labels for axes, datasets, and title.
    figsize : tuple
        Figure size.
    lims : tuple, optional
        Axis limits in the format ((x_min, x_max), (y_min, y_max)).
    """
    plt.figure(figsize=figsize)
    plt.plot(x1, y1, label=labels[2], color="blue")
    plt.scatter(x2, y2, color="green", marker="+", label=labels[3], zorder=3)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title(labels[4])
    plt.grid(True)
    plt.minorticks_on()
    if lims is not None:
        plt.xlim(lims[0][0],lims[0][1])
        plt.ylim(lims[1][0],lims[1][1])
    plt.legend()
    plt.show()
    return None

def heatmapplot(x,y,z,labels=["x","y","z",""],figsize=(8,6),alpha=0.75):
    plt.figure(figsize=figsize)
    plt.scatter(x, y, c=z, cmap='viridis', alpha=0.75)
    plt.colorbar(label=labels[2])
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title(labels[3])
    plt.grid(True)
    plt.minorticks_on()
    plt.show()
    return None

def surfaceplot(x, y, z, labels=["x", "y", "z", "Surface Plot"], figsize=(8, 6)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    xi = np.linspace(min(x), max(x), 200)
    yi = np.linspace(min(y), max(y), 200)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x, y), z, (Xi, Yi), method='cubic')

    surf = ax.plot_surface(Xi, Yi, Zi, cmap='viridis', edgecolor='none')
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.set_title(labels[3])
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    plt.show()
    return None

def heatmapplot_subplots(x, y, z1, z2, z3, labels1, labels2, labels3, figsize=(18, 6), alpha=0.75):
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # First heatmap (Prediction)
    im1 = axes[0].scatter(x, y, c=z1, cmap='viridis', alpha=alpha)
    fig.colorbar(im1, ax=axes[0], label=labels1[2])
    axes[0].set_xlabel(labels1[0])
    axes[0].set_ylabel(labels1[1])
    axes[0].set_title(labels1[3])
    axes[0].grid(True)
    
    # Second heatmap (True values)
    im2 = axes[1].scatter(x, y, c=z2, cmap='viridis', alpha=alpha)
    fig.colorbar(im2, ax=axes[1], label=labels2[2])
    axes[1].set_xlabel(labels2[0])
    axes[1].set_ylabel(labels2[1])
    axes[1].set_title(labels2[3])
    axes[1].grid(True)
    
    # Third heatmap (Deviation)
    im3 = axes[2].scatter(x, y, c=z3, cmap='viridis', alpha=alpha)
    fig.colorbar(im3, ax=axes[2], label=labels3[2])
    axes[2].set_xlabel(labels3[0])
    axes[2].set_ylabel(labels3[1])
    axes[2].set_title(labels3[3])
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
def surfaceplot_subplots(x, y, z1, z2, z3, labels1, labels2, labels3, figsize=(18, 6)):
    fig = plt.figure(figsize=figsize)
    
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_trisurf(x, y, z1, cmap='viridis')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, label=labels1[2])
    ax1.set_xlabel(labels1[0])
    ax1.set_ylabel(labels1[1])
    ax1.set_zlabel(labels1[2])
    ax1.set_title(labels1[3])
    
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_trisurf(x, y, z2, cmap='viridis')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, label=labels2[2])
    ax2.set_xlabel(labels2[0])
    ax2.set_ylabel(labels2[1])
    ax2.set_zlabel(labels2[2])
    ax2.set_title(labels2[3])
    
    ax3 = fig.add_subplot(133, projection='3d')
    surf3 = ax3.plot_trisurf(x, y, z3, cmap='viridis')
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10, label=labels3[2])
    ax3.set_xlabel(labels3[0])
    ax3.set_ylabel(labels3[1])
    ax3.set_zlabel(labels3[2])
    ax3.set_title(labels3[3])
    
    plt.show()

# def surfaceplot_subplots(x, y, z1, z2, z3, labels1, labels2, labels3, figsize=(18, 6)):
#     fig = plt.figure(figsize=figsize)
    
#     ax1 = fig.add_subplot(131, projection='3d')
#     ax1.plot_trisurf(x, y, z1, cmap='viridis')
#     ax1.set_xlabel(labels1[0])
#     ax1.set_ylabel(labels1[1])
#     ax1.set_zlabel(labels1[2])
#     ax1.set_title(labels1[3])
    
#     ax2 = fig.add_subplot(132, projection='3d')
#     ax2.plot_trisurf(x, y, z2, cmap='viridis')
#     ax2.set_xlabel(labels2[0])
#     ax2.set_ylabel(labels2[1])
#     ax2.set_zlabel(labels2[2])
#     ax2.set_title(labels2[3])
    
#     ax3 = fig.add_subplot(133, projection='3d')
#     ax3.plot_trisurf(x, y, z3, cmap='viridis')
#     ax3.set_xlabel(labels3[0])
#     ax3.set_ylabel(labels3[1])
#     ax3.set_zlabel(labels3[2])
#     ax3.set_title(labels3[3])
    
#     plt.show()