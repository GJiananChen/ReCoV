"""
Calculate uncertainity based on monte carlo dropout
"""
import torch, os
import torch._utils
import torch.nn as nn
import numpy as np
# import torchmetrics
import random

def dropout_uncertainity(model,x,num_iters):
    '''
    Calculated uncertaininty using monte carlo dropout
    Parameters:
        model: Model with dropout in its architecture
        x: input to the model
        num_iters: Number of iterations for the inference
    Returns:
        output_mean: output mean
        output_std: output standard deviation
    '''

    #Set model to train to switch on dropout
    output = []
    for _ in range(num_iters):
        y,_,_ = model(x)
        output.append(y)
    output = torch.cat(output,dim=0)
    return torch.mean(output,dim=0), torch.std(output,dim=0)