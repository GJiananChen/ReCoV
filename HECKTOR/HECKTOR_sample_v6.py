import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import os
import wandb
import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
from sklearn.model_selection import StratifiedKFold
from dataloader import AMINNDataset, MultiFocalBags, MultiFocalRegBags
from model import  NegativeLogLikelihood, c_index, MIL_reg_Ins
from torch.utils.data.sampler import SubsetRandomSampler
import pickle
from sklearn.metrics import roc_auc_score
import csv
import random
import seaborn as sns
from matplotlib import pyplot as plt

from HECKTOR import train, test, set_seed, smallest_index, get_parser
from sample import sample_folds
from uncertainity import dropout_uncertainity

def concordance_indvidual(risk_pred_all,bag_fu_all, bag_labels):
    '''
    Creates concordance metric consisting of each individual datapoint 
    Parameters:
        risk_pred_all: risk scores in the form of vector nx1
        bag_fu_all: time of the datapoints in form of vector nx1
    '''
    #NxN matrix consisting of 
    # X1>=X2 and T1<T2 = 1
    # T1=T2 = 0
    # X1>=X2 and T1>T2 = -1
    # Sample is valid if neither t1,t2 are censored or if t1 is censored after t2
    risk_pred_all = risk_pred_all.ravel()
    bag_fu_all = bag_fu_all.ravel()
    bag_labels = bag_labels.ravel()

    n = len(risk_pred_all)
    con_matrix = np.zeros((n,n))
    #By default we assume labels are good
    con_metrics = np.ones(n)*0.8
    cij = 0
    for i in range(1,n):
        for j in range(i):
            if bag_fu_all[i]==bag_fu_all[j]: #both times are equal hence no value
                cij = 0
            else:
                cond1 = risk_pred_all[i] >= risk_pred_all[j]
                cond2 = bag_fu_all[i] < bag_fu_all[j]
                if bag_labels[i]==0: #censored
                    if bag_labels[j]==0: #both censored no value
                        cij = 0
                    else:
                        if bag_fu_all[i] > bag_fu_all[j]: #ti censored after tj which experienced some event
                            cij = (2*cond1 - 1)*-1 #we know ti>tj hence ri<rj for conc and ri>rj for disc
                        else:
                            cij = 0 
                elif bag_labels[j]==1:
                    cij = (2*cond1 - 1)*(2*cond2 -1)
                else:
                    if bag_fu_all[i] < bag_fu_all[j]:
                        cij = (2*cond1 - 1) # we know ti<tj, hence ri>rj
                    else:
                        cij = 0 #ti which is time to event is after tj which is censored, hence we have no way of knowing if tj actually happend before or after
            con_matrix[j,i] = cij
            con_matrix[i,j] = cij
    
    for i in range(n):
        con_pairs = len(np.where(con_matrix[i,:]==1)[0])
        disc_pairs = len(np.where(con_matrix[i,:]==-1)[0])
        if (con_pairs+disc_pairs)>0:
            con_metrics[i] = (con_pairs)/(con_pairs+disc_pairs)
    return con_metrics
     
def auc_weight_function(auc_val):
    #0.5 is random hence easily achievable
    weight = np.exp(2.5*(auc_val - 0.5))
    #the val of weight will be 1 we debias it with exp**(-0.3)
    weight = weight - 0.6
    return weight

def rank_weights(aucs, test_ids, risk_pred_all, bag_fu_all, bag_labels, uncertainity_all, memory)->np.array:
    '''
    Gives weighting to all the samples in the dataset
    High weight implies more probability of being selected in the top fold
    '''
    n_folds = len(aucs)
    number_ids = [len(i) for i in test_ids]
    test_ids_all = np.concatenate(test_ids)
    # uncertainity_all = np.concatenate(uncertainity_all)

    # weights_auc = auc_weight_function(np.array(aucs))
    weights_auc = aucs
    weights_auc = np.concatenate([[weights_auc[i]]*number_ids[i] for i in range(n_folds)])
    
    con_metrics_all = concordance_indvidual(-np.concatenate(risk_pred_all),np.concatenate(bag_fu_all),np.concatenate(bag_labels))
    weights_like = con_metrics_all.copy()
    
    # weights_like =  1 - (uncertainity_all - uncertainity_all.min())/(uncertainity_all.max()-uncertainity_all.min())

    # weights = 3*weights_auc + weights_like
    # weights = 1.5*weights_auc + weights_like
    weights = 1*weights_auc + weights_like
    weights = weights[np.argsort(test_ids_all)]
    memory = 0.3*weights + 0.7*memory
    # print(weights[gt])
    return memory

def get_uncertainity(testloader, model, args):
    model.train()
    uncertainity = []
    with torch.no_grad():
        for batch_idx, (data, bag_label, bag_id, bag_fu,index) in enumerate(testloader):
            data = torch.stack(data).squeeze().float()
            if args.cuda:
                data = data.cuda()
            output_mean, output_std = dropout_uncertainity(model,data,50)
            uncertainity.append(output_std)
    return torch.stack(uncertainity).cpu().numpy()

def plot_weights(x,noise_labels):
    labels = np.zeros_like(x)
    labels[noise_labels] = 1
    data_idx = np.arange(len(x))
    sorting = np.argsort(x)
    labels = labels[sorting]
    x = x[sorting]
    x_clean = x[np.where(labels==0)[0]]
    x_noise = x[np.where(labels==1)[0]]
    plt.scatter(data_idx[np.where(labels==0)[0]],x_clean,s=1)
    plt.scatter(data_idx[np.where(labels==1)[0]],x_noise,s=1)
    plt.legend(["clean","noise"])
    plt.savefig("temp.png")


TOP_K = 20
TAU = 0.5

args = get_parser()
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

all_candidates = []
# For more information please check dataloader.py
data = AMINNDataset(data=args.dataset)
features, labels = data.extract_dataset(subset=args.subset, censor=args.censor, feature_class='original', normalize=args.normalize)

# MultiFocalRegBags gives survival regression labels, MultiFocalBags gives binary prediction labels
# but you need to change the model accordingly
dataset = MultiFocalRegBags(features,labels)
num_examples = len(dataset)
labels = [x[0] for x in dataset.labels_list]
memory = np.zeros(num_examples)
# Please comment out or change wandb tracking credentials
# Define the K-fold Cross Validator
kfold = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
fold_splits = list(kfold.split(dataset, labels))

for seed in range(args.seed,args.seed+args.n_runs):
    set_seed(seed)
    wandb.config = vars(args)
    wandb.init(project="recov_hecktor",
                config=wandb.config, 
                name=f'{args.dataset}_{args.pooling}_{args.normalize}_{args.subset}_{args.censor}_{seed}',
                # dir="/localdisk3/ramanav/Results/wandb",
                mode="disabled")
    # scripts for generating multiple instance survival regression datasets from radiomics spreadsheets
    aucs = []
    aucs_last = []
    aucs_stacked = []
    test_ids_all = []
    risk_all = []
    bag_fu_all = []
    test_ids_weight = []
    bag_labels = []
    uncertainity_all = []

    # Start print
    print('--------------------------------')
    print(f"Run {seed}")
    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(fold_splits):
        # print(f'Run {seed}, FOLD {fold}')
        # print('--------------------------------')
        test_ids_all.append(test_ids)
        train_subsampler = SubsetRandomSampler(train_ids)
        test_subsampler = SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = data_utils.DataLoader(
            dataset,
            batch_size=1, sampler=train_subsampler, drop_last=False)
        testloader = data_utils.DataLoader(
            dataset,
            batch_size=1, sampler=test_subsampler, drop_last=False)

        # print('Init Model')
        model = MIL_reg_Ins(args.pooling, n_input=features.shape[1], apply_dropout=False, p = 0.2)
        # model = MIL_reg_Ins(args.pooling, n_input=features.shape[1], apply_dropout=True, p = 0.2)
        if args.cuda:
            model.cuda()

        wandb.watch(model, log_freq=10)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)

        # print('Start Training')

        best_auc = 0

        for epoch in range(1, args.epochs + 1):
            train_error, train_loss, train_auc = train(epoch, trainloader, model, optimizer, args)
            test_error, test_loss, auc, bag_label_all, risk_pred_all, ids, y_instances, bag_fu = test(testloader, model, args)
            # uncertainity = get_uncertainity(testloader, model, args)
            uncertainity = 0
            wandb.log({"train_error": train_error, "test_error": test_error, "train_auc": train_auc, "test_auc": auc, "epoch": epoch})
            if epoch == args.epochs:
                aucs_last.append(auc)
                bag_fu_all.append(bag_fu.cpu().numpy())
                test_ids_weight.append(ids)
                risk_all.append(risk_pred_all.cpu().numpy())
                bag_labels.append(bag_label_all.cpu().numpy())
                uncertainity_all.append(uncertainity)

    memory = rank_weights(aucs_last,test_ids_weight, risk_all, bag_fu_all, bag_labels, uncertainity_all, memory)
    #Save memory
    with open("../results/memory_auc_cindex_corr_9_onerun.npy","wb") as file:
        np.save(file,memory)
    #Generate new set of folds based on weights
    fold_splits, fold_ids = sample_folds(args.folds,memory,TAU)
    #Get K worst samples
    identified = np.argsort(memory)[:TOP_K]
    jianan_indices = [10, 21, 58, 97, 135, 149, 163, 208, 235, 250, 269, 283, 285, 330, 358]
    all_indices = np.arange(num_examples)
    other_indicies = np.array(list(set(all_indices) - set(jianan_indices)))
    print(aucs_last)
    print(identified)
    fig = plt.figure()
    plt.subplot(1,2,1)
    sns.histplot(memory)
    plt.subplot(1,2,2)
    plt.scatter(other_indicies,memory[other_indicies])
    plt.scatter(jianan_indices,memory[jianan_indices])
    plt.legend(["other","jianan"])
    plt.savefig("../results/hecktor_weights_cindex_corr_v9_onerun.png")

    wandb.log({"last_aucs_average": np.mean(aucs_last)})

