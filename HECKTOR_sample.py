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
    risk_pred_all = risk_pred_all.ravel().cpu().numpy()
    bag_fu_all = bag_fu_all.ravel().cpu().numpy()
    bag_labels = bag_labels.ravel().cpu().numpy()

    n = len(risk_pred_all)
    con_matrix = np.zeros((n,n))
    #By default we assume labels are good
    con_metrics = np.ones(n)*0.8
    cij = 0
    for i in range(1,n):
        for j in range(i):
            if bag_fu_all[i]==bag_fu_all[j]:
                cij = 0
            else:
                cond1 = risk_pred_all[i] >= risk_pred_all[j]
                cond2 = bag_fu_all[i] < bag_fu_all[j]
                if bag_labels[i]==0:
                    if bag_labels[j]==0:
                        cij = 0
                    else:
                        if bag_fu_all[i] > bag_fu_all[j]:
                            cij = (2*cond1 - 1)*-1
                        else:
                            cij = 0 
                else:
                    cij = (2*cond1 - 1)*(2*cond2 -1)                 
            con_matrix[j,i] = cij
            con_matrix[i,j] = cij
    
    for i in range(n):
        con_pairs = len(np.where(con_matrix[i,:]==1)[0])
        disc_pairs = len(np.where(con_matrix[i,:]==-1)[0])
        if (con_pairs+disc_pairs)>0:
            con_metrics[i] = (con_pairs)/(con_pairs+disc_pairs)
    return con_metrics
     
def rank_weights(aucs, test_ids, risk_pred_all, bag_fu_all, bag_labels, memory)->np.array:
    '''
    Gives weighting to all the samples in the dataset
    High weight implies more probability of being selected in the top fold
    '''
    n_folds = len(aucs)
    number_ids = [len(i) for i in test_ids]
    test_ids_all = np.concatenate(test_ids)
    # pred_probs_all = np.concatenate(pred_probs)
    # test_labels_all = np.concatenate(test_labels)
    # aucsweights = np.linspace(0,1,n_folds)
    # weights_auc = aucsweights[np.argsort(aucs)]
    # weights_auc = np.max(aucs)/aucs
    weights_auc = np.max(aucs) - aucs
    weights_auc = 1 - (weights_auc - weights_auc.min())/(weights_auc.max()-weights_auc.min())
    weights_auc = np.concatenate([[weights_auc[i]]*number_ids[i] for i in range(n_folds)])
    con_metrics_all = []
    for i in range(n_folds):
        con_metrics_all.append(concordance_indvidual(-risk_pred_all[i],bag_fu_all[i], bag_labels[i]))

    con_metrics_all = np.concatenate(con_metrics_all)
    weights_like = 1 - (con_metrics_all.max() - con_metrics_all)
    # weights = (weights_auc**80)*weights_like
    weights = 7*weights_auc + weights_like
    # weights = weights_like
    weights = weights[np.argsort(test_ids_all)]
    memory = 0.5*weights + 0.5*memory
    # print(weights[gt])
    return memory

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
TAU = 1

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
    # artifact = wandb.Artifact(f'{wandb.run.name}_preds', 'predictions')
    # scripts for generating multiple instance survival regression datasets from radiomics spreadsheets
    aucs = []
    aucs_last = []
    aucs_stacked = []
    test_ids_all = []
    risk_all = []
    bag_fu_all = []
    test_ids_weight = []
    bag_labels = []

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
        model = MIL_reg_Ins(args.pooling, n_input=features.shape[1])
        if args.cuda:
            model.cuda()

        wandb.watch(model, log_freq=10)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)

        # print('Start Training')

        best_auc = 0

        for epoch in range(1, args.epochs + 1):
            train_error, train_loss, train_auc = train(epoch, trainloader, model, optimizer, args)
            test_error, test_loss, auc, bag_label_all, risk_pred_all, ids, y_instances, bag_fu = test(testloader, model, args)
            # print(f'Epoch: {epoch}, Train error: {train_error:.4f}, '
                # f'Test error: {test_error:.4f}, Train_AUC: {train_auc:.4f}, Test_AUC: {auc:.4f}')
            wandb.log({"train_error": train_error, "test_error": test_error, "train_auc": train_auc, "test_auc": auc, "epoch": epoch})
            if epoch == args.epochs:
                aucs_last.append(auc)
                bag_fu_all.append(bag_fu)
                test_ids_weight.append(ids)
                risk_all.append(risk_pred_all)
                bag_labels.append(bag_label_all)

    memory = rank_weights(aucs_last,test_ids_weight, risk_all, bag_fu_all, bag_labels, memory)
    # random.seed(run)
    # np.random.seed(run)
    # kf = KFold(n_splits=N_FOLDS, random_state=random_state,shuffle=True)
    # fold_splits = list(kf.split(X,y))
    #Generate new set of folds based on weights
    fold_splits, fold_ids = sample_folds(args.folds,memory,TAU)
    #Get K worst samples
    identified = np.argsort(memory)[:TOP_K]
    print(aucs_last)
    print(np.sort(identified))
    fig = plt.figure()
    plt.subplot(1,2,1)
    sns.histplot(memory)
    plt.subplot(1,2,2)
    plt.scatter(np.arange(num_examples),memory)
    plt.savefig("./hecktor_weights.png")

    # aucs_best= []
    # for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset, labels)):
    #     # Print
    #     print(f'Run {seed}, FOLD {fold}')
    #     print('--------------------------------')
    #     test_subsampler = SubsetRandomSampler(test_ids)
    #     testloader = data_utils.DataLoader(
    #         dataset,
    #         batch_size=1, sampler=test_subsampler, drop_last=False)

    #     print('Init Model')
    #     model = MIL_reg_Ins(args.pooling, n_input=features.shape[1] )
    #     if args.cuda:
    #         model.cuda()

    # for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset, labels)):
    #     test_ids_all.append(test_ids)
    #     print(test_ids)
    # candidates = test_ids_all[smallest_index(aucs_last)]
    # all_candidates.append(candidates)

    # wandb.log_artifact(artifact)
    wandb.log({"last_aucs_average": np.mean(aucs_last)})

# # save sample ids that belongs to the worst folds across runs
# with open(f'{args.dataset}_{seed}_{seed+args.n_runs}.pkl', 'wb') as f:
#     pickle.dump(all_candidates, f)

