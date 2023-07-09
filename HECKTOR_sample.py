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

from .HECKTOR import train, test, set_seed, smallest_index, get_parser

def rank_weights(aucs, test_ids, pred_probs, test_labels, memory)->np.array:
    '''
    Gives weighting to all the samples in the dataset
    High weight implies more probability of being selected in the top fold
    '''
    n_folds = len(aucs)
    number_ids = [len(i) for i in test_ids]
    test_ids_all = np.concatenate(test_ids)
    pred_probs_all = np.concatenate(pred_probs)
    test_labels_all = np.concatenate(test_labels)
    # aucsweights = np.linspace(0,1,n_folds)
    # weights_auc = aucsweights[np.argsort(aucs)]
    # weights_auc = np.max(aucs)/aucs
    weights_auc = np.max(aucs) - aucs
    weights_auc = 1 - (weights_auc - weights_auc.min())/(weights_auc.max()-weights_auc.min())
    weights_auc = np.concatenate([[weights_auc[i]]*number_ids[i] for i in range(n_folds)])
    idx_temp = np.stack((np.arange(len(test_labels_all)),test_labels_all))
    temp =  pred_probs_all[idx_temp[0,:],idx_temp[1,:]]
    weights_like = 1 - (temp.max() - temp)
    # weights = (weights_auc**80)*weights_like
    # weights = 1*weights_auc + weights_like
    weights = weights_like
    weights = weights[np.argsort(test_ids_all)]
    memory = 0.3*weights + 0.7*memory
    # print(weights[gt])
    return memory


args = get_parser()
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

all_candidates = []

# Please comment out or change wandb tracking credentials
for seed in range(args.seed,args.seed+args.n_runs):
    set_seed(seed)
    wandb.config = vars(args)
    wandb.init(project="recov_hecktor",
                config=wandb.config, 
                name=f'{args.dataset}_{args.pooling}_{args.normalize}_{args.subset}_{args.censor}_{seed}',
                dir="/localdisk3/ramanav/Results/wandb",
                mode="disabled")
    # artifact = wandb.Artifact(f'{wandb.run.name}_preds', 'predictions')

    # scripts for generating multiple instance survival regression datasets from radiomics spreadsheets
    # For more information please check dataloader.py
    data = AMINNDataset(data=args.dataset)
    features, labels = data.extract_dataset(subset=args.subset, censor=args.censor, feature_class='original', normalize=args.normalize)

    # MultiFocalRegBags gives survival regression labels, MultiFocalBags gives binary prediction labels
    # but you need to change the model accordingly
    dataset = MultiFocalRegBags(features,labels)
    aucs = []
    aucs_last = []
    aucs_stacked = []
    test_ids_all = []

    num_examples = len(dataset)
    labels = [x[0] for x in dataset.labels_list]
    # Define the K-fold Cross Validator
    kfold = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    # Start print
    print('--------------------------------')

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset, labels)):
        print(f'Run {seed}, FOLD {fold}')
        print('--------------------------------')
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

        print('Init Model')
        model = MIL_reg_Ins(args.pooling, n_input=features.shape[1])
        if args.cuda:
            model.cuda()

        wandb.watch(model, log_freq=10)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)

        print('Start Training')

        best_auc = 0

        for epoch in range(1, args.epochs + 1):
            train_error, train_loss, train_auc = train(epoch, trainloader)
            test_error, test_loss, auc, _, _, _, _ = test(testloader)
            print(f'Epoch: {epoch}, Train error: {train_error:.4f}, '
                f'Test error: {test_error:.4f}, Train_AUC: {train_auc:.4f}, Test_AUC: {auc:.4f}')
            wandb.log({"train_error": train_error, "test_error": test_error, "train_auc": train_auc, "test_auc": auc, "epoch": epoch})
            if epoch == args.epochs:
                aucs_last.append(auc)

    aucs_best= []
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset, labels)):
        # Print
        print(f'Run {seed}, FOLD {fold}')
        print('--------------------------------')
        test_subsampler = SubsetRandomSampler(test_ids)
        testloader = data_utils.DataLoader(
            dataset,
            batch_size=1, sampler=test_subsampler, drop_last=False)

        print('Init Model')
        model = MIL_reg_Ins(args.pooling, n_input=features.shape[1] )
        if args.cuda:
            model.cuda()

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset, labels)):
        test_ids_all.append(test_ids)
        print(test_ids)
    candidates = test_ids_all[smallest_index(aucs_last)]
    all_candidates.append(candidates)

    # wandb.log_artifact(artifact)
    wandb.log({"last_aucs_average": np.mean(aucs_last)})

# save sample ids that belongs to the worst folds across runs
with open(f'{args.dataset}_{seed}_{seed+args.n_runs}.pkl', 'wb') as f:
    pickle.dump(all_candidates, f)
