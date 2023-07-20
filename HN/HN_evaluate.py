import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import random
import csv
import os
import argparse
import pickle

import numpy as np
import pandas as pd
import wandb
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import roc_auc_score
import seaborn as sns
from matplotlib import pyplot as plt

from HECKTOR.model import  NegativeLogLikelihood, c_index, MIL_reg_Ins
from HECKTOR.dataloader import AMINNDataset, MultiFocalBags, MultiFocalRegBags
from HECKTOR.HECKTOR import train, test, set_seed, smallest_index, get_parser
from sample import sample_folds
from uncertainity import dropout_uncertainity


#Indices to remove ( seed 1234 performs better)
#Test performace: 0.6686601307189542 ± 0.155194161951401, seed 1234 0.6180228758169936 ± 0.09095581447193551
# exclusion = []
#jianan's indices: Test performace: 0.6232107843137256 ± 0.16471674363114716, seed 1234  0.6140604575163399 ± 0.0819084269107388
# exclusion = [45, 101, 113]
# hn_v4 Test performance: 0.6743218954248366 ± 0.13175441505066632 , seed 1234 0.681470588235294 ± 0.09958977275488806
# exclusion = [9,1,10,38]

with open("/home/ramanav/Projects/ReCoV/results/memory_cindex_1auc_0.5_1.npy","rb") as file:
    memory = np.load(file)
exclusion = list(np.argsort(memory)[:5])
print(exclusion)

args = get_parser()
args.dataset = "hn"
args.censor = 1825
args.subset = "outlier_removed"
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

all_candidates = []
# For more information please check dataloader.py
data = AMINNDataset(data=args.dataset,exclusion=exclusion)
features, labels = data.extract_dataset(subset=args.subset, censor=args.censor, feature_class='original', normalize=args.normalize)

# MultiFocalRegBags gives survival regression labels, MultiFocalBags gives binary prediction labels
# but you need to change the model accordingly
dataset = MultiFocalRegBags(features,labels)
num_examples = len(dataset)
labels = [x[0] for x in dataset.labels_list]
# Define the K-fold Cross Validator

set_seed(args.seed)
# scripts for generating multiple instance survival regression datasets from radiomics spreadsheets
aucs_train = []
aucs_last = []

# Start print
print('--------------------------------')
print(f"Run {args.seed}")
# K-fold Cross Validation model evaluation
kfold = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
fold_splits = list(kfold.split(dataset, labels))
for fold, (train_ids, test_ids) in enumerate(fold_splits):
    # print(f'Run {seed}, FOLD {fold}')
    # print('--------------------------------')
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

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)

    # print('Start Training')

    best_auc = 0

    for epoch in range(1, args.epochs + 1):
        train_error, train_loss, train_auc = train(epoch, trainloader, model, optimizer, args)
        test_error, test_loss, auc, bag_label_all, risk_pred_all, ids, y_instances, bag_fu = test(testloader, model, args)
        if epoch == args.epochs:
            aucs_train.append(train_auc)
            aucs_last.append(auc)
    # aucs_train_seed.append(np.mean(aucs_train))
    # aucs_test_seed.append(np.mean(aucs_last))
    # print("Train auc: {}".format(aucs_train))
    # print("Test auc: {}".format(aucs_last))
# print(u"Train performace: {} \u00B1 {}".format(np.mean(aucs_train_seed),np.std(aucs_train_seed)))
# print(u"Test performace: {} \u00B1 {}".format(np.mean(aucs_test_seed),np.std(aucs_test_seed)))
print(u"Train performace: {} \u00B1 {}".format(np.mean(aucs_train),np.std(aucs_train)))
print(u"Test performace: {} \u00B1 {}".format(np.mean(aucs_last),np.std(aucs_last)))

