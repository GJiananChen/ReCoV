from __future__ import print_function

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
# Training settings
# settings for HN and Lung: lr=0.0005, epoch=80
def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Multifocal Bags Example')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--reg', type=float, default=0.001, metavar='R',
                        help='weight decay')
    parser.add_argument('--dataset', type=str, default='hecktor_train', metavar='D',
                        help='bags have a positive labels if they contain at least one 9')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--model', type=str, default='ins', help='Choose b/w attention and gated_attention')
    parser.add_argument('--subset', default='all', type=str,
                        help='subset of dataset to use')
    parser.add_argument('--folds', default=5, type=int,
                        help='number of folds for cross validation')
    parser.add_argument('--n_runs', default=50, type=int,
                        help='number of runs for repeated validation')
    parser.add_argument('--censor',
                        default=730,
                        type=int, help='threshold for right censoring')
    parser.add_argument('--pooling', default='sum', type=str,
                        help='which multiple instance pooling to use')
    parser.add_argument('--norm', dest='normalize', action='store_true', help='two step normalization')
    parser.add_argument('--no-norm', dest='normalize', action='store_false', help='z-score normalization')
    parser.set_defaults(normalize=True)

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def smallest_index(lst):
    '''
    return the index of the smallest item in a list
    :param lst: target list
    :return: index of the smallest item
    '''
    return lst.index(min(lst))

def threshold_indice(lst, threshold=0.5):
    '''
    return the indice of items in a list that have a threshold smaller than threshold
    :param lst: target list
    :param threshold: cutoff
    :return: indice of items
    '''
    return [x[0] for x in enumerate(lst) if x[1] < threshold]

def train(epoch, train_loader, model, optimizer, args):
    '''
    code for training a model
    :param epoch: number of epoches for training the model
    :param train_loader: dataloader for training
    :return: train error (not working in survival regression), train loss, training c-index
    '''
    model.train()
    train_loss = 0.
    train_error = 0.

    # initialize lists for bag label, predictions and grount truth follow-ups
    bag_label_all = []
    risk_pred_all = []
    bag_fu_all = []

    # survival regression with negative loglikelihood, https://doi.org/10.1186/s12874-018-0482-1
    criterion = NegativeLogLikelihood().cuda()

    # training
    for batch_idx, (data, bag_label, bag_id, bag_fu,index) in enumerate(train_loader):
        data = torch.stack(data).squeeze().float()
        if args.cuda:
            data, bag_label, bag_fu = data.cuda(), bag_label.cuda(), bag_fu.cuda()
        data, bag_label, bag_fu = Variable(data), Variable(bag_label), Variable(bag_fu)

        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        error, predicted_label, risk_pred = model.calculate_classification_error(data, bag_label)
        train_error += error
        bag_label_all.append(bag_label)
        risk_pred_all.append(risk_pred)
        bag_fu_all.append(bag_fu)
    risk_pred_all, bag_label_all, bag_fu_all = torch.stack(risk_pred_all), torch.stack(bag_label_all), torch.stack(bag_fu_all)
    loss = criterion(risk_pred_all, bag_label_all, bag_fu_all, model.cuda())
    train_c = c_index(-risk_pred_all, bag_label_all, bag_fu_all)
    loss.sum().backward()
    # step
    optimizer.step()

    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_error /= len(train_loader)

    return train_error, train_loss, train_c

def test(test_loader, model, args):
    '''
    code for testing a model
    :param test_loader: dataloader for testing
    :return: test error (not working in survival regression), test loss, test c-index,
             all bag labels, all predictions, ids of bags, instance labels
    '''
    model.eval()
    test_loss = 0.
    test_error = 0.

    y_instances = []
    ids = []

    bag_label_all = []
    risk_pred_all = []
    bag_fu_all = []

    criterion = NegativeLogLikelihood().cuda()
    with torch.no_grad():
        # testing
        for batch_idx, (data, bag_label, bag_id, bag_fu, index) in enumerate(test_loader):
            data = torch.stack(data).squeeze().float()
            if args.cuda:
                data, bag_label, bag_fu = data.cuda(), bag_label.cuda(), bag_fu.cuda()
            data, bag_label, bag_fu = Variable(data), Variable(bag_label), Variable(bag_fu)

            loss, attention_weights = model.calculate_objective(data, bag_label)
            test_loss += loss.data[0]
            error, predicted_label, risk_pred = model.calculate_classification_error(data, bag_label)
            y_instances.append(predicted_label)
            bag_label_all.append(bag_label)
            risk_pred_all.append(risk_pred)
            bag_fu_all.append(bag_fu)

            # y_trues.append(bag_label.max().cpu().item())
            # y_preds.append(y_prob.cpu().item())
            # ids.append(bag_id[0][0])
            ids.extend(index.cpu().numpy())
            test_error += error

            # if batch_idx < 5:  # plot bag labels and instance labels for first 5 bags
            #     bag_level = (bag_label.cpu().data.numpy()[0], int(predicted_label.cpu().data.numpy()[0][0]))
            #     # instance_level = list(zip(instance_labels.numpy()[0].tolist(),
            #     #                      np.round(attention_weights.cpu().data.numpy()[0], decimals=3).tolist()))
            #
            #     print('\nTrue Bag Label, Predicted Bag Label: {}'.format(bag_level))
        risk_pred_all, bag_label_all, bag_fu_all = torch.stack(risk_pred_all), torch.stack(bag_label_all), torch.stack(
                bag_fu_all)
        test_c = c_index(-risk_pred_all, bag_label_all, bag_fu_all)
        test_error /= len(test_loader)
        test_loss = criterion(risk_pred_all, bag_label_all, bag_fu_all, model.cuda())
        # auc = roc_auc_score(y_trues, y_preds)
        # print('\nTest Set, Loss: {:.4f}, Test error: {:.4f}, Test AUC: {:.4f}'.format(test_loss.cpu().numpy()[0], test_error, auc))
        
        # #Per data point concordance
        # risk_matrix = 
    
    return test_error, test_loss, test_c, bag_label_all, risk_pred_all, ids, y_instances, bag_fu_all

if __name__ == '__main__':
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
                #    dir="/localdisk3/ramanav/Results/wandb",
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
                test_error, test_loss, auc, _, _, _, _, _ = test(testloader)
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

