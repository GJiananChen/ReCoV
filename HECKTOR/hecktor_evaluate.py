from __future__ import print_function
import random

import numpy as np
import pandas as pd
import os
import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
from sklearn.model_selection import StratifiedKFold
from dataloader import AMINNDataset, MultiFocalBags, MultiFocalRegBags
from model import  NegativeLogLikelihood, c_index, MIL_reg_Ins
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import roc_auc_score
import csv

# Training settings
# settings for HN and Lung: lr=0.0005, epoch=80
parser = argparse.ArgumentParser(description='PyTorch Multifocal Bags Example')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=0.001, metavar='R',
                    help='weight decay')
parser.add_argument('--dataset', type=str, default='hecktor_train', metavar='D',
                    help='bags have a positive labels if they contain at least one 9')
parser.add_argument('--seed', type=int, default=1234, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--model', type=str, default='ins', help='Choose b/w attention and gated_attention')
parser.add_argument('--subset', default='all', type=str,
                    help='subset of dataset to use')
parser.add_argument('--folds', default=5, type=int,
                    help='number of folds for cross validation')
parser.add_argument('--runs', default=1, type=int,
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

torch.manual_seed(args.seed)
#TODO: try train_val
def smallest_index(lst):
    return lst.index(min(lst))

def threshold_indice(lst, threshold=0.5):
    return [x[0] for x in enumerate(lst) if x[1] < threshold]

def train(epoch, train_loader, exclusion):
    model.train()
    train_loss = 0.
    train_error = 0.
    y_trues = []
    y_preds = []

    bag_label_all = []
    risk_pred_all = []
    bag_fu_all = []

    
    # No method: 0.545863453815261 ± 0.01993910181439524
    # exclusion = []
    # Jianan's method : 0.6070682730923694 ± 0.009657277664378548
    # exclusion = [10, 21, 58, 97, 135, 149, 163, 208, 235, 250, 269, 283, 285, 330, 358]
    # Test performace: 0.6019277108433734 ± 0.051782254806761484 - (2,1)
    # exclusion = [111 , 34 ,173 , 79, 215, 140, 283, 153 ,317 , 58 ,378  ,10 ,270  ,54 ,249 ,316 ,332 ,266, 149, 269]
    #Test performace: 0.6232931726907631 ± 0.015214471315728915 - (1.5, 1)
    # exclusion = [173  ,54 , 76 , 10 ,269, 215, 316 ,324 ,111  ,58, 249 ,135 ,332 ,378, 208, 153, 113, 193, 79, 222]
    #Test performnace: 0.6064257028112449 ± 0.04426570079529884 - Different auc calc, exponential weighting
    # exclusion = [173 ,194 , 58 , 10 , 76 ,300 ,270,  79 ,269 ,149 , 55 , 72 ,235,362, 193, 241, 303, 317 ,50 ,249]
    # Test performace: 0.61429718875502 ± 0.04313305362661224 - Repeat experiment with different seed
    # exclusion = [266 ,231, 316,  10, 215, 180,  16,  34, 194, 154, 173, 235 , 54 ,249, 269,   8,  12,  23, 245, 255]
    # Test performace: 0.6359839357429719 ± 0.036644116107534846 (1,1) cindex+auc 0.5 sample_v2
    # exclusion = [257, 241 , 72 ,316 , 54 ,324, 233 ,173 , 10 , 79 ,111  , 2 ,215 ,193 , 16, 149, 266 , 55 ,249 ,332]
    # Test performace: 0.5934136546184738 ± 0.038921941130044165 (1,1) correction
    # exclusion = [173, 104, 316, 294, 266, 54, 241, 249, 11, 40, 305, 149, 187, 135, 76, 72, 19, 194, 127, 321]
    # Test performace: 0.5791164658634538 ± 0.05488947650849761 (cindex correction, 1,1)
    # exclusion = [266, 104, 316, 332, 241, 173, 299, 79, 10, 324, 113, 236, 135, 283, 215, 311, 12, 140, 378, 278]
    # Test performace: 0.6371084337349397 ± 0.034037275928644764 (1,1,0.5 cindex correction)
    # exclusion = [72, 104, 54, 173, 316, 266, 40, 257, 235, 10, 270, 76, 233, 332, 362, 336, 245, 79, 269, 11]
    # Test performace: 0.6175100401606425 ± 0.04749535532320803 (cindex correction, no idea regd the parameters)
    # exclusion = [40, 316, 10, 76, 241, 257, 173, 193, 233, 332, 79, 299, 54, 104, 111, 149, 198, 266, 113, 8]
    #Without auc does not work well(058), 1.5 gives 0.62 similar 
    #The method performs better compared to sample 1 - cindex(61),1cindexauc(61),1.5cindexauc()
    criterion = NegativeLogLikelihood().cuda()

    for batch_idx, (data, bag_label, bag_id, bag_fu, t_id) in enumerate(train_loader):
        # if bag_id[0][0] not in exclusion:
        if t_id not in exclusion:
            data = torch.stack(data).squeeze().float()
            if args.cuda:
                data, bag_label, bag_fu = data.cuda(), bag_label.cuda(), bag_fu.cuda()
            data, bag_label, bag_fu = Variable(data), Variable(bag_label), Variable(bag_fu)
            # reset gradients
            optimizer.zero_grad()
            # calculate loss and metrics
            # loss, _ = model.calculate_objective(data, bag_label)
            # train_loss += loss.data[0]
            error, predicted_label, risk_pred = model.calculate_classification_error(data, bag_label)
            train_error += error
            bag_label_all.append(bag_label)
            risk_pred_all.append(risk_pred)
            bag_fu_all.append(bag_fu)
    risk_pred_all, bag_label_all, bag_fu_all = torch.stack(risk_pred_all), torch.stack(bag_label_all), torch.stack(bag_fu_all)
    loss = criterion(risk_pred_all, bag_label_all, bag_fu_all, model.cuda())
    train_c = c_index(-risk_pred_all, bag_label_all, bag_fu_all)

    # y_trues.append(bag_label.max().cpu().item())
    # y_preds.append(risk_pred.cpu().item())
    # backward pass
    loss.sum().backward()
    # step
    optimizer.step()

    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_error /= len(train_loader)
    # auc = roc_auc_score(y_trues, y_preds)

    return train_error, train_loss, train_c

def test(test_loader):
    model.eval()
    test_loss = 0.
    test_error = 0.
    y_trues = []
    y_preds = []
    y_instances = []
    ids = []

    bag_label_all = []
    risk_pred_all = []
    bag_fu_all = []
    criterion = NegativeLogLikelihood().cuda()
    for batch_idx, (data, bag_label, bag_id, bag_fu, _) in enumerate(test_loader):
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
    return test_error, test_loss, test_c, bag_label_all, risk_pred_all, ids, y_instances

if __name__ == '__main__':
    if args.cuda:
        # torch.cuda.manual_seed(args.seed)
        print('\nGPU is ON!')

    aucs_last = []
    testaucs_last = []
    
    # with open("/home/ramanav/Projects/ReCoV/results/HECKTOR/memory_cindex_15auc_01.npy","rb") as file:
    # with open("/home/ramanav/Projects/ReCoV/results/HECKTOR/memory_cindex_1auc_0.5_1.npy","rb") as file:
    # with open("/home/ramanav/Projects/ReCoV/results/HECKTOR/memory_cindex_1.5auc_0.4_1.npy","rb") as file:
    # with open("/home/ramanav/Projects/ReCoV/results/HECKTOR/memory_cindex_1auc_0.5_1_corr.npy","rb") as file:
    # with open("/home/ramanav/Projects/ReCoV/results/memory_auc_cindex_corr_8.npy","rb") as file:
    # with open("/home/ramanav/Projects/ReCoV/results/memory_auc_cindex_corr_9.npy","rb") as file:
        # memory = np.load(file)
    # exclusion = list(np.argsort(memory)[:20])
    # print(exclusion)
    exclusion = []

    for seeds in range(5):
        random_seed = args.seed + seeds
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        data = AMINNDataset(data=args.dataset)
        features, labels = data.extract_dataset(subset=args.subset, censor=args.censor, feature_class='original', normalize=args.normalize)
        dataset = MultiFocalRegBags(features,labels)
        if args.dataset == 'hecktor_train':
            data_test = AMINNDataset(data='hecktor_test')
            features_test, labels_test = data_test.extract_dataset(subset=args.subset, censor = args.censor, feature_class='original', normalize = args.normalize)
            dataset_test = MultiFocalRegBags(features_test,labels_test)

        aucs = []
        aucs_stacked = []
        test_ids_all = []
        for irun in range(args.runs):
            y_true_all = []
            y_prob_all = []
            ids_all = []
            y_instances_all = []
            num_examples = len(dataset)
            labels = [x[0] for x in dataset.labels_list]
            # Start print
            # print('--------------------------------')
            # print(f'Run {irun}')
            # print('--------------------------------')

            # Define data loaders for training and testing data in this fold
            trainloader = data_utils.DataLoader(
                dataset,
                batch_size=1, drop_last=False)
            testloader = data_utils.DataLoader(
                dataset_test,
                batch_size=1, drop_last=False)

            # print('Init Model')
            model = MIL_reg_Ins(args.pooling, n_input=features.shape[1])
            if args.cuda:
                model.cuda()

            optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)

            # print('Start Training')

            best_auc = 0
            
            for epoch in range(1, args.epochs + 1):
                train_error, train_loss, train_auc = train(epoch, trainloader,exclusion)
                test_error, test_loss, auc, _, _, _, _ = test(testloader)
                # print(f'Epoch: {epoch}, Train error: {train_error:.4f}, '
                    # f'Test error: {test_error:.4f}, Train_AUC: {train_auc:.4f}, Test_AUC: {auc:.4f}')
                if epoch == args.epochs:
                    best_auc = auc
                    # torch.save(model.state_dict(), os.path.join(wandb.run.dir, f'model_last_epoch.pt'))
                    # print(f'Model saved at epoch {epoch}.')
                    aucs_last.append(train_auc)

        testaucs_last.append(auc)
        # print("Train auc: {}".format(train_auc))
        # print("Test auc: {}".format(auc))
    print(u"Train performace: {} \u00B1 {}".format(np.mean(aucs_last),np.std(aucs_last)))
    print(u"Test performace: {} \u00B1 {}".format(np.mean(testaucs_last),np.std(testaucs_last)))

