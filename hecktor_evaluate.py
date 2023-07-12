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
from sklearn.metrics import roc_auc_score
import csv

# Training settings
# settings for HN and Lung: lr=0.0005, epoch=80
parser = argparse.ArgumentParser(description='PyTorch Multifocal Bags Example')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
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

def train(epoch, train_loader):
    model.train()
    train_loss = 0.
    train_error = 0.
    y_trues = []
    y_preds = []

    bag_label_all = []
    risk_pred_all = []
    bag_fu_all = []

    # exclusion = ['MDA-159', 'MDA-151', 'MDA-054', 'CHUV-049', 'CHUS-080', 'HGJ-052', 'MDA-013', 'HGJ-078', 'HGJ-086', 'CHUP-073', 'HMR-012']
    # exclusion_sv = [ 30,  50, 132, 183, 192, 338]
    # exclusion_sv = ['CHUS-100', 'MDA-190', 'MDA-140', 'MDA-020', 'CHUM-038', 'CHUP-036']
    exclusion = []
    # exclusion = ['CHUV-040', 'MDA-067', 'CHUM-021', 'HGJ-013']
    criterion = NegativeLogLikelihood().cuda()

    for batch_idx, (data, bag_label, bag_id, bag_fu) in enumerate(train_loader):
        if bag_id[0][0] not in exclusion:
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
    for batch_idx, (data, bag_label, bag_id, bag_fu) in enumerate(test_loader):
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
        torch.cuda.manual_seed(args.seed)
        print('\nGPU is ON!')

    wandb.config = vars(args)
    wandb.init(project="simulation_hn", entity="jiananchen", config=wandb.config, name=f'{args.dataset}_{args.pooling}_{args.normalize}_{args.subset}_{args.censor}_{args.seed}')
    artifact = wandb.Artifact(f'{wandb.run.name}_preds', 'predictions')

    data = AMINNDataset(data=args.dataset)
    features, labels = data.extract_dataset(subset=args.subset, censor=args.censor, feature_class='original', normalize=args.normalize)
    dataset = MultiFocalRegBags(features,labels)
    if args.dataset == 'hecktor_train':
        data_test = AMINNDataset(data='hecktor_test')
        features_test, labels_test = data_test.extract_dataset(subset=args.subset, censor = args.censor, feature_class='original', normalize = args.normalize)
        dataset_test = MultiFocalRegBags(features_test,labels_test)

    aucs = []
    aucs_last = []
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
        print('--------------------------------')
        print(f'Run {irun}')
        print('--------------------------------')

        # Define data loaders for training and testing data in this fold
        trainloader = data_utils.DataLoader(
            dataset,
            batch_size=1, drop_last=False)
        testloader = data_utils.DataLoader(
            dataset_test,
            batch_size=1, drop_last=False)

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
                best_auc = auc
                torch.save(model.state_dict(), os.path.join(wandb.run.dir, f'model_last_epoch.pt'))
                print(f'Model saved at epoch {epoch}.')
                aucs_last.append(auc)


            # model.save(os.path.join(wandb.run.dir, f'model_{fold}.h5'))
            # print('Start Testing')
            # _, _, auc, y_trues, y_probs, ids, y_instances = test(testloader)
            # y_true_all.extend(y_trues)
            # y_prob_all.extend(y_probs)
            # y_instances_all.append(y_instances)
            # ids_all.extend(ids)
            # aucs.append(auc)
        # print(y_trues)
        # print(y_probs)
        # print(ids)

    # auc_stacked = roc_auc_score(y_true_all, y_prob_all)
    # aucs_stacked.append(auc_stacked)
    # print(aucs)
    # table = wandb.Table(columns=['id', 'ytrue', 'yprob'])
    # for i, v in enumerate(y_true_all):
    #     table.add_data(ids_all[i], y_true_all[i], y_prob_all[i])
    # artifact.add(table, f'{wandb.run.name}_preds_run{irun+1}')


        print('Init Model')
        model = MIL_reg_Ins(args.pooling, n_input=features.shape[1] )
        if args.cuda:
            model.cuda()

        model.load_state_dict(torch.load(os.path.join(wandb.run.dir, f'model_last_epoch.pt')))
        test_error, test_loss, auc, _, _, _, _ = test(testloader)

    print(aucs_last)

    y_pred_all = [int(x >= 0.5) for x in y_prob_all]
    # acc = sum(1 for x,y in zip(y_pred_all,y_true_all) if x == y) / float(len(y_pred_all))
    # print(f'Average AUC = {np.mean(aucs):.4f}, Acc =  {np.mean(acc):.4f}')
    wandb.log_artifact(artifact)
    # wandb.log({"auc_average": np.mean(aucs)})
    # wandb.log({"auc_std": np.std(aucs)})
    wandb.log({"last_aucs_average": np.mean(aucs_last)})
    # wandb.log({"best_aucs": aucs_best})
    # wandb.log({"last_aucs": aucs_last})
    # wandb.log({"auc_stacked": np.mean(aucs_stacked)})
    # wandb.log({'acc': acc})
    # df = pd.DataFrame({'ID'
    # :ids_all, 'ytrue':y_true_all, 'yprob':y_prob_all})
    # df.to_csv('hn_real_df_km_reg.csv')

    # [0.6378205128205128, 0.45502998001332445, 0.42571618920719523, 0.6443994601889339, 0.5991902834008097]
    #[0.6025641025641025, 0.49700199866755496, 0.44770153231179216, 0.5809716599190283, 0.5607287449392713]
    # wandb.log({"auc_std": np.std(aucs_stacked})