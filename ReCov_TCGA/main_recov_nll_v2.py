from __future__ import print_function

import time
import argparse
import pdb
import os
import math
import sys
from timeit import default_timer as timer
from pathlib import Path
ROOT_PATH = str(Path(__file__).resolve().parent.parent/"Patch-GCN")
sys.path.append(ROOT_PATH)

import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

### Internal Imports
from datasets.dataset_survival import Generic_WSI_Survival_Dataset, Generic_MIL_Survival_Dataset
from utils.file_utils import save_pkl, load_pkl
from utils.core_utils import train
from utils.utils import get_custom_exp_code
from sklearn.model_selection import train_test_split, StratifiedKFold
sys.path.append(str(Path(__file__).resolve().parent.parent / "utils"))
from sample import sample_folds

### PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, sampler
from multiprocessing import set_start_method

### Sets Seed for reproducible experiments.
def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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
    # con_metrics = np.ones(n)*0.8
    con_metrics = np.ones(n)*0.5
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

def nll_metric(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat([torch.ones_like(c), S], 1) #S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    #h[y] = h(1)
    #S[1] = S(1)
    #First time is the patient does not have event till before that event time period, the second term forces hazard to be high for that time event
    # uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    # censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
    # uncensored_term = (1-c) * ((torch.gather(S_padded, 1, Y).clamp(min=eps)/2) + (torch.gather(hazards, 1, Y).clamp(min=eps))/2)
    # uncensored_term = (1-c)*torch.gather(S_padded, 1, Y).clamp(min=eps) * torch.gather(hazards, 1, Y).clamp(min=eps)
    uncensored_term = torch.gather(hazards,1,Y).clamp(min=eps)
    uncensored_term = (1-c)*(1/(1+np.exp(-35*uncensored_term + 3)))
    # uncensored_term = 
    #normalized by max value of uncensored term
    # uncensored_term = uncensored_term / uncensored_term.max()
    censored_term = c * torch.gather(S_padded, 1, Y+1).clamp(min=eps)

    # return censored_loss + uncensored_loss
    # return 4*uncensored_term + censored_term
    return (uncensored_term + censored_term).ravel()

def auc_weight_function(auc_val):
    #0.5 is random hence easily achievable
    weight = np.exp(2.5*(auc_val - 0.5))
    #the val of weight will be 1 we debias it with exp**(-0.3)
    weight = weight - 0.6
    return weight

def rank_weights(aucs, test_ids, risk_pred_all, bag_fu_all, bag_labels, hazards_all, labels_all, memory)->np.array:
    '''
    Gives weighting to all the samples in the dataset
    High weight implies more probability of being selected in the top fold
    '''
    n_folds = len(aucs)
    number_ids = [len(i) for i in test_ids]
    test_ids_all = np.concatenate(test_ids)
    # uncertainity_all = np.concatenate(uncertainity_all)

    weights_auc = aucs
    weights_auc = np.concatenate([[weights_auc[i]]*number_ids[i] for i in range(n_folds)])
    
    con_metrics_all = concordance_indvidual(np.concatenate(risk_pred_all),np.concatenate(bag_fu_all),np.concatenate(bag_labels))
    weights_like_2 = con_metrics_all.copy()

    # nll_metric_all = nll_metric(torch.cat(hazards_all,dim=0).cpu(),None,torch.cat(labels_all,dim=0),1-torch.tensor(np.concatenate(bag_labels)))
    # weights_like = nll_metric_all.numpy().copy()
    # weights = 3*weights_auc + weights_like
    # weights = LAMDA*weights_auc + weights_like
    # weights = LAMDA*weights_auc + weights_like + 1.2*weights_like_2
    # weights = LAMDA*weights_auc + weights_like_2 + 0.5*weights_like
    # weights = LAMDA*weights_auc + 1*weights_like_2
    # weights = weights_like
    # weights = weights_auc
    # weights = LAMDA*weights_auc
    weights = 1*weights_auc + weights_like_2
    # weights = weights_like
    weights = weights[np.argsort(test_ids_all)]
    memory = 0.3*weights + 0.7*memory
    # print(weights[gt])
    return memory


def train_one_run(args, run_id, fold_splits, identified, train_cases_list):
    latest_val_cindex = []
    risk_all = []
    bag_fu_all = []
    patient_ids = []
    bag_labels = []
    hazards_all = []
    labels_all = []

    ### Start 5-Fold CV Evaluation.
    for fold, (train_split, test_split) in enumerate(fold_splits):
        train_ids = [train_cases_list[ids] for ids in train_split]
        test_ids = [train_cases_list[ids] for ids in test_split]
        
        if FILTER_DROP and (len(identified)>0):
            #select 50% indice to drop randomly
            drop_idx = np.random.permutation(identified)[:int(NOISY_DROP*len(identified))]
            train_ids = list(set(train_ids) - set(drop_idx))
        
        seed_torch(args.seed + run_id)

        train_dataset, test_dataset = dataset.return_splits(from_id=True,  train_case_ids=np.array(train_ids), val_case_ids=np.array(test_ids))
        datasets = (train_dataset, test_dataset)

        ### Run Train-Val on Survival Task.
        if args.task_type == 'survival':
            val_latest, cindex_latest = train(datasets, fold, args)
            temp_dict = {"risk":[],"survival":[],"event":[],"case_id":[],"hazards":[],"label":[]}
            for keys, items in val_latest.items(): 
                temp_dict["risk"].append(items["risk"])
                temp_dict["survival"].append(items["survival"])
                temp_dict["event"].append(items["event"])
                temp_dict["case_id"].append(items["case_id"])
                temp_dict["hazards"].append(items["hazards"])
                temp_dict["label"].append(items["disc_label"])
            latest_val_cindex.append(cindex_latest)
            risk_all.append(np.array(temp_dict["risk"]))
            bag_fu_all.append(np.array(temp_dict["survival"]))
            bag_labels.append(np.array(temp_dict["event"]))
            patient_ids.append(np.array(temp_dict["case_id"]))
            hazards_all.append(torch.cat(temp_dict["hazards"],dim=0))
            labels_all.append(torch.tensor(temp_dict["label"]))

    return latest_val_cindex, patient_ids, risk_all, bag_fu_all, bag_labels, hazards_all, labels_all

def plot_weights(x,exp_name):
    all_indices = np.arange(len(x))
    fig = plt.figure()
    plt.subplot(1,2,1)
    sns.histplot(x)
    plt.subplot(1,2,2)
    plt.scatter(all_indices,x)
    plt.savefig(str(RESULT_DIR/f"{exp_name}.png"))


######################################################################################################################################################
### Training settings
parser = argparse.ArgumentParser(description='Configurations for Survival Analysis on TCGA Data.')
### Checkpoint + Misc. Pathing Parameters
# parser.add_argument('--data_root_dir', type=str, default='/scratch/localdata0/vramanathan/TCGA_MIL_Patches_Ctrans_1MPP/', help='data directory')
parser.add_argument('--data_root_dir', type=str, default='/localdisk3/ramanav/TCGA_processed/TCGA_MIL_Patches_Ctrans_1MPP/', help='data directory')
parser.add_argument('--seed',            type=int, default=1, help='Random seed for reproducible experiment (default: 1)')
parser.add_argument('--k',               type=int, default=5, help='Number of folds (default: 5)')
parser.add_argument('--k_start',         type=int, default=-1, help='Start fold (Default: -1, last fold)')
parser.add_argument('--k_end',           type=int, default=-1, help='End fold (Default: -1, first fold)')
# parser.add_argument('--results_dir',     type=str, default='/home/vramanathan/scratch/amgrp/TCGA_Results/10xfeature_MIL/Recov', help='Results directory (Default: ./results)')
parser.add_argument('--results_dir',     type=str, default='/localdisk3/ramanav/Results/TCGA_Results/Recov', help='Results directory (Default: ./results)')
parser.add_argument('--which_splits',    type=str, default='5foldcv', help='Which splits folder to use in ./splits/ (Default: ./splits/5foldcv')
parser.add_argument('--split_dir',       type=str, default='tcga_brca', help='Which cancer type within ./splits/<which_splits> to use for training. Used synonymously for "task" (Default: tcga_blca_100)')
parser.add_argument('--log_data',        action='store_true', default=True, help='Log data using tensorboard')
parser.add_argument('--overwrite',       action='store_true', default=False, help='Whether or not to overwrite experiments (if already ran)')
parser.add_argument('--testing',         action='store_true', default=False, help='debugging tool')
parser.add_argument('--censor_time',      type=float, default=150.0, help='Censoring patients beyond certain time period')

### Model Parameters.
parser.add_argument('--model_type',      type=str, choices=['deepset', 'amil', 'mifcn', 'dgc', 'patchgcn','tmil'], default='tmil', help='Type of model (Default: mcat)')
parser.add_argument('--mode',            type=str, choices=['path', 'cluster', 'graph'], default='path', help='Specifies which modalities to use / collate function in dataloader.')
parser.add_argument('--num_gcn_layers',  type=int, default=4, help = '# of GCN layers to use.')
parser.add_argument('--edge_agg',        type=str, default='spatial', help="What edge relationship to use for aggregation.")
parser.add_argument('--resample',        type=float, default=0.00, help='Dropping out random patches.')
parser.add_argument('--drop_out',        action='store_true', default=True, help='Enable dropout (p=0.25)')

### Optimizer Parameters + Survival Loss Function
parser.add_argument('--opt',             type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--batch_size',      type=int, default=1, help='Batch Size (Default: 1, due to varying bag sizes)')
parser.add_argument('--gc',              type=int, default=32, help='Gradient Accumulation Step.')
parser.add_argument('--max_epochs',      type=int, default=15, help='Maximum number of epochs to train (default: 20)')
# parser.add_argument('--max_epochs',      type=int, default=25, help='Maximum number of epochs to train (default: 20)')
parser.add_argument('--lr',              type=float, default=2e-4, help='Learning rate (default: 0.0001)')
# parser.add_argument('--lr',              type=float, default=5e-4, help='Learning rate (default: 0.0001)')
parser.add_argument('--bag_loss',        type=str, choices=['svm', 'ce', 'ce_surv', 'nll_surv', 'cox_surv'], default='nll_surv', help='slide-level classification loss function (default: ce)')
parser.add_argument('--label_frac',      type=float, default=1.0, help='fraction of training labels (default: 1.0)')
parser.add_argument('--bag_weight',      type=float, default=0.7, help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--reg',             type=float, default=1e-5, help='L2-regularization weight decay (default: 1e-5)')
parser.add_argument('--alpha_surv',      type=float, default=0.0, help='How much to weigh uncensored patients')
parser.add_argument('--reg_type',        type=str, choices=['None', 'omic', 'pathomic'], default='None', help='Which network submodules to apply L1-Regularization (default: None)')
parser.add_argument('--lambda_reg',      type=float, default=1e-4, help='L1-Regularization Strength (Default 1e-4)')
parser.add_argument('--weighted_sample', action='store_true', default=True, help='Enable weighted sampling')
parser.add_argument('--early_stopping',  action='store_true', default=False, help='Enable early stopping')

args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Creates Experiment Code from argparse + Folder Name to Save Results
args = get_custom_exp_code(args)
args.task = '_'.join(args.split_dir.split('_')[:2]) + '_survival'
print("Experiment Name:", args.exp_code)


seed_torch(args.seed)

encoding_size = 768
settings = {'num_splits': args.k, 
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'bag_weight': args.bag_weight,
            'seed': args.seed,
            'model_type': args.model_type,
            'weighted_sample': args.weighted_sample,
            'gc': args.gc,
            'opt': args.opt}
print('\nLoad Dataset')

if 'survival' in args.task:
    args.n_classes = 4
    study = '_'.join(args.task.split('_')[:2])
    if study == 'tcga_kirc' or study == 'tcga_kirp':
        combined_study = 'tcga_kidney'
    elif study == 'tcga_luad' or study == 'tcga_lusc':
        combined_study = 'tcga_lung'
    else:
        combined_study = study
    study_dir = '%s_10x_features' % combined_study
    dataset = Generic_MIL_Survival_Dataset(csv_path = '%s/%s/%s_all_clean.csv.zip' % (ROOT_PATH,args.dataset_path, combined_study),
                                           mode = args.mode,
                                        #    data_dir= os.path.join(args.data_root_dir, study_dir),
                                            data_dir = args.data_root_dir,
                                           shuffle = False, 
                                           seed = args.seed, 
                                           print_info = True,
                                           patient_strat= False,
                                           n_bins=4,
                                           label_col = 'survival_months',
                                           ignore=[])
else:
    raise NotImplementedError

if isinstance(dataset, Generic_MIL_Survival_Dataset):
    args.task_type = 'survival'
else:
    raise NotImplementedError

### Creates results_dir Directory.
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

### Appends to the results_dir path: 1) which splits were used for training (e.g. - 5foldcv), and then 2) the parameter code and 3) experiment code
args.results_dir = os.path.join(args.results_dir, args.which_splits, args.param_code, str(args.exp_code) + '_s{}'.format(args.seed) + time.strftime("_%d%b_%H_%M_%S", time.gmtime()))
if not os.path.isdir(args.results_dir):
    os.makedirs(args.results_dir)

if ('summary_latest.csv' in os.listdir(args.results_dir)) and (not args.overwrite):
    print("Exp Code <%s> already exists! Exiting script." % args.exp_code)
    sys.exit()

### Sets the absolute path of split_dir
args.split_dir = os.path.join(ROOT_PATH,'splits', args.which_splits, args.split_dir)
print("split_dir", args.split_dir)
assert os.path.isdir(args.split_dir)
settings.update({'split_dir': args.split_dir})
settings.update({'experiment details':"Testing with cindex metric and auc metric"})

with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

################################################################# RECOV SETTINGS #####################################################################
N_RUNS = 50
LAMDA = 1
TOP_K = 20
TAU = 0.5
FILTER_DROP = False
# NOISY_DROP = 0.4
NOISY_DROP = 0.25
EXP_NAME = f"TCGA_cindex_{LAMDA}auc_{TAU}_{FILTER_DROP*1}_nosplit"
RESULT_DIR = Path(args.results_dir) / "tcgabrca_recov_results"
if not (RESULT_DIR).is_dir():
    os.mkdir(RESULT_DIR)

########################################################################################################################################################

if __name__ == "__main__":
    #### Create Results Directory
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    patient_df = dataset.slide_data[["case_id","label"]].drop_duplicates(keep="first")
    case_list = patient_df["case_id"].values
    outcome_list = patient_df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(case_list,outcome_list,train_size=0.8,shuffle=True, stratify=outcome_list,random_state=args.seed)
    ### Gets the Train + Val Dataset Loader.
    train_dataset, test_dataset = dataset.return_splits(from_id=True,  train_case_ids=X_train, val_case_ids=X_test)
    train_df = train_dataset.slide_data[["case_id","label"]].drop_duplicates(keep="first")
    train_df = train_df.sort_values(by=["case_id"])
    train_cases_list = train_df["case_id"].values
    train_outcomes = train_df["label"].values
    kfold = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)
    fold_splits = list(kfold.split(train_cases_list, train_outcomes))
    identified = []
    memory = np.zeros(len(train_dataset))
    
    for seed in range(args.seed,args.seed+N_RUNS):
        latest_val_cindex, patient_ids, risk_all, bag_fu_all, bag_labels, hazards_all, labels_all = train_one_run(args, seed, fold_splits, identified, train_cases_list)
        memory = rank_weights(latest_val_cindex,patient_ids, risk_all, bag_fu_all, bag_labels, hazards_all, labels_all, memory)
        fold_splits, fold_ids = sample_folds(args.k,memory,TAU)
        #Get K worst samples
        temp = np.argsort(memory)[:TOP_K]
        identified = [train_cases_list[ids] for ids in temp]
        #Save memory
        with open(str(RESULT_DIR/f"memory_{EXP_NAME}.npy"),"wb") as file:
            np.save(file,memory)
        print(latest_val_cindex)
        print(temp)
        plot_weights(memory,EXP_NAME)    

    print("finished!")
    print("end script")