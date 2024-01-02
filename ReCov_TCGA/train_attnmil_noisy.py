import yaml
import sys
import os
import random

from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt
import seaborn as sns

from ReCov_TCGA.sample import sample_folds
from utils.trainfuncs import TrainEngine_MIL_folds
import trainer


def set_seed(random_seed): 
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def folds_tcgabraca(path):
    df = pd.read_csv(path)
    # df = df.loc[(df["is_female"]==1) & (df["train"]==1)]
    df = df.loc[(df["is_female"]==1)]
    # df = df.loc[df["is_female"]==1]
    #Form a searchable dictionary
    #replace slide id with the feature vector
    columns_needed = ["case_id","survival_months","censorship"]
    df = df[columns_needed]
    #rename some columns
    df.rename(columns={"survival_months":"durations","censorship":"vital_status"},inplace=True)
    df["vital_status"] = 1 - df["vital_status"]
    df.sort_values(by="case_id")
    df.drop_duplicates(keep="first",inplace=True)
    df.reset_index(drop=True,inplace=True)
    cases = df.to_dict()["case_id"]
    reverse_cases = {v: k for k, v in cases.items()}
    kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_splits = list(kfold.split(df["case_id"].values, df["vital_status"].values))
    return cases, reverse_cases, fold_splits

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
     
def auc_weight_function(auc_val):
    #0.5 is random hence easily achievable
    weight = np.exp(2.5*(auc_val - 0.5))
    #the val of weight will be 1 we debias it with exp**(-0.3)
    weight = weight - 0.6
    return weight

def rank_weights(aucs, test_ids, risk_pred_all, bag_fu_all, bag_labels, memory)->np.array:
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
    
    con_metrics_all = concordance_indvidual(-np.concatenate(risk_pred_all),np.concatenate(bag_fu_all),np.concatenate(bag_labels))
    weights_like = con_metrics_all.copy()
    
    # weights = 3*weights_auc + weights_like
    weights = LAMDA*weights_auc + weights_like
    # weights = 1*weights_auc + weights_like
    # weights = weights_like
    weights = weights[np.argsort(test_ids_all)]
    memory = 0.3*weights + 0.7*memory
    # print(weights[gt])
    return memory

def plot_weights(x,exp_name):
    all_indices = np.arange(len(x))
    fig = plt.figure()
    plt.subplot(1,2,1)
    sns.histplot(x)
    plt.subplot(1,2,2)
    plt.scatter(all_indices,x)
    plt.savefig(str(RESULT_DIR/f"{exp_name}.png"))

########################################################################################################################################################
CONFIG_PATH = "/home/vramanathan/Projects/TCGA_MIL/configs/attn_mil_tcga_folds_noisy.yml"
DATAFRAME_PATH = "/home/vramanathan/Projects/TCGA_MIL/data/tcga_brca_all_clean.csv"
RESULT_DIR = Path("./results/tcga_sample")
N_RUNS = 30
N_FOLDS = 5
RANDOM_STATE = 1
LAMDA = 1
TOP_K = 30
TAU = 0.5
FILTER_DROP = True
NOISY_DROP = 0.4

if not (RESULT_DIR).is_dir():
    os.mkdir(RESULT_DIR)

########################################################################################################################################################

identified = []
cases, reverse_cases, fold_splits = folds_tcgabraca(DATAFRAME_PATH)
memory = np.zeros(len(cases))

for seed in range(RANDOM_STATE,RANDOM_STATE+N_RUNS):
    set_seed(seed)
    EXP_NAME = f"TCGA_cindex_{LAMDA}auc_{TAU}_{FILTER_DROP*1}_nosplit"
    aucs_last = []
    risk_all = []
    bag_fu_all = []
    test_ids_weight = []
    bag_labels = []
    print('--------------------------------')
    print(f"Run {seed}")
    for fold, (train_split, test_split) in enumerate(fold_splits):
        train_split = [cases[ids] for ids in train_split]
        test_split = [cases[ids] for ids in test_split]
        # print(f"Fold number: {fold}")
        deep_trainer = TrainEngine_MIL_folds(config_pth=CONFIG_PATH, fold_no=fold, train_split=train_split, test_split=test_split, random_seed=seed)
        deep_trainer.run()
        auc, survstatus, survtime_all, riskval, patient_ids = deep_trainer.get_val_metrics
        patient_ids_num = [reverse_cases[ids] for ids in patient_ids]
        # print(f"Fold {fold} test performance: {auc}")
        aucs_last.append(auc)
        test_ids_weight.append(patient_ids_num)
        risk_all.append(riskval.cpu().numpy())
        bag_labels.append(survstatus.cpu().numpy())
        bag_fu_all.append(survtime_all.cpu().numpy())
    memory = rank_weights(aucs_last,test_ids_weight, risk_all, bag_fu_all, bag_labels, memory)
    #Generate new set of folds based on weights
    fold_splits, fold_ids = sample_folds(N_FOLDS,memory,TAU)
    #Get K worst samples
    identified = np.argsort(memory)[:TOP_K]
    #Save memory
    with open(str(RESULT_DIR/f"memory_{EXP_NAME}.npy"),"wb") as file:
        np.save(file,memory)
    print(aucs_last)
    print(identified)
    plot_weights(memory,EXP_NAME)    
# print(u"Test performance: {} \u00B1 {}".format(np.mean(aucs_last),np.std(aucs_last)))