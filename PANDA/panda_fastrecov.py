import sys
from pathlib import Path
import os
import warnings
import random
import time
import argparse
# warnings.filterwarnings("ignore")
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold

from utils.sample import sample_folds
from train_mil import train_full, preprocess_data
from mil_models import TransMIL_peg
from panda_dataloader import Pandas_Dataset

file_loc = Path(__file__).resolve().parent.parent

def calc_qwk_metric(pred_probs, test_labels):
    """
    based on the logic that predictions near the true should be penalized less
    """
    predicted = np.sum(pred_probs,axis=1)
    return np.abs(predicted - test_labels)

def rank_weights(test_metrics, test_ids, pred_probs, test_labels, memory, lamda = [1,0])->np.array:
    '''
    Gives weighting to all the samples in the dataset
    High weight implies more probability of being selected in the top fold
    '''
    test_ids_all = np.concatenate(test_ids)
    pred_probs_all = np.concatenate(pred_probs)
    test_labels_all = np.concatenate(test_labels)
    
    n_folds = len(test_metrics)
    number_ids = [len(i) for i in test_ids]
    weights_metric = np.concatenate([[test_metrics[i]]*number_ids[i] for i in range(n_folds)])
    weights_qwk = 5 - calc_qwk_metric(pred_probs_all,test_labels_all)
    weights = lamda[0]*weights_qwk + lamda[1]*weights_metric
    weights = weights[np.argsort(test_ids_all)]
    memory = 0.3*weights + 0.7*memory
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
    fig = plt.figure()
    plt.subplot(1,2,1)
    sns.histplot(x)
    plt.subplot(1,2,2)
    plt.scatter(data_idx[np.where(labels==0)[0]],x_clean,s=1)
    plt.scatter(data_idx[np.where(labels==1)[0]],x_noise,s=1)
    plt.legend(["karolinska","radboud"])
    plt.savefig(str(file_loc / f"results/panda/panda_{EXP_NAME}.png"))

def train_one_run(fold_splits, fulltraindataset, args, filter_noise, noisy_idx=[]):
    test_metric_folds = []
    pred_probs_folds = []
    true_labels_folds = []
    image_ids_folds = []
    for fold, (train_ids, test_ids) in enumerate(fold_splits):
        #model defination
        # model = TransMIL_peg(n_classes=6)
        model = TransMIL_peg(n_classes=args.num_classes-1, dim=512)
        if filter_noise and len(noisy_idx)>0 and (args.noisy_drop>0):
            #select 80% indice to drop randomly
            drop_idx = np.random.permutation(noisy_idx)[:int(args.noisy_drop*len(noisy_idx))]
            train_ids = list(set(train_ids) - set(drop_idx))            
        
        train_set = fulltraindataset.iloc[train_ids,:].copy()
        val_set = fulltraindataset.iloc[test_ids,:].copy()
        trainset = Pandas_Dataset(train_set,args.data_root_dir)
        valset = Pandas_Dataset(val_set,args.data_root_dir)
        test_metric, pred_probs, true_labels, image_ids  = train_full((trainset,valset),model,args)
        test_metric_folds.append(test_metric)
        pred_probs_folds.append(pred_probs)
        true_labels_folds.append(true_labels)
        image_ids_folds.append(image_ids)
    return test_metric_folds, image_ids_folds, pred_probs_folds, true_labels_folds

#hyperparameters settings
parser = argparse.ArgumentParser(description='Configurations for Gleason Grading in PANDA dataset')
#system settings
parser.add_argument('--seed',type=int,default=1)
parser.add_argument('--data_root_dir', type=str, default='../data/PANDA/PANDA_MIL_Patches_Selfpipeline_1MPP/', help='data directory')
parser.add_argument('--csv_path', type=str, default='../data/PANDA')
#model settings
parser.add_argument('--num_classes',type=int, default=6)
parser.add_argument('--lr',type=float,default=1e-4)
parser.add_argument('--lamda', type=float, default=0.0005, help="weight decay to use in adam optimizer")
parser.add_argument('--patience', type=int, default=10, help="number of epochs to wait in reducelronplateu lr scheduler")
parser.add_argument('--num_epochs', type=int, default=15)
parser.add_argument('--batch_size', type=int, default=60)
#recov settings
parser.add_argument('--recov_runs',type=int, default=20,help="number of recov runs")
parser.add_argument('--n_folds',type=int,default=5)
parser.add_argument('--tau',type=float,default=1.0,help="temperature value for softmax for probabilistic sampling in recov")
parser.add_argument('--noisy_drop',type=float,default=0.8,help="percentage of dropping random idx from the bottom memory indices")
parser.add_argument('--filter_noise',action='store_true',default=False,help="for dropping noisy labels randomly from the bottom memory indices")
args = parser.parse_args()

print("Defined settings: {}".format(args))

N_RUNS = args.recov_runs
N_FOLDS = args.n_folds
TAU = args.tau
RANDOM_STATE = args.seed
NOISY_DROP = args.noisy_drop
FILTER_NOISE = args.filter_noise
TOP_K = 500
LAMDA = [1,0]
EXP_NAME = f"{time.strftime('_%d%b_%H_%M_%S', time.localtime())}_{N_RUNS}_s{RANDOM_STATE}_{TAU}_{LAMDA}_{NOISY_DROP*FILTER_NOISE}_origfoldsplit_timetest"
Path.mkdir(Path("../results/panda"),parents=True,exist_ok=True)
(X_train_clean,y_train_clean),(X_val_clean,y_val_clean),(X_test_clean,y_test_clean), _ = preprocess_data(args)

X_train_clean["isup_grade"] = y_train_clean
train_split = X_train_clean.reset_index(drop=True)
X_val_clean["isup_grade"] = y_val_clean
val_split = X_val_clean.reset_index(drop=True)
X_test_clean["isup_grade"] = y_test_clean

train_split = pd.concat((X_train_clean,X_val_clean))
train_split = train_split.sort_values(by="image_id").reset_index(drop=True)
train_split["int_id"] = train_split.index
test_split = X_test_clean.reset_index(drop=True)
save_splits = train_split.drop(columns=["data"])
save_splits.to_csv(str(file_loc/f"results/panda/splitsave_{EXP_NAME}.csv"),index=False)

kfold = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
fold_splits = list(kfold.split(train_split["int_id"].values, train_split["isup_grade"].values))

memory = np.zeros_like(len(train_split))
identified = []

start_time = time.time()
for run in range(N_RUNS):
    print(f"Iteration {run}: {test_metric_folds}")
    test_metric_folds, image_ids_folds, pred_probs_folds, true_labels_folds = train_one_run(fold_splits,train_split,args,filter_noise=FILTER_NOISE,noisy_idx=identified)
    # rank the ids
    memory = rank_weights(test_metric_folds, image_ids_folds, pred_probs_folds, true_labels_folds, memory,lamda=LAMDA)
    # Generate new set of folds based on weights
    fold_splits, fold_ids = sample_folds(N_FOLDS, memory, TAU)
    # Get K worst samples for dropping from training
    identified = np.argsort(memory)[:TOP_K]
    plot_weights(memory, (train_split["data_provider"]=="radboud"))
    with open(str(file_loc/f"results/panda/memory_{EXP_NAME}_{run+1}_v2.npy"),"wb") as file:
        np.save(file,memory)

end_time = time.time()
print(end_time-start_time)